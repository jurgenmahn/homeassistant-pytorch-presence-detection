"""YOLO Presence Detection Server

This standalone server handles video processing and object detection
using YOLO models, designed to work with Home Assistant's yolo_presence
integration through HTTP requests.
"""

import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
import urllib.parse
import base64
import numpy as np
from typing import Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import cv2
import torch

try:
    import psutil
except ImportError:
    # If psutil is not available, create a dummy implementation
    class DummyPsutil:
        @staticmethod
        def cpu_percent(*args, **kwargs):
            return 50.0  # Return a default value

        @staticmethod
        def disk_usage(path):
            class DummyDiskUsage:
                total = 100
                used = 50
                free = 50
                percent = 50.0

            return DummyDiskUsage()

    psutil = DummyPsutil()
    logging.warning("psutil not available, using dummy implementation")

from ultralytics import YOLO

# Import and configure logging
from logging_config import configure_logging

# Import our RTSP stream processor
from rtsp_stream_processor import RTSPStreamProcessor

# Set up logging with DEBUG level for console output
logger = configure_logging(
    logger_name="yolo_server",
    log_level=logging.DEBUG,  # Set overall log level to DEBUG
    console_level=logging.DEBUG,  # Set console level to DEBUG
    file_level=logging.DEBUG,
    detailed_format=True,
)

# Constants
HTTP_PORT = int(os.environ.get("HTTP_PORT", 5505))
SUPPORTED_CLASSES = [0, 15, 16, 17]  # person, bird, cat, dog
CLASS_MAP = {0: "person", 15: "bird", 16: "cat", 17: "dog"}

# Default parameters that can be overridden with environment variables
DEFAULT_DETECTION_FRAME_COUNT = int(os.environ.get("DEFAULT_DETECTION_FRAME_COUNT", 5))
DEFAULT_CONSISTENT_DETECTION_COUNT = int(os.environ.get("DEFAULT_CONSISTENT_DETECTION_COUNT", 3))

# Basic authentication settings
# Get auth credentials from environment variables or use defaults
AUTH_ENABLED = os.environ.get("ENABLE_AUTH", "false").lower() in ("true", "1", "yes")
AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "yolopassword")
# Create auth protection pattern
PROTECTED_PATHS = [
    "/",
    "/view",
    "/stream",
    "/jpeg",
]  # Paths that require authentication

# Store for detector instances (key: detector_id)
detectors: Dict[str, Any] = {}  # Will store YoloDetector instances
detector_lock = threading.Lock()

# Store for processed images (key: detector_id)
processed_images: Dict[str, Any] = (
    {}
)  # Will store latest processed image for each detector
processed_images_lock = threading.Lock()


class YoloDetector:
    """YOLO-based object detector for video streams."""

    def __init__(self, detector_id: str, config: Dict[str, Any]):
        """Initialize the detector with the given configuration."""
        self.detector_id = detector_id
        self.stream_url = config.get("stream_url")
        self.name = config.get("name", "YOLO Detector")

        # YOLO model configuration
        self.model_name = config.get("model", "yolo11l")
        self.model_path = f"{self.model_name}.pt"
        self.model = None
        self.device = "cpu"  # Will be set during initialization

        # Parse input size
        input_size_str = config.get("input_size", "640x480")
        width, height = map(int, input_size_str.split("x"))
        self.input_width = width
        self.input_height = height

        # Detection parameters
        self.detection_interval = config.get("detection_interval", 10)
        self.confidence_threshold = config.get("confidence_threshold", 0.25)
        self.frame_skip_rate = config.get("frame_skip_rate", 5)
        self.detection_frame_count = config.get("detection_frame_count", DEFAULT_DETECTION_FRAME_COUNT)
        self.consistent_detection_count = config.get("consistent_detection_count", DEFAULT_CONSISTENT_DETECTION_COUNT)

        # Auto-optimization flag
        self.auto_optimization = config.get(
            "use_auto_optimization", config.get("auto_config", False)
        )

        # Detection results data
        self.inference_time = 0
        self.detected_objects = {}
        self.frame_dimensions = (0, 0)

        # Frame storage for visualization
        self.adjusted_dimensions = (0, 0)

        # Runtime state
        self.is_running = False
        self.stop_event = threading.Event()
        self.frame_count = 0

        # RTSP Stream Processor - replaces old cap and stream handling
        self.stream_processor = None

        # Detection state
        self.last_detection_time = None
        self.people_detected = False
        self.pets_detected = False
        self.people_count = 0
        self.pet_count = 0
        self.connection_status = "disconnected"

        # Last update time for clients to check for changes
        self.last_update_time = time.time()

        # Error recovery
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.error_recovery_delay = 5  # seconds
        self.last_error_time = 0
        self.stream_reconnect_attempts = 0
        self.max_stream_reconnect_attempts = 10
        self.stream_reconnect_delay = 5  # seconds

        # Error tracking by type
        self.error_counts = {"memory": 0, "cuda": 0, "stream": 0, "other": 0}

        # Optimization state
        self.optimization_level = 0
        self.degradation_level = 0
        self.last_optimization_check = 0
        self.optimization_check_interval = 60  # seconds

        # Idle detection
        self.last_client_request = time.time()
        self.max_idle_time = 3600  # 1 hour - shut down detector if no requests

    def initialize(self) -> bool:
        """Initialize the detector."""
        try:
            logger.info(
                f"Initializing detector {self.detector_id} with model {self.model_name}"
            )

            # Load the model
            self._load_model()

            # Open the video stream
            stream_opened = self._open_stream()
            if not stream_opened:
                logger.warning(
                    f"Could not open stream for {self.detector_id}, will retry on next poll"
                )

            # No need for stream monitor thread anymore - the RTSPStreamProcessor handles this

            self.is_running = True
            self.connection_status = "connected" if stream_opened else "connecting"
            self.last_update_time = time.time()

            logger.info(f"Detector {self.detector_id} initialized successfully")
            return True

        except Exception as ex:
            logger.error(
                f"Error initializing detector {self.detector_id}: {ex}", exc_info=True
            )
            self.is_running = False
            self.connection_status = "error"
            return False

    def _load_model(self) -> None:
        """Load the YOLO model."""
        # Check if we have CUDA available
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            # Use CUDA if available, with the first GPU
            self.device = "cuda:0"
            logger.info(f"CUDA is available, using device: {self.device}")
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = (
                torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
            )
            logger.info(f"GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
        else:
            self.device = "cpu"
            logger.info("CUDA not available, using CPU")

        # Look for model in various paths
        model_paths = [
            self.model_path,
            f"models/{self.model_path}",
            f"/models/{self.model_path}",
            os.path.join(os.getcwd(), self.model_path),
            os.path.join(os.getcwd(), "models", self.model_path),
        ]

        model_found = False
        for path in model_paths:
            if os.path.exists(path):
                logger.info(f"Found model at {path}")
                try:
                    self.model = YOLO(path)
                    model_found = True
                    break
                except Exception as ex:
                    logger.error(f"Error loading model from {path}: {ex}")

        if not model_found:
            raise FileNotFoundError(
                f"Could not find model {self.model_path} in any of: {model_paths}"
            )

        # Configure model
        logger.info(f"Model loaded successfully: {self.model.names}")

        # Set classes to detect (filtering happens later for flexibility)
        logger.info(f"Classes to detect: {SUPPORTED_CLASSES}")
        logger.info(f"Class mapping: {CLASS_MAP}")

    def _open_stream(self) -> bool:
        """Open the video stream using RTSPStreamProcessor."""
        if not self.stream_url:
            logger.error(f"No stream URL provided for detector {self.detector_id}")
            return False

        try:
            # Parse URL to mask password in logs
            try:
                parsed_url = urllib.parse.urlparse(self.stream_url)
                netloc = parsed_url.netloc

                # Log masked URL (hide password)
                if "@" in netloc:
                    userpass, hostport = netloc.split("@", 1)
                    if ":" in userpass:
                        user, _ = userpass.split(":", 1)
                        masked_netloc = f"{user}:****@{hostport}"

                    masked_url = (
                        f"{parsed_url.scheme}://{masked_netloc}{parsed_url.path}"
                    )
                    logger.info(f"Opening stream: {masked_url}")
                else:
                    logger.info(f"Opening stream: {self.stream_url}")
            except Exception:
                logger.info("Opening stream (URL parsing failed)")

            # Stop and clean up any existing stream processor
            if self.stream_processor is not None:
                logger.info(
                    f"Stopping existing stream processor for {self.detector_id}"
                )
                self.stream_processor.stop()
                self.stream_processor = None

            # For RTSP streams, try to use TCP transport by appending to URL first
            url_to_use = self.stream_url
            if self.stream_url.startswith("rtsp://"):
                # If URL doesn't already have transport parameter, add it
                if "?transport=" not in self.stream_url.lower():
                    url_to_use = f"{self.stream_url}?transport=tcp"
                    logger.info("Using TCP transport parameter in URL")

            # Create a new stream processor with appropriate settings
            self.stream_processor = RTSPStreamProcessor(
                rtsp_url=url_to_use,
                process_nth_frame=self.frame_skip_rate,
                reconnect_delay=self.stream_reconnect_delay,
                max_stored_frames=self.detection_frame_count,
            )

            # Start the stream processor
            logger.info(f"Starting stream processor for {self.detector_id}")
            if not self.stream_processor.start():
                logger.error(f"Failed to start stream processor for {self.detector_id}")
                return False

            # Successfully started the stream processor
            logger.info(f"Stream processor started successfully for {self.detector_id}")
            return True

        except Exception as e:
            logger.error(f"Error opening stream for {self.detector_id}: {e}")
            return False

    # The _stream_monitor_loop function has been removed
    # This functionality is now handled by the RTSPStreamProcessor class

    def perform_detection(self) -> bool:
        """
        Perform detection on the current frame.
        This is called on each poll request rather than continuously.

        Returns:
            bool: True if detection was successful, False otherwise
        """
        if (
            not self.is_running
            or self.stream_processor is None
            or not self.stream_processor.is_running()
        ):
            logger.warning(
                f"Cannot perform detection: detector {self.detector_id} not ready"
            )
            return False

        try:
            # Get multiple frames for detection
            frames = self.stream_processor.get_frames(self.detection_frame_count)

            if not frames or len(frames) == 0:
                logger.warning(
                    f"No frames available for detection from {self.detector_id}"
                )
                return False
                
            # Use the most recent frame as the primary frame
            frame = frames[0]

            # Log that we received frames for processing
            logger.debug(f"Got {len(frames)} frames for detection processing (primary frame: {frame.shape})")

            # Update connection status
            self.connection_status = "connected"

            # Increment frame counter
            self.frame_count += 1

            # Get frame dimensions for logging
            original_height, original_width = frame.shape[:2]

            # YOLO models have a stride requirement - dimensions should be multiples of stride
            # Default stride is 32 for most YOLO models
            STRIDE = 32

            # Adjust width and height to be multiples of STRIDE
            adjusted_width = (self.input_width // STRIDE) * STRIDE
            adjusted_height = (self.input_height // STRIDE) * STRIDE

            # Ensure we have at least one stride worth of pixels
            adjusted_width = max(adjusted_width, STRIDE)
            adjusted_height = max(adjusted_height, STRIDE)

            # Set the adjusted dimensions early for safety
            self.adjusted_dimensions = (adjusted_width, adjusted_height)

            if (
                adjusted_width != self.input_width
                or adjusted_height != self.input_height
            ):
                logger.info(
                    f"Adjusting input size from {self.input_width}x{self.input_height} to "
                    f"{adjusted_width}x{adjusted_height} (multiple of stride {STRIDE}) for detector {self.detector_id}"
                )
            else:
                logger.debug(
                    f"Using detection with configured size: {self.input_width}x{self.input_height} for detector {self.detector_id}"
                )

            # Explicitly resize the frame to the adjusted dimensions before detection
            resized_frame = cv2.resize(
                frame, (adjusted_width, adjusted_height), interpolation=cv2.INTER_LINEAR
            )
            logger.debug(
                f"Resized frame to {adjusted_width}x{adjusted_height} for detection"
            )

            # Process all frames
            all_frames_results = []
            all_annotated_frames = []
            total_inference_time = 0
            
            for idx, current_frame in enumerate(frames):
                # Resize the current frame
                current_resized = cv2.resize(
                    current_frame, (adjusted_width, adjusted_height), interpolation=cv2.INTER_LINEAR
                )
                
                # Pass the pre-resized frame to YOLO model
                start_time = time.time()
                current_results = self.model(
                    current_resized,
                    conf=self.confidence_threshold,
                    # No need to specify imgsz since we've already resized the frame
                )
                frame_inference_time = (time.time() - start_time) * 1000  # in milliseconds
                total_inference_time += frame_inference_time
                
                # Create a copy for annotation
                current_annotated = current_resized.copy()
                
                # Process results for this frame
                frame_people_detected = False
                frame_pets_detected = False
                frame_people_count = 0
                frame_pet_count = 0
                frame_detected_objects = {}
                
                if len(current_results) > 0:
                    # Extract detections
                    result = current_results[0]  # First batch result
                    boxes = result.boxes
                    
                    for box in boxes:
                        cls_id = int(box.cls.item())
                        confidence = box.conf.item()

                        # Only process supported classes with sufficient confidence
                        if (
                            cls_id in SUPPORTED_CLASSES
                            and confidence >= self.confidence_threshold
                        ):
                            class_name = CLASS_MAP.get(cls_id, f"class_{cls_id}")

                            # Count in detected objects
                            if class_name not in frame_detected_objects:
                                frame_detected_objects[class_name] = 0
                            frame_detected_objects[class_name] += 1

                            if cls_id == 0:  # Person
                                frame_people_detected = True
                                frame_people_count += 1
                            elif cls_id in [15, 16, 17]:  # Bird, cat, dog
                                frame_pets_detected = True
                                frame_pet_count += 1

                            # Get box coordinates and draw on the frame
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Ensure coordinates are within bounds
                            annotated_height, annotated_width = current_annotated.shape[:2]
                            x1 = max(0, min(annotated_width - 1, x1))
                            y1 = max(0, min(annotated_height - 1, y1))
                            x2 = max(0, min(annotated_width - 1, x2))
                            y2 = max(0, min(annotated_height - 1, y2))
                            
                            # Draw rectangle
                            color = (0, 255, 0) if cls_id == 0 else (0, 165, 255)
                            cv2.rectangle(current_annotated, (x1, y1), (x2, y2), color, 2)
                            
                            # Prepare label with class name and confidence
                            label = f"{class_name} {confidence:.2f}"
                            text_size, _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                            )
                            
                            # Draw label background
                            cv2.rectangle(
                                current_annotated,
                                (x1, y1 - text_size[1] - 5),
                                (x1 + text_size[0], y1),
                                color,
                                -1,
                            )
                            
                            # Draw label text
                            cv2.putText(
                                current_annotated,
                                label,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 0),
                                2,
                            )
                
                # Create function to add text with background
                def add_text_with_background(image, text, position, font, font_scale, text_color, text_thickness):
                    # Get text size
                    text_size, _ = cv2.getTextSize(text, font, font_scale, text_thickness)
                    text_w, text_h = text_size
                    x, y = position
                    
                    # Draw background rectangle (slightly larger than text)
                    padding = 5
                    cv2.rectangle(
                        image,
                        (x - padding, y - text_h - padding),
                        (x + text_w + padding, y + padding),
                        (0, 0, 0),
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        image,
                        text,
                        position,
                        font,
                        font_scale,
                        text_color,
                        text_thickness
                    )                
                
                # Add a frame number and timestamp to the annotated frame
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                add_text_with_background(
                    current_annotated,
                    f"Frame {idx+1}/{len(frames)} - {current_time}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Add detection stats to the frame
                add_text_with_background(
                    current_annotated,
                    f"People: {frame_people_count}, Pets: {frame_pet_count}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, 
                    (0, 255, 0),
                    2
                )
                
                # Add inference time
                add_text_with_background(
                    current_annotated,
                    f"Inference: {frame_inference_time:.1f}ms",
                    (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Store results for this frame
                all_frames_results.append({
                    "people_detected": frame_people_detected,
                    "pets_detected": frame_pets_detected,
                    "people_count": frame_people_count,
                    "pet_count": frame_pet_count,
                    "detected_objects": frame_detected_objects,
                    "inference_time": frame_inference_time
                })
                
                all_annotated_frames.append(current_annotated)
            
            # Calculate average inference time
            inference_time = total_inference_time / len(frames)
            
            # Count consistent detections to determine final state
            people_detection_count = sum(1 for r in all_frames_results if r["people_detected"])
            pets_detection_count = sum(1 for r in all_frames_results if r["pets_detected"])
            
            # Determine if we have consistent detections
            people_detected = people_detection_count >= self.consistent_detection_count
            pets_detected = pets_detection_count >= self.consistent_detection_count
            
            # For counts, let's use the median count across all frames
            all_people_counts = [r["people_count"] for r in all_frames_results]
            all_pet_counts = [r["pet_count"] for r in all_frames_results]
            
            # Sort and take the middle value as median
            all_people_counts.sort()
            all_pet_counts.sort()
            people_count = all_people_counts[len(all_people_counts) // 2] if all_people_counts else 0
            pet_count = all_pet_counts[len(all_pet_counts) // 2] if all_pet_counts else 0
            
            # Create a merged detected_objects dictionary from all frames
            detected_objects = {}
            
            # Combine all detected objects from all frames
            for result in all_frames_results:
                for obj_type, count in result["detected_objects"].items():
                    if obj_type not in detected_objects:
                        detected_objects[obj_type] = 0
                    detected_objects[obj_type] += count
            
            # Normalize the counts by dividing by the number of frames
            for obj_type in detected_objects:
                detected_objects[obj_type] = detected_objects[obj_type] // len(frames)
            
            # Create a combined visualization with all frames
            # For single frame case, just use the first annotated frame
            if len(all_annotated_frames) == 1:
                annotated_frame = all_annotated_frames[0]
            else:
                # Create a grid layout to show all frames
                # Determine grid dimensions (trying for square-ish layout)
                grid_size = int(np.ceil(np.sqrt(len(all_annotated_frames))))
                rows = grid_size
                cols = grid_size
                
                # Determine the size of each grid cell based on the frame dimensions
                cell_height, cell_width = all_annotated_frames[0].shape[:2]
                
                # Create the grid canvas
                grid_height = rows * cell_height
                grid_width = cols * cell_width
                grid_canvas = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
                
                # Place each frame in the grid
                for i, frame in enumerate(all_annotated_frames):
                    if i >= rows * cols:
                        break  # Skip if we run out of grid cells
                        
                    row = i // cols
                    col = i % cols
                    
                    y_start = row * cell_height
                    y_end = y_start + cell_height
                    x_start = col * cell_width
                    x_end = x_start + cell_width
                    
                    grid_canvas[y_start:y_end, x_start:x_end] = frame
                
                # Add a summary overlay across the bottom of all frames
                # Add a bar at the bottom with summary information
                status_bar_height = 120
                status_bar = np.zeros((status_bar_height, grid_width, 3), dtype=np.uint8)
                
                # Add detection summary to the status bar
                cv2.putText(
                    status_bar,
                    f"Overall detection: People {people_detection_count}/{len(frames)}, " +
                    f"Pets {pets_detection_count}/{len(frames)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
                
                # Show detection thresholds
                cv2.putText(
                    status_bar,
                    f"Required consistent detections: {self.consistent_detection_count}/{len(frames)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
                
                # Show detection status and counts
                status_color = (0, 255, 0) if people_detected or pets_detected else (0, 0, 255)
                cv2.putText(
                    status_bar,
                    f"Final: People={people_detected}({people_count}), Pets={pets_detected}({pet_count})",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2
                )
                
                # Combine the grid and status bar
                annotated_frame = np.vstack((grid_canvas, status_bar))

            # Update detection state
            self.people_detected = people_detected
            self.pets_detected = pets_detected
            self.people_count = people_count
            self.pet_count = pet_count
            self.last_detection_time = time.time()
            self.last_update_time = time.time()
            self.inference_time = inference_time
            self.detected_objects = detected_objects
            # Store original frame dimensions (width x height for consistency)
            self.frame_dimensions = (original_width, original_height)

            # Store the adjusted dimensions that were actually used
            self.adjusted_dimensions = (adjusted_width, adjusted_height)

            # Store the image in the global processed_images dict for the /jpeg endpoint
            with processed_images_lock:
                # Encode as high-quality JPEG for larger displays
                _, jpg_data = cv2.imencode(
                    ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                processed_images[self.detector_id] = jpg_data.tobytes()
                logger.debug(
                    f"Updated processed image for detector {self.detector_id}: {len(jpg_data)} bytes, people: {people_count}, pets: {pet_count}"
                )

            # Create detailed detection report
            detection_report = ", ".join(
                [f"{count} {obj}" for obj, count in detected_objects.items()]
            )
            if not detection_report:
                detection_report = "nothing detected"

            # Log detection with detailed information
            logger.info(
                f"Detection for {self.detector_id}: "
                f"people={people_detected}({people_count}) [{people_detection_count}/{len(frames)}], "
                f"pets={pets_detected}({pet_count}) [{pets_detection_count}/{len(frames)}], "
                f"consistent threshold={self.consistent_detection_count}, "
                f"objects: {detection_report}, "
                f"time: {inference_time:.1f}ms, "
                f"frames processed: {len(frames)}/{self.detection_frame_count}, "
                f"original size: {original_width}x{original_height}, "
                f"processed size: {adjusted_width}x{adjusted_height}"
            )

            return True

        except torch.cuda.OutOfMemoryError as cuda_err:
            logger.error(f"CUDA out of memory error during detection: {cuda_err}")
            self.error_counts["cuda"] += 1
            return False

        except MemoryError as mem_err:
            logger.error(f"Memory error during detection: {mem_err}")
            self.error_counts["memory"] += 1
            return False

        except cv2.error as cv_err:
            logger.error(f"OpenCV error during detection: {cv_err}")
            self.error_counts["stream"] += 1
            return False

        except Exception as ex:
            logger.error(f"Error during detection: {ex}", exc_info=True)
            self.error_counts["other"] += 1
            return False

    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info(f"Cleaning up detector {self.detector_id}")

        # Stop and clean up stream processor
        if self.stream_processor is not None:
            try:
                self.stream_processor.stop()
            except Exception as e:
                logger.error(f"Error stopping stream processor: {e}")
            self.stream_processor = None

        # Set status flags
        self.is_running = False
        self.connection_status = "disconnected"

    def shutdown(self) -> None:
        """Shut down the detector."""
        logger.info(f"Shutting down detector {self.detector_id}")

        # Signal thread to stop
        self.stop_event.set()

        # No need to wait for detection_thread as it's been replaced by RTSPStreamProcessor

        # Force cleanup
        self._cleanup()

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the detector."""
        # Update last client request time
        self.last_client_request = time.time()

        return {
            "detector_id": self.detector_id,
            "name": self.name,
            "stream_url": self.stream_url,
            "model": self.model_name,
            "input_size": f"{self.input_width}x{self.input_height}",
            "detection_interval": self.detection_interval,
            "confidence_threshold": self.confidence_threshold,
            "frame_skip_rate": self.frame_skip_rate,
            "detection_frame_count": self.detection_frame_count,
            "consistent_detection_count": self.consistent_detection_count,
            "is_running": self.is_running,
            "connection_status": self.connection_status,
            "device": self.device,
            "human_detected": self.people_detected,
            "pet_detected": self.pets_detected,
            "human_count": self.people_count,
            "pet_count": self.pet_count,
            "last_update": self.last_update_time,
            "last_detection": self.last_detection_time,
            "frame_count": self.frame_count,
            "optimization_level": self.optimization_level,
            "degradation_level": self.degradation_level,
            "auto_optimization": self.auto_optimization,
            "requested_resolution": f"{self.input_width}x{self.input_height}",  # User requested resolution
            "actual_resolution": (
                f"{self.adjusted_dimensions[0]}x{self.adjusted_dimensions[1]}"
                if hasattr(self, "adjusted_dimensions")
                else f"{self.input_width}x{self.input_height}"
            ),  # Adjusted for stride
            "inference_time_ms": (
                round(self.inference_time, 1) if hasattr(self, "inference_time") else 0
            ),
            "detected_objects": (
                self.detected_objects if hasattr(self, "detected_objects") else {}
            ),
            "original_frame_dimensions": (
                self.frame_dimensions if hasattr(self, "frame_dimensions") else (0, 0)
            ),
        }


def create_or_update_detector(
    detector_id: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create or update a detector with the given configuration."""
    with detector_lock:
        # Check for auto-configuration mode (both auto_config and use_auto_optimization for compatibility)
        auto_config = config.get(
            "auto_config", config.get("use_auto_optimization", False)
        )

        if detector_id in detectors:
            # Update existing detector
            detector = detectors[detector_id]
            logger.info(f"Updating existing detector {detector_id}")

            settings_changed = False

            # Save auto_optimization flag in detector
            detector.auto_optimization = auto_config

            # Track which settings are being updated
            updated_settings = []

            # Update detector configuration
            if config.get("stream_url") and config["stream_url"] != detector.stream_url:
                detector.stream_url = config["stream_url"]
                updated_settings.append("stream_url")
                # Stream URL change requires reconnection
                settings_changed = True

            if config.get("name") and config["name"] != detector.name:
                detector.name = config["name"]
                updated_settings.append("name")

            if config.get("model") and config["model"] != detector.model_name:
                # Model changes require reinitialization, which we'll log but not actually reinitialize
                detector.model_name = config["model"]
                updated_settings.append("model (requires restart)")

            if config.get("input_size"):
                try:
                    width, height = map(int, config["input_size"].split("x"))
                    if width != detector.input_width or height != detector.input_height:
                        detector.input_width = width
                        detector.input_height = height
                        updated_settings.append("input_size")
                except Exception as ex:
                    logger.warning(
                        f"Invalid input_size format: {config.get('input_size')} - {ex}"
                    )

            if config.get("detection_interval") is not None:
                try:
                    interval = float(config["detection_interval"])
                    if interval != detector.detection_interval:
                        detector.detection_interval = interval
                        updated_settings.append("detection_interval")
                except Exception as ex:
                    logger.warning(
                        f"Invalid detection_interval: {config.get('detection_interval')} - {ex}"
                    )

            if config.get("confidence_threshold") is not None:
                try:
                    threshold = float(config["confidence_threshold"])
                    if threshold != detector.confidence_threshold:
                        detector.confidence_threshold = threshold
                        updated_settings.append("confidence_threshold")
                except Exception as ex:
                    logger.warning(
                        f"Invalid confidence_threshold: {config.get('confidence_threshold')} - {ex}"
                    )

            if config.get("frame_skip_rate") is not None:
                try:
                    skip_rate = int(config["frame_skip_rate"])
                    if skip_rate != detector.frame_skip_rate:
                        detector.frame_skip_rate = skip_rate
                        updated_settings.append("frame_skip_rate")
                except Exception as ex:
                    logger.warning(
                        f"Invalid frame_skip_rate: {config.get('frame_skip_rate')} - {ex}"
                    )
                    
            if config.get("detection_frame_count") is not None:
                try:
                    frame_count = int(config["detection_frame_count"])
                    if frame_count != detector.detection_frame_count:
                        detector.detection_frame_count = frame_count
                        updated_settings.append("detection_frame_count")
                except Exception as ex:
                    logger.warning(
                        f"Invalid detection_frame_count: {config.get('detection_frame_count')} - {ex}"
                    )
                    
            if config.get("consistent_detection_count") is not None:
                try:
                    consistent_count = int(config["consistent_detection_count"])
                    if consistent_count != detector.consistent_detection_count:
                        detector.consistent_detection_count = consistent_count
                        updated_settings.append("consistent_detection_count")
                except Exception as ex:
                    logger.warning(
                        f"Invalid consistent_detection_count: {config.get('consistent_detection_count')} - {ex}"
                    )

            # Auto configure if requested and enabled
            if auto_config and detector.auto_optimization:
                # Determine if we need performance optimization or quality improvement based on resources
                try:
                    # Check CPU/GPU usage
                    cpu_usage = psutil.cpu_percent()

                    if (
                        hasattr(torch.cuda, "is_available")
                        and torch.cuda.is_available()
                    ):
                        gpu_usage = (
                            torch.cuda.memory_allocated()
                            / torch.cuda.get_device_properties(0).total_memory
                            * 100
                        )
                    else:
                        gpu_usage = 0

                    # Get current connection status
                    connection_ok = detector.connection_status == "connected"

                    if not connection_ok:
                        # Connection issues - reduce quality for better streaming
                        if detector.frame_skip_rate < 10:
                            detector.frame_skip_rate += 1
                            updated_settings.append("frame_skip_rate (auto increased)")

                        if detector.detection_interval < 20:
                            detector.detection_interval += 2
                            updated_settings.append(
                                "detection_interval (auto increased)"
                            )

                    elif cpu_usage > 80 or gpu_usage > 80:
                        # High resource usage - reduce quality
                        if detector.frame_skip_rate < 8:
                            detector.frame_skip_rate += 1
                            updated_settings.append("frame_skip_rate (auto increased)")

                        if detector.detection_interval < 15:
                            detector.detection_interval += 1
                            updated_settings.append(
                                "detection_interval (auto increased)"
                            )
                    elif cpu_usage < 40 and gpu_usage < 40:
                        # Low resource usage - increase quality
                        if detector.frame_skip_rate > 2:
                            detector.frame_skip_rate -= 1
                            updated_settings.append("frame_skip_rate (auto decreased)")

                        if detector.detection_interval > 3:
                            detector.detection_interval -= 1
                            updated_settings.append(
                                "detection_interval (auto decreased)"
                            )

                    # Log auto configuration changes
                    if updated_settings:
                        logger.info(
                            f"Auto-configured detector {detector_id}: {', '.join(updated_settings)}"
                        )

                except Exception as ex:
                    logger.warning(f"Auto-configuration error: {ex}")
            elif auto_config and not detector.auto_optimization:
                # Auto-config requested but detector doesn't have it enabled yet
                detector.auto_optimization = True
                updated_settings.append("auto_optimization (enabled)")
                logger.info(f"Enabled auto-optimization for detector {detector_id}")

            # Update last client request time to prevent idle shutdown
            detector.last_client_request = time.time()

            # Log the updated settings
            if updated_settings:
                logger.info(
                    f"Updated detector {detector_id} settings: {', '.join(updated_settings)}"
                )

                # If stream URL changed, try to reconnect immediately
                if "stream_url" in updated_settings and detector.is_running:
                    logger.info(
                        f"Stream URL changed, attempting to reconnect for {detector_id}"
                    )
                    if detector._open_stream():
                        logger.info(
                            f"Successfully reconnected to new stream for {detector_id}"
                        )
                    else:
                        logger.warning(
                            f"Failed to connect to new stream for {detector_id}, will retry in detection loop"
                        )

            return {
                "status": "success",
                "message": f"Detector updated"
                + (f" ({', '.join(updated_settings)})" if updated_settings else ""),
                "detector_id": detector_id,
                "state": detector.get_state(),
            }
        else:
            # Create new detector
            logger.info(f"Creating new detector {detector_id}")

            # Apply default configuration if not provided
            if not config.get("stream_url"):
                logger.error(f"No stream URL provided for detector {detector_id}")
                return {
                    "status": "error",
                    "message": "stream_url is required",
                    "detector_id": detector_id,
                }

            # Set default configuration if not provided
            if not config.get("name"):
                config["name"] = f"Detector {detector_id[:8]}"

            if not config.get("model"):
                config["model"] = "yolo11l"

            if not config.get("input_size"):
                config["input_size"] = "640x480"

            if not config.get("detection_interval"):
                config["detection_interval"] = 10

            if not config.get("confidence_threshold"):
                config["confidence_threshold"] = 0.25

            if not config.get("frame_skip_rate"):
                config["frame_skip_rate"] = 5

            # Set auto_optimization flag if provided, otherwise default to False
            if "use_auto_optimization" not in config and "auto_config" not in config:
                config["use_auto_optimization"] = False

            # Log the configuration being used
            logger.info(
                f"Creating detector {detector_id} with configuration: {json.dumps({k: v for k, v in config.items() if k != 'stream_url'})}"
            )
            if "stream_url" in config:
                masked_url = "[URL MASKED]"
                try:
                    url = config["stream_url"]
                    if "@" in url:
                        parsed = urllib.parse.urlparse(url)
                        netloc = parsed.netloc
                        if "@" in netloc:
                            userpass, hostport = netloc.split("@", 1)
                            if ":" in userpass:
                                user, _ = userpass.split(":", 1)
                                masked_netloc = f"{user}:****@{hostport}"
                                masked_url = (
                                    f"{parsed.scheme}://{masked_netloc}{parsed.path}"
                                )
                except:
                    pass
                logger.info(f"Stream URL: {masked_url}")

            detector = YoloDetector(detector_id, config)

            if detector.initialize():
                detectors[detector_id] = detector
                logger.info(f"Successfully initialized detector {detector_id}")

                return {
                    "status": "success",
                    "message": "Detector created",
                    "detector_id": detector_id,
                    "state": detector.get_state(),
                }
            else:
                logger.error(f"Failed to initialize detector {detector_id}")
                return {
                    "status": "error",
                    "message": "Failed to initialize detector",
                    "detector_id": detector_id,
                }


def check_disk_space() -> float:
    """Check disk space and return percentage used."""
    try:
        total, used, free = shutil.disk_usage("/")
        percent_used = (used / total) * 100
        return percent_used
    except Exception as ex:
        logger.error(f"Error checking disk space: {ex}")
        return 0.0


def load_template(template_name: str) -> str:
    """Load a template file from the templates directory.
    
    Args:
        template_name: Name of the template file without path.
        
    Returns:
        The template content as a string.
    """
    try:
        template_path = os.path.join(os.path.dirname(__file__), "templates", template_name)
        with open(template_path, 'r') as f:
            return f.read()
    except Exception as ex:
        logger.error(f"Error loading template {template_name}: {ex}")
        return f"<h1>Error loading template {template_name}</h1><p>{str(ex)}</p>"


def render_template(template_name: str, context: Dict[str, Any]) -> str:
    """Render a template with the given context.
    
    Args:
        template_name: Name of the template file without path.
        context: Dictionary of values to replace in the template.
        
    Returns:
        The rendered template as a string.
    """
    template = load_template(template_name)
    
    # Handle simple variable replacements
    for key, value in context.items():
        placeholder = f"{{{{{key}}}}}"
        template = template.replace(placeholder, str(value))
    
    # Handle conditional blocks (#if, #each)
    # #if block processing
    for match in ["{{#if no_detectors}}", "{{#if no_detectors}}\n"]:
        if match in template:
            parts = template.split(match)
            if len(parts) > 1:
                before = parts[0]
                after_parts = parts[1].split("{{else}}")
                if len(after_parts) > 1:
                    if_content = after_parts[0]
                    else_content = after_parts[1].split("{{/if}}")[0]
                    remaining = parts[1].split("{{/if}}")[1]
                    
                    if context.get("no_detectors", False):
                        template = before + if_content + remaining
                    else:
                        template = before + else_content + remaining
    
    # #each block processing
    if "{{#each detectors}}" in template:
        parts = template.split("{{#each detectors}}")
        if len(parts) > 1:
            before = parts[0]
            each_content = parts[1].split("{{/each}}")[0]
            after = parts[1].split("{{/each}}")[1]
            
            items_html = ""
            for item in context.get("detectors", []):
                item_html = each_content
                for key, value in item.items():
                    placeholder = f"{{{{{key}}}}}"
                    item_html = item_html.replace(placeholder, str(value))
                items_html += item_html
                
            template = before + items_html + after
    
    return template


class YoloHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for YOLO detectors."""

    # Add a more robust error handler
    def handle_one_request(self) -> None:
        """Handle a single HTTP request with improved error handling."""
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError) as e:
            logger.warning(f"Connection error with client {self.client_address}: {e}")
        except Exception as e:
            logger.error(
                f"Error handling request from {self.client_address}: {e}", exc_info=True
            )

    def _check_auth(self) -> bool:
        """Check if the request has valid authentication.

        Returns:
            bool: True if authentication is valid or disabled, False otherwise.
        """
        # If auth is disabled, always return True
        if not AUTH_ENABLED:
            return True

        # Check if path requires authentication
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # If path is not protected, no auth needed
        if path not in PROTECTED_PATHS:
            return True

        # Check for auth header
        auth_header = self.headers.get("Authorization")
        if not auth_header:
            # No auth header, authentication failed
            return False

        # Parse and validate auth header
        try:
            # Header format: "Basic base64encoded(username:password)"
            auth_type, auth_info = auth_header.split(" ", 1)

            if auth_type.lower() != "basic":
                # We only support Basic auth
                return False

            # Decode the base64 auth info
            auth_info_bytes = base64.b64decode(auth_info)
            auth_info_text = auth_info_bytes.decode("utf-8")
            username, password = auth_info_text.split(":", 1)

            # Check credentials
            return username == AUTH_USERNAME and password == AUTH_PASSWORD
        except Exception as ex:
            logger.warning(f"Authentication error: {ex}")
            return False

    def _send_response(self, data: Dict[str, Any], status_code: int = 200) -> None:
        """Send a JSON response."""
        try:
            # Prepare the response
            response_json = json.dumps(data)
            response_bytes = response_json.encode("utf-8")

            # Log the response
            client_addr = self.client_address[0]
            request_method = self.command
            request_path = self.path

            # Format the response for logging with some formatting
            log_data = data
            if isinstance(data, dict) and "state" in data:
                # If this contains detector state, mask any sensitive stream URLs
                state = data.get("state", {})
                if isinstance(state, dict) and "stream_url" in state:
                    stream_url = state.get("stream_url", "")
                    if "@" in stream_url:
                        # Parse and mask password
                        parsed_url = urllib.parse.urlparse(stream_url)
                        netloc = parsed_url.netloc
                        if "@" in netloc:
                            userpass, hostport = netloc.split("@", 1)
                            if ":" in userpass:
                                user, _ = userpass.split(":", 1)
                                masked_netloc = f"{user}:****@{hostport}"
                                masked_url = f"{parsed_url.scheme}://{masked_netloc}{parsed_url.path}"
                                # Create masked version for logging
                                masked_state = state.copy()
                                masked_state["stream_url"] = masked_url
                                log_data = data.copy()
                                log_data["state"] = masked_state

            # Log summary version or full version based on debug level
            response_log = json.dumps(log_data, indent=2)
            if len(response_log) > 1000:
                # For large responses, log a summary version
                if isinstance(data, dict):
                    summary = {"status": data.get("status")}
                    if "detector_id" in data:
                        summary["detector_id"] = data.get("detector_id")
                    if "message" in data:
                        summary["message"] = data.get("message")
                    if "state" in data and isinstance(data["state"], dict):
                        state = data["state"]
                        state_summary = {}
                        for key in [
                            "connection_status",
                            "human_detected",
                            "pet_detected",
                            "human_count",
                            "pet_count",
                            "is_running",
                        ]:
                            if key in state:
                                state_summary[key] = state[key]
                        summary["state"] = state_summary
                    response_log = json.dumps(summary, indent=2)
                else:
                    response_log = f"{response_log[:997]}..."

            logger.info(
                f"RESPONSE to {client_addr} - {request_method} {request_path} - Status: {status_code}"
            )
            logger.info(f"RESPONSE BODY: {response_log}")

            # Send the actual response
            try:
                self.send_response(status_code)
                self.send_header("Content-type", "application/json")
                self.send_header("Content-Length", str(len(response_bytes)))
                self.end_headers()
                self.wfile.write(response_bytes)
            except (BrokenPipeError, ConnectionResetError) as e:
                logger.warning(
                    f"Connection closed by client during response to {client_addr}: {e}"
                )
            except Exception as e:
                logger.error(f"Error sending response to {client_addr}: {e}")
        except Exception as e:
            logger.error(f"Error preparing response: {e}", exc_info=True)
            # Try to send a simple error response
            try:
                error_response = json.dumps(
                    {"status": "error", "message": "Internal server error"}
                ).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.send_header("Content-Length", str(len(error_response)))
                self.end_headers()
                self.wfile.write(error_response)
            except:
                # Last resort - we can't send a response
                logger.error("Failed to send error response")
                pass

    def _send_error_response(self, message: str, status_code: int = 400) -> None:
        """Send an error response."""
        self._send_response({"status": "error", "message": message}, status_code)

    def _parse_json_body(self) -> Dict[str, Any]:
        """Parse JSON request body."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                return {}

            try:
                # Read the raw body data
                try:
                    body_raw = self.rfile.read(content_length).decode("utf-8")
                except (ConnectionResetError, BrokenPipeError) as e:
                    logger.warning(
                        f"Connection closed by client while reading request body: {e}"
                    )
                    return {}

                # Log the raw request body
                client_addr = self.client_address[0]
                request_method = self.command
                request_path = self.path
                logger.info(
                    f"REQUEST from {client_addr} - {request_method} {request_path}"
                )

                # Mask any password in stream URLs for security
                body_log = body_raw
                try:
                    # Parse the JSON
                    body_json = json.loads(body_raw)

                    # If this contains a config with stream_url that has a password, mask it
                    if isinstance(body_json, dict) and "config" in body_json:
                        config = body_json.get("config", {})
                        if isinstance(config, dict) and "stream_url" in config:
                            stream_url = config.get("stream_url", "")
                            if "@" in stream_url:
                                # Parse and mask password
                                parsed_url = urllib.parse.urlparse(stream_url)
                                netloc = parsed_url.netloc
                                if "@" in netloc:
                                    userpass, hostport = netloc.split("@", 1)
                                    if ":" in userpass:
                                        user, _ = userpass.split(":", 1)
                                        masked_netloc = f"{user}:****@{hostport}"
                                        masked_url = f"{parsed_url.scheme}://{masked_netloc}{parsed_url.path}"
                                        # Create a new masked JSON for logging
                                        masked_config = config.copy()
                                        masked_config["stream_url"] = masked_url
                                        masked_body = body_json.copy()
                                        masked_body["config"] = masked_config
                                        body_log = json.dumps(masked_body, indent=2)
                except Exception as mask_err:
                    # If any error in masking, just use the original but truncated
                    logger.debug(f"Error while masking credentials: {mask_err}")
                    if len(body_log) > 1000:
                        body_log = body_log[:997] + "..."

                logger.info(f"REQUEST BODY: {body_log}")

                try:
                    return json.loads(body_raw)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")
                    logger.debug(f"Raw request body: {body_raw[:200]}")
                    return {}
            except Exception as read_err:
                logger.error(f"Error reading request body: {read_err}")
                return {}
        except Exception as e:
            logger.error(f"Unexpected error in _parse_json_body: {e}", exc_info=True)
            return {}

    def do_GET(self) -> None:
        """Handle GET requests."""
        try:
            # Authentication removed as requested

            # Parse URL and query parameters
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            query = parse_qs(parsed_url.query)

            # Get detector_id from query parameters
            detector_id = query.get("detector_id", [""])[0]

            if path == "/":
                # Root endpoint - show HTML index page with list of detectors
                try:
                    with detector_lock:
                        # Prepare data for index template
                        context = {
                            "detector_count": str(len(detectors)),
                            "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "no_detectors": len(detectors) == 0
                        }
                        
                        if detectors:
                            # Create a list of detector data for the template
                            detector_list = []
                            for d_id, detector in detectors.items():
                                state = detector.get_state()
                                
                                # Format detection information
                                detection_info = ""
                                if hasattr(detector, "detected_objects") and detector.detected_objects:
                                    items = [
                                        f"{count} {obj}"
                                        for obj, count in detector.detected_objects.items()
                                    ]
                                    if items:
                                        detection_info = f"Detected: {', '.join(items)}"
                                
                                detector_list.append({
                                    "detector_id": d_id,
                                    "name": detector.name,
                                    "connection_status": detector.connection_status,
                                    "connection_status_upper": detector.connection_status.upper(),
                                    "model_name": detector.model_name,
                                    "input_width": detector.input_width,
                                    "input_height": detector.input_height,
                                    "people_count": detector.people_count,
                                    "pet_count": detector.pet_count,
                                    "detection_info": detection_info
                                })
                            
                            context["detectors"] = detector_list
                        
                        # Render the template
                        html = render_template("index.html", context)

                        # Send the HTML response
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html")
                        self.send_header("Content-Length", str(len(html.encode())))
                        self.end_headers()
                        self.wfile.write(html.encode())
                        return
                except Exception as ex:
                    logger.error(f"Error generating index page: {ex}", exc_info=True)
                    self._send_error_response(
                        f"Error generating index page: {str(ex)}", 500
                    )
            elif path == "/health":
                # Health check endpoint
                self._send_response(
                    {
                        "status": "ok",
                        "message": "Server is running",
                        "detector_count": len(detectors),
                    }
                )
            elif path == "/state" and detector_id:
                # Get detector state
                try:
                    with detector_lock:
                        if detector_id in detectors:
                            detector = detectors[detector_id]
                            self._send_response(
                                {
                                    "status": "success",
                                    "detector_id": detector_id,
                                    "state": detector.get_state(),
                                }
                            )
                        else:
                            self._send_error_response(
                                f"Detector {detector_id} not found", 404
                            )
                except Exception as ex:
                    logger.error(f"Error getting detector state: {ex}", exc_info=True)
                    self._send_error_response(
                        f"Error getting detector state: {str(ex)}", 500
                    )
            elif path == "/detectors":
                # List all detectors
                try:
                    with detector_lock:
                        detector_list = []
                        for detector_id, detector in detectors.items():
                            detector_list.append(
                                {
                                    "detector_id": detector_id,
                                    "name": detector.name,
                                    "is_running": detector.is_running,
                                    "connection_status": detector.connection_status,
                                }
                            )
                        self._send_response(
                            {"status": "success", "detectors": detector_list}
                        )
                except Exception as ex:
                    logger.error(f"Error listing detectors: {ex}", exc_info=True)
                    self._send_error_response(
                        f"Error listing detectors: {str(ex)}", 500
                    )
            elif path == "/jpeg" and detector_id:
                # Simple JPEG endpoint that serves the processed image stored in processed_images
                try:
                    # Check if detector exists
                    with detector_lock:
                        if detector_id not in detectors:
                            self._send_error_response(
                                f"Detector {detector_id} not found", 404
                            )
                            return

                    # Get the pre-encoded JPEG data from processed_images
                    with processed_images_lock:
                        if detector_id not in processed_images:
                            logger.debug(
                                f"No processed image found for detector {detector_id}, creating placeholder"
                            )
                            # If no processed image exists yet, create a larger placeholder
                            placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
                            cv2.putText(
                                placeholder,
                                "Waiting for detection...",
                                (400, 360),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,
                                (255, 255, 255),
                                3,
                            )
                            _, jpg_data = cv2.imencode(
                                ".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 95]
                            )
                            jpg_bytes = jpg_data.tobytes()
                        else:
                            # Use the pre-encoded image stored in processed_images
                            jpg_bytes = processed_images[detector_id]
                            logger.debug(
                                f"Serving processed image for detector {detector_id}: {len(jpg_bytes)} bytes"
                            )

                    # Send the JPEG - simple headers
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg_bytes)))
                    self.send_header(
                        "Cache-Control", "no-cache, no-store, must-revalidate"
                    )
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.end_headers()
                    self.wfile.write(jpg_bytes)
                    return

                except Exception as ex:
                    logger.error(f"Error sending JPEG frame: {ex}", exc_info=True)
                    self._send_error_response(
                        f"Error sending JPEG frame: {str(ex)}", 500
                    )

            elif path == "/view" and detector_id:
                # Return an HTML page with the stream embedded
                try:
                    with detector_lock:
                        if detector_id not in detectors:
                            self._send_error_response(
                                f"Detector {detector_id} not found", 404
                            )
                            return

                        detector = detectors[detector_id]
                        
                        # Prepare context for the view template
                        context = {
                            "detector_id": detector_id,
                            "detector_name": detector.name,
                            "model_name": detector.model_name,
                            "connection_status": detector.connection_status,
                            "input_width": detector.input_width,
                            "input_height": detector.input_height,
                            "confidence_threshold": detector.confidence_threshold,
                            "frame_skip_rate": detector.frame_skip_rate,
                            "detection_frame_count": detector.detection_frame_count,
                            "consistent_detection_count": detector.consistent_detection_count
                        }
                        
                        # Render the template
                        html = render_template("view.html", context)

                        # Send the HTML response
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html")
                        self.send_header("Content-Length", str(len(html.encode())))
                        self.end_headers()
                        self.wfile.write(html.encode())
                        return
                except Exception as ex:
                    logger.error(f"Error generating view page: {ex}", exc_info=True)
                    self._send_error_response(
                        f"Error generating view page: {str(ex)}", 500
                    )
            else:
                self._send_error_response("Invalid endpoint", 404)
        except Exception as e:
            logger.error(f"Unhandled exception in do_GET: {e}", exc_info=True)
            try:
                self._send_error_response("Internal server error", 500)
            except Exception as send_err:
                logger.error("Failed to send error response", exc_info=True)

    def do_POST(self) -> None:
        """Handle POST requests."""
        try:
            # We don't check authentication for POST requests since they're used by the Home Assistant integration
            # If you want to add authentication to POST endpoints as well, uncomment and modify the lines below
            # if path in ["/custom_protected_endpoint"]:
            #     if not self._check_auth():
            #         self._send_auth_required()
            #         return

            # Parse URL
            parsed_url = urlparse(self.path)
            path = parsed_url.path

            # Parse request body
            body = self._parse_json_body()

            if path == "/poll":
                try:
                    # Poll endpoint - create or update detector, perform detection, and get state
                    detector_id = body.get("detector_id")

                    if not detector_id:
                        self._send_error_response("detector_id is required")
                        return

                    # Get detector configuration from request
                    config = body.get("config", {})

                    try:
                        # Create or update detector
                        result = create_or_update_detector(detector_id, config)

                        # If detector is successfully created/updated, perform detection
                        if (
                            result.get("status") == "success"
                            and detector_id in detectors
                        ):
                            try:
                                detector = detectors[detector_id]

                                # Perform detection on current frame
                                detection_start = time.time()
                                logger.info(
                                    f"Performing detection for {detector_id} on poll request - interval: {detector.detection_interval}s, auto_optimization: {detector.auto_optimization}"
                                )
                                logger.debug(
                                    f"DEBUG: Processing poll request for {detector_id} - about to call perform_detection()"
                                )

                                try:
                                    detection_success = detector.perform_detection()
                                    detection_time = time.time() - detection_start

                                    if detection_success:
                                        # Get the detailed detection information
                                        detection_info = ""
                                        if (
                                            hasattr(detector, "detected_objects")
                                            and detector.detected_objects
                                        ):
                                            items = [
                                                f"{count} {obj}"
                                                for obj, count in detector.detected_objects.items()
                                            ]
                                            detection_info = (
                                                f" - Found: {', '.join(items)}"
                                            )

                                        # Log with additional info
                                        logger.info(
                                            f"Detection completed in {detection_time:.3f}s for {detector_id}{detection_info}"
                                        )

                                        # Get updated state after detection
                                        result["state"] = detector.get_state()
                                        result["detection_time"] = detection_time
                                    else:
                                        logger.warning(
                                            f"Detection failed for {detector_id}"
                                        )
                                        result["detection_failed"] = True

                                except Exception as detection_err:
                                    logger.error(
                                        f"Error during detection for {detector_id}: {detection_err}",
                                        exc_info=True,
                                    )
                                    result["status"] = "error"
                                    result["message"] = (
                                        f"Detection error: {str(detection_err)}"
                                    )
                                    result["detection_failed"] = True
                            except Exception as detector_err:
                                logger.error(
                                    f"Error accessing detector {detector_id}: {detector_err}",
                                    exc_info=True,
                                )
                                result["status"] = "error"
                                result["message"] = (
                                    f"Detector access error: {str(detector_err)}"
                                )

                        self._send_response(result)
                    except Exception as create_err:
                        logger.error(
                            f"Error creating/updating detector {detector_id}: {create_err}",
                            exc_info=True,
                        )
                        self._send_error_response(
                            f"Error creating/updating detector: {str(create_err)}", 500
                        )
                except Exception as poll_err:
                    logger.error(
                        f"Error processing poll request: {poll_err}", exc_info=True
                    )
                    self._send_error_response(
                        f"Error processing poll request: {str(poll_err)}", 500
                    )

            elif path == "/shutdown":
                try:
                    # Shutdown detector
                    detector_id = body.get("detector_id")

                    if not detector_id:
                        self._send_error_response("detector_id is required")
                        return

                    with detector_lock:
                        if detector_id in detectors:
                            try:
                                detector = detectors[detector_id]
                                detector.shutdown()
                                del detectors[detector_id]
                                self._send_response(
                                    {
                                        "status": "success",
                                        "message": f"Detector {detector_id} shutdown",
                                    }
                                )
                            except Exception as shutdown_err:
                                logger.error(
                                    f"Error shutting down detector {detector_id}: {shutdown_err}",
                                    exc_info=True,
                                )
                                self._send_error_response(
                                    f"Error shutting down detector: {str(shutdown_err)}",
                                    500,
                                )
                        else:
                            self._send_error_response(
                                f"Detector {detector_id} not found", 404
                            )
                except Exception as shutdown_req_err:
                    logger.error(
                        f"Error processing shutdown request: {shutdown_req_err}",
                        exc_info=True,
                    )
                    self._send_error_response(
                        f"Error processing shutdown request: {str(shutdown_req_err)}",
                        500,
                    )
            else:
                self._send_error_response("Invalid endpoint", 404)
        except Exception as e:
            logger.error(f"Unhandled exception in do_POST: {e}", exc_info=True)
            try:
                self._send_error_response("Internal server error", 500)
            except Exception as send_err:
                logger.error("Failed to send error response", exc_info=True)


def start_http_server() -> None:
    """Start the HTTP server."""
    server = None
    try:
        # Import ThreadingMixIn for concurrent request handling
        from socketserver import ThreadingMixIn
        import socket

        # Create a threaded HTTP server for better concurrency
        class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
            """Threaded HTTP server for handling multiple concurrent connections."""

            # Set daemon_threads to True to ensure threads terminate when main thread exits
            daemon_threads = True
            # Allow socket reuse to prevent "Address already in use" errors
            allow_reuse_address = True

        # Customize the HTTP server for better error handling
        class RobustHTTPServer(ThreadedHTTPServer):
            """Enhanced HTTP server with improved error handling."""

            def handle_error(self, request, client_address):
                """Handle errors occurring during request processing."""
                # Extract the client IP
                client_ip = client_address[0] if client_address else "unknown"

                # Log the error with traceback
                logger.error(
                    f"Error processing request from {client_ip}:", exc_info=True
                )

                # Don't close the socket on errors
                return

            def server_bind(self):
                """Override server_bind to set socket timeout."""
                # Set socket options for quicker recovery from closed connections
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # Call the parent method
                super().server_bind()

        # Create the server with robust error handling
        server = RobustHTTPServer(("0.0.0.0", HTTP_PORT), YoloHTTPHandler)
        server.timeout = 60  # Set a timeout for socket operations

        # Log server startup information
        logger.info(f"Starting HTTP server on port {HTTP_PORT}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"OpenCV version: {cv2.__version__}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        # Start the server loop with restart capability
        try:
            logger.info("Server started and ready to handle requests")
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped by keyboard interrupt")
        except Exception as loop_ex:
            logger.error(f"Error in server main loop: {loop_ex}", exc_info=True)

    except OSError as os_err:
        if os_err.errno == 98:  # Address already in use
            logger.error(
                f"Port {HTTP_PORT} is already in use. Is another instance running?"
            )
        else:
            logger.error(f"OS error starting server: {os_err}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Server initialization interrupted by user")
    except Exception as ex:
        logger.error(f"Error starting server: {ex}", exc_info=True)
    finally:
        logger.info("Server shutting down")

        # Shutdown the server properly if it was created
        if server:
            try:
                logger.info("Shutting down HTTP server...")
                server.server_close()
            except Exception as server_err:
                logger.error(f"Error shutting down HTTP server: {server_err}")

        # Cleanup all detectors
        with detector_lock:
            for detector_id, detector in list(detectors.items()):
                try:
                    logger.info(f"Shutting down detector {detector_id}...")
                    detector.shutdown()
                except Exception as det_err:
                    logger.error(
                        f"Error shutting down detector {detector_id}: {det_err}"
                    )
            detectors.clear()

        logger.info("Server shutdown complete")


def handle_signals() -> None:
    """Handle signals for graceful shutdown."""

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down")
        # Cleanup all detectors
        with detector_lock:
            for detector_id, detector in list(detectors.items()):
                try:
                    detector.shutdown()
                except Exception:
                    pass
            detectors.clear()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    # Handle signals for graceful shutdown
    handle_signals()

    # Start the HTTP server
    start_http_server()