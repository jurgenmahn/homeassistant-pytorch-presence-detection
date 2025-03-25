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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("yolo_server.log")],
)
logger = logging.getLogger("yolo_server")

# Constants
HTTP_PORT = int(os.environ.get("HTTP_PORT", 5505))
SUPPORTED_CLASSES = [0, 15, 16, 17]  # person, bird, cat, dog
CLASS_MAP = {0: "person", 15: "bird", 16: "cat", 17: "dog"}

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

        # Auto-optimization flag
        self.auto_optimization = config.get(
            "use_auto_optimization", config.get("auto_config", False)
        )

        # Detection results data
        self.inference_time = 0
        self.detected_objects = {}
        self.frame_dimensions = (0, 0)

        # Frame storage for visualization
        self.last_annotated_frame = None  # Last frame with detection boxes drawn
        self.last_annotated_time = 0  # Timestamp when the annotated frame was created
        self.adjusted_dimensions = (0, 0)

        # Runtime state
        self.is_running = False
        self.detection_thread = None
        self.stop_event = threading.Event()
        self.cap = None
        self.frame_count = 0

        # Detection state
        self.last_detection_time = None
        self.people_detected = False
        self.pets_detected = False
        self.people_count = 0
        self.pet_count = 0
        self.connection_status = "disconnected"
        self.last_frame = None  # Store last processed frame for debugging

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

            # Start stream monitor thread to keep the stream connected
            self.stop_event.clear()
            self.detection_thread = threading.Thread(target=self._stream_monitor_loop)
            if self.detection_thread:  # Type checking validation
                self.detection_thread.daemon = True
                self.detection_thread.start()

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
        """Open the video stream."""
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
                    else:
                        masked_netloc = f"{userpass}@{hostport}"

                    masked_url = (
                        f"{parsed_url.scheme}://{masked_netloc}{parsed_url.path}"
                    )
                    logger.info(f"Opening stream: {masked_url}")
                else:
                    logger.info(f"Opening stream: {self.stream_url}")
            except Exception:
                logger.info("Opening stream (URL parsing failed)")

            # Close any existing stream
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

            # For RTSP streams, try to use TCP transport by appending to URL first
            url_to_use = self.stream_url
            if self.stream_url.startswith("rtsp://"):
                # If URL doesn't already have transport parameter, add it
                if "?transport=" not in self.stream_url.lower():
                    url_to_use = f"{self.stream_url}?transport=tcp"
                    logger.info("Using TCP transport parameter in URL")

            # Open the stream with OpenCV
            self.cap = cv2.VideoCapture(url_to_use)

            # Configure stream parameters if opened successfully
            if self.cap.isOpened():
                # Configure for optimal streaming performance
                self.cap.set(
                    cv2.CAP_PROP_BUFFERSIZE, 3
                )  # Small buffer to reduce latency

                # For RTSP streams, set additional parameters
                if self.stream_url.startswith("rtsp://"):
                    # Try to set preferred codec
                    try:
                        self.cap.set(
                            cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"H264")
                        )
                    except Exception as ex:
                        logger.debug(f"Could not set codec: {ex}")

                    # Try using TCP transport for better reliability
                    # Different OpenCV versions use different constant names
                    try:
                        # Try commonly used property IDs for RTSP transport
                        transport_props = [
                            ("CAP_PROP_RTSP_TRANSPORT", 0),  # Newer OpenCV
                            (78, 0),  # Direct property ID
                            ("CV_CAP_PROP_RTSP_TRANSPORT", 0),  # Older OpenCV
                        ]

                        for prop, value in transport_props:
                            try:
                                if isinstance(prop, str):
                                    if hasattr(cv2, prop):
                                        self.cap.set(getattr(cv2, prop), value)
                                        logger.info(
                                            f"Set RTSP transport to TCP using {prop}"
                                        )
                                        break
                                else:
                                    self.cap.set(prop, value)
                                    logger.info(
                                        "Set RTSP transport to TCP using property ID"
                                    )
                                    break
                            except Exception:
                                continue

                    except Exception as ex:
                        logger.debug(f"Could not set RTSP transport: {ex}")

                    # Set preferred frame rate
                    try:
                        self.cap.set(cv2.CAP_PROP_FPS, 15)
                    except Exception as ex:
                        logger.debug(f"Could not set FPS: {ex}")

                # Verify the connection by attempting to read a frame
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    logger.info(f"Stream opened and verified for {self.detector_id}")
                    return True
                else:
                    logger.warning(
                        f"Stream opened but failed to read frame for {self.detector_id}"
                    )
                    self.cap.release()
                    self.cap = None
                    return False
            else:
                logger.error(f"Failed to open stream for {self.detector_id}")
                return False

        except Exception as ex:
            logger.error(f"Error opening stream: {ex}", exc_info=True)
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            return False

    def _stream_monitor_loop(self) -> None:
        """
        Stream monitor loop that keeps the connection alive but doesn't perform detection.
        Detection will only happen on poll requests.
        """
        logger.info(f"Starting stream monitor for {self.detector_id}")

        # How often to check the stream health
        check_interval = 5  # seconds

        # When to try reading a frame to verify stream health
        frame_check_interval = 30  # seconds
        last_frame_check = 0

        while not self.stop_event.is_set():
            try:
                current_time = time.time()

                # If we've been idle too long, shut down the detector
                if current_time - self.last_client_request > self.max_idle_time:
                    logger.info(
                        f"Detector {self.detector_id} has been idle for too long, shutting down"
                    )
                    self.stop_event.set()
                    break

                # Make sure we have a valid capture device
                if self.cap is None or not self.cap.isOpened():
                    # Try to reconnect
                    logger.warning(
                        f"Stream not available for {self.detector_id}, attempting to reconnect"
                    )
                    self.connection_status = "reconnecting"
                    self.stream_reconnect_attempts += 1

                    if (
                        self.stream_reconnect_attempts
                        > self.max_stream_reconnect_attempts
                    ):
                        logger.error(
                            f"Max reconnection attempts ({self.max_stream_reconnect_attempts}) reached for {self.detector_id}"
                        )
                        self.connection_status = "error"

                        # Reset counter but use exponential backoff for next attempt series
                        time.sleep(
                            min(
                                30
                                * (
                                    self.stream_reconnect_attempts
                                    // self.max_stream_reconnect_attempts
                                ),
                                300,
                            )
                        )
                        logger.info(
                            "Resetting reconnection attempts counter and trying again"
                        )
                        self.stream_reconnect_attempts = 0
                    else:
                        # Calculate progressive backoff delay
                        current_delay = self.stream_reconnect_delay * (
                            2 ** (self.stream_reconnect_attempts - 1)
                        )
                        current_delay = min(current_delay, 30)  # Cap at 30 seconds

                        logger.info(
                            f"Reconnection attempt {self.stream_reconnect_attempts}/{self.max_stream_reconnect_attempts} "
                            + f"with {current_delay:.1f}s delay"
                        )

                        # Close existing capture if any
                        if self.cap is not None:
                            try:
                                self.cap.release()
                                self.cap = None
                            except Exception:
                                pass

                        # Wait before attempting reconnection
                        time.sleep(current_delay)

                        # Try to reopen the stream
                        if self._open_stream():
                            logger.info(
                                f"Successfully reconnected to stream for {self.detector_id}"
                            )
                            self.connection_status = "connected"
                            self.stream_reconnect_attempts = 0
                        else:
                            logger.warning(
                                f"Reconnection attempt {self.stream_reconnect_attempts} failed"
                            )

                    continue

                # Periodically check if the stream is still healthy by reading a frame
                if current_time - last_frame_check > frame_check_interval:
                    # Try to read a frame just to verify connection is still working
                    ret, frame = self.cap.read()

                    if not ret or frame is None:
                        logger.warning(
                            f"Failed to read frame from stream during health check for {self.detector_id}"
                        )
                        self.connection_status = "error"
                        # Try to reconnect
                        if self._open_stream():
                            logger.info(
                                "Successfully reconnected to stream after health check failure"
                            )
                            self.connection_status = "connected"
                        else:
                            # Skip this iteration and try again later
                            time.sleep(check_interval)
                            continue
                    else:
                        # Stream is healthy
                        self.connection_status = "connected"
                        # Store the most recent frame for potential use in detection
                        self.last_frame = frame.copy()
                        logger.debug(
                            f"Stream health check passed for {self.detector_id}"
                        )

                    # Update the last health check time
                    last_frame_check = current_time

                # Sleep before next check
                time.sleep(check_interval)

            except cv2.error as cv_err:
                logger.error(f"OpenCV error in stream monitor: {cv_err}")
                self.error_counts["stream"] += 1
                # Try to reconnect to the stream
                self._open_stream()
                time.sleep(self.error_recovery_delay)

            except Exception as ex:
                logger.error(f"Error in stream monitor: {ex}", exc_info=True)
                self.error_counts["other"] += 1
                time.sleep(self.error_recovery_delay)

                # Check if we need to apply error recovery
                if self.consecutive_errors >= self.max_consecutive_errors:
                    logger.warning(
                        f"Too many consecutive errors ({self.consecutive_errors}), applying recovery measures"
                    )
                    # Try to reconnect to the stream
                    self._open_stream()
                    # Reset error counter
                    self.consecutive_errors = 0
                    # Sleep for a longer time
                    time.sleep(10)

        # Clean up when thread exits
        self._cleanup()

    def perform_detection(self) -> bool:
        """
        Perform detection on the current frame.
        This is called on each poll request rather than continuously.

        Returns:
            bool: True if detection was successful, False otherwise
        """
        if not self.is_running or self.cap is None or not self.cap.isOpened():
            logger.warning(
                f"Cannot perform detection: detector {self.detector_id} not ready"
            )
            return False

        try:
            # Read frame from the video stream
            ret, frame = self.cap.read()

            if not ret or frame is None:
                logger.warning(
                    f"Failed to read frame for detection from {self.detector_id}"
                )

                # Try to reconnect
                if not self._open_stream():
                    return False

                # Try one more time to read a frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.error(
                        f"Failed to read frame after reconnection for {self.detector_id}"
                    )
                    return False

            # Store the frame
            self.last_frame = frame.copy()

            # Update connection status
            self.connection_status = "connected"

            # Increment frame counter
            self.frame_count += 1

            # Get frame dimensions for logging
            original_height, original_width = frame.shape[:2]

            # YOLO expects dimensions in (width, height) format
            # The imgsz parameter controls the internal resize that YOLO does

            # YOLO models have a stride requirement - dimensions should be multiples of stride
            # Default stride is 32 for most YOLO models
            STRIDE = 32

            # Adjust width and height to be multiples of STRIDE
            adjusted_width = (self.input_width // STRIDE) * STRIDE
            adjusted_height = (self.input_height // STRIDE) * STRIDE

            # Ensure we have at least one stride worth of pixels
            adjusted_width = max(adjusted_width, STRIDE)
            adjusted_height = max(adjusted_height, STRIDE)

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

            start_time = time.time()
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                imgsz=(
                    adjusted_width,
                    adjusted_height,
                ),  # Tell YOLO what size to use (stride-adjusted)
            )
            inference_time = (time.time() - start_time) * 1000  # in milliseconds

            # Process results
            people_detected = False
            pets_detected = False
            people_count = 0
            pet_count = 0

            # Store detected objects for detailed reporting
            detected_objects = {}

            if len(results) > 0:
                # Extract detections - YOLO v8 format
                result = results[0]  # First batch result

                # Get boxes and class information
                boxes = result.boxes

                # Create an annotated version of the frame for visualization
                annotated_frame = frame.copy()

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
                        if class_name not in detected_objects:
                            detected_objects[class_name] = 0
                        detected_objects[class_name] += 1

                        if cls_id == 0:  # Person
                            people_detected = True
                            people_count += 1
                        elif cls_id in [15, 16, 17]:  # Bird, cat, dog
                            pets_detected = True
                            pet_count += 1

                        # Draw bounding box on the annotated frame
                        # Get box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        # Convert to original frame coordinates if needed
                        orig_height, orig_width = annotated_frame.shape[:2]
                        x1 = int(x1 * orig_width / self.adjusted_dimensions[0])
                        y1 = int(y1 * orig_height / self.adjusted_dimensions[1])
                        x2 = int(x2 * orig_width / self.adjusted_dimensions[0])
                        y2 = int(y2 * orig_height / self.adjusted_dimensions[1])

                        # Ensure coordinates are within frame bounds
                        x1 = max(0, min(orig_width - 1, x1))
                        y1 = max(0, min(orig_height - 1, y1))
                        x2 = max(0, min(orig_width - 1, x2))
                        y2 = max(0, min(orig_height - 1, y2))

                        # Draw rectangle
                        color = (
                            (0, 255, 0) if cls_id == 0 else (0, 165, 255)
                        )  # Green for people, orange for pets
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                        # Prepare label with class name and confidence
                        label = f"{class_name} {confidence:.2f}"
                        text_size, _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )

                        # Draw label background
                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1 - text_size[1] - 5),
                            (x1 + text_size[0], y1),
                            color,
                            -1,
                        )

                        # Draw label text
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            2,
                        )

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

            # Add timestamp and extra info to the annotated frame
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                annotated_frame,
                f"Time: {current_time}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Add detection stats
            info_text = f"People: {people_count}, Pets: {pet_count}"
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Add inference time
            cv2.putText(
                annotated_frame,
                f"Inference: {inference_time:.1f}ms",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Store the annotated frame for visualization endpoint
            self.last_annotated_frame = annotated_frame
            self.last_annotated_time = time.time()

            # Create detailed detection report
            detection_report = ", ".join(
                [f"{count} {obj}" for obj, count in detected_objects.items()]
            )
            if not detection_report:
                detection_report = "nothing detected"

            # Log detection with detailed information
            logger.info(
                f"Detection for {self.detector_id}: people={people_detected}({people_count}), "
                f"pets={pets_detected}({pet_count}), "
                f"objects: {detection_report}, "
                f"time: {inference_time:.1f}ms, "
                f"original size: {original_width}x{original_height}, "
                f"requested size: {self.input_width}x{self.input_height}, "
                f"adjusted size: {adjusted_width}x{adjusted_height}"
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

        # Release video capture
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        # Set status flags
        self.is_running = False
        self.connection_status = "disconnected"

    def shutdown(self) -> None:
        """Shut down the detector."""
        logger.info(f"Shutting down detector {self.detector_id}")

        # Signal thread to stop
        self.stop_event.set()

        # Wait for thread to exit
        if self.detection_thread and self.detection_thread.is_alive():
            logger.info(f"Waiting for detection thread to exit for {self.detector_id}")
            self.detection_thread.join(timeout=5)

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

    def _send_auth_required(self) -> None:
        """Send authentication required response."""
        self.send_response(401)
        self.send_header(
            "WWW-Authenticate", 'Basic realm="YOLO Presence Detection Server"'
        )
        self.send_header("Content-type", "text/html")
        self.end_headers()

        # Create a simple HTML page
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Authentication Required</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 50px; text-align: center; }
                h1 { color: #d32f2f; }
                p { font-size: 18px; color: #333; }
            </style>
        </head>
        <body>
            <h1>Authentication Required</h1>
            <p>Please login with valid credentials to access this page.</p>
        </body>
        </html>
        """

        self.wfile.write(html.encode())

    def _send_mjpeg_frame(self, frame, boundary="--boundarydonotcross"):
        """Send a single frame as part of an MJPEG stream."""
        try:
            # Make sure we have a proper boundary format
            if not boundary.startswith("--"):
                boundary = f"--{boundary}"

            # Verify frame is valid
            if frame is None or frame.size == 0:
                logger.warning("Attempted to send invalid frame in MJPEG stream")
                # Create a simple error frame
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    "No valid frame",
                    (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            # Encode the frame as JPEG
            # Try with higher quality first
            try:
                _, jpeg_data = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
            except Exception as e:
                logger.warning(
                    f"Error encoding JPEG at high quality: {e}, trying lower quality"
                )
                _, jpeg_data = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50]
                )

            jpeg_bytes = jpeg_data.tobytes()

            # Write the MJPEG part header with proper boundary format
            # Note: If this is the first part, we've already written the initial boundary
            # For subsequent parts, we need the boundary with the leading CRLF
            self.wfile.write(f"{boundary}\r\n".encode())
            self.wfile.write(b"Content-Type: image/jpeg\r\n")
            self.wfile.write(f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode())

            # Write the JPEG data
            self.wfile.write(jpeg_bytes)
            self.wfile.flush()

            return True
        except (ConnectionResetError, BrokenPipeError) as e:
            logger.warning(f"Client disconnected during MJPEG streaming: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending MJPEG frame: {e}")
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
            # Check authentication for protected paths
            if not self._check_auth():
                self._send_auth_required()
                return

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
                        # Create a simple HTML page that lists all detectors with links to their view pages
                        html = (
                            """
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>YOLO Presence Detection Server</title>
                            <style>
                                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f0f0f0; }
                                h1 { color: #333; }
                                .container { max-width: 1200px; margin: 0 auto; }
                                .detector-list { list-style: none; padding: 0; }
                                .detector-card { 
                                    background-color: #fff; 
                                    border-radius: 5px; 
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                    margin-bottom: 15px; 
                                    padding: 15px; 
                                    display: flex;
                                    justify-content: space-between;
                                    align-items: center;
                                }
                                .detector-info { flex-grow: 1; }
                                .detector-name { font-weight: bold; font-size: 18px; margin-bottom: 5px; }
                                .detector-id { color: #666; font-size: 14px; margin-bottom: 5px; }
                                .detector-model { color: #444; margin-bottom: 5px; }
                                .detector-resolution { color: #444; margin-bottom: 5px; }
                                .detector-stats { color: #444; }
                                .status { 
                                    display: inline-block;
                                    padding: 5px 10px; 
                                    border-radius: 3px; 
                                    font-weight: bold; 
                                    margin-bottom: 10px;
                                }
                                .connected { background-color: #d4edda; color: #155724; }
                                .disconnected { background-color: #f8d7da; color: #721c24; }
                                .reconnecting { background-color: #fff3cd; color: #856404; }
                                .error { background-color: #f8d7da; color: #721c24; }
                                .detector-actions { 
                                    display: flex;
                                    gap: 10px;
                                }
                                .view-button { 
                                    display: inline-block;
                                    padding: 8px 15px; 
                                    background-color: #007bff; 
                                    color: white; 
                                    text-decoration: none; 
                                    border-radius: 3px; 
                                    font-weight: bold;
                                }
                                .view-button:hover { background-color: #0069d9; }
                                .no-detectors {
                                    background-color: #fff;
                                    padding: 20px;
                                    border-radius: 5px;
                                    text-align: center;
                                    color: #666;
                                }
                                .server-info {
                                    background-color: #fff;
                                    padding: 15px;
                                    border-radius: 5px;
                                    margin-bottom: 20px;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                }
                                .server-info h2 {
                                    margin-top: 0;
                                    color: #333;
                                }
                                .refresh {
                                    padding: 10px 15px;
                                    background-color: #6c757d;
                                    color: white;
                                    border: none;
                                    border-radius: 3px;
                                    cursor: pointer;
                                    font-size: 16px;
                                    margin: 10px 0;
                                }
                                .refresh:hover {
                                    background-color: #5a6268;
                                }
                            </style>
                            <meta http-equiv="refresh" content="30">
                        </head>
                        <body>
                            <div class="container">
                                <h1>YOLO Presence Detection Server</h1>
                                
                                <div class="server-info">
                                    <h2>Server Status</h2>
                                    <p>Active detectors: """
                            + str(len(detectors))
                            + """</p>
                                    <p>Server time: """
                            + time.strftime("%Y-%m-%d %H:%M:%S")
                            + """</p>
                                    <button class="refresh" onclick="window.location.reload()">Refresh Now</button>
                                    <p><small>Page auto-refreshes every 30 seconds</small></p>
                                </div>
                                
                                <h2>Detectors</h2>
                        """
                        )

                        if not detectors:
                            html += """
                                <div class="no-detectors">
                                    <p>No active detectors found.</p>
                                    <p>When a detector connects, it will appear here.</p>
                                </div>
                            """
                        else:
                            html += '<ul class="detector-list">'

                            for detector_id, detector in detectors.items():
                                state = detector.get_state()

                                # Format detection information
                                detection_info = ""
                                if (
                                    hasattr(detector, "detected_objects")
                                    and detector.detected_objects
                                ):
                                    items = [
                                        f"{count} {obj}"
                                        for obj, count in detector.detected_objects.items()
                                    ]
                                    if items:
                                        detection_info = f"Detected: {', '.join(items)}"

                                html += f"""
                                    <li class="detector-card">
                                        <div class="detector-info">
                                            <div class="detector-name">{detector.name}</div>
                                            <div class="detector-id">ID: {detector_id}</div>
                                            <div class="status {detector.connection_status}">{detector.connection_status.upper()}</div>
                                            <div class="detector-model">Model: {detector.model_name}</div>
                                            <div class="detector-resolution">Resolution: {detector.input_width}x{detector.input_height}</div>
                                            <div class="detector-stats">
                                                People: {detector.people_count}, 
                                                Pets: {detector.pet_count}
                                            </div>
                                            <div class="detector-stats">
                                                {detection_info}
                                            </div>
                                        </div>
                                        <div class="detector-actions">
                                            <a class="view-button" href="/view?detector_id={detector_id}">View Stream</a>
                                        </div>
                                    </li>
                                """

                            html += "</ul>"

                        html += """
                            </div>
                        </body>
                        </html>
                        """

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
            elif path == "/stream" and detector_id:
                # Stream the latest annotated frame as MJPEG
                try:
                    with detector_lock:
                        if detector_id not in detectors:
                            self._send_error_response(
                                f"Detector {detector_id} not found", 404
                            )
                            return

                        detector = detectors[detector_id]

                        # Check if we have an annotated frame
                        if detector.last_annotated_frame is None:
                            self._send_error_response(
                                "No annotated frame available yet", 404
                            )
                            return

                        # SAFETY BYPASS: Instead of streaming directly in the main server thread,
                        # Use a static image with a JavaScript-based stream approach
                        logger.info(
                            f"Using static image approach for detector {detector_id}"
                        )

                        # Set headers for HTML page
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html")
                        self.send_header(
                            "Content-Type",
                            f"multipart/x-mixed-replace; boundary={boundary}",
                        )
                        # Set caching headers to prevent any caching
                        self.send_header(
                            "Cache-Control", "no-cache, no-store, must-revalidate"
                        )
                        self.send_header("Pragma", "no-cache")
                        self.send_header("Expires", "0")
                        # Add CORS headers to allow embedding in other pages
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.send_header("Connection", "close")
                        self.end_headers()

                        # Send the initial boundary to properly start the multipart message
                        self.wfile.write(f"--{boundary}\r\n".encode())

                        # Create a placeholder frame with a message if no annotated frame is available
                        def create_placeholder_frame():
                            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(
                                placeholder,
                                "Waiting for detection...",
                                (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2,
                            )
                            return placeholder

                        # Track if a detector exists outside the loop to reduce lock contention
                        detector_exists = True
                        max_failures = 3  # Maximum consecutive failures before we exit
                        failure_count = 0

                        # Stream frames indefinitely
                        while detector_exists and failure_count < max_failures:
                            try:
                                # Use a very short-lived lock to just get a copy of the frame
                                frame_to_send = None
                                detector_name = "Unknown"

                                # Critical section - only hold the lock for the minimum time needed to copy the frame
                                with detector_lock:
                                    if detector_id not in detectors:
                                        logger.info(
                                            f"Detector {detector_id} no longer exists, ending stream"
                                        )
                                        detector_exists = False
                                        break

                                    detector = detectors[detector_id]
                                    detector_name = detector.name

                                    # Get the latest frame or none if not available
                                    if (
                                        detector.last_annotated_frame is not None
                                        and detector.last_annotated_time > 0
                                    ):
                                        frame_to_send = (
                                            detector.last_annotated_frame.copy()
                                        )

                                # We now release the lock and process the frame outside the lock

                                # If we didn't get a frame, create a placeholder
                                if frame_to_send is None:
                                    frame_to_send = create_placeholder_frame()
                                    # Add detector name to placeholder
                                    cv2.putText(
                                        frame_to_send,
                                        f"Detector: {detector_name}",
                                        (50, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (255, 255, 255),
                                        2,
                                    )
                                else:
                                    # Add "live" indicator (all processing done outside the lock)
                                    current_time = time.strftime("%H:%M:%S")

                                    # Calculate safe position for text
                                    h, w = frame_to_send.shape[:2]
                                    text_size = cv2.getTextSize(
                                        f"Live: {current_time}",
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        2,
                                    )[0]

                                    # Place text in top-right, but ensure it's within frame bounds
                                    text_x = max(
                                        10, min(w - text_size[0] - 10, w - 180)
                                    )
                                    text_y = 25

                                    cv2.putText(
                                        frame_to_send,
                                        f"Live: {current_time}",
                                        (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 255, 255),
                                        2,
                                    )

                                    # Add the detector name as well for context
                                    cv2.putText(
                                        frame_to_send,
                                        f"Detector: {detector_name}",
                                        (text_x, text_y + 25),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 255, 255),
                                        2,
                                    )

                                # Send the frame (outside the critical section)
                                if not self._send_mjpeg_frame(frame_to_send, boundary):
                                    # Client disconnected or error occurred
                                    logger.info(
                                        f"Client disconnected from stream for detector {detector_id}"
                                    )
                                    break

                                # Reset failure count on success
                                failure_count = 0

                                # Sleep for a bit to control frame rate
                                time.sleep(0.2)  # 5 FPS is fine for this

                            except Exception as stream_err:
                                # Log error but try to continue
                                logger.error(
                                    f"Error in stream loop for {detector_id}: {stream_err}"
                                )
                                failure_count += 1

                                # Create an error frame to send to the client
                                error_frame = np.zeros((240, 320, 3), dtype=np.uint8)
                                cv2.putText(
                                    error_frame,
                                    f"Stream error: {type(stream_err).__name__}",
                                    (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1,
                                )

                                # Try to send the error frame
                                try:
                                    self._send_mjpeg_frame(error_frame, boundary)
                                except Exception as e:
                                    logger.error(f"Failed to send error frame: {e}")

                                # Add a small delay before retrying
                                time.sleep(1)

                        logger.info(f"MJPEG stream for detector {detector_id} ended")

                        # Final cleanup - make sure connection is cleaned up
                        try:
                            # Add a final boundary to properly end the multipart message
                            self.wfile.write(f"\r\n--{boundary}--\r\n".encode())
                            self.wfile.flush()
                        except Exception as cleanup_err:
                            logger.debug(f"Error sending final boundary: {cleanup_err}")

                        return
                except Exception as ex:
                    logger.error(f"Error streaming frames: {ex}", exc_info=True)
                    self._send_error_response(f"Error streaming frames: {str(ex)}", 500)
            elif path == "/jpeg" and detector_id:
                # Return a single JPEG frame - much safer than continuous MJPEG streaming
                try:
                    with detector_lock:
                        if detector_id not in detectors:
                            self._send_error_response(
                                f"Detector {detector_id} not found", 404
                            )
                            return

                        detector = detectors[detector_id]

                        # Get the frame
                        if detector.last_annotated_frame is not None:
                            frame = detector.last_annotated_frame.copy()
                        else:
                            # Create placeholder
                            frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(
                                frame,
                                "Waiting for detection...",
                                (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2,
                            )

                    # Function to draw text with a white outline for visibility
                    def draw_outlined_text(
                        img, text, position, font_scale=1.0, line_thickness=3, margin=30
                    ):
                        # Coordinates for text
                        x, y = position

                        # Draw black text (main text)
                        cv2.putText(
                            img,
                            text,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (0, 0, 0),  # Black
                            line_thickness,
                        )

                        # Draw white outline
                        cv2.putText(
                            img,
                            text,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (255, 255, 255),  # White
                            int(line_thickness / 3),
                        )

                        return y + margin  # Return next y position

                    # Add rich information to the frame
                    current_time = time.strftime("%H:%M:%S")
                    y_pos = 40

                    # Time
                    y_pos = draw_outlined_text(
                        frame, f"Time: {current_time}", (10, y_pos), 1.2, 3
                    )

                    # Get detector info from the query string
                    try:
                        query = parse_qs(urlparse(self.path).query)
                        detector_id = query.get("detector_id", ["Unknown"])[0]

                        with detector_lock:
                            if detector_id in detectors:
                                detector = detectors[detector_id]

                                # Detector name
                                y_pos = draw_outlined_text(
                                    frame,
                                    f"Detector: {detector.name}",
                                    (10, y_pos),
                                    1.0,
                                    3,
                                )

                                # Detection counts
                                if (
                                    hasattr(detector, "people_count")
                                    and detector.people_count > 0
                                ):
                                    y_pos = draw_outlined_text(
                                        frame,
                                        f"People: {detector.people_count}",
                                        (10, y_pos),
                                        1.0,
                                        3,
                                    )

                                if (
                                    hasattr(detector, "pet_count")
                                    and detector.pet_count > 0
                                ):
                                    y_pos = draw_outlined_text(
                                        frame,
                                        f"Pets: {detector.pet_count}",
                                        (10, y_pos),
                                        1.0,
                                        3,
                                    )
                    except Exception as ex:
                        logger.error(f"Error adding text to frame: {ex}")

                    # Encode as JPEG
                    _, jpg_data = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
                    )
                    jpg_bytes = jpg_data.tobytes()

                    # Send the JPEG
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

                        # Create a simple HTML page with the embedded MJPEG stream
                        html = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>YOLO Detector: {detector.name}</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; background-color: #f0f0f0; }}
                                h1 {{ color: #333; }}
                                .container {{ max-width: 1200px; margin: 0 auto; }}
                                .stream {{ 
                                    width: 100%; 
                                    max-width: 1024px; 
                                    margin: 20px auto; 
                                    border: 2px solid #333; 
                                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                                    height: auto !important; /* Important - maintain aspect ratio */
                                    min-height: 240px; /* Ensure image is not too small */
                                    object-fit: contain; /* Maintain aspect ratio */
                                }}
                                .info {{ 
                                    background-color: #fff; 
                                    padding: 15px; 
                                    border-radius: 5px; 
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                    margin: 20px auto;
                                    max-width: 1024px;
                                    text-align: left;
                                }}
                                .detector-name {{ font-weight: bold; font-size: 18px; }}
                                .status {{ padding: 5px 10px; border-radius: 3px; font-weight: bold; }}
                                .connected {{ background-color: #d4edda; color: #155724; }}
                                .disconnected {{ background-color: #f8d7da; color: #721c24; }}
                                .reconnecting {{ background-color: #fff3cd; color: #856404; }}
                                .error {{ background-color: #f8d7da; color: #721c24; }}
                                .refresh {{ 
                                    padding: 10px 15px; 
                                    background-color: #007bff; 
                                    color: white; 
                                    border: none; 
                                    border-radius: 3px; 
                                    cursor: pointer; 
                                    font-size: 16px;
                                }}
                                .refresh:hover {{ background-color: #0069d9; }}
                            </style>
                        </head>
                        <body>
                            <div class="container">
                                <h1>YOLO Detector Live Stream</h1>
                                
                                <div class="info">
                                    <p class="detector-name">Name: {detector.name}</p>
                                    <p>Detector ID: {detector.detector_id}</p>
                                    <p>Model: {detector.model_name}</p>
                                    <p>Status: <span class="status {detector.connection_status}">{detector.connection_status}</span></p>
                                    <p>Resolution: {detector.input_width}x{detector.input_height}</p>
                                    <p>Confidence threshold: {detector.confidence_threshold}</p>
                                    <p>Auto-optimization: {"Enabled" if detector.auto_optimization else "Disabled"}</p>
                                </div>
                                
                                <div style="width:100%; max-width:1024px; margin:20px auto; border:2px solid #333; box-shadow:0 4px 8px rgba(0,0,0,0.1); background-color:#000; position:relative; min-height:320px;">
                                    <img id="stream-img" src="/jpeg?detector_id={detector_id}" alt="Live detection stream" style="width:100%; height:auto; min-height:320px; display:block;" />
                                    <!-- Status elements are hidden but still used by JavaScript -->
                                    <div id="connection-status" style="display:none;"></div>
                                    <div id="fps-counter" style="display:none;"></div>
                                    <div id="status-text" style="display:none;"></div>
                                </div>
                                <script>
                                    // Simple auto-refreshing image
                                    document.addEventListener('DOMContentLoaded', function() {{
                                        const streamImg = document.getElementById('stream-img');
                                        const refreshRate = 200; // milliseconds (5 fps)
                                        let frameCount = 0;
                                        
                                        // Function to refresh the image
                                        function refreshImage() {{
                                            // Add timestamp to prevent caching
                                            const url = `/jpeg?detector_id={detector_id}&t=${{Date.now()}}`;
                                            
                                            // Set the new source
                                            streamImg.src = url;
                                            frameCount++;
                                            
                                            // Schedule next refresh
                                            setTimeout(refreshImage, refreshRate);
                                        }}
                                        
                                        // Load first image immediately
                                        refreshImage();
                                        
                                        // Auto-refresh the whole page every 5 minutes to prevent memory issues
                                        setTimeout(() => {{
                                            window.location.reload();
                                        }}, 5 * 60 * 1000);
                                    }});
                                
                                <p>
                                    <button class="refresh" onclick="window.location.reload()">Refresh Page</button>
                                </p>
                            </div>
                        </body>
                        </html>
                        """

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
