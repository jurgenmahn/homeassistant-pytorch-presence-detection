"""YOLO Presence Detection Server

This standalone server handles video processing and object detection
using YOLO models, designed to work with Home Assistant's yolo_presence
integration through TCP socket connections.
"""
import asyncio
import json
import logging
import os
import signal
import socket
import selectors
import sys
import threading
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Set

import cv2
import numpy as np
import torch
import queue
from ultralytics import YOLO

# Import custom logging configuration
from logging_config import configure_logging, get_logger, LogExceptionContext

# Import watchdog and optimizer modules
from watchdog import system_watchdog, start_watchdog
from optimizer import get_resource_optimizer, get_graceful_degradation

# Define constants for detector persistence
DETECTORS_CONFIG_DIR = os.path.join(os.getcwd(), "config")
DETECTORS_CONFIG_FILE = os.path.join(DETECTORS_CONFIG_DIR, "detectors.json")

# Configure logging with rotation and structured format
logger = configure_logging(
    logger_name="yolo_presence_server",
    log_level=logging.DEBUG,  # Change to DEBUG level for maximum verbosity
    max_file_size=10 * 1024 * 1024,  # 10 MB
    backup_count=5,
    detailed_format=True
)

# Create module-specific loggers
detector_logger = get_logger("detector")
socket_logger = get_logger("socket")

# Socket server
TCP_PORT = int(os.environ.get("TCP_PORT", 5505))
socket_server = None
socket_clients = {}  # Maps detector_id to set of socket connections

# Global settings
SUPPORTED_CLASSES = [0, 15, 16, 17]  # person, bird, cat, dog
CLASS_MAP = {
    0: "person",
    15: "bird",
    16: "cat", 
    17: "dog"
}

# Store for detector instances (key: detector_id)
detectors = {}
detector_lock = threading.Lock()

# Store SSE client queues for each detector (key: detector_id)
sse_clients = {}
sse_clients_lock = threading.Lock()

# Get optimizer instances
resource_optimizer = get_resource_optimizer()
graceful_degradation = get_graceful_degradation()

# Periodic configuration save
CONFIG_SAVE_INTERVAL = 300  # 5 minutes
last_config_save_time = 0

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
        self.error_counts = {
            "memory": 0,
            "cuda": 0,
            "stream": 0,
            "other": 0
        }
        
        # Optimization state
        self.optimization_level = 0
        self.degradation_level = 0
        self.last_optimization_check = 0
        self.optimization_check_interval = 60  # seconds

    def initialize(self) -> bool:
        """Initialize the detector and start the detection process."""
        try:
            detector_logger.info(f"Initializing YOLO Detector for {self.name}")
            detector_logger.info(f"Stream URL: {self.stream_url}")
            detector_logger.info(f"Model: {self.model_name}")
            detector_logger.info(f"Input size: {self.input_width}x{self.input_height}")
            detector_logger.info(f"Detection interval: {self.detection_interval}s")
            detector_logger.info(f"Confidence threshold: {self.confidence_threshold}")
            
            # Load the YOLO model
            self._load_model()
            
            # Create output directory for debug frames
            self.debug_frames_dir = os.path.join(os.getcwd(), "debug_frames", self.detector_id)
            os.makedirs(self.debug_frames_dir, exist_ok=True)
            detector_logger.info(f"Debug frames will be saved to {self.debug_frames_dir}")
            
            # Start the detection process
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            self.is_running = True
            detector_logger.info(f"YOLO Detector initialized for {self.name}")
            return True
            
        except Exception as ex:
            detector_logger.error(f"Failed to initialize YOLO detector: {str(ex)}", exc_info=True)
            return False

    def _load_model(self) -> None:
        """Load the YOLO model."""
        # Determine model path - look in various possible locations
        model_locations = [
            self.model_path,                  # Current directory
            f"models/{self.model_path}",      # models subdirectory
            f"/app/models/{self.model_path}", # Docker volume mounted models
        ]
        
        # Find the first existing model file
        model_path = None
        for loc in model_locations:
            if os.path.exists(loc):
                model_path = loc
                detector_logger.info(f"Found YOLO model at {model_path}")
                break
        
        # If model not found locally, use pre-trained from ultralytics
        if not model_path:
            detector_logger.warning("Model not found locally, using pre-trained model")
            # Strip the .pt extension for pretrained models
            base_name = self.model_name
            if base_name.endswith(".pt"):
                base_name = base_name[:-3]
            model_path = base_name
        
        # Check if CUDA is available
        try:
            has_cuda = torch.cuda.is_available()
            torch_version = tuple(map(int, torch.__version__.split(".")[:2]))
            
            # ROCm support check
            has_rocm = False
            if hasattr(torch.backends, "hip") and hasattr(torch.backends.hip, "is_built"):
                has_rocm = torch.backends.hip.is_built()
        except Exception as ex:
            detector_logger.warning(f"Error checking CUDA availability: {str(ex)}", exc_info=True)
            has_cuda = False
            has_rocm = False
        
        # Select device
        device = "cpu"
        if has_cuda:
            device = "cuda"
            detector_logger.info("Using CUDA for YOLO inference")
        elif has_rocm:
            device = "cuda"  # ROCm uses CUDA API
            detector_logger.info("Using ROCm for YOLO inference") 
        else:
            detector_logger.info("Using CPU for YOLO inference")
        
        # Load the model with error handling
        with LogExceptionContext(detector_logger, f"Loading model {model_path}", {"device": device}):
            try:
                # First try with specified device
                model = YOLO(model_path)
                model.to(device)
            except Exception as ex:
                detector_logger.warning(f"Error loading model on {device}: {str(ex)}, falling back to CPU", exc_info=True)
                # Fall back to CPU if device-specific loading fails
                device = "cpu"
                model = YOLO(model_path)
                model.to(device)
            
            # Try to configure model parameters safely
            try:
                model.overrides['imgsz'] = max(self.input_width, self.input_height)
            except Exception as ex:
                detector_logger.warning(f"Could not set model image size: {str(ex)}", exc_info=True)
            
            self.model = model
            self.device = device
            detector_logger.info(
                f"YOLO model {self.model_name} loaded on {self.device} with input size {self.input_width}x{self.input_height}"
            )

    def _open_stream(self) -> bool:
        """Open the video stream."""
        try:
            # Close any existing stream
            if self.cap is not None:
                self.cap.release()
            
            detector_logger.info(f"Opening stream: {self.stream_url}")
            
            # Open the stream
            cap = cv2.VideoCapture(self.stream_url)
            
            # Configure stream parameters
            if cap.isOpened():
                # Try to set hardware acceleration
                try:
                    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                except:
                    detector_logger.debug("Hardware acceleration not supported")
                    pass
                
                # Set H.264 codec for better performance
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                except:
                    detector_logger.debug("H264 codec not supported")
                    pass
                
                # Get stream information
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                detector_logger.info(f"Stream opened successfully for {self.name}")
                detector_logger.info(f"Stream resolution: {width}x{height}, FPS: {fps}")
                
                self.cap = cap
                # Reset reconnect attempts on successful connection
                self.stream_reconnect_attempts = 0
                # Reset stream error count on successful connection
                self.error_counts["stream"] = 0
                
                # Read a test frame to verify stream is working
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    detector_logger.info(f"Successfully read initial frame from stream")
                    # Save test frame to debug directory
                    frame_path = os.path.join(self.debug_frames_dir, f"test_frame.jpg")
                    cv2.imwrite(frame_path, test_frame)
                    detector_logger.info(f"Saved test frame to {frame_path}")
                else:
                    detector_logger.warning(f"Could read initial frame but frame is empty or invalid")
                
                return True
            else:
                detector_logger.warning(f"Failed to open stream for {self.name}")
                self.stream_reconnect_attempts += 1
                self.error_counts["stream"] += 1
                return False
        except Exception as ex:
            detector_logger.error(f"Error opening stream: {str(ex)}", exc_info=True)
            self.stream_reconnect_attempts += 1
            self.error_counts["stream"] += 1
            return False

    def _close_stream(self) -> None:
        """Close the video stream."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as ex:
                detector_logger.debug(f"Error closing stream: {str(ex)}")
            self.cap = None

    def _check_and_apply_optimization(self) -> None:
        """Check if optimization is needed and apply it."""
        current_time = time.time()
        
        # Only check periodically
        if current_time - self.last_optimization_check < self.optimization_check_interval:
            return
        
        self.last_optimization_check = current_time
        
        try:
            # Get current resource usage
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Get GPU usage if available
            gpu_percent = None
            if self.device == "cuda":
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Get current and max memory allocated for the first GPU
                        current = torch.cuda.memory_allocated(0)
                        max_mem = torch.cuda.get_device_properties(0).total_memory
                        if max_mem > 0:
                            gpu_percent = (current / max_mem) * 100
                except Exception:
                    pass
            
            # Apply optimization based on resource usage
            settings = resource_optimizer.optimize_for_resources(cpu_percent, memory_percent, gpu_percent)
            
            # Update detector settings if optimization level changed
            new_level = resource_optimizer.get_optimization_level()
            if new_level != self.optimization_level:
                self.optimization_level = new_level
                self._apply_optimization_settings(settings)
                detector_logger.info(f"Applied optimization level {new_level} to detector {self.name}")
        
        except Exception as ex:
            detector_logger.error(f"Error in optimization check: {str(ex)}")

    def _apply_optimization_settings(self, settings: Dict[str, Any]) -> None:
        """Apply optimization settings to the detector."""
        # Update detection interval
        if "detection_interval" in settings:
            self.detection_interval = settings["detection_interval"]
        
        # Update frame skip rate
        if "frame_skip_rate" in settings:
            self.frame_skip_rate = settings["frame_skip_rate"]
        
        # Update confidence threshold
        if "confidence_threshold" in settings:
            self.confidence_threshold = settings["confidence_threshold"]
        
        # Update input size if needed
        if "input_size" in settings:
            try:
                input_size_str = settings["input_size"]
                width, height = map(int, input_size_str.split("x"))
                self.input_width = width
                self.input_height = height
            except Exception:
                pass
        
        detector_logger.debug(f"Applied optimization settings: {settings}")

    def _handle_error(self, error_type: str, ex: Exception) -> None:
        """Handle errors with appropriate recovery strategies."""
        # Increment error count for this type
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log the error
        if self.error_counts[error_type] <= 1 or time.time() - self.last_error_time > 60:
            detector_logger.error(f"{error_type.capitalize()} error in detector {self.name}: {str(ex)}", exc_info=True)
        else:
            detector_logger.debug(f"Repeated {error_type} error in detector {self.name}: {str(ex)}")
        
        self.last_error_time = time.time()
        
        # Apply graceful degradation if needed
        if sum(self.error_counts.values()) > 5:
            level, settings = graceful_degradation.degrade_gracefully(error_type, self.error_counts[error_type])
            
            if level != self.degradation_level:
                self.degradation_level = level
                self._apply_optimization_settings(settings)
                detector_logger.warning(f"Applied degradation level {level} to detector {self.name} due to {error_type} errors")
                
                # Reset error counts after applying degradation
                if level >= 2:  # Medium or severe degradation
                    for k in self.error_counts:
                        self.error_counts[k] = 0

    def _detection_loop(self) -> None:
        """Main detection loop."""
        last_detection_time = 0
        
        detector_logger.debug(f"Starting detection loop for {self.name}")
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Check if optimization is needed
                    self._check_and_apply_optimization()
                    
                    # Open the stream if not already open
                    if self.cap is None or not self.cap.isOpened():
                        if self.stream_reconnect_attempts >= self.max_stream_reconnect_attempts:
                            detector_logger.error(f"Failed to reconnect to stream after {self.stream_reconnect_attempts} attempts for {self.name}")
                            # Increase delay between reconnect attempts
                            reconnect_delay = min(60, self.stream_reconnect_delay * (1 + self.stream_reconnect_attempts // 5))
                            time.sleep(reconnect_delay)
                            # Reset counter periodically to allow future reconnection attempts
                            if self.stream_reconnect_attempts > self.max_stream_reconnect_attempts * 2:
                                self.stream_reconnect_attempts = 0
                        
                        # Try to open the stream
                        if not self._open_stream():
                            # If failed, wait before retrying
                            time.sleep(self.stream_reconnect_delay)
                            continue
                    
                    # Check if it's time to perform detection
                    current_time = time.time()
                    if current_time - last_detection_time < self.detection_interval:
                        # Sleep to avoid busy waiting
                        sleep_time = 0.1
                        time.sleep(sleep_time)
                        continue
                    
                    # Read a frame from the stream
                    detector_logger.debug(f"Reading frame from stream")
                    ret, frame = self.cap.read()
                    
                    if not ret or frame is None:
                        detector_logger.warning(f"Failed to read frame from stream for {self.name}")
                        self._close_stream()
                        continue
                    
                    # Process the frame
                    self.frame_count += 1
                    detector_logger.debug(f"Frame {self.frame_count} read successfully")
                    
                    # Skip frames to reduce processing load
                    if self.frame_count % self.frame_skip_rate != 0:
                        detector_logger.debug(f"Skipping frame {self.frame_count} (processing every {self.frame_skip_rate} frames)")
                        continue
                    
                    # Resize frame for processing
                    original_size = f"{frame.shape[1]}x{frame.shape[0]}"
                    frame_resized = cv2.resize(frame, (self.input_width, self.input_height))
                    detector_logger.debug(f"Resized frame from {original_size} to {self.input_width}x{self.input_height}")
                    
                    # Disabled debug frame saving to improve performance
                    # Uncomment the following code to re-enable frame saving for debugging
                    # if self.frame_count % (self.frame_skip_rate * 10) == 0:
                    #     frame_path = os.path.join(self.debug_frames_dir, f"frame_{self.frame_count}.jpg")
                    #     cv2.imwrite(frame_path, frame)
                    #     detector_logger.info(f"Saved frame {self.frame_count} to {frame_path}")
                    
                    # Run YOLO detection
                    detector_logger.debug(f"Running YOLO detection on frame {self.frame_count}")
                    results = self.model(frame_resized, conf=self.confidence_threshold, classes=SUPPORTED_CLASSES)
                    detector_logger.debug(f"YOLO detection completed on frame {self.frame_count}")
                    
                    # Process results
                    detections = []
                    detector_logger.info(f"Processing YOLO results for frame {self.frame_count}")
                    
                    # Log the raw results to better understand what's happening
                    try:
                        boxes_sum = sum(len(r.boxes) if hasattr(r, 'boxes') else 0 for r in results)
                        detector_logger.info(f"Total boxes detected: {boxes_sum} in frame {self.frame_count}")
                    except Exception as ex:
                        detector_logger.error(f"Error examining raw results: {ex}")
                    
                    for r in results:
                        if hasattr(r, 'boxes'):
                            boxes = r.boxes
                            box_count = len(boxes)
                            detector_logger.info(f"Found {box_count} potential objects in frame {self.frame_count}")
                            
                            for box in boxes:
                                try:
                                    # Get class and confidence
                                    cls_id = int(box.cls.item())
                                    confidence = box.conf.item()
                                    
                                    # Always log all detected objects regardless of class or confidence
                                    # This helps diagnose if detections are happening at all
                                    if cls_id in SUPPORTED_CLASSES:
                                        class_name = CLASS_MAP.get(cls_id, f"class_{cls_id}")
                                        threshold_status = "ABOVE" if confidence >= self.confidence_threshold else "BELOW"
                                        detector_logger.info(
                                            f"Detected {class_name} with confidence {confidence:.3f} ({threshold_status} threshold={self.confidence_threshold})"
                                        )
                                    else:
                                        detector_logger.debug(f"Detected non-supported class {cls_id} with confidence {confidence:.3f}")
                                    
                                    # Only include objects we care about above confidence threshold
                                    if cls_id in SUPPORTED_CLASSES and confidence >= self.confidence_threshold:
                                        class_name = CLASS_MAP.get(cls_id, f"class_{cls_id}")
                                        
                                        # Get bounding box coordinates for visualization
                                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                                        
                                        detector_logger.info(
                                            f"Adding {class_name} to detections with confidence {confidence:.3f} at coordinates "
                                            f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] in frame {self.frame_count}"
                                        )
                                        
                                        detections.append({
                                            "class": cls_id,
                                            "class_name": class_name,
                                            "confidence": confidence,
                                            "box": [x1, y1, x2, y2]
                                        })
                                except Exception as ex:
                                    detector_logger.error(f"Error processing detection box: {ex}")
                    
                    # Disabled debug detection frame saving to improve performance
                    # Only log detection information without saving images
                    if detections:
                        detector_logger.info(f"Found {len(detections)} detections in frame {self.frame_count}")
                        
                        # Uncomment the following code to re-enable detection visualization for debugging
                        # # Create a copy of the frame for drawing
                        # detection_frame = frame_resized.copy()
                        # 
                        # # Draw boxes for each detection
                        # for det in detections:
                        #     box = det["box"]
                        #     cls_name = det["class_name"]
                        #     conf = det["confidence"]
                        #     
                        #     # Convert to int for drawing
                        #     x1, y1, x2, y2 = map(int, box)
                        #     
                        #     # Draw rectangle
                        #     cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        #     
                        #     # Draw label
                        #     label = f"{cls_name}: {conf:.2f}"
                        #     cv2.putText(detection_frame, label, (x1, y1 - 10), 
                        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # 
                        # # Save annotated frame
                        # detection_path = os.path.join(self.debug_frames_dir, f"detection_{self.frame_count}.jpg")
                        # cv2.imwrite(detection_path, detection_frame)
                        # detector_logger.info(f"Saved detection results to {detection_path}")
                    
                    # Update detection status
                    person_detections = [d for d in detections if d["class"] == 0]
                    pet_detections = [d for d in detections if d["class"] in [15, 16, 17]]
                    
                    # Get current and new detection state
                    new_people_detected = len(person_detections) > 0
                    new_pets_detected = len(pet_detections) > 0
                    new_people_count = len(person_detections)
                    new_pet_count = len(pet_detections)
                    
                    # Always log detection counts for debugging
                    detector_logger.info(
                        f"Detection counts for frame {self.frame_count}: "
                        f"people={new_people_count}, pets={new_pet_count}"
                    )
                    
                    # Check for changes in detection state
                    people_detected_changed = new_people_detected != self.people_detected
                    pets_detected_changed = new_pets_detected != self.pets_detected
                    people_count_changed = new_people_count != self.people_count
                    pet_count_changed = new_pet_count != self.pet_count
                    
                    # Log any changes
                    if people_detected_changed:
                        detector_logger.info(
                            f"People detection status changed: {self.people_detected} -> {new_people_detected} for {self.name}"
                        )
                    
                    if pets_detected_changed:
                        detector_logger.info(
                            f"Pet detection status changed: {self.pets_detected} -> {new_pets_detected} for {self.name}"
                        )
                    
                    if people_count_changed:
                        detector_logger.info(
                            f"People count changed: {self.people_count} -> {new_people_count} for {self.name}"
                        )
                    
                    if pet_count_changed:
                        detector_logger.info(
                            f"Pet count changed: {self.pet_count} -> {new_pet_count} for {self.name}"
                        )
                    
                    # Update object state
                    self.people_detected = new_people_detected
                    self.pets_detected = new_pets_detected
                    self.people_count = new_people_count
                    self.pet_count = new_pet_count
                    self.last_detection_time = current_time
                    self.last_update_time = current_time
                    self.connection_status = "connected"
                    
                    # Save the last processed frame for debugging
                    self.last_frame = frame_resized
                    
                    # Notify clients of the new detection
                    self._notify_clients()
                    
                    # Reset consecutive errors on successful detection
                    self.consecutive_errors = 0
                    
                    # Update last detection time
                    last_detection_time = current_time
                    
                except Exception as ex:
                    # Determine error type for better handling
                    error_type = "other"
                    if "CUDA" in str(ex) or "cuda" in str(ex):
                        error_type = "cuda"
                    elif "memory" in str(ex).lower():
                        error_type = "memory"
                    elif "stream" in str(ex).lower() or "cap" in str(ex).lower():
                        error_type = "stream"
                    
                    self._handle_error(error_type, ex)
                    self.consecutive_errors += 1
                    
                    # Close stream on consecutive errors
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        self._close_stream()
                    
                    # Wait before retrying
                    time.sleep(self.error_recovery_delay)
        
        except Exception as ex:
            detector_logger.error(f"Fatal error in detection loop for {self.name}: {str(ex)}", exc_info=True)
        
        finally:
            detector_logger.info(f"Detection loop ending for {self.name}")
            self._close_stream()
            self.is_running = False

    def _notify_clients(self) -> None:
        """Notify all connected clients of the current detection state."""
        state = {
            "human_detected": self.people_detected,
            "pet_detected": self.pets_detected,
            "human_count": self.people_count,
            "pet_count": self.pet_count,
            "last_update": time.time(),
            "connection_status": self.connection_status,
        }
        
        message = {
            "type": "state_update",
            "detector_id": self.detector_id,
            "state": state
        }
        
        # Log notification details
        detector_logger.info(
            f"Notifying clients for detector {self.detector_id}: "
            f"human_detected={self.people_detected}, pet_detected={self.pets_detected}, "
            f"human_count={self.people_count}, pet_count={self.pet_count}"
        )
        
        if self.detector_id in socket_clients:
            clients_count = len(socket_clients[self.detector_id])
            if clients_count > 0:
                detector_logger.info(f"Sending state update to {clients_count} connected clients for detector {self.detector_id}")
                
                disconnected_clients = set()
                
                for client in socket_clients[self.detector_id].copy():
                    # Use the updated write function that handles connection errors
                    if not write_message_to_socket(client, message):
                        # Mark this client for removal
                        disconnected_clients.add(client)
                
                # Remove disconnected clients
                if disconnected_clients:
                    with detector_lock:
                        for client in disconnected_clients:
                            if client in socket_clients[self.detector_id]:
                                socket_clients[self.detector_id].discard(client)
                        detector_logger.info(f"Removed {len(disconnected_clients)} disconnected clients for detector {self.detector_id}")
            else:
                detector_logger.info(f"No clients connected for detector {self.detector_id}, state update not sent")
        else:
            detector_logger.info(f"No client list found for detector {self.detector_id}, state update not sent")

    def shutdown(self) -> None:
        """Shut down the detector."""
        # Signal the thread to stop
        self.stop_event.set()
        
        # Wait for the thread to finish
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5)
            
        # Close the stream if it's still open
        self._close_stream()
        
        detector_logger.info(f"Detector {self.name} shut down")

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the detector."""
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
        }


def write_message_to_socket(sock: socket.socket, message: Dict[str, Any]) -> bool:
    """
    Write a message to a socket.
    
    Returns:
        bool: True if successful, False if connection broken
    """
    # First validate the socket is valid
    if sock is None:
        socket_logger.warning("Cannot send message to None socket")
        return False
        
    # Check socket validity
    try:
        # Check if socket is closed or invalid
        # This doesn't guarantee the socket is valid, but helps catch some cases
        sock.getpeername()
    except OSError:
        socket_logger.warning("Socket appears to be invalid (getpeername failed)")
        return False
    except Exception as ex:
        socket_logger.warning(f"Socket validation failed: {str(ex)}")
        return False
        
    try:
        # Convert message to JSON
        json_data = json.dumps(message).encode("utf-8")
        
        # Prefix with message length (4 bytes)
        length = len(json_data)
        
        # Check if disk space is critically low before socket operations
        # This helps avoid errors from disk-full conditions
        if check_disk_space() > 95:  # extremely critical
            socket_logger.error("Disk space critically low (>95%), cannot reliably send messages")
            return False
            
        # Use a non-blocking socket with timeout to avoid hangs
        sock.settimeout(5)  # 5 second timeout
        
        # Send data in chunks to be safer
        try:
            # Send length prefix
            sock.sendall(length.to_bytes(4, byteorder="big"))
            # Send message data
            sock.sendall(json_data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as conn_err:
            socket_logger.warning(f"Connection error during send: {str(conn_err)}")
            return False
        except socket.timeout:
            socket_logger.warning("Socket send operation timed out")
            return False
        
        # Log the message being sent
        message_type = message.get("type", "unknown")
        detector_id = message.get("detector_id", "none")
        
        # Create a sanitized copy for logging (without large data fields)
        log_message = message.copy()
        if "state" in log_message:
            log_message["state"] = f"[State data with {len(str(log_message['state']))} chars]"
        
        # Enhanced logging for better visibility
        socket_logger.info(f"SENT to client: type={message_type}, detector={detector_id}")
        socket_logger.debug(f"SENT message details: {log_message}")
        
        return True
        
    except BrokenPipeError:
        socket_logger.warning(f"Client connection closed (broken pipe) when sending {message.get('type', 'unknown')}")
        return False
    except ConnectionResetError:
        socket_logger.warning(f"Connection reset by client when sending {message.get('type', 'unknown')}")
        return False
    except OSError as os_err:
        socket_logger.warning(f"OS error during socket operation: {str(os_err)}")
        return False
    except Exception as ex:
        socket_logger.error(f"Error writing to socket: {str(ex)}", exc_info=True)
        return False
    finally:
        # Reset timeout to default
        try:
            sock.settimeout(None)
        except:
            pass


def handle_client_connection(sock: socket.socket, addr: Tuple[str, int]) -> None:
    """Handle a client connection."""
    socket_logger.info(f"New client connected from {addr}")
    socket_logger.debug("Debug logging is enabled for socket connections")
    detector_id = None
    
    # Set socket timeout to prevent hangs
    try:
        sock.settimeout(30)  # 30 second timeout for initial authentication
    except Exception as timeout_ex:
        socket_logger.warning(f"Failed to set socket timeout: {timeout_ex}")
    
    try:
        # Check disk space before accepting new connections
        if check_disk_space() > 95:  # extremely critical
            socket_logger.error(f"Disk space critically low, refusing client connection from {addr}")
            try:
                error_msg = {"type": "error", "message": "Server disk space critical, please try again later"}
                json_data = json.dumps(error_msg).encode("utf-8")
                length = len(json_data)
                sock.sendall(length.to_bytes(4, byteorder="big"))
                sock.sendall(json_data)
                sock.close()
            except:
                pass
            return
            
        # Read client's authentication message
        socket_logger.debug(f"Waiting for authentication message from client {addr}")
        try:
            # Read message length with timeout
            length_data = sock.recv(4)
            if not length_data or len(length_data) < 4:
                socket_logger.warning(f"Client {addr} disconnected during authentication (invalid length prefix)")
                return
                
            length = int.from_bytes(length_data, byteorder="big")
            
            # Sanity check message length
            if length <= 0 or length > 1024 * 1024:  # Max 1MB
                socket_logger.warning(f"Invalid message length {length} from client {addr}")
                return
                
            socket_logger.debug(f"Received message length: {length} bytes from client {addr}")
            
            # Read message data with timeout
            data = b""
            remaining = length
            
            while remaining > 0:
                chunk = sock.recv(min(remaining, 8192))  # Read in chunks
                if not chunk:  # Connection closed
                    socket_logger.warning(f"Connection closed by client {addr} during read")
                    return
                data += chunk
                remaining -= len(chunk)
                
        except socket.timeout:
            socket_logger.warning(f"Timeout receiving authentication from client {addr}")
            return
        except (ConnectionResetError, BrokenPipeError, OSError) as conn_err:
            socket_logger.warning(f"Connection error during authentication from client {addr}: {conn_err}")
            return
        
        try:
            message = json.loads(data.decode("utf-8"))
            message_type = message.get("type", "unknown")
            detector_id = message.get("detector_id", "none")
            socket_logger.info(f"RECEIVED from client {addr}: type={message_type}, detector={detector_id}")
            socket_logger.debug(f"RECEIVED message details: {message}")
        except json.JSONDecodeError:
            socket_logger.warning(f"Invalid JSON from client {addr}: {data}")
            return
        except Exception as decode_ex:
            socket_logger.warning(f"Error decoding message from client {addr}: {decode_ex}")
            return
        
        if message.get("type") != "auth":
            socket_logger.warning(f"Expected auth message, got {message.get('type')} from client {addr}")
            response = {"type": "error", "message": "Authentication required"}
            if not write_message_to_socket(sock, response):
                socket_logger.warning(f"Connection lost with client {addr} while sending error message")
            return
        
        detector_id = message.get("detector_id")
        if not detector_id:
            socket_logger.warning(f"No detector_id provided by client {addr}")
            response = {"type": "error", "message": "detector_id required"}
            if not write_message_to_socket(sock, response):
                socket_logger.warning(f"Connection lost with client {addr} while sending error message")
            return
        
        # Respond with auth success
        response = {"type": "auth_success"}
        if not write_message_to_socket(sock, response):
            socket_logger.warning(f"Connection lost with client {addr} while sending auth success message")
            return
        
        # Register client with detector
        with detector_lock:
            if detector_id not in socket_clients:
                socket_clients[detector_id] = set()
            
            # Remove any dead connections with the same socket address
            to_remove = set()
            for existing_sock in socket_clients[detector_id]:
                try:
                    # Try to get peer name - will fail if socket is dead
                    existing_sock.getpeername()
                except:
                    to_remove.add(existing_sock)
            
            # Remove dead connections
            for dead_sock in to_remove:
                socket_clients[detector_id].discard(dead_sock)
                try:
                    dead_sock.close()
                except:
                    pass
                
            # Add new connection
            socket_clients[detector_id].add(sock)
        
        # Set a longer timeout for established connection
        try:
            sock.settimeout(60)  # 60 second timeout for normal operation
        except Exception as timeout_ex:
            socket_logger.warning(f"Failed to set socket timeout: {timeout_ex}")
            
        # Enter main loop
        while True:
            # Wait for client message
            try:
                socket_logger.debug(f"Waiting for next message from client {addr}")
                length_data = sock.recv(4)
                if not length_data or len(length_data) < 4:
                    socket_logger.debug(f"Client {addr} disconnected (invalid length prefix)")
                    break
                    
                length = int.from_bytes(length_data, byteorder="big")
                
                # Sanity check message length
                if length <= 0 or length > 1024 * 1024:  # Max 1MB
                    socket_logger.warning(f"Invalid message length {length} from client {addr}")
                    break
                    
                socket_logger.debug(f"Received message length: {length} bytes from client {addr}")
                
                # Read message data with timeout
                data = b""
                remaining = length
                
                while remaining > 0:
                    chunk = sock.recv(min(remaining, 8192))  # Read in chunks
                    if not chunk:  # Connection closed
                        socket_logger.warning(f"Connection closed by client {addr} during read")
                        return
                    data += chunk
                    remaining -= len(chunk)
            except socket.timeout:
                socket_logger.warning(f"Timeout receiving message from client {addr}")
                break
            except (ConnectionResetError, BrokenPipeError, OSError) as conn_err:
                socket_logger.warning(f"Connection error receiving message from client {addr}: {conn_err}")
                break
            
            try:
                message = json.loads(data.decode("utf-8"))
                message_type = message.get("type", "unknown")
                message_detector_id = message.get("detector_id", "none")
                socket_logger.info(f"RECEIVED message from client {addr}: type={message_type}, detector={message_detector_id}")
                socket_logger.debug(f"RECEIVED message details: {message}")
                
                # Update detector_id if it was provided in the message
                if message_detector_id and message_detector_id != "none":
                    detector_id = message_detector_id
            except json.JSONDecodeError:
                socket_logger.warning(f"Invalid JSON from client {addr}: {data}")
                continue
            except Exception as decode_ex:
                socket_logger.warning(f"Error decoding message from client {addr}: {decode_ex}")
                continue
            
            # Process message based on type
            try:
                message_type = message.get("type", "unknown")
                
                if message_type == "heartbeat":
                    # Respond with heartbeat
                    response = {"type": "heartbeat"}
                    if not write_message_to_socket(sock, response):
                        socket_logger.warning(f"Failed to send heartbeat response to client {addr}")
                        break
                    
                elif message_type == "get_state":
                    # Send current detector state
                    socket_logger.info(f"Received get_state request from client {addr} for detector {detector_id}")
                    
                    with detector_lock:
                        if detector_id in detectors:
                            detector = detectors[detector_id]
                            socket_logger.info(f"Found detector {detector_id}, checking state...")
                            
                            # Explicitly log detector properties
                            socket_logger.info(f"Detector {detector_id} availability: stream_url={detector.stream_url}, is_running={detector.is_running}")
                            socket_logger.info(f"Detector {detector_id} detection state: people_detected={detector.people_detected}, pets_detected={detector.pets_detected}")
                            socket_logger.info(f"Detector {detector_id} counts: people_count={detector.people_count}, pet_count={detector.pet_count}")
                            
                            # Check stream status but don't attempt to read frames
                            if detector.cap is not None and detector.cap.isOpened():
                                socket_logger.info(f"Stream appears to be open for detector {detector_id}")
                            else:
                                socket_logger.warning(f"No valid capture device for detector {detector_id}")
                            
                            # Get state and send response
                            try:
                                state = detector.get_state()
                                response = {
                                    "type": "state_update",
                                    "detector_id": detector_id,
                                    "state": {
                                        "human_detected": detector.people_detected,
                                        "pet_detected": detector.pets_detected,
                                        "human_count": detector.people_count,
                                        "pet_count": detector.pet_count,
                                        "last_update": time.time(),
                                        "connection_status": detector.connection_status,
                                    }
                                }
                                if not write_message_to_socket(sock, response):
                                    socket_logger.warning(f"Connection lost with client {addr} while sending state update")
                                    break
                            except Exception as state_ex:
                                socket_logger.error(f"Error getting detector state: {state_ex}")
                                response = {"type": "error", "message": "Error retrieving detector state"}
                                if not write_message_to_socket(sock, response):
                                    break
                        else:
                            # Detector not found - Send notification
                            response = {
                                "type": "detector_not_found",
                                "detector_id": detector_id
                            }
                            if not write_message_to_socket(sock, response):
                                socket_logger.warning(f"Connection lost with client {addr} while sending detector_not_found")
                                break
                            
                elif message_type == "create_detector":
                    # Check disk space before creating detector (which generates a lot of files)
                    if check_disk_space() > 90:
                        socket_logger.warning(f"Disk space critically low, refusing to create detector for client {addr}")
                        response = {"type": "error", "message": "Server disk space critical, cannot create detector"}
                        if not write_message_to_socket(sock, response):
                            break
                        continue
                    
                    # Create new detector
                    config = message.get("config", {})
                    
                    socket_logger.info(f"Processing create_detector request for detector_id={detector_id}")
                    socket_logger.debug(f"Detector config: {config}")
                    
                    with detector_lock:
                        if detector_id in detectors:
                            # Detector already exists, update config
                            detector = detectors[detector_id]
                            socket_logger.info(f"Updating existing detector {detector_id}")
                            
                            # Update detector configuration
                            if config.get("stream_url"):
                                detector.stream_url = config["stream_url"]
                            if config.get("name"):
                                detector.name = config["name"]
                            if config.get("model"):
                                # Model changes require reinitialization, which we'll skip here
                                # Just update the field but don't reload the model
                                detector.model_name = config["model"]
                            if config.get("input_size"):
                                try:
                                    width, height = map(int, config["input_size"].split("x"))
                                    detector.input_width = width
                                    detector.input_height = height
                                except:
                                    pass
                            if config.get("detection_interval"):
                                detector.detection_interval = config["detection_interval"]
                            if config.get("confidence_threshold"):
                                detector.confidence_threshold = config["confidence_threshold"]
                            if config.get("frame_skip_rate"):
                                detector.frame_skip_rate = config["frame_skip_rate"]
                            
                            # Save the updated configuration - only if disk space allows
                            if check_disk_space() < 90:
                                socket_logger.info(f"Saving updated configuration for detector {detector_id}")
                                save_detectors_config()
                            else:
                                socket_logger.warning("Skipping config save due to low disk space")
                                
                            response = {"type": "detector_updated", "detector_id": detector_id}
                        else:
                            # Create new detector
                            socket_logger.info(f"Creating new detector {detector_id}")
                            detector = YoloDetector(detector_id, config)
                            
                            # Log detector details before initialization
                            socket_logger.info(f"New detector details: stream_url={config.get('stream_url')}, name={config.get('name')}, model={config.get('model')}")
                            
                            if detector.initialize():
                                detectors[detector_id] = detector
                                # Save detector configuration when a new detector is created - if disk space allows
                                socket_logger.info(f"Successfully initialized detector {detector_id}, saving configuration")
                                
                                if check_disk_space() < 90:
                                    # Immediately save the configuration to ensure it's persisted
                                    save_detectors_config()
                                    # Verify the configuration was saved correctly
                                    try:
                                        if os.path.exists(DETECTORS_CONFIG_FILE):
                                            with open(DETECTORS_CONFIG_FILE, "r") as f:
                                                config_content = f.read()
                                                socket_logger.info(f"Verified detector config file contains: {config_content}")
                                        else:
                                            socket_logger.error(f"Config file not found after saving: {DETECTORS_CONFIG_FILE}")
                                            # Attempt to save again
                                            save_detectors_config()
                                    except Exception as verify_ex:
                                        socket_logger.error(f"Error verifying saved configuration: {str(verify_ex)}", exc_info=True)
                                else:
                                    socket_logger.warning("Skipping config save due to low disk space")
                                
                                response = {"type": "detector_created", "detector_id": detector_id}
                            else:
                                socket_logger.error(f"Failed to initialize detector {detector_id}")
                                response = {"type": "error", "message": "Failed to initialize detector"}
                                
                        if not write_message_to_socket(sock, response):
                            socket_logger.warning(f"Connection lost with client {addr} while sending detector create/update response")
                            break
                        
                else:
                    socket_logger.warning(f"Unknown message type: {message_type} from client {addr}")
                    
            except Exception as process_ex:
                socket_logger.error(f"Error processing message from client {addr}: {process_ex}", exc_info=True)
                # Don't exit the loop on a processing error unless it's critical
                if isinstance(process_ex, (OSError, BrokenPipeError, ConnectionResetError)):
                    break
                
    except ConnectionResetError:
        socket_logger.debug(f"Connection reset by client {addr}")
    except BrokenPipeError:
        socket_logger.debug(f"Connection broken with client {addr}")
    except socket.timeout:
        socket_logger.debug(f"Connection timed out for client {addr}")
    except OSError as os_err:
        socket_logger.debug(f"Socket error for client {addr}: {os_err}")
    except Exception as ex:
        socket_logger.error(f"Error handling client {addr}: {str(ex)}", exc_info=True)
    finally:
        # Clean up
        try:
            sock.close()
        except:
            pass
            
        # Remove from clients list
        if detector_id and detector_id in socket_clients:
            with detector_lock:
                if sock in socket_clients[detector_id]:
                    socket_clients[detector_id].remove(sock)
                    
                    # If no more clients for this detector, keep it running
                    # Don't shut down immediately, detector may be reused
                    if not socket_clients[detector_id]:
                        socket_logger.info(f"No more clients for detector {detector_id}, but keeping it running for now")


def check_periodic_save() -> None:
    """Check if it's time to periodically save detector configurations."""
    global last_config_save_time
    current_time = time.time()
    
    # First check disk space to avoid failures due to disk full
    if check_disk_space() > 90:  # If disk space usage is more than 90%
        logger.warning("Disk space is critically low, skipping configuration save")
        # Update the time anyway to avoid repeated attempts
        last_config_save_time = current_time
        return
    
    # Save configurations periodically
    if current_time - last_config_save_time > CONFIG_SAVE_INTERVAL:
        save_detectors_config()
        last_config_save_time = current_time
        
def check_disk_space() -> float:
    """Check available disk space and perform cleanup if needed.
    
    Returns:
        float: Disk usage percentage
    """
    try:
        import os
        import shutil
        
        # Get disk usage statistics
        total, used, free = shutil.disk_usage(os.getcwd())
        usage_percent = (used / total) * 100
        
        # Log disk usage
        logger.info(f"Disk usage: {usage_percent:.1f}% (used: {used/1024/1024:.1f} MB, free: {free/1024/1024:.1f} MB)")
        
        # If disk usage is above 85%, perform cleanup
        if usage_percent > 85:
            logger.warning(f"Disk usage is high ({usage_percent:.1f}%), cleaning up old files")
            
            # Clean up debug frames directory
            cleanup_old_files(os.path.join(os.getcwd(), "debug_frames"), days=1)
            
            # Clean up logs directory
            log_dir = os.path.join(os.getcwd(), "logs")
            if os.path.exists(log_dir):
                cleanup_old_files(log_dir, days=3)
                
        return usage_percent
    except Exception as ex:
        logger.error(f"Error checking disk space: {ex}", exc_info=True)
        return 0.0  # Return 0 on error
        
def cleanup_old_files(directory: str, days: int = 7) -> None:
    """Delete files older than the specified number of days.
    
    Args:
        directory: Directory to clean up
        days: Delete files older than this many days
    """
    if not os.path.exists(directory):
        return
        
    try:
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)
        
        count = 0
        size = 0
        
        logger.info(f"Cleaning up files older than {days} days in {directory}")
        
        for root, dirs, files in os.walk(directory, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_mtime = os.path.getmtime(file_path)
                    file_size = os.path.getsize(file_path)
                    
                    if file_mtime < cutoff_time:
                        os.remove(file_path)
                        count += 1
                        size += file_size
                except Exception as ex:
                    logger.error(f"Error removing file {file_path}: {ex}")
                    
        # Also remove empty directories
        for root, dirs, files in os.walk(directory, topdown=False):
            for d in dirs:
                dir_path = os.path.join(root, d)
                try:
                    if not os.listdir(dir_path):  # Check if directory is empty
                        os.rmdir(dir_path)
                except Exception as ex:
                    logger.error(f"Error removing directory {dir_path}: {ex}")
        
        if count > 0:
            logger.info(f"Cleaned up {count} files ({size/1024/1024:.1f} MB) in {directory}")
    except Exception as ex:
        logger.error(f"Error during cleanup: {ex}", exc_info=True)


def start_socket_server() -> None:
    """Start the socket server."""
    global socket_server
    
    # First check disk space on startup
    initial_disk_usage = check_disk_space()
    if initial_disk_usage > 90:
        logger.critical(f"WARNING: Disk space critically low ({initial_disk_usage:.1f}%)! This may cause server instability.")
        # Force an aggressive cleanup to free space
        cleanup_old_files(os.path.join(os.getcwd(), "debug_frames"), days=0)  # Delete all debug frames
        
        # Check again after cleanup
        post_cleanup_usage = check_disk_space()
        if post_cleanup_usage > 90:
            logger.critical(f"Disk space still critically low ({post_cleanup_usage:.1f}%) after cleanup attempt!")
            logger.critical("You should manually free up disk space to prevent server failures.")
    
    try:
        # Create a TCP socket
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Set socket timeout to avoid blocking forever
        server.settimeout(60)  # 60 second timeout
        
        # Bind to all interfaces
        server.bind(("0.0.0.0", TCP_PORT))
        
        # Start listening
        server.listen(5)
        socket_server = server
        
        socket_logger.info(f"Socket server started on port {TCP_PORT}")
        
        # Start configuration save thread
        save_thread = threading.Thread(target=periodic_save_loop)
        save_thread.daemon = True
        save_thread.start()
        
        # Start disk space monitoring thread
        disk_monitor_thread = threading.Thread(target=disk_monitor_loop)
        disk_monitor_thread.daemon = True
        disk_monitor_thread.start()
        
        # Accept connections
        while True:
            try:
                client_sock, client_addr = server.accept()
                
                # Check disk space on each new connection
                disk_usage = check_disk_space()
                if disk_usage > 95:  # extremely critical
                    logger.critical(f"Disk space critically low ({disk_usage:.1f}%)! Refusing new connections.")
                    try:
                        client_sock.close()
                    except:
                        pass
                    continue
                
                client_thread = threading.Thread(
                    target=handle_client_connection,
                    args=(client_sock, client_addr)
                )
                client_thread.daemon = True
                client_thread.start()
                
                # Check if we should save configurations
                check_periodic_save()
                
            except socket.timeout:
                # This is normal, just continue
                continue
            except Exception as accept_ex:
                socket_logger.error(f"Error accepting connection: {str(accept_ex)}")
                # Don't exit the loop on a single accept error
                # Just wait a bit and try again
                time.sleep(5)
            
    except Exception as ex:
        socket_logger.error(f"Fatal error in socket server: {str(ex)}", exc_info=True)
        
        # Try to shutdown gracefully
        if socket_server:
            try:
                socket_server.close()
            except:
                pass
                
        # Try to restart socket server after a delay
        logger.info("Attempting to restart socket server in 30 seconds...")
        time.sleep(30)
        start_socket_server()  # Recursive restart


def disk_monitor_loop() -> None:
    """Background thread to monitor disk space."""
    while True:
        try:
            # Check disk space every 5 minutes
            disk_usage = check_disk_space()
            
            # If disk space is critically low, take action
            if disk_usage > 95:  # extremely critical
                logger.critical(f"Disk space CRITICALLY LOW: {disk_usage:.1f}%")
                logger.critical("Performing emergency cleanup to free space")
                
                # Emergency cleanup - delete all debug frames
                cleanup_old_files(os.path.join(os.getcwd(), "debug_frames"), days=0)
                
                # Also remove older logs
                log_dir = os.path.join(os.getcwd(), "logs")
                if os.path.exists(log_dir):
                    cleanup_old_files(log_dir, days=1)  # Keep only 1 day of logs
                    
                # Check if the cleanup helped
                new_usage = check_disk_space()
                if new_usage > 90:
                    logger.critical(f"Disk space still critically low after cleanup: {new_usage:.1f}%")
                    logger.critical("Manual intervention required to free disk space!")
                else:
                    logger.info(f"Cleanup freed some space, disk usage now: {new_usage:.1f}%")
                    
            elif disk_usage > 85:  # high but not critical
                logger.warning(f"Disk usage high: {disk_usage:.1f}%")
                # Regular cleanup
                cleanup_old_files(os.path.join(os.getcwd(), "debug_frames"), days=1)
                
            # Sleep for 5 minutes before checking again
            time.sleep(300)
            
        except Exception as ex:
            logger.error(f"Error in disk monitor loop: {ex}", exc_info=True)
            # Sleep for 10 minutes on error
            time.sleep(600)


def periodic_save_loop() -> None:
    """Background thread to periodically save detector configurations."""
    global last_config_save_time
    last_config_save_time = time.time()  # Initialize with current time
    
    while True:
        try:
            current_time = time.time()
            if current_time - last_config_save_time > CONFIG_SAVE_INTERVAL:
                save_detectors_config()
                last_config_save_time = current_time
            
            # Sleep for a while before checking again
            time.sleep(60)  # Check every minute
        except Exception as ex:
            logger.error(f"Error in periodic save loop: {str(ex)}", exc_info=True)
            time.sleep(300)  # On error, wait 5 minutes before trying again


def save_detectors_config() -> None:
    """Save detector configurations to a JSON file."""
    logger.info(f"Attempting to save detector configurations, current count: {len(detectors)}")
    
    # Debug: Check detector contents
    with detector_lock:
        for detector_id, detector in detectors.items():
            logger.info(f"Found detector to save: {detector_id}, name={detector.name}, stream_url={detector.stream_url}")
    
    # Create config directory if it doesn't exist - be extra sure
    try:
        logger.info(f"Ensuring config directory exists: {DETECTORS_CONFIG_DIR}")
        os.makedirs(DETECTORS_CONFIG_DIR, exist_ok=True)
        # Check that the directory was created successfully
        if os.path.exists(DETECTORS_CONFIG_DIR):
            logger.info(f"Config directory exists and is accessible")
            # Check directory permissions
            dir_stat = os.stat(DETECTORS_CONFIG_DIR)
            logger.info(f"Directory permissions: {oct(dir_stat.st_mode)}")
        else:
            logger.error(f"Config directory doesn't exist after makedirs call")
    except Exception as dir_ex:
        logger.error(f"Error creating config directory: {dir_ex}", exc_info=True)
    
    # Collect configurations
    detector_configs = {}
    with detector_lock:
        for detector_id, detector in detectors.items():
            # Store only the configuration data, not the entire detector object
            detector_configs[detector_id] = {
                "stream_url": detector.stream_url,
                "name": detector.name,
                "model": detector.model_name,
                "input_size": f"{detector.input_width}x{detector.input_height}",
                "detection_interval": detector.detection_interval,
                "confidence_threshold": detector.confidence_threshold,
                "frame_skip_rate": detector.frame_skip_rate
            }
    
    # Debug: Check what we're about to save
    logger.info(f"Preparing to save {len(detector_configs)} detector configurations")
    
    try:
        # Make sure the config directory exists before trying to write to it
        os.makedirs(os.path.dirname(DETECTORS_CONFIG_FILE), exist_ok=True)
        
        # Debug: Check file path
        logger.info(f"Writing detector configurations to file: {DETECTORS_CONFIG_FILE}")
        
        # Debug: Dump what we're saving to the log
        config_json = json.dumps(detector_configs, indent=2)
        logger.info(f"Detector configurations to save: {config_json}")
        
        # Ensure we actually have data to save
        if not detector_configs:
            logger.warning("No detector configurations to save, writing empty object to avoid file errors")
            detector_configs = {}  # Ensure we write a valid empty JSON object
        
        # Write to file - use fsync to ensure data is physically written to disk
        with open(DETECTORS_CONFIG_FILE, "w") as f:
            # The bug is likely here - make sure we're writing the right JSON data
            # Make sure we're actually writing the detector_configs data, not an empty dict
            # If detector_configs is empty, initialize it as an empty dict for JSON
            if not detector_configs:
                logger.warning("No detector configurations to save, writing empty object")
                config_json = "{}"
            
            logger.info(f"About to write to file: {config_json}")
            f.write(config_json)
            f.flush()
            # Try to force physical write to disk
            try:
                os.fsync(f.fileno())
                logger.info("Successfully flushed file to disk with fsync")
            except Exception as fsync_ex:
                logger.warning(f"Could not fsync file: {fsync_ex}")
        
        # Verify the file was written
        if os.path.exists(DETECTORS_CONFIG_FILE):
            file_size = os.path.getsize(DETECTORS_CONFIG_FILE)
            logger.info(f"Config file written successfully, size: {file_size} bytes")
            
            # Read back the file to verify contents
            try:
                with open(DETECTORS_CONFIG_FILE, "r") as f:
                    saved_content = f.read()
                    logger.info(f"Verified file contains: {saved_content}")
                    
                    # Check if content is empty
                    if not saved_content.strip():
                        logger.error("File exists but content is empty after write!")
                        # Try writing again directly
                        with open(DETECTORS_CONFIG_FILE, "w") as fw:
                            fw.write(config_json)
                            fw.flush()
                            logger.info("Attempted second write to empty file")
            except Exception as read_ex:
                logger.error(f"Error reading back config file: {read_ex}", exc_info=True)
        else:
            logger.error(f"Config file does not exist after writing!")
        
        # Log additional information about what was saved
        if detector_configs:
            detector_ids = ", ".join(detector_configs.keys())
            logger.info(f"Saved {len(detector_configs)} detector configurations to {DETECTORS_CONFIG_FILE}: {detector_ids}")
        else:
            logger.info(f"No detectors to save to {DETECTORS_CONFIG_FILE}")
    except Exception as ex:
        logger.error(f"Failed to save detector configurations: {str(ex)}", exc_info=True)


def load_detectors_config() -> Dict[str, Dict[str, Any]]:
    """Load detector configurations from a JSON file."""
    logger.info(f"Attempting to load detector configurations from {DETECTORS_CONFIG_FILE}")
    
    if not os.path.exists(DETECTORS_CONFIG_FILE):
        logger.info(f"No detector configuration file found at {DETECTORS_CONFIG_FILE}")
        # Check if the directory exists
        config_dir = os.path.dirname(DETECTORS_CONFIG_FILE)
        if os.path.exists(config_dir):
            logger.info(f"Config directory exists: {config_dir}")
            # List contents of the directory for debugging
            try:
                dir_contents = os.listdir(config_dir)
                logger.info(f"Config directory contents: {dir_contents}")
            except Exception as dir_ex:
                logger.error(f"Error listing config directory: {dir_ex}")
        else:
            logger.info(f"Config directory does not exist: {config_dir}")
        return {}
    
    # File exists, check its size and permissions
    try:
        file_stat = os.stat(DETECTORS_CONFIG_FILE)
        logger.info(f"Config file exists, size: {file_stat.st_size} bytes, permissions: {oct(file_stat.st_mode)}")
        
        # Read file contents for debugging
        try:
            with open(DETECTORS_CONFIG_FILE, "r") as f:
                file_content = f.read()
                logger.info(f"Raw file content: {file_content}")
                
                if not file_content.strip():
                    logger.warning(f"Config file is empty")
                    return {}
                
                detector_configs = json.loads(file_content)
                
                # Ensure we have a dictionary, not an array
                if isinstance(detector_configs, list):
                    logger.warning(f"Invalid format in {DETECTORS_CONFIG_FILE} (got array instead of object), resetting to empty object")
                    detector_configs = {}
                
                if detector_configs:
                    detector_ids = ", ".join(detector_configs.keys())
                    logger.info(f"Loaded {len(detector_configs)} detector configurations from {DETECTORS_CONFIG_FILE}: {detector_ids}")
                    logger.info(f"Loaded configurations: {json.dumps(detector_configs, indent=2)}")
                else:
                    logger.info(f"No detector configurations found in {DETECTORS_CONFIG_FILE}")
                
                return detector_configs
        except json.JSONDecodeError as json_err:
            logger.error(f"JSON parsing error in {DETECTORS_CONFIG_FILE}: {json_err}")
            # If the file is empty or invalid, start with an empty dict
            return {}
    except Exception as ex:
        logger.error(f"Failed to load detector configurations: {str(ex)}", exc_info=True)
        return {}


def initialize_saved_detectors() -> None:
    """Initialize detectors from saved configurations."""
    logger.info("Starting to initialize saved detectors")
    detector_configs = load_detectors_config()
    
    if not detector_configs:
        logger.info("No saved detector configurations found")
        return
    
    logger.info(f"Found {len(detector_configs)} saved detector configurations to initialize")
    
    with detector_lock:
        for detector_id, config in detector_configs.items():
            try:
                logger.info(f"Initializing detector {detector_id} from saved configuration: {json.dumps(config)}")
                
                # Validate required configuration fields
                required_fields = ["stream_url"]
                missing_fields = [field for field in required_fields if not config.get(field)]
                
                if missing_fields:
                    logger.error(f"Cannot initialize detector {detector_id}: missing required fields: {missing_fields}")
                    continue
                
                # Create and initialize the detector
                detector = YoloDetector(detector_id, config)
                logger.info(f"Created detector object for {detector_id}, attempting to initialize")
                
                # Try to initialize the detector
                if detector.initialize():
                    detectors[detector_id] = detector
                    logger.info(f"Successfully initialized detector {detector_id} from saved configuration")
                else:
                    logger.error(f"Failed to initialize detector {detector_id} from saved configuration")
            except Exception as ex:
                logger.error(f"Error initializing detector {detector_id}: {str(ex)}", exc_info=True)
        
        # Log initialization summary
        if len(detectors) > 0:
            detector_ids = ", ".join(detectors.keys())
            logger.info(f"Successfully initialized {len(detectors)} detectors: {detector_ids}")
        else:
            logger.warning("No detectors could be initialized from saved configurations")


def cleanup_detectors() -> None:
    """Clean up detectors and other resources."""
    # Save detector configurations before shutting down
    save_detectors_config()
    
    with detector_lock:
        for detector_id, detector in detectors.items():
            try:
                detector.shutdown()
            except Exception as ex:
                detector_logger.error(f"Error shutting down detector {detector_id}: {str(ex)}")
                
    # Close socket server
    if socket_server:
        try:
            socket_server.close()
        except Exception as ex:
            socket_logger.error(f"Error closing socket server: {str(ex)}")


def handle_signals() -> None:
    """Handle signals for graceful shutdown."""
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        cleanup_detectors()
        os._exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main() -> None:
    """Main entry point."""
    # Print banner
    print("\n" + "=" * 80)
    print("YOLO Presence Detection Server")
    print("Version 1.0")
    print("=" * 80 + "\n")
    
    logger.info("Starting YOLO Presence Detection Server")
    
    # Print debug information about the environment
    logger.info("Python version: " + sys.version)
    logger.info("OpenCV version: " + cv2.__version__)
    logger.info("PyTorch version: " + torch.__version__)
    
    # Check for GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        logger.info(f"CUDA available, {gpu_count} device(s): {', '.join(device_names)}")
    else:
        logger.warning("CUDA not available, using CPU only")
    
    # Create debug frames directory in current working directory
    debug_dir = os.path.join(os.getcwd(), "debug_frames")
    os.makedirs(debug_dir, exist_ok=True)
    logger.info(f"Debug frames will be saved to {debug_dir}")
    
    # Create config directory if it doesn't exist
    os.makedirs(DETECTORS_CONFIG_DIR, exist_ok=True)
    logger.info(f"Detector configurations will be saved to {DETECTORS_CONFIG_DIR}")
    logger.info(f"Detector config file path: {DETECTORS_CONFIG_FILE}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Print directory permissions
    try:
        directory_info = os.stat(DETECTORS_CONFIG_DIR)
        logger.info(f"Config directory permissions: {oct(directory_info.st_mode)}")
    except Exception as ex:
        logger.error(f"Error checking config directory permissions: {ex}")
    
    # Force debug logging for detector module
    detector_logger.setLevel(logging.DEBUG)
    
    # Print a direct DEBUG message to test logging
    detector_logger.debug("This is a test DEBUG message from detector - if you see this, debug logging is working")
    detector_logger.info("This is a test INFO message from detector")
    
    # Set up signal handling
    handle_signals()
    
    # Start the system watchdog
    start_watchdog()
    
    # Load and initialize saved detectors
    initialize_saved_detectors()
    
    # Start the socket server
    start_socket_server()


if __name__ == "__main__":
    main()