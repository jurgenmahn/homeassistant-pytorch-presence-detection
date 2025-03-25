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
from typing import Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import cv2
import torch
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
        """Initialize the detector and start the detection process."""
        try:
            logger.info(
                f"Initializing detector {self.detector_id} with model {self.model_name}"
            )

            # Load the model
            self._load_model()

            # Open the video stream - don't fail if stream isn't available yet
            # We'll retry in the detection loop
            stream_opened = self._open_stream()
            if not stream_opened:
                logger.warning(
                    f"Could not open stream for {self.detector_id}, will retry in detection loop"
                )

            # Start detection thread
            self.stop_event.clear()
            self.detection_thread = threading.Thread(target=self._detection_loop)
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
                if '@' in netloc:
                    userpass, hostport = netloc.split('@', 1)
                    if ':' in userpass:
                        user, _ = userpass.split(':', 1)
                        masked_netloc = f"{user}:****@{hostport}"
                    else:
                        masked_netloc = f"{userpass}@{hostport}"
                    
                    masked_url = f"{parsed_url.scheme}://{masked_netloc}{parsed_url.path}"
                    logger.info(f"Opening stream: {masked_url}")
                else:
                    logger.info(f"Opening stream: {self.stream_url}")
            except Exception:
                logger.info(f"Opening stream (URL parsing failed)")
            
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
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Small buffer to reduce latency
                
                # For RTSP streams, set additional parameters
                if self.stream_url.startswith("rtsp://"):
                    # Try to set preferred codec
                    try:
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'H264'))
                    except Exception as ex:
                        logger.debug(f"Could not set codec: {ex}")
                    
                    # Try using TCP transport for better reliability
                    # Different OpenCV versions use different constant names
                    try:
                        # Try commonly used property IDs for RTSP transport
                        transport_props = [
                            ("CAP_PROP_RTSP_TRANSPORT", 0),  # Newer OpenCV
                            (78, 0),  # Direct property ID
                            ("CV_CAP_PROP_RTSP_TRANSPORT", 0)  # Older OpenCV
                        ]
                        
                        for prop, value in transport_props:
                            try:
                                if isinstance(prop, str):
                                    if hasattr(cv2, prop):
                                        self.cap.set(getattr(cv2, prop), value)
                                        logger.info(f"Set RTSP transport to TCP using {prop}")
                                        break
                                else:
                                    self.cap.set(prop, value)
                                    logger.info("Set RTSP transport to TCP using property ID")
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
                    logger.warning(f"Stream opened but failed to read frame for {self.detector_id}")
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

    def _detection_loop(self) -> None:
        """Main detection loop."""
        logger.info(f"Starting detection loop for {self.detector_id}")

        last_detection_time = 0
        frame_skip_counter = 0

        while not self.stop_event.is_set():
            try:
                # Check if we should process a frame based on detection interval
                current_time = time.time()

                # If we've been idle too long, shut down the detector
                if current_time - self.last_client_request > self.max_idle_time:
                    logger.info(
                        f"Detector {self.detector_id} has been idle for too long, shutting down"
                    )
                    self.stop_event.set()
                    break

                # Only process frames at the specified interval
                if current_time - last_detection_time < self.detection_interval:
                    # Sleep for a short time to avoid busy waiting
                    time.sleep(0.1)
                    continue

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
                        time.sleep(min(30 * (self.stream_reconnect_attempts // self.max_stream_reconnect_attempts), 300))
                        logger.info(f"Resetting reconnection attempts counter and trying again")
                        self.stream_reconnect_attempts = 0
                    else:
                        # Calculate progressive backoff delay
                        current_delay = self.stream_reconnect_delay * (2 ** (self.stream_reconnect_attempts - 1))
                        current_delay = min(current_delay, 30)  # Cap at 30 seconds
                        
                        logger.info(f"Reconnection attempt {self.stream_reconnect_attempts}/{self.max_stream_reconnect_attempts} " +
                                    f"with {current_delay:.1f}s delay")
                        
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
                            logger.warning(f"Reconnection attempt {self.stream_reconnect_attempts} failed")

                    continue

                # Read frame from the video stream
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    logger.warning(
                        f"Failed to read frame from stream for {self.detector_id}"
                    )
                    self.connection_status = "error"
                    # Try to reconnect
                    if self._open_stream():
                        logger.info(
                            "Successfully reconnected to stream after frame read error"
                        )
                        self.connection_status = "connected"
                    # Skip this iteration
                    time.sleep(1)
                    continue

                # Update connection status
                self.connection_status = "connected"

                # Increment frame counter
                self.frame_count += 1

                # Skip frames according to frame_skip_rate
                frame_skip_counter += 1
                if frame_skip_counter % self.frame_skip_rate != 0:
                    continue

                # Resize frame for model input
                frame_resized = cv2.resize(frame, (self.input_width, self.input_height))

                # Only perform detection at the specified interval
                if current_time - last_detection_time >= self.detection_interval:
                    # Store the most recent frame
                    self.last_frame = frame.copy()

                    # Perform detection
                    results = self.model(frame_resized, conf=self.confidence_threshold)

                    # Process results
                    people_detected = False
                    pets_detected = False
                    people_count = 0
                    pet_count = 0

                    if len(results) > 0:
                        # Extract detections - YOLO v8 format
                        result = results[0]  # First batch result

                        # Get boxes and class information
                        boxes = result.boxes

                        for box in boxes:
                            cls_id = int(box.cls.item())
                            _ = box.conf.item()  # Confidence value, unused

                            # Only process supported classes
                            if cls_id in SUPPORTED_CLASSES:
                                if cls_id == 0:  # Person
                                    people_detected = True
                                    people_count += 1
                                elif cls_id in [15, 16, 17]:  # Bird, cat, dog
                                    pets_detected = True
                                    pet_count += 1

                    # Update detection state
                    self.people_detected = people_detected
                    self.pets_detected = pets_detected
                    self.people_count = people_count
                    self.pet_count = pet_count
                    self.last_detection_time = current_time
                    self.last_update_time = current_time

                    # Log detection
                    logger.debug(
                        f"Detection: people={people_detected}({people_count}), pets={pets_detected}({pet_count})"
                    )

                    # Reset error counters on successful detection
                    self.consecutive_errors = 0

                    last_detection_time = current_time

            except torch.cuda.OutOfMemoryError as cuda_err:
                logger.error(f"CUDA out of memory error: {cuda_err}")
                self.error_counts["cuda"] += 1
                self.consecutive_errors += 1
                time.sleep(self.error_recovery_delay)

            except MemoryError as mem_err:
                logger.error(f"Memory error: {mem_err}")
                self.error_counts["memory"] += 1
                self.consecutive_errors += 1
                time.sleep(self.error_recovery_delay)

            except cv2.error as cv_err:
                logger.error(f"OpenCV error: {cv_err}")
                self.error_counts["stream"] += 1
                self.consecutive_errors += 1
                # Try to reconnect to the stream
                self._open_stream()
                time.sleep(self.error_recovery_delay)

            except Exception as ex:
                logger.error(f"Error in detection loop: {ex}", exc_info=True)
                self.error_counts["other"] += 1
                self.consecutive_errors += 1
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
        }


def create_or_update_detector(
    detector_id: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create or update a detector with the given configuration."""
    with detector_lock:
        if detector_id in detectors:
            # Update existing detector
            detector = detectors[detector_id]
            logger.info(f"Updating existing detector {detector_id}")

            # Update detector configuration
            if config.get("stream_url"):
                detector.stream_url = config["stream_url"]
            if config.get("name"):
                detector.name = config["name"]
            if config.get("model"):
                # Model changes require reinitialization, which we'll skip here
                detector.model_name = config["model"]
            if config.get("input_size"):
                try:
                    width, height = map(int, config["input_size"].split("x"))
                    detector.input_width = width
                    detector.input_height = height
                except Exception:
                    pass
            if config.get("detection_interval"):
                detector.detection_interval = config["detection_interval"]
            if config.get("confidence_threshold"):
                detector.confidence_threshold = config["confidence_threshold"]
            if config.get("frame_skip_rate"):
                detector.frame_skip_rate = config["frame_skip_rate"]

            # Update last client request time to prevent idle shutdown
            detector.last_client_request = time.time()

            return {
                "status": "success",
                "message": "Detector updated",
                "detector_id": detector_id,
                "state": detector.get_state(),
            }
        else:
            # Create new detector
            logger.info(f"Creating new detector {detector_id}")
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

    def _send_response(self, data: Dict[str, Any], status_code: int = 200) -> None:
        """Send a JSON response."""
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _send_error_response(self, message: str, status_code: int = 400) -> None:
        """Send an error response."""
        self._send_response({"status": "error", "message": message}, status_code)

    def _parse_json_body(self) -> Dict[str, Any]:
        """Parse JSON request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}

        try:
            body = self.rfile.read(content_length).decode("utf-8")
            return json.loads(body)
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:
        """Handle GET requests."""
        # Parse URL and query parameters
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        # Get detector_id from query parameters
        detector_id = query.get("detector_id", [""])[0]

        if path == "/health":
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
                    self._send_error_response(f"Detector {detector_id} not found", 404)
        elif path == "/detectors":
            # List all detectors
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
                self._send_response({"status": "success", "detectors": detector_list})
        else:
            self._send_error_response("Invalid endpoint", 404)

    def do_POST(self) -> None:
        """Handle POST requests."""
        # Parse URL
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # Parse request body
        body = self._parse_json_body()

        if path == "/poll":
            # Poll endpoint - create or update detector and get state
            detector_id = body.get("detector_id")

            if not detector_id:
                self._send_error_response("detector_id is required")
                return

            # Get detector configuration from request
            config = body.get("config", {})

            # Create or update detector
            result = create_or_update_detector(detector_id, config)
            self._send_response(result)

        elif path == "/shutdown":
            # Shutdown detector
            detector_id = body.get("detector_id")

            if not detector_id:
                self._send_error_response("detector_id is required")
                return

            with detector_lock:
                if detector_id in detectors:
                    detector = detectors[detector_id]
                    detector.shutdown()
                    del detectors[detector_id]
                    self._send_response(
                        {
                            "status": "success",
                            "message": f"Detector {detector_id} shutdown",
                        }
                    )
                else:
                    self._send_error_response(f"Detector {detector_id} not found", 404)
        else:
            self._send_error_response("Invalid endpoint", 404)


def start_http_server() -> None:
    """Start the HTTP server."""
    try:
        server = HTTPServer(("0.0.0.0", HTTP_PORT), YoloHTTPHandler)
        logger.info(f"Starting HTTP server on port {HTTP_PORT}")

        # Print server info
        logger.info(f"Python version: {sys.version}")
        logger.info(f"OpenCV version: {cv2.__version__}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by keyboard interrupt")
    except Exception as ex:
        logger.error(f"Error starting server: {ex}", exc_info=True)
    finally:
        logger.info("Server shutting down")
        # Cleanup all detectors
        with detector_lock:
            for detector_id, detector in list(detectors.items()):
                try:
                    detector.shutdown()
                except Exception:
                    pass
            detectors.clear()


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