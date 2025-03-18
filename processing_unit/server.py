#!/usr/bin/env python3
"""YOLO Presence Detection Server

This standalone server handles video processing and object detection
using YOLO models, designed to work with Home Assistant's yolo_presence
integration through a simple API.
"""
import asyncio
import json
import logging
import os
import signal
import threading
import time
from typing import Dict, List, Optional, Any, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import queue
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("yolo_presence_server")

# Flask app setup
app = Flask(__name__)
CORS(app)

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

    def initialize(self) -> bool:
        """Initialize the detector and start the detection process."""
        try:
            # Load the YOLO model
            self._load_model()
            
            # Start the detection process
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            self.is_running = True
            logger.info(f"YOLO Detector initialized for {self.name}")
            return True
            
        except Exception as ex:
            logger.error(f"Failed to initialize YOLO detector: {str(ex)}")
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
                logger.info(f"Found YOLO model at {model_path}")
                break
        
        # If model not found locally, use pre-trained from ultralytics
        if not model_path:
            logger.warning("Model not found locally, using pre-trained model")
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
            logger.warning(f"Error checking CUDA availability: {str(ex)}")
            has_cuda = False
            has_rocm = False
        
        # Select device
        device = "cpu"
        if has_cuda:
            device = "cuda"
            logger.info("Using CUDA for YOLO inference")
        elif has_rocm:
            device = "cuda"  # ROCm uses CUDA API
            logger.info("Using ROCm for YOLO inference") 
        else:
            logger.info("Using CPU for YOLO inference")
        
        # Load the model with error handling
        try:
            # First try with specified device
            model = YOLO(model_path)
            model.to(device)
        except Exception as ex:
            logger.warning(f"Error loading model on {device}: {str(ex)}, falling back to CPU")
            # Fall back to CPU if device-specific loading fails
            device = "cpu"
            model = YOLO(model_path)
            model.to(device)
        
        # Try to configure model parameters safely
        try:
            model.overrides['imgsz'] = max(self.input_width, self.input_height)
        except Exception as ex:
            logger.warning(f"Could not set model image size: {str(ex)}")
        
        self.model = model
        self.device = device
        logger.info(
            f"YOLO model {self.model_name} loaded on {self.device} with input size {self.input_width}x{self.input_height}"
        )

    def _open_stream(self) -> bool:
        """Open the video stream."""
        try:
            # Close any existing stream
            if self.cap is not None:
                self.cap.release()
            
            # Open the stream
            cap = cv2.VideoCapture(self.stream_url)
            
            # Configure stream parameters
            if cap.isOpened():
                # Try to set hardware acceleration
                try:
                    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                except:
                    pass
                
                # Set H.264 codec for better performance
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                except:
                    pass
                
                logger.debug(f"Stream opened for {self.name}")
                self.cap = cap
                return True
            else:
                logger.warning(f"Failed to open stream for {self.name}")
                return False
        except Exception as ex:
            logger.error(f"Error opening stream: {str(ex)}")
            return False

    def _close_stream(self) -> None:
        """Close the video stream."""
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

    def _detection_loop(self) -> None:
        """Main detection loop."""
        last_detection_time = 0
        
        logger.debug(f"Starting detection loop for {self.name}")
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Open the stream if not already open
                    if self.cap is None or not self.cap.isOpened():
                        if not self._open_stream():
                            self.connection_status = "disconnected"
                            # Wait before trying again
                            time.sleep(5)
                            continue
                        else:
                            self.connection_status = "connected"
                            self.last_update_time = time.time()
                    
                    current_time = time.time()
                    
                    # Keep the stream active by grabbing frames, but don't process them
                    ret = self.cap.grab()
                    if not ret:
                        logger.warning(f"Failed to grab frame from stream for {self.name}")
                        self._close_stream()
                        continue

                    # Check if we should run detection based on interval
                    if current_time - last_detection_time < self.detection_interval:
                        # Short sleep to avoid busy loop
                        time.sleep(0.1)
                        continue
                    
                    # Read a frame
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        logger.warning(f"Failed to read frame from stream for {self.name}")
                        self._close_stream()
                        continue
                    
                    # Process the frame
                    self._process_frame(frame)
                    
                    # Update last detection time
                    last_detection_time = time.time()
                    
                except Exception as ex:
                    logger.error(f"Error in detection loop for {self.name}: {str(ex)}")
                    time.sleep(5)  # Wait before retrying
                
        except Exception as ex:
            logger.error(f"Detection loop error for {self.name}: {str(ex)}")
        finally:
            self._close_stream()
            logger.debug(f"Detection loop ended for {self.name}")

    def _process_frame(self, frame: np.ndarray) -> None:
        """Process a video frame and detect people and pets."""
        try:
            # Resize the frame for processing
            resized = cv2.resize(frame, (self.input_width, self.input_height))
            
            # Convert to RGB (YOLO expects RGB, OpenCV uses BGR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Save a small version of the frame for debugging
            self.last_frame = cv2.resize(frame, (320, 240))
            
            # Run inference with torch optimizations
            with torch.no_grad():
                try:
                    # Run detection
                    results = self.model(
                        rgb,
                        conf=self.confidence_threshold,
                        iou=0.45,  # NMS threshold
                        max_det=50,  # Maximum detections
                        classes=SUPPORTED_CLASSES,  # Only detect people and pets
                        agnostic_nms=True,
                        verbose=False
                    )
                except Exception as ex:
                    logger.warning(f"Error with detection: {str(ex)}. Trying simplified approach.")
                    # Fall back to a more basic approach with fewer parameters
                    results = self.model(
                        rgb,
                        conf=self.confidence_threshold,
                        verbose=False
                    )
                
                # Count detections by class
                people_count = 0
                pet_count = 0
                
                if len(results) > 0:
                    # Handle different result formats
                    if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                        # Standard format
                        for box in results[0].boxes:
                            try:
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                
                                if cls_id == 0:  # Person
                                    people_count += 1
                                elif cls_id in [16, 17]:  # Cat or dog
                                    pet_count += 1
                            except Exception as ex:
                                logger.warning(f"Error processing box: {str(ex)}")
                                continue
                    else:
                        # Alternative format for older ultralytics versions
                        for det in results[0]:
                            try:
                                if len(det) >= 6:  # x1, y1, x2, y2, conf, cls
                                    cls_id = int(det[5])
                                    conf = float(det[4])
                                    
                                    if cls_id == 0:  # Person
                                        people_count += 1
                                    elif cls_id in [16, 17]:  # Cat or dog
                                        pet_count += 1
                            except Exception as ex:
                                logger.warning(f"Error processing detection: {str(ex)}")
                                continue
            
            # Update state
            old_people_detected = self.people_detected
            old_pets_detected = self.pets_detected
            old_people_count = self.people_count
            old_pet_count = self.pet_count
            
            self.people_detected = people_count > 0
            self.pets_detected = pet_count > 0
            self.people_count = people_count
            self.pet_count = pet_count
            self.last_detection_time = time.time()
            
            # If anything changed, update the last_update_time and notify clients
            if (old_people_detected != self.people_detected or 
                old_pets_detected != self.pets_detected or
                old_people_count != self.people_count or
                old_pet_count != self.pet_count):
                self.last_update_time = time.time()
                
                # Notify SSE clients of the state change
                self._notify_sse_clients()
                
                # Log detection results
                if self.people_count > 0 or self.pet_count > 0:
                    logger.debug(
                        f"{self.name}: Detected {self.people_count} people and {self.pet_count} pets"
                    )
            
        except Exception as ex:
            logger.error(f"Error processing frame for {self.name}: {str(ex)}")

    def shutdown(self) -> None:
        """Shutdown the detector."""
        logger.debug(f"Shutting down YOLO detector for {self.name}")
        
        # Prevent further processing
        self.is_running = False
        
        # Set the stop event to signal thread termination
        self.stop_event.set()
        
        # Wait for detection thread to finish
        if self.detection_thread and self.detection_thread.is_alive():
            try:
                # Give the thread a chance to exit gracefully
                self.detection_thread.join(timeout=2.0)
            except Exception as ex:
                logger.debug(f"Error waiting for detection thread: {str(ex)}")
        
        # Close the video stream
        self._close_stream()
        
        # Clean up PyTorch resources
        try:
            # Release model
            if self.model is not None:
                self.model = None
                
            # Clear CUDA cache if applicable
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as ex:
                    logger.debug(f"Error clearing CUDA cache: {str(ex)}")
        except Exception as ex:
            logger.debug(f"Error cleaning up resources: {str(ex)}")
        
        # Force Python garbage collection
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        
        # Notify all SSE clients that this detector is shutting down
        with sse_clients_lock:
            if self.detector_id in sse_clients:
                shutdown_message = f"event: detector_shutdown\ndata: {json.dumps({'detector_id': self.detector_id})}\n\n"
                for client_queue in sse_clients[self.detector_id]:
                    try:
                        client_queue.put_nowait(shutdown_message)
                    except queue.Full:
                        pass  # Ignore if queue is full
            
        logger.debug(f"YOLO detector shutdown complete for {self.name}")

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the detector."""
        return {
            "detector_id": self.detector_id,
            "name": self.name,
            "model": self.model_name,
            "device": self.device,
            "connection_status": self.connection_status,
            "is_running": self.is_running,
            "people_detected": self.people_detected,
            "pets_detected": self.pets_detected,
            "people_count": self.people_count,
            "pet_count": self.pet_count,
            "last_detection_time": self.last_detection_time,
            "last_update_time": self.last_update_time,
        }
        
    def _notify_sse_clients(self):
        """Notify SSE clients of state changes."""
        with sse_clients_lock:
            if self.detector_id in sse_clients:
                # Create state message
                state = self.get_state()
                message = f"data: {json.dumps(state)}\n\n"
                
                # Send to all clients
                dead_clients = []
                for client_queue in sse_clients[self.detector_id]:
                    try:
                        client_queue.put_nowait(message)
                    except queue.Full:
                        # Client queue is full and not being read, mark for removal
                        dead_clients.append(client_queue)
                
                # Remove dead clients
                for dead_client in dead_clients:
                    if dead_client in sse_clients[self.detector_id]:
                        sse_clients[self.detector_id].remove(dead_client)


# API Routes
@app.route('/api/status', methods=['GET'])
def get_status():
    """Get server status."""
    return jsonify({
        "status": "running",
        "version": "1.0.0",
        "detectors_count": len(detectors),
        "cuda_available": torch.cuda.is_available() if torch is not None else False,
    })

@app.route('/api/detectors', methods=['GET'])
def get_detectors():
    """Get all registered detectors."""
    with detector_lock:
        detector_list = []
        for detector_id, detector in detectors.items():
            detector_list.append({
                "detector_id": detector_id,
                "name": detector.name,
                "model": detector.model_name,
                "connection_status": detector.connection_status,
                "is_running": detector.is_running,
            })
    
    return jsonify(detector_list)

@app.route('/api/detectors', methods=['POST'])
def create_detector():
    """Create a new detector."""
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    required_fields = ["detector_id", "stream_url"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    detector_id = data["detector_id"]
    
    with detector_lock:
        # Check if detector already exists
        if detector_id in detectors:
            return jsonify({"error": f"Detector with ID {detector_id} already exists"}), 409
        
        # Create new detector
        detector = YoloDetector(detector_id, data)
        
        # Start the detector
        if detector.initialize():
            detectors[detector_id] = detector
            return jsonify({"success": True, "detector_id": detector_id}), 201
        else:
            return jsonify({"error": "Failed to initialize detector"}), 500

@app.route('/api/detectors/<detector_id>', methods=['GET'])
def get_detector(detector_id):
    """Get detector status."""
    with detector_lock:
        if detector_id not in detectors:
            return jsonify({"error": f"Detector {detector_id} not found"}), 404
        
        detector = detectors[detector_id]
        return jsonify(detector.get_state())

@app.route('/api/detectors/<detector_id>', methods=['DELETE'])
def delete_detector(detector_id):
    """Delete a detector."""
    with detector_lock:
        if detector_id not in detectors:
            return jsonify({"error": f"Detector {detector_id} not found"}), 404
        
        # Shutdown the detector
        detector = detectors[detector_id]
        detector.shutdown()
        
        # Remove from detectors
        del detectors[detector_id]
        
        return jsonify({"success": True})

@app.route('/api/detectors/<detector_id>/frame', methods=['GET'])
def get_detector_frame(detector_id):
    """Get the latest frame from the detector (for debugging)."""
    with detector_lock:
        if detector_id not in detectors:
            return jsonify({"error": f"Detector {detector_id} not found"}), 404
        
        detector = detectors[detector_id]
        if detector.last_frame is None:
            return jsonify({"error": "No frame available"}), 404
        
        # Convert the frame to JPEG
        try:
            _, buffer = cv2.imencode('.jpg', detector.last_frame)
            frame_bytes = buffer.tobytes()
            
            # Return as image
            return frame_bytes, 200, {'Content-Type': 'image/jpeg'}
        except Exception as ex:
            return jsonify({"error": f"Error encoding frame: {str(ex)}"}), 500

@app.route('/api/detectors/<detector_id>/events', methods=['GET'])
def detector_events(detector_id):
    """SSE endpoint for real-time detector state updates."""
    with detector_lock:
        if detector_id not in detectors:
            return jsonify({"error": f"Detector {detector_id} not found"}), 404
        
        detector = detectors[detector_id]
    
    def event_stream():
        """Generate SSE event stream."""
        # Create queue for this client
        client_queue = queue.Queue(maxsize=10)
        
        # Register this client
        with sse_clients_lock:
            if detector_id not in sse_clients:
                sse_clients[detector_id] = []
            sse_clients[detector_id].append(client_queue)
        
        # Send initial state
        state = detector.get_state()
        yield f"data: {json.dumps(state)}\n\n"
        
        try:
            while True:
                try:
                    # Wait for messages, timeout after 30 seconds to send keepalive
                    message = client_queue.get(timeout=30)
                    yield message
                except queue.Empty:
                    # Send keepalive message
                    yield ": keepalive\n\n"
        finally:
            # Remove client when connection is closed
            with sse_clients_lock:
                if detector_id in sse_clients and client_queue in sse_clients[detector_id]:
                    sse_clients[detector_id].remove(client_queue)
                    # Clean up empty detector list
                    if not sse_clients[detector_id]:
                        del sse_clients[detector_id]
    
    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable proxy buffering
            "Connection": "keep-alive"
        }
    )


# Graceful shutdown handler
def graceful_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Received shutdown signal, cleaning up...")
    
    # Shutdown all detectors
    with detector_lock:
        for detector_id, detector in list(detectors.items()):
            try:
                detector.shutdown()
            except Exception as ex:
                logger.error(f"Error shutting down detector {detector_id}: {str(ex)}")
    
    # Notify SSE clients that server is shutting down
    with sse_clients_lock:
        for detector_id, client_queues in list(sse_clients.items()):
            shutdown_message = f"event: shutdown\ndata: {json.dumps({'detector_id': detector_id})}\n\n"
            for client_queue in client_queues:
                try:
                    client_queue.put_nowait(shutdown_message)
                except queue.Full:
                    pass  # Ignore if queue is full
    
    # Exit
    os._exit(0)


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    
    # Start the server
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting YOLO Presence Detection Server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)