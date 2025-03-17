"""YOLO Presence Detection processing engine."""
import asyncio
import datetime
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, Event, callback
from homeassistant.helpers.event import async_track_time_interval
import homeassistant.util.dt as dt_util

from .const import (
    DOMAIN,
    CONF_STREAM_URL,
    CONF_NAME,
    CONF_DETECTION_INTERVAL,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_INPUT_SIZE,
    CONF_MODEL,
    CONF_FRAME_SKIP_RATE,
    DEFAULT_DETECTION_INTERVAL_CPU,
    DEFAULT_DETECTION_INTERVAL_GPU,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_INPUT_SIZE,
    DEFAULT_MODEL,
    DEFAULT_FRAME_SKIP_RATE_CPU,
    DEFAULT_FRAME_SKIP_RATE_GPU,
    SUPPORTED_CLASSES,
    CLASS_MAP,
    EVENT_HUMAN_DETECTED,
    EVENT_PET_DETECTED,
    EVENT_HUMAN_COUNT_CHANGED,
    EVENT_PET_COUNT_CHANGED,
    EVENT_CONNECTION_STATUS_CHANGED,
)

_LOGGER = logging.getLogger(__name__)


class YoloPresenceDetector:
    """Class to manage YOLO-based presence detection."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry):
        """Initialize the detector."""
        self.hass = hass
        self.config_entry = config_entry
        self.entry_id = config_entry.entry_id
        
        # Get configuration
        config = {**config_entry.data, **config_entry.options}
        self.stream_url = config.get(CONF_STREAM_URL)
        self.name = config.get(CONF_NAME, "YOLO Presence")
        self.device_id = config_entry.unique_id or self.entry_id
        
        # YOLO model configuration
        self.model_name = config.get(CONF_MODEL, DEFAULT_MODEL)
        self.model_path = f"{self.model_name}.pt"
        self.model = None
        self.device = "cpu"  # Will be set during initialization
        
        # Parse input size
        input_size_str = config.get(CONF_INPUT_SIZE, DEFAULT_INPUT_SIZE)
        width, height = map(int, input_size_str.split("x"))
        self.input_width = width
        self.input_height = height
        
        # Detection parameters
        self.detection_interval = config.get(
            CONF_DETECTION_INTERVAL, 
            DEFAULT_DETECTION_INTERVAL_CPU
        )
        self.confidence_threshold = config.get(
            CONF_CONFIDENCE_THRESHOLD, 
            DEFAULT_CONFIDENCE_THRESHOLD
        )
        self.frame_skip_rate = config.get(
            CONF_FRAME_SKIP_RATE, 
            DEFAULT_FRAME_SKIP_RATE_CPU
        )
        
        # Runtime state
        self.is_running = False
        self.detection_thread = None
        self.stop_event = None
        self.cap = None
        self.frame_count = 0
        
        # Detection state
        self.last_detection_time = None
        self.people_detected = False
        self.pets_detected = False
        self.people_count = 0
        self.pet_count = 0
        self.connection_status = "disconnected"
        self.detection_data = {}
        
        # Callback handler
        self.update_callbacks = []
        self.remove_track_time_interval = None

    async def async_initialize(self) -> None:
        """Initialize the detector and start the detection process."""
        try:
            # Load the YOLO model on a separate thread to avoid blocking
            await self.async_load_model()
            
            # Start the detection process
            self.stop_event = asyncio.Event()
            self.detection_thread = asyncio.create_task(self.async_detection_loop())
            
            # Schedule regular checks
            self.remove_track_time_interval = async_track_time_interval(
                self.hass, self.async_update_state, datetime.timedelta(seconds=1)
            )
            
            self.is_running = True
            _LOGGER.info("YOLO Presence detector initialized for %s", self.name)
            
        except Exception as ex:
            _LOGGER.error("Failed to initialize YOLO Presence detector: %s", str(ex))
            raise

    async def async_load_model(self) -> None:
        """Load the YOLO model asynchronously."""
        def _load_model():
            # Determine model path - look in current directory and models directory
            local_path = self.model_path
            models_path = f"models/{self.model_path}"
            
            model_path = local_path if os.path.exists(local_path) else models_path
            
            # Check if CUDA is available
            has_cuda = torch.cuda.is_available()
            has_rocm = (
                torch.backends.hip.is_built() 
                if hasattr(torch.backends, "hip") else False
            )
            
            # Select device
            device = "cpu"
            if has_cuda:
                device = "cuda"
                _LOGGER.info("Using CUDA for YOLO inference")
            elif has_rocm:
                device = "cuda"  # ROCm uses CUDA API
                _LOGGER.info("Using ROCm for YOLO inference") 
            else:
                _LOGGER.info("Using CPU for YOLO inference")
            
            # Load the model
            model = YOLO(model_path)
            model.to(device)
            
            # Configure model parameters
            model.overrides['imgsz'] = max(self.input_width, self.input_height)
            
            return model, device
        
        # Load the model in a separate thread
        self.model, self.device = await self.hass.async_add_executor_job(_load_model)
        _LOGGER.info(
            "YOLO model %s loaded on %s with input size %dx%d", 
            self.model_name, self.device, self.input_width, self.input_height
        )

    async def async_detection_loop(self) -> None:
        """Main detection loop."""
        last_detection_time = 0
        frame_skip_count = 0
        
        _LOGGER.debug("Starting detection loop for %s", self.name)
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Open the stream if not already open
                    if self.cap is None or not self.cap.isOpened():
                        await self.async_open_stream()
                        if self.cap is None or not self.cap.isOpened():
                            self.connection_status = "disconnected"
                            await self.async_fire_connection_event("disconnected")
                            # Wait before trying again
                            await asyncio.sleep(5)
                            continue
                        else:
                            self.connection_status = "connected"
                            await self.async_fire_connection_event("connected")
                    
                    current_time = time.time()
                    
                    # Check if we should run detection based on interval
                    if current_time - last_detection_time < self.detection_interval:
                        # Keep the stream active by grabbing frames, but don't process them
                        ret = await self.hass.async_add_executor_job(self.cap.grab)
                        if not ret:
                            _LOGGER.warning("Failed to grab frame from stream")
                            await self.async_close_stream()
                            continue
                        
                        # Short sleep to avoid busy loop
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Frame skipping for efficiency
                    frame_skip_count += 1
                    if frame_skip_count < self.frame_skip_rate:
                        ret = await self.hass.async_add_executor_job(self.cap.grab)
                        if not ret:
                            _LOGGER.warning("Failed to grab frame from stream")
                            await self.async_close_stream()
                            continue
                        
                        # Short sleep
                        await asyncio.sleep(0.01)
                        continue
                    
                    # Reset frame skip counter
                    frame_skip_count = 0
                    
                    # Read a frame
                    ret, frame = await self.hass.async_add_executor_job(self.cap.read)
                    if not ret or frame is None:
                        _LOGGER.warning("Failed to read frame from stream")
                        await self.async_close_stream()
                        continue
                    
                    # Process the frame
                    await self.async_process_frame(frame)
                    
                    # Update last detection time
                    last_detection_time = time.time()
                    
                except Exception as ex:
                    _LOGGER.error("Error in detection loop: %s", str(ex))
                    await asyncio.sleep(5)  # Wait before retrying
                
        except asyncio.CancelledError:
            _LOGGER.debug("Detection loop cancelled for %s", self.name)
        finally:
            await self.async_close_stream()
            _LOGGER.debug("Detection loop ended for %s", self.name)

    async def async_open_stream(self) -> None:
        """Open the video stream."""
        def _open_stream():
            # Close any existing stream
            if self.cap is not None:
                self.cap.release()
            
            # Open the stream
            cap = cv2.VideoCapture(self.stream_url)
            
            # Configure stream parameters
            if cap.isOpened():
                # Try to set hardware acceleration
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                
                # Set H.264 codec for better performance
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                
                _LOGGER.debug(
                    "Stream opened. Hardware acceleration: %s", 
                    cap.get(cv2.CAP_PROP_HW_ACCELERATION)
                )
            
            return cap
        
        self.cap = await self.hass.async_add_executor_job(_open_stream)

    async def async_close_stream(self) -> None:
        """Close the video stream."""
        if self.cap is not None:
            await self.hass.async_add_executor_job(self.cap.release)
            self.cap = None

    async def async_process_frame(self, frame: np.ndarray) -> None:
        """Process a video frame and detect people and pets."""
        def _prepare_frame():
            # Resize the frame for processing
            resized = cv2.resize(frame, (self.input_width, self.input_height))
            
            # Convert to RGB (YOLO expects RGB, OpenCV uses BGR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            return rgb
        
        def _run_detection(processed_frame):
            # Run inference with torch optimizations
            with torch.no_grad():
                # Pass just the classes we're interested in to save processing time
                results = self.model(
                    processed_frame,
                    conf=self.confidence_threshold,
                    iou=0.45,  # NMS threshold
                    max_det=50,  # Maximum detections
                    classes=SUPPORTED_CLASSES,  # Only detect people and pets
                    agnostic_nms=True,
                    verbose=False
                )
                
                # Count detections by class
                people_count = 0
                pet_count = 0
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if cls_id == 0:  # Person
                            people_count += 1
                        elif cls_id in [16, 17]:  # Cat or dog
                            pet_count += 1
                
                return results, people_count, pet_count
        
        try:
            # Prepare the frame
            processed_frame = await self.hass.async_add_executor_job(_prepare_frame)
            
            # Run detection
            results, people_count, pet_count = await self.hass.async_add_executor_job(
                _run_detection, processed_frame
            )
            
            # Update state
            old_people_detected = self.people_detected
            old_pets_detected = self.pets_detected
            old_people_count = self.people_count
            old_pet_count = self.pet_count
            
            self.people_detected = people_count > 0
            self.pets_detected = pet_count > 0
            self.people_count = people_count
            self.pet_count = pet_count
            self.last_detection_time = dt_util.utcnow()
            
            # Fire events for state changes
            if self.people_detected != old_people_detected:
                await self.async_fire_human_event(self.people_detected)
            
            if self.pets_detected != old_pets_detected:
                await self.async_fire_pet_event(self.pets_detected)
            
            if self.people_count != old_people_count:
                await self.async_fire_human_count_event(self.people_count)
            
            if self.pet_count != old_pet_count:
                await self.async_fire_pet_count_event(self.pet_count)
            
            # If anything changed, call update callbacks
            if (old_people_detected != self.people_detected or 
                old_pets_detected != self.pets_detected or
                old_people_count != self.people_count or
                old_pet_count != self.pet_count):
                self._call_update_callbacks()
            
            # Log detection results at debug level
            if self.people_count > 0 or self.pet_count > 0:
                _LOGGER.debug(
                    "%s: Detected %d people and %d pets", 
                    self.name, self.people_count, self.pet_count
                )
            
        except Exception as ex:
            _LOGGER.error("Error processing frame: %s", str(ex))

    async def async_update_state(self, now=None):
        """Update sensor state and check connection periodically."""
        # Check if the detector is still running
        if not self.is_running:
            return
        
        # Check if the connection status has changed
        is_connected = self.cap is not None and self.cap.isOpened()
        new_status = "connected" if is_connected else "disconnected"
        
        if new_status != self.connection_status:
            self.connection_status = new_status
            await self.async_fire_connection_event(new_status)
            self._call_update_callbacks()
            
        # Force refresh entities every 60 seconds even if no changes
        self._call_update_callbacks()

    async def async_shutdown(self) -> None:
        """Shutdown the detector."""
        _LOGGER.debug("Shutting down YOLO Presence detector for %s", self.name)
        
        self.is_running = False
        
        # Cancel the time interval tracker
        if self.remove_track_time_interval is not None:
            self.remove_track_time_interval()
            self.remove_track_time_interval = None
        
        # Stop the detection thread
        if self.stop_event is not None:
            self.stop_event.set()
        
        if self.detection_thread is not None:
            try:
                self.detection_thread.cancel()
                await self.detection_thread
            except (asyncio.CancelledError, Exception) as ex:
                _LOGGER.debug("Error cancelling detection thread: %s", str(ex))
            
            self.detection_thread = None
        
        # Close the stream
        await self.async_close_stream()
        
        # Clean up resources
        if self.model is not None:
            self.model = None
            
        # Clear CUDA cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        _LOGGER.debug("YOLO Presence detector shutdown complete for %s", self.name)

    def register_update_callback(self, callback_func):
        """Register a callback for state updates."""
        self.update_callbacks.append(callback_func)
        return lambda: self.update_callbacks.remove(callback_func)

    def _call_update_callbacks(self):
        """Call all registered callbacks."""
        for callback_func in self.update_callbacks:
            callback_func()

    async def async_fire_human_event(self, is_detected: bool) -> None:
        """Fire a Home Assistant event when human detection state changes."""
        data = {
            "device_id": self.device_id,
            "name": self.name,
            "detected": is_detected,
        }
        self.hass.bus.async_fire(EVENT_HUMAN_DETECTED, data)
        _LOGGER.debug("%s: Human detected state changed to %s", self.name, is_detected)

    async def async_fire_pet_event(self, is_detected: bool) -> None:
        """Fire a Home Assistant event when pet detection state changes."""
        data = {
            "device_id": self.device_id,
            "name": self.name,
            "detected": is_detected,
        }
        self.hass.bus.async_fire(EVENT_PET_DETECTED, data)
        _LOGGER.debug("%s: Pet detected state changed to %s", self.name, is_detected)

    async def async_fire_human_count_event(self, count: int) -> None:
        """Fire a Home Assistant event when human count changes."""
        data = {
            "device_id": self.device_id,
            "name": self.name,
            "count": count,
        }
        self.hass.bus.async_fire(EVENT_HUMAN_COUNT_CHANGED, data)
        _LOGGER.debug("%s: Human count changed to %d", self.name, count)

    async def async_fire_pet_count_event(self, count: int) -> None:
        """Fire a Home Assistant event when pet count changes."""
        data = {
            "device_id": self.device_id,
            "name": self.name,
            "count": count,
        }
        self.hass.bus.async_fire(EVENT_PET_COUNT_CHANGED, data)
        _LOGGER.debug("%s: Pet count changed to %d", self.name, count)

    async def async_fire_connection_event(self, status: str) -> None:
        """Fire a Home Assistant event when connection status changes."""
        data = {
            "device_id": self.device_id,
            "name": self.name,
            "status": status,
        }
        self.hass.bus.async_fire(EVENT_CONNECTION_STATUS_CHANGED, data)
        _LOGGER.info("%s: Connection status changed to %s", self.name, status)