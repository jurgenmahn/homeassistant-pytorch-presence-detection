#!/usr/bin/env python3
"""
RTSP Stream Processor - Dedicated module for reading RTSP streams
and storing frames for processing by YoloDetector
"""
import cv2
import time
import logging
import threading
from datetime import datetime

# We'll use the logger from the main application
logger = logging.getLogger("yolo_server.rtsp_processor")

class RTSPStreamProcessor:
    def __init__(self, rtsp_url, process_nth_frame=30, reconnect_delay=5):
        """
        Initialize the RTSP stream processor with CUDA acceleration using OpenCV.
        
        Args:
            rtsp_url (str): RTSP URL to connect to
            process_nth_frame (int): Process every Nth frame (skip others for efficiency)
            reconnect_delay (int): Seconds to wait before reconnection attempts
        """
        self.rtsp_url = rtsp_url
        self.process_nth_frame = process_nth_frame
        self.reconnect_delay = reconnect_delay
        self.frame_count = 0
        self.last_frame = None
        self.running = False
        self.stop_event = threading.Event()
        self.reconnect_count = 0
        self.max_reconnect_attempts = 0  # 0 = unlimited reconnection attempts
        self.capture = None
        self.process_thread = None
        
    def _create_capture(self):
        """Create OpenCV VideoCapture with CUDA hardware acceleration"""
        # Construct OpenCV backend string with CUDA acceleration
        # CUDA GStreamer pipeline or OpenCV's built-in CUDA support
        
        # Option 1: Using OpenCV's built-in CUDA support with VideoCapture
        capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Enable hardware acceleration if available
        capture.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        
        # Option 2: Alternative explicit CUDA GStreamer pipeline through OpenCV
        if not capture.isOpened():
            logger.info("Trying alternative CUDA pipeline")
            gst_pipeline = (
                f'rtspsrc location={self.rtsp_url} latency=0 ! '
                'rtph264depay ! h264parse ! '
                'nvdec ! '  # NVIDIA hardware decoder
                'videoconvert ! appsink'
            )
            capture = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        # Check if capture is successfully opened
        if not capture.isOpened():
            logger.error("Failed to open RTSP stream with CUDA acceleration")
            logger.info("Falling back to standard RTSP capture")
            # Fallback to standard capture without acceleration
            capture = cv2.VideoCapture(self.rtsp_url)
            
        if not capture.isOpened():
            logger.error("Failed to open RTSP stream with any method")
            return None
            
        # Set buffer size to minimize latency
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        logger.info(f"Successfully opened RTSP stream: {self.rtsp_url}")
        logger.info(f"Stream parameters: Width={capture.get(cv2.CAP_PROP_FRAME_WIDTH)}, "
                    f"Height={capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}, "
                    f"FPS={capture.get(cv2.CAP_PROP_FPS)}")
        
        return capture
        
    def _process_stream(self):
        """Process frames from the stream"""
        consecutive_failures = 0
        max_consecutive_failures = 10  # Maximum failures before reconnection
        
        logger.info("Stream processing started")
        
        while not self.stop_event.is_set():
            if self.capture is None or not self.capture.isOpened():
                logger.warning("Capture is not valid, attempting to reconnect")
                self._reconnect()
                continue
                
            try:
                # Read a frame from the capture
                ret, frame = self.capture.read()
                
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame: attempt {consecutive_failures}/{max_consecutive_failures}")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive failures, reconnecting")
                        self._reconnect()
                        consecutive_failures = 0
                    else:
                        # Short sleep before next attempt
                        time.sleep(0.1)
                    continue
                
                # Reset failures counter on successful frame read
                consecutive_failures = 0
                
                # Increment frame counter
                self.frame_count += 1
                
                # Store current frame
                self.last_frame = frame.copy()
                
                # Process only every Nth frame for efficiency
                if self.frame_count % self.process_nth_frame == 0:
                    self._process_frame(frame)
                    
                # Release this frame to free memory
                # frame = None
                
                # Sleep a tiny amount to prevent CPU overload
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error processing stream: {e}")
                consecutive_failures += 1
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Too many consecutive failures, reconnecting")
                    self._reconnect()
                    consecutive_failures = 0
                    
        logger.info("Stream processing stopped")
                
    def _process_frame(self, frame):
        """
        Process frame - in our case, we just store it
        This method is intentionally minimal, as requested
        """
        # We don't need to do any processing here - just store the frame
        # The main YoloDetector class will handle the actual processing
        
        # Just log that we received a frame at debug level
        logger.debug(f"Received frame {self.frame_count}")
        
    def _reconnect(self):
        """Attempt to reconnect to the RTSP stream after a delay"""
        self.reconnect_count += 1
        
        # Check if maximum reconnection attempts reached
        if self.max_reconnect_attempts > 0 and self.reconnect_count > self.max_reconnect_attempts:
            logger.error(f"Exceeded maximum reconnection attempts ({self.max_reconnect_attempts})")
            self.stop()
            return
        
        logger.info(f"Reconnection attempt {self.reconnect_count}")
        
        # Close existing capture if any
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        
        # Wait before reconnecting
        logger.info(f"Waiting {self.reconnect_delay} seconds before reconnecting")
        for i in range(self.reconnect_delay):
            if self.stop_event.is_set():
                return
            time.sleep(1)
        
        # Create new capture
        logger.info("Creating new capture")
        self.capture = self._create_capture()
        
        if self.capture is None or not self.capture.isOpened():
            logger.error("Reconnection failed")
            # Schedule another reconnection
            threading.Timer(1.0, self._reconnect).start()
        else:
            logger.info("Reconnection successful")
            
    def start(self):
        """Start processing the RTSP stream"""
        if self.running:
            logger.warning("Stream processor is already running")
            return False
            
        logger.info(f"Starting stream processor for {self.rtsp_url}")
        self.running = True
        self.stop_event.clear()
        
        # Create capture
        self.capture = self._create_capture()
        if self.capture is None:
            logger.error("Failed to create capture")
            self.running = False
            return False
            
        # Start processing in a separate thread
        self.process_thread = threading.Thread(target=self._process_stream)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        logger.info("Stream processor started successfully")
        return True
        
    def stop(self):
        """Stop processing the RTSP stream"""
        if not self.running:
            logger.warning("Stream processor is not running")
            return
            
        logger.info("Stopping stream processor")
        self.stop_event.set()
        self.running = False
        
        # Wait for the process thread to finish
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2)
            
        # Release capture
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            
        logger.info("Stream processor stopped")
        
    def is_running(self):
        """Check if the stream processor is running"""
        return self.running
        
    def get_latest_frame(self):
        """Get the latest processed frame"""
        return self.last_frame


# No example usage when imported as a module