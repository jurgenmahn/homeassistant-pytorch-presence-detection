"""Constants for the YOLO Presence Detection integration."""

DOMAIN = "yolo_presence"
DATA_YOLO_PRESENCE = "yolo_presence_data"

# Config keys
CONF_STREAM_URL = "stream_url"
CONF_NAME = "name"
CONF_DETECTION_INTERVAL = "detection_interval"
CONF_CONFIDENCE_THRESHOLD = "confidence_threshold"
CONF_INPUT_SIZE = "input_size"
CONF_MODEL = "model"
CONF_FRAME_SKIP_RATE = "frame_skip_rate"
CONF_PROCESSING_SERVER = "processing_server"

# Entity attributes
ATTR_DEVICE_ID = "device_id"
ATTR_HUMANS_DETECTED = "humans_detected"
ATTR_PETS_DETECTED = "pets_detected"
ATTR_HUMAN_COUNT = "human_count"
ATTR_PET_COUNT = "pet_count"
ATTR_LAST_DETECTION = "last_detection"
ATTR_MODEL_TYPE = "model_type"
ATTR_CONNECTION_STATUS = "connection_status"

# Default values
DEFAULT_NAME = "YOLO Presence"
DEFAULT_DETECTION_INTERVAL_CPU = 10
DEFAULT_DETECTION_INTERVAL_GPU = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_INPUT_SIZE = "640x480"
DEFAULT_MODEL = "yolo11l"
DEFAULT_FRAME_SKIP_RATE_CPU = 5
DEFAULT_FRAME_SKIP_RATE_GPU = 3
DEFAULT_PROCESSING_SERVER = "http://localhost:5000"

# SCAN_INTERVAL for sensors updates (seconds)
SCAN_INTERVAL = 1

# Model options - friendly names
MODEL_OPTIONS = {
    "yolo11n": "YOLO11 Nano (Fastest, lowest accuracy)",
    "yolo11s": "YOLO11 Small (Fast, good accuracy)",
    "yolo11m": "YOLO11 Medium (Balanced speed/accuracy)",
    "yolo11l": "YOLO11 Large (Slower, high accuracy)",
    "yolo11x": "YOLO11 Extra Large (Slowest, highest accuracy)"
}

# Input size options
INPUT_SIZE_OPTIONS = ["320x240", "640x480", "1280x720"]

# Supported classes
# 0=person, 15=bird, 16=cat, 17=dog
SUPPORTED_CLASSES = [0, 15, 16, 17]

# Class mapping
CLASS_MAP = {
    0: "person",
    15: "bird",
    16: "cat", 
    17: "dog"
}

# Connection status
CONNECTION_STATUS_OPTIONS = {
    "connected": "Connected to camera and processing server",
    "server_unavailable": "Processing server unavailable",
    "server_disconnected": "Disconnected from processing server",
    "disconnected": "Camera disconnected",
}

# Event types
EVENT_HUMAN_DETECTED = f"{DOMAIN}_human_detected"
EVENT_PET_DETECTED = f"{DOMAIN}_pet_detected"
EVENT_HUMAN_COUNT_CHANGED = f"{DOMAIN}_human_count_changed"
EVENT_PET_COUNT_CHANGED = f"{DOMAIN}_pet_count_changed"
EVENT_CONNECTION_STATUS_CHANGED = f"{DOMAIN}_connection_status_changed"