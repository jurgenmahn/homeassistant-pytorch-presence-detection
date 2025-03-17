# YOLO Presence Detection for Home Assistant

This integration uses YOLOv11 AI models to detect people and pets in camera feeds, providing presence detection for your home.

## Features

- **Person Detection**: Detects when people are present in the camera feed
- **Pet Detection**: Detects cats and dogs in the camera feed
- **Counting**: Counts the number of people and pets detected
- **GPU Acceleration**: Automatically uses GPU acceleration (CUDA/ROCm) when available
- **Adjustable Parameters**: Configure detection thresholds, intervals, and resolution
- **Multiple Models**: Choose from different YOLO models based on your hardware capabilities

## Installation

### HACS Installation (Recommended)

1. Ensure [HACS](https://hacs.xyz/) is installed
2. Add this repository as a custom repository in HACS:
   - Go to HACS → Integrations → ⋮ → Custom repositories
   - Add URL: `https://github.com/jurgen/yolo-presence-homeassistant`
   - Category: Integration
3. Click "Install" on the YOLO Presence Detection integration
4. Restart Home Assistant

### Manual Installation

1. Copy the `custom_components/yolo_presence` directory to your Home Assistant's `custom_components` directory
2. Restart Home Assistant

## Configuration

The integration can be configured through the Home Assistant UI:

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for "YOLO Presence" and select it
3. Enter the RTSP stream URL of your camera
4. Configure optional parameters like detection interval, model, etc.

## Available Entities

Each camera instance will create the following entities:

### Binary Sensors
- **Person Detected**: ON when a person is detected
- **Pet Detected**: ON when a pet (cat/dog) is detected

### Sensors
- **People Count**: Number of people detected
- **Pet Count**: Number of pets detected
- **Last Detection**: Timestamp of the last detection
- **Model Type**: The YOLO model being used
- **Connection Status**: Status of the camera connection

## Events

The integration fires the following events:

- `yolo_presence_human_detected`: When human detection state changes
- `yolo_presence_pet_detected`: When pet detection state changes
- `yolo_presence_human_count_changed`: When human count changes
- `yolo_presence_pet_count_changed`: When pet count changes
- `yolo_presence_connection_status_changed`: When connection status changes

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| Name | Name of the camera instance | YOLO Presence |
| Stream URL | RTSP URL of the camera | Required |
| Model | YOLO model to use | YOLOv11L |
| Detection Interval | Seconds between detections | 5s (GPU), 10s (CPU) |
| Confidence Threshold | Minimum detection confidence (0.1-0.9) | 0.25 |
| Input Resolution | Resolution for processing | 640x480 |
| Frame Skip Rate | Process 1 out of X frames | 3 (GPU), 5 (CPU) |

## Hardware Requirements

- For CPU mode: Any modern CPU (at least 2 cores recommended)
- For GPU acceleration: NVIDIA GPU with CUDA support or AMD GPU with ROCm support
- At least 4GB of RAM (8GB+ recommended for larger models)

## Models

- **YOLO11 Nano**: Fastest, lowest power usage, less accurate
- **YOLO11 Small**: Good balance for lower-end hardware
- **YOLO11 Medium**: Balanced performance and accuracy
- **YOLO11 Large**: Good accuracy, moderate performance requirements
- **YOLO11 Extra Large**: Best accuracy, highest resource usage

## Troubleshooting

- If the connection fails, check your RTSP URL format and network connectivity
- If detection is inaccurate, try increasing the confidence threshold or using a larger model
- If performance is poor, try a smaller model or increasing the detection interval

## License

This project is licensed under the MIT License.

---

Created by Jurgen Mahn, based on Ultralytics YOLO.