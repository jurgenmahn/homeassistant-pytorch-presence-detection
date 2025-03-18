# Home Assistant YOLO Presence Detection Integration

This integration provides presence detection using YOLO object detection on camera streams. It uses a separate processing server to handle the video processing and ML inference, while providing seamless integration with Home Assistant entities and automations.

## Major Update: New Architecture

This is a major update to the YOLO Presence Detection integration, introducing a new split architecture:

1. **Home Assistant Integration**: Handles the UI, configuration, entities, and events within Home Assistant
2. **Processing Server**: A separate Docker container that handles video stream processing and ML inference

This architecture provides several benefits:
- **Python Version Compatibility**: Home Assistant requires Python 3.13, but PyTorch and Ultralytics don't fully support Python 3.13 yet
- **Resource Isolation**: Processing can happen on a separate machine with a GPU
- **Reduced Dependencies**: The HA component has minimal dependencies
- **Scalability**: Multiple HA instances can connect to a single processing server, or one server can handle multiple cameras

## Features

- **Person Detection**: Detects when people are present in the camera feed
- **Pet Detection**: Detects cats and dogs in the camera feed
- **Counting**: Counts the number of people and pets detected
- **GPU Acceleration**: Automatically uses GPU acceleration (CUDA/ROCm) when available
- **Adjustable Parameters**: Configure detection thresholds, intervals, and resolution
- **Multiple Models**: Choose from different YOLO models based on your hardware capabilities

## Installation

### Step 1: Set up the Processing Server

1. Clone the repository to your machine
2. Navigate to the processing_unit directory
3. Start the Docker container:

```bash
cd processing_unit
docker-compose up -d
```

For more details, see the [processing server README](../../processing_unit/README.md).

### Step 2: Install the Home Assistant Integration

#### HACS Installation (Recommended)

1. Ensure [HACS](https://hacs.xyz/) is installed
2. Add this repository as a custom repository in HACS:
   - Go to HACS → Integrations → ⋮ → Custom repositories
   - Add URL: `https://github.com/jurgen/yolo-presence-homeassistant`
   - Category: Integration
3. Click "Install" on the YOLO Presence Detection integration
4. Restart Home Assistant

#### Manual Installation

1. Copy the `custom_components/yolo_presence` directory to your Home Assistant's `custom_components` directory
2. Restart Home Assistant

## Configuration

The integration can be configured through the Home Assistant UI:

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for "YOLO Presence" and select it
3. Enter the processing server URL (e.g., http://192.168.1.100:5000)
4. Enter the RTSP stream URL of your camera
5. Configure optional parameters like detection interval, model, etc.

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
- **Connection Status**: Status of the processing server and camera connection

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
| Processing Server | URL of the YOLO processing server | http://localhost:5000 |
| Stream URL | RTSP URL of the camera | Required |
| Model | YOLO model to use | YOLOv11L |
| Detection Interval | Seconds between detections | 5s (GPU), 10s (CPU) |
| Confidence Threshold | Minimum detection confidence (0.1-0.9) | 0.25 |
| Input Resolution | Resolution for processing | 640x480 |
| Frame Skip Rate | Process 1 out of X frames | 3 (GPU), 5 (CPU) |

## Hardware Requirements

### For the Processing Server:
- Any machine capable of running Docker
- For GPU acceleration: NVIDIA GPU with CUDA support or AMD GPU with ROCm support
- At least 4GB of RAM (8GB+ recommended for larger models)

### For Home Assistant:
- Regular Home Assistant requirements (much lighter than before as processing happens externally)

## Models

- **YOLO11 Nano**: Fastest, lowest power usage, less accurate
- **YOLO11 Small**: Good balance for lower-end hardware
- **YOLO11 Medium**: Balanced performance and accuracy
- **YOLO11 Large**: Good accuracy, moderate performance requirements
- **YOLO11 Extra Large**: Best accuracy, highest resource usage

## Troubleshooting

- If you experience connection issues, verify that your processing server is running and accessible from Home Assistant
- Check that your camera stream URL is correct and accessible from the processing server
- For CPU resource issues, try using a smaller model (nano or small) and increasing the detection interval
- See the logs for more detailed error messages

## License

This project is licensed under the MIT License.

---

Created by Jurgen Mahn with assistance from Claude Code, based on Ultralytics YOLO.