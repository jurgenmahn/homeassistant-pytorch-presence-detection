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
3. Enter the processing server URL (e.g., http://192.168.1.100:5505)
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
| Processing Server | URL of the YOLO processing server | http://localhost:5505 |
| Stream URL | RTSP URL of the camera | Required |
| Auto-Optimization | Enable automatic resource optimization | Disabled |
| Model | YOLO model to use | YOLOv11L |
| Detection Interval | Seconds between detections | 5s (GPU), 10s (CPU) |
| Confidence Threshold | Minimum detection confidence (0.1-0.9) | 0.25 |
| Input Resolution | Resolution for processing | 640x480 |
| Frame Skip Rate | Process 1 out of X frames | 3 (GPU), 5 (CPU) |

### Auto-Optimization

The integration includes an automatic resource optimization feature that:

- Automatically adjusts detection settings based on server resource usage
- Dynamically selects the optimal model, resolution, and frame rate
- Monitors CPU, memory, and GPU usage in real-time
- Scales back resource usage when the server is under heavy load
- Improves performance automatically when resources are available

When auto-optimization is enabled:
- Manual configuration fields for model, interval, resolution, etc. are disabled
- The server continuously monitors resource usage and adjusts settings
- Optimization happens transparently without disrupting detection

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

## Automatic Detector Recovery

The integration includes a detector health monitoring system that:

- Checks every minute if each configured detector is running on the server
- Monitors detector responses to verify they're properly functioning
- Automatically recreates detectors if they're missing or not responding
- Ensures continuous operation even if the processing server restarts

This feature helps maintain system reliability by automatically recovering from:
- Server restarts or crashes
- Network interruptions
- Configuration loss on the server side
- Detector process termination

The health check runs in the background and requires no manual intervention.

## License

This project is licensed under the MIT License.

---



# YOLO Presence Detection Processing Server

This is the processing server component for the Home Assistant YOLO Presence Detection integration. It handles video stream processing and object detection using PyTorch and YOLO models.

## Why a Separate Server?

The processing server is separated from the Home Assistant integration for several reasons:

1. **Python Version Compatibility**: Home Assistant requires Python 3.13, but some ML libraries like PyTorch and ultralytics don't fully support Python 3.13 yet.
2. **Dependency Management**: ML libraries have complex dependencies that can conflict with Home Assistant's environment.
3. **Resource Isolation**: Video processing and ML inference are resource-intensive tasks that can be run on a separate machine with a GPU.
4. **Scalability**: Multiple Home Assistant instances can connect to a single processing server, or a processing server can handle multiple camera streams.

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU (optional, but recommended for better performance)
  - If using GPU, make sure you have the NVIDIA Container Toolkit installed
- RTSP or HTTP camera streams

## Getting Started

1. Clone this repository
2. Place your YOLO models in the `models` directory (if you don't have models, the server will download them from ultralytics)
3. Start the server with Docker Compose:

```bash
cd processing_unit
docker-compose up -d
```

The server will be available at http://localhost:5505.

## API Endpoints

- `GET /api/status`: Get server status information
- `GET /api/detectors`: List all detector instances
- `POST /api/detectors`: Create a new detector instance
- `GET /api/detectors/{detector_id}`: Get detector status
- `DELETE /api/detectors/{detector_id}`: Delete a detector
- `GET /api/detectors/{detector_id}/frame`: Get the latest processed frame (for debugging)

## Usage with Home Assistant

1. Install the YOLO Presence Detection integration in Home Assistant
2. When adding a new device, set the processing server URL to `http://<your-server-ip>:5505`
3. Configure the remaining options as needed

## Configuration

The following environment variables can be set in the `docker-compose.yml` file:

- `PORT`: The port to run the server on (default: 5505)
- `DEBUG`: Enable debug mode (default: false)

## Troubleshooting

- If your computer doesn't have an NVIDIA GPU, remove the `deploy.resources` section from `docker-compose.yml`
- If you experience memory issues, you can adjust CUDA memory allocation by setting the environment variable `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` in the docker-compose file
- Check logs with `docker-compose logs -f`

## Models

The server supports the following YOLO models:

- yolo11n: Fastest, lowest accuracy
- yolo11s: Fast, good accuracy
- yolo11m: Balanced speed/accuracy
- yolo11l: Slower, high accuracy
- yolo11x: Slowest, highest accuracy
