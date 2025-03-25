# Home Assistant YOLO Presence Detection Integration

This integration provides presence detection using YOLO object detection on camera streams. It uses a separate processing server to handle the video processing and ML inference, while providing seamless integration with Home Assistant entities and automations.

## Major Update: Poll-Based Architecture

This is a major update to the YOLO Presence Detection integration, introducing a new poll-based HTTP architecture:

1. **Home Assistant Integration**: Controls when detection happens through regular HTTP polling
2. **Processing Server**: A separate Docker container that maintains stream connections and performs detection on demand

This architecture provides several benefits:
- **Reliability**: Stream connections are maintained in the background, but resource-intensive detection only happens when requested
- **Control**: Home Assistant controls the frequency of detection through configurable polling intervals
- **Resource Management**: Processing server can handle multiple streams efficiently with automatic resource optimization
- **Resilience**: Polling architecture is more resilient to network issues and interruptions
- **Recovery**: Automatic reconnection and health monitoring ensures continuous operation

## Features

- **Person Detection**: Detects when people are present in the camera feed
- **Pet Detection**: Detects cats and dogs in the camera feed
- **Counting**: Counts the number of people and pets detected
- **GPU Acceleration**: Automatically uses GPU acceleration (CUDA) when available
- **Adjustable Parameters**: Configure detection thresholds, intervals, and resolution
- **Multiple Models**: Choose from different YOLO models based on your hardware capabilities
- **Auto-Optimization**: Optional automatic resource management based on system usage
- **Detailed Logging**: Comprehensive logging with masked credentials for security

## Installation

### Step 1: Set up the Processing Server

1. Clone the repository to your machine
2. Navigate to the processing_unit directory
3. Start the Docker container:

```bash
cd processing_unit
docker-compose up -d
```

For more details, see the processing server documentation below.

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
| Detection Interval | Seconds between detection polls | 10s (CPU), 5s (GPU) |
| Confidence Threshold | Minimum detection confidence (0.1-0.9) | 0.25 |
| Input Resolution | Resolution for processing (adjusted to be multiple of 32) | 640x480 |
| Frame Skip Rate | Process 1 out of X frames | 5 (CPU), 3 (GPU) |

### Poll-Based Detection

The integration uses a poll-based architecture where:

- Home Assistant polls the server at regular intervals (as set by Detection Interval)
- The server maintains camera stream connections in the background
- Detection only happens when a poll request is received
- Results are returned immediately in the HTTP response
- Each poll includes the full detector configuration

Benefits of this approach:
- Reduced resource usage when detection isn't needed
- Client controls exactly when detection happens
- More resilient to network interruptions
- Easier to monitor and debug
- Configurable detection frequency

### Auto-Optimization

The integration includes an automatic resource optimization feature that:

- Automatically adjusts detection settings based on server resource usage
- Monitors CPU, memory, and GPU usage in real-time
- Scales back resource usage when the server is under heavy load
- Improves performance automatically when resources are available
- Adjusts parameters based on stream connection health

When auto-optimization is enabled:
- Manual configuration fields for model, interval, resolution, etc. are disabled
- The server continuously monitors resource usage and adjusts settings
- Optimization happens transparently without disrupting detection

## Hardware Requirements

### For the Processing Server:
- Any machine capable of running Docker
- For GPU acceleration: NVIDIA GPU with CUDA support
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

- **HTTP Polling Issues**: If entities aren't updating, check that the configured detection interval is appropriate
- **Inconsistent Detection**: Try increasing the confidence threshold or using a more accurate model
- **High CPU Usage**: Use a smaller model or increase the detection interval
- **Memory Issues**: Lower the input resolution or reduce the frame skip rate
- **Connection Problems**: Verify your RTSP URL is correct and accessible from the processing server
- **Performance Insights**: Review the detector output in the logs, which includes detailed information about:
  - Detection times in milliseconds
  - Original frame dimensions vs. model input dimensions
  - Auto-optimization status and adjustments
  - Detected objects and their counts

## RTSP Stream Compatibility

The system supports various RTSP stream formats:

- Standard RTSP streams from IP cameras
- RTSP streams with credentials (username:password format)
- HTTP video streams
- ONVIF-compliant camera streams

For optimal performance:
- Configure your camera for H.264 encoding when possible
- Use a resolution appropriate for detection (720p or lower recommended)
- Ensure your network has sufficient bandwidth between the processing server and camera

## Processing Server Architecture

The processing server component has the following key components:

1. **HTTP Server**: Handles requests from Home Assistant clients
2. **YOLO Model Management**: Loads and configures YOLO models for detection
3. **Stream Monitor**: Background threads that maintain camera connections
4. **Detector Instances**: One per configured camera, maintaining state
5. **Resource Monitor**: Monitors system resources for auto-optimization

### API Endpoints

- `GET /health`: Server health check
- `POST /poll`: Main endpoint for detection polling
- `POST /shutdown`: Gracefully shut down a detector
- `GET /state`: Get current detector state
- `GET /detectors`: List all active detectors

## License

This project is licensed under the MIT License.

---

## 🚀 Built with human ingenuity & a dash of AI wizardry

This project emerged from late-night coding sessions, unexpected inspiration, and the occasional debugging dance. Every line of code has a story behind it.

Found a bug? Have a wild idea? The issues tab is your canvas.

Authored By: [👨‍💻 Jurgen Mahn](https://github.com/jurgenmahn) with some help from AI code monkies [Claude](https://claude.ai) & [Manus.im](https://manus.im/app)

*"Sometimes the code writes itself. Other times, we collaborate with the machines."*

⚡ Happy hacking, fellow explorer ⚡