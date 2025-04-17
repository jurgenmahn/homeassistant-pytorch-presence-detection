# Home Assistant YOLO Presence Detection Integration

## Why Use This Integration?

Traditional motion sensors fail in one critical scenario: detecting people who are sitting still. Whether you're reading a book, watching TV, or working at your desk, standard motion sensors will falsely report an empty room once movement stops. This leads to lights turning off while you're still present, thermostats reducing heating in occupied spaces, and security systems making incorrect assessments.

The YOLO Presence Detection Integration solves this problem by using computer vision and machine learning to actively recognize people and pets in your camera feeds - whether they're moving or completely still. This enables truly reliable presence detection for:

- Living rooms where people watch TV or read
- Home offices with seated workers
- Bedrooms where people might be sleeping
- Any space where traditional motion detection falls short

Get accurate presence data for smarter automations, better energy management, and improved security.

## Features

- **Person Detection**: Detects when people are present in the camera feed, even when sitting still
- **Pet Detection**: Identifies cats and dogs in the camera feed
- **Counting**: Tracks the number of people and pets detected
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

## System Architecture

The integration uses a poll-based architecture where:

- Home Assistant polls the server at regular intervals
- The server maintains camera stream connections in the background
- Detection only happens when a poll request is received
- Results are returned immediately in the HTTP response

Benefits of this approach:
- Reduced resource usage when detection isn't needed
- Client controls exactly when detection happens
- More resilient to network interruptions
- Easier to monitor and debug
- Configurable detection frequency

## Auto-Optimization

When auto-optimization is enabled:
- The server monitors CPU, memory, and GPU usage in real-time
- Detection settings are automatically adjusted based on server load
- Performance improves automatically when resources are available
- Manual configuration fields are disabled

## API Documentation

The processing server exposes several HTTP endpoints for integration with Home Assistant and other clients. Here's how to interact with the API:

### Key Endpoints

#### 1. Poll Endpoint - `/poll` (POST)

The primary endpoint used by the Home Assistant integration to create/update a detector, perform detection, and get results.

**Request payload:**
```json
{
  "detector_id": "unique_id",
  "config": {
    "stream_url": "rtsp://username:password@camera-ip:554/stream",
    "name": "Living Room Camera",
    "model": "yolo11l",
    "input_size": "640x480",
    "detection_interval": 10,
    "confidence_threshold": 0.25,
    "frame_skip_rate": 5,
    "detection_frame_count": 5,
    "consistent_detection_count": 3,
    "use_auto_optimization": false
  }
}
```

**Response:**
```json
{
  "status": "success",
  "detector_id": "unique_id",
  "detection_time": 0.123,
  "state": {
    "human_detected": true,
    "pet_detected": false,
    "human_count": 2,
    "pet_count": 0,
    "connection_status": "connected",
    "last_update": 1616867123.45,
    "inference_time_ms": 124.5,
    "detected_objects": {"person": 2},
    "requested_resolution": "640x480",
    "actual_resolution": "640x480"
  }
}
```

#### 2. Shutdown Endpoint - `/shutdown` (POST)

Gracefully shuts down a detector instance and frees associated resources.

**Request payload:**
```json
{
  "detector_id": "unique_id"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Detector unique_id shutdown"
}
```

#### 3. State Endpoint - `/state` (GET)

Gets the current state of a detector without performing detection.

**Query parameter:**
- `detector_id=unique_id`

**Response:**
```json
{
  "status": "success",
  "detector_id": "unique_id",
  "state": {
    "detector_id": "unique_id",
    "name": "Living Room Camera",
    "stream_url": "rtsp://username:password@camera-ip:554/stream",
    "model": "yolo11l",
    "input_size": "640x480",
    "detection_interval": 10,
    "confidence_threshold": 0.25,
    "frame_skip_rate": 5,
    "detection_frame_count": 5,
    "consistent_detection_count": 3,
    "is_running": true,
    "connection_status": "connected",
    "device": "cuda:0",
    "human_detected": true,
    "pet_detected": false,
    "human_count": 2,
    "pet_count": 0
  }
}
```

#### 4. Detectors List - `/detectors` (GET)

Lists all active detectors with basic information.

**Response:**
```json
{
  "status": "success",
  "detectors": [
    {
      "detector_id": "unique_id1",
      "name": "Living Room Camera",
      "is_running": true,
      "connection_status": "connected"
    },
    {
      "detector_id": "unique_id2",
      "name": "Front Door Camera",
      "is_running": true,
      "connection_status": "connected"
    }
  ]
}
```

#### 5. Health Check - `/health` (GET)

Simple endpoint to verify the server is running.

**Response:**
```json
{
  "status": "ok",
  "message": "Server is running",
  "detector_count": 2
}
```

### Visual Endpoints

The following endpoints provide visual access to detector streams:

- **View UI** - `/view?detector_id=unique_id`: HTML page with live stream and detection visualizations
- **Direct JPEG** - `/jpeg?detector_id=unique_id`: Latest processed JPEG image with detection annotations
- **Main Dashboard** - `/`: Lists all active detectors with status and links to individual views

## Hardware Requirements

### For the Processing Server:
- Any machine capable of running Docker
- For GPU acceleration: NVIDIA GPU with CUDA support, at least 1GB of VRAM (2GB for larger models)

### For Home Assistant:
- Regular Home Assistant requirements (as processing happens externally)

## Models

- **YOLO11 Nano**: Fastest, lowest power usage, less accurate
- **YOLO11 Small**: Good balance for lower-end hardware
- **YOLO11 Medium**: Balanced performance and accuracy
- **YOLO11 Large**: Good accuracy, moderate performance requirements
- **YOLO11 Extra Large**: Best accuracy, highest resource usage

## Processing Server Details

The processing server includes:

- HTTP Server for handling requests
- YOLO Model Management
- Stream Monitoring in background threads
- Resource Monitoring for auto-optimization
- Web Interface for visual monitoring

### API Endpoints

- `GET /`: Main web interface
- `GET /health`: Server health check
- `POST /poll`: Main endpoint for detection polling
- `POST /shutdown`: Gracefully shut down a detector
- `GET /state`: Get current detector state
- `GET /detectors`: List all active detectors
- `GET /view?detector_id=ID`: Visual detector monitoring
- `GET /stream?detector_id=ID`: MJPEG stream access

## Web Interface

The processing server includes a built-in web interface that provides visual monitoring and management of your detection streams. This interface is accessible at `http://<server-ip>:5505/` by default.

### Key Features

- **Dashboard Overview**: The main page lists all active detectors with their status, model, and detection counts
- **Live Stream Viewing**: Each detector has a dedicated view page showing the camera feed with real-time detection overlays
- **Detection Visualization**: Bounding boxes highlight detected people and pets with confidence scores
- **Detection Statistics**: View counts, inference times, and detection history
- **Multi-frame Analysis**: See all frames used in the detection process with a summary of results
- **Mobile-Friendly Design**: Responsive layout works on phones, tablets, and desktops

### Usage Scenarios

- **Setup & Configuration**: Verify camera streams are properly connected during initial setup
- **Performance Tuning**: Monitor inference times and detection quality to optimize settings 
- **Troubleshooting**: Diagnose connection issues or unexpected detection results visually
- **Monitoring**: Keep an eye on protected areas from any web browser

### Web Interface Security

For security in shared environments, you can enable basic authentication for the web interface by setting `ENABLE_AUTH=true` in the docker-compose.yml file.

## Troubleshooting

- **HTTP Polling Issues**: Check that the detection interval is appropriate
- **Inconsistent Detection**: Try increasing the confidence threshold or using a more accurate model
- **High CPU Usage**: Use a smaller model or increase the detection interval
- **Memory Issues**: Lower the input resolution or reduce the frame skip rate
- **Connection Problems**: Verify your RTSP URL is correct and accessible
- **Logs**: Check Docker logs with `docker-compose logs -f`
- **RTSP with Credentials**: Ensure they're properly URL-encoded
- **For CPU-only Systems**: Remove the `deploy.resources` section from `docker-compose.yml`

## RTSP Stream Compatibility

The system supports:
- Standard RTSP streams from IP cameras
- RTSP streams with credentials (username:password format)
- HTTP video streams
- ONVIF-compliant camera streams

For optimal performance:
- Use H.264 encoding when possible
- Use 720p or lower resolution for detection
- Ensure sufficient network bandwidth

## License

This project is licensed under the MIT License.

---

## 🚀 Built with human ingenuity & a dash of AI wizardry

This project emerged from late-night coding sessions, unexpected inspiration, and the occasional debugging dance. Every line of code has a story behind it.

Found a bug? Have a wild idea? The issues tab is your canvas.

Authored By: [👨‍💻 Jurgen Mahn](https://github.com/jurgenmahn) with some help from AI code monkies [Claude](https://claude.ai) & [Manus.im](https://manus.im/app)

*"Sometimes the code writes itself. Other times, we collaborate with the machines."*

⚡ Happy hacking, fellow explorer ⚡