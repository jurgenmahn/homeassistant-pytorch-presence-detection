# YOLO Presence Detection Processing Server

This is the processing server component for the Home Assistant YOLO Presence Detection integration. It handles video stream processing and object detection using PyTorch and YOLO models through an HTTP API.

## Poll-Based Detection Architecture

This server implements a poll-based detection architecture:

1. **Stream Monitoring**: Maintains camera stream connections in background threads
2. **On-Demand Detection**: Performs detection only when explicitly requested via HTTP poll
3. **HTTP API**: Provides REST-like API endpoints for integration with Home Assistant
4. **Resource Monitoring**: Monitors system resources for auto-optimization
5. **Error Recovery**: Automatically recovers from stream connection issues

## Key Features

- **Efficient Stream Handling**: Maintains stream connections with minimal resources
- **On-Demand Detection**: Only performs resource-intensive detection when requested
- **Multiple Model Support**: Compatible with different YOLO model sizes
- **GPU Acceleration**: Automatically uses CUDA when available
- **Resolution Auto-Adjustment**: Ensures dimensions are multiples of YOLO stride
- **Detailed Detection Information**: Returns comprehensive detection results
- **Password Security**: Securely masks credentials in logs
- **Automatic Resource Optimization**: Dynamically adjusts settings based on system load

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU (optional, but recommended for better performance)
  - If using GPU, make sure you have the NVIDIA Container Toolkit installed
- RTSP or HTTP camera streams

## Getting Started

1. Clone this repository
2. Place your YOLO models in the `models` directory (if you don't have models, the server will download them automatically)
3. Start the server with Docker Compose:

```bash
cd processing_unit
docker-compose up -d
```

The server will be available at http://localhost:5505.

## API Endpoints

- `GET /health`: Get server health status
- `GET /detectors`: List all active detectors
- `POST /poll`: Main endpoint for detection polling
- `POST /shutdown`: Gracefully shut down a detector
- `GET /state`: Get current detector state

### Poll Endpoint

The `/poll` endpoint is the primary API endpoint, used for creating/updating detector configurations and performing detection. 

#### Request Format:
```json
{
  "detector_id": "unique_id",
  "config": {
    "stream_url": "rtsp://user:pass@camera-ip:554/stream",
    "name": "Living Room Camera",
    "model": "yolo11l",
    "input_size": "640x480",
    "detection_interval": 10,
    "confidence_threshold": 0.25,
    "frame_skip_rate": 5,
    "use_auto_optimization": false
  }
}
```

#### Response Format:
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

## Configuration

The following environment variables can be set in the `docker-compose.yml` file:

- `HTTP_PORT`: The port to run the server on (default: 5505)
- `DEBUG`: Enable debug mode (default: false)

## Usage with Home Assistant

1. Install the YOLO Presence Detection integration in Home Assistant
2. When adding a new device, set the processing server URL to `http://<your-server-ip>:5505`
3. Configure the remaining options as needed

## Troubleshooting

- If your computer doesn't have an NVIDIA GPU, remove the `deploy.resources` section from `docker-compose.yml`
- If you experience memory issues, you can adjust CUDA memory allocation by setting the environment variable `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` in the docker-compose file
- Check logs with `docker-compose logs -f`
- Connection issues with RTSP streams: The server will automatically try multiple transport methods (TCP/UDP)
- For RTSP URLs with credentials, ensure they're properly URL-encoded
- Output resolution may be adjusted to be a multiple of 32 (YOLO's stride requirement)

## Models

The server supports the following YOLO models:

- **yolo11n**: Fastest, lowest accuracy (great for low-power CPUs)
- **yolo11s**: Fast, good accuracy (good balance for most CPUs)
- **yolo11m**: Balanced speed/accuracy (recommended for entry-level GPUs)
- **yolo11l**: Slower, high accuracy (recommended default)
- **yolo11x**: Slowest, highest accuracy (for powerful GPUs)

## Auto-Optimization

When auto-optimization is enabled, the server will:

1. Monitor CPU/GPU usage and memory consumption
2. Adjust frame skip rate based on system load
3. Modify detection interval based on available resources
4. Reduce quality settings when connection issues are detected
5. Gradually increase quality when resources are abundant

This helps maintain reliable detection even as system conditions change.

## Docker Image Options

The included Dockerfile uses Python 3.12.3 with optimized dependencies for both CPU and GPU operation. The container is designed to be lightweight while still supporting all necessary ML libraries.

For machines with limited resources, you can also use a pre-built image from Docker Hub:
```
image: your-dockerhub-username/yolo-presence-server:latest
```

## License

This project is licensed under the MIT License.