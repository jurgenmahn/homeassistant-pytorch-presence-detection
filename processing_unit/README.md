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

The server will be available at http://localhost:5000.

## API Endpoints

- `GET /api/status`: Get server status information
- `GET /api/detectors`: List all detector instances
- `POST /api/detectors`: Create a new detector instance
- `GET /api/detectors/{detector_id}`: Get detector status
- `DELETE /api/detectors/{detector_id}`: Delete a detector
- `GET /api/detectors/{detector_id}/frame`: Get the latest processed frame (for debugging)

## Usage with Home Assistant

1. Install the YOLO Presence Detection integration in Home Assistant
2. When adding a new device, set the processing server URL to `http://<your-server-ip>:5000`
3. Configure the remaining options as needed

## Configuration

The following environment variables can be set in the `docker-compose.yml` file:

- `PORT`: The port to run the server on (default: 5000)
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

Models should be placed in the `models` directory with the `.pt` extension (e.g., `yolo11l.pt`).