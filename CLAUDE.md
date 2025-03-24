# PyTorch YOLO Project Commands & Guidelines

## Environment and Setup
- Python version: 3.12.3
- Default runtime: `docker-compose up -d` in processing_unit/ directory
- For development: `python processing_unit/server.py` for the processing server

## Testing and Linting
- Install linting tools: `pip install black ruff isort mypy`
- Format code: `black .`
- Lint code: `ruff check .`
- Type checking: `mypy --ignore-missing-imports .`
- Run single test: `python development/test-gpu-acceleration.py` (or other test files in development/)

## Code Style & Conventions
- **Imports**: Group in order: standard library, third-party packages, local modules
- **Formatting**: 4-space indentation, 88 char line length (Black default)
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Type Hints**: Use them for function parameters and return values
- **Error Handling**: Use specific exceptions, handle exceptions gracefully
- **Comments**: Document complex logic, not obvious behavior
- **Documentation**: Docstrings for classes and functions (Google style)

## Project Structure
- Processing Server: processing_unit/server.py
- HA Integration: custom_components/yolo_presence/
- YOLO models in models/ directory (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
- Uses ultralytics YOLO package for model inference