# PyTorch YOLO Project Commands & Guidelines

## Environment and Setup
- Python version: 3.12.3
- Runtime: `python demo.py` - Runs YOLO object detection on RTSP stream

## Testing and Linting
- Install linting tools: `pip install black ruff isort mypy`
- Format code: `black .`
- Lint code: `ruff check .`
- Type checking: `mypy --ignore-missing-imports .`
- Run tests: `pytest tests/` or `pytest tests/test_file.py::test_function`

## Code Style & Conventions
- **Imports**: Group in order: standard library, third-party packages, local modules
- **Formatting**: 4-space indentation, 88 char line length (Black default)
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Type Hints**: Use them for function parameters and return values
- **Error Handling**: Use specific exceptions, handle exceptions gracefully
- **Comments**: Document complex logic, not obvious behavior
- **Documentation**: Docstrings for classes and functions (Google style)

## Project Structure
- Main application: demo.py
- YOLO models in models/ directory
- Use ultralytics YOLO package for model inference