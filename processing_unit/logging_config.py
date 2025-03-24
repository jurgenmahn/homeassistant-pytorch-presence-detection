"""
Logging configuration for the YOLO Presence Detection Server.

This module provides structured logging with rotation capabilities
to improve diagnostics and troubleshooting.
"""
import os
import sys
import logging
import logging.handlers
from typing import Optional, Dict, Any

# Default log directory - prefer current directory for Docker compatibility
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(os.getcwd(), "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
print(f"Logging to {LOG_DIR}")  # Print directly to be sure this is visible

# Log file paths
MAIN_LOG_FILE = os.path.join(LOG_DIR, "yolo_presence.log")
ERROR_LOG_FILE = os.path.join(LOG_DIR, "yolo_presence_error.log")
DEBUG_LOG_FILE = os.path.join(LOG_DIR, "yolo_presence_debug.log")

# Log format with more detailed information
DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
SIMPLE_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def configure_logging(
    logger_name: str = "yolo_presence_server",
    log_level: int = logging.INFO,
    console_level: int = logging.DEBUG,  # Set console level to DEBUG by default
    file_level: int = logging.DEBUG,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    log_to_console: bool = True,
    log_to_file: bool = True,
    detailed_format: bool = True
) -> logging.Logger:
    """
    Configure logging with console and rotating file handlers.
    
    Args:
        logger_name: Name of the logger
        log_level: Overall logging level
        console_level: Logging level for console output
        file_level: Logging level for file output
        max_file_size: Maximum size of each log file in bytes
        backup_count: Number of backup log files to keep
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        detailed_format: Whether to use detailed log format
        
    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    log_format = DETAILED_FORMAT if detailed_format else SIMPLE_FORMAT
    formatter = logging.Formatter(log_format)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handlers (if enabled)
    if log_to_file:
        # Main log file (all levels)
        file_handler = logging.handlers.RotatingFileHandler(
            MAIN_LOG_FILE,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Error log file (ERROR and above)
        error_handler = logging.handlers.RotatingFileHandler(
            ERROR_LOG_FILE,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
        
        # Debug log file (DEBUG and above) if debug level is enabled
        if log_level <= logging.DEBUG:
            debug_handler = logging.handlers.RotatingFileHandler(
                DEBUG_LOG_FILE,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(formatter)
            logger.addHandler(debug_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the given name, creating it if necessary.
    
    Args:
        name: Logger name (if None, returns the root yolo_presence_server logger)
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"yolo_presence_server.{name}")
    return logging.getLogger("yolo_presence_server")

# Context manager for logging exceptions with additional context
class LogExceptionContext:
    """Context manager for logging exceptions with additional context."""
    
    def __init__(self, logger: logging.Logger, context: str, extra: Optional[Dict[str, Any]] = None):
        """
        Initialize the context manager.
        
        Args:
            logger: Logger to use
            context: Context description for the log message
            extra: Additional context information to include in the log
        """
        self.logger = logger
        self.context = context
        self.extra = extra or {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Format the exception details
            import traceback
            tb_str = ''.join(traceback.format_exception(exc_type, exc_val, exc_tb))
            
            # Log the exception with context
            context_str = f"{self.context}"
            if self.extra:
                context_str += f" (Context: {self.extra})"
            
            self.logger.error(f"Exception in {context_str}: {exc_val}\n{tb_str}")
        return False  # Don't suppress the exception
