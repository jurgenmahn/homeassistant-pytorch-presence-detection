"""
Watchdog module for the YOLO Presence Detection Server.

This module provides watchdog mechanisms to monitor and recover from
various failure conditions, ensuring the system remains stable.
"""
import os
import sys
import time
import threading
import logging
import signal
import psutil
import subprocess
from typing import Optional, Dict, Any, Callable, List

# Import custom logging configuration
try:
    from logging_config import get_logger
    logger = get_logger("watchdog")
except ImportError:
    # Fallback logging if logging_config is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("watchdog")

class ResourceMonitor:
    """Monitor system resources and take action if thresholds are exceeded."""
    
    def __init__(
        self,
        check_interval: int = 60,  # seconds
        cpu_threshold: float = 90.0,  # percent
        memory_threshold: float = 90.0,  # percent
        gpu_threshold: float = 90.0,  # percent
        disk_threshold: float = 90.0,  # percent
        action_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize the resource monitor.
        
        Args:
            check_interval: How often to check resources (seconds)
            cpu_threshold: CPU usage threshold (percent)
            memory_threshold: Memory usage threshold (percent)
            gpu_threshold: GPU memory usage threshold (percent)
            disk_threshold: Disk usage threshold (percent)
            action_callback: Function to call when thresholds are exceeded
        """
        self.check_interval = check_interval
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.gpu_threshold = gpu_threshold
        self.disk_threshold = disk_threshold
        self.action_callback = action_callback
        
        self.running = False
        self.thread = None
        self.stop_event = threading.Event()
        
        # GPU monitoring
        self.has_gpu = False
        self.gpu_type = None
        self._check_gpu_availability()
    
    def _check_gpu_availability(self):
        """Check if GPU is available and what type."""
        try:
            import torch
            if torch.cuda.is_available():
                self.has_gpu = True
                self.gpu_type = "cuda"
                logger.info(f"GPU monitoring enabled: CUDA device count: {torch.cuda.device_count()}")
            elif hasattr(torch.backends, "hip") and hasattr(torch.backends.hip, "is_built") and torch.backends.hip.is_built():
                self.has_gpu = True
                self.gpu_type = "rocm"
                logger.info("GPU monitoring enabled: ROCm")
            else:
                logger.info("No GPU detected, GPU monitoring disabled")
        except ImportError:
            logger.info("PyTorch not available, GPU monitoring disabled")
        except Exception as ex:
            logger.warning(f"Error checking GPU availability: {str(ex)}")
    
    def start(self):
        """Start the resource monitor thread."""
        if self.running:
            return
        
        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Resource monitor started")
    
    def stop(self):
        """Stop the resource monitor thread."""
        if not self.running:
            return
        
        self.running = False
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Resource monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                self._check_resources()
            except Exception as ex:
                logger.error(f"Error in resource monitor: {str(ex)}")
            
            # Wait for next check interval or until stop event
            for _ in range(self.check_interval):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
    
    def _check_resources(self):
        """Check system resources and take action if thresholds are exceeded."""
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.cpu_threshold:
            logger.warning(f"CPU usage above threshold: {cpu_percent:.1f}% > {self.cpu_threshold:.1f}%")
            if self.action_callback:
                self.action_callback("cpu", cpu_percent)
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > self.memory_threshold:
            logger.warning(f"Memory usage above threshold: {memory.percent:.1f}% > {self.memory_threshold:.1f}%")
            if self.action_callback:
                self.action_callback("memory", memory.percent)
        
        # Check disk usage
        disk = psutil.disk_usage('/')
        if disk.percent > self.disk_threshold:
            logger.warning(f"Disk usage above threshold: {disk.percent:.1f}% > {self.disk_threshold:.1f}%")
            if self.action_callback:
                self.action_callback("disk", disk.percent)
        
        # Check GPU usage if available
        if self.has_gpu:
            try:
                if self.gpu_type == "cuda":
                    self._check_nvidia_gpu()
                elif self.gpu_type == "rocm":
                    self._check_amd_gpu()
            except Exception as ex:
                logger.error(f"Error checking GPU usage: {str(ex)}")
    
    def _check_nvidia_gpu(self):
        """Check NVIDIA GPU usage using nvidia-smi."""
        try:
            # Try to use nvidia-smi to get GPU memory usage
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse output
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                    
                used, total = map(int, line.split(','))
                if total > 0:
                    percent = (used / total) * 100
                    if percent > self.gpu_threshold:
                        logger.warning(f"GPU memory usage above threshold: {percent:.1f}% > {self.gpu_threshold:.1f}%")
                        if self.action_callback:
                            self.action_callback("gpu", percent)
        except (subprocess.SubprocessError, FileNotFoundError):
            # Fall back to PyTorch if nvidia-smi fails
            try:
                import torch
                for i in range(torch.cuda.device_count()):
                    # Get current and max memory allocated
                    current = torch.cuda.memory_allocated(i)
                    max_mem = torch.cuda.get_device_properties(i).total_memory
                    if max_mem > 0:
                        percent = (current / max_mem) * 100
                        if percent > self.gpu_threshold:
                            logger.warning(f"GPU {i} memory usage above threshold: {percent:.1f}% > {self.gpu_threshold:.1f}%")
                            if self.action_callback:
                                self.action_callback("gpu", percent)
            except Exception as ex:
                logger.error(f"Error checking CUDA memory: {str(ex)}")
                self.has_gpu = False  # Disable GPU monitoring if it fails
    
    def _check_amd_gpu(self):
        """Check AMD GPU usage using rocm-smi."""
        try:
            # Try to use rocm-smi to get GPU memory usage
            result = subprocess.run(
                ["rocm-smi", "--showmemuse"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse output (simplified, actual parsing would depend on rocm-smi output format)
            if "GPU memory use" in result.stdout:
                # This is a very simplified check - actual parsing would need to be adapted
                # to the specific output format of rocm-smi
                logger.info("ROCm GPU monitoring not fully implemented")
            
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("rocm-smi not available, disabling GPU monitoring")
            self.has_gpu = False  # Disable GPU monitoring if it fails

class ProcessWatchdog:
    """Monitor and restart processes if they fail."""
    
    def __init__(
        self,
        process_name: str,
        start_cmd: List[str],
        check_interval: int = 30,  # seconds
        restart_delay: int = 5,  # seconds
        max_restarts: int = 3,
        restart_window: int = 300,  # seconds
        working_dir: Optional[str] = None
    ):
        """
        Initialize the process watchdog.
        
        Args:
            process_name: Name of the process to monitor
            start_cmd: Command to start the process
            check_interval: How often to check the process (seconds)
            restart_delay: Delay before restarting the process (seconds)
            max_restarts: Maximum number of restarts in restart_window
            restart_window: Time window for max_restarts (seconds)
            working_dir: Working directory for the process
        """
        self.process_name = process_name
        self.start_cmd = start_cmd
        self.check_interval = check_interval
        self.restart_delay = restart_delay
        self.max_restarts = max_restarts
        self.restart_window = restart_window
        self.working_dir = working_dir or os.getcwd()
        
        self.running = False
        self.thread = None
        self.stop_event = threading.Event()
        self.process = None
        self.restart_times = []
    
    def start(self):
        """Start the watchdog thread."""
        if self.running:
            return
        
        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Process watchdog started for {self.process_name}")
    
    def stop(self):
        """Stop the watchdog thread and the monitored process."""
        if not self.running:
            return
        
        self.running = False
        self.stop_event.set()
        
        # Stop the monitored process
        self._stop_process()
        
        # Stop the watchdog thread
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info(f"Process watchdog stopped for {self.process_name}")
    
    def _start_process(self):
        """Start the monitored process."""
        try:
            logger.info(f"Starting process: {' '.join(self.start_cmd)}")
            self.process = subprocess.Popen(
                self.start_cmd,
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return True
        except Exception as ex:
            logger.error(f"Error starting process {self.process_name}: {str(ex)}")
            return False
    
    def _stop_process(self):
        """Stop the monitored process."""
        if not self.process:
            return
        
        try:
            logger.info(f"Stopping process {self.process_name}")
            self.process.terminate()
            
            # Wait for process to terminate
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                logger.warning(f"Process {self.process_name} did not terminate, force killing")
                self.process.kill()
                self.process.wait(timeout=5)
            
            self.process = None
        except Exception as ex:
            logger.error(f"Error stopping process {self.process_name}: {str(ex)}")
    
    def _check_process(self):
        """Check if the process is running."""
        if not self.process:
            return False
        
        # Check if process is still running
        if self.process.poll() is not None:
            return False
        
        return True
    
    def _can_restart(self):
        """Check if we can restart the process based on restart limits."""
        current_time = time.time()
        
        # Remove old restart times outside the window
        self.restart_times = [t for t in self.restart_times if current_time - t <= self.restart_window]
        
        # Check if we've exceeded the maximum number of restarts
        return len(self.restart_times) < self.max_restarts
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        # Start the process initially
        self._start_process()
        
        while not self.stop_event.is_set():
            try:
                # Check if process is running
                if not self._check_process():
                    logger.warning(f"Process {self.process_name} is not running")
                    
                    # Check if we can restart
                    if self._can_restart():
                        logger.info(f"Restarting process {self.process_name}")
                        time.sleep(self.restart_delay)
                        if self._start_process():
                            # Record restart time
                            self.restart_times.append(time.time())
                    else:
                        logger.error(f"Too many restarts for {self.process_name}, giving up")
                        break
            except Exception as ex:
                logger.error(f"Error in process watchdog: {str(ex)}")
            
            # Wait for next check interval or until stop event
            for _ in range(self.check_interval):
                if self.stop_event.is_set():
                    break
                time.sleep(1)

class SystemWatchdog:
    """System-wide watchdog to monitor and recover from various failure conditions."""
    
    def __init__(self):
        """Initialize the system watchdog."""
        self.resource_monitor = None
        self.process_watchdogs = {}
        self.running = False
    
    def start(self):
        """Start all watchdog components."""
        if self.running:
            return
        
        # Start resource monitor
        self.resource_monitor = ResourceMonitor(
            action_callback=self._handle_resource_threshold_exceeded
        )
        self.resource_monitor.start()
        
        self.running = True
        logger.info("System watchdog started")
    
    def stop(self):
        """Stop all watchdog components."""
        if not self.running:
            return
        
        # Stop resource monitor
        if self.resource_monitor:
            self.resource_monitor.stop()
        
        # Stop all process watchdogs
        for watchdog in self.process_watchdogs.values():
            watchdog.stop()
        
        self.running = False
        logger.info("System watchdog stopped")
    
    def add_process_watchdog(self, name, start_cmd, **kwargs):
        """Add a process to be monitored."""
        if name in self.process_watchdogs:
            logger.warning(f"Process watchdog for {name} already exists, replacing")
            self.process_watchdogs[name].stop()
        
        watchdog = ProcessWatchdog(name, start_cmd, **kwargs)
        self.process_watchdogs[name] = watchdog
        watchdog.start()
        logger.info(f"Added process watchdog for {name}")
    
    def remove_process_watchdog(self, name):
        """Remove a process from being monitored."""
        if name in self.process_watchdogs:
            watchdog = self.process_watchdogs[name]
            watchdog.stop()
            del self.process_watchdogs[name]
            logger.info(f"Removed process watchdog for {name}")
        else:
            logger.warning(f"No process watchdog found for {name}")
            
    def _handle_resource_threshold_exceeded(self, resource_type, value):
        """Handle a resource threshold being exceeded."""
        logger.warning(f"Resource threshold exceeded: {resource_type} = {value:.1f}%")
        
        # Take appropriate action based on resource type
        if resource_type == "memory":
            # For memory issues, try to free up memory
            self._handle_memory_pressure()
        elif resource_type == "cpu":
            # For CPU issues, might throttle processing
            self._handle_cpu_pressure()
        elif resource_type == "gpu":
            # For GPU issues, might throttle processing
            self._handle_gpu_pressure()
            
    def _handle_memory_pressure(self):
        """Handle memory pressure situation."""
        logger.info("Taking action to handle memory pressure")
        # Could implement memory-saving measures here
        
    def _handle_cpu_pressure(self):
        """Handle CPU pressure situation."""
        logger.info("Taking action to handle CPU pressure")
        # Could implement CPU throttling here
        
    def _handle_gpu_pressure(self):
        """Handle GPU pressure situation."""
        logger.info("Taking action to handle GPU pressure")
        # Could implement GPU throttling here


# Global system watchdog instance
system_watchdog = SystemWatchdog()

def start_watchdog():
    """Start the system watchdog."""
    system_watchdog.start()

def stop_watchdog():
    """Stop the system watchdog."""
    system_watchdog.stop()

# Start watchdog if module is run directly
if __name__ == "__main__":
    start_watchdog()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_watchdog()