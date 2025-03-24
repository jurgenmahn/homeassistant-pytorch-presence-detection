"""
Resource optimization module for the YOLO Presence Detection Server.

This module provides utilities to optimize resource usage and implement
graceful degradation when system resources are constrained.
"""
import os
import sys
import time
import threading
import logging
import gc
from typing import Optional, Dict, Any, List, Tuple

# Import custom logging configuration
try:
    from logging_config import get_logger
    logger = get_logger("optimizer")
except ImportError:
    # Fallback logging if logging_config is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("optimizer")

class ResourceOptimizer:
    """Optimize resource usage based on system conditions."""
    
    def __init__(self):
        """Initialize the resource optimizer."""
        self.optimization_level = 0  # 0=none, 1=light, 2=medium, 3=aggressive
        self.last_optimization_time = 0
        self.optimization_cooldown = 300  # seconds between optimization level changes
        
        # Optimization thresholds
        self.cpu_thresholds = [70, 80, 90]  # percent
        self.memory_thresholds = [70, 80, 90]  # percent
        self.gpu_thresholds = [70, 80, 90]  # percent
        
        # Optimization settings
        self.settings = {
            # Detection settings
            "detection_interval": [5, 10, 15, 20],  # seconds between detections
            "frame_skip_rate": [2, 5, 10, 15],  # frames to skip
            "input_size": ["640x480", "480x360", "320x240", "160x120"],  # input resolution
            
            # Model settings
            "model_complexity": ["full", "medium", "light", "minimal"],  # conceptual complexity levels
            
            # Processing settings
            "max_detections": [50, 30, 20, 10],  # maximum detections per frame
            "confidence_threshold": [0.25, 0.3, 0.4, 0.5],  # minimum confidence
        }
        
        # Current optimization settings
        self.current_settings = {}
        self._initialize_current_settings()
    
    def _initialize_current_settings(self):
        """Initialize current settings to the default (level 0) values."""
        for setting, values in self.settings.items():
            self.current_settings[setting] = values[0]
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Get the current optimization settings."""
        return self.current_settings.copy()
    
    def get_optimization_level(self) -> int:
        """Get the current optimization level."""
        return self.optimization_level
    
    def set_optimization_level(self, level: int) -> Dict[str, Any]:
        """
        Set the optimization level and return the new settings.
        
        Args:
            level: Optimization level (0-3)
            
        Returns:
            Dictionary of new settings
        """
        # Validate level
        if level < 0 or level > 3:
            logger.warning(f"Invalid optimization level: {level}, must be 0-3")
            level = max(0, min(level, 3))
        
        # Check if we're in cooldown period
        current_time = time.time()
        if current_time - self.last_optimization_time < self.optimization_cooldown:
            logger.info(f"Optimization in cooldown period, ignoring level change to {level}")
            return self.current_settings.copy()
        
        # Update level and settings
        old_level = self.optimization_level
        self.optimization_level = level
        self.last_optimization_time = current_time
        
        # Update all settings based on the new level
        for setting, values in self.settings.items():
            self.current_settings[setting] = values[level]
        
        logger.info(f"Changed optimization level from {old_level} to {level}")
        logger.debug(f"New settings: {self.current_settings}")
        
        return self.current_settings.copy()
    
    def optimize_for_resources(self, cpu_percent: float, memory_percent: float, gpu_percent: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize settings based on current resource usage.
        
        Args:
            cpu_percent: CPU usage percentage
            memory_percent: Memory usage percentage
            gpu_percent: GPU memory usage percentage (optional)
            
        Returns:
            Dictionary of new settings
        """
        # Determine the highest resource usage
        resource_levels = []
        
        # Check CPU usage
        for i, threshold in enumerate(self.cpu_thresholds):
            if cpu_percent >= threshold:
                resource_levels.append(i + 1)
        
        # Check memory usage
        for i, threshold in enumerate(self.memory_thresholds):
            if memory_percent >= threshold:
                resource_levels.append(i + 1)
        
        # Check GPU usage if available
        if gpu_percent is not None:
            for i, threshold in enumerate(self.gpu_thresholds):
                if gpu_percent >= threshold:
                    resource_levels.append(i + 1)
        
        # Determine the new optimization level
        if not resource_levels:
            # No thresholds exceeded, use level 0
            new_level = 0
        else:
            # Use the highest level from any resource
            new_level = max(resource_levels)
        
        # Set the new level if it's different
        if new_level != self.optimization_level:
            logger.info(f"Resource usage triggered optimization level change to {new_level}")
            return self.set_optimization_level(new_level)
        
        return self.current_settings.copy()
    
    def suggest_model_for_level(self, current_model: str) -> str:
        """
        Suggest an appropriate model for the current optimization level.
        
        Args:
            current_model: Current model name
            
        Returns:
            Suggested model name
        """
        # Model complexity mapping
        model_complexity = self.current_settings.get("model_complexity", "full")
        
        # Model suggestions based on complexity
        if model_complexity == "minimal":
            return "yolo11n"  # Nano model
        elif model_complexity == "light":
            return "yolo11s"  # Small model
        elif model_complexity == "medium":
            return "yolo11m"  # Medium model
        else:
            return "yolo11l"  # Large model
    
    def optimize_memory_usage(self):
        """Perform immediate memory optimization."""
        logger.info("Performing memory optimization")
        
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared PyTorch CUDA cache")
        except (ImportError, AttributeError):
            pass
        
        return True

class GracefulDegradation:
    """Implement graceful degradation strategies when resources are constrained."""
    
    def __init__(self, resource_optimizer: ResourceOptimizer):
        """
        Initialize the graceful degradation handler.
        
        Args:
            resource_optimizer: Resource optimizer instance
        """
        self.resource_optimizer = resource_optimizer
        self.degradation_level = 0  # 0=none, 1=light, 2=medium, 3=severe
        self.degradation_strategies = {
            0: self._no_degradation,
            1: self._light_degradation,
            2: self._medium_degradation,
            3: self._severe_degradation
        }
    
    def degrade_gracefully(self, error_type: str, error_count: int) -> Tuple[int, Dict[str, Any]]:
        """
        Implement graceful degradation based on error type and count.
        
        Args:
            error_type: Type of error (e.g., "memory", "cuda", "stream")
            error_count: Number of consecutive errors
            
        Returns:
            Tuple of (degradation_level, settings)
        """
        # Determine degradation level based on error type and count
        if error_type == "memory":
            # Memory errors are serious, degrade more aggressively
            if error_count >= 10:
                new_level = 3
            elif error_count >= 5:
                new_level = 2
            elif error_count >= 2:
                new_level = 1
            else:
                new_level = 0
        elif error_type == "cuda":
            # CUDA errors might be recoverable with lighter degradation
            if error_count >= 15:
                new_level = 3
            elif error_count >= 8:
                new_level = 2
            elif error_count >= 3:
                new_level = 1
            else:
                new_level = 0
        elif error_type == "stream":
            # Stream errors might be temporary, degrade lightly
            if error_count >= 20:
                new_level = 2
            elif error_count >= 10:
                new_level = 1
            else:
                new_level = 0
        else:
            # Unknown error type, use general approach
            if error_count >= 15:
                new_level = 3
            elif error_count >= 8:
                new_level = 2
            elif error_count >= 3:
                new_level = 1
            else:
                new_level = 0
        
        # Apply the degradation strategy
        if new_level != self.degradation_level:
            logger.info(f"Changing degradation level from {self.degradation_level} to {new_level} due to {error_type} errors")
            self.degradation_level = new_level
        
        # Apply the corresponding strategy
        strategy = self.degradation_strategies.get(new_level, self._no_degradation)
        return new_level, strategy()
    
    def _no_degradation(self) -> Dict[str, Any]:
        """No degradation strategy."""
        # Reset optimization level to 0
        return self.resource_optimizer.set_optimization_level(0)
    
    def _light_degradation(self) -> Dict[str, Any]:
        """Light degradation strategy."""
        # Set optimization level to 1
        settings = self.resource_optimizer.set_optimization_level(1)
        
        # Perform memory optimization
        self.resource_optimizer.optimize_memory_usage()
        
        return settings
    
    def _medium_degradation(self) -> Dict[str, Any]:
        """Medium degradation strategy."""
        # Set optimization level to 2
        settings = self.resource_optimizer.set_optimization_level(2)
        
        # Perform memory optimization
        self.resource_optimizer.optimize_memory_usage()
        
        return settings
    
    def _severe_degradation(self) -> Dict[str, Any]:
        """Severe degradation strategy."""
        # Set optimization level to 3 (most aggressive)
        settings = self.resource_optimizer.set_optimization_level(3)
        
        # Perform memory optimization
        self.resource_optimizer.optimize_memory_usage()
        
        # Additional severe degradation actions could be added here
        
        return settings

# Singleton instances
resource_optimizer = ResourceOptimizer()
graceful_degradation = GracefulDegradation(resource_optimizer)

def get_resource_optimizer() -> ResourceOptimizer:
    """Get the resource optimizer instance."""
    return resource_optimizer

def get_graceful_degradation() -> GracefulDegradation:
    """Get the graceful degradation instance."""
    return graceful_degradation
