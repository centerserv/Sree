"""
Resource Throttling System for SREE
===================================

This module provides intelligent resource management to prevent system stalls
when processing large datasets (30k+ rows or 1M+ rows). It implements:

1. CPU Usage Monitoring and Throttling
2. Memory Usage Monitoring
3. Internal Processing Throttling
4. Adaptive Batch Processing
5. Process Priority Management

The system ensures that SREE analysis completes successfully even on weak CPUs
or with massive datasets by implementing hard limits and intelligent throttling.
"""

import time
import psutil
import os
import signal
import threading
from typing import Dict, Any, Optional, Callable
import logging
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class ThrottleConfig:
    """Configuration for resource throttling"""
    # CPU limits
    max_cpu_percent: float = 80.0  # Maximum CPU usage percentage
    cpu_throttle_threshold: float = 70.0  # CPU threshold to start throttling
    
    # Memory limits
    max_memory_percent: float = 85.0  # Maximum memory usage percentage
    memory_throttle_threshold: float = 75.0  # Memory threshold to start throttling
    
    # Processing throttling
    base_sleep_time: float = 0.01  # Base sleep time in seconds
    sleep_time_multiplier: float = 1.5  # Multiplier for sleep time when throttling
    max_sleep_time: float = 0.1  # Maximum sleep time
    
    # Batch processing
    batch_size: int = 1000  # Number of rows to process before checking resources
    adaptive_batch_size: bool = True  # Automatically adjust batch size based on performance
    
    # Process priority
    lower_process_priority: bool = True  # Lower process priority to prevent system freeze
    
    # Monitoring
    monitor_interval: float = 0.5  # How often to check resource usage (seconds)
    log_resource_usage: bool = True  # Log resource usage periodically

class ResourceMonitor:
    """Monitors system resources and provides throttling recommendations"""
    
    def __init__(self, config: ThrottleConfig):
        self.config = config
        self.process = psutil.Process()
        self._monitoring = False
        self._monitor_thread = None
        self._current_cpu_percent = 0.0
        self._current_memory_percent = 0.0
        self._throttle_level = 0.0  # 0.0 = no throttling, 1.0 = maximum throttling
        
    def start_monitoring(self):
        """Start background resource monitoring"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
        
    def stop_monitoring(self):
        """Stop background resource monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Resource monitoring stopped")
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                # Get current resource usage
                cpu_percent = self.process.cpu_percent()
                memory_percent = self.process.memory_percent()
                
                # Update current values
                self._current_cpu_percent = cpu_percent
                self._current_memory_percent = memory_percent
                
                # Calculate throttle level
                self._calculate_throttle_level()
                
                # Log if enabled
                if self.config.log_resource_usage and self._throttle_level > 0.1:
                    logger.info(f"Resource usage - CPU: {cpu_percent:.1f}%, "
                              f"Memory: {memory_percent:.1f}%, Throttle: {self._throttle_level:.2f}")
                
                time.sleep(self.config.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.config.monitor_interval)
    
    def _calculate_throttle_level(self):
        """Calculate current throttle level based on resource usage"""
        cpu_factor = max(0, (self._current_cpu_percent - self.config.cpu_throttle_threshold) / 
                        (self.config.max_cpu_percent - self.config.cpu_throttle_threshold))
        memory_factor = max(0, (self._current_memory_percent - self.config.memory_throttle_threshold) / 
                           (self.config.max_memory_percent - self.config.memory_throttle_threshold))
        
        self._throttle_level = min(1.0, max(cpu_factor, memory_factor))
    
    def get_throttle_level(self) -> float:
        """Get current throttle level (0.0 to 1.0)"""
        return self._throttle_level
    
    def get_sleep_time(self) -> float:
        """Get recommended sleep time based on current throttle level"""
        if self._throttle_level <= 0:
            return 0.0
            
        sleep_time = self.config.base_sleep_time * (self.config.sleep_time_multiplier ** self._throttle_level)
        return min(sleep_time, self.config.max_sleep_time)
    
    def get_current_resources(self) -> Dict[str, float]:
        """Get current resource usage"""
        return {
            'cpu_percent': self._current_cpu_percent,
            'memory_percent': self._current_memory_percent,
            'throttle_level': self._throttle_level
        }

class ResourceThrottler:
    """
    Main resource throttling system for SREE
    
    This class provides intelligent throttling to prevent system stalls
    while ensuring analysis completion even with massive datasets.
    """
    
    def __init__(self, config: Optional[ThrottleConfig] = None):
        self.config = config or ThrottleConfig()
        self.monitor = ResourceMonitor(self.config)
        self._batch_counter = 0
        self._total_processed = 0
        self._start_time = None
        
        # Set process priority if requested
        if self.config.lower_process_priority:
            self._set_process_priority()
    
    def _set_process_priority(self):
        """Lower process priority to prevent system freeze"""
        try:
            if os.name == 'posix':  # Linux/Mac
                os.nice(10)  # Lower priority
                logger.info("Process priority lowered (nice=10)")
            elif os.name == 'nt':  # Windows
                # Windows priority adjustment would go here
                pass
        except Exception as e:
            logger.warning(f"Could not set process priority: {e}")
    
    def start(self):
        """Start the throttling system"""
        self.monitor.start_monitoring()
        self._start_time = time.time()
        logger.info("Resource throttling system started")
    
    def stop(self):
        """Stop the throttling system"""
        self.monitor.stop_monitoring()
        if self._start_time:
            elapsed = time.time() - self._start_time
            logger.info(f"Resource throttling completed. Processed {self._total_processed} items in {elapsed:.2f}s")
    
    def throttle(self, items_processed: int = 1):
        """
        Apply throttling based on current resource usage
        
        Args:
            items_processed: Number of items processed since last throttle call
        """
        self._batch_counter += items_processed
        self._total_processed += items_processed
        
        # Check if we need to throttle
        if self._batch_counter >= self.config.batch_size:
            self._apply_throttling()
            self._batch_counter = 0
    
    def _apply_throttling(self):
        """Apply throttling based on current resource usage"""
        throttle_level = self.monitor.get_throttle_level()
        sleep_time = self.monitor.get_sleep_time()
        
        if sleep_time > 0:
            time.sleep(sleep_time)
            
            # Log significant throttling
            if throttle_level > 0.3:
                resources = self.monitor.get_current_resources()
                logger.info(f"Applied throttling - Level: {throttle_level:.2f}, "
                          f"Sleep: {sleep_time:.3f}s, CPU: {resources['cpu_percent']:.1f}%")
    
    def get_adaptive_batch_size(self, current_performance: float) -> int:
        """
        Get adaptive batch size based on current performance
        
        Args:
            current_performance: Current processing speed (items per second)
            
        Returns:
            Recommended batch size
        """
        if not self.config.adaptive_batch_size:
            return self.config.batch_size
        
        # Adjust batch size based on performance and resource usage
        base_size = self.config.batch_size
        throttle_level = self.monitor.get_throttle_level()
        
        if throttle_level > 0.7:
            # Heavy throttling - reduce batch size
            return max(100, int(base_size * 0.5))
        elif throttle_level > 0.3:
            # Moderate throttling - slightly reduce batch size
            return max(500, int(base_size * 0.8))
        else:
            # Low throttling - can increase batch size
            return min(2000, int(base_size * 1.2))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current throttling system status"""
        resources = self.monitor.get_current_resources()
        return {
            'total_processed': self._total_processed,
            'batch_counter': self._batch_counter,
            'throttle_level': resources['throttle_level'],
            'cpu_percent': resources['cpu_percent'],
            'memory_percent': resources['memory_percent'],
            'sleep_time': self.monitor.get_sleep_time(),
            'adaptive_batch_size': self.get_adaptive_batch_size(0.0)
        }

@contextmanager
def throttled_processing(config: Optional[ThrottleConfig] = None):
    """
    Context manager for throttled processing
    
    Usage:
        with throttled_processing() as throttler:
            for item in large_dataset:
                process_item(item)
                throttler.throttle()
    """
    throttler = ResourceThrottler(config)
    try:
        throttler.start()
        yield throttler
    finally:
        throttler.stop()

def create_throttle_config_for_dataset_size(dataset_size: int) -> ThrottleConfig:
    """
    Create optimal throttle configuration based on dataset size
    
    Args:
        dataset_size: Number of rows in the dataset
        
    Returns:
        Optimized ThrottleConfig
    """
    if dataset_size <= 50:
        # Very small datasets (50 rows or less) - NO throttling at all
        return ThrottleConfig(
            max_cpu_percent=100.0,  # Use all CPU
            cpu_throttle_threshold=95.0,
            max_memory_percent=100.0,
            memory_throttle_threshold=95.0,
            base_sleep_time=0.0,  # No sleep at all
            sleep_time_multiplier=1.0,
            max_sleep_time=0.0,
            batch_size=50,  # Process all rows at once
            adaptive_batch_size=False,  # No adaptation
            lower_process_priority=False,  # Don't lower priority
            monitor_interval=5.0,  # Minimal monitoring
            log_resource_usage=False  # No logging
        )
    elif dataset_size <= 100:
        # Small datasets (100 rows or less) - NO throttling for maximum speed
        return ThrottleConfig(
            max_cpu_percent=95.0,
            cpu_throttle_threshold=90.0,
            max_memory_percent=95.0,
            memory_throttle_threshold=90.0,
            base_sleep_time=0.0,  # No sleep for maximum speed
            sleep_time_multiplier=1.0,
            max_sleep_time=0.0,
            batch_size=100,  # Process all rows at once
            adaptive_batch_size=False,  # No adaptation needed
            lower_process_priority=False,  # Don't lower priority
            monitor_interval=2.0,  # Less frequent monitoring
            log_resource_usage=False  # No logging for speed
        )
    elif dataset_size <= 1000:
        # Small dataset - minimal throttling
        return ThrottleConfig(
            max_cpu_percent=90.0,
            cpu_throttle_threshold=85.0,
            max_memory_percent=90.0,
            memory_throttle_threshold=85.0,
            base_sleep_time=0.001,  # Very short sleep
            sleep_time_multiplier=1.2,
            max_sleep_time=0.01,
            batch_size=500,
            adaptive_batch_size=True,
            lower_process_priority=False,  # Don't lower priority for small datasets
            monitor_interval=1.0,
            log_resource_usage=False
        )
    elif dataset_size <= 10000:
        # Medium dataset - moderate throttling
        return ThrottleConfig(
            max_cpu_percent=85.0,
            cpu_throttle_threshold=75.0,
            max_memory_percent=85.0,
            memory_throttle_threshold=75.0,
            base_sleep_time=0.005,
            sleep_time_multiplier=1.3,
            max_sleep_time=0.05,
            batch_size=1000,
            adaptive_batch_size=True,
            lower_process_priority=True,
            monitor_interval=0.5,
            log_resource_usage=True
        )
    elif dataset_size <= 100000:
        # Large dataset - aggressive throttling
        return ThrottleConfig(
            max_cpu_percent=80.0,
            cpu_throttle_threshold=70.0,
            max_memory_percent=80.0,
            memory_throttle_threshold=70.0,
            base_sleep_time=0.01,
            sleep_time_multiplier=1.5,
            max_sleep_time=0.1,
            batch_size=2000,
            adaptive_batch_size=True,
            lower_process_priority=True,
            monitor_interval=0.5,
            log_resource_usage=True
        )
    else:
        # Massive dataset - very aggressive throttling
        return ThrottleConfig(
            max_cpu_percent=75.0,
            cpu_throttle_threshold=65.0,
            max_memory_percent=75.0,
            memory_throttle_threshold=65.0,
            base_sleep_time=0.02,
            sleep_time_multiplier=2.0,
            max_sleep_time=0.2,
            batch_size=5000,
            adaptive_batch_size=True,
            lower_process_priority=True,
            monitor_interval=0.3,
            log_resource_usage=True
        )

# Convenience functions for easy integration
def throttle_iteration(iteration: int, items_processed: int = 1, 
                      config: Optional[ThrottleConfig] = None):
    """
    Convenience function to throttle during iterations
    
    Args:
        iteration: Current iteration number
        items_processed: Number of items processed
        config: Throttle configuration
    """
    if not hasattr(throttle_iteration, '_throttler'):
        throttle_iteration._throttler = ResourceThrottler(config)
        throttle_iteration._throttler.start()
    
    throttle_iteration._throttler.throttle(items_processed)

def get_resource_status() -> Dict[str, Any]:
    """Get current resource status"""
    if hasattr(throttle_iteration, '_throttler'):
        return throttle_iteration._throttler.get_status()
    return {'error': 'Throttler not initialized'}

def cleanup_throttler():
    """Clean up the global throttler instance"""
    if hasattr(throttle_iteration, '_throttler'):
        throttle_iteration._throttler.stop()
        delattr(throttle_iteration, '_throttler') 