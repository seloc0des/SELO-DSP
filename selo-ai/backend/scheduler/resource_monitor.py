"""
Resource Monitor Module

This module monitors system resources (CPU, memory, disk) and provides
this information to the scheduler for resource-aware decision making.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from datetime import timedelta

from ..utils.datetime import utc_now
import time
import psutil

logger = logging.getLogger("selo.scheduler.resources")

# Minimum time difference to avoid noise in I/O speed calculations
MIN_TIME_DIFF_SECONDS = 0.1  # 100ms minimum

# Tier-aware default intervals (seconds) - 2-tier system
# Standard tier (<12GB GPU): Less frequent monitoring to reduce overhead
# High tier (>=12GB GPU): More frequent monitoring for responsiveness
TIER_UPDATE_INTERVALS = {
    "standard": 90,  # 90 seconds for standard tier (<12GB GPU)
    "high": 45,      # 45 seconds for high-tier (>=12GB GPU)
}

def _detect_system_tier() -> str:
    """Detect system tier for appropriate resource monitoring interval."""
    try:
        from ..utils.system_profile import detect_system_profile
        profile = detect_system_profile()
        return profile.get("tier", "standard")
    except Exception:
        return "standard"


class ResourceMonitor:
    """
    System resource monitoring for resource-aware scheduling.
    
    This class monitors CPU, memory, and disk usage and provides
    this information to the scheduler to make resource-aware decisions.
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 update_callback: Optional[Callable] = None):
        """
        Initialize the resource monitor.
        
        Args:
            config: Configuration options
            update_callback: Callback function for resource updates
        """
        self.config = config or {}
        self.update_callback = update_callback
        
        # Detect system tier for appropriate default intervals
        self._system_tier = _detect_system_tier()
        tier_default_interval = TIER_UPDATE_INTERVALS.get(self._system_tier, 60)
        
        # Configuration - use tier-aware default if not explicitly configured
        self.base_update_interval = self.config.get("update_interval_seconds", tier_default_interval)
        self.update_interval = self.base_update_interval  # Current interval (may be adaptive)
        self.cpu_threshold = self.config.get("cpu_threshold", 80)  # percent
        self.memory_threshold = self.config.get("memory_threshold", 80)  # percent
        self.disk_threshold = self.config.get("disk_threshold", 90)  # percent
        
        # Adaptive monitoring settings - 2-tier system (standard/high)
        self.adaptive_enabled = self.config.get("adaptive_monitoring", True)
        if self._system_tier == "high":
            # High-tier: more responsive monitoring
            self.min_update_interval = self.config.get("min_update_interval_seconds", 15)
            self.max_update_interval = self.config.get("max_update_interval_seconds", 90)
        else:  # standard (<12GB GPU)
            # Standard tier: less frequent to reduce overhead, won't interfere with user tasks
            self.min_update_interval = self.config.get("min_update_interval_seconds", 30)
            self.max_update_interval = self.config.get("max_update_interval_seconds", 180)
        
        # Predictive throttling settings
        self.prediction_enabled = self.config.get("predictive_throttling", True)
        self.prediction_window_samples = 10  # Number of samples for trend analysis
        
        # Resource history
        self.history = {
            "cpu": [],
            "memory": [],
            "disk": [],
            "io": []
        }
        
        self.history_max_items = self.config.get("history_max_items", 1000)
        
        # Current values
        self.current = {
            "cpu": 0,
            "memory": 0,
            "disk": 0,
            "io": {
                "read_bytes": 0,
                "write_bytes": 0
            },
            "timestamp": None
        }
        
        # Monitor task
        self.monitor_task = None
        self.is_running = False
        
        # Callbacks for resource state changes
        self._resource_available_callbacks: list = []
        self._resource_constrained_callbacks: list = []
        self._was_constrained: bool = False
        
        logger.info(f"Resource monitor initialized (tier={self._system_tier}, interval={self.base_update_interval}s, adaptive={self.adaptive_enabled}, predictive={self.prediction_enabled})")
        
    async def start(self):
        """Start the resource monitor."""
        if self.is_running:
            logger.warning("Resource monitor is already running")
            return
            
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Resource monitor started")
        
    async def stop(self):
        """Stop the resource monitor."""
        if not self.is_running:
            logger.warning("Resource monitor is not running")
            return
            
        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            self.monitor_task = None
            
        logger.info("Resource monitor stopped")
        
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        logger.info(f"Resource monitoring loop started with {self.update_interval}s interval")
        
        # Initialize IO counters
        try:
            io_counters = psutil.disk_io_counters()
            last_read_bytes = io_counters.read_bytes if io_counters else 0
            last_write_bytes = io_counters.write_bytes if io_counters else 0
            last_io_time = time.time()
        except Exception as e:
            logger.error(f"Error reading IO counters: {str(e)}")
            last_read_bytes = 0
            last_write_bytes = 0
            last_io_time = time.time()
        
        while self.is_running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                
                # IO usage
                try:
                    io_counters = psutil.disk_io_counters()
                    current_time = time.time()
                    
                    if io_counters:
                        time_diff = current_time - last_io_time
                        read_bytes = io_counters.read_bytes
                        write_bytes = io_counters.write_bytes
                        
                        # Use minimum time threshold to avoid noise and artificially inflated speeds
                        if time_diff >= MIN_TIME_DIFF_SECONDS:
                            read_speed = (read_bytes - last_read_bytes) / time_diff
                            write_speed = (write_bytes - last_write_bytes) / time_diff
                        else:
                            # Time difference too small, use zero to avoid noise
                            read_speed = 0
                            write_speed = 0
                        
                        last_read_bytes = read_bytes
                        last_write_bytes = write_bytes
                        last_io_time = current_time
                        
                        io_usage = {
                            "read_bytes_per_sec": read_speed,
                            "write_bytes_per_sec": write_speed
                        }
                    else:
                        io_usage = {
                            "read_bytes_per_sec": 0,
                            "write_bytes_per_sec": 0
                        }
                except Exception as e:
                    logger.error(f"Error calculating IO usage: {str(e)}")
                    io_usage = {
                        "read_bytes_per_sec": 0,
                        "write_bytes_per_sec": 0
                    }
                
                # Update current values
                now = utc_now()
                self.current = {
                    "cpu": cpu_percent,
                    "memory": memory_percent,
                    "disk": disk_percent,
                    "io": io_usage,
                    "timestamp": now
                }
                
                # Add to history
                self.history["cpu"].append({"value": cpu_percent, "timestamp": now})
                self.history["memory"].append({"value": memory_percent, "timestamp": now})
                self.history["disk"].append({"value": disk_percent, "timestamp": now})
                self.history["io"].append({"value": io_usage, "timestamp": now})
                
                # Trim history if needed
                for key in self.history:
                    if len(self.history[key]) > self.history_max_items:
                        self.history[key] = self.history[key][-self.history_max_items:]
                        
                # Check for threshold breaches
                cpu_breach = cpu_percent >= self.cpu_threshold
                memory_breach = memory_percent >= self.memory_threshold
                disk_breach = disk_percent >= self.disk_threshold
                
                if cpu_breach:
                    logger.warning(f"CPU usage above threshold: {cpu_percent}% >= {self.cpu_threshold}%")
                    
                if memory_breach:
                    logger.warning(f"Memory usage above threshold: {memory_percent}% >= {self.memory_threshold}%")
                    
                if disk_breach:
                    logger.warning(f"Disk usage above threshold: {disk_percent}% >= {self.disk_threshold}%")
                
                # Call update callback if provided
                if self.update_callback:
                    try:
                        await self.update_callback(self.current)
                    except Exception as e:
                        logger.error(f"Error in update callback: {str(e)}", exc_info=True)
                
                # Log resource usage periodically
                if self.config.get("log_usage", False):
                    logger.info(f"Resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, "
                              f"Disk {disk_percent:.1f}%")
                
                # Check for resource state transitions and notify callbacks
                is_constrained = self.is_resource_constrained()
                if is_constrained != self._was_constrained:
                    self._was_constrained = is_constrained
                    if is_constrained:
                        await self._notify_constrained()
                    else:
                        await self._notify_available()
                
                # Adjust monitoring interval adaptively
                if self.adaptive_enabled:
                    self._adjust_monitoring_interval(cpu_percent, memory_percent)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}", exc_info=True)
                
            # Wait for next update (using current adaptive interval)
            try:
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
                
        logger.info("Resource monitoring loop stopped")
        
    def get_current_usage(self) -> Dict[str, Any]:
        """
        Get current resource usage.
        
        Returns:
            Dict with current resource usage
        """
        return self.current
        
    def get_history(self, 
                  resource_type: str, 
                  minutes: Optional[int] = None) -> list:
        """
        Get resource usage history.
        
        Args:
            resource_type: Type of resource ('cpu', 'memory', 'disk', 'io')
            minutes: Number of minutes to look back, or None for all
            
        Returns:
            List of usage data points
        """
        if resource_type not in self.history:
            return []
            
        if minutes is None:
            return self.history[resource_type]
            
        # Filter by time window
        cutoff = utc_now() - timedelta(minutes=minutes)
        return [
            point for point in self.history[resource_type]
            if point["timestamp"] >= cutoff
        ]
        
    def is_resource_constrained(self) -> bool:
        """
        Check if the system is currently resource constrained.
        
        Returns:
            True if any resource is above threshold
        """
        return (
            self.current["cpu"] >= self.cpu_threshold or
            self.current["memory"] >= self.memory_threshold or
            self.current["disk"] >= self.disk_threshold
        )
    
    def predict_resource_exhaustion(self, lookahead_minutes: int = 5) -> bool:
        """
        Predict if resources will be exhausted in the near future.
        Uses trend analysis on recent history.
        
        Args:
            lookahead_minutes: How far ahead to predict
            
        Returns:
            True if resources are predicted to exceed thresholds
        """
        if not self.prediction_enabled:
            return False
        
        # Need enough samples for prediction
        cpu_history = self.history.get("cpu", [])
        mem_history = self.history.get("memory", [])
        
        if len(cpu_history) < self.prediction_window_samples:
            return False
        
        # Analyze CPU trend
        recent_cpu = [p["value"] for p in cpu_history[-self.prediction_window_samples:]]
        cpu_trend = self._calculate_trend(recent_cpu)
        predicted_cpu = recent_cpu[-1] + (cpu_trend * lookahead_minutes)
        
        # Analyze memory trend
        recent_mem = [p["value"] for p in mem_history[-self.prediction_window_samples:]]
        mem_trend = self._calculate_trend(recent_mem)
        predicted_mem = recent_mem[-1] + (mem_trend * lookahead_minutes)
        
        # Predict constraint at 90% of threshold
        cpu_warning = predicted_cpu >= (self.cpu_threshold * 0.9)
        mem_warning = predicted_mem >= (self.memory_threshold * 0.9)
        
        if cpu_warning or mem_warning:
            logger.debug(
                f"Predicted resource constraint in {lookahead_minutes}min: "
                f"CPU={predicted_cpu:.1f}% (trend={cpu_trend:.2f}/min), "
                f"Memory={predicted_mem:.1f}% (trend={mem_trend:.2f}/min)"
            )
        
        return cpu_warning or mem_warning
    
    def should_defer_task(self, importance: float = 0.5) -> bool:
        """
        Determine if a task should be deferred based on resource state.
        
        Args:
            importance: Task importance (0-1). Higher importance = less likely to defer
            
        Returns:
            True if task should be deferred
        """
        # Always defer if currently constrained
        if self.is_resource_constrained():
            return True
        
        # For low-importance tasks, also check predictive throttling
        if importance < 0.7 and self.predict_resource_exhaustion(lookahead_minutes=5):
            logger.debug(f"Deferring low-importance task (importance={importance}) due to predicted resource constraint")
            return True
        
        return False
    
    def _calculate_trend(self, values: list) -> float:
        """
        Calculate the trend (rate of change) in a series of values.
        Uses simple linear regression.
        
        Args:
            values: List of numeric values
            
        Returns:
            Trend per minute (positive = increasing, negative = decreasing)
        """
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        # Slope per sample, convert to per minute based on update interval
        slope = numerator / denominator
        samples_per_minute = 60 / self.update_interval
        return slope * samples_per_minute
    
    def _adjust_monitoring_interval(self, cpu_percent: float, memory_percent: float) -> None:
        """
        Adjust monitoring interval based on resource usage.
        Monitor more frequently under high load, less frequently when idle.
        """
        # Calculate load factor (0-1)
        cpu_load = cpu_percent / 100
        mem_load = memory_percent / 100
        max_load = max(cpu_load, mem_load)
        
        if max_load > 0.8:  # High load
            new_interval = self.min_update_interval
        elif max_load > 0.6:  # Moderate load
            new_interval = self.base_update_interval // 2
        elif max_load < 0.2:  # Very low load
            new_interval = min(self.max_update_interval, self.base_update_interval * 2)
        else:  # Normal load
            new_interval = self.base_update_interval
        
        # Clamp to bounds
        new_interval = max(self.min_update_interval, min(new_interval, self.max_update_interval))
        
        # Only log if significantly different
        if abs(new_interval - self.update_interval) >= 5:
            logger.debug(f"Adjusting monitoring interval: {self.update_interval}s -> {new_interval}s (load={max_load:.2f})")
            self.update_interval = new_interval
    
    def register_available_callback(self, callback) -> None:
        """
        Register a callback to be called when resources become available.
        
        Args:
            callback: Async function to call when resources are available
        """
        self._resource_available_callbacks.append(callback)
    
    def register_constrained_callback(self, callback) -> None:
        """
        Register a callback to be called when resources become constrained.
        
        Args:
            callback: Async function to call when resources are constrained
        """
        self._resource_constrained_callbacks.append(callback)
    
    async def _notify_available(self) -> None:
        """Notify all registered callbacks that resources are now available."""
        logger.info("Resources now available - notifying registered callbacks")
        for callback in self._resource_available_callbacks:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in resource available callback: {e}", exc_info=True)
    
    async def _notify_constrained(self) -> None:
        """Notify all registered callbacks that resources are constrained."""
        logger.warning("Resources now constrained - notifying registered callbacks")
        for callback in self._resource_constrained_callbacks:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in resource constrained callback: {e}", exc_info=True)
