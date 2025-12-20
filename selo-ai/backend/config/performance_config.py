"""
Performance Configuration for SELO AI

Centralized performance optimization settings including caching,
streaming, model selection, and hardware utilization.
"""

import os
from typing import Dict, Any
import logging

logger = logging.getLogger("selo.config.performance")

class PerformanceConfig:
    """Performance optimization configuration."""
    
    def __init__(self):
        # Response caching configuration
        self.cache_enabled = os.getenv("RESPONSE_CACHE_ENABLED", "true").lower() == "true"
        self.cache_max_entries = int(os.getenv("CACHE_MAX_ENTRIES", "1000"))
        self.cache_ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
        self.cache_similarity_threshold = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85"))
        
        # Streaming configuration
        self.streaming_enabled = os.getenv("STREAMING_ENABLED", "true").lower() == "true"
        self.streaming_chunk_size = int(os.getenv("STREAMING_CHUNK_SIZE", "50"))
        self.streaming_flush_interval = float(os.getenv("STREAMING_FLUSH_INTERVAL", "0.1"))
        
        # Model optimization configuration
        self.smart_model_selection = os.getenv("SMART_MODEL_SELECTION", "true").lower() == "true"
        self.prioritize_speed = os.getenv("PRIORITIZE_SPEED", "false").lower() == "true"
        self.max_response_time = int(os.getenv("MAX_RESPONSE_TIME_SECONDS", "120"))
        self.min_quality_threshold = float(os.getenv("MIN_QUALITY_THRESHOLD", "0.7"))
        
        # Hardware optimization
        self.gpu_memory_fraction = float(os.getenv("GPU_MEMORY_FRACTION", "0.8"))
        self.cpu_threads = int(os.getenv("CPU_THREADS", "12"))
        self.parallel_requests = int(os.getenv("PARALLEL_REQUESTS", "2"))
        
        # Performance monitoring
        self.performance_logging = os.getenv("PERFORMANCE_LOGGING", "true").lower() == "true"
        self.metrics_collection = os.getenv("METRICS_COLLECTION", "true").lower() == "true"
        
        logger.info("Performance configuration loaded")
        if self.cache_enabled:
            logger.info(f"Response caching enabled: {self.cache_max_entries} entries, {self.cache_ttl_seconds}s TTL")
        if self.streaming_enabled:
            logger.info(f"Streaming responses enabled: {self.streaming_chunk_size} char chunks")
        if self.smart_model_selection:
            logger.info("Smart model selection enabled")
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get caching configuration."""
        return {
            "enabled": self.cache_enabled,
            "max_entries": self.cache_max_entries,
            "ttl_seconds": self.cache_ttl_seconds,
            "similarity_threshold": self.cache_similarity_threshold
        }
    
    def get_streaming_config(self) -> Dict[str, Any]:
        """Get streaming configuration."""
        return {
            "enabled": self.streaming_enabled,
            "chunk_size": self.streaming_chunk_size,
            "flush_interval": self.streaming_flush_interval
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model optimization configuration."""
        return {
            "smart_selection": self.smart_model_selection,
            "prioritize_speed": self.prioritize_speed,
            "max_response_time": self.max_response_time,
            "min_quality_threshold": self.min_quality_threshold
        }
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware optimization configuration."""
        return {
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "cpu_threads": self.cpu_threads,
            "parallel_requests": self.parallel_requests
        }

# Global performance config instance
_performance_config = None

def get_performance_config() -> PerformanceConfig:
    """Get global performance configuration instance."""
    global _performance_config
    if _performance_config is None:
        _performance_config = PerformanceConfig()
    return _performance_config
