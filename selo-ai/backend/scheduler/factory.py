"""
Scheduler Factory Module

This module provides factory methods for creating and initializing
the enhanced scheduling system components.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
import os
import pathlib

from .integration import SchedulerIntegration
from ..reflection.processor import ReflectionProcessor
from ..db.repositories.user import UserRepository
from ..db.repositories.reflection import ReflectionRepository

logger = logging.getLogger("selo.scheduler.factory")

class SchedulerFactory:
    """
    Factory for creating and initializing scheduler components.
    
    This class provides static methods for creating preconfigured
    scheduler instances with appropriate dependencies.
    """
    
    @staticmethod
    async def create_scheduler_integration(
            reflection_processor: Optional[ReflectionProcessor] = None,
            user_repository: Optional[UserRepository] = None,
            reflection_repository: Optional[ReflectionRepository] = None,
            conversation_repository: Optional[Any] = None,
            db_url: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
            event_trigger_system: Optional[Any] = None,
            agent_loop_runner: Optional[Any] = None) -> SchedulerIntegration:
        """
        Create and initialize a scheduler integration.
        
        Args:
            reflection_processor: Reflection processor instance
            user_repository: User repository instance
            reflection_repository: Reflection repository instance
            db_url: Database URL
            config: Configuration options
            
        Returns:
            Initialized scheduler integration
        """
        # Use environment variables if not provided
        if not db_url:
            db_url = os.environ.get("DATABASE_URL", None)

        if not db_url:
            default_store_dir = pathlib.Path(__file__).resolve().parents[2] / "data" / "scheduler"
            try:
                default_store_dir.mkdir(parents=True, exist_ok=True)
                default_store_path = default_store_dir / "jobs.sqlite"
                db_url = f"sqlite:///{default_store_path}"
                logger.info("Scheduler factory defaulting to persistent SQLite job store at %s", default_store_path)
            except Exception as dir_err:
                logger.warning("Failed to prepare default scheduler job store directory: %s", dir_err)

            
        # Default configuration with reasonable values
        default_config = {
            "scheduler_service": {
                # Core scheduler settings
            },
            "adaptive_scheduler": {
                "min_interval_seconds": 300,  # 5 minutes
                "max_interval_seconds": 86400 * 2,  # 2 days
                "default_interval_seconds": 86400,  # 1 day
                "adaptation_rate": 0.2,
                "activity_weight": 0.6,
                "importance_weight": 0.3,
                "resource_weight": 0.1
            },
            "event_trigger_system": {
                "max_event_history": 1000
            },
            "resource_monitor": {
                "update_interval_seconds": 60,
                "cpu_threshold": 80,
                "memory_threshold": 80,
                "disk_threshold": 90,
                "log_usage": False
            }
        }
        
        # Merge with provided config
        merged_config = default_config.copy()
        if config:
            for section, values in config.items():
                if section in merged_config:
                    merged_config[section].update(values)
                else:
                    merged_config[section] = values
        
        # Create integration
        integration = SchedulerIntegration(
            reflection_processor=reflection_processor,
            user_repository=user_repository,
            reflection_repository=reflection_repository,
            conversation_repository=conversation_repository,
            db_url=db_url,
            config=merged_config,
            event_trigger_system=event_trigger_system,
            agent_loop_runner=agent_loop_runner,
        )
        
        # Initialize components
        await integration.setup()
        
        logger.info("Scheduler integration created and initialized")
        return integration
        
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get default configuration for scheduler components.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "scheduler_service": {
                # Core scheduler settings
            },
            "adaptive_scheduler": {
                "min_interval_seconds": 300,  # 5 minutes
                "max_interval_seconds": 86400 * 2,  # 2 days
                "default_interval_seconds": 86400,  # 1 day
                "adaptation_rate": 0.2,
                "activity_weight": 0.6,
                "importance_weight": 0.3,
                "resource_weight": 0.1
            },
            "event_trigger_system": {
                "max_event_history": 1000
            },
            "resource_monitor": {
                "update_interval_seconds": 60,
                "cpu_threshold": 80,
                "memory_threshold": 80,
                "disk_threshold": 90,
                "log_usage": False
            }
        }
        
    @staticmethod
    def load_config_from_env() -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Returns:
            Configuration dictionary
        """
        config = SchedulerFactory.get_default_config()
        
        # Load scheduler service config
        if "SCHEDULER_DB_URL" in os.environ:
            config["scheduler_service"]["db_url"] = os.environ["SCHEDULER_DB_URL"]
            
        # Load adaptive scheduler config
        if "SCHEDULER_MIN_INTERVAL" in os.environ:
            config["adaptive_scheduler"]["min_interval_seconds"] = int(os.environ["SCHEDULER_MIN_INTERVAL"])
            
        if "SCHEDULER_MAX_INTERVAL" in os.environ:
            config["adaptive_scheduler"]["max_interval_seconds"] = int(os.environ["SCHEDULER_MAX_INTERVAL"])
            
        if "SCHEDULER_DEFAULT_INTERVAL" in os.environ:
            config["adaptive_scheduler"]["default_interval_seconds"] = int(os.environ["SCHEDULER_DEFAULT_INTERVAL"])
            
        # Load resource monitor config
        if "RESOURCE_MONITOR_INTERVAL" in os.environ:
            config["resource_monitor"]["update_interval_seconds"] = int(os.environ["RESOURCE_MONITOR_INTERVAL"])
            
        if "RESOURCE_CPU_THRESHOLD" in os.environ:
            config["resource_monitor"]["cpu_threshold"] = int(os.environ["RESOURCE_CPU_THRESHOLD"])
            
        if "RESOURCE_MEMORY_THRESHOLD" in os.environ:
            config["resource_monitor"]["memory_threshold"] = int(os.environ["RESOURCE_MEMORY_THRESHOLD"])
            
        if "RESOURCE_LOG_USAGE" in os.environ:
            config["resource_monitor"]["log_usage"] = os.environ["RESOURCE_LOG_USAGE"].lower() in ("true", "1", "yes")
            
        return config
