"""
SELO AI Enhanced Scheduler Package

This package provides a comprehensive scheduling system for SELO AI with:
- Core scheduler service based on APScheduler
- Adaptive scheduling based on usage patterns and importance
- Event-driven triggers for real-time responsiveness
- Resource monitoring for optimal system performance
"""

from .scheduler_service import SchedulerService
from .adaptive_scheduler import AdaptiveScheduler
from .event_triggers import EventTriggerSystem, EventType
from .resource_monitor import ResourceMonitor

__all__ = [
    'SchedulerService',
    'AdaptiveScheduler', 
    'EventTriggerSystem',
    'EventType',
    'ResourceMonitor'
]
