"""
Configuration Package

Centralized configuration management for SELO AI application.

This module provides centralized access to configuration constants that are
frequently used throughout the application. These constants are initialized
once from the singleton configuration instances.
"""

from .app_config import get_app_config, AppConfig
from .reflection_config import get_reflection_config, ReflectionConfig

# Initialize reflection config singleton once to get constants
_reflection_config = get_reflection_config()

# Export commonly-used reflection configuration constants
# These are the ONLY authoritative source for these values
REFLECTION_WORD_MIN = _reflection_config.word_count_min
REFLECTION_WORD_MAX = _reflection_config.word_count_max

__all__ = [
    'get_app_config',
    'AppConfig', 
    'get_reflection_config',
    'ReflectionConfig',
    'REFLECTION_WORD_MIN',
    'REFLECTION_WORD_MAX'
]
