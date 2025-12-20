"""
Graceful Degradation Patterns for SELO AI

Provides fallback mechanisms and degraded functionality when components fail.
Ensures system remains partially functional even during component failures.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps
import json

from ..utils.datetime import utc_now, isoformat_utc_now

logger = logging.getLogger("selo.graceful_degradation")

class DegradationLevel:
    """Defines different levels of service degradation."""
    FULL = "full"           # All features available
    REDUCED = "reduced"     # Some features disabled
    MINIMAL = "minimal"     # Only core features
    EMERGENCY = "emergency" # Basic functionality only

class FallbackRegistry:
    """Registry for fallback functions and degraded services."""
    
    def __init__(self):
        self.fallbacks: Dict[str, Dict[str, Callable]] = {}
        self.degradation_level = DegradationLevel.FULL
        
    def register_fallback(self, service: str, level: str, fallback_func: Callable):
        """Register a fallback function for a service at a specific degradation level."""
        if service not in self.fallbacks:
            self.fallbacks[service] = {}
        self.fallbacks[service][level] = fallback_func
        logger.info(f"🔄 Registered fallback for {service} at {level} level")
        
    def get_fallback(self, service: str, level: str = None) -> Optional[Callable]:
        """Get fallback function for service at specified or current degradation level."""
        if service not in self.fallbacks:
            return None
            
        target_level = level or self.degradation_level
        return self.fallbacks[service].get(target_level)
        
    def set_degradation_level(self, level: str):
        """Set system-wide degradation level."""
        self.degradation_level = level
        logger.warning(f"🔻 System degradation level set to: {level}")

# Global fallback registry
fallback_registry = FallbackRegistry()

# Idempotence guard to prevent duplicate fallback registration and log noise
_FALLBACKS_REGISTERED = False

def with_fallback(service_name: str, fallback_level: str = DegradationLevel.REDUCED):
    """
    Decorator to add fallback functionality to service methods.
    
    Args:
        service_name: Name of the service
        fallback_level: Degradation level to use for fallbacks
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                # Try primary function
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                logger.warning(f"⚠️ Primary function {func.__name__} failed: {e}")
                
                # Try fallback
                fallback = fallback_registry.get_fallback(service_name, fallback_level)
                if fallback:
                    logger.info(f"🔄 Using fallback for {service_name}")
                    try:
                        if asyncio.iscoroutinefunction(fallback):
                            return await fallback(*args, **kwargs)
                        else:
                            return fallback(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"❌ Fallback also failed: {fallback_error}")
                        raise e  # Raise original error
                else:
                    logger.error(f"❌ No fallback available for {service_name}")
                    raise e
                    
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"⚠️ Primary function {func.__name__} failed: {e}")
                
                fallback = fallback_registry.get_fallback(service_name, fallback_level)
                if fallback:
                    logger.info(f"🔄 Using fallback for {service_name}")
                    try:
                        return fallback(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"❌ Fallback also failed: {fallback_error}")
                        raise e
                else:
                    logger.error(f"❌ No fallback available for {service_name}")
                    raise e
                    
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator

# LLM Service Fallbacks
async def llm_simple_fallback(*args, **kwargs) -> Dict[str, Any]:
    """Simple fallback for LLM operations when service is unavailable."""
    return {
        "response": "I'm currently experiencing technical difficulties. Please try again later.",
        "model": "fallback",
        "status": "degraded",
        "timestamp": isoformat_utc_now()
    }

async def llm_cached_fallback(*args, **kwargs) -> Dict[str, Any]:
    """Cached response fallback for LLM operations."""
    # In a real implementation, this would check a cache of previous responses
    return {
        "response": "Based on previous interactions, here's a general response to your query.",
        "model": "cached",
        "status": "cached_fallback",
        "timestamp": isoformat_utc_now()
    }

def llm_template_fallback(*args, **kwargs) -> Dict[str, Any]:
    """Template-based fallback for LLM operations."""
    templates = {
        "greeting": "Hello! How can I help you today?",
        "error": "I apologize, but I'm having trouble processing your request right now.",
        "general": "Thank you for your message. I'll do my best to help once my systems are fully operational."
    }
    
    # Simple keyword matching for template selection
    prompt = kwargs.get('prompt', '') or (args[0] if args else '')
    
    if any(word in prompt.lower() for word in ['hello', 'hi', 'hey']):
        response = templates["greeting"]
    elif any(word in prompt.lower() for word in ['error', 'problem', 'issue']):
        response = templates["error"]
    else:
        response = templates["general"]
        
    return {
        "response": response,
        "model": "template",
        "status": "template_fallback",
        "timestamp": isoformat_utc_now()
    }

# Database Fallbacks
async def database_memory_fallback(*args, **kwargs) -> List[Dict[str, Any]]:
    """In-memory fallback for database operations."""
    logger.warning("🔄 Using in-memory database fallback")
    
    # Return empty results with degraded status
    return []

def database_readonly_fallback(*args, **kwargs) -> Dict[str, Any]:
    """Read-only fallback for database write operations."""
    logger.warning("🔄 Database in read-only mode, write operation blocked")
    
    return {
        "status": "blocked",
        "message": "Database is in read-only mode due to system issues",
        "timestamp": isoformat_utc_now()
    }

# Reflection System Fallbacks
async def reflection_simple_fallback(*args, **kwargs) -> Dict[str, Any]:
    """Simple fallback for reflection processing."""
    return {
        "reflection_id": f"fallback_{int(utc_now().timestamp())}",
        "content": "Reflection processing is currently unavailable. Your thoughts have been noted.",
        "type": "fallback",
        "status": "degraded",
        "timestamp": isoformat_utc_now()
    }

def reflection_queue_fallback(*args, **kwargs) -> Dict[str, Any]:
    """Queue-based fallback for reflection processing."""
    # In a real implementation, this would queue the reflection for later processing
    logger.info("🔄 Queueing reflection for later processing")
    
    return {
        "status": "queued",
        "message": "Reflection queued for processing when system recovers",
        "queue_position": 1,  # Simplified
        "timestamp": isoformat_utc_now()
    }

# Vector Store Fallbacks
async def vector_store_simple_search_fallback(*args, **kwargs) -> List[Dict[str, Any]]:
    """Simple text matching fallback for vector search."""
    query = kwargs.get('query', '') or (args[0] if args else '')
    
    # Return empty results with degraded status
    return [{
        "content": "Vector search is currently unavailable. Please try a different search method.",
        "score": 0.0,
        "metadata": {"status": "fallback", "query": query}
    }]

def vector_store_keyword_fallback(*args, **kwargs) -> List[Dict[str, Any]]:
    """Keyword-based fallback for vector operations."""
    # Simplified keyword matching
    return []

# Health Check Fallbacks
def health_basic_fallback(*args, **kwargs) -> Dict[str, Any]:
    """Basic health status when monitoring is unavailable."""
    return {
        "status": "unknown",
        "message": "Health monitoring is currently unavailable",
        "timestamp": isoformat_utc_now(),
        "degraded": True
    }

# Register all fallbacks
def register_all_fallbacks():
    """Register all predefined fallback functions.

    Safe to call multiple times; subsequent calls are no-ops to avoid duplicate
    registration and log noise.
    """

    global _FALLBACKS_REGISTERED
    if _FALLBACKS_REGISTERED:
        logger.debug("🔄 Fallback functions already registered; skipping re-registration")
        return

    # LLM Service fallbacks
    fallback_registry.register_fallback("llm", DegradationLevel.REDUCED, llm_cached_fallback)
    fallback_registry.register_fallback("llm", DegradationLevel.MINIMAL, llm_template_fallback)
    fallback_registry.register_fallback("llm", DegradationLevel.EMERGENCY, llm_simple_fallback)
    
    # Database fallbacks
    fallback_registry.register_fallback("database_read", DegradationLevel.REDUCED, database_memory_fallback)
    fallback_registry.register_fallback("database_write", DegradationLevel.REDUCED, database_readonly_fallback)
    fallback_registry.register_fallback("database_write", DegradationLevel.MINIMAL, database_readonly_fallback)
    
    # Reflection system fallbacks
    fallback_registry.register_fallback("reflection", DegradationLevel.REDUCED, reflection_queue_fallback)
    fallback_registry.register_fallback("reflection", DegradationLevel.MINIMAL, reflection_simple_fallback)
    
    # Vector store fallbacks
    fallback_registry.register_fallback("vector_search", DegradationLevel.REDUCED, vector_store_keyword_fallback)
    fallback_registry.register_fallback("vector_search", DegradationLevel.MINIMAL, vector_store_simple_search_fallback)
    
    # Health monitoring fallbacks
    fallback_registry.register_fallback("health", DegradationLevel.REDUCED, health_basic_fallback)

    logger.info("🔄 All fallback functions registered")
    _FALLBACKS_REGISTERED = True

# Degradation Manager
class DegradationManager:
    """Manages system degradation levels and automatic fallback activation."""
    
    def __init__(self):
        self.current_level = DegradationLevel.FULL
        self.degradation_history = []
        self.component_health = {}
        
    def assess_system_health(self, health_data: Dict[str, Any]):
        """Assess system health and adjust degradation level accordingly."""
        critical_components = ['database', 'llm_service']
        warning_components = ['vector_store', 'reflection_processor']
        
        critical_failures = 0
        warning_failures = 0
        
        for component, status in health_data.get('checks', {}).items():
            if component in critical_components and status.get('status') == 'critical':
                critical_failures += 1
            elif component in warning_components and status.get('status') in ['critical', 'warning']:
                warning_failures += 1
                
        # Determine appropriate degradation level
        if critical_failures >= 2:
            new_level = DegradationLevel.EMERGENCY
        elif critical_failures >= 1:
            new_level = DegradationLevel.MINIMAL
        elif warning_failures >= 2:
            new_level = DegradationLevel.REDUCED
        else:
            new_level = DegradationLevel.FULL
            
        if new_level != self.current_level:
            self.set_degradation_level(new_level)
            
    def set_degradation_level(self, level: str):
        """Set system degradation level and log the change."""
        old_level = self.current_level
        self.current_level = level
        fallback_registry.set_degradation_level(level)
        
        self.degradation_history.append({
            "from": old_level,
            "to": level,
            "timestamp": isoformat_utc_now()
        })
        
        logger.warning(f"🔻 Degradation level changed: {old_level} → {level}")
        
    def get_status(self) -> Dict[str, Any]:
        """Get current degradation status."""
        return {
            "current_level": self.current_level,
            "available_services": self._get_available_services(),
            "degradation_history": self.degradation_history[-10:],  # Last 10 changes
            "timestamp": isoformat_utc_now()
        }
        
    def _get_available_services(self) -> Dict[str, str]:
        """Get list of available services at current degradation level."""
        services = {
            DegradationLevel.FULL: {
                "chat": "full",
                "reflection": "full", 
                "vector_search": "full",
                "analytics": "full"
            },
            DegradationLevel.REDUCED: {
                "chat": "cached_responses",
                "reflection": "queued",
                "vector_search": "keyword_based",
                "analytics": "disabled"
            },
            DegradationLevel.MINIMAL: {
                "chat": "template_responses",
                "reflection": "basic_logging",
                "vector_search": "disabled",
                "analytics": "disabled"
            },
            DegradationLevel.EMERGENCY: {
                "chat": "error_messages_only",
                "reflection": "disabled",
                "vector_search": "disabled", 
                "analytics": "disabled"
            }
        }
        
        return services.get(self.current_level, {})

# Global degradation manager
degradation_manager = DegradationManager()

# Initialize fallbacks
register_all_fallbacks()

# Convenience functions
def get_degradation_status() -> Dict[str, Any]:
    """Get current system degradation status."""
    return degradation_manager.get_status()

def set_degradation_level(level: str):
    """Set system degradation level."""
    degradation_manager.set_degradation_level(level)

def assess_and_adjust_degradation(health_data: Dict[str, Any]):
    """Assess health data and adjust degradation level automatically."""
    degradation_manager.assess_system_health(health_data)
