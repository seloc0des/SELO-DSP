"""
Reflection Configuration Module

Centralized configuration for reflection processor settings including
model mappings, thresholds, and processing parameters.
"""

import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("selo.config.reflection")


class ReflectionConfig:
    """
    Configuration manager for reflection processor settings.
    
    This class uses the singleton pattern. Do not instantiate directly.
    Use get_reflection_config() to access the singleton instance.
    """
    
    _initialized = False
    _allow_init = False  # Internal flag to allow singleton creation
    
    def __init__(self):
        """
        Initialize configuration with environment variables and defaults.
        
        Raises:
            RuntimeError: If called directly instead of via get_reflection_config()
        """
        if not ReflectionConfig._allow_init and ReflectionConfig._initialized:
            raise RuntimeError(
                "ReflectionConfig is a singleton. "
                "Use get_reflection_config() to access the instance instead of "
                "instantiating directly."
            )
        ReflectionConfig._initialized = True
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables with fallbacks."""
        
        # Get the base reflection model from environment (respects profile selection)
        # Fall back to REFLECTION_LLM which should be set by the installer based on profile
        # Default to qwen2.5:3b for proper word count generation when not configured
        base_reflection_model = os.getenv("REFLECTION_LLM", "qwen2.5:3b")
        
        # Model mappings for different reflection types
        # Use profile-specific REFLECTION_LLM as the default for all reflection types
        self.model_mappings = {
            "daily": os.getenv("REFLECTION_MODEL_DAILY", base_reflection_model),
            "weekly": os.getenv("REFLECTION_MODEL_WEEKLY", base_reflection_model),
            "emotional": os.getenv("REFLECTION_MODEL_EMOTIONAL", base_reflection_model),
            "manifesto": os.getenv("REFLECTION_MODEL_MANIFESTO", base_reflection_model),
            "message": os.getenv("REFLECTION_MODEL_MESSAGE", base_reflection_model),
            "periodic": os.getenv("REFLECTION_MODEL_PERIODIC", base_reflection_model),
            "default": os.getenv("REFLECTION_MODEL_DEFAULT", base_reflection_model)
        }
        
        # Processing thresholds and limits
        self.max_context_items = int(os.getenv("REFLECTION_MAX_CONTEXT_ITEMS", "10"))
        self.memory_importance_threshold = float(os.getenv("REFLECTION_MEMORY_THRESHOLD", "3.0"))
        self.max_memories_per_reflection = int(os.getenv("REFLECTION_MAX_MEMORIES", "10"))
        self.max_conversation_messages = int(os.getenv("REFLECTION_MAX_MESSAGES", "20"))
        
        # Persona evolution settings
        self.trait_adjustment_max = float(os.getenv("REFLECTION_TRAIT_MAX_DELTA", "0.2"))
        self.trait_adjustment_min = float(os.getenv("REFLECTION_TRAIT_MIN_DELTA", "-0.2"))
        self.evolution_confidence_threshold = float(os.getenv("REFLECTION_EVOLUTION_THRESHOLD", "0.7"))
        
        # Reflection generation settings (not used to gate /chat or LLM; informational caps only)
        # Default to 0 (unbounded) to align with strict no-timeouts policy
        self.reflection_timeout_seconds = int(os.getenv("REFLECTION_TIMEOUT", "0"))
        self.max_reflection_length = int(os.getenv("REFLECTION_MAX_LENGTH", "2000"))
        self.min_reflection_length = int(os.getenv("REFLECTION_MIN_LENGTH", "100"))
        
        # Word count bounds for reflection narrative (allows full emotional depth)
        # Updated to 70-200 words to give LLMs more flexibility while maintaining quality
        # Lower bound of 70 allows concise reflections without being too restrictive
        # Upper bound of 200 permits deeper exploration when context warrants it
        # Validation uses ±20 word tolerance, so actual acceptance range is 50-220
        try:
            self.word_count_min = int(os.getenv("REFLECTION_WORD_MIN", "70"))
        except Exception:
            self.word_count_min = 70
        
        # Use tier-aware fallback for word count max
        try:
            word_max_env = os.getenv("REFLECTION_WORD_MAX", "0")
            self.word_count_max = int(word_max_env)
            if self.word_count_max <= 0:
                # Detect system tier for appropriate fallback
                try:
                    from ..utils.system_profile import detect_system_profile
                    profile = detect_system_profile()
                    # High-tier and standard tier share the same cap
                    # Widened to 200 words to reduce LLM word-count struggles
                    self.word_count_max = 200
                    logger.debug(f"Using tier-aware word count max: {self.word_count_max} (tier={profile.get('tier', 'unknown')})")
                except Exception as e:
                    # Final fallback to 200 words
                    self.word_count_max = 200
                    logger.warning(f"Failed to detect system tier for word count, using fallback: {e}")
        except Exception:
            self.word_count_max = 200

        # Coherence and quality checks
        self.coherence_check_enabled = os.getenv("REFLECTION_COHERENCE_CHECK", "true").lower() == "true"
        self.identity_constraint_check_enabled = os.getenv("REFLECTION_IDENTITY_CHECK", "true").lower() == "true"
        self.min_coherence_score = float(os.getenv("REFLECTION_MIN_COHERENCE", "0.6"))
        
        # Selective reflection classifier settings
        self.classifier_enabled = os.getenv("REFLECTION_CLASSIFIER_ENABLED", "true").lower() == "true"
        self.classifier_model = os.getenv("REFLECTION_CLASSIFIER_MODEL", "same")  # "same" uses base reflection model
        self.classifier_threshold = os.getenv("REFLECTION_CLASSIFIER_THRESHOLD", "balanced")  # conservative/balanced/aggressive
        self.classifier_max_tokens = int(os.getenv("REFLECTION_CLASSIFIER_MAX_TOKENS", "50"))
        self.classifier_temperature = float(os.getenv("REFLECTION_CLASSIFIER_TEMPERATURE", "0.2"))
        self.mandatory_reflection_interval = int(os.getenv("REFLECTION_MANDATORY_INTERVAL", "10"))  # Force every Nth turn
        self.mandatory_early_turns = int(os.getenv("REFLECTION_MANDATORY_EARLY_TURNS", "5"))  # Always reflect for first N turns
        
        # Fallback settings
        self.enable_fallback_generation = os.getenv("REFLECTION_ENABLE_FALLBACK", "true").lower() == "true"
        self.fallback_template_count = int(os.getenv("REFLECTION_FALLBACK_TEMPLATES", "3"))
        
        logger.info("Reflection configuration loaded successfully")
        logger.debug(f"Model mappings: {self.model_mappings}")
        logger.debug(f"Max context items: {self.max_context_items}")
        logger.debug(f"Memory threshold: {self.memory_importance_threshold}")
    
    def get_model_for_reflection_type(self, reflection_type: str) -> str:
        """
        Get the appropriate LLM model for a reflection type.
        
        Args:
            reflection_type: The reflection type
            
        Returns:
            Model name to use
        """
        return self.model_mappings.get(reflection_type, self.model_mappings["default"])
    
    def get_context_limits(self) -> Dict[str, int]:
        """
        Get context limits for reflection generation.
        
        Returns:
            Dictionary with context limits
        """
        return {
            "max_context_items": self.max_context_items,
            "max_memories": self.max_memories_per_reflection,
            "max_messages": self.max_conversation_messages
        }
    
    def get_quality_thresholds(self) -> Dict[str, float]:
        """
        Get quality and coherence thresholds.
        
        Returns:
            Dictionary with quality thresholds
        """
        return {
            "memory_importance": self.memory_importance_threshold,
            "evolution_confidence": self.evolution_confidence_threshold,
            "min_coherence": self.min_coherence_score
        }
    
    def get_trait_adjustment_bounds(self) -> tuple[float, float]:
        """
        Get the bounds for trait adjustments.
        
        Returns:
            Tuple of (min_delta, max_delta)
        """
        return (self.trait_adjustment_min, self.trait_adjustment_max)
    
    def is_coherence_check_enabled(self) -> bool:
        """Check if coherence checking is enabled."""
        return self.coherence_check_enabled
    
    def is_identity_check_enabled(self) -> bool:
        """Check if identity constraint checking is enabled."""
        return self.identity_constraint_check_enabled
    
    def is_fallback_enabled(self) -> bool:
        """Check if fallback generation is enabled."""
        return self.enable_fallback_generation
    
    def is_classifier_enabled(self) -> bool:
        """Check if selective reflection classifier is enabled."""
        return self.classifier_enabled
    
    def get_classifier_config(self) -> Dict[str, Any]:
        """
        Get classifier configuration settings.
        
        Returns:
            Dictionary with classifier settings
        """
        return {
            "enabled": self.classifier_enabled,
            "model": self.classifier_model,
            "threshold": self.classifier_threshold,
            "max_tokens": self.classifier_max_tokens,
            "temperature": self.classifier_temperature,
            "mandatory_interval": self.mandatory_reflection_interval,
            "mandatory_early_turns": self.mandatory_early_turns
        }
    
    def get_processing_timeouts(self) -> Dict[str, int]:
        """
        Get processing timeout settings.
        
        Returns:
            Dictionary with timeout settings
        """
        return {
            "reflection_timeout": self.reflection_timeout_seconds,
            "max_length": self.max_reflection_length,
            "min_length": self.min_reflection_length
        }
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check that all required models are specified
            for reflection_type, model in self.model_mappings.items():
                if not model or not isinstance(model, str):
                    logger.error(f"Invalid model for reflection type {reflection_type}: {model}")
                    return False
            
            # Check numeric bounds
            if self.trait_adjustment_min >= self.trait_adjustment_max:
                logger.error("Trait adjustment min must be less than max")
                return False
            
            if self.memory_importance_threshold < 0 or self.memory_importance_threshold > 10:
                logger.error("Memory importance threshold must be between 0 and 10")
                return False
            
            if self.min_coherence_score < 0 or self.min_coherence_score > 1:
                logger.error("Min coherence score must be between 0 and 1")
                return False
            
            if self.word_count_min < 0 or self.word_count_max <= 0:
                logger.error("Word count bounds must be positive")
                return False

            if self.word_count_min >= self.word_count_max:
                logger.error("REFLECTION_WORD_MIN must be less than REFLECTION_WORD_MAX")
                return False

            logger.info("Reflection configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


# Global configuration instance and thread lock
import threading
_reflection_config: Optional[ReflectionConfig] = None
_config_lock = threading.Lock()


def get_reflection_config() -> ReflectionConfig:
    """
    Get the global reflection configuration instance with thread-safe double-check locking.
    
    Returns:
        ReflectionConfig instance
    """
    global _reflection_config
    # First check (without lock for performance)
    if _reflection_config is None:
        with _config_lock:
            # Second check (with lock to ensure thread safety)
            if _reflection_config is None:
                # Allow singleton creation
                ReflectionConfig._allow_init = True
                _reflection_config = ReflectionConfig()
                ReflectionConfig._allow_init = False
                if not _reflection_config.validate_config():
                    logger.warning("Reflection configuration validation failed, using defaults")
    return _reflection_config


def reload_reflection_config() -> ReflectionConfig:
    """
    Reload the reflection configuration from environment variables with thread safety.
    
    Returns:
        New ReflectionConfig instance
    """
    global _reflection_config
    with _config_lock:
        # Allow singleton creation for reload
        ReflectionConfig._allow_init = True
        ReflectionConfig._initialized = False  # Reset to allow re-initialization
        _reflection_config = ReflectionConfig()
        ReflectionConfig._allow_init = False
        if not _reflection_config.validate_config():
            logger.warning("Reflection configuration validation failed after reload")
    return _reflection_config
