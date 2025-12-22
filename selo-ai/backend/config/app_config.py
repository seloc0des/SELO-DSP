"""
Application Configuration Module

Centralized configuration management for SELO AI application including
database, LLM, security, and system-wide settings.
"""

import os
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger("selo.config.app")


class AppConfig:
    """Main application configuration manager."""
    
    def __init__(self):
        """Initialize configuration with environment variables and defaults."""
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables with fallbacks."""
        
        # Database configuration (align default with db/session.py)
        # Note: db/session.py converts postgresql:// to postgresql+asyncpg:// automatically
        # SECURITY: No default credentials - DATABASE_URL must be explicitly configured
        self.database_url = os.getenv("DATABASE_URL", "")
        self.database_pool_size = int(os.getenv("DATABASE_POOL_SIZE", "10"))
        self.database_max_overflow = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))
        self.database_pool_timeout = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))
        
        # LLM configuration
        self.conversational_model = os.getenv("CONVERSATIONAL_MODEL", "llama3:8b")
        self.analytical_model = os.getenv("ANALYTICAL_MODEL", "qwen2.5:3b")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        # Default to unbounded generation unless explicitly limited via env
        self.llm_timeout = int(os.getenv("LLM_TIMEOUT", "0"))
        self.llm_max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        
        # Frontend configuration
        self.frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        self.static_files_path = os.getenv("STATIC_FILES_PATH", "./frontend/build")

        # Security configuration
        self.api_key = os.getenv("SELO_SYSTEM_API_KEY", "")
        # If CORS_ORIGINS is not set, default to FRONTEND_URL and localhost to avoid hardcoded IPs
        cors_raw = os.getenv("CORS_ORIGINS", "").strip()
        if cors_raw:
            self.cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
        else:
            # Use the actual FRONTEND_URL (user's IP/domain) plus localhost
            self.cors_origins = [self.frontend_url, "http://localhost:3000"]
        self.session_secret = os.getenv("SESSION_SECRET", "default-session-secret-change-me")
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        
        # Server configuration
        self.host = os.getenv("HOST", "localhost")
        self.port = int(os.getenv("PORT", "8000"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        
        # Storage and backup configuration
        self.backup_enabled = os.getenv("BACKUP_ENABLED", "true").lower() == "true"
        self.backup_interval_hours = int(os.getenv("BACKUP_INTERVAL_HOURS", "24"))
        self.backup_retention_days = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
        self.backup_path = os.getenv("BACKUP_PATH", "./backups")
        
        # Memory and performance configuration
        self.max_conversation_history = int(os.getenv("MAX_CONVERSATION_HISTORY", "50"))
        self.memory_cleanup_interval = int(os.getenv("MEMORY_CLEANUP_INTERVAL", "3600"))
        self.vector_store_dimension = int(os.getenv("VECTOR_STORE_DIMENSION", "384"))
        
        # Persona system configuration
        self.persona_evolution_enabled = os.getenv("PERSONA_EVOLUTION_ENABLED", "true").lower() == "true"
        self.persona_evolution_threshold = float(os.getenv("PERSONA_EVOLUTION_THRESHOLD", "0.7"))
        self.max_trait_adjustment = float(os.getenv("MAX_TRAIT_ADJUSTMENT", "0.2"))
        
        # Reflection system configuration
        self.reflection_enabled = os.getenv("REFLECTION_ENABLED", "true").lower() == "true"
        self.reflection_schedule_enabled = os.getenv("REFLECTION_SCHEDULE_ENABLED", "true").lower() == "true"
        self.daily_reflection_time = os.getenv("DAILY_REFLECTION_TIME", "00:00")
        self.weekly_reflection_day = int(os.getenv("WEEKLY_REFLECTION_DAY", "0"))  # Sunday
        
        # SDL (Self-Directed Learning) configuration
        self.sdl_enabled = os.getenv("SDL_ENABLED", "true").lower() == "true"
        self.sdl_learning_rate = float(os.getenv("SDL_LEARNING_RATE", "0.1"))
        self.sdl_concept_threshold = float(os.getenv("SDL_CONCEPT_THRESHOLD", "0.8"))
        
        # Socket.IO configuration
        self.socketio_cors_origins = self.cors_origins
        self.socketio_ping_timeout = int(os.getenv("SOCKETIO_PING_TIMEOUT", "60"))
        self.socketio_ping_interval = int(os.getenv("SOCKETIO_PING_INTERVAL", "25"))
        
        logger.info("Application configuration loaded successfully")
        logger.debug(f"Database URL: {self.database_url}")
        logger.debug(f"Conversational model: {self.conversational_model}")
        logger.debug(f"Analytical model: {self.analytical_model}")
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
            "pool_timeout": self.database_pool_timeout
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return {
            "conversational_model": self.conversational_model,
            "analytical_model": self.analytical_model,
            "ollama_base_url": self.ollama_base_url,
            "timeout": self.llm_timeout,
            "max_retries": self.llm_max_retries
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return {
            "api_key": self.api_key,
            "cors_origins": self.cors_origins,
            "session_secret": self.session_secret,
            "rate_limit": {
                "requests": self.rate_limit_requests,
                "window": self.rate_limit_window
            }
        }
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration."""
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "log_level": self.log_level,
            "frontend_url": self.frontend_url,
            "static_files_path": self.static_files_path
        }
    
    def get_backup_config(self) -> Dict[str, Any]:
        """Get backup configuration."""
        return {
            "enabled": self.backup_enabled,
            "interval_hours": self.backup_interval_hours,
            "retention_days": self.backup_retention_days,
            "path": self.backup_path
        }
    
    def get_persona_config(self) -> Dict[str, Any]:
        """Get persona system configuration."""
        return {
            "evolution_enabled": self.persona_evolution_enabled,
            "evolution_threshold": self.persona_evolution_threshold,
            "max_trait_adjustment": self.max_trait_adjustment
        }
    
    def get_reflection_config(self) -> Dict[str, Any]:
        """Get reflection system configuration."""
        return {
            "enabled": self.reflection_enabled,
            "schedule_enabled": self.reflection_schedule_enabled,
            "daily_time": self.daily_reflection_time,
            "weekly_day": self.weekly_reflection_day
        }
    
    def get_sdl_config(self) -> Dict[str, Any]:
        """Get SDL configuration."""
        return {
            "enabled": self.sdl_enabled,
            "learning_rate": self.sdl_learning_rate,
            "concept_threshold": self.sdl_concept_threshold
        }
    
    def get_socketio_config(self) -> Dict[str, Any]:
        """Get Socket.IO configuration."""
        return {
            "cors_origins": self.socketio_cors_origins,
            "ping_timeout": self.socketio_ping_timeout,
            "ping_interval": self.socketio_ping_interval
        }
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.debug
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration with comprehensive checks.
        
        Validates:
        - Required fields and production settings
        - Database URL format and connectivity
        - Numeric parameter ranges
        - URL and path formats
        - Model names (optional check)
        
        Returns:
            True if configuration is valid, False otherwise
        """
        validation_errors = []
        
        try:
            # Check required configuration
            if not self.database_url:
                validation_errors.append("DATABASE_URL is required")
            else:
                # Validate database URL format
                if not (self.database_url.startswith("postgresql://") or 
                        self.database_url.startswith("postgresql+asyncpg://")):
                    validation_errors.append(
                        "DATABASE_URL must start with 'postgresql://' or 'postgresql+asyncpg://'"
                    )
            
            if not self.api_key and self.is_production():
                validation_errors.append("SELO_SYSTEM_API_KEY is required in production")
            
            # Warn about default credentials
            if self.api_key in ("dev-secret-key-change-me", "default-session-secret-change-me"):
                if self.is_production():
                    validation_errors.append(
                        "Default API key detected in production. Change SELO_SYSTEM_API_KEY."
                    )
                else:
                    logger.warning("Using default API key. Change for production deployment.")
            
            # Check numeric bounds
            if self.port < 1 or self.port > 65535:
                validation_errors.append("PORT must be between 1 and 65535")
            
            if self.database_pool_size < 1:
                validation_errors.append("DATABASE_POOL_SIZE must be at least 1")
            
            if self.database_pool_size > 100:
                logger.warning(f"DATABASE_POOL_SIZE is very high ({self.database_pool_size}). Consider reducing.")
            
            if self.database_max_overflow < 0:
                validation_errors.append("DATABASE_MAX_OVERFLOW must be >= 0")
            
            if self.database_pool_timeout < 1:
                validation_errors.append("DATABASE_POOL_TIMEOUT must be at least 1 second")
            
            # Allow 0 or positive values for LLM_TIMEOUT (0 = unbounded)
            if self.llm_timeout < 0:
                validation_errors.append("LLM_TIMEOUT must be >= 0 (0 = unbounded)")
            
            if self.llm_max_retries < 0:
                validation_errors.append("LLM_MAX_RETRIES must be >= 0")
            
            # Validate CORS origins format
            for origin in self.cors_origins:
                if not origin.startswith(("http://", "https://")):
                    validation_errors.append(
                        f"Invalid CORS origin '{origin}'. Must start with http:// or https://"
                    )
            
            # Validate Ollama base URL format
            if not self.ollama_base_url.startswith(("http://", "https://")):
                validation_errors.append(
                    f"OLLAMA_BASE_URL must start with http:// or https://"
                )
            
            # Validate model names are not empty
            if not self.conversational_model or not self.conversational_model.strip():
                validation_errors.append("CONVERSATIONAL_MODEL cannot be empty")
            
            if not self.analytical_model or not self.analytical_model.strip():
                validation_errors.append("ANALYTICAL_MODEL cannot be empty")
            
            # Validate persona configuration ranges
            if self.persona_evolution_threshold < 0 or self.persona_evolution_threshold > 1:
                validation_errors.append("PERSONA_EVOLUTION_THRESHOLD must be between 0 and 1")
            
            if self.max_trait_adjustment < 0 or self.max_trait_adjustment > 1:
                validation_errors.append("MAX_TRAIT_ADJUSTMENT must be between 0 and 1")
            
            # Validate SDL configuration
            if self.sdl_learning_rate < 0 or self.sdl_learning_rate > 1:
                validation_errors.append("SDL_LEARNING_RATE must be between 0 and 1")
            
            if self.sdl_concept_threshold < 0 or self.sdl_concept_threshold > 1:
                validation_errors.append("SDL_CONCEPT_THRESHOLD must be between 0 and 1")
            
            # Validate rate limiting
            if self.rate_limit_requests < 1:
                validation_errors.append("RATE_LIMIT_REQUESTS must be at least 1")
            
            if self.rate_limit_window < 1:
                validation_errors.append("RATE_LIMIT_WINDOW must be at least 1 second")
            
            # Validate backup configuration
            if self.backup_interval_hours < 1:
                validation_errors.append("BACKUP_INTERVAL_HOURS must be at least 1")
            
            if self.backup_retention_days < 1:
                validation_errors.append("BACKUP_RETENTION_DAYS must be at least 1")
            
            # Check paths exist or can be created
            backup_path = Path(self.backup_path)
            if self.backup_enabled:
                try:
                    backup_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    validation_errors.append(f"Cannot create backup directory {backup_path}: {e}")
            
            # Check static files path if specified
            if self.static_files_path:
                static_path = Path(self.static_files_path)
                if not static_path.exists() and self.is_production():
                    logger.warning(
                        f"Static files path does not exist: {static_path}. "
                        f"Frontend may not be accessible."
                    )
            
            # Report all validation errors
            if validation_errors:
                logger.error("Configuration validation failed with the following errors:")
                for error in validation_errors:
                    logger.error(f"  - {error}")
                return False
            
            logger.info("Application configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed with exception: {e}", exc_info=True)
            return False
    
    def get_env_template(self) -> str:
        """
        Generate a template .env file with all configuration options.
        
        Returns:
            String containing .env template
        """
        return """# SELO AI Configuration Template
# Copy this to .env and customize for your installation

# Database Configuration
DATABASE_URL=postgresql://localhost/selo_ai
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
DATABASE_POOL_TIMEOUT=30

# LLM Configuration (default profile)
CONVERSATIONAL_MODEL=llama3:8b
ANALYTICAL_MODEL=qwen2.5:3b
REFLECTION_LLM=qwen2.5:3b
OLLAMA_BASE_URL=http://127.0.0.1:11434
LLM_TIMEOUT=0
LLM_MAX_RETRIES=3

# Security Configuration
SELO_SYSTEM_API_KEY=your-secure-api-key-here
CORS_ORIGINS=http://localhost:3000
SESSION_SECRET=your-secure-session-secret-here
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Server Configuration
HOST=localhost
PORT=8000
DEBUG=false
LOG_LEVEL=INFO
FRONTEND_URL=http://localhost:3000
STATIC_FILES_PATH=./frontend/build

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_INTERVAL_HOURS=24
BACKUP_RETENTION_DAYS=30
BACKUP_PATH=./backups

# Performance Configuration
MAX_CONVERSATION_HISTORY=50
MEMORY_CLEANUP_INTERVAL=3600
VECTOR_STORE_DIMENSION=384

# Persona System Configuration
PERSONA_EVOLUTION_ENABLED=true
PERSONA_EVOLUTION_THRESHOLD=0.7
MAX_TRAIT_ADJUSTMENT=0.2

# Reflection System Configuration
REFLECTION_ENABLED=true
REFLECTION_SCHEDULE_ENABLED=true
DAILY_REFLECTION_TIME=00:00
WEEKLY_REFLECTION_DAY=0

# Self-Directed Learning Configuration
SDL_ENABLED=true
SDL_LEARNING_RATE=0.1
SDL_CONCEPT_THRESHOLD=0.8

# Socket.IO Configuration
SOCKETIO_PING_TIMEOUT=60
SOCKETIO_PING_INTERVAL=25

# Reflection-specific Configuration
REFLECTION_MODEL_DAILY=qwen:latest
REFLECTION_MODEL_WEEKLY=llama3:latest
REFLECTION_MODEL_EMOTIONAL=qwen:latest
REFLECTION_MODEL_MANIFESTO=llama3:latest
REFLECTION_MODEL_MESSAGE=qwen:latest
REFLECTION_MODEL_PERIODIC=qwen:latest
REFLECTION_MODEL_DEFAULT=qwen:latest
REFLECTION_MAX_CONTEXT_ITEMS=10
REFLECTION_MEMORY_THRESHOLD=3.0
REFLECTION_MAX_MEMORIES=10
REFLECTION_MAX_MESSAGES=20
REFLECTION_TRAIT_MAX_DELTA=0.2
REFLECTION_TRAIT_MIN_DELTA=-0.2
REFLECTION_EVOLUTION_THRESHOLD=0.7
REFLECTION_TIMEOUT=30
REFLECTION_MAX_LENGTH=2000
REFLECTION_MIN_LENGTH=100
REFLECTION_COHERENCE_CHECK=true
REFLECTION_IDENTITY_CHECK=true
REFLECTION_MIN_COHERENCE=0.6
REFLECTION_ENABLE_FALLBACK=true
REFLECTION_FALLBACK_TEMPLATES=3
"""


# Global configuration instance
_app_config: Optional[AppConfig] = None


def get_app_config() -> AppConfig:
    """
    Get the global application configuration instance.
    
    Returns:
        AppConfig instance
    """
    global _app_config
    if _app_config is None:
        _app_config = AppConfig()
        if not _app_config.validate_config():
            logger.warning("Application configuration validation failed, using defaults")
    return _app_config


def reload_app_config() -> AppConfig:
    """
    Reload the application configuration from environment variables.
    
    Returns:
        New AppConfig instance
    """
    global _app_config
    _app_config = AppConfig()
    if not _app_config.validate_config():
        logger.warning("Application configuration validation failed after reload")
    return _app_config
