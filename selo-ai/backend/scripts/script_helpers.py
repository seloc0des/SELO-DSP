#!/usr/bin/env python3
"""
Script Helper Module

Provides centralized initialization and dependency injection for scripts,
ensuring consistency with the main application's DI container.
"""

import os
import sys
import logging
from typing import Dict, Any
from unittest.mock import Mock

# Add backend and project root directories to sys.path for robust imports
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_BACKEND_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_BACKEND_DIR, ".."))
for _p in (_PROJECT_ROOT, _BACKEND_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger("selo.scripts.helpers")


def setup_script_logging(level: str = "INFO") -> None:
    """
    Configure logging for scripts with consistent format.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

def get_script_app_context(mock_mode: bool = False) -> Dict[str, Any]:
    """
    Initialize and return the application context for scripts.
    
    This provides a centralized way to initialize all application services,
    ensuring scripts use the same routing, configuration, and dependencies.
    
    Args:
        mock_mode: If True, use mock implementations for testing (DEPRECATED - not allowed for installation)
        
    Returns:
        Dictionary containing all initialized services and dependencies
    """
    if mock_mode:
        logger.error("Mock mode is not allowed for installation scripts")
        logger.error("Installation requires real database, LLM, and all services")
        raise ValueError("Mock mode is disabled - installation requires production services")
    
    try:
        # Import the main application's service initialization (prefer package path)
        # Try multiple import strategies to handle different execution contexts
        initialize_services = None
        import_error = None
        
        # Strategy 1: Try backend.main (when run as module from project root)
        try:
            from backend.main import initialize_services  # type: ignore
            logger.debug("Successfully imported from backend.main")
        except (ImportError, ModuleNotFoundError) as e1:
            import_error = e1
            # Strategy 2: Try main (when run from backend directory)
            try:
                from main import initialize_services  # type: ignore
                logger.debug("Successfully imported from main")
            except (ImportError, ModuleNotFoundError) as e2:
                # Strategy 3: Try sys.path manipulation
                try:
                    import sys
                    import os
                    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if backend_dir not in sys.path:
                        sys.path.insert(0, backend_dir)
                    from main import initialize_services  # type: ignore
                    logger.debug("Successfully imported from main after sys.path fix")
                except (ImportError, ModuleNotFoundError) as e3:
                    raise ImportError(f"Failed all import strategies: {e1}, {e2}, {e3}")
        
        if not initialize_services:
            raise ImportError(f"Could not import initialize_services: {import_error}")
        
        logger.info("Initializing script context with production services")
        
        # initialize_services is now async, so we need to run it in an event loop
        # NOTE: This helper is for standalone scripts, not async scheduled jobs.
        # Scheduled jobs should directly await initialize_services to avoid event loop conflicts.
        import asyncio
        
        # Create a completely fresh event loop for this script
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            services = loop.run_until_complete(initialize_services())
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}", exc_info=True)
            raise
        finally:
            # Clean up the loop after use
            try:
                # Cancel any pending tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                # Give them a chance to finish
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                logger.error("Silent exception caught", exc_info=True)
                pass
            # Don't close the loop here - let it stay open for the script's duration
            # loop.close()
        
        # Validate critical services are present
        required_services = [
            "llm_router", "conversation_repo", "persona_repo", 
            "user_repo", "reflection_repo", "vector_store"
        ]
        
        missing_services = [svc for svc in required_services if svc not in services]
        if missing_services:
            logger.warning(f"Missing services in context: {missing_services}")
        
        # Tag context so callers can detect mock vs production
        try:
            services["is_mock_context"] = False
        except Exception:
            logger.error("Silent exception caught", exc_info=True)
            pass
        logger.info(f"Script context initialized with {len(services)} services")
        return services
        
    except Exception as e:
        logger.error(f"Failed to initialize production context: {e}")
        logger.error("CRITICAL: Cannot proceed without real services")
        logger.error("Installation requires functional database, LLM, and all repositories")
        raise RuntimeError(f"Failed to initialize production services: {e}") from e


def _get_mock_app_context() -> Dict[str, Any]:
    """
    Create a mock application context for testing/development.
    
    Returns:
        Dictionary with mock implementations of all services
    """
    from unittest.mock import Mock, AsyncMock
    
    logger.info("Creating mock application context")
    
    # Create mock LLM router
    mock_llm_router = Mock()
    mock_llm_router.route = AsyncMock(return_value={
        "completion": "Mock LLM response",
        "model": "mock-model",
        "usage": {"tokens": 100}
    })
    
    # Create mock repositories
    mock_conversation_repo = Mock()
    mock_conversation_repo.get_conversation_history = AsyncMock(return_value=[])
    mock_conversation_repo.add_message = AsyncMock(return_value=Mock(id="mock-msg-id"))
    
    mock_persona_repo = Mock()
    mock_persona_repo.get_persona = AsyncMock(return_value=Mock(
        id="mock-persona-id",
        attributes=[],
        to_dict=lambda: {"id": "mock-persona-id", "attributes": []}
    ))
    
    mock_user_repo = Mock()
    mock_user_repo.get_user = AsyncMock(return_value=Mock(id="mock-user-id"))
    
    mock_reflection_repo = Mock()
    mock_reflection_repo.get_reflections = AsyncMock(return_value=[])
    
    # Create mock vector store
    mock_vector_store = Mock()
    mock_vector_store.search = AsyncMock(return_value=[])
    mock_vector_store.add = AsyncMock()
    
    # Create mock reflection processor
    mock_reflection_processor = Mock()
    mock_reflection_processor.generate_reflection = AsyncMock(return_value={
        "success": True,
        "reflection_id": "mock-reflection-id"
    })
    
    return {
        "llm_router": mock_llm_router,
        "conversational_llm_controller": mock_llm_router,  # Legacy compatibility
        "analytical_llm_controller": mock_llm_router,      # Legacy compatibility
        "conversation_repo": mock_conversation_repo,
        "persona_repo": mock_persona_repo,
        "user_repo": mock_user_repo,
        "reflection_repo": mock_reflection_repo,
        "vector_store": mock_vector_store,
        "reflection_processor": mock_reflection_processor,
        "prompt_builder": Mock(),
        "reflection_scheduler": Mock(),
        "scheduler_config": {},
        "scheduler_integration": Mock(),
        "socketio_server": Mock(),
        # Tag for downstream detection
        "is_mock_context": True
    }


def get_test_config() -> Dict[str, Any]:
    """
    Get test configuration for scripts.
    
    Returns:
        Test configuration dictionary
    """
    return {
        "database_url": "sqlite:///test_selo.db",
        "llm_timeout": 10,
        "max_retries": 1,
        "log_level": "DEBUG",
        "mock_mode": True
    }


class ScriptContext:
    """
    Context manager for script execution with proper cleanup.
    """
    
    def __init__(self, mock_mode: bool = False, log_level: str = "INFO"):
        """
        Initialize script context.
        
        Args:
            mock_mode: Use mock implementations
            log_level: Logging level
        """
        self.mock_mode = mock_mode
        self.log_level = log_level
        self.services = None
    
    def __enter__(self) -> Dict[str, Any]:
        """Enter context and initialize services."""
        setup_script_logging(self.log_level)
        self.services = get_script_app_context(self.mock_mode)
        return self.services
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and cleanup services."""
        if self.services and not self.mock_mode:
            # Cleanup background tasks to prevent script from hanging
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                
                # Cancel background monitoring tasks if present
                health_monitor_task = self.services.get("health_monitor_task")
                if health_monitor_task and not health_monitor_task.done():
                    health_monitor_task.cancel()
                    logger.debug("Cancelled health monitoring task")
                
                memory_consolidation_task = self.services.get("memory_consolidation_task")
                if memory_consolidation_task and not memory_consolidation_task.done():
                    memory_consolidation_task.cancel()
                    logger.debug("Cancelled memory consolidation task")
                
                # Cancel all remaining pending tasks in the event loop
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    if not task.done():
                        task.cancel()
                
                # Give tasks a brief moment to clean up
                if pending:
                    try:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    except Exception:
                        logger.error("Silent exception caught", exc_info=True)

                        # pass
                
                # Close the event loop to ensure clean exit
                loop.close()
                logger.debug("Event loop closed")
                
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
            
            logger.info("Script context cleanup completed")
        
        if exc_type:
            logger.error(f"Script execution failed: {exc_val}")
            return False  # Re-raise exception
        
        logger.info("Script execution completed successfully")
        return True


# Convenience functions for common script patterns
async def run_with_llm_router(func, *args, mock_mode: bool = False, **kwargs):
    """
    Run a function with LLM router from the application context.
    
    Args:
        func: Async function to run
        *args: Function arguments
        mock_mode: Use mock implementations
        **kwargs: Function keyword arguments
    """
    with ScriptContext(mock_mode=mock_mode) as services:
        llm_router = services["llm_router"]
        return await func(llm_router, *args, **kwargs)


def get_persona_components(mock_mode: bool = False) -> tuple:
    """
    Get persona engine and integration components.
    
    Args:
        mock_mode: Use mock implementations
        
    Returns:
        Tuple of (persona_engine, persona_integration, services)
    """
    services = get_script_app_context(mock_mode)
    
    # Import persona components
    from persona.engine import PersonaEngine
    from persona.integration import PersonaIntegration
    from sdl.repository import LearningRepository
    from events.triggers import EventTriggerSystem
    
    # Initialize persona engine with router
    persona_engine = PersonaEngine(
        llm_router=services["llm_router"],
        vector_store=services["vector_store"],
        persona_repo=services["persona_repo"],
        learning_repo=LearningRepository() if not mock_mode else Mock()
    )
    
    # Initialize persona integration
    persona_integration = PersonaIntegration(
        persona_engine=persona_engine,
        llm_router=services["llm_router"],
        vector_store=services["vector_store"],
        persona_repo=services["persona_repo"],
        learning_repo=LearningRepository() if not mock_mode else Mock(),
        event_trigger_system=EventTriggerSystem() if not mock_mode else Mock()
    )
    
    return persona_engine, persona_integration, services


def get_sdl_components(mock_mode: bool = False) -> tuple:
    """
    Get SDL engine and related components.
    
    Args:
        mock_mode: Use mock implementations
        
    Returns:
        Tuple of (sdl_engine, services)
    """
    services = get_script_app_context(mock_mode)
    
    # Import SDL components
    from sdl.engine import SDLEngine
    from sdl.repository import LearningRepository
    
    # Initialize SDL engine with router
    sdl_engine = SDLEngine(
        llm_router=services["llm_router"],
        vector_store=services["vector_store"],
        learning_repo=LearningRepository() if not mock_mode else Mock(),
        reflection_repo=services["reflection_repo"]
    )
    
    return sdl_engine, services
