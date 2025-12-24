"""
API Dependencies

This module provides FastAPI dependency functions for injecting
components into API endpoints.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Dict, Any
import pathlib

from ..llm.router import LLMRouter
from ..memory.vector_store import VectorStore
from ..scheduler.event_triggers import EventTriggerSystem
from ..persona.integration import PersonaIntegration
from ..sdl.integration import SDLIntegration
from ..scheduler.scheduler_service import SchedulerService

# ReflectionService does not exist; the concrete implementation is ReflectionProcessor
from ..reflection.processor import ReflectionProcessor
from ..db.repositories.reflection import ReflectionRepository
from ..db.repositories.agent_state import (
    AffectiveStateRepository,
    AgentGoalRepository,
    PlanStepRepository,
    MetaReflectionRepository,
)
from ..prompt.builder import PromptBuilder

logger = logging.getLogger("selo.api.dependencies")

# Service registry to avoid circular imports
_service_registry: Dict[str, Any] = {}

# Caches
_llm_router: Optional[LLMRouter] = None
_vector_store: Optional[VectorStore] = None
_event_trigger_system: Optional[EventTriggerSystem] = None
_persona_integration: Optional[PersonaIntegration] = None
_sdl_integration: Optional[SDLIntegration] = None
_reflection_service: Optional[ReflectionProcessor] = None
_scheduler_service: Optional[SchedulerService] = None
_episode_service: Optional[Any] = None


def register_service(name: str, service: Any) -> None:
    """Register a service in the global registry to avoid circular imports."""
    _service_registry[name] = service
    logger.debug(f"Registered service: {name}")


def get_service(name: str) -> Optional[Any]:
    """Get a service from the global registry."""
    return _service_registry.get(name)

async def initialize_dependencies():
    """Initialize all shared dependencies.
    
    Note: This function expects app.state.services to already be initialized
    by main.py's lifespan startup. It does NOT call initialize_services() again
    to avoid duplicate service creation.
    """
    logger.info("Initializing API dependencies")

    # Reference already-initialized services from app state (no re-initialization)
    try:
        # Services should already be initialized in main.py lifespan
        # We just cache references here for dependency injection
        # NOTE: This is a no-op if called before lifespan completes
        logger.info("API dependencies initialization complete (services managed by main.py)")
    except Exception as e:
        logger.warning(f"Error during API dependencies init: {e}")


async def close_dependencies():
    """Close all shared dependencies."""
    global _vector_store, _event_trigger_system
    global _persona_integration, _sdl_integration, _reflection_service, _scheduler_service
    
    logger.info("Closing API dependencies")
    
    # Close in reverse order of initialization
    if _persona_integration:
        await _persona_integration.close()
        _persona_integration = None
    
    if _sdl_integration:
        await _sdl_integration.close()
        _sdl_integration = None
    
    if _reflection_service:
        # ReflectionProcessor has no close(); just drop reference
        _reflection_service = None
    
    # Scheduler service is managed by scheduler_integration in main.py
    # Don't stop it here to avoid duplicate cleanup
    if _scheduler_service:
        _scheduler_service = None

    global _episode_service
    if _episode_service:
        _episode_service = None
    
    # Close core components last
    if _event_trigger_system:
        _event_trigger_system = None
    
    if _vector_store:
        # VectorStore has no close(); just drop reference
        _vector_store = None
    
    # LLMRouter is managed by app services; no explicit close here
    
    logger.info("All API dependencies closed")


# === Dependency injection functions ===

async def get_llm_router() -> LLMRouter:
    """Get the LLMRouter instance for dynamic routing and logging.
    
    Retrieves from service registry with retry logic to handle startup race conditions.
    Implements exponential backoff to wait for service initialization.
    
    Returns:
        LLMRouter instance
        
    Raises:
        RuntimeError: If service not initialized after maximum retry attempts
    """
    global _llm_router
    if not _llm_router:
        # Retry with exponential backoff to handle startup race conditions
        max_retries = 5
        base_delay = 0.1  # 100ms
        
        for attempt in range(max_retries):
            _llm_router = get_service("llm_router")
            
            if _llm_router:
                logger.debug(f"LLMRouter retrieved successfully on attempt {attempt + 1}")
                break
                
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.debug(
                    f"LLMRouter not ready, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)
        
        if not _llm_router:
            raise RuntimeError(
                "LLMRouter not initialized after maximum retry attempts. "
                "Ensure main.py lifespan startup has completed before making API calls. "
                "This prevents race conditions and ensures proper service initialization order. "
                f"Retried {max_retries} times with exponential backoff."
            )
    return _llm_router

# Removed: get_llm_controller() — all code should use get_llm_router()


async def get_vector_store() -> VectorStore:
    """Get vector store instance.
    
    Retrieves from service registry with retry logic to handle startup race conditions.
    Implements exponential backoff to wait for service initialization.
    
    Returns:
        VectorStore instance
        
    Raises:
        RuntimeError: If service not initialized after maximum retry attempts
    """
    global _vector_store
    if not _vector_store:
        # Retry with exponential backoff to handle startup race conditions
        max_retries = 5
        base_delay = 0.1  # 100ms
        
        for attempt in range(max_retries):
            _vector_store = get_service("vector_store")
            
            if _vector_store:
                logger.debug(f"VectorStore retrieved successfully on attempt {attempt + 1}")
                break
                
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.debug(
                    f"VectorStore not ready, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)
        
        if not _vector_store:
            raise RuntimeError(
                "VectorStore not initialized after maximum retry attempts. "
                "Ensure main.py lifespan startup has completed before making API calls. "
                "This prevents race conditions and ensures proper service initialization order. "
                f"Retried {max_retries} times with exponential backoff."
            )
    return _vector_store


async def get_event_trigger_system() -> EventTriggerSystem:
    """Get event trigger system instance.
    
    Retrieves from service registry to avoid creating duplicate scheduler instances.
    The event system is initialized in main.py and shared across the application.
    """
    global _event_trigger_system
    if not _event_trigger_system:
        # Use the shared event system from service registry (initialized in main.py)
        _event_trigger_system = get_service("event_system")
        if not _event_trigger_system:
            raise RuntimeError(
                "Event trigger system not initialized. "
                "Ensure main.py lifespan startup has completed before making API calls."
            )
    return _event_trigger_system


async def get_persona_integration() -> PersonaIntegration:
    """Get persona integration instance."""
    global _persona_integration
    if not _persona_integration:
        _persona_integration = PersonaIntegration(
            llm_router=await get_llm_router(),
            vector_store=await get_vector_store(),
            event_system=await get_event_trigger_system(),
            conversation_repo=await get_conversation_repository(),
        )
        await _persona_integration.initialize()
    return _persona_integration


async def get_sdl_integration() -> SDLIntegration:
    """Get SDL integration instance."""
    global _sdl_integration
    if not _sdl_integration:
        _sdl_integration = SDLIntegration(
            llm_router=await get_llm_router(),
            vector_store=await get_vector_store(),
            event_system=await get_event_trigger_system()
        )
        await _sdl_integration.start()
    return _sdl_integration


async def get_reflection_service() -> ReflectionProcessor:
    """Return the singleton reflection processor used throughout the app."""
    global _reflection_service

    if not _reflection_service:
        # Prefer instance registered by main.initialize_services()
        shared_processor = get_service("reflection_processor")
        if shared_processor:
            _reflection_service = shared_processor
            return _reflection_service

        # Optional Socket.IO server injection
        try:
            from ..socketio.registry import get_socketio_server
            socketio_server = get_socketio_server()
        except Exception:
            socketio_server = None

        templates_dir = str(pathlib.Path(__file__).resolve().parent.parent / "prompt" / "templates")

        # Reuse shared services when available to mirror Phase 0 wiring
        event_bus = get_service("event_system")
        meta_processor = get_service("meta_reflection_processor")
        affective_manager = get_service("affective_state_manager")
        goal_manager = get_service("goal_manager")
        persona_repo = get_service("persona_repo")
        user_repo = get_service("user_repo")
        conversation_repo = get_service("conversation_repo") or await get_conversation_repository()

        _reflection_service = ReflectionProcessor(
            reflection_repo=get_service("reflection_repo") or ReflectionRepository(),
            prompt_builder=get_service("prompt_builder") or PromptBuilder(templates_dir=templates_dir),
            llm_controller=await get_llm_router(),
            vector_store=await get_vector_store(),
            socketio_server=socketio_server,
            event_bus=event_bus,
            meta_reflection_processor=meta_processor,
            affective_state_manager=affective_manager,
            goal_manager=goal_manager,
            conversation_repo=conversation_repo,
            persona_repo=persona_repo,
            user_repo=user_repo,
        )

    return _reflection_service


async def get_affective_state_manager():
    manager = get_service("affective_state_manager")
    if manager:
        return manager
    from ..agent.affective_state_manager import AffectiveStateManager
    from ..db.repositories.persona import PersonaRepository
    state_repo = get_service("affective_state_repo") or AffectiveStateRepository()
    persona_repo = get_service("persona_repo") or PersonaRepository()
    if not state_repo or not persona_repo:
        raise RuntimeError("Affective state dependencies unavailable")
    manager = AffectiveStateManager(state_repo=state_repo, persona_repo=persona_repo)
    register_service("affective_state_manager", manager)
    return manager


async def get_goal_manager():
    manager = get_service("goal_manager")
    if manager:
        return manager
    from ..agent.goal_manager import GoalManager
    goal_repo = get_service("agent_goal_repo") or AgentGoalRepository()
    plan_repo = get_service("plan_step_repo") or PlanStepRepository()
    meta_repo = get_service("meta_reflection_repo") or MetaReflectionRepository()
    if not goal_repo or not plan_repo or not meta_repo:
        raise RuntimeError("Goal manager dependencies unavailable")
    manager = GoalManager(goal_repo=goal_repo, plan_repo=plan_repo, meta_repo=meta_repo)
    register_service("goal_manager", manager)
    return manager


async def get_episode_service() -> Any:
    """Get autobiographical episode service instance."""
    global _episode_service
    if not _episode_service:
        service = get_service("episode_service")
        if service is None:
            raise RuntimeError("Autobiographical episode service not initialized")
        _episode_service = service
    return _episode_service


async def get_scheduler_service() -> SchedulerService:
    """Get scheduler service instance.
    
    Retrieves from service registry to avoid creating duplicate scheduler instances.
    The scheduler service is initialized by SchedulerFactory in main.py.
    """
    global _scheduler_service
    if not _scheduler_service:
        # Try to get from scheduler integration first
        scheduler_integration = get_service("scheduler_integration")
        if scheduler_integration and hasattr(scheduler_integration, "scheduler_service"):
            _scheduler_service = scheduler_integration.scheduler_service
        
        if not _scheduler_service:
            raise RuntimeError(
                "Scheduler service not initialized. "
                "Ensure main.py lifespan startup has completed and scheduler integration is set up."
            )
    return _scheduler_service

async def get_conversation_repository():
    """Get conversation repository instance."""
    from ..db.repositories.conversation import ConversationRepository
    return ConversationRepository()


async def get_reflection_repository():
    """Get reflection repository instance."""
    from ..db.repositories.reflection import ReflectionRepository
    return ReflectionRepository()


async def get_reflection_processor() -> ReflectionProcessor:
    """Alias for :func:`get_reflection_service` for backward compatibility."""
    return await get_reflection_service()
