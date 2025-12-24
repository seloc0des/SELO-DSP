"""
Scheduler Integration Module

This module integrates the enhanced scheduling system with the existing
reflection system and other application components.
"""

import logging
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime, timedelta, timezone
import os
import time

from .scheduler_service import SchedulerService
from .adaptive_scheduler import AdaptiveScheduler
from .event_triggers import EventTriggerSystem, EventType
from .resource_monitor import ResourceMonitor

# Import existing reflection components
from ..reflection.scheduler import ReflectionScheduler
from ..reflection.processor import ReflectionProcessor
from ..db.repositories.user import UserRepository
from ..db.repositories.reflection import ReflectionRepository

if TYPE_CHECKING:  # pragma: no cover
    from ..agent.agent_loop_runner import AgentLoopRunner

logger = logging.getLogger("selo.scheduler.integration")

class SchedulerIntegration:
    """
    Integrates the enhanced scheduling system with existing application components.
    
    This class connects the new enhanced scheduler with the existing reflection
    system and other application components.
    """
    
    def __init__(self, 
                 reflection_processor: Optional[ReflectionProcessor] = None,
                 user_repository: Optional[UserRepository] = None,
                 reflection_repository: Optional[ReflectionRepository] = None,
                 conversation_repository: Optional[Any] = None,
                 db_url: Optional[str] = None,
                 config: Dict[str, Any] = None,
                 event_trigger_system: Optional[EventTriggerSystem] = None,
                 agent_loop_runner: Optional["AgentLoopRunner"] = None):
        """
        Initialize the scheduler integration.
        
        Args:
            reflection_processor: Reflection processor for generating reflections
            user_repository: User repository for accessing user data
            reflection_repository: Reflection repository for storing reflections
            db_url: Database URL for job persistence
            config: Configuration options
        """
        self.reflection_processor = reflection_processor
        self.user_repository = user_repository
        self.reflection_repository = reflection_repository
        self.db_url = db_url or os.environ.get("DATABASE_URL", None)
        self.config = config or {}
        self.conversation_repository = conversation_repository
        self.event_trigger_system = event_trigger_system
        self.agent_loop_runner = agent_loop_runner
        self._agent_loop_job_id = "agent_loop_runner"
        
        # Create enhanced scheduling components
        self.scheduler_service = None
        self.adaptive_scheduler = None
        self.resource_monitor = None
        
        # Legacy reflection scheduler
        self.reflection_scheduler = None
        
        # Session episode generator (set externally for event-driven mode)
        self._session_episode_generator = None
        
        # Integration state
        self.initialized = False
        # Cooldown map for memory-triggered reflections to prevent duplicate storms
        self._memory_trigger_cooldowns: Dict[str, float] = {}
        
        logger.info("Scheduler integration initialized")
    
    def bind_session_episode_generator(self, generator) -> None:
        """
        Bind a session episode generator for event-driven episode generation.
        
        Args:
            generator: SessionEpisodeGenerator instance
        """
        self._session_episode_generator = generator
        
        # Bind event system and resource monitor to the generator
        if self.event_trigger_system:
            generator.bind_event_system(self.event_trigger_system)
        if self.resource_monitor:
            generator.bind_resource_monitor(self.resource_monitor)
        
        logger.info("Session episode generator bound to scheduler integration")
        
    async def setup(self):
        """Set up all scheduler components and integrate them."""
        if self.initialized:
            logger.warning("Scheduler integration already initialized")
            return
            
        try:
            # Initialize scheduler service
            self.scheduler_service = SchedulerService(
                db_url=self.db_url,
                config=self.config.get("scheduler_service", {})
            )
            await self.scheduler_service.start()
            
            # Initialize resource monitor
            self.resource_monitor = ResourceMonitor(
                config=self.config.get("resource_monitor", {}),
                update_callback=self._handle_resource_update
            )
            await self.resource_monitor.start()
            
            # Initialize adaptive scheduler
            self.adaptive_scheduler = AdaptiveScheduler(
                scheduler_service=self.scheduler_service,
                config=self.config.get("adaptive_scheduler", {})
            )
            
            # Initialize or reuse event trigger system
            if self.event_trigger_system:
                try:
                    existing_config = self.config.get("event_trigger_system", {}) or {}
                    if hasattr(self.event_trigger_system, "config") and existing_config:
                        self.event_trigger_system.config.update(existing_config)
                except Exception:
                    logger.debug("Unable to merge event trigger config into existing system", exc_info=True)
                self.event_trigger_system.scheduler_service = self.scheduler_service
                self.event_trigger_system.adaptive_scheduler = self.adaptive_scheduler
            else:
                self.event_trigger_system = EventTriggerSystem(
                    scheduler_service=self.scheduler_service,
                    adaptive_scheduler=self.adaptive_scheduler,
                    config=self.config.get("event_trigger_system", {})
                )
            
            conversation_repo = self.conversation_repository
            if not conversation_repo and self.reflection_processor:
                conversation_repo = getattr(self.reflection_processor, "conversation_repo", None)

            # Set up reflection scheduler with our enhanced scheduler
            self.reflection_scheduler = ReflectionScheduler(
                reflection_processor=self.reflection_processor,
                user_repo=self.user_repository,
                scheduler_service=self.scheduler_service,
                conversation_repo=conversation_repo,
                config=self.config,
            )
            
            # Set up event handlers
            await self._setup_event_handlers()
            
            # Set up default triggers
            await self._setup_default_triggers()
            # Set up daily persona reassessment at midnight US/Eastern
            await self._setup_persona_reassessment_job()
            # Schedule agent loop runner if enabled
            await self._schedule_agent_loop()
            
            # Initialize reflection schedules
            await self.reflection_scheduler.initialize()
            
            self.initialized = True
            logger.info("Scheduler integration setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up scheduler integration: {str(e)}", exc_info=True)
            await self.shutdown()
            raise
            
    async def shutdown(self):
        """Shut down all scheduler components."""
        logger.info("Shutting down scheduler integration...")
        
        # Stop resource monitor first
        if self.resource_monitor:
            await self.resource_monitor.stop()
            
        # Remove agent loop job if scheduled
        if self.scheduler_service and self._agent_loop_job_id in getattr(self.scheduler_service, "jobs", {}):
            try:
                await self.scheduler_service.remove_job(self._agent_loop_job_id)
            except Exception:
                logger.debug("Failed to remove agent loop job during shutdown", exc_info=True)

        # Stop scheduler service
        if self.scheduler_service:
            await self.scheduler_service.stop()
            
        self.initialized = False
        logger.info("Scheduler integration shutdown complete")
        
    async def close(self):
        """Alias for shutdown to match application lifecycle expectations.
        
        main.py calls `await scheduler_integration.close()` during app shutdown.
        Implementing this alias prevents warnings about a missing close() method
        and ensures a graceful teardown by delegating to `shutdown()`.
        """
        await self.shutdown()
        
    async def _handle_resource_update(self, resource_data: Dict[str, Any]):
        """
        Handle resource updates from the monitor.
        
        Args:
            resource_data: Current resource usage data
        """
        # Pass resource updates to the adaptive scheduler
        if self.adaptive_scheduler:
            await self.adaptive_scheduler.update_resource_usage(
                resource_data["cpu"],
                resource_data["memory"]
            )
            
    async def _setup_event_handlers(self):
        """Set up event handlers for different event types."""
        if not self.event_trigger_system:
            return
            
        # Handle conversation events
        await self.event_trigger_system.register_event_handler(
            EventType.CONVERSATION,
            self._handle_conversation_event
        )
        
        # Handle user interaction events
        await self.event_trigger_system.register_event_handler(
            EventType.USER_INTERACTION,
            self._handle_user_interaction_event
        )
        
        # Handle memory events
        await self.event_trigger_system.register_event_handler(
            EventType.MEMORY_CREATED,
            self._handle_memory_event
        )
        
        # Register conversation end handler for event-driven episode generation
        await self.event_trigger_system.register_event_handler(
            "conversation.ended",
            self._handle_conversation_end_event
        )
        
        # Register resource available callback for processing deferred tasks
        if self.resource_monitor:
            self.resource_monitor.register_available_callback(
                self._on_resources_available
            )
    
    async def _handle_conversation_end_event(self, event_data: Dict[str, Any], user_id: str):
        """
        Handle conversation end events for event-driven episode generation.
        
        Args:
            event_data: Conversation end event data
            user_id: Associated user ID
        """
        # Forward to session episode generator if available
        if hasattr(self, '_session_episode_generator') and self._session_episode_generator:
            await self._session_episode_generator.on_conversation_end(event_data, user_id)
    
    async def _on_resources_available(self):
        """Called when resources become available after being constrained."""
        logger.info("Resources available - processing deferred tasks")
        
        # Process deferred episode generations
        if hasattr(self, '_session_episode_generator') and self._session_episode_generator:
            try:
                await self._session_episode_generator.process_deferred_generations()
            except Exception as e:
                logger.error(f"Error processing deferred episode generations: {e}")
        
    async def _handle_conversation_event(self, event_data: Dict[str, Any], user_id: str):
        """
        Handle a conversation event.
        
        Args:
            event_data: Conversation event data
            user_id: Associated user ID
        """
        logger.debug(f"Processing conversation event for user {user_id}")
        
        if not user_id or not self.adaptive_scheduler:
            return
            
        # Estimate activity level based on conversation
        message_count = event_data.get("message_count", 1)
        content_length = len(event_data.get("content", ""))
        
        # Normalize to activity level between 0-1
        # More messages and longer content = higher activity
        activity_level = min(1.0, (message_count * 0.2) + (content_length / 1000))
        
        # Update user activity in adaptive scheduler
        await self.adaptive_scheduler.update_user_activity(user_id, activity_level)
        
    async def _handle_user_interaction_event(self, event_data: Dict[str, Any], user_id: str):
        """
        Handle a user interaction event.
        
        Args:
            event_data: User interaction event data
            user_id: Associated user ID
        """
        logger.debug(f"Processing user interaction event for user {user_id}")
        
        if not user_id or not self.adaptive_scheduler:
            return
            
        # Extract activity level from event data
        activity_level = event_data.get("activity_level", 0.5)
        
        # Update user activity in adaptive scheduler
        await self.adaptive_scheduler.update_user_activity(user_id, activity_level)
        
    async def _handle_memory_event(self, event_data: Dict[str, Any], user_id: str):
        """
        Handle a memory creation/update event.
        
        Args:
            event_data: Memory event data
            user_id: Associated user ID
        """
        logger.debug(f"Processing memory event for user {user_id}")
        
        # Check if this memory should trigger a reflection
        memory_type = event_data.get("memory_type", "")
        importance = event_data.get("importance", 0.0)
        memory_id = event_data.get("memory_id") or ""

        # Apply per-memory cooldown to avoid duplicate storms while keeping immediacy
        cooldown_seconds = 120  # 2 minutes
        now = time.time()
        cooldown_key = f"{user_id}:{memory_id or memory_type}"
        next_allowed = self._memory_trigger_cooldowns.get(cooldown_key, 0)
        if now < next_allowed:
            logger.debug("Skipping memory-triggered reflection for %s (cooldown active)", cooldown_key)
            return
        
        # High importance memories might trigger immediate reflections
        if importance >= 0.8:
            logger.info(f"High importance memory detected for user {user_id}, considering reflection")
            
            # Implement logic to trigger reflection based on important memories
            try:
                self._memory_trigger_cooldowns[cooldown_key] = now + cooldown_seconds

                # First attempt immediate reflection (must-have behavior)
                if hasattr(self, "reflection_processor") and self.reflection_processor:
                    try:
                        await self.reflection_processor.generate_reflection(
                            reflection_type="memory_triggered",
                            user_profile_id=user_id,
                            memory_ids=[memory_id] if memory_id else None,
                            trigger_source="important_memory_immediate",
                        )
                        logger.info("Immediate memory-triggered reflection executed for user %s", user_id)
                        return
                    except Exception as immediate_err:
                        logger.warning(
                            "Immediate reflection failed for user %s (will schedule fallback): %s",
                            user_id,
                            immediate_err,
                        )

                # Fallback: schedule a reflection if scheduler is available
                if self.reflection_scheduler:
                    await self.reflection_scheduler.schedule_reflection(
                        user_profile_id=user_id,
                        reflection_type="memory_triggered",
                        trigger_source="important_memory_fallback",
                        context={
                            "memory_id": memory_id,
                            "importance_score": importance,
                            "trigger_reason": "High importance memory detected (fallback)"
                        },
                        delay_seconds=5  # Small delay to allow memory to be fully processed
                    )
                    logger.info(f"Scheduled fallback memory-triggered reflection for user {user_id}")
                    
            except Exception as e:
                logger.error(f"Error triggering reflection for important memory: {str(e)}", exc_info=True)
            
    async def _setup_default_triggers(self):
        """Set up default event triggers."""
        if not self.event_trigger_system or not self.reflection_scheduler:
            return
            
        # Emotional spike trigger - triggers a reflection when emotional content is detected
        try:
            await self.event_trigger_system.register_trigger(
                trigger_id="emotional_spike_reflection",
                event_type=EventType.EMOTIONAL_SPIKE,
                condition={
                    "type": "simple",
                    "field": "intensity",
                    "operator": "gte",
                    "value": 0.7
                },
                action=self._trigger_emotional_reflection,
                cooldown_seconds=14400,  # 4 hours
                importance=0.8
            )
            logger.info("Registered emotional spike reflection trigger")
        except (ValueError, Exception) as e:
            logger.error(f"Failed to register emotional spike trigger: {e}")
            # Continue with other triggers
        
        # Knowledge update pattern - triggers a reflection when multiple related knowledge items are updated
        try:
            await self.event_trigger_system.register_pattern(
                pattern_id="knowledge_update_pattern",
                event_types=[EventType.KNOWLEDGE_UPDATED, EventType.MEMORY_CREATED],
                pattern_config={
                    "type": "frequency",
                    "thresholds": {
                        EventType.KNOWLEDGE_UPDATED: 3,
                        EventType.MEMORY_CREATED: 2
                    },
                    "time_window_seconds": 3600,  # 1 hour
                    "cooldown_seconds": 86400  # 1 day
                },
                action=self._trigger_knowledge_reflection
            )
            logger.info("Registered knowledge update pattern trigger")
        except (ValueError, Exception) as e:
            logger.error(f"Failed to register knowledge update pattern: {e}")
            # Continue with other triggers

    async def _schedule_agent_loop(self):
        """Register the agent loop runner with the scheduler.
        
        Uses a module-level function reference that can be serialized to the
        persistent job store, allowing the agent loop to survive restarts.
        """
        if not self.scheduler_service or not self.agent_loop_runner:
            return

        if not self.agent_loop_runner.enabled:
            logger.info("Agent loop runner disabled; skipping scheduler registration")
            return

        # Bind resource monitor for resource-aware adaptive scheduling
        if self.resource_monitor:
            self.agent_loop_runner.bind_resource_monitor(self.resource_monitor)
        
        # Bind scheduler service for dynamic rescheduling
        if self.scheduler_service:
            self.agent_loop_runner.bind_scheduler(self.scheduler_service, self._agent_loop_job_id)

        # Register the runner instance globally so the serializable function can access it
        from ..agent.agent_loop_runner import set_agent_loop_runner_instance
        set_agent_loop_runner_instance(self.agent_loop_runner)

        interval = max(60, self.agent_loop_runner.interval_seconds)
        try:
            # Use string reference for serialization to persistent job store
            await self.scheduler_service.add_job(
                job_id=self._agent_loop_job_id,
                func="backend.agent.agent_loop_runner:run_agent_loop_job",  # String reference - serializable!
                trigger="interval",
                seconds=interval,
            )
            logger.info("Registered agent loop runner job (persistent) at %s-second interval", interval)
        except Exception as err:
            logger.error("Failed to register agent loop runner job: %s", err, exc_info=True)

    async def _setup_persona_reassessment_job(self):
        """Schedule a daily persona reassessment event at midnight US/Eastern.
        Uses a module-level callable path for APScheduler persistence.
        """
        if not self.scheduler_service:
            return
        try:
            job_id = "persona_reassessment_daily"
            # Determine a valid tzinfo for America/New_York (handles DST)
            tzinfo = None
            try:
                import pytz  # type: ignore
                tzinfo = pytz.timezone("America/New_York")
            except Exception:
                try:
                    import pytz  # type: ignore
                    tzinfo = pytz.timezone("US/Eastern")
                except Exception:
                    tzinfo = None
                    logger.warning("Falling back to scheduler default timezone (UTC) for persona reassessment job")

            # Schedule a cron job at 00:00 local (America/New_York when available)
            await self.scheduler_service.add_job(
                job_id=job_id,
                func="backend.scheduler.tasks:run_daily_persona_reassessment",
                trigger="cron",
                hour=0,
                minute=0,
                **({"timezone": tzinfo} if tzinfo else {}),
            )
            logger.info("Registered daily persona reassessment job at midnight America/New_York")
        except Exception as e:
            logger.error(f"Error scheduling persona reassessment job: {e}", exc_info=True)

    async def _emit_daily_persona_reassessment(self):
        """Emit a scheduler persona evolution event for the default user/persona."""
        try:
            # Resolve default user and persona lazily
            from ..db.repositories.user import UserRepository
            from ..db.repositories.persona import PersonaRepository
            user_repo = self.user_repository or UserRepository()
            persona_repo = PersonaRepository()
            user = await user_repo.get_or_create_default_user()
            if not user:
                logger.warning("No default user found for persona reassessment")
                return
            persona = await persona_repo.get_persona_by_user(user_id=user.id, is_default=True)
            if not persona:
                logger.warning("No default persona found for persona reassessment")
                return

            event_data = {
                "persona_id": getattr(persona, "id", None),
                "user_id": getattr(user, "id", None),
                "trigger_type": "scheduled",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            # Prefer immediate publish so handlers run without extra delay
            if self.event_trigger_system:
                await self.event_trigger_system.publish_event(
                    event_type="scheduler.task.persona_evolution",
                    event_data=event_data,
                )
                logger.info("Emitted daily persona reassessment event")
        except Exception as e:
            logger.error(f"Failed to emit daily persona reassessment: {e}", exc_info=True)
        
    async def _trigger_emotional_reflection(self, event_data: Dict[str, Any], user_id: str):
        """
        Trigger a reflection based on emotional content.
        
        Args:
            event_data: Event data that triggered the reflection
            user_id: Associated user ID
        """
        if not self.reflection_scheduler or not user_id:
            return
            
        logger.info(f"Triggering emotional reflection for user {user_id}")
        
        try:
            # Create a one-time job for generating an emotional reflection
            job_id = f"emotional_reflection_{user_id}_{int(time.time())}"
            
            emotion = event_data.get("emotion", "unknown")
            intensity = event_data.get("intensity", 0.0)
            context = event_data.get("context", "")
            
            # Prepare reflection context
            reflection_context = {
                "trigger_source": "system",
                "trigger_type": "emotional_event",
                "emotion": emotion,
                "intensity": intensity,
                "context": context
            }
            
            # Schedule an immediate reflection
            await self.scheduler_service.add_job(
                job_id=job_id,
                func=self._generate_emotional_reflection,
                trigger="date",
                run_date=datetime.now() + timedelta(seconds=10),
                args=[user_id, reflection_context]
            )
            
            logger.info(f"Scheduled emotional reflection job {job_id} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error scheduling emotional reflection: {str(e)}", exc_info=True)
            
    async def _generate_emotional_reflection(self, user_id: str, context: Dict[str, Any]):
        """
        Generate an emotional reflection.
        
        Args:
            user_id: User ID
            context: Reflection context
        """
        if not self.reflection_processor or not user_id:
            return
            
        try:
            logger.debug(f"Generating emotional reflection for user {user_id}")
            
            # Use the reflection processor to generate the reflection
            await self.reflection_processor.generate_reflection(
                reflection_type="emotional",
                user_profile_id=user_id,
                trigger_source="system"
            )
            
        except Exception as e:
            logger.error(f"Error generating emotional reflection: {str(e)}", exc_info=True)
            
    async def _trigger_knowledge_reflection(self, events: List[Dict[str, Any]], user_id: str):
        """
        Trigger a reflection based on knowledge updates.
        
        Args:
            events: List of events that matched the pattern
            user_id: Associated user ID
        """
        if not self.reflection_scheduler or not user_id:
            return
            
        logger.info(f"Triggering knowledge reflection for user {user_id}")
        
        try:
            # Create a one-time job for generating a knowledge reflection
            job_id = f"knowledge_reflection_{user_id}_{int(time.time())}"
            
            # Extract topics from events
            topics = set()
            for event in events:
                if "data" in event and "topic" in event["data"]:
                    topics.add(event["data"]["topic"])
                    
            # Prepare reflection context
            reflection_context = {
                "trigger_source": "system",
                "trigger_type": "knowledge_update",
                "topics": list(topics),
                "event_count": len(events)
            }
            
            # Schedule a knowledge reflection soon
            await self.scheduler_service.add_job(
                job_id=job_id,
                func=self._generate_knowledge_reflection,
                trigger="date",
                run_date=datetime.now() + timedelta(minutes=5),
                args=[user_id, reflection_context]
            )
            
            logger.info(f"Scheduled knowledge reflection job {job_id} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error scheduling knowledge reflection: {str(e)}", exc_info=True)
            
    async def _generate_knowledge_reflection(self, user_id: str, context: Dict[str, Any]):
        """
        Generate a knowledge reflection.
        
        Args:
            user_id: User ID
            context: Reflection context
        """
        if not self.reflection_processor or not user_id:
            return
            
        try:
            logger.debug(f"Generating knowledge reflection for user {user_id}")
            
            # Use the reflection processor to generate the reflection
            await self.reflection_processor.generate_reflection(
                reflection_type="knowledge",
                user_profile_id=user_id,
                trigger_source="system"
            )
            
        except Exception as e:
            logger.error(f"Error generating knowledge reflection: {str(e)}", exc_info=True)
            
    async def process_event(self, event_type: str, event_data: Dict[str, Any], user_id: Optional[str] = None):
        """
        Process an application event.
        
        Args:
            event_type: Type of event
            event_data: Event data
            user_id: Associated user ID
            
        This method should be called from other parts of the application
        when events occur that might affect scheduling.
        """
        if not self.initialized or not self.event_trigger_system:
            logger.warning("Scheduler integration not initialized, can't process event")
            return
            
        await self.event_trigger_system.process_event(event_type, event_data, user_id)
