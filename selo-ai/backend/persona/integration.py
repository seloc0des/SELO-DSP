"""
Persona Integration Module

This module integrates the Persona Engine with other SELO AI components,
including SDL, Reflection, and Conversation systems.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from .engine import PersonaEngine
from ..memory.vector_store import VectorStore
from ..db.repositories.persona import PersonaRepository
from ..sdl.repository import LearningRepository
from ..scheduler.event_triggers import EventTriggerSystem
from ..db.repositories.reflection import ReflectionRepository

logger = logging.getLogger("selo.persona.integration")


class PersonaIntegration:
    """
    Integration layer for the Persona System.
    
    Connects the persona engine with event triggers, SDL, reflection,
    and conversation systems to enable autonomous persona evolution.
    """
    
    def __init__(
        self,
        llm_router,
        vector_store: Optional[VectorStore] = None,
        event_system: Optional[EventTriggerSystem] = None,
        persona_engine: Optional[PersonaEngine] = None,
        conversation_repo: Optional[Any] = None,
        use_saga: bool = True,  # Enabled by default for data consistency
    ):
        """Initialize persona integration."""
        self.llm_router = llm_router
        self.vector_store = vector_store
        self.event_system = event_system or EventTriggerSystem()
        self.conversation_repo = conversation_repo
        self.use_saga = use_saga

        # Initialize repositories
        self.persona_repo = PersonaRepository()
        self.learning_repo = LearningRepository()
        self.reflection_repo = ReflectionRepository()

        # Initialize persona engine with LLMRouter
        self.persona_engine = persona_engine or PersonaEngine(
            llm_router=self.llm_router,
            vector_store=self.vector_store,
            persona_repo=self.persona_repo,
            learning_repo=self.learning_repo
        )
        
        # Initialize saga integration if enabled
        self.saga_integration = None
        if use_saga:
            from ..saga.integration import SagaIntegration
            self.saga_integration = SagaIntegration(
                persona_repo=self.persona_repo,
                learning_repo=self.learning_repo
            )

        # Track initialization status
        self.initialized = False

        logger.info(f"Persona Integration initialized with LLMRouter (saga_enabled={use_saga})")
    
    async def initialize(self):
        """Initialize the integration and register event handlers."""
        if self.initialized:
            logger.debug("Persona Integration already initialized")
            return
        
        try:
            # Register event handlers
            await self._register_event_handlers()
            
            # Set initialization flag
            self.initialized = True
            logger.info("Persona Integration fully initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Persona Integration: {str(e)}", exc_info=True)
            raise
    
    async def close(self):
        """Close all resources."""
        if self.persona_engine:
            await self.persona_engine.close()
    
    # === Helper Methods ===
    
    def _is_bootstrap_reflection(
        self,
        event_data: Dict[str, Any],
        reflection: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Centralized check to determine if a reflection is from bootstrap process.
        
        Args:
            event_data: Event data containing metadata
            reflection: Optional reflection object with additional metadata
            
        Returns:
            True if reflection is from bootstrap, False otherwise
        """
        # Check event_data trigger_source
        trigger_source = event_data.get("trigger_source")
        if trigger_source and str(trigger_source).lower() == "bootstrap":
            return True
        
        # Check reflection metadata if available
        if reflection:
            result = reflection.get("result") or {}
            metadata = result.get("metadata", {})
            reflection_trigger = metadata.get("trigger_source")
            if reflection_trigger and str(reflection_trigger).lower() == "bootstrap":
                return True
        
        return False
    
    # === Event Handler Registration ===
    
    async def _register_event_handlers(self):
        """Register event handlers for autonomous operation."""
        # Register handlers for learning events
        await self.event_system.register_handler(
            event_type="sdl.learning.created",
            handler=self._handle_new_learning
        )
        
        # Register handlers for reflection events
        await self.event_system.register_handler(
            event_type="reflection.created", 
            handler=self._handle_new_reflection
        )
        
        # Register handlers for conversation events
        await self.event_system.register_handler(
            event_type="conversation.completed",
            handler=self._handle_conversation_completed
        )
        
        # Register handler for scheduled persona evolution
        await self.event_system.register_handler(
            event_type="scheduler.task.persona_evolution",
            handler=self._handle_scheduled_evolution
        )
        
        logger.info("Registered persona event handlers")
    
    # === Event Handlers ===
    
    async def _handle_new_learning(self, event_data: Dict[str, Any], user_id: str):
        """Handle new learning event from SDL."""
        try:
            learning_id = event_data.get("learning_id")
            # user_id is passed as parameter from event system
            if not user_id:
                user_id = event_data.get("user_id")
            
            if not learning_id or not user_id:
                logger.warning("Missing required data in learning event")
                return
            
            # Get active persona for user
            persona = await self.persona_repo.get_persona_by_user(
                user_id=user_id,
                is_active=True
            )
            
            if not persona:
                logger.info(f"No active persona found for user {user_id}")
                return
                
            # Get learning
            learning = await self.learning_repo.get_learning(learning_id)
            if not learning:
                logger.warning(f"Learning {learning_id} not found")
                return
                
            # Check if this learning should trigger persona evolution
            # For now, we'll use a simple rule: evolve if learning confidence is high
            if learning.confidence > 0.8:
                # Schedule persona evolution
                await self._schedule_persona_evolution(
                    persona_id=persona.id,
                    user_id=user_id,
                    trigger_type="learning",
                    trigger_id=learning_id
                )
            else:
                # Log that this learning wasn't significant enough
                logger.debug(f"Learning {learning_id} not significant enough for evolution")
            
        except Exception as e:
            logger.error(f"Error handling new learning: {str(e)}", exc_info=True)
    
    async def _handle_new_reflection(self, event_data: Dict[str, Any], user_id: str):
        """Handle new reflection event."""
        try:
            reflection_id = event_data.get("reflection_id")
            # user_id is passed as parameter from event system
            if not user_id:
                user_id = event_data.get("user_id")
            
            if not reflection_id or not user_id:
                logger.warning("Missing required data in reflection event")
                return
            
            # Get active persona for user
            persona = await self.persona_repo.get_persona_by_user(
                user_id=user_id,
                is_active=True
            )
            
            if not persona:
                logger.info(f"No active persona found for user {user_id}")
                return
            
            # Skip bootstrap-triggered reflections from seeding evolution
            if self._is_bootstrap_reflection(event_data):
                logger.info(f"Skipping persona evolution scheduling for bootstrap reflection {reflection_id}")
                return

            # Only schedule evolutions when user has interacted at least once
            if self.conversation_repo:
                try:
                    has_interacted = await self.conversation_repo.has_user_messages(user_id)
                except Exception as interaction_err:
                    logger.warning(f"Failed to verify user interaction for {user_id}: {interaction_err}")
                    has_interacted = True
                if not has_interacted:
                    logger.info(
                        f"Deferring reflection-driven evolution for user {user_id}; no user messages yet"
                    )
                    return

            # Schedule persona evolution based on this reflection
            await self._schedule_persona_evolution(
                persona_id=persona.id,
                user_id=user_id,
                trigger_type="reflection",
                trigger_id=reflection_id
            )
            
        except Exception as e:
            logger.error(f"Error handling new reflection: {str(e)}", exc_info=True)
    
    async def _handle_conversation_completed(self, event_data: Dict[str, Any], user_id: str):
        """Handle completed conversation event."""
        try:
            conversation_id = event_data.get("conversation_id")
            # user_id is passed as parameter from event system
            if not user_id:
                user_id = event_data.get("user_id")
            
            if not conversation_id or not user_id:
                logger.warning("Missing required data in conversation event")
                return
            
            # For conversation events, we don't immediately trigger evolution
            # Instead, we let SDL process the conversation first
            # SDL will then create learnings which will trigger evolution
            logger.debug(f"Conversation {conversation_id} completed, waiting for SDL processing")
            
        except Exception as e:
            logger.error(f"Error handling conversation completed: {str(e)}", exc_info=True)
    
    async def _handle_scheduled_evolution(self, event_data: Dict[str, Any], user_id: str):
        """Handle scheduled persona evolution event."""
        try:
            persona_id = event_data.get("persona_id")
            # user_id is passed as parameter from event system
            if not user_id:
                user_id = event_data.get("user_id")
            trigger_type = event_data.get("trigger_type", "scheduled")
            trigger_id = event_data.get("trigger_id")
            
            if not persona_id or not user_id:
                logger.warning("Missing required data in scheduled evolution event")
                return
            
            # Execute persona evolution
            if trigger_type == "reflection" and trigger_id:
                # Fetch reflection and extract trait_changes + content
                reflection = await self.reflection_repo.get_reflection(trigger_id)
                trait_changes = []
                reflection_content = None
                reflection_themes = []
                reflection_metadata = {}

                if reflection and hasattr(reflection, 'trait_changes'):
                    trait_changes = getattr(reflection, 'trait_changes', [])
                elif reflection and isinstance(reflection, dict):
                    # Check both top-level and in result dict
                    trait_changes = reflection.get('trait_changes', [])
                    if not trait_changes and 'result' in reflection:
                        result_dict = reflection.get('result', {})
                        if isinstance(result_dict, dict):
                            trait_changes = result_dict.get('trait_changes', [])
                            reflection_metadata = result_dict.get('metadata', {})

                    # Extract reflection content and themes for evidence
                    if 'result' in reflection:
                        result_dict = reflection.get('result', {})
                        if isinstance(result_dict, dict):
                            reflection_content = result_dict.get('content', '')
                            reflection_themes = result_dict.get('themes', [])
                            reflection_metadata = result_dict.get('metadata', reflection_metadata)

                # Skip bootstrap reflections or reflections without trait deltas
                if self._is_bootstrap_reflection(event_data, reflection):
                    logger.info(f"Skipping scheduled evolution for bootstrap reflection {trigger_id}")
                    return

                if not trait_changes:
                    logger.info(
                        f"Skipping evolution for reflection {trigger_id}: no trait changes present"
                    )
                    return

                # Use saga orchestration if enabled, otherwise direct method call
                if self.use_saga and self.saga_integration:
                    logger.info(f"Using saga orchestration for reflection-driven evolution (reflection={trigger_id})")
                    saga_result = await self.saga_integration.execute_persona_evolution_saga(
                        user_id=user_id,
                        persona_id=persona_id,
                        reflection_id=trigger_id,
                        trait_changes=trait_changes,
                        reasoning=f"Reflection-driven evolution from {trigger_id}",
                        confidence=0.8,
                        correlation_id=f"reflection_{trigger_id}"
                    )
                    
                    # Convert saga result to standard result format
                    if saga_result.get('status') == 'completed':
                        result = {
                            'success': True,
                            'changed': True,
                            'persona_id': persona_id,
                            'reflection_id': trigger_id,
                            'saga_id': saga_result.get('id'),
                            'changes': saga_result.get('output_data', {})
                        }
                    else:
                        result = {
                            'success': False,
                            'error': saga_result.get('error_data', 'Saga failed'),
                            'saga_id': saga_result.get('id')
                        }
                        logger.error(f"Saga evolution failed for reflection {trigger_id}: {result.get('error')}")
                else:
                    # Direct method call (legacy path)
                    result = await self.persona_engine.evolve_persona_from_reflection(
                        persona_id=persona_id,
                        reflection_id=trigger_id,
                        trait_changes=trait_changes,
                        reflection_content=reflection_content,
                        reflection_themes=reflection_themes
                    )
            else:
                # Scheduled: run template-driven reassessment first, then refine via learnings
                reassess_changed = False
                impact_score = 0.0
                try:
                    reassess = await self.persona_engine.reassess_persona(
                        persona_id=persona_id,
                        user_id=user_id,
                        reflections=None,
                        learnings_limit=10,
                    )
                    if reassess.get("success") and reassess.get("changed"):
                        reassess_changed = True
                        impact_score = max(impact_score, 0.6)
                except Exception as e:
                    logger.warning(f"Scheduled reassessment skipped: {e}")

                # Evolve based on recent learnings as a complementary step
                result_learn = await self.persona_engine.evolve_persona_from_learnings(
                    persona_id=persona_id,
                    user_id=user_id,
                    limit=10
                )
                # Synthesize a combined result for downstream handling
                result = result_learn or {}
                if reassess_changed:
                    result["changed"] = True
                    result["success"] = True
                    # prefer max impact score
                    try:
                        result["impact_score"] = max(float(result.get("impact_score", 0.0) or 0.0), impact_score)
                    except Exception:
                        result["impact_score"] = impact_score
            
            # Check if evolution succeeded
            if result.get("success") and result.get("changed", False):
                # Publish evolution completed event
                await self.event_system.publish_event(
                    event_type="persona.evolution.completed",
                    event_data={
                        "persona_id": persona_id,
                        "user_id": user_id,
                        "evolution_id": result.get("evolution_id"),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                # If changes were significant enough, trigger prompt regeneration
                if result.get("impact_score", 0) > 0.5:
                    await self._regenerate_system_prompt(persona_id)
            
            logger.info(f"Persona evolution for {persona_id} completed")
            
        except Exception as e:
            logger.error(f"Error handling scheduled evolution: {str(e)}", exc_info=True)
    
    # === Helper Methods ===
    
    async def _schedule_persona_evolution(
        self,
        persona_id: str,
        user_id: str,
        trigger_type: str,
        trigger_id: Optional[str] = None,
        delay_seconds: int = 5
    ):
        """Schedule persona evolution with a delay."""
        try:
            # Create event data
            event_data = {
                "persona_id": persona_id,
                "user_id": user_id,
                "trigger_type": trigger_type,
                "trigger_id": trigger_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Schedule event
            await self.event_system.schedule_event(
                event_type="scheduler.task.persona_evolution",
                event_data=event_data,
                delay_seconds=delay_seconds
            )
            
            logger.info(f"Scheduled persona evolution for {persona_id} in {delay_seconds}s")
            
        except Exception as e:
            logger.error(f"Error scheduling persona evolution: {str(e)}", exc_info=True)
    
    async def _regenerate_system_prompt(self, persona_id: str):
        """Regenerate system prompt for a persona after evolution."""
        try:
            # Generate new prompt
            prompt_result = await self.persona_engine.generate_persona_prompt(persona_id)
            
            if not prompt_result.get("success"):
                logger.warning(f"Failed to regenerate prompt for persona {persona_id}")
                return
            
            # Publish prompt updated event
            await self.event_system.publish_event(
                event_type="persona.prompt.updated",
                event_data={
                    "persona_id": persona_id,
                    "system_prompt": prompt_result.get("system_prompt"),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
            logger.info(f"Regenerated system prompt for persona {persona_id}")
            
        except Exception as e:
            logger.error(f"Error regenerating system prompt: {str(e)}", exc_info=True)
    
    # === Public API Methods ===
    
    async def ensure_default_persona(self, user_id: str, name: str = "SELO") -> Dict[str, Any]:
        """
        Ensure user has a default persona, creating one if needed.
        
        Args:
            user_id: User ID
            name: Name for the persona if created
            
        Returns:
            Persona data
        """
        try:
            # Check if user already has a persona
            existing = await self.persona_repo.get_persona_by_user(
                user_id=user_id,
                is_default=True
            )
            
            if existing:
                return {
                    "success": True,
                    "persona_id": existing.id,
                    "name": existing.name,
                    "created": False
                }
            
            # Create initial persona
            persona_data = await self.persona_engine.create_initial_persona(
                user_id=user_id,
                name=name
            )
            
            return {
                "success": True,
                "persona_id": persona_data.get("id"),
                "name": persona_data.get("name"),
                "created": True
            }
            
        except Exception as e:
            logger.error(f"Error ensuring default persona: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_system_prompt(self, persona_id: str) -> Dict[str, Any]:
        """
        Get system prompt for a persona.
        
        Args:
            persona_id: Persona ID
            
        Returns:
            Dictionary with prompt and metadata
        """
        try:
            return await self.persona_engine.generate_persona_prompt(persona_id)
        except Exception as e:
            logger.error(f"Error getting system prompt: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_persona_history(self, persona_id: str) -> Dict[str, Any]:
        """
        Get evolution history for a persona.
        
        Args:
            persona_id: Persona ID
            
        Returns:
            Dictionary with evolution history
        """
        try:
            # Get persona
            persona = await self.persona_repo.get_persona(persona_id, include_traits=True)
            if not persona:
                return {
                    "success": False,
                    "error": "Persona not found"
                }
            
            # Get evolution history (ensure JSON-serializable)
            evolutions = await self.persona_repo.get_evolutions_for_persona(persona_id, limit=20)
            # Convert ORM models to dicts to avoid Pydantic serialization errors
            evolutions_serialized: List[dict] = []
            try:
                for ev in evolutions or []:
                    if hasattr(ev, "to_dict") and callable(getattr(ev, "to_dict")):
                        evolutions_serialized.append(ev.to_dict())
                    elif isinstance(ev, dict):
                        evolutions_serialized.append(ev)
                    else:
                        # Best-effort fallback serialization
                        item = {k: v for k, v in getattr(ev, "__dict__", {}).items() if not k.startswith("_")}
                        if item:
                            evolutions_serialized.append(item)
                        else:
                            evolutions_serialized.append({"id": getattr(ev, "id", None)})
            except Exception:
                # If serialization fails, fall back to empty list to keep API stable
                evolutions_serialized = []
            
            # Get trait history (aggregated across all traits)
            traits = persona.traits if persona.traits else []
            trait_histories = []
            
            # Parallelize trait history queries to reduce N+1 query latency
            # Previously: sequential queries (1-2s × 6 traits = 6-12s)
            # Now: parallel queries (max 1-2s total for all traits)
            if traits:
                import asyncio
                history_tasks = [
                    self.persona_repo.get_trait_evolution(
                        persona_id=persona_id,
                        trait_name=trait.name,
                        trait_category=trait.category,
                        limit=10
                    )
                    for trait in traits
                ]
                histories = await asyncio.gather(*history_tasks)
                
                trait_histories = [
                    {
                        "trait_id": trait.id,
                        "trait_name": trait.name,
                        "trait_category": trait.category,
                        "history": history
                    }
                    for trait, history in zip(traits, histories)
                ]
            
            result = {
                "success": True,
                "persona_id": persona_id,
                "persona": persona.to_dict(),
                "evolutions": evolutions_serialized,
                "evolution_count": len(evolutions_serialized),
                "trait_histories": trait_histories
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting persona history: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
