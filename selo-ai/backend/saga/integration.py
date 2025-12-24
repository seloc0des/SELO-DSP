"""
Saga pattern integration for SELO AI systems.

Provides high-level saga coordination for persona evolution and other
multi-step operations with automatic compensation on failure.
"""

import logging
from typing import Dict, Any, Optional, List

from .orchestrator import SagaOrchestrator
from .handlers import (
    PersonaEvolutionHandlers,
    GoalManagementHandlers,
    ConversationProcessingHandlers,
    EpisodeGenerationHandlers
)
from ..db.repositories.saga import SagaRepository
from ..db.repositories.persona import PersonaRepository
from ..db.repositories.agent_state import (
    AgentGoalRepository,
    PlanStepRepository,
    AutobiographicalEpisodeRepository
)
from ..sdl.repository import LearningRepository
from ..db.repositories.conversation import ConversationRepository
from ..db.repositories.reflection import ReflectionRepository

logger = logging.getLogger("selo.saga.integration")


class SagaIntegration:
    """
    High-level saga integration for SELO AI operations.
    
    Provides convenience methods for executing common saga patterns
    with proper handler registration and error handling.
    """
    
    def __init__(
        self,
        persona_repo: Optional[PersonaRepository] = None,
        learning_repo: Optional[LearningRepository] = None,
        goal_repo: Optional[AgentGoalRepository] = None,
        plan_repo: Optional[PlanStepRepository] = None,
        saga_repo: Optional[SagaRepository] = None,
        conversation_repo: Optional[ConversationRepository] = None,
        episode_repo: Optional[AutobiographicalEpisodeRepository] = None,
        reflection_repo: Optional[ReflectionRepository] = None
    ):
        self.persona_repo = persona_repo or PersonaRepository()
        self.learning_repo = learning_repo or LearningRepository()
        self.goal_repo = goal_repo or AgentGoalRepository()
        self.plan_repo = plan_repo or PlanStepRepository()
        self.saga_repo = saga_repo or SagaRepository()
        self.conversation_repo = conversation_repo or ConversationRepository()
        self.episode_repo = episode_repo or AutobiographicalEpisodeRepository()
        self.reflection_repo = reflection_repo or ReflectionRepository()
        
        # Initialize orchestrator
        self.orchestrator = SagaOrchestrator(self.saga_repo)
        
        # Initialize handlers
        self.persona_handlers = PersonaEvolutionHandlers(
            self.persona_repo,
            self.learning_repo
        )
        self.goal_handlers = GoalManagementHandlers(
            self.goal_repo,
            self.plan_repo
        )
        self.conversation_handlers = ConversationProcessingHandlers(
            self.conversation_repo,
            self.learning_repo,
            self.persona_repo
        )
        self.episode_handlers = EpisodeGenerationHandlers(
            self.episode_repo,
            self.persona_repo,
            self.reflection_repo,
            self.conversation_repo
        )
        
        # Register all handlers
        self._register_handlers()
        
        logger.info("Saga integration initialized with all handlers")
    
    def _register_handlers(self):
        """Register all step and compensation handlers."""
        # Persona evolution handlers
        self.orchestrator.register_step_handler(
            "extract_learnings",
            self.persona_handlers.extract_learnings_step
        )
        self.orchestrator.register_compensation_handler(
            "compensate_extract_learnings",
            self.persona_handlers.compensate_extract_learnings
        )
        
        self.orchestrator.register_step_handler(
            "update_persona_traits",
            self.persona_handlers.update_persona_traits_step
        )
        self.orchestrator.register_compensation_handler(
            "compensate_update_persona_traits",
            self.persona_handlers.compensate_update_persona_traits
        )
        
        self.orchestrator.register_step_handler(
            "create_evolution_record",
            self.persona_handlers.create_evolution_record_step
        )
        self.orchestrator.register_compensation_handler(
            "compensate_create_evolution_record",
            self.persona_handlers.compensate_create_evolution_record
        )
        
        # Goal management handlers
        self.orchestrator.register_step_handler(
            "create_goal",
            self.goal_handlers.create_goal_step
        )
        self.orchestrator.register_compensation_handler(
            "compensate_create_goal",
            self.goal_handlers.compensate_create_goal
        )
        
        self.orchestrator.register_step_handler(
            "create_plan_steps",
            self.goal_handlers.create_plan_steps_step
        )
        self.orchestrator.register_compensation_handler(
            "compensate_create_plan_steps",
            self.goal_handlers.compensate_create_plan_steps
        )
        
        # Conversation processing handlers
        self.orchestrator.register_step_handler(
            "store_conversation",
            self.conversation_handlers.store_conversation_step
        )
        self.orchestrator.register_compensation_handler(
            "compensate_store_conversation",
            self.conversation_handlers.compensate_store_conversation
        )
        
        self.orchestrator.register_step_handler(
            "extract_conversation_learnings",
            self.conversation_handlers.extract_conversation_learnings_step
        )
        self.orchestrator.register_compensation_handler(
            "compensate_extract_conversation_learnings",
            self.conversation_handlers.compensate_extract_conversation_learnings
        )
        
        self.orchestrator.register_step_handler(
            "update_conversation_summary",
            self.conversation_handlers.update_conversation_summary_step
        )
        self.orchestrator.register_compensation_handler(
            "compensate_update_conversation_summary",
            self.conversation_handlers.compensate_update_conversation_summary
        )
        
        # Episode generation handlers
        self.orchestrator.register_step_handler(
            "gather_episode_context",
            self.episode_handlers.gather_episode_context_step
        )
        self.orchestrator.register_compensation_handler(
            "compensate_gather_episode_context",
            self.episode_handlers.compensate_gather_episode_context
        )
        
        self.orchestrator.register_step_handler(
            "generate_episode_narrative",
            self.episode_handlers.generate_episode_narrative_step
        )
        self.orchestrator.register_compensation_handler(
            "compensate_generate_episode_narrative",
            self.episode_handlers.compensate_generate_episode_narrative
        )
        
        self.orchestrator.register_step_handler(
            "persist_episode",
            self.episode_handlers.persist_episode_step
        )
        self.orchestrator.register_compensation_handler(
            "compensate_persist_episode",
            self.episode_handlers.compensate_persist_episode
        )
        
        logger.debug("Registered all saga handlers (persona, goal, conversation, episode)")
    
    async def execute_persona_evolution_saga(
        self,
        user_id: str,
        persona_id: str,
        reflection_id: str,
        trait_changes: list,
        reasoning: str,
        confidence: float = 0.7,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute persona evolution as a saga with automatic compensation.
        
        This orchestrates the multi-step process:
        1. Extract learnings from reflection
        2. Update persona traits
        3. Create evolution audit record
        
        If any step fails, all completed steps are automatically compensated.
        
        Args:
            user_id: User ID
            persona_id: Persona ID to evolve
            reflection_id: Source reflection ID
            trait_changes: List of trait changes to apply
            reasoning: Evolution reasoning
            confidence: Confidence score
            correlation_id: Optional correlation ID
            
        Returns:
            Saga execution result with final state
        """
        logger.info(
            f"Starting persona evolution saga for persona {persona_id} "
            f"from reflection {reflection_id}"
        )
        
        # Define saga steps
        steps = [
            {
                "step_name": "extract_learnings",
                "step_type": "extract_learnings",
                "input_data": {
                    "reflection_id": reflection_id,
                    "user_id": user_id
                },
                "compensation_handler": "compensate_extract_learnings",
                "max_retries": 2
            },
            {
                "step_name": "update_persona_traits",
                "step_type": "update_persona_traits",
                "input_data": {
                    "persona_id": persona_id,
                    "trait_changes": trait_changes
                },
                "compensation_handler": "compensate_update_persona_traits",
                "max_retries": 3
            },
            {
                "step_name": "create_evolution_record",
                "step_type": "create_evolution_record",
                "input_data": {
                    "persona_id": persona_id,
                    "changes": {"traits": trait_changes},
                    "reasoning": reasoning,
                    "confidence": confidence
                },
                "compensation_handler": "compensate_create_evolution_record",
                "max_retries": 2
            }
        ]
        
        # Create and execute saga
        saga_id = await self.orchestrator.create_saga(
            saga_type="persona_evolution",
            user_id=user_id,
            input_data={
                "persona_id": persona_id,
                "reflection_id": reflection_id,
                "trait_changes": trait_changes,
                "reasoning": reasoning
            },
            steps=steps,
            persona_id=persona_id,
            correlation_id=correlation_id
        )
        
        result = await self.orchestrator.execute_saga(saga_id)
        
        logger.info(
            f"Persona evolution saga {saga_id} completed with status: {result['status']}"
        )
        
        return result
    
    async def execute_goal_creation_saga(
        self,
        user_id: str,
        persona_id: str,
        goal_data: Dict[str, Any],
        plan_steps: list,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute goal creation with plan steps as a saga.
        
        This orchestrates:
        1. Create agent goal
        2. Create plan steps for the goal
        
        If step 2 fails, the goal is automatically cancelled.
        
        Args:
            user_id: User ID
            persona_id: Persona ID
            goal_data: Goal configuration
            plan_steps: List of plan step configurations
            correlation_id: Optional correlation ID
            
        Returns:
            Saga execution result
        """
        logger.info(f"Starting goal creation saga for persona {persona_id}")
        
        # Ensure persona_id is in goal_data
        goal_data['persona_id'] = persona_id
        goal_data['user_id'] = user_id
        
        # Note: For goal creation saga, we need to pass goal_id from step 1 to step 2
        # This is handled by the orchestrator which passes output_data from previous steps
        # The create_plan_steps handler should extract goal_id from the saga context
        steps = [
            {
                "step_name": "create_goal",
                "step_type": "create_goal",
                "input_data": goal_data,
                "compensation_handler": "compensate_create_goal",
                "max_retries": 2
            },
            {
                "step_name": "create_plan_steps",
                "step_type": "create_plan_steps",
                "input_data": {
                    "steps": plan_steps,
                    "persona_id": persona_id
                    # goal_id will be added by orchestrator from previous step output
                },
                "compensation_handler": "compensate_create_plan_steps",
                "max_retries": 2
            }
        ]
        
        saga_id = await self.orchestrator.create_saga(
            saga_type="goal_creation",
            user_id=user_id,
            input_data={
                "goal_data": goal_data,
                "plan_steps": plan_steps
            },
            steps=steps,
            persona_id=persona_id,
            correlation_id=correlation_id
        )
        
        result = await self.orchestrator.execute_saga(saga_id)
        
        logger.info(
            f"Goal creation saga {saga_id} completed with status: {result['status']}"
        )
        
        return result
    
    async def execute_conversation_processing_saga(
        self,
        user_id: str,
        session_id: str,
        messages: List[Dict[str, Any]],
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute conversation processing as a saga with automatic compensation.
        
        This orchestrates:
        1. Store conversation messages
        2. Extract learnings from conversation
        3. Update conversation summary
        
        If any step fails, all completed steps are automatically compensated.
        
        Args:
            user_id: User ID
            session_id: Session identifier
            messages: List of message objects
            correlation_id: Optional correlation ID
            
        Returns:
            Saga execution result with final state
        """
        logger.info(
            f"Starting conversation processing saga for session {session_id}"
        )
        
        # Define saga steps
        steps = [
            {
                "step_name": "store_conversation",
                "step_type": "store_conversation",
                "input_data": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "messages": messages
                },
                "compensation_handler": "compensate_store_conversation",
                "max_retries": 2
            },
            {
                "step_name": "extract_conversation_learnings",
                "step_type": "extract_conversation_learnings",
                "input_data": {
                    "messages": messages
                    # conversation_id will be added by orchestrator from previous step
                },
                "compensation_handler": "compensate_extract_conversation_learnings",
                "max_retries": 3
            },
            {
                "step_name": "update_conversation_summary",
                "step_type": "update_conversation_summary",
                "input_data": {
                    "summary": "Conversation processed",
                    "topics": [],
                    "sentiment": None
                    # conversation_id will be added by orchestrator
                },
                "compensation_handler": "compensate_update_conversation_summary",
                "max_retries": 2
            }
        ]
        
        # Create and execute saga
        saga_id = await self.orchestrator.create_saga(
            saga_type="conversation_processing",
            user_id=user_id,
            input_data={
                "session_id": session_id,
                "messages": messages,
                "message_count": len(messages)
            },
            steps=steps,
            correlation_id=correlation_id
        )
        
        result = await self.orchestrator.execute_saga(saga_id)
        
        logger.info(
            f"Conversation processing saga {saga_id} completed with status: {result['status']}"
        )
        
        return result
    
    async def execute_episode_generation_saga(
        self,
        user_id: str,
        persona_id: str,
        trigger_reason: str = "manual",
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute episode generation as a saga with automatic compensation.
        
        This orchestrates:
        1. Gather context (persona, reflections, conversations)
        2. Generate narrative using LLM
        3. Persist episode to database
        
        If any step fails, all completed steps are automatically compensated.
        
        Args:
            user_id: User ID
            persona_id: Persona ID
            trigger_reason: Reason for episode generation
            correlation_id: Optional correlation ID
            
        Returns:
            Saga execution result with final state
        """
        logger.info(
            f"Starting episode generation saga for persona {persona_id}"
        )
        
        # Define saga steps
        steps = [
            {
                "step_name": "gather_episode_context",
                "step_type": "gather_episode_context",
                "input_data": {
                    "persona_id": persona_id,
                    "user_id": user_id,
                    "trigger_reason": trigger_reason
                },
                "compensation_handler": "compensate_gather_episode_context",
                "max_retries": 2
            },
            {
                "step_name": "generate_episode_narrative",
                "step_type": "generate_episode_narrative",
                "input_data": {
                    # context, persona_id, user_id will be added from previous step
                },
                "compensation_handler": "compensate_generate_episode_narrative",
                "max_retries": 3
            },
            {
                "step_name": "persist_episode",
                "step_type": "persist_episode",
                "input_data": {
                    "user_id": user_id
                    # narrative_data, persona_id will be added from previous step
                },
                "compensation_handler": "compensate_persist_episode",
                "max_retries": 2
            }
        ]
        
        # Create and execute saga
        saga_id = await self.orchestrator.create_saga(
            saga_type="episode_generation",
            user_id=user_id,
            input_data={
                "persona_id": persona_id,
                "trigger_reason": trigger_reason
            },
            steps=steps,
            persona_id=persona_id,
            correlation_id=correlation_id
        )
        
        result = await self.orchestrator.execute_saga(saga_id)
        
        logger.info(
            f"Episode generation saga {saga_id} completed with status: {result['status']}"
        )
        
        return result
    
    async def get_saga_status(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a saga.
        
        Args:
            saga_id: Saga ID
            
        Returns:
            Saga state with all steps
        """
        return await self.saga_repo.get_saga(saga_id)
    
    async def retry_failed_saga(self, saga_id: str) -> Dict[str, Any]:
        """
        Retry a failed saga.
        
        Args:
            saga_id: ID of failed saga
            
        Returns:
            Updated saga state after retry
        """
        logger.info(f"Retrying failed saga {saga_id}")
        return await self.orchestrator.retry_failed_saga(saga_id)
    
    async def list_active_sagas(
        self,
        saga_type: Optional[str] = None
    ) -> list:
        """
        List all active sagas.
        
        Args:
            saga_type: Optional filter by saga type
            
        Returns:
            List of active sagas
        """
        return await self.saga_repo.list_active_sagas(saga_type=saga_type)
    
    async def list_failed_sagas(
        self,
        saga_type: Optional[str] = None
    ) -> list:
        """
        List failed sagas for monitoring.
        
        Args:
            saga_type: Optional filter by saga type
            
        Returns:
            List of failed sagas
        """
        return await self.saga_repo.list_failed_sagas(saga_type=saga_type)
