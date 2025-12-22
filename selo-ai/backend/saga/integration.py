"""
Saga pattern integration for SELO AI systems.

Provides high-level saga coordination for persona evolution and other
multi-step operations with automatic compensation on failure.
"""

import logging
from typing import Dict, Any, Optional

from .orchestrator import SagaOrchestrator
from .handlers import PersonaEvolutionHandlers, GoalManagementHandlers
from ..db.repositories.saga import SagaRepository
from ..db.repositories.persona import PersonaRepository
from ..db.repositories.agent_state import AgentGoalRepository, PlanStepRepository
from ..sdl.repository import LearningRepository

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
        saga_repo: Optional[SagaRepository] = None
    ):
        self.persona_repo = persona_repo or PersonaRepository()
        self.learning_repo = learning_repo or LearningRepository()
        self.goal_repo = goal_repo or AgentGoalRepository()
        self.plan_repo = plan_repo or PlanStepRepository()
        self.saga_repo = saga_repo or SagaRepository()
        
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
        
        # Register all handlers
        self._register_handlers()
        
        logger.info("Saga integration initialized")
    
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
        
        logger.debug("Registered all saga handlers")
    
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
