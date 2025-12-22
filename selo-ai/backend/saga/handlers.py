"""
Saga step handlers and compensation handlers for SELO AI operations.

This module contains the forward execution handlers and their corresponding
compensation handlers for various saga operations.
"""

import logging
from typing import Dict, Any

from ..db.repositories.persona import PersonaRepository
from ..db.repositories.agent_state import AgentGoalRepository, PlanStepRepository
from ..sdl.repository import LearningRepository

logger = logging.getLogger("selo.saga.handlers")


class PersonaEvolutionHandlers:
    """Handlers for persona evolution saga."""
    
    def __init__(
        self,
        persona_repo: PersonaRepository,
        learning_repo: LearningRepository
    ):
        self.persona_repo = persona_repo
        self.learning_repo = learning_repo
    
    async def extract_learnings_step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 1: Extract learnings from reflection.
        
        Args:
            input_data: Contains reflection_id
            
        Returns:
            output_data: learning_ids
            compensation_data: learning_ids for rollback
        """
        reflection_id = input_data['reflection_id']
        user_id = input_data['user_id']
        
        logger.info(f"Extracting learnings from reflection {reflection_id}")
        
        # This would call the SDL engine to extract learnings
        # For now, we'll simulate the operation
        learning_ids = []  # Would be populated by SDL engine
        
        return {
            "output_data": {
                "learning_ids": learning_ids,
                "reflection_id": reflection_id
            },
            "compensation_data": {
                "learning_ids": learning_ids
            }
        }
    
    async def compensate_extract_learnings(self, compensation_data: Dict[str, Any]):
        """
        Compensation: Delete created learnings.
        
        Args:
            compensation_data: Contains learning_ids to delete
        """
        learning_ids = compensation_data.get('learning_ids', [])
        
        if not learning_ids:
            return
        
        logger.info(f"Compensating: Deleting {len(learning_ids)} learnings")
        
        for learning_id in learning_ids:
            try:
                await self.learning_repo.delete_learning(learning_id)
                logger.debug(f"Deleted learning {learning_id}")
            except Exception as e:
                logger.error(f"Failed to delete learning {learning_id}: {e}")
    
    async def update_persona_traits_step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Update persona traits based on learnings.
        
        Args:
            input_data: Contains persona_id, trait_changes
            
        Returns:
            output_data: updated_traits
            compensation_data: previous_trait_values for rollback
        """
        persona_id = input_data['persona_id']
        trait_changes = input_data.get('trait_changes', [])
        
        logger.info(f"Updating {len(trait_changes)} traits for persona {persona_id}")
        
        # Get current trait values for compensation
        persona = await self.persona_repo.get_persona(persona_id, include_traits=True)
        if not persona:
            raise ValueError(f"Persona {persona_id} not found")
        
        previous_values = {}
        traits = getattr(persona, 'traits', [])
        
        for change in trait_changes:
            trait_name = change.get('name')
            # Find existing trait
            for trait in traits:
                if trait.name == trait_name:
                    previous_values[trait_name] = {
                        'trait_id': trait.id,
                        'value': trait.value
                    }
                    break
        
        # Apply trait changes (would use PersonaEngine in real implementation)
        updated_traits = []
        for change in trait_changes:
            # Simulate trait update
            updated_traits.append({
                'name': change['name'],
                'new_value': change.get('delta', 0)
            })
        
        return {
            "output_data": {
                "updated_traits": updated_traits,
                "persona_id": persona_id
            },
            "compensation_data": {
                "persona_id": persona_id,
                "previous_values": previous_values
            }
        }
    
    async def compensate_update_persona_traits(self, compensation_data: Dict[str, Any]):
        """
        Compensation: Restore previous trait values.
        
        Args:
            compensation_data: Contains persona_id and previous_values
        """
        persona_id = compensation_data.get('persona_id')
        previous_values = compensation_data.get('previous_values', {})
        
        if not previous_values:
            return
        
        logger.info(f"Compensating: Restoring {len(previous_values)} trait values for persona {persona_id}")
        
        for trait_name, trait_data in previous_values.items():
            try:
                trait_id = trait_data['trait_id']
                old_value = trait_data['value']
                
                await self.persona_repo.update_trait(trait_id, {'value': old_value})
                logger.debug(f"Restored trait {trait_name} to value {old_value}")
            except Exception as e:
                logger.error(f"Failed to restore trait {trait_name}: {e}")
    
    async def create_evolution_record_step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3: Create persona evolution audit record.
        
        Args:
            input_data: Contains persona_id, changes, reasoning
            
        Returns:
            output_data: evolution_id
            compensation_data: evolution_id for deletion
        """
        persona_id = input_data['persona_id']
        changes = input_data.get('changes', {})
        reasoning = input_data.get('reasoning', '')
        
        logger.info(f"Creating evolution record for persona {persona_id}")
        
        evolution_data = {
            'persona_id': persona_id,
            'changes': changes,
            'reasoning': reasoning,
            'source_type': 'saga_orchestrated',
            'confidence': input_data.get('confidence', 0.7),
            'impact_score': input_data.get('impact_score', 0.5)
        }
        
        # Create evolution record
        evolution = await self.persona_repo.create_evolution(evolution_data)
        
        return {
            "output_data": {
                "evolution_id": evolution.id,
                "persona_id": persona_id
            },
            "compensation_data": {
                "evolution_id": evolution.id
            }
        }
    
    async def compensate_create_evolution_record(self, compensation_data: Dict[str, Any]):
        """
        Compensation: Delete evolution record.
        
        Args:
            compensation_data: Contains evolution_id to delete
        """
        evolution_id = compensation_data.get('evolution_id')
        
        if not evolution_id:
            return
        
        logger.info(f"Compensating: Deleting evolution record {evolution_id}")
        
        try:
            await self.persona_repo.delete_evolution(evolution_id)
            logger.debug(f"Deleted evolution record {evolution_id}")
        except Exception as e:
            logger.error(f"Failed to delete evolution record {evolution_id}: {e}")


class GoalManagementHandlers:
    """Handlers for goal management saga."""
    
    def __init__(
        self,
        goal_repo: AgentGoalRepository,
        plan_repo: PlanStepRepository
    ):
        self.goal_repo = goal_repo
        self.plan_repo = plan_repo
    
    async def create_goal_step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 1: Create agent goal.
        
        Args:
            input_data: Goal configuration
            
        Returns:
            output_data: goal_id
            compensation_data: goal_id for deletion
        """
        logger.info(f"Creating goal for persona {input_data.get('persona_id')}")
        
        goal = await self.goal_repo.create_goal(input_data)
        
        return {
            "output_data": {
                "goal_id": goal['id']
            },
            "compensation_data": {
                "goal_id": goal['id']
            }
        }
    
    async def compensate_create_goal(self, compensation_data: Dict[str, Any]):
        """
        Compensation: Delete created goal.
        
        Args:
            compensation_data: Contains goal_id
        """
        goal_id = compensation_data.get('goal_id')
        
        if not goal_id:
            return
        
        logger.info(f"Compensating: Deleting goal {goal_id}")
        
        try:
            # Archive goal instead of hard delete
            await self.goal_repo.set_goal_status(goal_id, "cancelled", progress=0.0)
            logger.debug(f"Cancelled goal {goal_id}")
        except Exception as e:
            logger.error(f"Failed to cancel goal {goal_id}: {e}")
    
    async def create_plan_steps_step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Create plan steps for goal.
        
        Args:
            input_data: Contains goal_id, steps
            
        Returns:
            output_data: step_ids
            compensation_data: step_ids for deletion
        """
        goal_id = input_data['goal_id']
        steps = input_data.get('steps', [])
        
        logger.info(f"Creating {len(steps)} plan steps for goal {goal_id}")
        
        step_ids = []
        for step_data in steps:
            step_data['goal_id'] = goal_id
            step = await self.plan_repo.create_step(step_data)
            step_ids.append(step['id'])
        
        return {
            "output_data": {
                "step_ids": step_ids,
                "goal_id": goal_id
            },
            "compensation_data": {
                "step_ids": step_ids
            }
        }
    
    async def compensate_create_plan_steps(self, compensation_data: Dict[str, Any]):
        """
        Compensation: Delete created plan steps.
        
        Args:
            compensation_data: Contains step_ids
        """
        step_ids = compensation_data.get('step_ids', [])
        
        if not step_ids:
            return
        
        logger.info(f"Compensating: Deleting {len(step_ids)} plan steps")
        
        for step_id in step_ids:
            try:
                await self.plan_repo.update_step(step_id, {'status': 'cancelled'})
                logger.debug(f"Cancelled plan step {step_id}")
            except Exception as e:
                logger.error(f"Failed to cancel plan step {step_id}: {e}")
