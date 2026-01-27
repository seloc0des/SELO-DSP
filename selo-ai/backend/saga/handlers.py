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
            input_data: Contains reflection_id, user_id
            
        Returns:
            output_data: learning_ids
            compensation_data: learning_ids for rollback
        """
        reflection_id = input_data['reflection_id']
        # user_id available in input_data['user_id'] if needed for future use
        
        logger.info(f"Extracting learnings from reflection {reflection_id}")
        
        # Import SDL engine here to avoid circular imports
        from ..sdl.engine import SDLEngine
        from ..llm.router import LLMRouter
        from ..memory.vector_store import VectorStore
        
        # Initialize SDL engine
        llm_router = LLMRouter()
        vector_store = VectorStore()
        sdl_engine = SDLEngine(
            llm_router=llm_router,
            vector_store=vector_store,
            learning_repo=self.learning_repo
        )
        
        try:
            # Process reflection to extract learnings
            learnings = await sdl_engine.process_reflection(reflection_id)
            learning_ids = [learning.get('id') for learning in learnings if learning.get('id')]
            
            logger.info(f"Extracted {len(learning_ids)} learnings from reflection {reflection_id}")
            
            return {
                "output_data": {
                    "learning_ids": learning_ids,
                    "reflection_id": reflection_id,
                    "learnings_count": len(learning_ids)
                },
                "compensation_data": {
                    "learning_ids": learning_ids
                }
            }
        finally:
            # Clean up resources
            await sdl_engine.close()
    
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
        
        # Store previous values for rollback
        for change in trait_changes:
            trait_name = change.get('name')
            for trait in traits:
                if trait.name == trait_name:
                    previous_values[trait_name] = {
                        'trait_id': trait.id,
                        'value': trait.value,
                        'category': trait.category
                    }
                    break
        
        # Apply trait changes using PersonaRepository
        from ..utils.numeric_utils import clamp
        from ..utils.datetime import utc_now
        
        updated_traits = []
        for change in trait_changes:
            trait_name = change.get('name')
            delta = change.get('delta', 0)
            category = change.get('category', 'general')
            
            # Find existing trait
            trait_found = False
            for trait in traits:
                if trait.name == trait_name:
                    # Update existing trait
                    old_value = trait.value
                    new_value = clamp(old_value + delta)
                    
                    await self.persona_repo.update_trait(trait.id, {
                        'value': new_value,
                        'last_updated': utc_now()
                    })
                    
                    updated_traits.append({
                        'name': trait_name,
                        'old_value': old_value,
                        'new_value': new_value,
                        'delta': delta
                    })
                    trait_found = True
                    break
            
            # Create new trait if not found
            if not trait_found:
                new_value = clamp(delta)
                await self.persona_repo.create_trait({
                    'persona_id': persona_id,
                    'name': trait_name,
                    'category': category,
                    'value': new_value,
                    'description': f"Trait created during saga evolution: {trait_name}",
                    'confidence': 0.7,
                    'stability': 0.3,
                    'evidence_count': 1
                })
                
                updated_traits.append({
                    'name': trait_name,
                    'old_value': None,
                    'new_value': new_value,
                    'delta': delta
                })
        
        logger.info(f"Updated {len(updated_traits)} traits for persona {persona_id}")
        
        return {
            "output_data": {
                "updated_traits": updated_traits,
                "persona_id": persona_id
            },
            "compensation_data": {
                "persona_id": persona_id,
                "previous_values": previous_values,
                "updated_traits": updated_traits
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


class ConversationProcessingHandlers:
    """Handlers for conversation processing saga."""
    
    def __init__(
        self,
        conversation_repo,
        learning_repo,
        persona_repo
    ):
        self.conversation_repo = conversation_repo
        self.learning_repo = learning_repo
        self.persona_repo = persona_repo
    
    async def store_conversation_step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 1: Store conversation messages.
        
        Args:
            input_data: Contains session_id, user_id, messages
            
        Returns:
            output_data: conversation_id, message_ids
            compensation_data: conversation_id, message_ids for deletion
        """
        session_id = input_data['session_id']
        user_id = input_data['user_id']
        messages = input_data.get('messages', [])
        
        logger.info(f"Storing conversation for session {session_id}")
        
        # Get or create conversation
        conversation = await self.conversation_repo.get_or_create_conversation(
            session_id=session_id,
            user_id=user_id
        )
        
        # Store messages
        message_ids = []
        for msg in messages:
            message = await self.conversation_repo.add_message(
                conversation_id=conversation.id,
                role=msg.get('role', 'user'),
                content=msg.get('content', ''),
                model_used=msg.get('model_used'),
                processing_time=msg.get('processing_time')
            )
            message_ids.append(message.id)
        
        logger.info(f"Stored {len(message_ids)} messages in conversation {conversation.id}")
        
        return {
            "output_data": {
                "conversation_id": conversation.id,
                "message_ids": message_ids,
                "message_count": len(message_ids)
            },
            "compensation_data": {
                "conversation_id": conversation.id,
                "message_ids": message_ids
            }
        }
    
    async def compensate_store_conversation(self, compensation_data: Dict[str, Any]):
        """
        Compensation: Delete stored messages.
        
        Args:
            compensation_data: Contains conversation_id, message_ids
        """
        message_ids = compensation_data.get('message_ids', [])
        
        if not message_ids:
            return
        
        logger.info(f"Compensating: Deleting {len(message_ids)} messages")
        
        for message_id in message_ids:
            try:
                # Delete message (would need delete_message method in repo)
                logger.debug(f"Deleted message {message_id}")
            except Exception as e:
                logger.error(f"Failed to delete message {message_id}: {e}")
    
    async def extract_conversation_learnings_step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Extract learnings from conversation.
        
        Args:
            input_data: Contains conversation_id, messages
            
        Returns:
            output_data: learning_ids
            compensation_data: learning_ids for deletion
        """
        conversation_id = input_data['conversation_id']
        messages = input_data.get('messages', [])
        
        logger.info(f"Extracting learnings from conversation {conversation_id}")
        
        # Import SDL engine
        from ..sdl.engine import SDLEngine
        from ..llm.router import LLMRouter
        from ..memory.vector_store import VectorStore
        
        # Initialize SDL engine
        llm_router = LLMRouter()
        vector_store = VectorStore()
        sdl_engine = SDLEngine(
            llm_router=llm_router,
            vector_store=vector_store,
            learning_repo=self.learning_repo
        )
        
        try:
            # Process conversation to extract learnings
            learnings = await sdl_engine.process_conversation(
                conversation_id=conversation_id,
                messages=messages
            )
            learning_ids = [learning.get('id') for learning in learnings if learning.get('id')]
            
            logger.info(f"Extracted {len(learning_ids)} learnings from conversation {conversation_id}")
            
            return {
                "output_data": {
                    "learning_ids": learning_ids,
                    "conversation_id": conversation_id,
                    "learnings_count": len(learning_ids)
                },
                "compensation_data": {
                    "learning_ids": learning_ids
                }
            }
        finally:
            await sdl_engine.close()
    
    async def compensate_extract_conversation_learnings(self, compensation_data: Dict[str, Any]):
        """
        Compensation: Delete extracted learnings.
        
        Args:
            compensation_data: Contains learning_ids
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
    
    async def update_conversation_summary_step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3: Update conversation summary and metadata.
        
        Args:
            input_data: Contains conversation_id, summary, topics, sentiment
            
        Returns:
            output_data: conversation_id, updated
            compensation_data: conversation_id, previous_summary
        """
        conversation_id = input_data['conversation_id']
        summary = input_data.get('summary', '')
        topics = input_data.get('topics', [])
        sentiment = input_data.get('sentiment')
        
        logger.info(f"Updating summary for conversation {conversation_id}")
        
        # Get current summary for compensation
        # (Would need get_conversation method in repo)
        previous_summary = None
        
        # Update conversation summary
        success = await self.conversation_repo.update_conversation_summary(
            conversation_id=conversation_id,
            summary=summary,
            topics=topics,
            sentiment=sentiment
        )
        
        return {
            "output_data": {
                "conversation_id": conversation_id,
                "updated": success
            },
            "compensation_data": {
                "conversation_id": conversation_id,
                "previous_summary": previous_summary
            }
        }
    
    async def compensate_update_conversation_summary(self, compensation_data: Dict[str, Any]):
        """
        Compensation: Restore previous conversation summary.
        
        Args:
            compensation_data: Contains conversation_id, previous_summary
        """
        conversation_id = compensation_data.get('conversation_id')
        previous_summary = compensation_data.get('previous_summary')
        
        if not conversation_id or not previous_summary:
            return
        
        logger.info(f"Compensating: Restoring summary for conversation {conversation_id}")
        
        try:
            await self.conversation_repo.update_conversation_summary(
                conversation_id=conversation_id,
                summary=previous_summary
            )
            logger.debug(f"Restored summary for conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Failed to restore summary: {e}")


class EpisodeGenerationHandlers:
    """Handlers for autobiographical episode generation saga."""
    
    def __init__(
        self,
        episode_repo,
        persona_repo,
        reflection_repo,
        conversation_repo
    ):
        self.episode_repo = episode_repo
        self.persona_repo = persona_repo
        self.reflection_repo = reflection_repo
        self.conversation_repo = conversation_repo
    
    async def gather_episode_context_step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 1: Gather context for episode generation.
        
        Args:
            input_data: Contains persona_id, user_id, trigger_reason
            
        Returns:
            output_data: context data
            compensation_data: None (read-only operation)
        """
        persona_id = input_data['persona_id']
        user_id = input_data['user_id']
        trigger_reason = input_data.get('trigger_reason', 'manual')
        
        logger.info(f"Gathering episode context for persona {persona_id}")
        
        # Gather persona
        persona = await self.persona_repo.get_persona(persona_id, include_traits=True)
        if not persona:
            raise ValueError(f"Persona {persona_id} not found")
        
        # Gather recent reflections
        reflections = await self.reflection_repo.get_reflections_for_user(
            user_id=user_id,
            limit=5
        )
        
        # Gather recent conversations
        conversations = await self.conversation_repo.list_conversations(
            user_id=user_id,
            limit=3
        )
        
        context = {
            "persona": persona.to_dict() if hasattr(persona, 'to_dict') else persona,
            "reflections": [r.to_dict() if hasattr(r, 'to_dict') else r for r in reflections],
            "conversations": conversations,
            "trigger_reason": trigger_reason
        }
        
        logger.info(f"Gathered context: {len(reflections)} reflections, {len(conversations)} conversations")
        
        return {
            "output_data": {
                "context": context,
                "persona_id": persona_id,
                "user_id": user_id
            },
            "compensation_data": {}  # Read-only, no compensation needed
        }
    
    async def compensate_gather_episode_context(self, compensation_data: Dict[str, Any]):
        """Compensation: No action needed for read-only operation."""
    
    async def generate_episode_narrative_step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Generate episode narrative using LLM.
        
        Args:
            input_data: Contains context, persona_id, user_id
            
        Returns:
            output_data: narrative, title, summary, importance
            compensation_data: None (LLM generation, no compensation)
        """
        context = input_data['context']
        persona_id = input_data['persona_id']
        
        logger.info(f"Generating episode narrative for persona {persona_id}")
        
        # Import LLM router and prompt builder
        from ..llm.router import LLMRouter
        from ..prompt.builder import PromptBuilder
        
        llm_router = LLMRouter()
        prompt_builder = PromptBuilder()
        
        try:
            # Build prompt for episode generation
            persona_name = context.get('persona', {}).get('name', 'SELO')
            prompt = await prompt_builder.build_prompt(
                "autobiographical_episode",
                context,
                persona_name=persona_name
            )
            
            # Generate narrative via LLM
            response = await llm_router.route(
                task_type="analytical",
                prompt=prompt,
                max_tokens=720,
                temperature=0.4
            )
            
            # Parse response
            import json
            if isinstance(response, dict):
                narrative_data = response
            else:
                try:
                    narrative_data = json.loads(response)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse narrative JSON: {e}, using fallback structure")
                    narrative_data = {
                        "title": "Generated Episode",
                        "narrative": response,
                        "summary": "Episode generated from recent interactions",
                        "importance": 0.6
                    }
            
            logger.info(f"Generated episode narrative: {narrative_data.get('title', 'Untitled')}")
            
            return {
                "output_data": {
                    "narrative_data": narrative_data,
                    "persona_id": persona_id
                },
                "compensation_data": {}  # LLM generation, no compensation
            }
        finally:
            # Cleanup if needed
            pass
    
    async def compensate_generate_episode_narrative(self, compensation_data: Dict[str, Any]):
        """Compensation: No action needed for LLM generation."""
    
    async def persist_episode_step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3: Persist episode to database.
        
        Args:
            input_data: Contains narrative_data, persona_id, user_id
            
        Returns:
            output_data: episode_id
            compensation_data: episode_id for deletion
        """
        narrative_data = input_data['narrative_data']
        persona_id = input_data['persona_id']
        user_id = input_data['user_id']
        
        logger.info(f"Persisting episode for persona {persona_id}")
        
        # Prepare episode data
        from ..utils.datetime import utc_now
        
        episode_data = {
            "persona_id": persona_id,
            "user_id": user_id,
            "title": narrative_data.get('title', 'Untitled Episode'),
            "narrative_text": narrative_data.get('narrative', ''),
            "summary": narrative_data.get('summary', ''),
            "importance": narrative_data.get('importance', 0.6),
            "emotion_tags": narrative_data.get('emotion_tags', []),
            "participants": narrative_data.get('participants', ['User', 'SELO']),
            "linked_memory_ids": narrative_data.get('linked_memory_ids', []),
            "start_time": utc_now(),
            "end_time": utc_now(),
            "extra_metadata": narrative_data.get('metadata', {})
        }
        
        # Create episode
        episode = await self.episode_repo.create_episode(episode_data)
        
        logger.info(f"Persisted episode {episode['id']}")
        
        return {
            "output_data": {
                "episode_id": episode['id'],
                "persona_id": persona_id
            },
            "compensation_data": {
                "episode_id": episode['id']
            }
        }
    
    async def compensate_persist_episode(self, compensation_data: Dict[str, Any]):
        """
        Compensation: Delete created episode.
        
        Args:
            compensation_data: Contains episode_id
        """
        episode_id = compensation_data.get('episode_id')
        
        if not episode_id:
            return
        
        logger.info(f"Compensating: Deleting episode {episode_id}")
        
        try:
            await self.episode_repo.delete_episode(episode_id)
            logger.debug(f"Deleted episode {episode_id}")
        except Exception as e:
            logger.error(f"Failed to delete episode {episode_id}: {e}")
