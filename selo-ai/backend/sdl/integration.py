"""
SDL Integration module for connecting the Self-Development Learning system
with other SELO AI components like the reflection system and conversation handlers.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Set, Tuple

from ..llm.router import LLMRouter
from ..memory.vector_store import VectorStore
from ..db.repositories.reflection import ReflectionRepository
from ..db.repositories.user import UserRepository
from ..scheduler.event_triggers import EventType, EventTriggerSystem
from .repository import LearningRepository
from .concept_mapper import ConceptMapper
from .engine import SDLEngine

logger = logging.getLogger("selo.sdl.integration")

class SDLIntegration:
    """
    Integration layer for the Self-Development Learning system.
    """
    def __init__(
        self,
        llm_router: LLMRouter,
        vector_store: VectorStore,
        event_system: Optional[EventTriggerSystem] = None,
        learning_repo: Optional[LearningRepository] = None,
        reflection_repo: Optional[ReflectionRepository] = None,
        user_repo: Optional[UserRepository] = None
    ):
        """Initialize the SDL Integration with required components."""
        self.llm_router = llm_router
        self.vector_store = vector_store
        self.event_system = event_system
        self.learning_repo = learning_repo or LearningRepository()
        self.reflection_repo = reflection_repo or ReflectionRepository()
        self.user_repo = user_repo or UserRepository()
        self.concept_mapper = ConceptMapper(self.llm_router, self.learning_repo)
        self.sdl_engine = SDLEngine(
            self.llm_router,
            self.vector_store,
            self.learning_repo,
            self.reflection_repo,
            self.concept_mapper
        )
        self.processing_reflections = set()
        self.processing_conversations = set()
        logger.info("SDL Integration initialized")
    
    async def start(self):
        """Start the SDL Integration and register event handlers."""
        if self.event_system:
            await self._register_event_handlers()
        logger.info("SDL Integration started")
    
    async def stop(self):
        """Stop the SDL Integration and cleanup resources."""
        await self.sdl_engine.close()
        if self.learning_repo:
            await self.learning_repo.close()
        if self.reflection_repo:
            await self.reflection_repo.close()
        logger.info("SDL Integration stopped")
    
    async def process_new_reflection(self, reflection_id: str) -> Dict[str, Any]:
        """
        Process a new reflection and extract learnings.
        
        Args:
            reflection_id: ID of the reflection to process
            
        Returns:
            Summary of processing results
        """
        # FIXED: Check database for existing learnings (idempotency check)
        existing_learnings = await self._check_already_processed(
            source_type="reflection",
            source_id=reflection_id
        )
        
        if existing_learnings:
            logger.info(
                f"Reflection {reflection_id} already processed with {len(existing_learnings)} learnings, skipping"
            )
            return {
                "status": "skipped",
                "reason": "already_processed",
                "existing_learnings_count": len(existing_learnings)
            }
        
        # Also check in-memory processing set for concurrent requests
        if reflection_id in self.processing_reflections:
            logger.info(f"Reflection {reflection_id} currently being processed by another request, skipping")
            return {
                "status": "skipped",
                "reason": "concurrent_processing"
            }
            
        self.processing_reflections.add(reflection_id)
        
        try:
            # Process the reflection
            start_time = datetime.now()
            learnings = await self.sdl_engine.process_reflection(reflection_id)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "status": "success",
                "reflection_id": reflection_id,
                "learnings_count": len(learnings),
                "learnings": learnings,
                "processing_time_seconds": processing_time
            }
            
            # Trigger learning event if available with verification
            if self.event_system and learnings:
                try:
                    event_result = await self.event_system.process_event(
                        event_type=EventType.LEARNING_CREATED,
                        event_data={
                            "reflection_id": reflection_id,
                            "learnings_count": len(learnings),
                            "domains": list(set(l.get("domain", "") for l in learnings)),
                            "importance": max([l.get("importance", 0) for l in learnings], default=0.5)
                        },
                        user_id=learnings[0].get("user_id") if learnings else None
                    )
                    # FIXED: Verify event was accepted
                    if not event_result:
                        logger.warning(
                            f"Event system did not accept LEARNING_CREATED event for reflection {reflection_id}"
                        )
                    else:
                        logger.debug(f"LEARNING_CREATED event emitted successfully for reflection {reflection_id}")
                except Exception as event_error:
                    logger.error(
                        f"Failed to emit LEARNING_CREATED event for reflection {reflection_id}: {event_error}",
                        exc_info=True
                    )
            
            logger.info(f"Processed reflection {reflection_id}, extracted {len(learnings)} learnings")
            return result
            
        except Exception as e:
            logger.error(f"Error processing reflection {reflection_id}: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "reflection_id": reflection_id,
                "error": str(e)
            }
        finally:
            # Remove from processing set
            self.processing_reflections.discard(reflection_id)
    
    async def process_conversation(
        self, 
        conversation_id: str,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a conversation and extract learnings.
        
        Args:
            conversation_id: ID of the conversation
            messages: List of message objects
            
        Returns:
            Summary of processing results
        """
        # FIXED: Check database for existing learnings (idempotency check)
        existing_learnings = await self._check_already_processed(
            source_type="conversation",
            source_id=conversation_id
        )
        
        if existing_learnings:
            logger.info(
                f"Conversation {conversation_id} already processed with {len(existing_learnings)} learnings, skipping"
            )
            return {
                "status": "skipped",
                "reason": "already_processed",
                "existing_learnings_count": len(existing_learnings)
            }
        
        # Also check in-memory processing set for concurrent requests
        if conversation_id in self.processing_conversations:
            logger.info(f"Conversation {conversation_id} already being processed, skipping")
            return {
                "status": "skipped",
                "reason": "already_processing"
            }
            
        self.processing_conversations.add(conversation_id)
        
        try:
            # Process the conversation
            start_time = datetime.now()
            learnings = await self.sdl_engine.process_conversation(conversation_id, messages)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "status": "success",
                "conversation_id": conversation_id,
                "learnings_count": len(learnings),
                "learnings": learnings,
                "processing_time_seconds": processing_time
            }
            
            # Trigger learning event if available
            if self.event_system and learnings:
                user_id = None
                for msg in messages:
                    if msg.get("role") == "user" and "user_id" in msg:
                        user_id = msg["user_id"]
                        break
                        
                if user_id:
                    await self.event_system.process_event(
                        event_type=EventType.LEARNING_CREATED,
                        event_data={
                            "conversation_id": conversation_id,
                            "learnings_count": len(learnings),
                            "domains": list(set(l.get("domain", "") for l in learnings)),
                            "importance": max([l.get("importance", 0) for l in learnings], default=0.5)
                        },
                        user_id=user_id
                    )
            
            logger.info(f"Processed conversation {conversation_id}, extracted {len(learnings)} learnings")
            return result
            
        except Exception as e:
            logger.error(f"Error processing conversation {conversation_id}: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "conversation_id": conversation_id,
                "error": str(e)
            }
        finally:
            # Remove from processing set
            self.processing_conversations.discard(conversation_id)
    
    async def consolidate_user_learnings(
        self, 
        user_id: str,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Consolidate all learnings for a user across domains with pagination.
        
        Args:
            user_id: ID of the user to consolidate learnings for
            batch_size: Number of learnings to fetch per batch (default: 100)
            
        Returns:
            Consolidated insights
        """
        try:
            # FIXED: Use pagination to handle large learning sets
            all_learnings = []
            offset = 0
            total_fetched = 0
            max_learnings = 1000  # Safety limit
            
            while total_fetched < max_learnings:
                batch = await self.learning_repo.get_learnings_for_user(
                    user_id=user_id,
                    limit=batch_size,
                    offset=offset
                )
                
                if not batch:
                    break
                
                all_learnings.extend(batch)
                total_fetched += len(batch)
                offset += batch_size
                
                # Stop if we got fewer than requested (end of data)
                if len(batch) < batch_size:
                    break
                
                logger.debug(f"Fetched {total_fetched} learnings for user {user_id} so far...")
            
            logger.info(f"Retrieved {len(all_learnings)} learnings for user {user_id} consolidation")
            
            domains = list(set(learning.domain for learning in all_learnings))
            
            # Consolidate each domain
            domain_insights = {}
            for domain in domains:
                consolidation = await self.sdl_engine.consolidate_learnings(user_id, domain)
                domain_insights[domain] = consolidation
            
            # Generate meta-learning
            meta_learning = await self.sdl_engine.generate_meta_learning(user_id)
            
            result = {
                "status": "success",
                "user_id": user_id,
                "domains_analyzed": domains,
                "domain_insights": domain_insights,
                "meta_learning": meta_learning,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Consolidated learnings for user {user_id} across {len(domains)} domains")
            return result
            
        except Exception as e:
            logger.error(f"Error consolidating learnings for user {user_id}: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "user_id": user_id,
                "error": str(e)
            }
    
    async def reorganize_user_concepts(self, user_id: str) -> Dict[str, Any]:
        """
        Reorganize concepts for a user to improve knowledge organization.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Reorganization results
        """
        try:
            # Perform reorganization
            reorganization = await self.concept_mapper.reorganize_concepts(user_id)
            
            logger.info(f"Reorganized concepts for user {user_id}")
            return {
                "status": "success",
                "user_id": user_id,
                "reorganization": reorganization
            }
            
        except Exception as e:
            logger.error(f"Error reorganizing concepts for user {user_id}: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "user_id": user_id,
                "error": str(e)
            }
    
    async def get_user_knowledge_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about a user's knowledge base.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Knowledge statistics
        """
        try:
            # Get counts
            learnings = await self.learning_repo.get_learnings_for_user(
                user_id=user_id,
                limit=1000
            )
            
            concepts = await self.learning_repo.get_concepts_for_user(
                user_id=user_id,
                limit=1000
            )
            
            # Calculate domain distribution
            domain_counts = {}
            for learning in learnings:
                domain = learning.domain
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Calculate concept categories
            category_counts = {}
            for concept in concepts:
                category = concept.category or "uncategorized"
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Get recent reflections
            reflections = await self.reflection_repo.get_reflections_for_user(
                user_id=user_id,
                limit=10
            )
            
            result = {
                "status": "success",
                "user_id": user_id,
                "stats": {
                    "total_learnings": len(learnings),
                    "total_concepts": len(concepts),
                    "domain_distribution": domain_counts,
                    "category_distribution": category_counts,
                    "recent_reflections_count": len(reflections),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
            
            logger.debug(f"Retrieved knowledge stats for user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting knowledge stats for user {user_id}: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "user_id": user_id,
                "error": str(e)
            }
    
    # Internal methods
    
    async def _check_already_processed(
        self,
        source_type: str,
        source_id: str
    ) -> List[Dict[str, Any]]:
        """
        Check if a reflection or conversation has already been processed.
        
        This provides database-backed idempotency checking that survives restarts,
        unlike the in-memory processing sets.
        
        Args:
            source_type: Type of source ("reflection" or "conversation")
            source_id: ID of the source
            
        Returns:
            List of existing learnings if already processed, empty list otherwise
        """
        try:
            # Query for learnings from this source
            existing = await self.learning_repo.get_learnings_for_user(
                user_id="",  # We'll get all users for this source
                source_type=source_type,
                limit=100  # Reasonable limit
            )
            
            # Filter to exact source_id match
            matching = [
                learning for learning in existing
                if getattr(learning, 'source_id', None) == source_id
            ]
            
            return [l.to_dict() if hasattr(l, 'to_dict') else l for l in matching]
            
        except Exception as e:
            logger.warning(
                f"Failed to check if {source_type} {source_id} already processed: {e}. "
                "Proceeding with processing to be safe."
            )
            return []  # On error, allow processing to proceed
    
    async def _register_event_handlers(self):
        """Register event handlers with the event system."""
        if not self.event_system:
            logger.warning("No event system available, skipping handler registration")
            return
            
        # Handler for new reflections
        async def reflection_created_handler(event_data: Dict[str, Any], user_id: str):
            reflection_id = event_data.get("reflection_id")
            if reflection_id:
                # Process asynchronously
                asyncio.create_task(self.process_new_reflection(reflection_id))
        
        # Handler for learning pattern detection
        async def learning_pattern_handler(events: List[Dict[str, Any]], user_id: str):
            # When multiple learning events happen, trigger consolidation
            asyncio.create_task(self.consolidate_user_learnings(user_id))
        
        # Register handlers
        await self.event_system.register_trigger(
            trigger_id="sdl_reflection_processing",
            event_type=EventType.REFLECTION_CREATED,
            condition={"type": "simple", "field": "status", "operator": "eq", "value": "complete"},
            action=reflection_created_handler,
            cooldown_seconds=10,
            importance=0.7
        )
        
        # Register pattern for learning consolidation
        await self.event_system.register_pattern(
            pattern_id="learning_consolidation_pattern",
            event_types=[EventType.LEARNING_CREATED],
            pattern_config={
                "type": "frequency",
                "thresholds": {
                    EventType.LEARNING_CREATED: 5  # After 5 new learnings
                },
                "time_window_seconds": 3600,  # Within an hour
                "cooldown_seconds": 3600  # Run at most once per hour
            },
            action=learning_pattern_handler
        )
        
        logger.info("SDL event handlers registered")
