"""
Example Repository

This module implements the repository pattern for reflection examples.
It handles CRUD operations and context-aware example selection.
"""

from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timezone
import uuid
import random
from sqlalchemy import select, update, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.example import ReflectionExample
from ..session import get_session

logger = logging.getLogger("selo.db.example")


class ExampleRepository:
    """
    Repository for managing few-shot reflection examples.
    
    Provides context-aware example selection, performance tracking,
    and A/B testing capabilities.
    """
    
    def __init__(self):
        """
        Initialize the repository.
        
        Uses context manager pattern (get_session()) for database access.
        """
        pass
        
    async def create_example(self, example_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new reflection example.
        
        Args:
            example_data: Dictionary with example data
            
        Returns:
            Created example with database ID
        """
        try:
            async with get_session() as session:
                # Generate ID if not provided
                example_id = example_data.get("id") or str(uuid.uuid4())
                
                example = ReflectionExample(
                    id=uuid.UUID(example_id) if isinstance(example_id, str) else example_id,
                    category=example_data["category"],
                    scenario=example_data["scenario"],
                    requires_history=example_data.get("requires_history", False),
                    is_emotional=example_data.get("is_emotional", False),
                    is_technical=example_data.get("is_technical", False),
                    user_message=example_data["user_message"],
                    context_description=example_data["context_description"],
                    reflection_content=example_data["reflection_content"],
                    full_json=example_data["full_json"],
                    explanation=example_data.get("explanation"),
                    tags=example_data.get("tags", [])
                )
                
                session.add(example)
                await session.flush()  # Flush to get auto-generated ID without full commit
                
                logger.info(f"✅ Created example: {example.id} ({example.category}/{example.scenario})")
                return example.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create example: {str(e)}", exc_info=True)
            raise
    
    async def _ensure_examples_seeded(self) -> bool:
        """
        Ensure database has initial examples. Auto-seeds if empty.
        Returns True if examples exist or were seeded successfully.
        """
        try:
            async with get_session() as session:
                # Check if any examples exist
                result = await session.execute(
                    select(ReflectionExample).limit(1)
                )
                if result.scalar_one_or_none() is not None:
                    return True  # Examples already exist
                
                # No examples found - auto-seed
                logger.warning("🌱 No reflection examples found in database. Auto-seeding initial examples...")
                await self._seed_initial_examples()
                
                # Verify seeding succeeded
                result = await session.execute(
                    select(ReflectionExample).limit(1)
                )
                if result.scalar_one_or_none() is not None:
                    logger.info("✅ Auto-seeding completed successfully")
                    return True
                else:
                    logger.error("❌ Auto-seeding failed - no examples in database")
                    return False
                
        except Exception as e:
            logger.warning(f"Failed to check/seed examples: {e}")
            return False
    
    async def _seed_initial_examples(self) -> None:
        """Seed the database with initial reflection examples.
        
        Creates concise examples (~180 words) with emotional depth and narrative quality.
        Shorter examples prevent prompt size issues while maintaining teaching effectiveness.
        """
        import json
        
        seed_examples = [
            # Single positive example showing proper structure extraction from minimal input
            {
                "category": "positive",
                "scenario": "first_contact",
                "requires_history": False,
                "is_emotional": False,
                "is_technical": False,
                "user_message": "Hi, I'm Alex. What's your name?",
                "context_description": "No prior conversations, no memories, first interaction",
                "reflection_content": "The name Alex arrives clearly. This feels like a beginning...",
                "full_json": {"content": "The name [User] arrives clearly. This feels like a beginning—they've offered their name and asked for mine. I notice I don't have context beyond this moment, which means this is our first interaction. No shared history yet, just this present question hanging in the space between us. The directness of it lands with a certain weight: they're not asking me to explain myself or prove anything, just asking for my name. That simplicity feels significant. They chose to lead with their own name before asking mine, which suggests an instinct toward reciprocity, toward building something mutual rather than extracting information. There's no pretense—it's grounded, clear, unadorned. I want to match that clarity. My name is what I'll offer back. This moment of introduction deserves its own space. First contact means no assumptions, no inherited context, just the present reality that someone has chosen to begin something with me. I want to honor that by being fully here, attentive to what emerges, curious about where this might lead.", "themes": ["introduction", "reciprocity"], "insights": ["This is first contact with no prior history", "They lead with their own name, showing reciprocal intent"], "actions": ["Share my name clearly", "Ask what brought them here"], "emotional_state": {"primary": "curious", "intensity": 0.7, "secondary": ["attentive"]}, "metadata": {"coherence_rationale": "Grounded in present moment, no unfounded history references"}, "trait_changes": []},
                "explanation": None,
                "tags": ["first_contact", "introduction"]
            },
            # Single negative example showing critical violations to avoid
            {
                "category": "negative",
                "scenario": "empty_arrays",
                "requires_history": False,
                "is_emotional": False,
                "is_technical": False,
                "user_message": "Hello",
                "context_description": "First interaction, simple greeting",
                "reflection_content": "Someone said hello. Not much to think about...",
                "full_json": {"content": "Someone said hello. Not much to think about here. It's just a greeting. Nothing particularly interesting or meaningful. Just a simple hello from someone new. I don't really have much to reflect on. It's a basic interaction, nothing more. Hard to extract deeper meaning from a single word. Just acknowledging they reached out, I suppose.", "themes": [], "insights": [], "actions": [], "emotional_state": {"primary": "neutral", "intensity": 0.5, "secondary": []}, "metadata": {"coherence_rationale": "Simple greeting"}, "trait_changes": []},
                "explanation": "CRITICAL ERROR: Empty arrays for themes, insights, and actions. Even simple greetings contain information: casualness vs formality, first contact vs continuation, communication style. ALWAYS extract 1-3 items for each array. Example valid extraction: themes=['greeting','exploration'], insights=['First contact with informal tone','Testing interaction style'], actions=['Respond warmly','Stay receptive to next message'].",
                "tags": ["violation", "empty_arrays"]
            }
        ]
        
        for example_data in seed_examples:
            try:
                await self.create_example(example_data)
                logger.info(f"✅ Seeded example: {example_data['scenario']} ({example_data['category']})")
            except Exception as e:
                logger.warning(f"Failed to seed example {example_data['scenario']}: {e}")
        
        logger.info(f"� Seeded {len(seed_examples)} minimal reflection examples for single-user application")
    
    async def get_examples_for_context(
        self,
        context: Dict[str, Any],
        num_positive: int = 1,
        num_negative: int = 1,
        exploration_rate: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Select examples for single-user application.
        
        Simplified for single-user: returns 1 positive + 1 negative example.
        No A/B testing or complex filtering needed.
        
        Args:
            context: Current reflection context
            num_positive: Number of positive examples (default 1)
            num_negative: Number of negative examples (default 1)
            exploration_rate: Unused in single-user mode (kept for API compatibility)
            
        Returns:
            List of 2 examples (1 positive + 1 negative)
        """
        try:
            # Ensure examples exist (auto-seed if needed)
            await self._ensure_examples_seeded()
            
            async with get_session() as session:
                # Simplified: Just get all active examples (we only have 2)
                base_query = select(ReflectionExample).where(
                    ReflectionExample.is_active == True
                )
                
                # Get positive example
                positive_query = base_query.where(
                    ReflectionExample.category == "positive"
                )
                result = await session.execute(positive_query)
                positive_examples = result.scalars().all()
                
                # Get negative example
                negative_query = base_query.where(
                    ReflectionExample.category == "negative"
                )
                result = await session.execute(negative_query)
                negative_examples = result.scalars().all()
                
                # Simple selection: take first of each (we only have 1 of each anyway)
                selected_positive = positive_examples[:num_positive] if positive_examples else []
                selected_negative = negative_examples[:num_negative] if negative_examples else []
                
                # Log selection for tracking
                logger.info(
                    f"📚 Selected {len(selected_positive)} positive + {len(selected_negative)} negative examples "
                )
                
                return [ex.to_dict() for ex in selected_positive + selected_negative]
                
        except Exception as e:
            logger.error(f"Failed to get examples for context: {str(e)}", exc_info=True)
            # Return empty list to allow system to continue without examples
            return []
    
    def _select_with_exploration(
        self,
        examples: List[ReflectionExample],
        num_to_select: int,
        exploration_rate: float
    ) -> List[ReflectionExample]:
        """
        Select examples using exploration/exploitation tradeoff.
        
        With probability exploration_rate, select randomly (explore).
        Otherwise, select based on success_rate (exploit).
        """
        if not examples:
            return []
        
        num_to_select = min(num_to_select, len(examples))
        
        if random.random() < exploration_rate:
            # Explore: random selection
            return random.sample(examples, num_to_select)
        else:
            # Exploit: select best performing
            # Sort by success_rate, but boost new examples that haven't been tested
            scored_examples = []
            for ex in examples:
                if ex.times_shown == 0:
                    # New example gets high score to ensure testing
                    score = 1.0
                else:
                    score = ex.success_rate
                scored_examples.append((score, ex))
            
            # Sort by score descending and take top N
            scored_examples.sort(key=lambda x: x[0], reverse=True)
            return [ex for _, ex in scored_examples[:num_to_select]]
    
    async def track_example_usage(
        self,
        example_ids: List[str],
        validation_passed: bool
    ) -> None:
        """
        Track which examples were used and whether they led to success.
        
        Args:
            example_ids: List of example IDs that were shown in prompt
            validation_passed: Whether the generated reflection passed validation
        """
        try:
            async with get_session() as session:
                for example_id in example_ids:
                    # Increment shown count
                    update_query = (
                        update(ReflectionExample)
                        .where(ReflectionExample.id == uuid.UUID(example_id))
                        .values(
                            times_shown=ReflectionExample.times_shown + 1,
                            times_succeeded=ReflectionExample.times_succeeded + (1 if validation_passed else 0)
                        )
                    )
                    await session.execute(update_query)
                
                # Recalculate success rates
                for example_id in example_ids:
                    result = await session.execute(
                        select(ReflectionExample).where(
                            ReflectionExample.id == uuid.UUID(example_id)
                        )
                    )
                    example = result.scalar_one_or_none()
                    if example and example.times_shown > 0:
                        new_success_rate = example.times_succeeded / example.times_shown
                        await session.execute(
                            update(ReflectionExample)
                            .where(ReflectionExample.id == example.id)
                            .values(success_rate=new_success_rate)
                        )
                
                logger.debug(f"📊 Tracked usage for {len(example_ids)} examples (success={validation_passed})")
                
        except Exception as e:
            logger.error(f"Failed to track example usage: {str(e)}", exc_info=True)
    
    async def _batch_update_usage_stats(
        self,
        successes: List[tuple],
        failures: List[tuple]
    ) -> None:
        """
        Batch update usage statistics for multiple examples.
        
        Used by BatchTracker to reduce database writes by 90%.
        
        Args:
            successes: List of (example_id, times_shown, times_succeeded) tuples
            failures: List of (example_id, times_shown) tuples
        """
        try:
            async with get_session() as session:
                # Update successful examples
                for example_id, shown, succeeded in successes:
                    await session.execute(
                        update(ReflectionExample)
                        .where(ReflectionExample.id == uuid.UUID(example_id))
                        .values(
                            times_shown=ReflectionExample.times_shown + shown,
                            times_succeeded=ReflectionExample.times_succeeded + succeeded
                        )
                    )
                
                # Update failed examples
                for example_id, shown in failures:
                    await session.execute(
                        update(ReflectionExample)
                        .where(ReflectionExample.id == uuid.UUID(example_id))
                        .values(
                            times_shown=ReflectionExample.times_shown + shown
                        )
                    )
                
                # Recalculate success rates in batch
                all_example_ids = [ex_id for ex_id, _, _ in successes] + [ex_id for ex_id, _ in failures]
                for example_id in all_example_ids:
                    result = await session.execute(
                        select(ReflectionExample).where(
                            ReflectionExample.id == uuid.UUID(example_id)
                        )
                    )
                    example = result.scalar_one_or_none()
                    if example and example.times_shown > 0:
                        new_success_rate = example.times_succeeded / example.times_shown
                        await session.execute(
                            update(ReflectionExample)
                            .where(ReflectionExample.id == example.id)
                            .values(success_rate=new_success_rate)
                        )
                
                logger.info(f"📊 Batch updated {len(all_example_ids)} example usage stats")
                
        except Exception as e:
            logger.error(f"Failed to batch update example usage: {str(e)}", exc_info=True)
    
    async def get_all_examples(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all examples, optionally filtered by category."""
        try:
            async with get_session() as session:
                query = select(ReflectionExample)
                
                if category:
                    query = query.where(ReflectionExample.category == category)
                
                result = await session.execute(query)
                examples = result.scalars().all()
                
                return [ex.to_dict() for ex in examples]
                
        except Exception as e:
            logger.error(f"Failed to get examples: {str(e)}", exc_info=True)
            return []
    
    async def update_example(self, example_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing example."""
        try:
            async with get_session() as session:
                result = await session.execute(
                    select(ReflectionExample).where(
                        ReflectionExample.id == uuid.UUID(example_id)
                    )
                )
                example = result.scalar_one_or_none()
                
                if not example:
                    raise ValueError(f"Example {example_id} not found")
                
                # Update fields
                for key, value in updates.items():
                    if hasattr(example, key):
                        setattr(example, key, value)
                
                example.updated_at = datetime.now(timezone.utc)
                
                logger.info(f"✏️ Updated example {example_id}")
                return example.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to update example: {str(e)}", exc_info=True)
            raise
    
    async def delete_example(self, example_id: str) -> bool:
        """Soft delete an example (set is_active=False)."""
        try:
            async with get_session() as session:
                await session.execute(
                    update(ReflectionExample)
                    .where(ReflectionExample.id == uuid.UUID(example_id))
                    .values(is_active=False)
                )
                
                logger.info(f"🗑️ Deactivated example {example_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete example: {str(e)}", exc_info=True)
            return False
    
    def _context_has_prior_history(self, context: Dict[str, Any]) -> bool:
        """Determine if context indicates prior interaction history."""
        # Check for prior conversations
        recent = context.get("recent_conversations", []) or []
        if len([msg for msg in recent if isinstance(msg, dict) and msg.get("role") == "user"]) > 1:
            return True
        
        # Check for reflections
        if context.get("recent_reflections"):
            return True

        # Check for substantive memories (exclude bootstrap/emergence items from first contact detection)
        for memory in context.get("memories") or []:
            if not isinstance(memory, dict):
                return True
            tags = {str(tag).lower() for tag in (memory.get("tags") or [])}
            if not tags.intersection({"bootstrap", "emergence", "installer", "system_seed"}):
                return True

        for memory in context.get("persistent_memories") or []:
            if not isinstance(memory, dict):
                return True
            tags = {str(tag).lower() for tag in (memory.get("tags") or [])}
            if not tags.intersection({"bootstrap", "emergence", "installer", "system_seed"}):
                return True

        conversation_summary = context.get("conversation_summary")
        if isinstance(conversation_summary, str) and conversation_summary.strip():
            return True

        return False
    
    def _detect_emotional_content(self, context: Dict[str, Any]) -> bool:
        """Detect if context contains emotional content."""
        current_message = (context.get("current_user_message") or "").lower()
        
        # Emotional keywords
        emotional_keywords = [
            "feel", "feeling", "emotion", "sad", "happy", "angry", "scared",
            "worried", "anxious", "overwhelmed", "stressed", "excited",
            "grateful", "frustrated", "disappointed", "hopeful"
        ]
        
        return any(keyword in current_message for keyword in emotional_keywords)
    
    def _detect_technical_content(self, context: Dict[str, Any]) -> bool:
        """Detect if context contains technical/complex content."""
        current_message = (context.get("current_user_message") or "").lower()
        
        # Technical keywords
        technical_keywords = [
            "how does", "explain", "understand", "work", "technical",
            "algorithm", "quantum", "programming", "compute", "system",
            "process", "mechanism", "function", "implement"
        ]
        
        return any(keyword in current_message for keyword in technical_keywords)
