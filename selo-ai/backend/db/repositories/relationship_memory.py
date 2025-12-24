"""
Relationship Memory Repository

This module provides the repository layer for relationship memory operations
in a single-user application context.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select, desc, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..session import get_session
from ..models.relationship_memory import RelationshipMemory, AnticipatedEvent

logger = logging.getLogger("selo.db.repositories.relationship_memory")


class RelationshipMemoryRepository:
    """Repository for relationship memory operations."""
    
    def __init__(self):
        """Initialize the relationship memory repository."""
        logger.debug("RelationshipMemoryRepository initialized")
    
    async def close(self):
        """Close any resources."""
        logger.debug("RelationshipMemoryRepository closed (no-op)")
    
    async def create_memory(
        self,
        persona_id: str,
        memory_type: str,
        narrative: str,
        emotional_significance: float = 0.5,
        intimacy_delta: float = 0.0,
        trust_delta: float = 0.0,
        user_perspective: Optional[str] = None,
        context: Optional[str] = None,
        conversation_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        emotional_tone: Optional[str] = None,
        session: Optional[AsyncSession] = None
    ) -> RelationshipMemory:
        """
        Create a new relationship memory.
        
        Args:
            persona_id: Persona ID
            memory_type: Type of memory (shared_moment, inside_joke, etc.)
            narrative: SELO's perspective on this memory
            emotional_significance: 0.0-1.0
            intimacy_delta: Change in intimacy from this memory
            trust_delta: Change in trust from this memory
            user_perspective: What user said/did
            context: Additional context
            conversation_id: Associated conversation
            tags: Searchable tags
            emotional_tone: positive, negative, mixed, neutral
            session: Optional database session
            
        Returns:
            Created RelationshipMemory object
        """
        async with get_session(session) as session:
            memory = RelationshipMemory(
                persona_id=persona_id,
                memory_type=memory_type,
                narrative=narrative,
                emotional_significance=emotional_significance,
                intimacy_delta=intimacy_delta,
                trust_delta=trust_delta,
                user_perspective=user_perspective,
                context=context,
                conversation_id=conversation_id,
                tags=tags or [],
                emotional_tone=emotional_tone
            )
            
            session.add(memory)
            await session.flush()
            
            logger.info(f"Created relationship memory {memory.id} for persona {persona_id}: {memory_type}")
            return memory
    
    async def get_recent_memories(
        self,
        persona_id: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        session: Optional[AsyncSession] = None
    ) -> List[RelationshipMemory]:
        """
        Get recent relationship memories.
        
        Args:
            persona_id: Persona ID
            limit: Maximum number of memories
            memory_type: Filter by memory type
            session: Optional database session
            
        Returns:
            List of RelationshipMemory objects
        """
        async with get_session(session) as session:
            conditions = [RelationshipMemory.persona_id == persona_id]
            
            if memory_type:
                conditions.append(RelationshipMemory.memory_type == memory_type)
            
            query = (
                select(RelationshipMemory)
                .where(and_(*conditions))
                .order_by(desc(RelationshipMemory.occurred_at))
                .limit(limit)
            )
            
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def get_significant_memories(
        self,
        persona_id: str,
        min_significance: float = 0.7,
        limit: int = 5,
        session: Optional[AsyncSession] = None
    ) -> List[RelationshipMemory]:
        """
        Get most emotionally significant memories.
        
        Args:
            persona_id: Persona ID
            min_significance: Minimum emotional significance
            limit: Maximum number of memories
            session: Optional database session
            
        Returns:
            List of RelationshipMemory objects
        """
        async with get_session(session) as session:
            query = (
                select(RelationshipMemory)
                .where(
                    and_(
                        RelationshipMemory.persona_id == persona_id,
                        RelationshipMemory.emotional_significance >= min_significance
                    )
                )
                .order_by(desc(RelationshipMemory.emotional_significance))
                .limit(limit)
            )
            
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def increment_recall(
        self,
        memory_id: str,
        session: Optional[AsyncSession] = None
    ) -> None:
        """
        Increment recall count for a memory.
        
        Args:
            memory_id: Memory ID
            session: Optional database session
        """
        async with get_session(session) as session:
            query = select(RelationshipMemory).where(RelationshipMemory.id == memory_id)
            result = await session.execute(query)
            memory = result.scalars().first()
            
            if memory:
                memory.recall_count += 1
                memory.last_recalled = datetime.now(timezone.utc)
    
    # === Anticipated Events ===
    
    async def create_anticipated_event(
        self,
        persona_id: str,
        event_description: str,
        anticipated_date: Optional[datetime] = None,
        event_type: Optional[str] = None,
        importance: float = 0.5,
        conversation_id: Optional[str] = None,
        session: Optional[AsyncSession] = None
    ) -> AnticipatedEvent:
        """
        Create a new anticipated event.
        
        Args:
            persona_id: Persona ID
            event_description: Description of the event
            anticipated_date: When it's expected to happen
            event_type: Type of event
            importance: 0.0-1.0
            conversation_id: Associated conversation
            session: Optional database session
            
        Returns:
            Created AnticipatedEvent object
        """
        async with get_session(session) as session:
            event = AnticipatedEvent(
                persona_id=persona_id,
                event_description=event_description,
                anticipated_date=anticipated_date,
                event_type=event_type,
                importance=importance,
                conversation_id=conversation_id
            )
            
            session.add(event)
            await session.flush()
            
            logger.info(f"Created anticipated event {event.id} for persona {persona_id}")
            return event
    
    async def get_pending_events(
        self,
        persona_id: str,
        session: Optional[AsyncSession] = None
    ) -> List[AnticipatedEvent]:
        """
        Get events that haven't been followed up on yet.
        
        Args:
            persona_id: Persona ID
            session: Optional database session
            
        Returns:
            List of AnticipatedEvent objects
        """
        async with get_session(session) as session:
            now = datetime.now(timezone.utc)
            
            query = (
                select(AnticipatedEvent)
                .where(
                    and_(
                        AnticipatedEvent.persona_id == persona_id,
                        AnticipatedEvent.followed_up < 0.5,
                        or_(
                            AnticipatedEvent.anticipated_date.is_(None),
                            AnticipatedEvent.anticipated_date <= now
                        )
                    )
                )
                .order_by(desc(AnticipatedEvent.importance))
            )
            
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def mark_event_followed_up(
        self,
        event_id: str,
        outcome: Optional[str] = None,
        session: Optional[AsyncSession] = None
    ) -> None:
        """
        Mark an event as followed up.
        
        Args:
            event_id: Event ID
            outcome: What happened
            session: Optional database session
        """
        async with get_session(session) as session:
            query = select(AnticipatedEvent).where(AnticipatedEvent.id == event_id)
            result = await session.execute(query)
            event = result.scalars().first()
            
            if event:
                event.followed_up = 1.0
                event.follow_up_at = datetime.now(timezone.utc)
                if outcome:
                    event.outcome = outcome
                
                logger.info(f"Marked event {event_id} as followed up")
