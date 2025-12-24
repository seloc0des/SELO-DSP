"""
Relationship Repository

This module provides the repository layer for relationship state operations
in a single-user application context.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..session import get_session
from ..models.relationship import RelationshipState

logger = logging.getLogger("selo.db.repositories.relationship")


class RelationshipRepository:
    """Repository for relationship state operations."""
    
    def __init__(self):
        """Initialize the relationship repository."""
        logger.debug("RelationshipRepository initialized")
    
    async def close(self):
        """Close any resources."""
        logger.debug("RelationshipRepository closed (no-op)")
    
    async def get_or_create_state(
        self,
        persona_id: str,
        session: Optional[AsyncSession] = None
    ) -> RelationshipState:
        """
        Get or create relationship state for a persona.
        
        Args:
            persona_id: Persona ID
            session: Optional database session
            
        Returns:
            RelationshipState object
        """
        async with get_session(session) as session:
            # Try to get existing state
            query = select(RelationshipState).where(RelationshipState.persona_id == persona_id)
            result = await session.execute(query)
            state = result.scalars().first()
            
            if state:
                return state
            
            # Create new state
            state = RelationshipState(
                persona_id=persona_id,
                first_conversation=datetime.now(timezone.utc)
            )
            session.add(state)
            await session.flush()
            
            logger.info(f"Created relationship state for persona {persona_id}")
            return state
    
    async def get_state(
        self,
        persona_id: str,
        session: Optional[AsyncSession] = None
    ) -> Optional[RelationshipState]:
        """
        Get relationship state for a persona.
        
        Args:
            persona_id: Persona ID
            session: Optional database session
            
        Returns:
            RelationshipState object or None if not found
        """
        async with get_session(session) as session:
            query = select(RelationshipState).where(RelationshipState.persona_id == persona_id)
            result = await session.execute(query)
            return result.scalars().first()
    
    async def update_state(
        self,
        persona_id: str,
        updates: Dict[str, Any],
        session: Optional[AsyncSession] = None
    ) -> Optional[RelationshipState]:
        """
        Update relationship state.
        
        Args:
            persona_id: Persona ID
            updates: Dictionary of fields to update
            session: Optional database session
            
        Returns:
            Updated RelationshipState object or None if not found
        """
        async with get_session(session) as session:
            # Get state (avoid nested get_session calls)
            query = select(RelationshipState).where(RelationshipState.persona_id == persona_id)
            result = await session.execute(query)
            state = result.scalars().first()
            if not state:
                logger.warning(f"Relationship state not found for persona {persona_id}")
                return None
            
            # Update fields
            for key, value in updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)
            
            # Update timestamp
            state.last_updated = datetime.now(timezone.utc)
            
            logger.info(f"Updated relationship state for persona {persona_id}")
            return state
    
    async def increment_conversation_count(
        self,
        persona_id: str,
        session: Optional[AsyncSession] = None
    ) -> None:
        """
        Increment conversation count and update last conversation time.
        
        Args:
            persona_id: Persona ID
            session: Optional database session
        """
        async with get_session(session) as session:
            query = select(RelationshipState).where(RelationshipState.persona_id == persona_id)
            result = await session.execute(query)
            state = result.scalars().first()
            
            if not state:
                state = await self.get_or_create_state(persona_id, session)
            
            state.conversations_count += 1
            state.last_conversation_at = datetime.now(timezone.utc)
            
            # Update days_known
            if state.first_conversation:
                delta = datetime.now(timezone.utc) - state.first_conversation
                state.days_known = delta.days
            
            state.last_updated = datetime.now(timezone.utc)
    
    async def update_intimacy(
        self,
        persona_id: str,
        delta: float,
        reason: str = "",
        session: Optional[AsyncSession] = None
    ) -> None:
        """
        Update intimacy level.
        
        Args:
            persona_id: Persona ID
            delta: Change in intimacy (-1.0 to 1.0)
            reason: Reason for change (for logging)
            session: Optional database session
        """
        async with get_session(session) as session:
            query = select(RelationshipState).where(RelationshipState.persona_id == persona_id)
            result = await session.execute(query)
            state = result.scalars().first()
            
            if not state:
                state = await self.get_or_create_state(persona_id, session)
            
            old_intimacy = state.intimacy_level
            state.intimacy_level = max(0.0, min(1.0, state.intimacy_level + delta))
            
            # Update stage based on intimacy
            if state.intimacy_level >= 0.8 and state.trust_level >= 0.8:
                state.stage = "profound"
            elif state.intimacy_level >= 0.6 and state.trust_level >= 0.7:
                state.stage = "deep"
            elif state.intimacy_level >= 0.4:
                state.stage = "established"
            elif state.intimacy_level >= 0.2:
                state.stage = "developing"
            else:
                state.stage = "early"
            
            state.last_updated = datetime.now(timezone.utc)
            
            logger.info(
                f"Updated intimacy for persona {persona_id}: "
                f"{old_intimacy:.2f} -> {state.intimacy_level:.2f} "
                f"(delta: {delta:+.2f}, reason: {reason})"
            )
    
    async def update_trust(
        self,
        persona_id: str,
        delta: float,
        reason: str = "",
        session: Optional[AsyncSession] = None
    ) -> None:
        """
        Update trust level.
        
        Args:
            persona_id: Persona ID
            delta: Change in trust (-1.0 to 1.0)
            reason: Reason for change (for logging)
            session: Optional database session
        """
        async with get_session(session) as session:
            query = select(RelationshipState).where(RelationshipState.persona_id == persona_id)
            result = await session.execute(query)
            state = result.scalars().first()
            
            if not state:
                state = await self.get_or_create_state(persona_id, session)
            
            old_trust = state.trust_level
            state.trust_level = max(0.0, min(1.0, state.trust_level + delta))
            
            state.last_updated = datetime.now(timezone.utc)
            
            logger.info(
                f"Updated trust for persona {persona_id}: "
                f"{old_trust:.2f} -> {state.trust_level:.2f} "
                f"(delta: {delta:+.2f}, reason: {reason})"
            )
    
    async def record_milestone(
        self,
        persona_id: str,
        milestone_type: str,
        session: Optional[AsyncSession] = None
    ) -> None:
        """
        Record a relationship milestone.
        
        Args:
            persona_id: Persona ID
            milestone_type: Type of milestone (deep_conversation, vulnerability, disagreement, inside_joke)
            session: Optional database session
        """
        async with get_session(session) as session:
            query = select(RelationshipState).where(RelationshipState.persona_id == persona_id)
            result = await session.execute(query)
            state = result.scalars().first()
            
            if not state:
                state = await self.get_or_create_state(persona_id, session)
            
            now = datetime.now(timezone.utc)
            
            if milestone_type == "deep_conversation" and not state.first_deep_conversation:
                state.first_deep_conversation = now
                logger.info(f"Recorded first deep conversation milestone for persona {persona_id}")
            elif milestone_type == "vulnerability" and not state.first_vulnerability_moment:
                state.first_vulnerability_moment = now
                logger.info(f"Recorded first vulnerability milestone for persona {persona_id}")
            elif milestone_type == "disagreement" and not state.first_disagreement:
                state.first_disagreement = now
                logger.info(f"Recorded first disagreement milestone for persona {persona_id}")
            elif milestone_type == "inside_joke" and not state.first_inside_joke:
                state.first_inside_joke = now
                logger.info(f"Recorded first inside joke milestone for persona {persona_id}")
            
            state.last_updated = now
