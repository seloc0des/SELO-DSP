"""
Learning Repository for the SDL (Self-Development Learning) module.

This module provides database operations for SDL models using SQLAlchemy.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from sqlalchemy import select, update, delete, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..db.session import get_session
from .learning_models import Learning, Concept, Connection, LearningConcept

logger = logging.getLogger("selo.sdl.repository")


class LearningRepository:
    """Repository for accessing and managing learning data."""
    
    def __init__(self, db_session=None):
        """Initialize the repository with optional database session."""
        self.db_session = db_session
    
    # Learning methods
    
    async def create_learning(self, learning_data: Dict[str, Any]) -> Learning:
        """Create a new learning entry."""
        async with get_session(self.db_session) as session:
            # Create the learning
            learning = Learning(**{
                k: v for k, v in learning_data.items() 
                if k not in ['concepts', 'id'] and hasattr(Learning, k)
            })
            
            if 'id' not in learning_data or not learning_data['id']:
                learning.id = str(uuid.uuid4())
            else:
                learning.id = learning_data['id']
                
            # Add concepts if provided
            if 'concepts' in learning_data and learning_data['concepts']:
                for concept_name in learning_data['concepts']:
                    # FIXED: Use get-or-create pattern with proper handling of race conditions
                    concept = await self._get_or_create_concept(
                        session=session,
                        user_id=learning_data['user_id'],
                        concept_name=concept_name
                    )
                    
                    learning.concepts.append(concept)
            
            session.add(learning)
            await session.commit()  # Learning creation uses commit, not just flush
            
            logger.info(f"Created learning {learning.id} for user {learning.user_id}")
            return learning
    
    async def get_learning(self, learning_id: str) -> Optional[Learning]:
        """Get a learning by ID."""
        async with get_session(self.db_session) as session:
            query = (
                select(Learning)
                .where(Learning.id == learning_id)
                .options(selectinload(Learning.concepts))
            )
            
            result = await session.execute(query)
            learning = result.scalar_one_or_none()
            
            return learning
    
    async def get_learnings_for_user(
        self, 
        user_id: str, 
        limit: int = 50,
        offset: int = 0,
        domain: Optional[str] = None,
        source_type: Optional[str] = None,
        active_only: bool = True
    ) -> List[Learning]:
        """Get learnings for a specific user."""
        async with get_session(self.db_session) as session:
            # Build query with filters
            filters = [Learning.user_id == user_id]
            
            if domain:
                filters.append(Learning.domain == domain)
            
            if source_type:
                filters.append(Learning.source_type == source_type)
                
            if active_only:
                filters.append(Learning.active == True)
                
            query = (
                select(Learning)
                .where(and_(*filters))
                .options(selectinload(Learning.concepts))
                .order_by(desc(Learning.importance), desc(Learning.created_at))
                .offset(offset)
                .limit(limit)
            )
            
            result = await session.execute(query)
            learnings = result.scalars().all()
            
            return learnings
    
    async def get_learnings_by_source(
        self,
        source_type: str,
        source_id: str,
        limit: int = 100
    ) -> List[Learning]:
        """Get learnings by source type and source ID for idempotency checking."""
        async with get_session(self.db_session) as session:
            query = (
                select(Learning)
                .where(
                    and_(
                        Learning.source_type == source_type,
                        Learning.source_id == source_id
                    )
                )
                .options(selectinload(Learning.concepts))
                .order_by(desc(Learning.created_at))
                .limit(limit)
            )
            
            result = await session.execute(query)
            learnings = result.scalars().all()
            
            return list(learnings)
    
    async def update_learning(self, learning_id: str, learning_data: Dict[str, Any]) -> Optional[Learning]:
        """Update an existing learning."""
        # Get the learning to update
        learning = await self.get_learning(learning_id)
        if not learning:
            return None
        
        async with get_session(self.db_session) as session:
            # Update fields
            update_data = {
                k: v for k, v in learning_data.items() 
                if k not in ['id', 'user_id', 'concepts', 'created_at'] and hasattr(Learning, k)
            }
            
            for key, value in update_data.items():
                setattr(learning, key, value)
                
            # Handle concepts if provided
            if 'concepts' in learning_data:
                # Clear existing concepts
                learning.concepts = []
                
                # Add new concepts
                for concept_name in learning_data['concepts']:
                    concept = await self.get_concept_by_name(
                        learning.user_id, 
                        concept_name
                    )
                    
                    if not concept:
                        concept = Concept(
                            id=str(uuid.uuid4()),
                            user_id=learning.user_id,
                            name=concept_name,
                            description=f"Auto-created concept for {concept_name}"
                        )
                        session.add(concept)
                    
                    learning.concepts.append(concept)
            
            # Update the timestamp
            learning.updated_at = datetime.now(timezone.utc)
            
            await session.commit()
            
            logger.info(f"Updated learning {learning.id}")
            return learning
    
    async def delete_learning(self, learning_id: str) -> bool:
        """Delete a learning by ID."""
        async with get_session(self.db_session) as session:
            # Query and delete in same session to avoid detached instance errors
            query = (
                select(Learning)
                .where(Learning.id == learning_id)
            )
            result = await session.execute(query)
            learning = result.scalar_one_or_none()
            
            if not learning:
                return False
            
            # Delete the learning
            await session.delete(learning)
            await session.commit()
        
        logger.info(f"Deleted learning {learning_id}")
        return True
    
    async def get_recent_learnings(
        self,
        user_id: str,
        limit: int = 10,
        domain: Optional[str] = None
    ) -> List[Learning]:
        """
        Get recent learnings for a user, ordered by creation date.
        This is an alias for get_learnings_for_user with ordering by recency.
        
        Args:
            user_id: User ID
            limit: Maximum number of learnings to return
            domain: Optional domain filter
            
        Returns:
            List of recent learning objects
        """
        return await self.get_learnings_for_user(
            user_id=user_id,
            limit=limit,
            domain=domain,
            active_only=True
        )
    
    async def close(self):
        """Close repository resources."""
        # Context managers handle session cleanup
        logger.debug("LearningRepository closed")
    
    async def _get_or_create_concept(
        self,
        session,
        user_id: str,
        concept_name: str
    ) -> Concept:
        """
        Get or create a concept with proper race condition handling.
        
        This method handles concurrent concept creation by catching IntegrityError
        and retrying the get operation.
        
        Args:
            session: Active database session
            user_id: User ID
            concept_name: Name of the concept
            
        Returns:
            Existing or newly created Concept object
        """
        from sqlalchemy.exc import IntegrityError
        
        # Try to get existing concept first
        query = (
            select(Concept)
            .where(
                and_(
                    Concept.user_id == user_id,
                    Concept.name == concept_name
                )
            )
        )
        result = await session.execute(query)
        concept = result.scalar_one_or_none()
        
        if concept:
            return concept
        
        # Try to create new concept
        try:
            concept = Concept(
                id=str(uuid.uuid4()),
                user_id=user_id,
                name=concept_name,
                description=f"Auto-created concept for {concept_name}"
            )
            session.add(concept)
            await session.flush()  # Flush to detect unique constraint violations
            return concept
        except IntegrityError:
            # Another transaction created this concept - retry get
            await session.rollback()
            result = await session.execute(query)
            concept = result.scalar_one_or_none()
            if not concept:
                # This should be extremely rare
                logger.error(f"Failed to get or create concept {concept_name} for user {user_id}")
                raise
            logger.debug(f"Concept {concept_name} created by concurrent transaction, using existing")
            return concept
    
    # Concept methods
    
    async def create_concept(self, concept_data: Dict[str, Any]) -> Concept:
        """Create a new concept."""
        async with get_session(self.db_session) as session:
            # Create the concept
            concept = Concept(**{
                k: v for k, v in concept_data.items() 
                if k not in ['id'] and hasattr(Concept, k)
            })
            
            if 'id' not in concept_data or not concept_data['id']:
                concept.id = str(uuid.uuid4())
            else:
                concept.id = concept_data['id']
                
            session.add(concept)
            await session.commit()  # Concept creation uses commit, not just flush
            
            logger.info(f"Created concept {concept.id} ({concept.name}) for user {concept.user_id}")
            return concept
    
    async def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get a concept by ID."""
        async with get_session(self.db_session) as session:
            query = (
                select(Concept)
                .where(Concept.id == concept_id)
                .options(
                    selectinload(Concept.learnings),
                    selectinload(Concept.outgoing_connections),
                    selectinload(Concept.incoming_connections)
                )
            )
            
            result = await session.execute(query)
            concept = result.scalar_one_or_none()
            
            return concept
    
    async def get_concept_by_name(self, user_id: str, name: str) -> Optional[Concept]:
        """Get a concept by name for a specific user."""
        async with get_session(self.db_session) as session:
            query = (
                select(Concept)
                .where(
                    and_(
                        Concept.user_id == user_id,
                        Concept.name == name
                    )
                )
            )
            
            result = await session.execute(query)
            concept = result.scalar_one_or_none()
            
            return concept
    
    async def get_concepts_for_user(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        category: Optional[str] = None,
        active_only: bool = True
    ) -> List[Concept]:
        """Get concepts for a specific user."""
        async with get_session(self.db_session) as session:
            # Build query with filters
            filters = [Concept.user_id == user_id]
            
            if category:
                filters.append(Concept.category == category)
                
            if active_only:
                filters.append(Concept.active == True)
                
            query = (
                select(Concept)
                .where(and_(*filters))
                .options(selectinload(Concept.learnings))
                .order_by(desc(Concept.importance), Concept.name)
                .offset(offset)
                .limit(limit)
            )
            
            result = await session.execute(query)
            concepts = result.scalars().all()
            
            return concepts
    
    async def update_concept(self, concept_id: str, concept_data: Dict[str, Any]) -> Optional[Concept]:
        """Update an existing concept."""
        # Get the concept to update
        concept = await self.get_concept(concept_id)
        if not concept:
            return None
        
        async with get_session(self.db_session) as session:
            # Update fields
            update_data = {
                k: v for k, v in concept_data.items() 
                if k not in ['id', 'user_id', 'created_at'] and hasattr(Concept, k)
            }
            
            for key, value in update_data.items():
                setattr(concept, key, value)
            
            # Update the timestamp
            concept.updated_at = datetime.now(timezone.utc)
            
            await session.commit()
            
            logger.info(f"Updated concept {concept.id} ({concept.name})")
            return concept
    
    async def delete_concept(self, concept_id: str) -> bool:
        """Delete a concept by ID."""
        # Find the concept
        concept = await self.get_concept(concept_id)
        if not concept:
            return False
        
        async with get_session(self.db_session) as session:
            # Remove it
            await session.delete(concept)
            await session.commit()
        
        logger.info(f"Deleted concept {concept_id}")
        return True
    
    # Connection methods
    
    async def create_connection(self, connection_data: Dict[str, Any]) -> Connection:
        """Create a new connection between concepts."""
        async with get_session(self.db_session) as session:
            # Create the connection
            connection = Connection(**{
                k: v for k, v in connection_data.items() 
                if k not in ['id'] and hasattr(Connection, k)
            })
            
            if 'id' not in connection_data or not connection_data['id']:
                connection.id = str(uuid.uuid4())
            else:
                connection.id = connection_data['id']
                
            session.add(connection)
            
            # If bidirectional, create the reverse connection too
            if connection.bidirectional:
                reverse_connection = Connection(
                    id=str(uuid.uuid4()),
                    user_id=connection.user_id,
                    source_id=connection.target_id,
                    target_id=connection.source_id,
                    relation_type=connection.relation_type,
                    strength=connection.strength,
                    bidirectional=True
                )
                session.add(reverse_connection)
            
            await session.commit()
            
            logger.info(f"Created connection {connection.id} between concepts")
            return connection
    
    async def get_connection(self, connection_id: str) -> Optional[Connection]:
        """Get a connection by ID."""
        async with get_session(self.db_session) as session:
            query = (
                select(Connection)
                .where(Connection.id == connection_id)
                .options(
                    selectinload(Connection.source),
                    selectinload(Connection.target)
                )
            )
            
            result = await session.execute(query)
            connection = result.scalar_one_or_none()
            
            return connection
    
    async def get_connections_for_concept(
        self, 
        concept_id: str,
        include_incoming: bool = True,
        include_outgoing: bool = True,
        active_only: bool = True
    ) -> List[Connection]:
        """Get connections for a specific concept."""
        filters = []
        
        if include_incoming and include_outgoing:
            filters.append(
                or_(
                    Connection.source_id == concept_id,
                    Connection.target_id == concept_id
                )
            )
        elif include_incoming:
            filters.append(Connection.target_id == concept_id)
        elif include_outgoing:
            filters.append(Connection.source_id == concept_id)
        else:
            return []  # No connections to fetch
        
        async with get_session(self.db_session) as session:
            if active_only:
                filters.append(Connection.active == True)
                
            query = (
                select(Connection)
                .where(and_(*filters))
                .options(
                    selectinload(Connection.source),
                    selectinload(Connection.target)
                )
                .order_by(desc(Connection.strength))
            )
            
            result = await session.execute(query)
            connections = result.scalars().all()
            
            return connections
    
    async def update_connection(self, connection_id: str, connection_data: Dict[str, Any]) -> Optional[Connection]:
        """Update an existing connection."""
        # Get the connection to update
        connection = await self.get_connection(connection_id)
        if not connection:
            return None
        
        async with get_session(self.db_session) as session:
            # Check if bidirectionality is changing
            bidirectional_changed = (
                'bidirectional' in connection_data and
                connection_data['bidirectional'] != connection.bidirectional
            )
            
            # Update fields
            update_data = {
                k: v for k, v in connection_data.items() 
                if k not in ['id', 'user_id', 'created_at', 'source_id', 'target_id'] and hasattr(Connection, k)
            }
            
            for key, value in update_data.items():
                setattr(connection, key, value)
            
            # Update the timestamp
            connection.updated_at = datetime.now(timezone.utc)
            
            # Handle bidirectional changes
            if bidirectional_changed:
                # Find any reverse connections
                reverse_query = (
                    select(Connection)
                    .where(
                        and_(
                            Connection.source_id == connection.target_id,
                            Connection.target_id == connection.source_id,
                            Connection.user_id == connection.user_id
                        )
                    )
                )
                
                result = await session.execute(reverse_query)
                reverse_connection = result.scalar_one_or_none()
                
                if connection.bidirectional and not reverse_connection:
                    # Create a new reverse connection
                    new_reverse = Connection(
                        id=str(uuid.uuid4()),
                        user_id=connection.user_id,
                        source_id=connection.target_id,
                        target_id=connection.source_id,
                        relation_type=connection.relation_type,
                        strength=connection.strength,
                        bidirectional=True
                    )
                    session.add(new_reverse)
                    
                elif not connection.bidirectional and reverse_connection:
                    # Remove the reverse connection
                    await session.delete(reverse_connection)
            
            await session.commit()
            
            logger.info(f"Updated connection {connection.id}")
            return connection
    
    async def delete_connection(self, connection_id: str) -> bool:
        """Delete a connection by ID."""
        # Find the connection
        connection = await self.get_connection(connection_id)
        if not connection:
            return False
        
        async with get_session(self.db_session) as session:
            # If bidirectional, delete the reverse connection too
            if connection.bidirectional:
                reverse_query = (
                    select(Connection)
                    .where(
                        and_(
                            Connection.source_id == connection.target_id,
                            Connection.target_id == connection.source_id,
                            Connection.user_id == connection.user_id
                        )
                    )
                )
                
                result = await session.execute(reverse_query)
                reverse_connection = result.scalar_one_or_none()
                
                if reverse_connection:
                    await session.delete(reverse_connection)
                
            # Remove the connection
            await session.delete(connection)
            await session.commit()
        
        logger.info(f"Deleted connection {connection_id}")
        return True
