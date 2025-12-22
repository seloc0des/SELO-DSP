"""
Persona Repository

This module provides the repository layer for persona-related database operations,
including CRUD operations for personas, traits, and evolutions.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Tuple

from sqlalchemy import select, update, delete, and_, or_, desc, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..base import Base
from ..session import get_session, AsyncSessionLocal
from ..models.persona import Persona, PersonaTrait, PersonaEvolution
from ..models.conversation import Conversation, ConversationMessage, Memory
from ..models.reflection import Reflection, ReflectionMemory, ReflectionSchedule, RelationshipQuestionQueue

logger = logging.getLogger("selo.db.repositories.persona")


class PersonaRepository:
    """Repository for persona-related database operations."""
    
    def __init__(self):
        """Initialize the persona repository."""
        # No persistent resources to initialize eagerly; session is acquired per-call
        logger.debug("PersonaRepository initialized")
    
    async def close(self):
        """Close any resources."""
        # Explicit no-op for interface compatibility; log for observability
        logger.debug("PersonaRepository closed (no-op)")
    
    # === Persona CRUD operations ===
    
    async def create_persona(
        self, 
        persona_data: Dict[str, Any], 
        session: Optional[AsyncSession] = None
    ) -> Persona:
        """
        Create a new persona.
        
        Args:
            persona_data: Dictionary with persona data
            session: Optional database session
            
        Returns:
            Created persona object
        """
        async with get_session(session) as session:
            # Create persona object
            persona = Persona(**persona_data)
            
            # Add to session
            session.add(persona)
            await session.flush()  # Flush to get auto-generated ID without full commit
            
            logger.info(f"Created persona {persona.id} for user {persona.user_id}")
            return persona

    async def get_persona(
        self, 
        persona_id: str, 
        session: Optional[AsyncSession] = None,
        include_traits: bool = False,
        include_evolutions: bool = False
    ) -> Optional[Persona]:
        """
        Get a persona by ID.
        
        Args:
            persona_id: Persona ID
            session: Optional database session
            include_traits: Whether to include persona traits
            include_evolutions: Whether to include persona evolutions
            
        Returns:
            Persona object or None if not found
        """
        async with get_session(session) as session:
            # Build query
            query = select(Persona).where(Persona.id == persona_id)
            
            # Include relationships if requested
            if include_traits:
                query = query.options(selectinload(Persona.traits))
            if include_evolutions:
                query = query.options(selectinload(Persona.evolutions))
                
            # Execute query
            result = await session.execute(query)
            persona = result.scalars().first()
            
            return persona
    
    async def get_persona_by_id(
        self, 
        persona_id: str, 
        session: Optional[AsyncSession] = None,
        include_traits: bool = False,
        include_evolutions: bool = False
    ) -> Optional[Persona]:
        """
        Get a persona by ID (alias for get_persona for backward compatibility).
        
        Args:
            persona_id: Persona ID
            session: Optional database session
            include_traits: Whether to include persona traits
            include_evolutions: Whether to include persona evolutions
            
        Returns:
            Persona object or None if not found
        """
        return await self.get_persona(
            persona_id=persona_id,
            session=session,
            include_traits=include_traits,
            include_evolutions=include_evolutions
        )
    
    async def get_persona_by_user(
        self, 
        user_id: str, 
        is_default: bool = None,
        is_active: bool = None,
        session: Optional[AsyncSession] = None,
        include_traits: bool = False
    ) -> Optional[Persona]:
        """
        Get a persona for a user, optionally filtering by default/active status.
        
        Args:
            user_id: User ID
            is_default: Filter by default status (optional)
            is_active: Filter by active status (optional)
            session: Optional database session
            include_traits: Whether to include persona traits
            
        Returns:
            Persona object or None if not found
        """
        async with get_session(session) as session:
            # Build query conditions
            conditions = [Persona.user_id == user_id]
            
            if is_default is not None:
                conditions.append(Persona.is_default == is_default)
            if is_active is not None:
                conditions.append(Persona.is_active == is_active)
                
            # Build query
            query = select(Persona).where(and_(*conditions))
            
            # Include traits if requested
            if include_traits:
                query = query.options(selectinload(Persona.traits))
                
            # Execute query
            result = await session.execute(query)
            persona = result.scalars().first()
            
            return persona
    
    async def get_personas_for_user(
        self, 
        user_id: str, 
        limit: int = 10,
        offset: int = 0,
        include_inactive: bool = False,
        session: Optional[AsyncSession] = None
    ) -> List[Persona]:
        """
        Get all personas for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of personas to return
            offset: Pagination offset
            include_inactive: Whether to include inactive personas
            session: Optional database session
            
        Returns:
            List of persona objects
        """
        async with get_session(session) as session:
            # Build query conditions
            conditions = [Persona.user_id == user_id]
            
            if not include_inactive:
                conditions.append(Persona.is_active == True)
                
            # Build query
            query = (select(Persona)
                    .where(and_(*conditions))
                    .order_by(Persona.is_default.desc(), Persona.last_modified.desc())
                    .offset(offset)
                    .limit(limit))
                    
            # Execute query
            result = await session.execute(query)
            personas = result.scalars().all()
            
            return list(personas)
    
    async def update_persona(
        self, 
        persona_id: str, 
        persona_data: Dict[str, Any], 
        session: Optional[AsyncSession] = None
    ) -> Optional[Persona]:
        """
        Update a persona.
        
        Args:
            persona_id: Persona ID
            persona_data: Dictionary with persona data to update
            session: Optional database session
            
        Returns:
            Updated persona object or None if not found
        """
        async with get_session(session) as session:
            # Get persona
            persona = await self.get_persona(persona_id, session)
            if not persona:
                logger.warning(f"Persona {persona_id} not found for update")
                return None
            
            # Update persona attributes
            for key, value in persona_data.items():
                if hasattr(persona, key):
                    setattr(persona, key, value)
            
            # Update last_modified timestamp
            persona.last_modified = datetime.now(timezone.utc)
            
            # Commit changes to database
            await session.commit()
            
            logger.info(f"Updated persona {persona_id}")
            return persona

    async def ensure_schema(self, session: Optional[AsyncSession] = None) -> None:
        """Ensure required schema (mantra, first_thoughts, boot_directive columns) exists. Safe and idempotent."""
        async with get_session(session) as session:
            try:
                await session.execute(text("ALTER TABLE personas ADD COLUMN IF NOT EXISTS mantra TEXT"))
                await session.execute(text("ALTER TABLE personas ADD COLUMN IF NOT EXISTS first_thoughts TEXT"))
                await session.execute(text("ALTER TABLE personas ADD COLUMN IF NOT EXISTS boot_directive TEXT"))
            except Exception as e:
                logger.debug(f"ensure_schema skipped or failed softly: {e}")

    async def count_personas_for_user(
        self,
        user_id: str,
        session: Optional[AsyncSession] = None,
    ) -> int:
        async with get_session(session) as session:
            result = await session.execute(
                select(func.count(Persona.id)).where(Persona.user_id == user_id)
            )
            return result.scalar() or 0

    async def count_traits_for_user(
        self,
        user_id: str,
        session: Optional[AsyncSession] = None,
    ) -> int:
        async with get_session(session) as session:
            result = await session.execute(
                select(func.count(PersonaTrait.id)).join(Persona, PersonaTrait.persona_id == Persona.id).where(Persona.user_id == user_id)
            )
            return result.scalar() or 0

    async def count_evolutions_for_user(
        self,
        user_id: str,
        session: Optional[AsyncSession] = None,
    ) -> int:
        async with get_session(session) as session:
            result = await session.execute(
                select(func.count(PersonaEvolution.id)).join(Persona, PersonaEvolution.persona_id == Persona.id).where(Persona.user_id == user_id)
            )
            return result.scalar() or 0

    async def count_conversations_for_user(
        self,
        user_id: str,
        session: Optional[AsyncSession] = None,
    ) -> int:
        async with get_session(session) as session:
            result = await session.execute(
                select(func.count(Conversation.id)).where(Conversation.user_id == user_id)
            )
            return result.scalar() or 0

    async def count_memories_for_user(
        self,
        user_id: str,
        session: Optional[AsyncSession] = None,
    ) -> int:
        async with get_session(session) as session:
            result = await session.execute(
                select(func.count(Memory.id)).where(Memory.user_id == user_id)
            )
            return result.scalar() or 0

    async def persona_is_initialized(
        self,
        user_id: str,
        session: Optional[AsyncSession] = None,
    ) -> bool:
        try:
            persona = await self.get_persona_by_user(
                user_id=user_id,
                is_default=True,
                is_active=True,
                session=session,
                include_traits=True,
            )
            if not persona:
                return False

            name = (getattr(persona, "name", "") or "").strip()
            if not name or name == "SELO":
                return False

            mantra = (getattr(persona, "mantra", "") or "").strip()
            if not mantra:
                return False

            description = (getattr(persona, "description", "") or "").strip()
            if len(description) < 10:
                return False

            values = getattr(persona, "values", None)
            if not values or not isinstance(values, dict):
                return False

            traits = getattr(persona, "traits", []) or []
            if len(traits) < 1:
                return False

            evolutions = await self.get_evolutions_for_persona(
                persona_id=persona.id,
                limit=1,
                session=session,
            )
            if not evolutions:
                return False

            return True
        except Exception:
            return False

    async def delete_all_persona_data_for_user(
        self,
        user_id: str,
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, int]:
        """Delete persona-related data for the given user and return deletion counts."""

        deletions = {
            "personas": 0,
            "persona_traits": 0,
            "persona_evolutions": 0,
            "memories": 0,
            "reflections": 0,
            "reflection_memories": 0,
            "reflection_schedule": 0,
            "relationship_question_queue": 0,
            "conversation_messages": 0,
            "conversations": 0,
        }

        async with get_session(session) as session:
            async def _delete(stmt, key: str) -> None:
                result = await session.execute(stmt)
                count = result.rowcount if result.rowcount is not None else 0
                deletions[key] = deletions.get(key, 0) + count

            user_id_str = str(user_id)

            # Collect persona IDs for user first
            result = await session.execute(select(Persona.id).where(Persona.user_id == user_id))
            persona_ids = [row[0] for row in result.all()]

            # Delete reflections and related tables first
            result = await session.execute(select(Reflection.id).where(Reflection.user_profile_id == user_id_str))
            reflection_ids = [row[0] for row in result.all()]

            if reflection_ids:
                await _delete(delete(ReflectionMemory).where(ReflectionMemory.reflection_id.in_(reflection_ids)), "reflection_memories")
                await _delete(delete(Reflection).where(Reflection.id.in_(reflection_ids)), "reflections")

            await _delete(delete(ReflectionSchedule).where(ReflectionSchedule.user_profile_id == user_id_str), "reflection_schedule")
            await _delete(delete(RelationshipQuestionQueue).where(RelationshipQuestionQueue.user_profile_id == user_id_str), "relationship_question_queue")

            # Delete memories associated with user
            await _delete(delete(Memory).where(Memory.user_id == user_id), "memories")

            # Delete conversation messages and conversations for user
            result = await session.execute(select(Conversation.id).where(Conversation.user_id == user_id))
            conversation_ids = [row[0] for row in result.all()]

            if conversation_ids:
                await _delete(
                    delete(ConversationMessage).where(ConversationMessage.conversation_id.in_(conversation_ids)),
                    "conversation_messages",
                )
                await _delete(delete(Conversation).where(Conversation.id.in_(conversation_ids)), "conversations")

            if persona_ids:
                await _delete(delete(PersonaEvolution).where(PersonaEvolution.persona_id.in_(persona_ids)), "persona_evolutions")
                await _delete(delete(PersonaTrait).where(PersonaTrait.persona_id.in_(persona_ids)), "persona_traits")
                await _delete(delete(Persona).where(Persona.id.in_(persona_ids)), "personas")

            logger.info(
                "Cleared persona-related data for user %s (details: %s)",
                user_id,
                deletions,
            )

        return deletions
    
    async def delete_persona(
        self, 
        persona_id: str, 
        session: Optional[AsyncSession] = None
    ) -> bool:
        """
        Delete a persona.
        
        Args:
            persona_id: Persona ID
            session: Optional database session
            
        Returns:
            True if deleted, False if not found
        """
        async with get_session(session) as session:
            # Delete persona
            query = delete(Persona).where(Persona.id == persona_id)
            result = await session.execute(query)
            
            # Check if deleted
            if result.rowcount > 0:
                logger.info(f"Deleted persona {persona_id}")
                return True
            else:
                logger.warning(f"Persona {persona_id} not found for delete")
                return False
    
    # === Trait operations ===
    
    async def create_trait(
        self, 
        trait_data: Dict[str, Any], 
        session: Optional[AsyncSession] = None
    ) -> PersonaTrait:
        """
        Create a new persona trait.
        
        Args:
            trait_data: Dictionary with trait data
            session: Optional database session
            
        Returns:
            Created trait object
        """
        async with get_session(session) as session:
            # Create trait object
            trait = PersonaTrait(**trait_data)
            
            # Add to session
            session.add(trait)
            await session.flush()  # Flush to get auto-generated ID without full commit
            
            logger.info(f"Created trait {trait.id} for persona {trait.persona_id}")
            return trait
    
    async def get_trait(
        self, 
        trait_id: str, 
        session: Optional[AsyncSession] = None
    ) -> Optional[PersonaTrait]:
        """
        Get a trait by ID.
        
        Args:
            trait_id: Trait ID
            session: Optional database session
            
        Returns:
            Trait object or None if not found
        """
        async with get_session(session) as session:
            # Query trait
            query = select(PersonaTrait).where(PersonaTrait.id == trait_id)
            result = await session.execute(query)
            trait = result.scalars().first()
            
            return trait
    
    async def get_traits_for_persona(
        self, 
        persona_id: str, 
        category: Optional[str] = None,
        session: Optional[AsyncSession] = None
    ) -> List[PersonaTrait]:
        """
        Get all traits for a persona.
        
        Args:
            persona_id: Persona ID
            category: Optional category filter
            session: Optional database session
            
        Returns:
            List of trait objects
        """
        async with get_session(session) as session:
            # Build query conditions
            conditions = [PersonaTrait.persona_id == persona_id]
            
            if category:
                conditions.append(PersonaTrait.category == category)
                
            # Build query
            query = (select(PersonaTrait)
                    .where(and_(*conditions))
                    .order_by(PersonaTrait.category, PersonaTrait.name))
                    
            # Execute query
            result = await session.execute(query)
            traits = result.scalars().all()
            
            return list(traits)
    
    async def update_trait(
        self, 
        trait_id: str, 
        trait_data: Dict[str, Any], 
        session: Optional[AsyncSession] = None
    ) -> Optional[PersonaTrait]:
        """
        Update a trait.
        
        Args:
            trait_id: Trait ID
            trait_data: Dictionary with trait data to update
            session: Optional database session
            
        Returns:
            Updated trait object or None if not found
        """
        async with get_session(session) as session:
            # Get trait
            trait = await self.get_trait(trait_id, session)
            if not trait:
                logger.warning(f"Trait {trait_id} not found for update")
                return None
            
            # Update trait attributes
            for key, value in trait_data.items():
                if hasattr(trait, key):
                    setattr(trait, key, value)
            
            # Update last_updated timestamp
            trait.last_updated = datetime.now(timezone.utc)
            
            logger.info(f"Updated trait {trait_id}")
            return trait
    
    async def delete_trait(
        self, 
        trait_id: str, 
        session: Optional[AsyncSession] = None
    ) -> bool:
        """
        Delete a trait.
        
        Args:
            trait_id: Trait ID
            session: Optional database session
            
        Returns:
            True if deleted, False if not found
        """
        async with get_session(session) as session:
            # Delete trait
            query = delete(PersonaTrait).where(PersonaTrait.id == trait_id)
            result = await session.execute(query)
            
            # Check if deleted
            if result.rowcount > 0:
                logger.info(f"Deleted trait {trait_id}")
                return True
            else:
                logger.warning(f"Trait {trait_id} not found for delete")
                return False
    
    # === Evolution operations ===
    
    async def create_evolution(
        self, 
        evolution_data: Dict[str, Any], 
        session: Optional[AsyncSession] = None
    ) -> PersonaEvolution:
        """
        Create a new persona evolution.
        
        Args:
            evolution_data: Dictionary with evolution data
            session: Optional database session
            
        Returns:
            Created evolution object
        """
        async with get_session(session) as session:
            # Create evolution object
            evolution = PersonaEvolution(**evolution_data)
            
            # Add to session
            session.add(evolution)
            await session.flush()  # Flush to get auto-generated ID without full commit
            
            # Update persona's evolution count and last_evolution
            persona_id = evolution.persona_id
            await session.execute(
                update(Persona)
                .where(Persona.id == persona_id)
                .values(
                    evolution_count=Persona.evolution_count + 1,
                    last_evolution=datetime.now(timezone.utc)
                )
            )
            
            logger.info(f"Created evolution {evolution.id} for persona {evolution.persona_id}")

            # Generate and update persona summary based on evolution (only on evolution, not bootstrap)
            # Skip summary generation for installation evolutions to maintain clean-slate initialization
            if getattr(evolution, 'source_type', '') not in ('bootstrap', 'boot_reflection', 'persona_created'):
                try:
                    await self._update_persona_summary_on_evolution(persona_id, evolution, session)
                except Exception as summary_err:
                    logger.warning(f"Failed to update persona summary on evolution: {summary_err}")
            else:
                logger.debug(f"Skipping summary generation for installation evolution {evolution.id} (source_type={getattr(evolution, 'source_type', None)})")

            # Emit live socket event to notify frontend of persona evolution (default namespace)
            try:
                # Resolve user_id by loading the persona (lightweight select)
                result = await session.execute(select(Persona).where(Persona.id == persona_id))
                persona_obj = result.scalars().first()
                user_id = getattr(persona_obj, 'user_id', None)

                # Defer import to avoid circular dependencies
                from ...socketio.registry import get_socketio_server  # type: ignore
                sio = get_socketio_server()
                if sio is not None:
                    payload = {
                        "event": "persona.evolution",
                        "persona_id": persona_id,
                        "user_id": user_id,
                        "evolution_id": evolution.id,
                        "timestamp": getattr(evolution, 'timestamp', datetime.now(timezone.utc)).isoformat(),
                        # Provide a brief, best-effort summary for UI
                        "summary": {
                            "source_type": getattr(evolution, 'source_type', None),
                            "impact_score": getattr(evolution, 'impact_score', None),
                            "changes_keys": list((evolution.changes or {}).keys()) if getattr(evolution, 'changes', None) else [],
                        },
                    }
                    # Emit on default namespace so personaService.js (default socket) receives it
                    await sio.emit('persona.evolution', payload)
                else:
                    logger.debug("Socket.IO server not available; skipping persona.evolution emit")
            except Exception as emit_err:
                logger.warning(f"Failed to emit persona.evolution event: {emit_err}")
            return evolution
    
    async def get_evolution(
        self, 
        evolution_id: str, 
        session: Optional[AsyncSession] = None
    ) -> Optional[PersonaEvolution]:
        """
        Get an evolution by ID.
        
        Args:
            evolution_id: Evolution ID
            session: Optional database session
            
        Returns:
            Evolution object or None if not found
        """
        async with get_session(session) as session:
            # Query evolution
            query = select(PersonaEvolution).where(PersonaEvolution.id == evolution_id)
            result = await session.execute(query)
            evolution = result.scalars().first()
            
            return evolution
    
    async def get_evolutions_for_persona(
        self, 
        persona_id: str, 
        limit: int = 10,
        offset: int = 0,
        source_type: Optional[str] = None,
        session: Optional[AsyncSession] = None
    ) -> List[PersonaEvolution]:
        """
        Get all evolutions for a persona.
        
        Args:
            persona_id: Persona ID
            limit: Maximum number of evolutions to return
            offset: Pagination offset
            source_type: Optional source type filter
            session: Optional database session
            
        Returns:
            List of evolution objects
        """
        async with get_session(session) as session:
            # Build query conditions
            conditions = [PersonaEvolution.persona_id == persona_id]
            
            if source_type:
                conditions.append(PersonaEvolution.source_type == source_type)
                
            # Build query
            query = (select(PersonaEvolution)
                    .where(and_(*conditions))
                    .order_by(PersonaEvolution.timestamp.desc())
                    .offset(offset)
                    .limit(limit))
                    
            # Execute query
            result = await session.execute(query)
            evolutions = result.scalars().all()
            
            return list(evolutions)
    
    async def update_evolution(
        self, 
        evolution_id: str, 
        evolution_data: Dict[str, Any], 
        session: Optional[AsyncSession] = None
    ) -> Optional[PersonaEvolution]:
        """
        Update an evolution.
        
        Args:
            evolution_id: Evolution ID
            evolution_data: Dictionary with evolution data to update
            session: Optional database session
            
        Returns:
            Updated evolution object or None if not found
        """
        async with get_session(session) as session:
            # Get evolution
            evolution = await self.get_evolution(evolution_id, session)
            if not evolution:
                logger.warning(f"Evolution {evolution_id} not found for update")
                return None
            
            # Update evolution attributes
            for key, value in evolution_data.items():
                if hasattr(evolution, key):
                    setattr(evolution, key, value)
            
            logger.info(f"Updated evolution {evolution_id}")
            return evolution
    
    async def delete_evolution(
        self,
        evolution_id: str,
        session: Optional[AsyncSession] = None
    ) -> bool:
        """
        Delete a persona evolution record.
        
        Args:
            evolution_id: Evolution ID to delete
            session: Optional database session
            
        Returns:
            True if deleted, False if not found
        """
        async with get_session(session) as session:
            # Delete evolution
            query = delete(PersonaEvolution).where(PersonaEvolution.id == evolution_id)
            result = await session.execute(query)
            
            # Check if deleted
            if result.rowcount > 0:
                logger.info(f"Deleted evolution {evolution_id}")
                return True
            else:
                logger.warning(f"Evolution {evolution_id} not found for delete")
                return False
    
    # === Advanced operations ===
    
    async def set_default_persona(
        self,
        persona_id: str,
        user_id: str,
        session: Optional[AsyncSession] = None
    ) -> bool:
        """
        Set a persona as the default for a user.
        
        Args:
            persona_id: Persona ID to set as default
            user_id: User ID
            session: Optional database session
            
        Returns:
            True if successful, False otherwise
        """
        async with get_session(session) as session:
            # Clear default flag from all user personas
            await session.execute(
                update(Persona)
                .where(Persona.user_id == user_id)
                .values(is_default=False)
            )
            
            # Set default flag for specified persona
            result = await session.execute(
                update(Persona)
                .where(and_(Persona.id == persona_id, Persona.user_id == user_id))
                .values(is_default=True)
            )
            
            # Check if updated
            if result.rowcount > 0:
                logger.info(f"Set persona {persona_id} as default for user {user_id}")
                return True
            else:
                logger.warning(f"Failed to set persona {persona_id} as default for user {user_id}")
                return False

    async def ensure_singleton_for_user(
        self,
        user_id: str,
        session: Optional[AsyncSession] = None
    ) -> Optional[Persona]:
        """
        Ensure there is exactly one active default persona for a user.

        - If no persona exists: create a default one.
        - If multiple defaults exist: keep the most recently modified and unset default on others.
        - If multiple active personas exist: keep the default active and set others to inactive.

        Returns the single default Persona.
        """
        async with get_session(session) as session:
            # Fetch all active personas for the user ordered by default flag then last_modified desc
            personas = await self.get_personas_for_user(
                user_id=user_id,
                include_inactive=False,
                session=session,
                limit=100,
                offset=0,
            )

            # If none exist, create one
            if not personas:
                persona = await self.create_persona(
                    {
                        "id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "name": "SELO",
                        "description": "",  # Empty description so bootstrap can detect it needs initialization
                        "is_default": True,
                        "is_active": True,
                    },
                    session=session,
                )
                return persona

            # Identify default personas
            default_personas = [p for p in personas if getattr(p, "is_default", False)]

            # If multiple defaults, keep the most recently modified as default
            if len(default_personas) > 1:
                default_personas.sort(key=lambda p: getattr(p, "last_modified", datetime.min), reverse=True)
                keeper = default_personas[0]
                to_unset = [p for p in default_personas[1:]]
                for p in to_unset:
                    await self.update_persona(p.id, {"is_default": False}, session=session)
                default_personas = [keeper]

            # If no default among active, choose the most recently modified as the default
            if not default_personas:
                personas.sort(key=lambda p: getattr(p, "last_modified", datetime.min), reverse=True)
                keeper = personas[0]
                await self.set_default_persona(keeper.id, user_id, session=session)
                default_personas = [keeper]

            default_persona = default_personas[0]

            # Deactivate non-default active personas
            for p in personas:
                if p.id != default_persona.id and getattr(p, "is_active", True):
                    await self.update_persona(p.id, {"is_active": False, "is_default": False}, session=session)

            return await self.get_persona(default_persona.id, session=session)

    async def get_or_create_default_persona(
        self,
        user_id: str,
        session: Optional[AsyncSession] = None,
        include_traits: bool = False,
        include_evolutions: bool = False,
    ) -> Persona:
        """
        Get the single default persona for a user, creating and enforcing singleton if necessary.
        
        When include flags are provided, the returned Persona will have the requested
        relationships eagerly loaded so it can be safely serialized after the session closes.
        """
        persona = await self.get_persona_by_user(
            user_id=user_id,
            is_default=True,
            is_active=True,
            session=session,
            include_traits=include_traits,
        )
        if persona:
            # If caller requested additional relationships beyond what was loaded by get_persona_by_user,
            # re-fetch the persona with the full include set to avoid lazy-load on a detached instance.
            if include_evolutions:
                persona = await self.get_persona(
                    persona.id,
                    session=session,
                    include_traits=include_traits,
                    include_evolutions=include_evolutions,
                )
            return persona
        # Enforce singleton (creates if none)
        default_persona = await self.ensure_singleton_for_user(user_id=user_id, session=session)
        # If include flags requested, fetch with relationships loaded
        if default_persona and (include_traits or include_evolutions):
            default_persona = await self.get_persona(
                default_persona.id,
                session=session,
                include_traits=include_traits,
                include_evolutions=include_evolutions,
            )
        return default_persona
    
    async def get_trait_evolution(
        self,
        persona_id: str,
        trait_name: str,
        trait_category: Optional[str] = None,
        limit: int = 10,
        session: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the evolution history of a specific trait.
        
        Args:
            persona_id: Persona ID
            trait_name: Name of the trait
            trait_category: Optional category of the trait
            limit: Maximum number of evolutions to return
            session: Optional database session
            
        Returns:
            List of evolution history entries for the trait
        """
        async with get_session(session) as session:
            # Get evolutions for the persona
            evolutions = await self.get_evolutions_for_persona(
                persona_id=persona_id,
                limit=100,  # Get more to filter for trait changes
                session=session
            )
            
            # Extract trait changes
            trait_history = []
            
            for evolution in evolutions:
                # Check if this evolution contains changes to the specified trait
                if "traits" not in evolution.changes:
                    continue
                    
                trait_changes = evolution.changes["traits"]
                
                # Ensure trait_changes is a list, handle legacy string format
                if not isinstance(trait_changes, list):
                    continue
                
                for change in trait_changes:
                    # Skip if change is not a dict (legacy data format)
                    if not isinstance(change, dict):
                        continue
                        
                    if change.get("name") != trait_name:
                        continue
                        
                    if trait_category and change.get("category") != trait_category:
                        continue
                        
                    # Found a change for this trait
                    trait_history.append({
                        "evolution_id": evolution.id,
                        "timestamp": evolution.timestamp,
                        "old_value": change.get("old_value"),
                        "new_value": change.get("new_value"),
                        "reasoning": evolution.reasoning,
                        "confidence": evolution.confidence,
                        "source_type": evolution.source_type
                    })
                    
                    # Break if we have enough history entries
                    if len(trait_history) >= limit:
                        break
                
                # Break if we have enough history entries
                if len(trait_history) >= limit:
                    break
            
            return trait_history
    
    async def upsert_trait(
        self,
        user_id: str,
        name: str,
        value: float,
        weight: float = 1.0,  # Map to confidence for compatibility
        description: str = "",
        category: str = "general",
        locked: bool = False,  # Ignored - not in PersonaTrait model
        session: Optional[AsyncSession] = None
    ) -> PersonaTrait:
        """
        Create or update a trait for the user's persona.
        
        Args:
            user_id: User ID
            name: Trait name
            value: Trait value (0.0-1.0)
            weight: Trait weight
            description: Trait description
            category: Trait category
            locked: Whether trait is locked
            session: Optional database session
            
        Returns:
            The created or updated PersonaTrait
        """
        async with get_session(session) as session:
            try:
                # Get the user's persona
                persona = await self.get_or_create_default_persona(user_id, session=session)
                
                # Check if trait already exists
                existing_trait_query = select(PersonaTrait).where(
                    and_(
                        PersonaTrait.persona_id == persona.id,
                        PersonaTrait.name == name
                    )
                )
                result = await session.execute(existing_trait_query)
                existing_trait = result.scalar_one_or_none()
                
                if existing_trait:
                    # Update existing trait
                    existing_trait.value = value
                    existing_trait.confidence = weight  # Map weight to confidence
                    existing_trait.description = description
                    existing_trait.category = category
                    existing_trait.last_updated = datetime.now(timezone.utc)
                    
                    logger.info(f"Updated trait {name} for persona {persona.id}")
                    return existing_trait
                else:
                    # Create new trait
                    trait_data = {
                        "persona_id": persona.id,
                        "name": name,
                        "value": value,
                        "confidence": weight,  # Map weight to confidence
                        "description": description,
                        "category": category
                    }
                    
                    return await self.create_trait(trait_data, session=session)
                    
            except Exception as e:
                logger.error(f"Error upserting trait {name}: {str(e)}", exc_info=True)
                await session.rollback()
                raise

    async def _update_persona_summary_on_evolution(
        self,
        persona_id: str,
        evolution: PersonaEvolution,
        session: AsyncSession
    ) -> None:
        """
        Generate and update persona summary based on evolution.
        Only called during evolution events, not during bootstrap.
        
        Args:
            persona_id: Persona ID
            evolution: Evolution object that triggered this update
            session: Database session
        """
        try:
            # Get current persona with traits
            persona = await self.get_persona(persona_id, session=session, include_traits=True)
            if not persona:
                logger.warning(f"Persona {persona_id} not found for summary update")
                return

            # Generate summary from current persona state and evolution context
            summary = await self._generate_persona_summary(persona, evolution, session)
            
            if summary:
                # Validate compliance before persisting
                if self._validate_summary_compliance(summary):
                    # Update persona description with the new summary
                    await session.execute(
                        update(Persona)
                        .where(Persona.id == persona_id)
                        .values(description=summary)
                    )
                    logger.info(f"Updated persona {persona_id} summary on evolution {evolution.id}")
                else:
                    logger.warning(f"Summary failed compliance check for persona {persona_id}, not persisting")
            else:
                logger.warning(f"Failed to generate summary for persona {persona_id}")
                
        except Exception as e:
            logger.error(f"Error updating persona summary on evolution: {e}", exc_info=True)

    async def _generate_persona_summary(
        self,
        persona: Persona,
        evolution: PersonaEvolution,
        session: AsyncSession
    ) -> Optional[str]:
        """
        Generate a context-rich summary of the persona's current state.
        
        Args:
            persona: Current persona object
            evolution: Evolution that triggered this summary
            session: Database session
            
        Returns:
            Generated summary or None if generation failed
        """
        try:
            # Import LLM router for summary generation
            try:
                from ...llm.router import LLMRouter
                from ...api.dependencies import get_llm_router
                llm_router = await get_llm_router()
            except Exception:
                logger.warning("Could not get LLM router for summary generation")
                return None

            # Build context for summary generation
            persona_dict = persona.to_dict() if hasattr(persona, "to_dict") else {
                "description": getattr(persona, "description", ""),
                "values": getattr(persona, "values", {}) or {},
                "knowledge_domains": getattr(persona, "knowledge_domains", []) or [],
                "communication_style": getattr(persona, "communication_style", {}) or {},
            }

            # Get recent evolutions for context (last 3)
            recent_evolutions = await self.get_evolutions_for_persona(
                persona_id=persona.id,
                limit=3,
                session=session
            )

            # Build summary prompt
            summary_prompt = self._build_summary_prompt(persona_dict, evolution, recent_evolutions)
            
            # Generate summary using analytical model for precision
            response = await llm_router.route(
                task_type="analytical",
                prompt=summary_prompt,
                max_tokens=500,
                temperature=0.2
            )
            
            content = (response or {}).get("content") or (response or {}).get("completion") or ""
            
            if content:
                # Clean and extract summary
                summary = self._extract_summary_from_response(content)
                return summary
            else:
                logger.warning("Empty response from LLM for summary generation")
                return None
                
        except Exception as e:
            logger.error(f"Error generating persona summary: {e}", exc_info=True)
            return None

    def _build_summary_prompt(
        self,
        persona_dict: Dict[str, Any],
        current_evolution: PersonaEvolution,
        recent_evolutions: List[PersonaEvolution]
    ) -> str:
        """
        Build prompt for persona summary generation.
        
        Args:
            persona_dict: Current persona data
            current_evolution: Evolution that triggered this summary
            recent_evolutions: Recent evolution history
            
        Returns:
            Summary generation prompt
        """
        # Extract current state
        description = persona_dict.get("description", "")
        values = persona_dict.get("values", {})
        knowledge_domains = persona_dict.get("knowledge_domains", [])
        communication_style = persona_dict.get("communication_style", {})
        
        # Build evolution context
        evolution_context = []
        for evo in recent_evolutions[-3:]:  # Last 3 evolutions
            source_type = getattr(evo, "source_type", "unknown")
            reasoning = getattr(evo, "reasoning", "")
            changes = getattr(evo, "changes", {})
            evolution_context.append(f"- {source_type}: {reasoning} (changes: {list(changes.keys())})")
        
        evolution_text = "\n".join(evolution_context) if evolution_context else "No recent evolutions"
        
        # Current evolution details
        current_source = getattr(current_evolution, "source_type", "unknown")
        current_reasoning = getattr(current_evolution, "reasoning", "")
        current_changes = getattr(current_evolution, "changes", {})
        
        prompt = f"""Generate a concise, context-rich summary of SELO's current state as a Self Evolving Learning Organism.

CRITICAL CONSTRAINTS:
- Use SELO-specific identity language, never "AI" or "assistant"
- Base summary ONLY on provided persona data and evolution history
- Do NOT invent, assume, or fabricate any experiences or memories
- Keep summary factual and grounded in actual persona attributes
- Maximum 3-4 sentences

CURRENT PERSONA STATE:
Description: {description}
Values: {values}
Knowledge Domains: {knowledge_domains}
Communication Style: {communication_style}

RECENT EVOLUTION HISTORY:
{evolution_text}

CURRENT EVOLUTION:
Source: {current_source}
Reasoning: {current_reasoning}
Changes: {list(current_changes.keys())}

Generate a summary that reflects SELO's current evolved state based on this factual information:"""

        return prompt

    def _extract_summary_from_response(self, response: str) -> str:
        """
        Extract clean summary from LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Cleaned summary text
        """
        # Remove any markdown formatting
        summary = response.strip()
        if summary.startswith("```"):
            lines = summary.split("\n")
            summary = "\n".join(line for line in lines if not line.startswith("```"))
        
        # Remove any JSON formatting if present
        if summary.startswith("{") and summary.endswith("}"):
            try:
                import json
                data = json.loads(summary)
                summary = data.get("summary", summary)
            except:
                pass
        
        # Clean up extra whitespace
        summary = " ".join(summary.split())
        
        return summary

    def _validate_summary_compliance(self, summary: str) -> bool:
        """
        Validate that summary complies with identity constraints.
        
        Args:
            summary: Generated summary text
            
        Returns:
            True if compliant, False otherwise
        """
        if not summary or not isinstance(summary, str):
            return False
            
        # Use centralized ValidationHelper for consistent validation
        try:
            from ...constraints.validation_helpers import ValidationHelper
        except ImportError:
            from backend.constraints.validation_helpers import ValidationHelper
        
        is_compliant, violations = ValidationHelper.validate_text_with_fabrication(
            text=summary,
            context="summary",
            has_history=False,
            context_stage="summary"
        )
        
        if not is_compliant:
            ValidationHelper.log_violations(
                violations=violations,
                stage="summary generation",
                level="warning"
            )
            return False
        
        return True
