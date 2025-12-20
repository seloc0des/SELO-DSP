"""
User Repository for Single-User SELO AI Installation

SELO AI is designed as a single-user, single-persona system.
This repository ensures there is exactly ONE user per installation,
representing the sole owner/operator of this SELO AI instance.

Architecture:
- One user per installation
- One persona per user (managed by PersonaRepository)
- Session alignment for frontend/backend consistency
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from ..models.user import User
from ..session import AsyncSessionLocal, get_session

logger = logging.getLogger(__name__)

class UserRepository:
    """
    Repository for managing the single user in a SELO AI installation.
    
    In a single-user system, this typically manages one user record
    representing the installation owner/operator.
    """
    
    def __init__(self):
        """Initialize the user repository."""
        self.logger = logger
    
    async def get_or_create_default_user(self, session: Optional[AsyncSession] = None, user_id: Optional[str] = None) -> User:
        """
        Get the default (and typically only) user for this installation.
        Creates one if it doesn't exist.
        
        Args:
            session: Optional database session
            user_id: Optional specific user ID to use (for session alignment)
            
        Returns:
            User: The default user for this installation
        """
        async with get_session(session) as db:
            return await self._get_or_create_default_user_impl(db, user_id)
    
    async def _get_or_create_default_user_impl(self, session: AsyncSession, user_id: Optional[str] = None) -> User:
        """Implementation of get_or_create_default_user.
        
        For single-user SELO AI installations, this ensures there is exactly ONE user.
        The user_id parameter is used for session alignment but doesn't create multiple users.
        """
        try:
            # SINGLE-USER ARCHITECTURE: Always try to get the existing single user first
            result = await session.execute(select(User).limit(1))
            user = result.scalar_one_or_none()
            
            if user is not None:
                # Update last active and return the single existing user
                await self.update_last_active(user.id, session)
                self.logger.debug(f"Returning existing single user: {user.id}")
                return user
            
            # No user exists - create the single installation user
            # Use provided user_id for session alignment, or generate one
            final_user_id = user_id if user_id else None  # Let SQLAlchemy generate if None
            
            user = User(
                id=final_user_id,  # This will be auto-generated if None
                username="user",
                display_name="SELO AI User",
                is_active=True
            )
            session.add(user)
            await session.flush()  # Flush to get auto-generated ID without full commit
            self.logger.info(f"Created single installation user: {user.id}")
            return user
            
        except Exception as e:
            
            # Handle race condition: if INSERT failed due to duplicate key, try SELECT again
            from sqlalchemy.exc import IntegrityError
            if isinstance(e, IntegrityError) and 'users_username_key' in str(e):
                self.logger.warning(f"User already exists (race condition detected), retrying SELECT...")
                try:
                    result = await session.execute(select(User).limit(1))
                    user = result.scalar_one_or_none()
                    if user:
                        self.logger.info(f"Successfully retrieved existing user after race condition: {user.id}")
                        return user
                except Exception as retry_err:
                    self.logger.error(f"Retry SELECT also failed: {retry_err}", exc_info=True)
            
            self.logger.error(f"Error getting/creating single user: {str(e)}", exc_info=True)
            raise
    
    async def get_user_by_id(self, user_id: str, session: Optional[AsyncSession] = None) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID to look up
            session: Optional database session
            
        Returns:
            User or None if not found
        """
        async with get_session(session) as db:
            return await self._get_user_by_id_impl(user_id, db)
    
    async def _get_user_by_id_impl(self, user_id: str, session: AsyncSession) -> Optional[User]:
        """Implementation of get_user_by_id."""
        try:
            result = await session.execute(select(User).where(User.id == user_id))
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error getting user by ID {user_id}: {str(e)}", exc_info=True)
            return None
    
    async def update_last_active(self, user_id: str, session: Optional[AsyncSession] = None) -> bool:
        """
        Update user's last active timestamp.
        
        Args:
            user_id: User ID to update
            session: Optional database session
            
        Returns:
            bool: True if updated successfully
        """
        async with get_session(session) as db:
            return await self._update_last_active_impl(user_id, db)
    
    async def _update_last_active_impl(self, user_id: str, session: AsyncSession) -> bool:
        """Implementation of update_last_active."""
        try:
            await session.execute(
                update(User)
                .where(User.id == user_id)
                .values(last_active=datetime.now(timezone.utc))
            )
            return True
        except Exception as e:
            self.logger.error(f"Error updating last active for user {user_id}: {str(e)}", exc_info=True)
            return False
    
    async def update_preferences(self, user_id: str, preferences: Dict[str, Any], session: Optional[AsyncSession] = None) -> bool:
        """
        Update user preferences.
        
        Args:
            user_id: User ID to update
            preferences: Preferences dictionary (will be JSON serialized)
            session: Optional database session
            
        Returns:
            bool: True if updated successfully
        """
        import json
        
        async with get_session(session) as db:
            return await self._update_preferences_impl(user_id, preferences, db)
    
    async def _update_preferences_impl(self, user_id: str, preferences: Dict[str, Any], session: AsyncSession) -> bool:
        """Implementation of update_preferences."""
        import json
        
        try:
            preferences_json = json.dumps(preferences)
            await session.execute(
                update(User)
                .where(User.id == user_id)
                .values(preferences=preferences_json)
            )
            return True
        except Exception as e:
            self.logger.error(f"Error updating preferences for user {user_id}: {str(e)}", exc_info=True)
            return False
    
    async def get_user_preferences(self, user_id: str, session: Optional[AsyncSession] = None) -> Dict[str, Any]:
        """
        Get user preferences.
        
        Args:
            user_id: User ID to get preferences for
            session: Optional database session
            
        Returns:
            Dict: User preferences (empty dict if none or error)
        """
        async with get_session(session) as db:
            return await self._get_user_preferences_impl(user_id, db)
    
    async def _get_user_preferences_impl(self, user_id: str, session: AsyncSession) -> Dict[str, Any]:
        """Implementation of get_user_preferences."""
        import json
        
        try:
            user = await self._get_user_by_id_impl(user_id, session)
            if user and user.preferences:
                return json.loads(user.preferences)
            return {}
        except Exception as e:
            self.logger.error(f"Error getting preferences for user {user_id}: {str(e)}", exc_info=True)
            return {}
    
    async def get_installation_user_id(self) -> Optional[str]:
        """
        Get the ID of the installation's default user.
        Convenience method for single-user systems.
        
        Returns:
            str: User ID of the installation user, or None if error
        """
        try:
            user = await self.get_or_create_default_user()
            return user.id if user else None
        except Exception as e:
            self.logger.error(f"Error getting installation user ID: {str(e)}", exc_info=True)
            return None
