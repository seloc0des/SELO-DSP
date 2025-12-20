"""Enhanced episode trigger system for autobiographical episodes.

Provides additional triggers beyond high-intensity reflections:
- Conversation milestones (every N messages)
- Daily summary episodes
- Goal completion celebrations
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .autobiographical_episode_service import AutobiographicalEpisodeService
    from ..db.repositories.conversation import ConversationRepository
    from ..db.repositories.persona import PersonaRepository
    from ..db.repositories.user import UserRepository

logger = logging.getLogger("selo.agent.episode_triggers")


class EpisodeTriggerManager:
    """Manages additional episode generation triggers."""

    def __init__(
        self,
        *,
        episode_service: "AutobiographicalEpisodeService",
        conversation_repo: "ConversationRepository",
        persona_repo: "PersonaRepository",
        user_repo: "UserRepository",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._episode_service = episode_service
        self._conversation_repo = conversation_repo
        self._persona_repo = persona_repo
        self._user_repo = user_repo
        
        # Configuration with defaults
        config = config or {}
        self._milestone_enabled = bool(config.get("milestone_enabled", True))
        self._milestone_interval = int(config.get("milestone_interval", 10))
        self._daily_summary_enabled = bool(config.get("daily_summary_enabled", True))
        self._goal_celebration_enabled = bool(config.get("goal_celebration_enabled", True))
        
        # Track last milestone for each user
        self._last_milestone_count: Dict[str, int] = {}
        self._last_daily_summary: Dict[str, datetime] = {}

    async def check_conversation_milestone(
        self,
        user_id: str,
        persona_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Check if conversation has reached a milestone (every N messages).
        
        Args:
            user_id: User identifier
            persona_id: Optional persona identifier (will resolve if not provided)
            
        Returns:
            Generated episode dict if milestone reached, None otherwise
        """
        if not self._milestone_enabled:
            return None
            
        try:
            # Count total messages for this session (user_id is session_id in SELO)
            # Note: In SELO's architecture, user_id is actually the session_id
            messages = await self._conversation_repo.get_conversation_history(
                session_id=user_id,
                limit=1000,  # Get enough to count
            )
            total_count = len(messages)
            
            # Check if we've crossed a milestone threshold
            last_count = self._last_milestone_count.get(user_id, 0)
            current_milestone = (total_count // self._milestone_interval) * self._milestone_interval
            last_milestone = (last_count // self._milestone_interval) * self._milestone_interval
            
            if current_milestone > last_milestone and current_milestone > 0:
                # Milestone reached!
                self._last_milestone_count[user_id] = total_count
                
                # Resolve persona if needed
                if not persona_id:
                    persona_id = await self._resolve_persona_id(user_id)
                    if not persona_id:
                        return None
                
                logger.info(
                    f"🎉 Conversation milestone reached: {current_milestone} messages for user {user_id}"
                )
                
                return await self._episode_service.generate_episode(
                    persona_id=persona_id,
                    user_id=user_id,
                    trigger_reason=f"conversation_milestone_{current_milestone}",
                )
            
            # Update count but no milestone
            self._last_milestone_count[user_id] = total_count
            return None
            
        except Exception as exc:
            logger.error(f"Error checking conversation milestone: {exc}", exc_info=True)
            return None

    async def check_daily_summary(
        self,
        user_id: str,
        persona_id: Optional[str] = None,
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Check if it's time for a daily summary episode.
        
        Args:
            user_id: User identifier
            persona_id: Optional persona identifier (will resolve if not provided)
            force: Force generation regardless of time check
            
        Returns:
            Generated episode dict if summary created, None otherwise
        """
        if not self._daily_summary_enabled and not force:
            return None
            
        try:
            now = datetime.now(timezone.utc)
            last_summary = self._last_daily_summary.get(user_id)
            
            # Check if 24 hours have passed since last summary
            if not force and last_summary:
                time_since_last = now - last_summary
                if time_since_last < timedelta(hours=24):
                    return None
            
            # Resolve persona if needed
            if not persona_id:
                persona_id = await self._resolve_persona_id(user_id)
                if not persona_id:
                    return None
            
            # Check if there's been any activity in the last 24 hours
            # Note: In SELO's architecture, user_id is actually the session_id
            messages = await self._conversation_repo.get_conversation_history(
                session_id=user_id,
                limit=50,
            )
            
            if not messages:
                logger.debug(f"Skipping daily summary for user {user_id} - no recent activity")
                return None
            
            # Check if latest message is recent
            # Note: messages are ordered oldest first, so get the last one
            if messages:
                latest = messages[-1]
                timestamp = latest.get("timestamp")
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        timestamp = None
                
                if timestamp:
                    age = now - timestamp
                    if age > timedelta(hours=24):
                        logger.debug(f"Skipping daily summary for user {user_id} - no recent activity")
                        return None
            
            logger.info(f"📅 Generating daily summary episode for user {user_id}")
            self._last_daily_summary[user_id] = now
            
            return await self._episode_service.generate_episode(
                persona_id=persona_id,
                user_id=user_id,
                trigger_reason="daily_summary",
            )
            
        except Exception as exc:
            logger.error(f"Error generating daily summary: {exc}", exc_info=True)
            return None

    async def handle_goal_completion(
        self,
        goal: Dict[str, Any],
        persona_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Generate a celebration episode when a goal is completed.
        
        Args:
            goal: Goal dictionary with completion information
            persona_id: Persona identifier
            user_id: User identifier
            
        Returns:
            Generated episode dict if celebration created, None otherwise
        """
        if not self._goal_celebration_enabled:
            return None
            
        try:
            goal_title = goal.get("title", "goal")
            goal_id = goal.get("id", "unknown")
            
            logger.info(
                f"🎊 Goal completed, generating celebration episode: '{goal_title}' "
                f"(goal_id={goal_id}, user={user_id})"
            )
            
            return await self._episode_service.generate_episode(
                persona_id=persona_id,
                user_id=user_id,
                trigger_reason=f"goal_completion:{goal_title}",
            )
            
        except Exception as exc:
            logger.error(f"Error generating goal completion episode: {exc}", exc_info=True)
            return None

    async def _resolve_persona_id(self, user_id: str) -> Optional[str]:
        """Resolve persona ID for a user."""
        try:
            persona = await self._persona_repo.get_persona_by_user(
                user_id=user_id,
                is_default=True,
                include_traits=False,
            )
            return getattr(persona, "id", None) if persona else None
        except Exception as exc:
            logger.debug(f"Unable to resolve persona for user {user_id}: {exc}")
            return None

    def update_config(self, config: Dict[str, Any]) -> None:
        """Update trigger configuration."""
        if "milestone_enabled" in config:
            self._milestone_enabled = bool(config["milestone_enabled"])
        if "milestone_interval" in config:
            self._milestone_interval = int(config["milestone_interval"])
        if "daily_summary_enabled" in config:
            self._daily_summary_enabled = bool(config["daily_summary_enabled"])
        if "goal_celebration_enabled" in config:
            self._goal_celebration_enabled = bool(config["goal_celebration_enabled"])
        
        logger.info(
            f"Episode triggers updated: milestones={self._milestone_enabled} "
            f"(every {self._milestone_interval}), daily={self._daily_summary_enabled}, "
            f"goals={self._goal_celebration_enabled}"
        )

    @property
    def config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "milestone_enabled": self._milestone_enabled,
            "milestone_interval": self._milestone_interval,
            "daily_summary_enabled": self._daily_summary_enabled,
            "goal_celebration_enabled": self._goal_celebration_enabled,
        }
