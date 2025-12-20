"""
Session-Based Episode Generation

Generates comprehensive autobiographical episodes after conversation sessions,
during natural idle periods. Episodes capture the full narrative arc of a
conversation session rather than fragmenting into per-reflection episodes.

Configuration:
- EPISODE_IDLE_THRESHOLD_MIN: Minutes of idle before episode (default: 15)
- EPISODE_MIN_REFLECTIONS: Minimum reflections needed (default: 1)
- EPISODE_SCAN_INTERVAL_SEC: Check interval (default: 180)
- EPISODE_SESSION_GAP_MIN: Gap to consider new session (default: 60)
"""

import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .event_triggers import EventTriggerSystem

logger = logging.getLogger("selo.scheduler.session_episodes")

# Constants for memory management
MAX_PROCESSED_REFLECTIONS_PER_USER = 500
PROCESSED_REFLECTION_EXPIRY_DAYS = 7


class SessionEpisodeGenerator:
    """
    Generates episodes for conversation sessions during idle periods.
    
    A "session" is a series of interactions with <60min gaps between messages.
    When a session has been idle for 15+ minutes, generate one episode covering
    the entire session.
    """
    
    def __init__(
        self,
        user_repo,
        persona_repo,
        reflection_repo,
        conversation_repo,
        episode_service,
        idle_threshold_minutes: int = 15,
        min_reflections: int = 1,
        session_gap_minutes: int = 60,
    ):
        self.user_repo = user_repo
        self.persona_repo = persona_repo
        self.reflection_repo = reflection_repo
        self.conversation_repo = conversation_repo
        self.episode_service = episode_service
        self.idle_threshold_minutes = idle_threshold_minutes
        self.min_reflections = min_reflections
        self.session_gap_minutes = session_gap_minutes
        
        # Track last processed session per user to avoid duplicates
        self._last_episode_time: Dict[str, datetime] = {}
        # Track reflection IDs that have been processed into episodes (with expiry)
        self._processed_reflection_ids: Dict[str, set] = {}
        self._processed_reflection_expiry: Dict[str, datetime] = {}  # reflection_id -> expiry_time
        
        # Event system reference for event-driven triggering
        self._event_system: Optional["EventTriggerSystem"] = None
        self._event_driven_enabled: bool = False
        
        # Resource-aware scheduling
        self._resource_monitor = None
        self._defer_on_high_load: bool = True
        self._deferred_generations: List[str] = []  # user_ids waiting for resources
        
        # Activity-aware scanning optimization
        self._last_activity_check: Optional[datetime] = None
        self._last_known_activity_time: Optional[datetime] = None
        self._extended_idle_threshold_hours: float = 1.0  # Skip scans after 1 hour of inactivity
        
        logger.info(
            f"Session episode generator initialized: "
            f"idle_threshold={idle_threshold_minutes}min, "
            f"min_reflections={min_reflections}, "
            f"session_gap={session_gap_minutes}min"
        )
    
    def bind_event_system(self, event_system: "EventTriggerSystem") -> None:
        """Bind event system for event-driven episode generation."""
        self._event_system = event_system
        self._event_driven_enabled = True
        logger.info("Session episode generator bound to event system (event-driven mode enabled)")
    
    def bind_resource_monitor(self, resource_monitor) -> None:
        """Bind resource monitor for resource-aware scheduling."""
        self._resource_monitor = resource_monitor
        logger.info("Session episode generator bound to resource monitor")
    
    async def on_conversation_end(self, event_data: Dict[str, Any], user_id: Optional[str]) -> None:
        """
        Event handler for conversation end events.
        Triggers episode generation check immediately instead of waiting for next poll.
        
        Args:
            event_data: Event data from the event system
            user_id: User ID from the event
        """
        if not user_id:
            user_id = event_data.get("user_id")
        
        if not user_id:
            logger.debug("Conversation end event received but no user_id")
            return
        
        logger.debug(f"Conversation end event received for user {user_id}")
        
        # Check resource availability before generating
        if self._defer_on_high_load and self._resource_monitor:
            if self._resource_monitor.is_resource_constrained():
                logger.info(f"Deferring episode generation for user {user_id} due to high resource usage")
                if user_id not in self._deferred_generations:
                    self._deferred_generations.append(user_id)
                return
        
        # Schedule episode check with small delay to allow final reflections to process
        asyncio.create_task(self._delayed_episode_check(user_id, delay_seconds=30))
    
    async def _delayed_episode_check(self, user_id: str, delay_seconds: int = 30) -> None:
        """Check for episode generation after a short delay."""
        try:
            await asyncio.sleep(delay_seconds)
            should_generate = await self._check_user_session(user_id)
            if should_generate:
                await self._generate_session_episode(user_id)
                logger.info(f"✅ Event-triggered episode generated for user {user_id}")
        except Exception as e:
            logger.error(f"Error in delayed episode check: {e}", exc_info=True)
    
    async def process_deferred_generations(self) -> int:
        """
        Process any deferred episode generations when resources become available.
        Called by the scheduler when resource usage drops.
        
        Returns:
            Number of episodes generated
        """
        if not self._deferred_generations:
            return 0
        
        # Check if resources are now available
        if self._resource_monitor and self._resource_monitor.is_resource_constrained():
            return 0
        
        generated = 0
        users_to_process = self._deferred_generations.copy()
        self._deferred_generations.clear()
        
        for user_id in users_to_process:
            try:
                should_generate = await self._check_user_session(user_id)
                if should_generate:
                    await self._generate_session_episode(user_id)
                    generated += 1
            except Exception as e:
                logger.error(f"Error processing deferred generation for {user_id}: {e}")
        
        if generated > 0:
            logger.info(f"✅ Processed {generated} deferred episode generation(s)")
        
        return generated
    
    def _cleanup_expired_reflections(self) -> None:
        """
        Remove expired reflection IDs to prevent unbounded memory growth.
        Called periodically during episode checks.
        """
        now = datetime.now(timezone.utc)
        expired_ids = [
            rid for rid, expiry in self._processed_reflection_expiry.items()
            if expiry < now
        ]
        
        if not expired_ids:
            return
        
        # Remove from expiry tracking
        for rid in expired_ids:
            del self._processed_reflection_expiry[rid]
        
        # Remove from user sets
        for user_id, id_set in self._processed_reflection_ids.items():
            id_set.difference_update(expired_ids)
        
        logger.debug(f"Cleaned up {len(expired_ids)} expired processed reflection IDs")
    
    async def check_and_generate_session_episodes(self) -> int:
        """
        Check default user for idle session and generate episode if needed.
        Also processes any deferred generations and cleans up expired data.
        
        Uses activity-aware scanning to skip unnecessary DB queries during
        extended idle periods (>1 hour of inactivity).
        
        Returns:
            Number of episodes generated
        """
        try:
            # Periodic cleanup to prevent memory growth
            self._cleanup_expired_reflections()
            
            # Process any deferred generations first
            deferred_count = await self.process_deferred_generations()
            
            # Check resource availability before proceeding
            if self._defer_on_high_load and self._resource_monitor:
                if self._resource_monitor.is_resource_constrained():
                    logger.debug("Skipping episode check due to high resource usage")
                    return deferred_count
            
            # Activity-aware optimization: skip full scan if no recent activity
            now = datetime.now(timezone.utc)
            if await self._should_skip_scan(now):
                logger.debug("Skipping episode scan - no recent activity detected")
                return deferred_count
            
            # SELO is single-user system - get default user
            user = await self.user_repo.get_or_create_default_user()
            user_id = user.id
            
            # Check if this user has an idle session ready for episode
            should_generate = await self._check_user_session(user_id)
            
            if should_generate:
                # Generate episode in background
                await self._generate_session_episode(user_id)
                logger.info(f"✅ Generated session episode for idle period")
                return deferred_count + 1
            
            return deferred_count
            
        except Exception as e:
            logger.error(f"Error in session episode generation: {e}", exc_info=True)
            return 0
    
    async def _check_user_session(self, user_id: str) -> bool:
        """
        Check if user has an idle session ready for episode generation.
        
        Returns:
            True if should generate episode, False otherwise
        """
        try:
            # Get user's last message time
            last_message_time = await self._get_last_message_time(user_id)
            if not last_message_time:
                return False
            
            now = datetime.now(timezone.utc)
            idle_minutes = (now - last_message_time).total_seconds() / 60
            
            # Check if idle threshold reached
            if idle_minutes < self.idle_threshold_minutes:
                return False
            
            # Check if we already generated episode for this session
            last_episode_time = self._last_episode_time.get(user_id)
            if last_episode_time:
                # If last episode was recent (within session gap), skip
                time_since_last_episode = (now - last_episode_time).total_seconds() / 60
                if time_since_last_episode < self.session_gap_minutes:
                    return False
            
            # Get session reflections (reflections since last episode or session start)
            session_start = self._calculate_session_start(last_message_time, last_episode_time)
            session_reflections = await self._get_session_reflections(user_id, session_start)
            
            # Filter out already-processed reflections
            processed_ids = self._processed_reflection_ids.get(user_id, set())
            new_reflections = [
                r for r in session_reflections 
                if r.get('id') not in processed_ids
            ]
            
            # Check minimum NEW reflection count
            if len(new_reflections) < self.min_reflections:
                logger.debug(
                    f"Session idle {idle_minutes:.1f}min but only "
                    f"{len(new_reflections)} NEW reflection(s) (need {self.min_reflections}, "
                    f"{len(session_reflections)} total, {len(processed_ids)} already processed)"
                )
                return False
            
            logger.info(
                f"🎬 Session ready for episode: "
                f"{idle_minutes:.1f}min idle, {len(session_reflections)} reflection(s)"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error checking session: {e}")
            return False
    
    def _calculate_session_start(
        self,
        last_message_time: datetime,
        last_episode_time: Optional[datetime]
    ) -> datetime:
        """
        Calculate when current session started.
        
        Session starts either:
        1. After last episode was generated, OR
        2. session_gap_minutes before last message (if no recent episode)
        """
        now = datetime.now(timezone.utc)
        
        # If we have a recent episode, session starts after it
        if last_episode_time:
            time_since_episode = (now - last_episode_time).total_seconds() / 60
            if time_since_episode < self.session_gap_minutes * 2:
                return last_episode_time
        
        # Otherwise, look back from last message
        # Session could span multiple messages with <60min gaps
        session_start = last_message_time - timedelta(minutes=self.session_gap_minutes * 2)
        return session_start
    
    async def _should_skip_scan(self, now: datetime) -> bool:
        """
        Determine if we should skip the episode scan based on activity patterns.
        
        Skips scans when:
        - No activity has been detected for over 1 hour
        - Last activity check was recent and showed no change
        
        Returns:
            True if scan should be skipped, False otherwise
        """
        try:
            # If we've never checked, don't skip
            if self._last_activity_check is None:
                return False
            
            # If last check was very recent (<2 min), skip redundant check
            time_since_last_check = (now - self._last_activity_check).total_seconds()
            if time_since_last_check < 120:  # 2 minutes
                # Only skip if we know there's been extended idle
                if self._last_known_activity_time:
                    hours_idle = (now - self._last_known_activity_time).total_seconds() / 3600
                    if hours_idle > self._extended_idle_threshold_hours:
                        return True
            
            # If last known activity was over 1 hour ago, skip the full scan
            # (but still update the activity check periodically)
            if self._last_known_activity_time:
                hours_idle = (now - self._last_known_activity_time).total_seconds() / 3600
                if hours_idle > self._extended_idle_threshold_hours:
                    # Only do a full scan every 30 minutes during extended idle
                    if time_since_last_check < 1800:  # 30 minutes
                        return True
                    
            return False
            
        except Exception as e:
            logger.debug(f"Error in activity skip check: {e}")
            return False
    
    async def _get_last_message_time(self, user_id: str) -> Optional[datetime]:
        """Get timestamp of user's last message and update activity tracking."""
        try:
            # Get most recent conversation
            conversations = await self.conversation_repo.list_conversations(
                user_id=user_id,
                limit=1,
                sort_by="updated_at",
                sort_order="desc"
            )
            
            if not conversations:
                return None
            
            conversation = conversations[0]
            
            # Get last message from conversation history
            history = await self.conversation_repo.get_conversation_history(
                session_id=conversation.get('session_id'),
                limit=1
            )
            
            if not history:
                return None
            
            last_message = history[0]
            timestamp = last_message.get('timestamp')
            
            if isinstance(timestamp, str):
                # Parse ISO timestamp
                from dateutil import parser
                timestamp = parser.parse(timestamp)
            
            if isinstance(timestamp, datetime):
                # Update activity tracking
                self._last_activity_check = datetime.now(timezone.utc)
                self._last_known_activity_time = timestamp
                return timestamp
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting last message time for user {user_id}: {e}")
            return None
    
    async def _get_session_reflections(
        self,
        user_id: str,
        session_start: datetime
    ) -> List[Dict[str, Any]]:
        """Get all reflections created during this session."""
        try:
            # Get reflections created after session start
            all_reflections = await self.reflection_repo.list_reflections(
                user_profile_id=user_id,
                limit=50  # Get recent reflections
            )
            
            if not all_reflections:
                return []
            
            # Filter to session timeframe
            session_reflections = []
            for reflection in all_reflections:
                created_at = reflection.get('created_at')
                
                if isinstance(created_at, str):
                    from dateutil import parser
                    created_at = parser.parse(created_at)
                
                if created_at and created_at >= session_start:
                    session_reflections.append(reflection)
            
            return session_reflections
            
        except Exception as e:
            logger.error(f"Error getting session reflections for user {user_id}: {e}")
            return []
    
    async def _generate_session_episode(self, user_id: str) -> None:
        """
        Generate episode for user's idle session (background task).
        """
        async def _generate_background():
            try:
                # Get persona
                persona = await self.persona_repo.get_persona_by_user(user_id)
                if not persona:
                    logger.warning(f"No persona found, skipping episode generation")
                    return
                
                persona_id = persona.id
                
                # Count session reflections for logging
                last_episode_time = self._last_episode_time.get(user_id)
                last_message_time = await self._get_last_message_time(user_id)
                if not last_message_time:
                    last_message_time = datetime.now(timezone.utc)
                session_start = self._calculate_session_start(
                    last_message_time,
                    last_episode_time
                )
                session_reflections = await self._get_session_reflections(user_id, session_start)
                
                # Filter out already-processed reflections
                if user_id not in self._processed_reflection_ids:
                    self._processed_reflection_ids[user_id] = set()
                processed_ids = self._processed_reflection_ids[user_id]
                new_reflections = [
                    r for r in session_reflections 
                    if r.get('id') not in processed_ids
                ]
                reflection_count = len(new_reflections)
                
                if reflection_count == 0:
                    logger.debug(f"No new reflections to process for episode (all {len(session_reflections)} already processed)")
                    return
                
                logger.info(
                    f"🎬 Starting session episode generation: "
                    f"{reflection_count} NEW reflection(s), persona {persona_id}"
                )
                
                # Generate episode
                result = await self.episode_service.generate_episode(
                    persona_id=persona_id,
                    user_id=user_id,
                    trigger_reason=f"session_idle_{reflection_count}_reflections",
                    plan_steps=None,
                )
                
                if result:
                    episode_id = result.get('id', 'unknown')
                    logger.info(
                        f"✅ Session episode {episode_id} generated "
                        f"({reflection_count} reflections)"
                    )
                    
                    # Mark reflections as processed with expiry
                    expiry_time = datetime.now(timezone.utc) + timedelta(days=PROCESSED_REFLECTION_EXPIRY_DAYS)
                    for reflection in new_reflections:
                        reflection_id = reflection.get('id')
                        if reflection_id:
                            processed_ids.add(reflection_id)
                            self._processed_reflection_expiry[reflection_id] = expiry_time
                    
                    # Enforce max size per user
                    if len(processed_ids) > MAX_PROCESSED_REFLECTIONS_PER_USER:
                        # Remove oldest entries (those closest to expiry)
                        sorted_ids = sorted(
                            processed_ids,
                            key=lambda rid: self._processed_reflection_expiry.get(rid, datetime.max.replace(tzinfo=timezone.utc))
                        )
                        to_remove = sorted_ids[:len(processed_ids) - MAX_PROCESSED_REFLECTIONS_PER_USER]
                        for rid in to_remove:
                            processed_ids.discard(rid)
                            self._processed_reflection_expiry.pop(rid, None)
                    
                    logger.debug(f"Marked {len(new_reflections)} reflection(s) as processed (total: {len(processed_ids)})")
                    
                    # Update last episode time
                    self._last_episode_time[user_id] = datetime.now(timezone.utc)
                else:
                    logger.warning(f"⚠️ Session episode generation returned None")
                
            except Exception as e:
                logger.error(
                    f"❌ Session episode generation failed: {e}",
                    exc_info=True
                )
        
        # Launch background task
        asyncio.create_task(_generate_background())


async def check_session_episodes(
    user_repo,
    persona_repo,
    reflection_repo,
    conversation_repo,
    episode_service,
    config: Optional[Dict[str, Any]] = None
) -> int:
    """
    Standalone function for scheduler integration.
    
    Args:
        config: Optional configuration override
            - idle_threshold_minutes: Minutes of idle (default: 15)
            - min_reflections: Minimum reflections (default: 1)
            - session_gap_minutes: Session boundary gap (default: 60)
    
    Returns:
        Number of episodes generated
    """
    config = config or {}
    
    generator = SessionEpisodeGenerator(
        user_repo=user_repo,
        persona_repo=persona_repo,
        reflection_repo=reflection_repo,
        conversation_repo=conversation_repo,
        episode_service=episode_service,
        idle_threshold_minutes=config.get('idle_threshold_minutes', 15),
        min_reflections=config.get('min_reflections', 1),
        session_gap_minutes=config.get('session_gap_minutes', 60),
    )
    
    return await generator.check_and_generate_session_episodes()
