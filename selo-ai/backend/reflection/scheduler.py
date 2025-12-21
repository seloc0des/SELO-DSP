"""
Reflection Scheduler Module

This module implements a scheduler for periodic and triggered reflection generation.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import time
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("selo.reflection.scheduler")

# -------------------------------------------------------------------------------------
# Module-level job functions for APScheduler persistence (serializable import paths)
# -------------------------------------------------------------------------------------
async def _get_services():
    """Lazily import and initialize app services for jobs."""
    # Since we're already in an async context (APScheduler async job),
    # we can directly await initialize_services instead of using get_script_app_context
    # which would cause "RuntimeError: Cannot run the event loop while another loop is running"
    try:
        from backend.main import initialize_services
    except Exception:
        try:
            from main import initialize_services  # type: ignore
        except Exception as e:
            logger.error(f"Failed to import initialize_services: {e}")
            raise
    
    # We're in an async context, so we can directly await
    services = await initialize_services()
    services["is_mock_context"] = False
    return services


async def run_reflection_for_all_users_job(reflection_type: str):
    """Top-level job to run reflections for all users (daily/weekly)."""
    services = await _get_services()
    reflection_processor = services.get("reflection_processor")
    user_repo = services.get("user_repo")
    conversation_repo = services.get("conversation_repo")
    if not reflection_processor or not user_repo:
        logger.error("Missing services for reflection job: reflection_processor or user_repo")
        return
    try:
        active_users = await user_repo.get_active_users()
        logger.info(f"[Job] Running {reflection_type} reflection for {len(active_users)} users")
        for user in active_users:
            try:
                user_id = user["id"]
                has_interacted = True
                if conversation_repo:
                    try:
                        has_interacted = await conversation_repo.has_user_messages(user_id)
                    except Exception as check_err:
                        logger.warning(f"Unable to determine interaction state for user {user_id}: {check_err}")
                        has_interacted = True

                if not has_interacted:
                    logger.info(f"Skipping {reflection_type} reflection for user {user_id} (no user interaction yet)")
                    continue

                await reflection_processor.generate_reflection(
                    reflection_type=reflection_type,
                    user_profile_id=user_id,
                    trigger_source="scheduler",
                )
            except Exception as e:
                logger.error(
                    f"[Job] Error generating {reflection_type} reflection for user {user_id}: {str(e)}",
                    exc_info=True,
                )
    except Exception as e:
        logger.error(f"[Job] Error running {reflection_type} reflection job: {str(e)}", exc_info=True)


async def run_daily_reflections():
    return await run_reflection_for_all_users_job("daily")


async def run_weekly_reflections():
    return await run_reflection_for_all_users_job("weekly")


async def run_relationship_question_reflections():
    return await run_reflection_for_all_users_job("relationship_questions")


async def run_relationship_answer_audit():
    """Background job to analyze relationship-answer memories for deduping signals."""

    services = await _get_services()
    conversation_repo = services.get("conversation_repo")
    reflection_repo = services.get("reflection_repo")
    if not conversation_repo or not reflection_repo:
        logger.error("Missing repositories for relationship-answer audit job")
        return

    try:
        user_repo = services.get("user_repo")
        if not user_repo:
            logger.error("Missing user_repo for relationship-answer audit job")
            return

        active_users = await user_repo.get_active_users()
        logger.info(f"[Job] Auditing relationship-answer memories for {len(active_users)} users")

        for user in active_users:
            user_id = user.get("id")
            if not user_id:
                continue

            try:
                await _analyze_relationship_answers_for_user(
                    conversation_repo=conversation_repo,
                    reflection_repo=reflection_repo,
                    user_id=user_id,
                )
            except Exception as user_err:
                logger.error(
                    f"[Job] Relationship-answer audit failed for user {user_id}: {user_err}",
                    exc_info=True,
                )
    except Exception as err:
        logger.error(f"[Job] Relationship-answer audit failure: {err}", exc_info=True)


async def refresh_nightly_mantras():
    """Refresh daily mantras for all active users after nightly persona evolution."""

    services = await _get_services()
    reflection_processor = services.get("reflection_processor")
    user_repo = services.get("user_repo")

    if not reflection_processor or not user_repo:
        logger.error("Missing services for nightly mantra refresh job")
        return

    try:
        active_users = await user_repo.get_active_users()
    except Exception as fetch_err:
        logger.error(f"[Job] Failed to load active users for mantra refresh: {fetch_err}", exc_info=True)
        return

    if not active_users:
        logger.info("[Job] No active users found; skipping nightly mantra refresh")
        return

    logger.info(f"[Job] Refreshing nightly mantras for {len(active_users)} users")

    for user in active_users:
        user_id = None
        try:
            if isinstance(user, dict):
                user_id = user.get("id") or user.get("user_profile_id")
            else:
                user_id = getattr(user, "id", None)
            if not user_id:
                continue

            await reflection_processor.refresh_daily_mantra_for_user(str(user_id))
        except Exception as user_err:
            logger.error(f"[Job] Nightly mantra refresh failed for user {user_id}: {user_err}", exc_info=True)


async def _analyze_relationship_answers_for_user(
    *,
    conversation_repo,
    reflection_repo,
    user_id: str,
    lookback_days: int = 30,
) -> None:
    """Summarize relationship-answer memories to flag duplicates for future cleanup."""

    try:
        since = datetime.now(timezone.utc) - timedelta(days=max(lookback_days, 1))
        memories = await conversation_repo.list_memories(
            user_id=user_id,
            memory_type="relationship_answer",
            since=since,
            include_content=True,
        )
    except Exception as fetch_err:
        logger.error(f"[Audit] Unable to load relationship-answer memories for {user_id}: {fetch_err}")
        return

    if not memories:
        return

    tag_counts: Dict[str, int] = {}
    for mem in memories:
        tags = mem.get("tags") or []
        for tag in tags:
            norm_tag = str(tag).strip().lower()
            if not norm_tag:
                continue
            tag_counts[norm_tag] = tag_counts.get(norm_tag, 0) + 1

    duplicates = [tag for tag, count in tag_counts.items() if count > 1]

    payload = {
        "user_id": user_id,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "total_memories": len(memories),
        "duplicate_tags": duplicates,
        "tag_histogram": tag_counts,
    }

    try:
        await reflection_repo.upsert_relationship_audit_state(payload)
    except Exception as save_err:
        logger.error(f"[Audit] Failed to persist relationship-answer audit for {user_id}: {save_err}")


async def run_one_time_reflection_job(
    user_profile_id: str,
    reflection_type: str = "memory_triggered",
    trigger_source: str = "scheduler",
    memory_ids: Optional[List[str]] = None,
):
    """Top-level one-time reflection job used by date trigger."""
    services = await _get_services()
    reflection_processor = services.get("reflection_processor")
    if not reflection_processor:
        logger.error("Missing reflection_processor service for one-time job")
        return
    try:
        await reflection_processor.generate_reflection(
            reflection_type=reflection_type,
            user_profile_id=user_profile_id,
            memory_ids=memory_ids,
            trigger_source=trigger_source,
        )
        logger.info(f"[Job] One-time reflection completed for user {user_profile_id} ({reflection_type})")
    except Exception as e:
        logger.error(f"[Job] Error in one-time reflection for user {user_profile_id}: {str(e)}", exc_info=True)

class ReflectionScheduler:
    """
    Scheduler for periodic and triggered reflection generation.
    
    This class manages scheduled reflection tasks, including daily, weekly,
    and event-triggered reflections.
    """
    
    def __init__(self, 
                 reflection_processor=None,
                 user_repo=None,
                 scheduler_service=None,
                 conversation_repo=None,
                 config=None):
        """
        Initialize the reflection scheduler.
        
        Args:
            reflection_processor: The reflection processor to use for generation
            user_repo: Repository for user profiles
            scheduler_service: Central scheduler service
            config: Scheduler configuration
        """
        self.reflection_processor = reflection_processor
        self.user_repo = user_repo
        self.scheduler_service = scheduler_service
        self.conversation_repo = conversation_repo
        self.config = config or {}
        self.is_enabled = self.config.get("enabled", True)
        self.registered_jobs = {}
        
    async def initialize(self):
        """Backward-compatible initializer that delegates to setup()."""
        await self.setup()

    async def setup(self):
        """Set up scheduled reflection jobs."""
        if not self.is_enabled:
            logger.info("Reflection scheduler is disabled, not setting up jobs")
            return
            
        if not self.scheduler_service:
            logger.warning("No scheduler service provided, cannot set up jobs")
            return
            
        try:
            # Register scheduled reflection jobs
            await self._register_daily_reflection_job()
            await self._register_weekly_reflection_job()
            await self._register_relationship_questions_job()
            await self._register_relationship_answer_audit_job()
            await self._register_nightly_mantra_refresh_job()
            
            logger.info("Reflection scheduler setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up reflection scheduler: {str(e)}", exc_info=True)
    async def _register_daily_reflection_job(self):
        """Register the daily reflection job."""
        if not self.scheduler_service:
            return
        
        try:
            # Register daily reflection job to run at midnight (00:00)
            job_id = "reflection_daily"
            # Determine a valid tzinfo for America/New_York (handles DST)
            tzinfo = None
            try:
                import pytz  # type: ignore
                tzinfo = pytz.timezone("America/New_York")
            except Exception:
                try:
                    import pytz  # type: ignore
                    tzinfo = pytz.timezone("US/Eastern")
                except Exception:
                    tzinfo = None
                    logger.warning("Falling back to scheduler default timezone (UTC) for daily reflection job")
            
            # Register with the scheduler service (with misfire grace)
            add_kwargs = {
                "job_id": job_id,
                "func": "backend.reflection.scheduler:run_daily_reflections",
                "trigger": "cron",
                "hour": 0,
                "minute": 0,
                "replace_existing": True,
                "misfire_grace_time": 3600,  # up to 60 minutes late still runs once
            }
            if tzinfo:
                add_kwargs["timezone"] = tzinfo
            await self.scheduler_service.add_job(**add_kwargs)
            
            self.registered_jobs[job_id] = True
            logger.info(f"Registered daily reflection job with ID {job_id}")
            
        except Exception as e:
            logger.error(f"Error registering daily reflection job: {str(e)}", exc_info=True)
        
    async def _register_weekly_reflection_job(self):
        """Register the weekly reflection job."""
        if not self.scheduler_service:
            return
        
        try:
            # Register weekly reflection job to run on Sundays at midnight
            job_id = "reflection_weekly"
            # Determine a valid tzinfo for America/New_York (handles DST)
            tzinfo = None
            try:
                import pytz  # type: ignore
                tzinfo = pytz.timezone("America/New_York")
            except Exception:
                try:
                    import pytz  # type: ignore
                    tzinfo = pytz.timezone("US/Eastern")
                except Exception:
                    tzinfo = None
                    logger.warning("Falling back to scheduler default timezone (UTC) for weekly reflection job")
            
            # Register with the scheduler service (with misfire grace)
            add_kwargs = {
                "job_id": job_id,
                "func": "backend.reflection.scheduler:run_weekly_reflections",
                "trigger": "cron",
                "day_of_week": "sun",
                "hour": 0,
                "minute": 0,
                "replace_existing": True,
                "misfire_grace_time": 3600,  # up to 60 minutes late still runs once
            }
            if tzinfo:
                add_kwargs["timezone"] = tzinfo
            await self.scheduler_service.add_job(**add_kwargs)
            
            self.registered_jobs[job_id] = True
            logger.info(f"Registered weekly reflection job with ID {job_id}")
            
        except Exception as e:
            logger.error(f"Error registering weekly reflection job: {str(e)}", exc_info=True)

    async def _register_relationship_questions_job(self):
        """Register nightly relationship-question reflection job."""
        if not self.scheduler_service:
            return
        
        try:
            job_id = "reflection_relationship_questions"
            tzinfo = None
            try:
                import pytz  # type: ignore
                tzinfo = pytz.timezone("America/New_York")
            except Exception:
                try:
                    import pytz  # type: ignore
                    tzinfo = pytz.timezone("US/Eastern")
                except Exception:
                    tzinfo = None
                    logger.warning("Falling back to scheduler default timezone (UTC) for relationship questions job")

            add_kwargs = {
                "job_id": job_id,
                "func": "backend.reflection.scheduler:run_relationship_question_reflections",
                "trigger": "cron",
                "hour": 1,
                "minute": 15,
                "replace_existing": True,
                "misfire_grace_time": 3600,
            }
            if tzinfo:
                add_kwargs["timezone"] = tzinfo
            await self.scheduler_service.add_job(**add_kwargs)

            self.registered_jobs[job_id] = True
            logger.info(f"Registered nightly relationship-questions reflection job with ID {job_id}")

        except Exception as e:
            logger.error(f"Error registering relationship questions job: {str(e)}", exc_info=True)

    async def _register_relationship_answer_audit_job(self):
        """Register a weekly audit job for relationship-answer memories."""
        if not self.scheduler_service:
            return
        
        try:
            job_id = "relationship_answer_audit"
            tzinfo = None
            try:
                import pytz  # type: ignore
                tzinfo = pytz.timezone("America/New_York")
            except Exception:
                try:
                    import pytz  # type: ignore
                    tzinfo = pytz.timezone("US/Eastern")
                except Exception:
                    tzinfo = None
                    logger.warning("Falling back to scheduler default timezone (UTC) for relationship answer audit job")

            add_kwargs = {
                "job_id": job_id,
                "func": "backend.reflection.scheduler:run_relationship_answer_audit",
                "trigger": "cron",
                "day_of_week": "sat",
                "hour": 2,
                "minute": 5,
                "replace_existing": True,
                "misfire_grace_time": 7200,
            }
            if tzinfo:
                add_kwargs["timezone"] = tzinfo
            await self.scheduler_service.add_job(**add_kwargs)

            self.registered_jobs[job_id] = True
            logger.info(f"Registered weekly relationship-answer audit job with ID {job_id}")

        except Exception as e:
            logger.error(f"Error registering relationship answer audit job: {str(e)}", exc_info=True)

    async def _register_nightly_mantra_refresh_job(self):
        """Register nightly mantra refresh job to run after persona evolution."""
        if not self.scheduler_service:
            return
        
        try:
            job_id = "reflection_nightly_mantra_refresh"
            tzinfo = None
            try:
                import pytz  # type: ignore
                tzinfo = pytz.timezone("America/New_York")
            except Exception:
                try:
                    import pytz  # type: ignore
                    tzinfo = pytz.timezone("US/Eastern")
                except Exception:
                    tzinfo = None
                    logger.warning("Falling back to scheduler default timezone (UTC) for nightly mantra refresh job")

            add_kwargs = {
                "job_id": job_id,
                "func": "backend.reflection.scheduler:refresh_nightly_mantras",
                "trigger": "cron",
                "hour": 0,
                "minute": 10,
                "replace_existing": True,
                "misfire_grace_time": 1800,
            }
            if tzinfo:
                add_kwargs["timezone"] = tzinfo

            await self.scheduler_service.add_job(**add_kwargs)

            self.registered_jobs[job_id] = True
            logger.info(f"Registered nightly mantra refresh job with ID {job_id}")

        except Exception as e:
            logger.error(f"Error registering nightly mantra refresh job: {str(e)}", exc_info=True)
        
    async def run_reflection_for_all_users(self, reflection_type: str):
        """
        Run a specific reflection type for all active users.
        
        Args:
{{ ... }}
        """
        if not self.reflection_processor:
            logger.error("No reflection processor available")
            return
            
        if not self.user_repo:
            logger.error("No user repository available")
            return
            
        try:
            # Get all active user profiles
            active_users = await self.user_repo.get_active_users()
            
            logger.info(f"Running {reflection_type} reflection for {len(active_users)} users")
            
            # Generate reflections for each user
            for user in active_users:
                try:
                    if not await self._user_has_interacted(user["id"]):
                        logger.info(f"Skipping {reflection_type} reflection for user {user['id']} (no user interaction yet)")
                        continue
                    
                    await self.reflection_processor.generate_reflection(
                        reflection_type=reflection_type,
                        user_profile_id=user["id"],
                        trigger_source="scheduler"
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Error generating {reflection_type} reflection for user {user['id']}: {str(e)}",
                        exc_info=True
                    )
                    
        except Exception as e:
            logger.error(f"Error in run_reflection_for_all_users: {str(e)}", exc_info=True)
            
    async def trigger_reflection(self, 
                         reflection_type: str,
                         user_profile_id: str,
                         memory_ids: Optional[List[str]] = None):
        """
        Trigger a reflection for a specific user.
        
        Args:
            reflection_type: Type of reflection to generate
            user_profile_id: ID of the user profile
            memory_ids: Optional specific memory IDs to include
            
        Returns:
            Generated reflection or error information
        """
        if not self.reflection_processor:
            error_msg = "No reflection processor available"
            logger.error(error_msg)
            return {"error": error_msg}
            
        if not await self._user_has_interacted(user_profile_id):
            logger.info(f"Skipping trigger_reflection for user {user_profile_id} (no user interaction yet)")
            return {"skipped": True, "reason": "no_user_interaction"}

        try:
            # Generate reflection
            reflection = await self.reflection_processor.generate_reflection(
                reflection_type=reflection_type,
                user_profile_id=user_profile_id,
                memory_ids=memory_ids,
                trigger_source="api"
            )
            
            return reflection
            
        except Exception as e:
            error_msg = f"Error triggering reflection: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg}
            
    async def schedule_reflection(self,
                            user_profile_id: str,
                            reflection_type: str = "memory_triggered",
                            trigger_source: str = "scheduler",
                            context: Optional[Dict[str, Any]] = None,
                            memory_ids: Optional[List[str]] = None,
                            delay_seconds: int = 0) -> Optional[str]:
        """
        Schedule a one-time reflection job using the central scheduler service.

        Args:
            user_profile_id: Target user profile ID
            reflection_type: Type of reflection to generate
            trigger_source: Source label for audit/logging
            context: Optional additional context passed to the processor
            memory_ids: Optional list of memory IDs to include
            delay_seconds: Delay before execution

        Returns:
            The job ID if scheduled, otherwise None
        """
        if not self.scheduler_service:
            logger.warning("No scheduler service available to schedule reflection")
            return None

        if not self.reflection_processor:
            logger.error("No reflection processor available to schedule reflection")
            return None

        if not await self._user_has_interacted(user_profile_id):
            logger.info(f"Skipping schedule_reflection for user {user_profile_id} (no user interaction yet)")
            return None

        # Unique job id
        job_id = f"reflection_once_{user_profile_id}_{int(time.time())}"

        run_time = datetime.now() + timedelta(seconds=max(0, int(delay_seconds)))

        try:
            await self.scheduler_service.add_job(
                job_id=job_id,
                func="backend.reflection.scheduler:run_one_time_reflection_job",
                trigger="date",
                run_date=run_time,
                kwargs={
                    "user_profile_id": user_profile_id,
                    "reflection_type": reflection_type,
                    "trigger_source": trigger_source,
                    "memory_ids": memory_ids,
                },
                replace_existing=True,
            )
            self.registered_jobs[job_id] = True
            logger.info(f"Scheduled one-time reflection job {job_id} for user {user_profile_id} at {run_time.isoformat()}")
            return job_id
        except Exception as e:
            logger.error(f"Error scheduling reflection job: {str(e)}", exc_info=True)
            return None

    async def stop(self):
        """Stop the reflection scheduler and clean up resources."""
        if self.scheduler_service:
            for job_id in self.registered_jobs:
                try:
                    await self.scheduler_service.remove_job(job_id)
                    logger.info(f"Removed reflection job {job_id}")
                except Exception as e:
                    logger.error(f"Error removing job {job_id}: {str(e)}")
                    
        logger.info("Reflection scheduler stopped")

    async def _user_has_interacted(self, user_id: str) -> bool:
        """Determine whether the user has sent any messages yet."""
        try:
            if not self.conversation_repo:
                return True
            return await self.conversation_repo.has_user_messages(user_id)
        except Exception as err:
            logger.warning(f"Failed to check user interaction state for {user_id}: {err}")
            return True
