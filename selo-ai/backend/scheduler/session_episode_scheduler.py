"""
Session Episode Scheduler Integration

Registers the session-based episode generation job with the scheduler system.
"""

import logging

logger = logging.getLogger("selo.scheduler.session_episodes")


async def setup_session_episode_scheduler(
    scheduler_service,
    session_episode_generator,
    scan_interval_seconds: int = 600  # Increased from 180 (3min) to 600 (10min) to reduce CPU load
) -> None:
    """
    Register session episode generation job with scheduler.
    
    Args:
        scheduler_service: SchedulerService instance
        session_episode_generator: SessionEpisodeGenerator instance
        scan_interval_seconds: How often to check for idle sessions (default: 600 = 10 min)
    """
    try:
        # Define the scheduled job function
        async def _run_session_episode_check():
            """Periodic check for idle sessions ready for episodes."""
            try:
                episodes_generated = await session_episode_generator.check_and_generate_session_episodes()
                if episodes_generated > 0:
                    logger.info(f"✅ Session episode cycle completed: {episodes_generated} episode(s) generated")
            except Exception as e:
                logger.error(f"Session episode check failed: {e}", exc_info=True)
        
        # Register with scheduler
        # add_job signature: add_job(job_id, func, trigger, **trigger_args)
        await scheduler_service.add_job(
            "session_episode_generator",  # job_id (positional)
            _run_session_episode_check,   # func (positional)
            "interval",                    # trigger (positional)
            seconds=scan_interval_seconds, # trigger_args (kwargs)
            name="Session Episode Generator",
            replace_existing=True,
            misfire_grace_time=60,
        )
        
        logger.info(
            f"✨ Session episode scheduler registered: checking every {scan_interval_seconds}s "
            f"(idle_threshold={session_episode_generator.idle_threshold_minutes}min, "
            f"min_reflections={session_episode_generator.min_reflections})"
        )
        
    except Exception as e:
        logger.error(f"Failed to setup session episode scheduler: {e}", exc_info=True)
        raise
