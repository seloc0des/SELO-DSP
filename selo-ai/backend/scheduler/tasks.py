"""
Scheduler Task Functions

Module-level APScheduler job functions to allow SQLAlchemyJobStore serialization
via textual references (module:function). These functions reconstruct required
services using the centralized script helpers to avoid closures.
"""
from __future__ import annotations

import logging
from ..utils.datetime import isoformat_utc_now
from typing import Any, Dict

logger = logging.getLogger("selo.scheduler.tasks")


async def _get_services() -> Dict[str, Any]:
    """Lazily initialize app services using the main application's DI container."""
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


async def run_daily_persona_reassessment() -> None:
    """Emit a scheduler persona evolution event for the default installation persona."""
    try:
        services = await _get_services()
        event_trigger_system = services.get("event_trigger_system")
        user_repo = services.get("user_repo")
        persona_repo = services.get("persona_repo")

        if not (event_trigger_system and user_repo and persona_repo):
            logger.error("Missing services for persona reassessment job")
            return

        user = await user_repo.get_or_create_default_user()
        if not user:
            logger.warning("No default user found for persona reassessment")
            return

        persona = await persona_repo.get_persona_by_user(user_id=user.id, is_default=True)
        if not persona:
            logger.warning("No default persona found for persona reassessment")
            return

        event_data = {
            "persona_id": getattr(persona, "id", None),
            "user_id": getattr(user, "id", None),
            "trigger_type": "scheduled",
            "timestamp": isoformat_utc_now(),
        }
        await event_trigger_system.publish_event(
            event_type="scheduler.task.persona_evolution",
            event_data=event_data,
        )
        logger.info("[Job] Emitted daily persona reassessment event")
    except Exception as e:
        logger.error(f"[Job] Failed to emit daily persona reassessment: {e}", exc_info=True)
