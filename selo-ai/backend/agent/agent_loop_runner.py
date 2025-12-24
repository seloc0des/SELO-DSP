"""Agent loop runner coordinating Phase 1 continuous background behavior."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..agent.affective_state_manager import AffectiveStateManager
    from ..agent.goal_manager import GoalManager
    from ..agent.planner_service import PlannerService
    from ..agent.episode_builder import EpisodeBuilder
    from ..agent.autobiographical_episode_service import AutobiographicalEpisodeService
    from ..db.repositories.persona import PersonaRepository
    from ..db.repositories.user import UserRepository
    from ..scheduler.event_triggers import EventTriggerSystem


logger = logging.getLogger("selo.agent.loop_runner")


# Global reference for the agent loop runner instance (set by scheduler integration)
_agent_loop_runner_instance: Optional["AgentLoopRunner"] = None


def set_agent_loop_runner_instance(runner: "AgentLoopRunner") -> None:
    """
    Set the global agent loop runner instance for serializable scheduler jobs.
    
    This allows the scheduler to use a string reference to `run_agent_loop_job`
    which can be persisted to the database, while still accessing the configured
    runner instance.
    
    Args:
        runner: The configured AgentLoopRunner instance
    """
    global _agent_loop_runner_instance
    _agent_loop_runner_instance = runner


async def run_agent_loop_job() -> Dict[str, Any]:
    """
    Module-level function for scheduler serialization.
    
    This function can be referenced as a string ("backend.agent.agent_loop_runner:run_agent_loop_job")
    and persisted to the APScheduler job store. It delegates to the configured runner instance.
    
    Returns:
        Dict with execution summary
    """
    if _agent_loop_runner_instance is None:
        logger.warning("Agent loop job triggered but no runner instance configured")
        return {"skipped": True, "reason": "no_runner_instance"}
    
    try:
        return await _agent_loop_runner_instance.run(reason="scheduler_interval")
    except Exception as exc:
        logger.error("Agent loop runner job failed: %s", exc, exc_info=True)
        return {"error": str(exc), "skipped": True}


class AgentLoopRunner:
    """Coordinates the periodic agent loop for Phase 1 background systems."""

    def __init__(
        self,
        *,
        affective_state_manager: "AffectiveStateManager",
        goal_manager: "GoalManager",
        planner_service: "PlannerService",
        persona_repo: "PersonaRepository",
        user_repo: "UserRepository",
        episode_builder: Optional["EpisodeBuilder"] = None,
        episode_service: Optional["AutobiographicalEpisodeService"] = None,
        episode_trigger_manager: Optional[Any] = None,
        event_system: Optional["EventTriggerSystem"] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._config = self._build_config(config or {})

        self._affective_state_manager = affective_state_manager
        self._goal_manager = goal_manager
        self._planner_service = planner_service
        self._persona_repo = persona_repo
        self._user_repo = user_repo
        self._episode_builder = episode_builder
        self._episode_service = episode_service
        self._episode_trigger_manager = episode_trigger_manager
        self._event_system = event_system

        self._lock = asyncio.Lock()
        self._last_run_at: Optional[datetime] = None
        
        # Adaptive interval tracking
        self._activity_history: List[Dict[str, Any]] = []
        self._consecutive_idle_runs: int = 0
        self._consecutive_active_runs: int = 0
        self._current_interval: Optional[int] = None
        self._resource_monitor = None
        
        # Scheduler reference for dynamic rescheduling
        self._scheduler_service = None
        self._job_id: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self._config.get("enabled", True))

    @property
    def interval_seconds(self) -> int:
        """Get the current interval, which may be adaptively adjusted."""
        if self._current_interval is not None:
            return self._current_interval
        try:
            value = int(self._config.get("interval_seconds", 900))
        except (TypeError, ValueError):
            value = 900
        return max(60, value)
    
    @property
    def base_interval_seconds(self) -> int:
        """Get the configured base interval (before adaptive adjustments)."""
        try:
            value = int(self._config.get("interval_seconds", 900))
        except (TypeError, ValueError):
            value = 900
        return max(60, value)
    
    @property
    def adaptive_enabled(self) -> bool:
        """Whether adaptive interval adjustment is enabled."""
        return bool(self._config.get("adaptive_interval_enabled", True))

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._config)

    def bind_event_system(self, event_system: "EventTriggerSystem") -> None:
        self._event_system = event_system
    
    def bind_resource_monitor(self, resource_monitor) -> None:
        """Bind resource monitor for resource-aware scheduling."""
        self._resource_monitor = resource_monitor
    
    def bind_scheduler(self, scheduler_service, job_id: str) -> None:
        """
        Bind scheduler service for dynamic job rescheduling.
        
        Args:
            scheduler_service: SchedulerService instance
            job_id: The job ID used for the agent loop job
        """
        self._scheduler_service = scheduler_service
        self._job_id = job_id
        logger.info(f"Agent loop runner bound to scheduler (job_id={job_id})")

    def update_config(self, updates: Dict[str, Any]) -> None:
        self._config = self._build_config({**self._config, **(updates or {})})

    async def run(self, *, reason: str = "scheduled") -> Dict[str, Any]:
        """Execute a single agent loop tick.
        
        Defers execution if:
        - Agent loop is disabled
        - Resources are constrained (checked via resource monitor)
        - This ensures user interactions always take priority over background tasks
        """
        if not self.enabled:
            logger.debug("Agent loop tick skipped (disabled)")
            return {"skipped": True, "reason": "disabled"}
        
        # Check resource constraints - defer if system is under load
        # This ensures we don't compete with user's active tasks
        if self._resource_monitor:
            if self._resource_monitor.is_resource_constrained():
                logger.debug("Agent loop tick deferred - resources constrained (user task priority)")
                self._consecutive_idle_runs += 1  # Count as idle for adaptive scheduling
                return {"skipped": True, "reason": "resources_constrained"}
            
            # Also defer if resources are predicted to be exhausted soon
            if self._resource_monitor.should_defer_task(importance=0.4):
                logger.debug("Agent loop tick deferred - predicted resource constraint")
                return {"skipped": True, "reason": "predicted_constraint"}

        async with self._lock:
            return await self._execute(reason=reason)

    async def _execute(self, *, reason: str) -> Dict[str, Any]:
        started_at = datetime.now(timezone.utc)
        summary: Dict[str, Any] = {
            "reason": reason,
            "started_at": started_at.isoformat(),
        }

        try:
            context = await self._ensure_persona_context()
            if not context:
                summary["skipped"] = True
                summary["reason"] = "no_persona"
                return summary

            user_id = context["user_id"]
            persona_id = context["persona_id"]

            affective_state = await self._affective_state_manager.ensure_state_available(
                persona_id=persona_id,
                user_id=user_id,
            )

            if self._config.get("homeostasis_enabled", True):
                try:
                    await self._affective_state_manager.run_homeostasis_decay(persona_id)
                    summary["homeostasis_applied"] = True
                except Exception as exc:
                    logger.debug("Homeostasis decay failed for persona %s: %s", persona_id, exc)
                    summary["homeostasis_applied"] = False
            
            # Check for daily summary episode trigger
            if self._episode_trigger_manager:
                try:
                    daily_episode = await self._episode_trigger_manager.check_daily_summary(
                        user_id=user_id,
                        persona_id=persona_id,
                    )
                    summary["daily_summary_generated"] = daily_episode is not None
                except Exception as exc:
                    logger.debug("Daily summary check failed: %s", exc)
                    summary["daily_summary_generated"] = False

            active_goals = await self._goal_manager.list_active_goals(persona_id)
            pending_steps_before = await self._goal_manager.list_pending_steps(persona_id)
            directives = await self._goal_manager.list_meta_directives(
                persona_id,
                statuses=["pending", "in_progress"],
                limit=25,
            )

            generated_steps = await self._planner_service.generate_plan_steps(
                persona_id=persona_id,
                affective_state=affective_state or {},
            )

            pending_steps_after = await self._goal_manager.list_pending_steps(persona_id)

            summary.update(
                {
                    "active_goals": len(active_goals or []),
                    "pending_steps_before": len(pending_steps_before or []),
                    "pending_steps_after": len(pending_steps_after or []),
                    "meta_directives": len(directives or []),
                    "new_steps_generated": len(generated_steps or []),
                }
            )

            if generated_steps and self._episode_builder and self._config.get("episode_builder_enabled", True):
                await self._persist_episode(persona_id=persona_id, user_id=user_id, plan_steps=generated_steps)

            await self._emit_event(payload=summary)

            self._last_run_at = datetime.now(timezone.utc)
            summary["completed_at"] = self._last_run_at.isoformat()
            summary["duration_seconds"] = (self._last_run_at - started_at).total_seconds()
            
            # Track activity for adaptive scheduling
            is_active = summary.get("new_steps_generated", 0) > 0 or summary.get("active_goals", 0) > 0
            await self._update_activity_tracking(is_active, summary)
            
            # Add current interval to summary
            summary["current_interval_seconds"] = self.interval_seconds
            summary["adaptive_enabled"] = self.adaptive_enabled
            
            logger.info(
                "Agent loop tick complete (reason=%s, new_steps=%d, pending=%d, interval=%ds)",
                reason,
                summary["new_steps_generated"],
                summary["pending_steps_after"],
                self.interval_seconds,
            )
            return summary

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Agent loop tick failed: %s", exc, exc_info=True)
            summary["error"] = str(exc)
            return summary

    async def _ensure_persona_context(self) -> Optional[Dict[str, str]]:
        user = await self._user_repo.get_or_create_default_user()
        if not user:
            logger.warning("Agent loop runner could not resolve default user")
            return None

        persona = await self._persona_repo.get_or_create_default_persona(user_id=user.id, include_traits=True)
        if not persona:
            logger.warning("Agent loop runner could not resolve default persona")
            return None

        return {
            "user_id": getattr(user, "id", None),
            "persona_id": getattr(persona, "id", None),
        }

    async def _persist_episode(self, *, persona_id: str, user_id: str, plan_steps: List[Dict[str, Any]]) -> None:
        if not plan_steps:
            return
        try:
            if self._episode_service:
                await self._episode_service.generate_episode(
                    persona_id=persona_id,
                    user_id=user_id,
                    trigger_reason="agent_loop_plan_update",
                    plan_steps=plan_steps,
                )
                return

            if not self._episode_builder:
                logger.debug("Agent loop has no episode builder or service; skipping episode persistence.")
                return

            now = datetime.now(timezone.utc)
            lines = []
            for step in plan_steps:
                description = (step or {}).get("description") or "Agent action"
                priority = (step or {}).get("priority")
                if priority is not None:
                    lines.append(f"- {description} (priority {priority:.2f})")
                else:
                    lines.append(f"- {description}")

            narrative = "Agent loop generated new plan steps:\n" + "\n".join(lines)
            summary = f"Generated {len(plan_steps)} plan step(s) to advance active goals."

            artifacts: Dict[str, Any] = {
                "title": "Agent loop planning update",
                "narrative": narrative,
                "summary": summary,
                "emotion_tags": ["focused"],
                "participants": ["User", "SELO"],
                "linked_memory_ids": [],
                "importance": 0.55,
                "source": "agent_loop_runner",
                "artifacts_used": [step.get("id") for step in plan_steps if isinstance(step, dict) and step.get("id")],
                "start_time": now,
                "end_time": now,
            }
            await self._episode_builder.build_episode(
                persona_id=persona_id,
                user_id=user_id,
                artifacts=artifacts,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Episode builder failed for agent loop tick: %s", exc, exc_info=True)

    async def _emit_event(self, *, payload: Dict[str, Any]) -> None:
        if not self._event_system or not self._config.get("audit_events_enabled", True):
            return
        try:
            await self._event_system.publish_event(
                event_type="agent.loop.tick",
                event_data=payload,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Agent loop event emission failed: %s", exc, exc_info=True)

    async def _update_activity_tracking(self, is_active: bool, summary: Dict[str, Any]) -> None:
        """
        Update activity tracking and adjust interval if adaptive scheduling is enabled.
        
        Args:
            is_active: Whether this run was active (generated steps or had active goals)
            summary: Run summary for metrics
        """
        # Update consecutive counters
        if is_active:
            self._consecutive_active_runs += 1
            self._consecutive_idle_runs = 0
        else:
            self._consecutive_idle_runs += 1
            self._consecutive_active_runs = 0
        
        # Track activity history (keep last 20 runs)
        self._activity_history.append({
            "timestamp": datetime.now(timezone.utc),
            "is_active": is_active,
            "new_steps": summary.get("new_steps_generated", 0),
            "active_goals": summary.get("active_goals", 0),
        })
        if len(self._activity_history) > 20:
            self._activity_history = self._activity_history[-20:]
        
        # Adjust interval if adaptive scheduling is enabled
        if not self.adaptive_enabled:
            return
        
        # Adaptive scheduling constants - extracted for easier tuning
        INTERVAL_MIN_FLOOR = 60           # Absolute minimum interval in seconds
        INTERVAL_MIN_RATIO = 0.25         # Minimum as ratio of base (25%)
        INTERVAL_MAX_RATIO = 4.0          # Maximum as ratio of base (4x)
        
        base_interval = self.base_interval_seconds
        min_interval = max(INTERVAL_MIN_FLOOR, int(base_interval * INTERVAL_MIN_RATIO))
        max_interval = int(base_interval * INTERVAL_MAX_RATIO)  # Maximum: 4x base (up to 2 hours)
        
        # Calculate activity rate from recent history
        if len(self._activity_history) >= 3:
            recent = self._activity_history[-10:]
            activity_rate = sum(1 for r in recent if r["is_active"]) / len(recent)
        else:
            activity_rate = 0.5  # Default to neutral
        
        # Adaptive scheduling thresholds and multipliers - extracted for easier tuning
        EXTENDED_IDLE_THRESHOLD = 8      # Consecutive idle runs for extended idle
        VERY_IDLE_THRESHOLD = 5          # Consecutive idle runs for very idle
        MODERATE_IDLE_THRESHOLD = 3      # Consecutive idle runs for moderate idle
        ACTIVE_THRESHOLD = 3             # Consecutive active runs for active state
        ACTIVITY_HIGH_THRESHOLD = 0.6    # Activity rate above this = generally active
        ACTIVITY_LOW_THRESHOLD = 0.3     # Activity rate below this = generally idle
        
        STRETCH_MAJOR = 2.0              # Interval multiplier for very idle (100% increase)
        STRETCH_MODERATE = 1.5           # Interval multiplier for moderate idle (50% increase)
        SHRINK_ACTIVE = 0.7              # Interval multiplier for active (30% decrease)
        SHRINK_GENERAL = 0.75            # Interval multiplier for generally active
        STRETCH_GENERAL = 1.5            # Interval multiplier for generally idle
        RESOURCE_CONSTRAINT_MULTIPLIER = 1.5  # Multiplier when under resource load
        
        # Adjust interval based on activity
        if self._consecutive_idle_runs >= EXTENDED_IDLE_THRESHOLD:
            # Extended idle (2+ hours) - use maximum interval
            new_interval = max_interval
            logger.debug(f"Extended idle detected ({self._consecutive_idle_runs} runs), using max interval")
            
        elif self._consecutive_idle_runs >= VERY_IDLE_THRESHOLD:
            # Very idle (1+ hour) - stretch interval significantly
            current = self._current_interval or base_interval
            new_interval = int(current * STRETCH_MAJOR)
            new_interval = min(new_interval, max_interval)  # Cap at maximum
            
        elif self._consecutive_idle_runs >= MODERATE_IDLE_THRESHOLD:
            # Moderately idle (45+ min) - stretch interval moderately
            current = self._current_interval or base_interval
            new_interval = int(current * STRETCH_MODERATE)
            new_interval = min(new_interval, max_interval)  # Cap at maximum
            
        elif self._consecutive_active_runs >= ACTIVE_THRESHOLD:
            # Very active - shorten interval
            current = self._current_interval or base_interval
            new_interval = int(current * SHRINK_ACTIVE)
            new_interval = max(new_interval, min_interval)  # Floor at minimum
            
        elif activity_rate > ACTIVITY_HIGH_THRESHOLD:
            # Generally active - use shorter interval
            new_interval = int(base_interval * SHRINK_GENERAL)
            
        elif activity_rate < ACTIVITY_LOW_THRESHOLD:
            # Generally idle - use longer interval
            new_interval = int(base_interval * STRETCH_GENERAL)
            
        else:
            # Normal activity - use base interval
            new_interval = base_interval
        
        # Check resource constraints
        if self._resource_monitor and self._resource_monitor.is_resource_constrained():
            # Under load - use longer interval
            new_interval = max(new_interval, int(base_interval * RESOURCE_CONSTRAINT_MULTIPLIER))
            logger.debug("Agent loop extending interval due to resource constraints")
        
        # Apply new interval if significantly different
        old_interval = self._current_interval or base_interval
        if abs(new_interval - old_interval) > (base_interval * 0.1):
            self._current_interval = new_interval
            logger.info(
                f"Agent loop interval adjusted: {old_interval}s -> {new_interval}s "
                f"(activity_rate={activity_rate:.2f}, idle={self._consecutive_idle_runs}, active={self._consecutive_active_runs})"
            )
            
            # Reschedule the job with the new interval
            await self._reschedule_job(new_interval)
    
    async def _reschedule_job(self, new_interval: int) -> None:
        """
        Reschedule the agent loop job with a new interval.
        
        Implements robust error handling with fallback to base interval if rescheduling fails.
        This ensures the agent loop continues running even if dynamic rescheduling encounters issues.
        
        Args:
            new_interval: New interval in seconds
        """
        if not self._scheduler_service or not self._job_id:
            logger.debug("Cannot reschedule - no scheduler service or job_id bound")
            return
        
        try:
            # Use reschedule_job if available, otherwise remove and re-add
            if hasattr(self._scheduler_service, 'reschedule_job'):
                await self._scheduler_service.reschedule_job(
                    self._job_id,
                    trigger="interval",
                    seconds=new_interval
                )
            else:
                # Fallback: remove and re-add the job
                try:
                    await self._scheduler_service.remove_job(self._job_id)
                except Exception as remove_err:
                    logger.debug(f"Job removal failed (may not exist): {remove_err}")
                
                await self._scheduler_service.add_job(
                    job_id=self._job_id,
                    func="backend.agent.agent_loop_runner:run_agent_loop_job",
                    trigger="interval",
                    seconds=new_interval,
                )
            
            logger.info(f"Agent loop job rescheduled with interval: {new_interval}s")
            
        except AttributeError as e:
            # Scheduler service missing required methods
            logger.error(
                f"Scheduler service missing required methods for rescheduling: {e}. "
                f"Agent loop will continue with current interval.",
                exc_info=True
            )
            # Reset to base interval to prevent drift
            self._current_interval = self.base_interval_seconds
            
        except asyncio.TimeoutError:
            # Scheduler operation timed out
            logger.error(
                f"Timeout while rescheduling agent loop job. "
                f"Falling back to base interval: {self.base_interval_seconds}s"
            )
            # Reset to base interval as safety measure
            self._current_interval = self.base_interval_seconds
            
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(
                f"Failed to reschedule agent loop job: {e}. "
                f"Falling back to base interval: {self.base_interval_seconds}s",
                exc_info=True
            )
            # Reset to base interval to ensure job continues
            self._current_interval = self.base_interval_seconds
            
            # Attempt to verify job is still scheduled
            try:
                if hasattr(self._scheduler_service, 'get_job'):
                    job = await self._scheduler_service.get_job(self._job_id)
                    if job is None:
                        logger.critical(
                            f"Agent loop job {self._job_id} is no longer scheduled! "
                            f"Manual intervention may be required."
                        )
            except Exception as verify_err:
                logger.debug(f"Could not verify job status: {verify_err}")
    
    def get_adaptive_interval(self) -> int:
        """
        Get the current adaptive interval for external schedulers.
        
        Returns:
            Current interval in seconds
        """
        return self.interval_seconds
    
    def reset_adaptive_interval(self) -> None:
        """Reset adaptive interval to base configuration."""
        self._current_interval = None
        self._consecutive_idle_runs = 0
        self._consecutive_active_runs = 0
        logger.info(f"Agent loop interval reset to base: {self.base_interval_seconds}s")

    @staticmethod
    def _build_config(raw: Dict[str, Any]) -> Dict[str, Any]:
        config = {
            "enabled": bool(str(raw.get("enabled", True)).lower() in {"1", "true", "yes"}),
            "interval_seconds": raw.get("interval_seconds", 1800),  # Increased from 900 (15min) to 1800 (30min) to reduce CPU load
            "homeostasis_enabled": bool(str(raw.get("homeostasis_enabled", True)).lower() in {"1", "true", "yes"}),
            "episode_builder_enabled": bool(str(raw.get("episode_builder_enabled", True)).lower() in {"1", "true", "yes"}),
            "audit_events_enabled": bool(str(raw.get("audit_events_enabled", True)).lower() in {"1", "true", "yes"}),
            "adaptive_interval_enabled": bool(str(raw.get("adaptive_interval_enabled", True)).lower() in {"1", "true", "yes"}),
        }
        try:
            config["interval_seconds"] = int(config["interval_seconds"])
        except (TypeError, ValueError):
            config["interval_seconds"] = 1800  # Increased from 900 to 1800
        config["interval_seconds"] = max(60, config["interval_seconds"])
        return config
