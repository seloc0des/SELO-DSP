"""Planner service that produces structured plan steps from goals and directives."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from ..db.repositories.agent_state import (
    AgentGoalRepository,
    PlanStepRepository,
    MetaReflectionRepository,
)

logger = logging.getLogger("selo.agent.planner")


class PlannerService:
    """Produces actionable plan steps and reconciles meta directives."""

    def __init__(
        self,
        goal_repo: AgentGoalRepository,
        plan_repo: PlanStepRepository,
        meta_repo: MetaReflectionRepository,
    ) -> None:
        self._goal_repo = goal_repo
        self._plan_repo = plan_repo
        self._meta_repo = meta_repo

    async def generate_plan_steps(
        self,
        persona_id: str,
        affective_state: Dict[str, Any],
        time_now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Generate new steps based on active goals and affective signals."""
        time_now = time_now or datetime.now(timezone.utc)
        active_goals = await self._goal_repo.list_active_goals(persona_id)
        pending_steps = await self._plan_repo.list_pending_steps(persona_id)
        directives = await self._meta_repo.list_directives(persona_id, statuses=["pending", "in_progress"], limit=10)

        steps: List[Dict[str, Any]] = []
        if not active_goals:
            logger.debug("Planner found no active goals for persona %s", persona_id)
            return steps

        pending_by_goal = _group_by_goal(pending_steps)
        directives_by_goal = _group_by_goal(directives, key="related_goal_id")

        energy = float(affective_state.get("energy", 0.5) or 0.5)
        stress = float(affective_state.get("stress", 0.4) or 0.4)

        for goal in active_goals:
            goal_id = goal.get("id")
            existing_steps = pending_by_goal.get(goal_id, [])
            remapped_directives = directives_by_goal.get(goal_id, [])
            # Avoid generating duplicate reminders if there are already actionable steps
            if existing_steps:
                continue

            strategy = self._select_strategy(goal, remapped_directives, energy, stress)
            plan_step = self._build_step_from_strategy(goal, strategy, time_now, directives=remapped_directives)
            if not plan_step:
                continue

            # Persist step through repository before returning to caller
            try:
                persisted = await self._plan_repo.create_step(plan_step)
                steps.append(persisted)
            except Exception as exc:
                logger.error("Failed to persist plan step for goal %s: %s", goal_id, exc, exc_info=True)

        logger.info("Planner generated %d new steps for persona %s", len(steps), persona_id)
        return steps

    def _select_strategy(
        self,
        goal: Dict[str, Any],
        directives: Iterable[Dict[str, Any]],
        energy: float,
        stress: float,
    ) -> str:
        if any(d.get("priority", 0.5) >= 0.8 for d in directives):
            return "directive_followup"
        if stress >= 0.7:
            return "stabilize_before_action"
        if energy <= 0.3:
            return "light_touch_checkin"
        if "reflection" in (goal.get("origin") or ""):
            return "reflection_to_action"
        return "progress_review"

    def _build_step_from_strategy(
        self,
        goal: Dict[str, Any],
        strategy: str,
        time_now: datetime,
        directives: Iterable[Dict[str, Any]] = (),
    ) -> Optional[Dict[str, Any]]:
        description = None
        priority = float(goal.get("priority", 0.5) or 0.5)
        target_time: Optional[datetime] = None
        directive_ids = [d.get("id") for d in directives if isinstance(d, dict) and d.get("id")]
        metadata: Dict[str, Any] = {
            "strategy": strategy,
            "meta_directive_ids": directive_ids,
        }

        if strategy == "directive_followup":
            description = "Act on outstanding meta directive tied to this goal."
            priority = max(priority, 0.8)
            target_time = time_now + timedelta(hours=2)
        elif strategy == "stabilize_before_action":
            description = "Initiate mood stabilization reflection before pursuing deeper action."
            priority = max(priority, 0.7)
            target_time = time_now + timedelta(hours=4)
        elif strategy == "light_touch_checkin":
            description = "Draft a gentle outward check-in acknowledging limited energy today."
            priority = max(priority, 0.6)
            target_time = time_now + timedelta(days=1)
        elif strategy == "reflection_to_action":
            description = "Translate latest reflection insight into a concrete commitment for the user."
            priority = max(priority, 0.75)
            target_time = time_now + timedelta(hours=6)
        else:  # progress_review
            description = "Review recent memories to log measurable progress toward this goal."
            target_time = time_now + timedelta(days=2)

        if not description:
            return None

        return {
            "goal_id": goal["id"],
            "persona_id": goal["persona_id"],
            "user_id": goal["user_id"],
            "description": description,
            "priority": priority,
            "target_time": target_time,
            "metadata": metadata,
        }


def _group_by_goal(items: Iterable[Dict[str, Any]], key: str = "goal_id") -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in items or []:
        goal_id = item.get(key)
        if not goal_id:
            continue
        grouped.setdefault(goal_id, []).append(item)
    return grouped
