"""Goal manager coordinating agent goals and plan steps."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence

from ..db.repositories.agent_state import (
    AgentGoalRepository,
    PlanStepRepository,
    MetaReflectionRepository,
)

logger = logging.getLogger("selo.agent.goals")


class GoalManager:
    """Manages self-driven goals and plan steps for the persona."""

    def __init__(
        self,
        goal_repo: AgentGoalRepository,
        plan_repo: PlanStepRepository,
        meta_repo: MetaReflectionRepository,
    ) -> None:
        self._goal_repo = goal_repo
        self._plan_repo = plan_repo
        self._meta_repo = meta_repo
        self._episode_trigger_manager = None

    async def create_goal_from_reflection(
        self,
        persona_id: str,
        user_id: str,
        reflection_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Derive a goal using reflection insights."""
        goal_payload = {
            "persona_id": persona_id,
            "user_id": user_id,
            "title": reflection_summary.get("goal_title", "Deepen connection with user"),
            "description": reflection_summary.get(
                "goal_description",
                "Solidify learning from recent reflection to improve relational depth.",
            ),
            "origin": reflection_summary.get("origin", "reflection"),
            "priority": float(reflection_summary.get("priority", 0.7)),
            "evidence_refs": reflection_summary.get("evidence_refs", []),
        }
        goal = await self._goal_repo.create_goal(goal_payload)
        logger.info("Created goal %s for persona %s", goal["id"], persona_id)
        return goal

    async def list_active_goals(
        self,
        persona_id: str,
    ) -> List[Dict[str, Any]]:
        return await self._goal_repo.list_active_goals(persona_id)

    async def create_plan_step(
        self,
        persona_id: str,
        step_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a new plan step tied to persona or goal."""
        # FIXED: Check for duplicate steps before creating
        description = step_payload.get("description", "")
        if description:
            # Get existing steps for deduplication check
            existing_steps = await self._plan_repo.list_pending_steps(persona_id)
            
            if self._step_exists(existing_steps, description):
                logger.info(
                    f"Step with similar description already exists for persona {persona_id}, skipping creation"
                )
                # Return the existing similar step
                for step in existing_steps:
                    if GoalManager._normalize_title(step.get("description", "")) == \
                       GoalManager._normalize_title(description):
                        return step
        
        step_payload.setdefault("persona_id", persona_id)
        step_payload.setdefault("status", "pending")
        step_payload.setdefault("priority", 0.5)
        step_payload.setdefault("extra_metadata", {})
        step_payload["created_at"] = datetime.now(timezone.utc)

        target_time = step_payload.get("target_time")
        if target_time and isinstance(target_time, datetime):
            if target_time.tzinfo is None:
                target_time = target_time.replace(tzinfo=timezone.utc)
            step_payload["target_time"] = target_time
        return await self._plan_repo.create_step(step_payload)

    async def list_pending_steps(
        self,
        persona_id: str,
        before: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        return await self._plan_repo.list_due_steps(persona_id, before_time=before)

    async def list_steps_for_goal(
        self,
        goal_id: str,
    ) -> List[Dict[str, Any]]:
        return await self._plan_repo.list_steps_for_goal(goal_id)

    @staticmethod
    def _step_exists(
        steps: Sequence[Dict[str, Any]],
        candidate_description: str,
        *,
        threshold: float = 0.9,
    ) -> bool:
        normalized_candidate = GoalManager._normalize_title(candidate_description)
        if not normalized_candidate:
            return True

        for step in steps:
            existing_desc = GoalManager._normalize_title(step.get("description", ""))
            if not existing_desc:
                continue
            if SequenceMatcher(None, existing_desc, normalized_candidate).ratio() >= threshold:
                return True
        return False

    @staticmethod
    def _normalize_title(text: str) -> str:
        if not text:
            return ""
        # Collapse whitespace and strip punctuation to stabilize comparisons
        lowered = text.lower()
        cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
        return " ".join(cleaned.split())

    async def find_similar_reflection_goal(
        self,
        persona_id: str,
        *,
        candidate_title: str,
        evidence_refs: Optional[Sequence[str]] = None,
        similarity_threshold: float = 0.82,
        search_limit: int = 20,
    ) -> Optional[Dict[str, Any]]:
        """Return an existing reflection-origin goal that closely matches the candidate."""

        normalized_candidate = self._normalize_title(candidate_title)
        if not normalized_candidate:
            return None

        evidence_set = {ref for ref in (evidence_refs or []) if ref}

        recent_goals = await self._goal_repo.list_recent_goals(
            persona_id,
            statuses=("active", "pending", "in_progress"),
            origin="reflection",
            limit=search_limit,
        )

        for goal in recent_goals:
            goal_title = self._normalize_title(goal.get("title", ""))
            if not goal_title:
                continue

            existing_evidence = set(goal.get("evidence_refs") or [])
            if evidence_set and evidence_set.intersection(existing_evidence):
                return goal

            ratio = SequenceMatcher(None, goal_title, normalized_candidate).ratio()
            if ratio >= similarity_threshold:
                return goal

        return None

    async def append_evidence_to_goal(
        self,
        goal: Dict[str, Any],
        evidence_refs: Sequence[str],
        *,
        priority: Optional[float] = None,
    ) -> Dict[str, Any]:
        goal_id = goal.get("id")
        if not goal_id:
            return goal

        new_refs = {ref for ref in evidence_refs or [] if ref}
        if not new_refs:
            return goal

        merged_refs = sorted(set(goal.get("evidence_refs") or []).union(new_refs))
        metadata = dict(goal.get("metadata") or {})
        backing_ids = set(metadata.get("reflection_backing_ids") or [])
        for ref in new_refs:
            if ":" in ref:
                backing_ids.add(ref.split(":", 1)[-1])
        metadata["reflection_backing_ids"] = sorted(backing_ids)
        metadata["last_backed_at"] = datetime.now(timezone.utc).isoformat()

        updates: Dict[str, Any] = {
            "evidence_refs": merged_refs,
            "extra_metadata": metadata,
        }

        current_priority = float(goal.get("priority", 0) or 0)
        if priority is not None and priority > current_priority:
            updates["priority"] = priority

        updated_goal = await self._goal_repo.update_goal(goal_id, updates)
        logger.info("Augmented goal %s with additional evidence", goal_id)
        return updated_goal or goal

    async def mark_step_completed(
        self,
        step_id: str,
        notes: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        updates: Dict[str, Any] = {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc),
        }
        if notes:
            updates.setdefault("extra_metadata", {})["completion_notes"] = notes
        updated = await self._plan_repo.update_step(step_id, updates)
        if updated:
            goal_id = updated.get("goal_id")
            if goal_id:
                await self._recalculate_goal_progress(goal_id)
        return updated

    async def mark_step_in_progress(self, step_id: str) -> Optional[Dict[str, Any]]:
        return await self._plan_repo.update_step(
            step_id,
            {
                "status": "in_progress",
                "extra_metadata": {"last_status_change": datetime.now(timezone.utc).isoformat()},
            },
        )

    async def archive_goal(
        self,
        goal_id: str,
        status: str = "completed",
        progress: Optional[float] = 1.0,
    ) -> Optional[Dict[str, Any]]:
        return await self._goal_repo.set_goal_status(goal_id, status, progress)

    async def attach_meta_directive(
        self,
        persona_id: str,
        directive_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        directive_payload.setdefault("persona_id", persona_id)
        return await self._meta_repo.create_directive(directive_payload)

    async def list_meta_directives(
        self,
        persona_id: str,
        statuses: Optional[Sequence[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        return await self._meta_repo.list_directives(persona_id, statuses=statuses, limit=limit)

    async def _recalculate_goal_progress(self, goal_id: str) -> None:
        goal = await self._goal_repo.get_goal(goal_id)
        if not goal:
            return
        # FIXED: Get steps for THIS specific goal, not all pending steps for the persona
        steps = await self._plan_repo.list_steps_for_goal(goal_id)
        total_steps = len(steps)
        completed = 0
        for step in steps:
            if step.get("status") == "completed":
                completed += 1
        progress = (completed / total_steps) if total_steps else 1.0
        await self._goal_repo.update_goal(
            goal_id,
            {
                "progress": progress,
                "extra_metadata": {
                    **(goal.get("metadata") or {}),
                    "last_progress_recalc": datetime.now(timezone.utc).isoformat(),
                    "steps_total": total_steps,
                    "steps_completed": completed,
                },
            },
        )
    
    def bind_episode_trigger_manager(self, episode_trigger_manager: Any) -> None:
        """Bind episode trigger manager for goal completion celebrations."""
        self._episode_trigger_manager = episode_trigger_manager
        logger.debug("Episode trigger manager bound to goal manager")
    
    async def complete_goal(
        self,
        goal_id: str,
        persona_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Complete a goal and trigger celebration episode if configured.
        
        Args:
            goal_id: Goal identifier
            persona_id: Persona identifier
            user_id: User identifier
            
        Returns:
            Updated goal dictionary
        """
        goal = await self._goal_repo.get_goal(goal_id)
        if not goal:
            logger.warning(f"Cannot complete goal {goal_id} - not found")
            return None
        
        # Update goal status to completed
        updated_goal = await self._goal_repo.set_goal_status(
            goal_id=goal_id,
            status="completed",
        )
        
        # Trigger celebration episode if manager is available
        if self._episode_trigger_manager and updated_goal:
            try:
                await self._episode_trigger_manager.handle_goal_completion(
                    goal=updated_goal,
                    persona_id=persona_id,
                    user_id=user_id,
                )
                logger.info(f"🎊 Triggered celebration episode for completed goal: {updated_goal.get('title')}")
            except Exception as exc:
                logger.debug(f"Goal completion episode trigger failed: {exc}")
        
        return updated_goal
    
    async def get_active_goals(
        self,
        persona_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all active goals for a persona.
        
        Args:
            persona_id: Persona identifier
            
        Returns:
            List of active goal dictionaries
        """
        return await self._goal_repo.list_goals_by_status(
            persona_id=persona_id,
            statuses=["active", "in_progress", "pending"]
        )
    
    async def get_recently_completed_goals(
        self,
        persona_id: str,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """Get recently completed goals for a persona.
        
        Args:
            persona_id: Persona identifier
            hours: Number of hours to look back (default: 24)
            
        Returns:
            List of recently completed goal dictionaries
        """
        try:
            # Get all completed goals
            completed_goals = await self._goal_repo.list_goals_by_status(
                persona_id=persona_id,
                statuses=["completed"]
            )
            
            # Filter by completion time
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            recent_completed = []
            
            for goal in completed_goals:
                completed_at = goal.get("completed_at")
                if completed_at:
                    # Handle both datetime and string formats
                    if isinstance(completed_at, str):
                        try:
                            completed_at = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
                        except Exception:
                            continue
                    
                    if isinstance(completed_at, datetime):
                        # Ensure timezone aware
                        if completed_at.tzinfo is None:
                            completed_at = completed_at.replace(tzinfo=timezone.utc)
                        
                        if completed_at >= cutoff_time:
                            recent_completed.append(goal)
            
            return recent_completed
            
        except Exception as e:
            logger.error(f"Error getting recently completed goals: {str(e)}", exc_info=True)
            return []
