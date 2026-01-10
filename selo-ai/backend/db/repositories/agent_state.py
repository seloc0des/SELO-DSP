"""
Agent state repositories for affective state, goals, plan steps, and autobiographical episodes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select, update, delete, asc, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.agent_state import (
    AffectiveState,
    AgentGoal,
    PlanStep,
    AutobiographicalEpisode,
    MetaReflectionDirective,
)
from ..session import get_session

logger = logging.getLogger("selo.db.agent_state")


class AffectiveStateRepository:
    """Persistence helper for persona affective state snapshots."""

    async def get_latest_state(
        self,
        persona_id: str,
        session: Optional[AsyncSession] = None,
    ) -> Optional[Dict[str, Any]]:
        async with get_session(session) as db:
            stmt = (
                select(AffectiveState)
                .where(AffectiveState.persona_id == persona_id)
                .order_by(AffectiveState.last_update.desc())
                .limit(1)
            )
            result = await db.execute(stmt)
            model = result.scalar_one_or_none()
            return model.to_dict() if model else None

    async def upsert_state(
        self,
        state_payload: Dict[str, Any],
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        async with get_session(session) as db:
            state_id = state_payload.get("id")
            if state_id:
                await db.execute(
                    update(AffectiveState)
                    .where(AffectiveState.id == state_id)
                    .values(**state_payload)
                )
                await db.commit()
                logger.debug("Updated affective state %s", state_id)
                result = await db.execute(
                    select(AffectiveState).where(AffectiveState.id == state_id)
                )
                model = result.scalar_one_or_none()
                return model.to_dict() if model else state_payload
            else:
                model = AffectiveState(**state_payload)
                db.add(model)
                await db.commit()
                await db.refresh(model)
                state_payload = model.to_dict()
                logger.debug("Created new affective state %s", state_payload.get("id"))
            return state_payload

    async def ensure_state(
        self,
        persona_id: str,
        user_id: str,
        baseline: Optional[Dict[str, Any]] = None,
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        async with get_session(session) as db:
            stmt = (
                select(AffectiveState)
                .where(AffectiveState.persona_id == persona_id)
                .order_by(desc(AffectiveState.last_update))
                .limit(1)
            )
            result = await db.execute(stmt)
            model = result.scalar_one_or_none()
            if model:
                return model.to_dict()
            payload = {
                "persona_id": persona_id,
                "user_id": user_id,
            }
            if baseline:
                mood_vector = None
                baseline_mood = baseline.get("mood_vector") if isinstance(baseline, dict) else None
                if isinstance(baseline_mood, dict):
                    mood_vector = {
                        "valence": float(baseline_mood.get("valence", 0.0) or 0.0),
                        "arousal": float(baseline_mood.get("arousal", 0.0) or 0.0),
                    }
                else:
                    empathy_val = float(baseline.get("empathy", 0.0) or 0.0)
                    arousal_val = float(baseline.get("energy", 0.0) or 0.0)
                    mood_vector = {"valence": empathy_val, "arousal": arousal_val}

                payload["mood_vector"] = mood_vector or {"valence": 0.0, "arousal": 0.0}
                payload["energy"] = float(baseline.get("energy", 0.5) or 0.5)
                payload["stress"] = float(baseline.get("stress", 0.4) or 0.4)
                payload["confidence"] = float(baseline.get("confidence", 0.6) or 0.6)

                metadata = baseline.get("state_metadata") if isinstance(baseline, dict) else {}
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata.setdefault("seeded_at", datetime.now(timezone.utc).isoformat())
                metadata.setdefault("source", "baseline_seed")
                payload["state_metadata"] = metadata

            payload.setdefault("mood_vector", {"valence": 0.0, "arousal": 0.0})
            payload.setdefault("energy", 0.5)
            payload.setdefault("stress", 0.4)
            payload.setdefault("confidence", 0.6)
            payload.setdefault(
                "state_metadata",
                {
                    "seeded_at": datetime.now(timezone.utc).isoformat(),
                    "source": "ensure_state_autoseed",
                },
            )
            created = await self.upsert_state(payload, session=db)
            return created


class AgentGoalRepository:
    """Repository for agent goals."""

    async def create_goal(
        self,
        goal_payload: Dict[str, Any],
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        async with get_session(session) as db:
            model = AgentGoal(**goal_payload)
            db.add(model)
            await db.commit()
            await db.refresh(model)
            logger.info("Created agent goal %s", model.id)
            return model.to_dict()

    async def list_active_goals(
        self,
        persona_id: str,
        session: Optional[AsyncSession] = None,
    ) -> List[Dict[str, Any]]:
        async with get_session(session) as db:
            stmt = select(AgentGoal).where(
                AgentGoal.persona_id == persona_id,
                AgentGoal.status == "active",
            ).order_by(AgentGoal.priority.desc())
            result = await db.execute(stmt)
            return [goal.to_dict() for goal in result.scalars().all()]

    async def get_goal(
        self,
        goal_id: str,
        session: Optional[AsyncSession] = None,
    ) -> Optional[Dict[str, Any]]:
        async with get_session(session) as db:
            result = await db.execute(select(AgentGoal).where(AgentGoal.id == goal_id))
            model = result.scalar_one_or_none()
            return model.to_dict() if model else None

    async def update_goal(
        self,
        goal_id: str,
        updates: Dict[str, Any],
        session: Optional[AsyncSession] = None,
    ) -> Optional[Dict[str, Any]]:
        async with get_session(session) as db:
            await db.execute(
                update(AgentGoal)
                .where(AgentGoal.id == goal_id)
                .values(**updates)
            )
            await db.commit()
            result = await db.execute(select(AgentGoal).where(AgentGoal.id == goal_id))
            model = result.scalar_one_or_none()
            if model:
                logger.debug("Updated goal %s", goal_id)
            return model.to_dict() if model else None

    async def set_goal_status(
        self,
        goal_id: str,
        status: str,
        progress: Optional[float] = None,
        session: Optional[AsyncSession] = None,
    ) -> Optional[Dict[str, Any]]:
        updates: Dict[str, Any] = {"status": status}
        if progress is not None:
            updates["progress"] = progress
            if status == "completed":
                updates.setdefault("completed_at", datetime.now(timezone.utc))
        return await self.update_goal(goal_id, updates, session=session)

    async def list_goals_by_status(
        self,
        persona_id: str,
        statuses: Sequence[str],
        session: Optional[AsyncSession] = None,
    ) -> List[Dict[str, Any]]:
        async with get_session(session) as db:
            stmt = (
                select(AgentGoal)
                .where(AgentGoal.persona_id == persona_id, AgentGoal.status.in_(list(statuses)))
                .order_by(desc(AgentGoal.priority))
            )
            result = await db.execute(stmt)
            return [goal.to_dict() for goal in result.scalars().all()]

    async def list_recent_goals(
        self,
        persona_id: str,
        *,
        statuses: Optional[Sequence[str]] = None,
        origin: Optional[str] = None,
        limit: int = 20,
        session: Optional[AsyncSession] = None,
    ) -> List[Dict[str, Any]]:
        async with get_session(session) as db:
            stmt = select(AgentGoal).where(AgentGoal.persona_id == persona_id)
            if statuses:
                stmt = stmt.where(AgentGoal.status.in_(list(statuses)))
            if origin:
                stmt = stmt.where(AgentGoal.origin == origin)
            stmt = stmt.order_by(desc(AgentGoal.created_at)).limit(max(1, limit))
            result = await db.execute(stmt)
            return [goal.to_dict() for goal in result.scalars().all()]


class PlanStepRepository:
    """Repository for structured plan steps."""

    async def create_step(
        self,
        step_payload: Dict[str, Any],
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        async with get_session(session) as db:
            model = PlanStep(**step_payload)
            db.add(model)
            await db.commit()
            await db.refresh(model)
            logger.debug("Created plan step %s", model.id)
            return model.to_dict()

    async def list_pending_steps(
        self,
        persona_id: str,
        session: Optional[AsyncSession] = None,
    ) -> List[Dict[str, Any]]:
        async with get_session(session) as db:
            stmt = select(PlanStep).where(
                PlanStep.persona_id == persona_id,
                PlanStep.status == "pending",
            ).order_by(PlanStep.priority.desc())
            result = await db.execute(stmt)
            return [step.to_dict() for step in result.scalars().all()]

    async def get_step(
        self,
        step_id: str,
        session: Optional[AsyncSession] = None,
    ) -> Optional[Dict[str, Any]]:
        async with get_session(session) as db:
            result = await db.execute(select(PlanStep).where(PlanStep.id == step_id))
            model = result.scalar_one_or_none()
            return model.to_dict() if model else None

    async def update_step(
        self,
        step_id: str,
        updates: Dict[str, Any],
        session: Optional[AsyncSession] = None,
    ) -> Optional[Dict[str, Any]]:
        async with get_session(session) as db:
            await db.execute(
                update(PlanStep)
                .where(PlanStep.id == step_id)
                .values(**updates)
            )
            await db.commit()
            result = await db.execute(select(PlanStep).where(PlanStep.id == step_id))
            model = result.scalar_one_or_none()
            return model.to_dict() if model else None

    async def list_due_steps(
        self,
        persona_id: str,
        before_time: Optional[datetime] = None,
        session: Optional[AsyncSession] = None,
    ) -> List[Dict[str, Any]]:
        async with get_session(session) as db:
            stmt = select(PlanStep).where(
                PlanStep.persona_id == persona_id,
                PlanStep.status == "pending",
            )
            if before_time:
                stmt = stmt.where(PlanStep.target_time <= before_time)
            stmt = stmt.order_by(
                PlanStep.target_time.asc().nulls_last(),
                desc(PlanStep.priority),
            )
            result = await db.execute(stmt)
            return [step.to_dict() for step in result.scalars().all()]

    async def list_steps_for_goal(
        self,
        goal_id: str,
        session: Optional[AsyncSession] = None,
    ) -> List[Dict[str, Any]]:
        async with get_session(session) as db:
            stmt = select(PlanStep).where(PlanStep.goal_id == goal_id)
            result = await db.execute(stmt)
            return [step.to_dict() for step in result.scalars().all()]


class AutobiographicalEpisodeRepository:
    """Repository for autobiographical episodes."""

    async def find_similar_recent_episodes(
        self,
        persona_id: str,
        narrative_text: str,
        within_hours: int = 24,
        session: Optional[AsyncSession] = None,
    ) -> List[Dict[str, Any]]:
        """Find recent episodes with similar narrative content for deduplication."""
        from datetime import datetime, timezone, timedelta
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=within_hours)
        
        async with get_session(session) as db:
            stmt = (
                select(AutobiographicalEpisode)
                .where(
                    AutobiographicalEpisode.persona_id == persona_id,
                    AutobiographicalEpisode.created_at >= cutoff_time
                )
                .order_by(AutobiographicalEpisode.created_at.desc())
            )
            result = await db.execute(stmt)
            episodes = [episode.to_dict() for episode in result.scalars().all()]
            
            # Simple text similarity check (can be enhanced with embedding similarity later)
            similar = []
            narrative_lower = narrative_text.lower().strip()
            narrative_words = set(narrative_lower.split())
            
            for episode in episodes:
                existing_narrative = (episode.get("narrative_text") or "").lower().strip()
                existing_words = set(existing_narrative.split())
                
                # Calculate Jaccard similarity
                if narrative_words and existing_words:
                    intersection = len(narrative_words & existing_words)
                    union = len(narrative_words | existing_words)
                    similarity = intersection / union if union > 0 else 0.0
                    
                    # Consider similar if >70% word overlap
                    if similarity > 0.7:
                        episode["similarity_score"] = similarity
                        similar.append(episode)
            
            return similar

    async def create_episode(
        self,
        episode_payload: Dict[str, Any],
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        async with get_session(session) as db:
            model = AutobiographicalEpisode(**episode_payload)
            db.add(model)
            await db.commit()
            await db.refresh(model)
            logger.info("Created autobiographical episode %s", model.id)
            return model.to_dict()

    async def list_recent_episodes(
        self,
        persona_id: str,
        limit: int = 5,
        session: Optional[AsyncSession] = None,
    ) -> List[Dict[str, Any]]:
        async with get_session(session) as db:
            stmt = (
                select(AutobiographicalEpisode)
                .where(AutobiographicalEpisode.persona_id == persona_id)
                .order_by(AutobiographicalEpisode.created_at.desc())
                .limit(limit)
            )
            result = await db.execute(stmt)
            return [episode.to_dict() for episode in result.scalars().all()]

    async def get_episode(
        self,
        episode_id: str,
        session: Optional[AsyncSession] = None,
    ) -> Optional[Dict[str, Any]]:
        async with get_session(session) as db:
            result = await db.execute(select(AutobiographicalEpisode).where(AutobiographicalEpisode.id == episode_id))
            model = result.scalar_one_or_none()
            return model.to_dict() if model else None


class MetaReflectionRepository:
    """Repository for meta reflection directives."""

    async def create_directive(
        self,
        payload: Dict[str, Any],
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        async with get_session(session) as db:
            model = MetaReflectionDirective(**payload)
            db.add(model)
            await db.commit()
            await db.refresh(model)
            logger.info("Created meta reflection directive %s", model.id)
            return model.to_dict()

    async def update_directive(
        self,
        directive_id: str,
        updates: Dict[str, Any],
        session: Optional[AsyncSession] = None,
    ) -> Optional[Dict[str, Any]]:
        async with get_session(session) as db:
            await db.execute(
                update(MetaReflectionDirective)
                .where(MetaReflectionDirective.id == directive_id)
                .values(**updates)
            )
            await db.commit()
            result = await db.execute(
                select(MetaReflectionDirective).where(MetaReflectionDirective.id == directive_id)
            )
            model = result.scalar_one_or_none()
            if model:
                logger.debug("Updated meta reflection directive %s", directive_id)
            return model.to_dict() if model else None

    async def list_directives(
        self,
        persona_id: str,
        statuses: Optional[Sequence[str]] = None,
        limit: int = 10,
        session: Optional[AsyncSession] = None,
    ) -> List[Dict[str, Any]]:
        async with get_session(session) as db:
            stmt = select(MetaReflectionDirective).where(
                MetaReflectionDirective.persona_id == persona_id
            )
            if statuses:
                stmt = stmt.where(MetaReflectionDirective.status.in_(list(statuses)))
            stmt = stmt.order_by(
                asc(MetaReflectionDirective.status),
                desc(MetaReflectionDirective.priority),
                MetaReflectionDirective.due_time.asc().nulls_last(),
            ).limit(limit)
            result = await db.execute(stmt)
            return [directive.to_dict() for directive in result.scalars().all()]

    async def delete_directive(
        self,
        directive_id: str,
        session: Optional[AsyncSession] = None,
    ) -> None:
        async with get_session(session) as db:
            await db.execute(
                delete(MetaReflectionDirective).where(MetaReflectionDirective.id == directive_id)
            )
            await db.commit()
            logger.debug("Deleted meta reflection directive %s", directive_id)
