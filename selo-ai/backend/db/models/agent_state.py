"""
Agent state models for persistent affective state, self-driven goals, and autobiographical episodes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Float,
    JSON,
    Boolean,
    ForeignKey,
)

from ..base import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


class AffectiveState(Base):
    """Persistent affective state snapshot for a persona."""

    __tablename__ = "affective_states"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    persona_id = Column(String, ForeignKey("personas.id"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)

    mood_vector = Column(JSON, nullable=False, default=lambda: {"valence": 0.0, "arousal": 0.0})
    energy = Column(Float, nullable=False, default=0.5)
    stress = Column(Float, nullable=False, default=0.5)
    confidence = Column(Float, nullable=False, default=0.5)
    last_update = Column(DateTime(timezone=True), default=_utc_now)
    state_metadata = Column(JSON, nullable=False, default=lambda: {})
    homeostasis_active = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "persona_id": self.persona_id,
            "user_id": self.user_id,
            "mood_vector": self.mood_vector,
            "energy": self.energy,
            "stress": self.stress,
            "confidence": self.confidence,
            "last_update": _iso(self.last_update),
            "state_metadata": self.state_metadata,
            "homeostasis_active": self.homeostasis_active,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


class MetaReflectionDirective(Base):
    """Meta directives generated after comparing recent reflections."""

    __tablename__ = "meta_reflections"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    persona_id = Column(String, ForeignKey("personas.id"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)

    directive_text = Column(Text, nullable=False)
    priority = Column(Float, nullable=False, default=0.5)
    status = Column(String, nullable=False, default="pending")
    due_time = Column(DateTime(timezone=True), nullable=True)
    related_goal_id = Column(String, ForeignKey("agent_goals.id"), nullable=True)
    source_reflection_ids = Column(JSON, nullable=False, default=lambda: [])
    review_notes = Column(Text, nullable=True)
    extra_metadata = Column(JSON, nullable=False, default=lambda: {})

    created_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "persona_id": self.persona_id,
            "user_id": self.user_id,
            "directive_text": self.directive_text,
            "priority": self.priority,
            "status": self.status,
            "due_time": _iso(self.due_time),
            "related_goal_id": self.related_goal_id,
            "source_reflection_ids": self.source_reflection_ids,
            "review_notes": self.review_notes,
            "metadata": self.extra_metadata,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


class AgentGoal(Base):
    """Self-directed agent goal linked to persona intent."""

    __tablename__ = "agent_goals"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    persona_id = Column(String, ForeignKey("personas.id"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)

    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    origin = Column(String, nullable=False)  # reflection, directive, user_prompt, etc.
    status = Column(String, nullable=False, default="active")
    priority = Column(Float, nullable=False, default=0.5)
    deadline = Column(DateTime(timezone=True), nullable=True)
    progress = Column(Float, nullable=False, default=0.0)
    evidence_refs = Column(JSON, nullable=False, default=lambda: [])
    extra_metadata = Column(JSON, nullable=False, default=lambda: {})

    created_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "persona_id": self.persona_id,
            "user_id": self.user_id,
            "title": self.title,
            "description": self.description,
            "origin": self.origin,
            "status": self.status,
            "priority": self.priority,
            "deadline": _iso(self.deadline),
            "progress": self.progress,
            "evidence_refs": self.evidence_refs,
            "metadata": self.extra_metadata,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "completed_at": _iso(self.completed_at),
        }


class PlanStep(Base):
    """Structured plan step tied to a goal."""

    __tablename__ = "plan_steps"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    goal_id = Column(String, ForeignKey("agent_goals.id"), nullable=False, index=True)
    persona_id = Column(String, ForeignKey("personas.id"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)

    description = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="pending")
    priority = Column(Float, nullable=False, default=0.5)
    target_time = Column(DateTime(timezone=True), nullable=True)
    evidence_refs = Column(JSON, nullable=False, default=lambda: [])
    extra_metadata = Column(JSON, nullable=False, default=lambda: {})

    created_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "persona_id": self.persona_id,
            "user_id": self.user_id,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "target_time": _iso(self.target_time),
            "evidence_refs": self.evidence_refs,
            "metadata": self.extra_metadata,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "completed_at": _iso(self.completed_at),
        }


class AutobiographicalEpisode(Base):
    """Narrative episode built from reflections, memories, and conversations."""

    __tablename__ = "autobiographical_episodes"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    persona_id = Column(String, ForeignKey("personas.id"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)

    title = Column(String, nullable=False)
    narrative_text = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    importance = Column(Float, nullable=False, default=0.5)
    emotion_tags = Column(JSON, nullable=False, default=lambda: [])
    participants = Column(JSON, nullable=False, default=lambda: [])
    linked_memory_ids = Column(JSON, nullable=False, default=lambda: [])
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    extra_metadata = Column(JSON, nullable=False, default=lambda: {})

    created_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "persona_id": self.persona_id,
            "user_id": self.user_id,
            "title": self.title,
            "narrative_text": self.narrative_text,
            "summary": self.summary,
            "importance": self.importance,
            "emotion_tags": self.emotion_tags,
            "participants": self.participants,
            "linked_memory_ids": self.linked_memory_ids,
            "start_time": _iso(self.start_time),
            "end_time": _iso(self.end_time),
            "metadata": self.extra_metadata,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }
