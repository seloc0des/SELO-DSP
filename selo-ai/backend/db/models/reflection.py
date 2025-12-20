"""
SQLAlchemy models for the SELO AI reflection system.
These models map to the tables created in init_db.py.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, String, DateTime, ForeignKey, Float, LargeBinary, Integer, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from ..base import Base


class Reflection(Base):
    """
    SQLAlchemy model for reflection entries.
    Corresponds to the 'reflections' table in the database.
    """
    __tablename__ = "reflections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_profile_id = Column(String, nullable=False, index=True)
    reflection_type = Column(String, nullable=False, index=True)
    result = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    embedding = Column(LargeBinary, nullable=True)
    reflection_metadata = Column(JSONB, default={})

    # Relationships
    memories = relationship("ReflectionMemory", back_populates="reflection", cascade="all, delete-orphan")
    schedule_entries = relationship("ReflectionSchedule", back_populates="reflection")
    relationship_question_entries = relationship(
        "RelationshipQuestionQueue",
        back_populates="reflection",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        """String representation of the Reflection."""
        return f"<Reflection(id={self.id}, user_profile_id={self.user_profile_id}, type={self.reflection_type})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": str(self.id),
            "user_profile_id": self.user_profile_id,
            "reflection_type": self.reflection_type,
            "result": self.result,
            "created_at": self._iso_utc(self.created_at),
            "updated_at": self._iso_utc(self.updated_at),
            "metadata": self.reflection_metadata or {}
        }

    @staticmethod
    def _iso_utc(value: Optional[datetime]) -> Optional[str]:
        if not value:
            return None
        coerced = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return coerced.isoformat().replace("+00:00", "Z")


class ReflectionMemory(Base):
    """
    SQLAlchemy model for tracking which memories were used in each reflection.
    Corresponds to the 'reflection_memories' table in the database.
    """
    __tablename__ = "reflection_memories"

    reflection_id = Column(UUID(as_uuid=True), ForeignKey("reflections.id", ondelete="CASCADE"), primary_key=True)
    memory_id = Column(UUID(as_uuid=True), primary_key=True)
    relevance_score = Column(Float, nullable=True)

    # Relationship back to the parent reflection
    reflection = relationship("Reflection", back_populates="memories")

    def __repr__(self) -> str:
        """String representation of the ReflectionMemory."""
        return f"<ReflectionMemory(reflection_id={self.reflection_id}, memory_id={self.memory_id})>"


class ReflectionSchedule(Base):
    """
    SQLAlchemy model for scheduled reflections.
    Corresponds to the 'reflection_schedule' table in the database.
    """
    __tablename__ = "reflection_schedule"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_profile_id = Column(String, nullable=False, index=True)
    reflection_type = Column(String, nullable=False)
    scheduled_time = Column(DateTime(timezone=True), nullable=False)
    status = Column(String, default="pending", index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)
    reflection_id = Column(UUID(as_uuid=True), ForeignKey("reflections.id", ondelete="SET NULL"), nullable=True)

    # Relationship to the generated reflection (if completed)
    reflection = relationship("Reflection", back_populates="schedule_entries")

    def __repr__(self) -> str:
        """String representation of the ReflectionSchedule."""
        return f"<ReflectionSchedule(id={self.id}, user={self.user_profile_id}, type={self.reflection_type}, status={self.status})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": str(self.id),
            "user_profile_id": self.user_profile_id,
            "reflection_type": self.reflection_type,
            "scheduled_time": Reflection._iso_utc(self.scheduled_time),
            "status": self.status,
            "created_at": Reflection._iso_utc(self.created_at),
            "completed_at": Reflection._iso_utc(self.completed_at),
            "reflection_id": str(self.reflection_id) if self.reflection_id else None
        }


class RelationshipQuestionQueue(Base):
    """Pending relationship questions awaiting delivery."""

    __tablename__ = "relationship_question_queue"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_profile_id = Column(String, nullable=False, index=True)
    reflection_id = Column(UUID(as_uuid=True), ForeignKey("reflections.id", ondelete="CASCADE"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    topic = Column(String, nullable=False, index=True)
    priority = Column(Integer, nullable=False, default=3)
    suggested_delay_days = Column(Integer, nullable=False, default=7)
    prompt = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="pending", index=True)
    insight_value = Column(Text, nullable=True)
    existing_conflicts = Column(JSONB, default=list)
    queued_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    available_at = Column(DateTime(timezone=True), nullable=False)
    raw_payload = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    answered_at = Column(DateTime(timezone=True), nullable=True)
    answer_memory_id = Column(UUID(as_uuid=True), nullable=True)
    answer_tags = Column(JSONB, default=list)
    answer_importance_score = Column(Integer, nullable=True)
    answer_confidence_score = Column(Integer, nullable=True)

    reflection = relationship("Reflection", back_populates="relationship_question_entries")

    def __repr__(self) -> str:
        return (
            f"<RelationshipQuestionQueue(id={self.id}, user={self.user_profile_id}, "
            f"topic={self.topic}, status={self.status})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "user_profile_id": self.user_profile_id,
            "reflection_id": str(self.reflection_id),
            "question": self.question,
            "topic": self.topic,
            "priority": self.priority,
            "suggested_delay_days": self.suggested_delay_days,
            "prompt": self.prompt,
            "status": self.status,
            "insight_value": self.insight_value,
            "existing_conflicts": self.existing_conflicts or [],
            "queued_at": Reflection._iso_utc(self.queued_at),
            "available_at": Reflection._iso_utc(self.available_at),
            "raw_payload": self.raw_payload or {},
            "created_at": Reflection._iso_utc(self.created_at),
            "delivered_at": Reflection._iso_utc(self.delivered_at),
            "answered_at": Reflection._iso_utc(self.answered_at),
            "answer_memory_id": str(self.answer_memory_id) if self.answer_memory_id else None,
            "answer_tags": self.answer_tags or [],
            "answer_importance_score": self.answer_importance_score,
            "answer_confidence_score": self.answer_confidence_score,
        }
