"""
Relationship Memory Model

This module defines the database model for relationship-specific memories
in a single-user context - shared moments, inside jokes, milestones.
"""

from datetime import datetime, timezone
import uuid
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Text, JSON

from ..base import Base


def _iso_utc(value: Optional[datetime]) -> Optional[str]:
    if not value:
        return None
    coerced = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return coerced.isoformat().replace("+00:00", "Z")


class RelationshipMemory(Base):
    """
    Memories specifically about the relationship itself.
    
    Tracks shared moments, inside jokes, conflicts, growth milestones, and
    other relationship-specific experiences that build intimacy over time.
    """
    __tablename__ = "relationship_memories"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    persona_id = Column(String, ForeignKey("personas.id"), nullable=False, index=True)
    
    # Memory classification
    memory_type = Column(String, nullable=False, index=True)
    # Types: shared_moment, inside_joke, conflict_resolution, growth_milestone, 
    #        vulnerability_shared, trust_building, disappointment, celebration
    
    # Emotional significance
    emotional_significance = Column(Float, nullable=False, default=0.5)  # 0.0-1.0
    emotional_tone = Column(String, nullable=True)  # positive, negative, mixed, neutral
    
    # Relationship impact
    intimacy_delta = Column(Float, nullable=False, default=0.0)  # How much this deepened bond
    trust_delta = Column(Float, nullable=False, default=0.0)  # How much this affected trust
    
    # Content
    narrative = Column(Text, nullable=False)  # SELO's perspective on this memory
    user_perspective = Column(Text, nullable=True)  # What user said/did (if captured)
    context = Column(Text, nullable=True)  # Additional context
    
    # Metadata
    conversation_id = Column(String, nullable=True, index=True)
    tags = Column(JSON, nullable=False, default=list)  # Searchable tags
    
    # Temporal info
    occurred_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Recall tracking
    recall_count = Column(Float, nullable=False, default=0)  # How often referenced
    last_recalled = Column(DateTime(timezone=True), nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship memory to dictionary representation."""
        return {
            "id": self.id,
            "persona_id": self.persona_id,
            "memory_type": self.memory_type,
            "emotional_significance": self.emotional_significance,
            "emotional_tone": self.emotional_tone,
            "intimacy_delta": self.intimacy_delta,
            "trust_delta": self.trust_delta,
            "narrative": self.narrative,
            "user_perspective": self.user_perspective,
            "context": self.context,
            "conversation_id": self.conversation_id,
            "tags": self.tags,
            "occurred_at": _iso_utc(self.occurred_at),
            "created_at": _iso_utc(self.created_at),
            "recall_count": self.recall_count,
            "last_recalled": _iso_utc(self.last_recalled)
        }


class AnticipatedEvent(Base):
    """
    Events that SELO is aware will happen in the future.
    
    Enables temporal awareness and follow-up behavior.
    """
    __tablename__ = "anticipated_events"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    persona_id = Column(String, ForeignKey("personas.id"), nullable=False, index=True)
    
    # Event details
    event_description = Column(Text, nullable=False)
    event_type = Column(String, nullable=True)  # interview, presentation, trip, deadline, etc.
    
    # Temporal info
    anticipated_date = Column(DateTime(timezone=True), nullable=True)  # When it's expected
    mentioned_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    conversation_id = Column(String, nullable=True)
    
    # Follow-up tracking
    followed_up = Column(Float, nullable=False, default=0.0)  # 0=no, 1=yes
    follow_up_at = Column(DateTime(timezone=True), nullable=True)
    outcome = Column(Text, nullable=True)  # What happened (filled in after follow-up)
    
    # Importance
    importance = Column(Float, nullable=False, default=0.5)  # 0.0-1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert anticipated event to dictionary representation."""
        return {
            "id": self.id,
            "persona_id": self.persona_id,
            "event_description": self.event_description,
            "event_type": self.event_type,
            "anticipated_date": _iso_utc(self.anticipated_date),
            "mentioned_at": _iso_utc(self.mentioned_at),
            "conversation_id": self.conversation_id,
            "followed_up": self.followed_up,
            "follow_up_at": _iso_utc(self.follow_up_at),
            "outcome": self.outcome,
            "importance": self.importance
        }
