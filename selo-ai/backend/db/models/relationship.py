"""
Relationship State Model

This module defines the database model for tracking the relationship state
between SELO and the user in a single-user application context.
"""

from datetime import datetime, timezone
import uuid
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON

from ..base import Base


def _iso_utc(value: Optional[datetime]) -> Optional[str]:
    if not value:
        return None
    coerced = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return coerced.isoformat().replace("+00:00", "Z")


class RelationshipState(Base):
    """
    Current state of the human-SELO relationship.
    
    Tracks intimacy, trust, and relationship progression in a single-user context.
    This enables SELO to understand and reference the depth and history of the relationship.
    """
    __tablename__ = "relationship_state"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    persona_id = Column(String, ForeignKey("personas.id"), nullable=False, unique=True, index=True)
    
    # Relationship metrics (0.0-1.0)
    intimacy_level = Column(Float, nullable=False, default=0.0)
    trust_level = Column(Float, nullable=False, default=0.5)
    comfort_level = Column(Float, nullable=False, default=0.3)
    
    # Relationship stage
    stage = Column(String, nullable=False, default="early")  # early, developing, established, deep, profound
    days_known = Column(Integer, nullable=False, default=0)
    conversations_count = Column(Integer, nullable=False, default=0)
    
    # Communication characteristics
    communication_style = Column(String, nullable=True)  # formal, casual, playful, deep
    shared_interests = Column(JSON, nullable=False, default=list)
    inside_jokes = Column(JSON, nullable=False, default=list)
    
    # Relationship milestones
    first_conversation = Column(DateTime(timezone=True), nullable=True)
    first_deep_conversation = Column(DateTime(timezone=True), nullable=True)
    first_vulnerability_moment = Column(DateTime(timezone=True), nullable=True)
    first_disagreement = Column(DateTime(timezone=True), nullable=True)
    first_inside_joke = Column(DateTime(timezone=True), nullable=True)
    
    # User information (learned through conversation)
    user_name = Column(String, nullable=True)  # Learned from conversation, not required
    user_preferences = Column(JSON, nullable=False, default=dict)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    last_updated = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    last_conversation_at = Column(DateTime(timezone=True), nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship state to dictionary representation."""
        return {
            "id": self.id,
            "persona_id": self.persona_id,
            "intimacy_level": self.intimacy_level,
            "trust_level": self.trust_level,
            "comfort_level": self.comfort_level,
            "stage": self.stage,
            "days_known": self.days_known,
            "conversations_count": self.conversations_count,
            "communication_style": self.communication_style,
            "shared_interests": self.shared_interests,
            "inside_jokes": self.inside_jokes,
            "first_conversation": _iso_utc(self.first_conversation),
            "first_deep_conversation": _iso_utc(self.first_deep_conversation),
            "first_vulnerability_moment": _iso_utc(self.first_vulnerability_moment),
            "first_disagreement": _iso_utc(self.first_disagreement),
            "first_inside_joke": _iso_utc(self.first_inside_joke),
            "user_name": self.user_name,
            "user_preferences": self.user_preferences,
            "created_at": _iso_utc(self.created_at),
            "last_updated": _iso_utc(self.last_updated),
            "last_conversation_at": _iso_utc(self.last_conversation_at)
        }
