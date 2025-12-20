"""
SQLAlchemy models for the SELO AI few-shot example system.
These models store and manage reflection examples for dynamic prompt engineering.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any
import json

from sqlalchemy import Column, String, DateTime, Boolean, Integer, Float, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB

from ..base import Base


class ReflectionExample(Base):
    """
    SQLAlchemy model for few-shot reflection examples.
    Stores examples used to teach the LLM correct reflection patterns.
    """
    __tablename__ = "reflection_examples"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Example classification
    category = Column(String(50), nullable=False, index=True)  # 'positive' or 'negative'
    scenario = Column(String(100), nullable=False, index=True)  # 'first_contact', 'emotional', 'technical', etc.
    
    # Context flags for selection
    requires_history = Column(Boolean, default=False, index=True)  # Does example assume prior context?
    is_emotional = Column(Boolean, default=False, index=True)
    is_technical = Column(Boolean, default=False, index=True)
    
    # Example content
    user_message = Column(Text, nullable=False)  # The user input in the example
    context_description = Column(Text, nullable=False)  # What context is available
    reflection_content = Column(Text, nullable=False)  # The 170-500 word reflection
    full_json = Column(JSONB, nullable=False)  # Complete JSON output
    explanation = Column(Text)  # For negative examples: why it's wrong
    
    # Performance tracking
    success_rate = Column(Float, default=0.0)  # Percentage of successful reflections when shown
    times_shown = Column(Integer, default=0)  # How many times used in prompts
    times_succeeded = Column(Integer, default=0)  # How many times led to valid reflection
    
    # Versioning and management
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True, index=True)  # Can be disabled without deleting
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Optional: tag examples for easier management
    tags = Column(JSONB, default=lambda: [])  # ['beginner', 'advanced', 'emotional', etc.]
    
    def __repr__(self) -> str:
        """String representation of the ReflectionExample."""
        return f"<ReflectionExample(id={self.id}, category={self.category}, scenario={self.scenario}, success_rate={self.success_rate:.2f})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": str(self.id),
            "category": self.category,
            "scenario": self.scenario,
            "requires_history": self.requires_history,
            "is_emotional": self.is_emotional,
            "is_technical": self.is_technical,
            "user_message": self.user_message,
            "context_description": self.context_description,
            "reflection_content": self.reflection_content,
            "full_json": self.full_json,
            "explanation": self.explanation,
            "success_rate": self.success_rate,
            "times_shown": self.times_shown,
            "times_succeeded": self.times_succeeded,
            "version": self.version,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "tags": self.tags or []
        }
    
    def _render_full_json(self) -> str:
        """Render full_json as valid JSON for prompt examples."""
        if isinstance(self.full_json, str):
            candidate = self.full_json.strip()
            if candidate:
                return candidate
        try:
            return json.dumps(self.full_json, ensure_ascii=False, indent=2, sort_keys=True)
        except (TypeError, ValueError):
            return json.dumps({"content": "Invalid example payload"}, ensure_ascii=False)

    def format_for_prompt(self) -> str:
        """Format example for inclusion in prompt."""
        rendered_json = self._render_full_json()
        if self.category == "positive":
            return (
                f"EXAMPLE — {self.scenario.replace('_', ' ').title()} (CORRECT):\n"
                f"User: \"{self.user_message}\"\n"
                f"Context: {self.context_description}\n\n"
                f"{rendered_json}\n"
            )
        return (
            f"EXAMPLE — Wrong: {self.scenario.replace('_', ' ').title()} (DO NOT DO THIS):\n"
            f"User: \"{self.user_message}\"\n"
            f"Context: {self.context_description}\n\n"
            f"{rendered_json}\n\n"
            f"WHY THIS IS WRONG:\n{self.explanation}\n"
        )
