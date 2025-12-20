"""
Persona Database Models

This module defines the database models for the Dynamic Persona System,
including persona core attributes, persona evolution history, and trait models.
"""

from datetime import datetime, timezone
import uuid
from typing import Dict, Any, List, Optional

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, 
    DateTime, ForeignKey, JSON, Text, Table
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.asyncio import AsyncSession

from ..base import Base


def _iso_utc(value: Optional[datetime]) -> Optional[str]:
    if not value:
        return None
    coerced = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return coerced.isoformat().replace("+00:00", "Z")


class Persona(Base):
    """
    Persona model representing the core attributes of a SELO AI persona.
    
    A persona contains fundamental personality traits, preferences, knowledge domains,
    communication style, and other attributes that define the AI's identity.
    """
    __tablename__ = "personas"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    mantra = Column(Text, nullable=True)
    first_thoughts = Column(Text, nullable=True)
    
    # Core persona attributes stored as JSON
    personality = Column(JSON, nullable=False, default=lambda: {})
    communication_style = Column(JSON, nullable=False, default=lambda: {})
    expertise = Column(JSON, nullable=False, default=lambda: {"domains": [], "skills": [], "knowledge_depth": 0.5})
    preferences = Column(JSON, nullable=False, default=lambda: {})
    goals = Column(JSON, nullable=False, default=lambda: {})
    values = Column(JSON, nullable=False, default=lambda: {})
    
    # Boot directive that created this persona
    boot_directive = Column(Text, nullable=True)
    
    # Persona state
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    evolution_locked = Column(Boolean, default=False)
    
    # Metadata
    creation_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_modified = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_evolution = Column(DateTime(timezone=True), nullable=True)
    evolution_count = Column(Integer, default=0)
    stability_score = Column(Float, default=1.0)  # 0.0-1.0, higher means more stable
    
    # Relationships
    evolutions = relationship("PersonaEvolution", back_populates="persona", 
                             cascade="all, delete-orphan")
    traits = relationship("PersonaTrait", back_populates="persona",
                          cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary representation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "mantra": self.mantra,
            "first_thoughts": self.first_thoughts,
            "boot_directive": self.boot_directive,
            "personality": self.personality,
            "communication_style": self.communication_style,
            "expertise": self.expertise,
            "preferences": self.preferences,
            "goals": self.goals,
            "values": self.values,
            "is_active": self.is_active,
            "is_default": self.is_default,
            "evolution_locked": self.evolution_locked,
            "creation_date": _iso_utc(self.creation_date),
            "last_modified": _iso_utc(self.last_modified),
            "last_evolution": _iso_utc(self.last_evolution),
            "evolution_count": self.evolution_count,
            "stability_score": self.stability_score,
            "traits": [trait.to_dict() for trait in self.traits] if self.traits else []
        }


class PersonaEvolution(Base):
    """
    PersonaEvolution model representing a single evolution step for a persona.
    
    Each evolution records the changes made to a persona, the reason for those changes,
    and metadata about the evolution process. This creates an audit trail of how
    the persona has evolved over time.
    """
    __tablename__ = "persona_evolutions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    persona_id = Column(String, ForeignKey("personas.id"), nullable=False, index=True)
    
    # Evolution details
    changes = Column(JSON, nullable=False)  # JSON of attribute changes
    reasoning = Column(Text, nullable=False)  # Why these changes were made
    evidence = Column(JSON, nullable=False)  # References to learnings, reflections, etc.
    confidence = Column(Float, nullable=False)  # 0.0-1.0 confidence in changes
    impact_score = Column(Float, nullable=False, default=0.0)  # 0.0-1.0 impact of changes
    
    # Evolution source
    source_type = Column(String, nullable=False)  # "learning", "reflection", "user_feedback", etc.
    source_id = Column(String, nullable=True)  # ID of source if applicable
    
    # Metadata
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    reviewed = Column(Boolean, default=False)  # Has been reviewed by system
    approved = Column(Boolean, default=True)  # Changes were applied
    
    # Relationships
    persona = relationship("Persona", back_populates="evolutions")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evolution to dictionary representation."""
        return {
            "id": self.id,
            "persona_id": self.persona_id,
            "changes": self.changes,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "impact_score": self.impact_score,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "timestamp": _iso_utc(self.timestamp),
            "reviewed": self.reviewed,
            "approved": self.approved
        }


class PersonaTrait(Base):
    """
    PersonaTrait model representing a specific trait or characteristic of a persona.
    
    Traits are more granular than the core persona attributes and can represent
    specific qualities, skills, preferences, or other aspects of the persona.
    """
    __tablename__ = "persona_traits"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    persona_id = Column(String, ForeignKey("personas.id"), nullable=False, index=True)
    
    # Trait details
    category = Column(String, nullable=False)  # Category of trait
    name = Column(String, nullable=False)  # Name of trait
    value = Column(Float, nullable=False)  # -1.0 to 1.0, or 0.0 to 1.0 depending on trait
    description = Column(Text, nullable=True)  # Description of this trait for this persona
    
    # Trait metadata
    confidence = Column(Float, nullable=False, default=1.0)  # 0.0-1.0 confidence in trait value
    stability = Column(Float, nullable=False, default=1.0)  # 0.0-1.0 stability of trait
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    evidence_count = Column(Integer, default=0)  # Number of pieces of evidence for this trait
    
    # Relationships
    persona = relationship("Persona", back_populates="traits")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trait to dictionary representation."""
        return {
            "id": self.id,
            "persona_id": self.persona_id,
            "category": self.category,
            "name": self.name,
            "value": self.value,
            "description": self.description,
            "confidence": self.confidence,
            "stability": self.stability,
            "last_updated": _iso_utc(self.last_updated),
            "evidence_count": self.evidence_count
        }


# Junction table for persona and concepts
persona_concept_association = Table(
    'persona_concept_association',
    Base.metadata,
    Column('persona_id', String, ForeignKey('personas.id'), primary_key=True),
    Column('concept_id', String, ForeignKey('concepts.id'), primary_key=True),
    Column('relevance_score', Float, default=0.5),
    Column('last_updated', DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
)
