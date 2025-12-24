"""
Database models for the SDL (Self-Development Learning) module.

These models represent the core structures for storing and organizing
learnings, concepts, and their connections.
"""

import uuid
from typing import Dict, Any

from sqlalchemy import Column, String, Text, DateTime, Float, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from ..db.base import Base
from ..utils.datetime import utc_now, isoformat_utc


class Learning(Base):
    """
    Represents an individual learning derived from reflections or interactions.
    
    Learnings are the fundamental units of knowledge that the AI acquires
    through its experiences and reflections.
    """
    __tablename__ = "learnings"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    
    # Learning content and metadata
    content = Column(Text, nullable=False)
    source_type = Column(String(50), nullable=False)  # reflection, conversation, etc.
    source_id = Column(String(36), nullable=False, index=True)
    
    # Learning properties
    confidence = Column(Float, nullable=False, default=0.7)
    importance = Column(Float, nullable=False, default=0.5)
    domain = Column(String(100), nullable=False)  # The knowledge domain this applies to
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    active = Column(Boolean, default=True)
    vector_id = Column(String(36), nullable=True)  # ID in the vector database
    
    # JSON field for additional attributes
    attributes = Column(JSON, nullable=True)
    
    # Relationships
    concepts = relationship("Concept", secondary="learning_concept", back_populates="learnings")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the learning to a dictionary representation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "confidence": self.confidence,
            "importance": self.importance,
            "domain": self.domain,
            "created_at": isoformat_utc(self.created_at),
            "updated_at": isoformat_utc(self.updated_at),
            "active": self.active,
            "vector_id": self.vector_id,
            "attributes": self.attributes or {},
            "concepts": [c.name for c in self.concepts] if self.concepts else []
        }


class Concept(Base):
    """
    Represents a concept or idea that can be linked to multiple learnings.
    
    Concepts provide a way to organize and connect related learnings together,
    forming a knowledge graph.
    """
    __tablename__ = "concepts"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=True)
    
    # Concept properties
    importance = Column(Float, nullable=False, default=0.5)
    familiarity = Column(Float, nullable=False, default=0.3)  # How familiar the AI is with this concept
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    active = Column(Boolean, default=True)
    vector_id = Column(String(36), nullable=True)  # ID in the vector database
    
    # JSON field for additional attributes
    attributes = Column(JSON, nullable=True)
    
    # Relationships
    learnings = relationship("Learning", secondary="learning_concept", back_populates="concepts")
    outgoing_connections = relationship(
        "Connection", foreign_keys="Connection.source_id", back_populates="source"
    )
    incoming_connections = relationship(
        "Connection", foreign_keys="Connection.target_id", back_populates="target"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the concept to a dictionary representation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "importance": self.importance,
            "familiarity": self.familiarity,
            "created_at": isoformat_utc(self.created_at),
            "updated_at": isoformat_utc(self.updated_at),
            "active": self.active,
            "vector_id": self.vector_id,
            "attributes": self.attributes or {},
            "learning_count": len(self.learnings) if self.learnings else 0
        }


class LearningConcept(Base):
    """Association table for many-to-many relationship between Learning and Concept."""
    __tablename__ = "learning_concept"
    
    learning_id = Column(String(36), ForeignKey("learnings.id"), primary_key=True)
    concept_id = Column(String(36), ForeignKey("concepts.id"), primary_key=True)
    strength = Column(Float, default=0.5)  # Strength of association
    created_at = Column(DateTime(timezone=True), default=utc_now)


class Connection(Base):
    """
    Represents a connection between two concepts.
    
    Connections form the edges in the concept graph, allowing for
    complex knowledge representation and reasoning.
    """
    __tablename__ = "concept_connections"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    
    source_id = Column(String(36), ForeignKey("concepts.id"), nullable=False)
    target_id = Column(String(36), ForeignKey("concepts.id"), nullable=False)
    
    # Connection properties
    relation_type = Column(String(100), nullable=False)  # is-a, has-a, related-to, etc.
    strength = Column(Float, nullable=False, default=0.5)
    bidirectional = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    active = Column(Boolean, default=True)
    
    # Relationships
    source = relationship("Concept", foreign_keys=[source_id], back_populates="outgoing_connections")
    target = relationship("Concept", foreign_keys=[target_id], back_populates="incoming_connections")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the connection to a dictionary representation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "source_name": self.source.name if self.source else None,
            "target_name": self.target.name if self.target else None,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "bidirectional": self.bidirectional,
            "created_at": isoformat_utc(self.created_at),
            "updated_at": isoformat_utc(self.updated_at),
            "active": self.active
        }
