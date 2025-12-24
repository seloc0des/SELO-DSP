"""
Conversation Model for SELO AI Persistent Memory

Stores all conversations for SELO's long-term memory and continuity.
Essential for SELO's growth and self-improvement across sessions.
"""

from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid
from typing import Dict, Any, Optional

from ..base import Base

class Conversation(Base):
    """
    Model for storing complete conversations between user and SELO.
    
    Each conversation represents a complete chat session with all
    messages, context, and metadata for SELO's persistent memory.
    """
    __tablename__ = "conversations"
    
    # Primary identifier
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Session identification
    session_id = Column(String(255), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    # Conversation metadata
    title = Column(String(500), nullable=True)  # Auto-generated conversation title
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    last_message_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    
    # Conversation status
    is_active = Column(Boolean, default=True, nullable=False)
    message_count = Column(Integer, default=0, nullable=False)
    
    # Conversation summary and context
    summary = Column(Text, nullable=True)  # AI-generated conversation summary
    topics = Column(JSONB, nullable=True)  # Extracted topics and themes
    sentiment = Column(JSONB, nullable=True)  # Overall conversation sentiment
    
    # Metadata (avoid reserved name 'metadata')
    meta_json = Column(JSONB, nullable=True)  # Additional conversation metadata
    
    # Relationships
    messages = relationship("ConversationMessage", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, session_id={self.session_id}, messages={self.message_count})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary representation."""
        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "title": self.title,
            "started_at": self._iso_utc(self.started_at),
            "last_message_at": self._iso_utc(self.last_message_at),
            "is_active": self.is_active,
            "message_count": self.message_count,
            "summary": self.summary,
            "topics": self.topics or [],
            "sentiment": self.sentiment or {},
            "metadata": self.meta_json or {}
        }

    @staticmethod
    def _iso_utc(value: Optional[datetime]) -> Optional[str]:
        if not value:
            return None
        coerced = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return coerced.isoformat().replace("+00:00", "Z")

class ConversationMessage(Base):
    """
    Model for individual messages within conversations.
    
    Stores each message (user and SELO) with full context,
    timing, and metadata for SELO's detailed memory.
    """
    __tablename__ = "conversation_messages"
    
    # Primary identifier
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Conversation relationship
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Message identification
    message_index = Column(Integer, nullable=False)  # Order within conversation
    role = Column(String(50), nullable=False, index=True)  # 'user' or 'assistant'
    
    # Message content
    content = Column(Text, nullable=False)
    
    # Message metadata
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    model_used = Column(String(100), nullable=True)  # Which LLM model was used
    token_count = Column(Integer, nullable=True)  # Token count for this message
    
    # Processing metadata
    processing_time = Column(Integer, nullable=True)  # Processing time in milliseconds
    reflection_triggered = Column(Boolean, default=False, nullable=False)  # Did this message trigger a reflection?
    
    # Message analysis
    sentiment_score = Column(JSONB, nullable=True)  # Message sentiment analysis
    topics_extracted = Column(JSONB, nullable=True)  # Topics/themes in this message
    entities_mentioned = Column(JSONB, nullable=True)  # Named entities mentioned
    
    # Additional metadata (avoid reserved name 'metadata')
    meta_json = Column(JSONB, nullable=True)  # Additional message metadata
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<ConversationMessage(id={self.id}, role={self.role}, index={self.message_index})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "id": str(self.id),
            "conversation_id": str(self.conversation_id),
            "message_index": self.message_index,
            "role": self.role,
            "content": self.content,
            "timestamp": Conversation._iso_utc(self.timestamp),
            "model_used": self.model_used,
            "token_count": self.token_count,
            "processing_time": self.processing_time,
            "reflection_triggered": self.reflection_triggered,
            "sentiment_score": self.sentiment_score or {},
            "topics_extracted": self.topics_extracted or [],
            "entities_mentioned": self.entities_mentioned or [],
            "metadata": self.meta_json or {}
        }

class Memory(Base):
    """
    Model for SELO's persistent memories extracted from conversations.
    
    Represents important information, experiences, and learnings
    that SELO should remember across all future interactions.
    """
    __tablename__ = "memories"
    
    # Primary identifier
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Memory identification
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    memory_type = Column(String(100), nullable=False, index=True)  # 'fact', 'preference', 'experience', etc.
    
    # Memory content
    content = Column(Text, nullable=False)
    summary = Column(String(500), nullable=True)  # Brief summary of the memory
    
    # Memory metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    last_accessed = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    access_count = Column(Integer, default=0, nullable=False)
    
    # Memory importance and relevance
    importance_score = Column(Integer, default=5, nullable=False)  # 1-10 scale
    confidence_score = Column(Integer, default=5, nullable=False)  # 1-10 scale
    
    # Memory relationships
    source_conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True)
    source_message_id = Column(UUID(as_uuid=True), ForeignKey("conversation_messages.id"), nullable=True)
    
    # Memory categorization
    tags = Column(JSONB, nullable=True)  # Tags for categorization
    topics = Column(JSONB, nullable=True)  # Related topics
    
    # Memory status
    is_active = Column(Boolean, default=True, nullable=False)
    is_validated = Column(Boolean, default=False, nullable=False)  # Has this memory been confirmed?
    
    # Additional metadata (avoid reserved name 'metadata')
    meta_json = Column(JSONB, nullable=True)
    
    def __repr__(self):
        return f"<Memory(id={self.id}, type={self.memory_type}, importance={self.importance_score})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary representation."""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "summary": self.summary,
            "created_at": Conversation._iso_utc(self.created_at),
            "last_accessed": Conversation._iso_utc(self.last_accessed),
            "access_count": self.access_count,
            "importance_score": self.importance_score,
            "confidence_score": self.confidence_score,
            "source_conversation_id": str(self.source_conversation_id) if self.source_conversation_id else None,
            "source_message_id": str(self.source_message_id) if self.source_message_id else None,
            "tags": self.tags or [],
            "topics": self.topics or [],
            "is_active": self.is_active,
            "is_validated": self.is_validated,
            "metadata": self.meta_json or {}
        }
