"""
User Model for Single-User SELO AI Installation

Since SELO AI is designed for single-user-per-install, this model
represents the single installation user with minimal complexity.
"""

from sqlalchemy import Column, String, DateTime, Boolean, Text
from datetime import datetime, timezone
import uuid

from ..base import Base

class User(Base):
    """
    Single user model for SELO AI installation.
    
    In a single-user system, there's typically only one user record
    representing the owner/operator of this SELO AI installation.
    """
    __tablename__ = "users"
    
    # Primary identifier
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Basic user information
    username = Column(String(100), unique=True, nullable=False, default="user")
    display_name = Column(String(200), nullable=True)
    
    # Installation metadata
    installation_id = Column(String, unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    last_active = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Simple authentication (optional)
    api_key = Column(String(255), nullable=True)  # Optional API key for basic auth
    
    # User preferences
    preferences = Column(Text, nullable=True)  # JSON string for user preferences
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, installation_id={self.installation_id})>"
    
    def to_dict(self):
        """Convert user to dictionary representation."""
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "installation_id": self.installation_id,
            "created_at": self._iso_utc(self.created_at),
            "last_active": self._iso_utc(self.last_active),
            "is_active": self.is_active,
            "preferences": self.preferences
        }

    @staticmethod
    def _iso_utc(value: datetime | None) -> str | None:
        if not value:
            return None
        coerced = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return coerced.isoformat().replace("+00:00", "Z")
