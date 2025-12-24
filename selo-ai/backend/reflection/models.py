"""
Reflection Models

This module defines the data models for reflections and related entities.
These models are used for API serialization/deserialization and validation.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

try:
    from ..utils.datetime import utc_now, isoformat_utc
except ImportError:
    from backend.utils.datetime import utc_now, isoformat_utc

def _iso_utc(dt: datetime) -> str:
    """Wrapper for compatibility with pydantic json_encoders."""
    return isoformat_utc(dt)


class ReflectionBase(BaseModel):
    """Base model for reflections with common fields."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reflection_type: str
    user_profile_id: str
    trigger_source: str = "system"
    created_at: datetime = Field(default_factory=utc_now)
    
    model_config = {
        "json_encoders": {
            datetime: _iso_utc
        }
    }

class ReflectionCreateRequest(BaseModel):
    """Model for requesting reflection generation via API."""
    reflection_type: str
    user_profile_id: str
    memory_ids: Optional[List[str]] = None
    max_context_items: int = 10
    trigger_source: str = "api"
    additional_context: Optional[Dict[str, Any]] = None

class ReflectionCreate(ReflectionBase):
    """Model for creating new reflections."""
    content: str
    themes: List[str] = []
    insights: List[str] = []
    actions: List[str] = []
    # Accept arbitrary structure (labels, lists, numbers) coming from LLM output
    emotional_state: Optional[Dict[str, Any]] = None
    context_summary: Dict[str, int] = {}
    metadata: Dict[str, Any] = {}

class ReflectionDB(ReflectionBase):
    """Model for reflections as stored in the database."""
    content: str
    themes: List[str] = []
    insights: List[str] = []
    actions: List[str] = []
    emotional_state: Optional[Dict[str, Any]] = None
    context_summary: Dict[str, int] = {}
    metadata: Dict[str, Any] = {}
    updated_at: Optional[datetime] = None
    embedding: Optional[bytes] = None

class ReflectionResponse(ReflectionBase):
    """Model for reflection responses returned to clients."""
    content: str
    themes: List[str] = []
    insights: List[str] = []
    actions: List[str] = []
    emotional_state: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}
    updated_at: Optional[datetime] = None
    # Optional fields provided by list endpoint for UI convenience
    traits: Optional[List[Dict[str, Any]]] = []
    trait_changes: Optional[List[Dict[str, Any]]] = []

class ReflectionResult(BaseModel):
    """Model for LLM-generated reflection results."""
    content: str
    model: str = "unknown"
    themes: List[str] = []
    insights: List[str] = []
    actions: List[str] = []
    emotional_state: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}

class ReflectionRequest(BaseModel):
    """Model for internal reflection generation requests."""
    reflection_type: str
    user_profile_id: str
    memory_ids: Optional[List[str]] = None
    max_context_items: int = 10
    trigger_source: str = "system"
    

class ReflectionListResponse(BaseModel):
    """Model for listing multiple reflections."""
    reflections: List[ReflectionResponse]
    count: int
    limit: int
    offset: int
