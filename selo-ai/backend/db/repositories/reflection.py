"""
Reflection Repository

This module implements the repository pattern for reflection persistence.
It handles CRUD operations for reflections in the database.
"""

from typing import Dict, List, Optional, Any, Union
import json
import logging
from datetime import datetime, timedelta, timezone
import uuid
from sqlalchemy import select, update, delete, and_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from ..models.reflection import Reflection, ReflectionMemory, RelationshipQuestionQueue
from ..session import get_session

logger = logging.getLogger("selo.db.reflection")

class ReflectionRepository:
    """
    Repository for storing and retrieving reflections from the database.
    
    This class implements the repository pattern for reflections, providing
    a clean abstraction over the database operations.
    """
    
    def __init__(self, db_session=None):
        """
        Initialize the repository with a database session.
        
        Args:
            db_session: SQLAlchemy database session (AsyncSession)
        """
        self.db_session = db_session
        # Keep cache for fallback when DB is not available
        self._reflections_cache = {}
        self._relationship_answer_audit_cache: Dict[str, Dict[str, Any]] = {}
        
    async def create_reflection(self, reflection_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new reflection in the database.
        
        Args:
            reflection_data: Dictionary with reflection data
            
        Returns:
            Created reflection with database ID
        """
        try:
            # Generate ID if not provided
            reflection_id = reflection_data.get("id") or str(uuid.uuid4())
            reflection_data["id"] = reflection_id
            
            # Set timestamps if not provided and coerce to datetime if provided as epoch or string
            now = datetime.now(timezone.utc)
            def _to_dt(val):
                if val is None:
                    return None
                try:
                    # Accept datetime as-is
                    if isinstance(val, datetime):
                        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
                    # Accept UNIX epoch seconds (int/float)
                    if isinstance(val, (int, float)):
                        return datetime.fromtimestamp(val, tz=timezone.utc)
                    # Accept ISO8601 string
                    if isinstance(val, str):
                        try:
                            # fromisoformat supports 'YYYY-MM-DDTHH:MM:SS.mmmmmm'
                            dt = datetime.fromisoformat(val.replace('Z', '+00:00'))
                            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                        except Exception:
                            pass
                    # Fallback to now if unrecognized
                    return now
                except Exception:
                    return now
            reflection_data["created_at"] = _to_dt(reflection_data.get("created_at", now)) or now
            reflection_data["updated_at"] = _to_dt(reflection_data.get("updated_at", now)) or now
            
            # Extract memory_ids if provided to create relationships
            memory_ids = reflection_data.pop("memory_ids", [])
            relationship_questions = reflection_data.pop("relationship_questions", [])
            
            async with get_session(self.db_session) as session:
                # Compose result if not provided directly
                result_blob = reflection_data.get("result")
                if result_blob is None:
                    # Build from commonly provided fields in ReflectionProcessor._store_reflection
                    result_blob = {
                        "content": reflection_data.get("content"),
                        "themes": reflection_data.get("themes", []),
                        "insights": reflection_data.get("insights", []),
                        "actions": reflection_data.get("actions", []),
                        "emotional_state": reflection_data.get("emotional_state"),
                    }
                    # Optional trailing metadata inside result
                    if reflection_data.get("metadata"):
                        # Don't duplicate model metadata at top-level; store only informative bits here
                        result_blob.setdefault("metadata", {})
                        try:
                            for k, v in (reflection_data.get("metadata") or {}).items():
                                # Keep lightweight keys only
                                if k in ("model", "coherence_rationale", "notes") or isinstance(v, (str, int, float, list, dict)):
                                    result_blob["metadata"][k] = v
                        except Exception:
                            pass
                    if reflection_data.get("trait_changes"):
                        result_blob["trait_changes"] = reflection_data.get("trait_changes")

                # Create new reflection object
                reflection_model = Reflection(
                    id=uuid.UUID(reflection_id),
                    user_profile_id=reflection_data["user_profile_id"],
                    reflection_type=reflection_data["reflection_type"],
                    result=result_blob,
                    created_at=reflection_data["created_at"],
                    updated_at=reflection_data["updated_at"],
                    embedding=reflection_data.get("embedding"),
                    reflection_metadata=reflection_data.get("metadata", {})
                )
                
                session.add(reflection_model)
                
                # Add memory relationships if provided
                for memory_id in memory_ids:
                    memory_relation = ReflectionMemory(
                        reflection_id=uuid.UUID(reflection_id),
                        memory_id=uuid.UUID(memory_id)
                    )
                    session.add(memory_relation)
                
                await session.flush()

                if relationship_questions:
                    queue_entries = []
                    for question in relationship_questions:
                        context = question or {}
                        if not isinstance(context, dict):
                            continue
                        question_text = context.get("question")
                        prompt = context.get("prompt")
                        if not question_text or not prompt:
                            continue
                        suggested_delay_days = context.get("suggested_delay_days", 0) or 0
                        queued_at = datetime.now(timezone.utc)
                        available_at = queued_at + timedelta(days=max(0, suggested_delay_days))
                        queue_entries.append(
                            RelationshipQuestionQueue(
                                user_profile_id=reflection_data["user_profile_id"],
                                reflection_id=reflection_model.id,
                                question=question_text,
                                topic=context.get("topic", "general"),
                                priority=int(context.get("priority", 3) or 3),
                                suggested_delay_days=suggested_delay_days,
                                prompt=prompt,
                                status=context.get("status", "pending"),
                                insight_value=context.get("insight_value"),
                                existing_conflicts=context.get("existing_conflicts", []),
                                queued_at=queued_at,
                                available_at=available_at,
                                raw_payload=context,
                            )
                        )
                    if queue_entries:
                        session.add_all(queue_entries)
                
                # Convert to dictionary for return
                result = reflection_model.to_dict()
                if relationship_questions:
                    result.setdefault("metadata", {})["relationship_questions"] = relationship_questions
                
                # Keep in cache for quick access
                self._reflections_cache[reflection_id] = result
                logger.info(f"Created reflection with ID {reflection_id}")
                
                return result
                
        except Exception as e:
            logger.error(f"Error creating reflection: {str(e)}", exc_info=True)
            raise
            
    async def get_reflection(self, reflection_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a reflection by ID.
        
        Args:
            reflection_id: ID of the reflection to retrieve
            
        Returns:
            Reflection data or None if not found
        """
        try:
            # Check cache first for performance
            if reflection_id in self._reflections_cache:
                return self._reflections_cache[reflection_id]
                
            async with get_session(self.db_session) as session:
                # Query the database
                stmt = select(Reflection).where(Reflection.id == uuid.UUID(reflection_id))
                result = await session.execute(stmt)
                reflection = result.scalar_one_or_none()
                
                if reflection:
                    # Convert to dictionary and cache
                    reflection_dict = reflection.to_dict()
                    self._reflections_cache[reflection_id] = self._serialize_reflection(reflection_dict)
                    return self._reflections_cache[reflection_id]
                return None
                
        except Exception as e:
            logger.error(f"Error getting reflection {reflection_id}: {str(e)}", exc_info=True)
            return None
            
    async def update_reflection(self, 
                               reflection_id: str, 
                               update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an existing reflection.
        
        Args:
            reflection_id: ID of the reflection to update
            update_data: Dictionary with fields to update
            
        Returns:
            Updated reflection or None if not found
        """
        try:
            # Set updated timestamp
            update_data["updated_at"] = datetime.now(timezone.utc)
            
            async with get_session(self.db_session) as session:
                # First check if reflection exists
                stmt = select(Reflection).where(Reflection.id == uuid.UUID(reflection_id))
                result = await session.execute(stmt)
                reflection = result.scalar_one_or_none()
                
                if not reflection:
                    return None
                
                # Update the fields
                for key, value in update_data.items():
                    if hasattr(reflection, key):
                        setattr(reflection, key, value if key not in {"created_at", "updated_at"} else (value if isinstance(value, datetime) and value.tzinfo else (value.replace(tzinfo=timezone.utc) if isinstance(value, datetime) else reflection.updated_at)))
                
                # Special handling for embedding which might be numpy array or bytes
                if "embedding" in update_data:
                    reflection.embedding = update_data["embedding"]
                
                # Update cache and return
                reflection_dict = reflection.to_dict()
                self._reflections_cache[reflection_id] = reflection_dict
                
                logger.info(f"Updated reflection {reflection_id} in database")
                return reflection_dict
                
        except Exception as e:
            logger.error(f"Error updating reflection {reflection_id}: {str(e)}", exc_info=True)
            return None

    async def _format_reflection_list(self, reflections):
        reflection_dicts = []
        for reflection in reflections:
            reflection_dict = reflection.to_dict()
            self._reflections_cache[str(reflection.id)] = reflection_dict
            reflection_dicts.append(reflection_dict)
        return reflection_dicts

    async def list_reflections(self,
                              user_profile_id: Optional[str] = None,
                              reflection_type: Optional[str] = None,
                              limit: int = 10,
                              offset: int = 0,
                              sort_by: str = "created_at",
                              sort_order: str = "desc") -> List[Dict[str, Any]]:
        """
        List reflections with optional filtering.
        
        Args:
            user_profile_id: Optional filter by user profile
            reflection_type: Optional filter by reflection type
            limit: Maximum number of reflections to return
            offset: Pagination offset
            sort_by: Field to sort by (default: created_at)
            sort_order: Sort order - 'asc' or 'desc' (default: desc)
            
        Returns:
            List of reflections matching the criteria
        """
        try:
            async with get_session(self.db_session) as session:
                # Build query with filters
                query = select(Reflection)
                conditions = []
                
                if user_profile_id:
                    conditions.append(Reflection.user_profile_id == user_profile_id)
                    
                if reflection_type:
                    conditions.append(Reflection.reflection_type == reflection_type)
                
                if conditions:
                    query = query.where(and_(*conditions))
                
                # Apply ordering and pagination
                sort_column = getattr(Reflection, sort_by, Reflection.created_at)
                if sort_order.lower() == "asc":
                    query = query.order_by(sort_column)
                else:
                    query = query.order_by(desc(sort_column))
                query = query.limit(limit).offset(offset)
                
                result = await session.execute(query)
                reflections = result.scalars().all()
                
                # Convert to dictionaries and update cache
                reflection_dicts = []
                for reflection in reflections:
                    reflection_dict = reflection.to_dict()
                    self._reflections_cache[str(reflection.id)] = reflection_dict
                    reflection_dicts.append(reflection_dict)
                
                return reflection_dicts
                
        except Exception as e:
            logger.error(f"Error listing reflections from database: {str(e)}", exc_info=True)
            # Fall back to memory cache if database query failed
            reflections = list(self._reflections_cache.values())
            
            # Apply filters
            if user_profile_id:
                reflections = [r for r in reflections if r.get("user_profile_id") == user_profile_id]
                
            if reflection_type:
                reflections = [r for r in reflections if r.get("reflection_type") == reflection_type]
                
            # Sort by created_at (descending)
            reflections.sort(key=lambda r: r.get("created_at", 0), reverse=True)
            
            # Apply pagination
            return reflections[offset:offset + limit]

    async def get_latest_reflection(
        self,
        user_profile_id: str,
        *,
        include_baseline: bool = True,
        limit: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """Return the most recent reflection for a given user, optionally skipping boot baselines."""

        if not user_profile_id:
            return None

        try:
            reflections = await self.list_reflections(
                user_profile_id=user_profile_id,
                limit=max(1, limit),
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug(
                "Latest reflection lookup failed for user %s: %s",
                user_profile_id,
                exc,
            )
            return None

        baseline_candidate: Optional[Dict[str, Any]] = None
        for reflection in reflections:
            if not self._is_baseline_reflection(reflection):
                return reflection
            if baseline_candidate is None:
                baseline_candidate = reflection

        if include_baseline:
            return baseline_candidate

        return None

    @staticmethod
    def _serialize_reflection(data: Dict[str, Any]) -> Dict[str, Any]:
        def _iso(value):
            if not isinstance(value, datetime):
                return value
            coerced = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
            return coerced.isoformat().replace("+00:00", "Z")

        serialized = dict(data)
        for field in ("created_at", "updated_at", "completed_at", "queued_at", "available_at", "delivered_at", "answered_at"):
            if field in serialized:
                serialized[field] = _iso(serialized[field])
        return serialized

    @staticmethod
    def _is_baseline_reflection(reflection: Dict[str, Any]) -> bool:
        """Determine whether a reflection originates from boot-time baseline seeding."""

        if not isinstance(reflection, dict):
            return False

        metadata = reflection.get("metadata")
        if isinstance(metadata, dict):
            if metadata.get("baseline") is True:
                return True
            source = metadata.get("source")
            if isinstance(source, str) and source.startswith("boot_"):
                return True

        result = reflection.get("result")
        if isinstance(result, dict):
            result_metadata = result.get("metadata")
            if isinstance(result_metadata, dict) and result_metadata.get("baseline") is True:
                return True

        return False

    async def count_reflections(self,
                               user_profile_id: Optional[str] = None,
                               reflection_type: Optional[str] = None) -> int:
        """
        Count reflections with optional filtering.
        
        Args:
            user_profile_id: Optional filter by user profile
            reflection_type: Optional filter by reflection type
            
        Returns:
            Total count of reflections matching the criteria
        """
        try:
            async with get_session(self.db_session) as session:
                # Build count query with filters
                query = select(func.count(Reflection.id))
                conditions = []
                
                if user_profile_id:
                    conditions.append(Reflection.user_profile_id == user_profile_id)
                    
                if reflection_type:
                    conditions.append(Reflection.reflection_type == reflection_type)
                
                if conditions:
                    query = query.where(and_(*conditions))
                
                result = await session.execute(query)
                count = result.scalar()
                
                return count or 0
                
        except Exception as e:
            logger.error(f"Error counting reflections from database: {str(e)}", exc_info=True)
            # Fallback to memory cache if database query failed
            reflections = list(self._reflections_cache.values())
            
            # Apply filters
            if user_profile_id:
                reflections = [r for r in reflections if r.get("user_profile_id") == user_profile_id]
                
            if reflection_type:
                reflections = [r for r in reflections if r.get("reflection_type") == reflection_type]
                
            return len(reflections)

    async def list_relationship_question_queue(
        self,
        user_profile_id: Optional[str] = None,
        status: Optional[str] = "pending",
        limit: int = 50,
        include_future: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve queued relationship questions for scheduling and chat injection."""

        try:
            async with get_session(self.db_session) as session:
                query = select(RelationshipQuestionQueue)
                conditions = []

                if user_profile_id:
                    conditions.append(RelationshipQuestionQueue.user_profile_id == user_profile_id)
                if status:
                    conditions.append(RelationshipQuestionQueue.status == status)
                if not include_future:
                    conditions.append(RelationshipQuestionQueue.available_at <= datetime.now(timezone.utc))

                if conditions:
                    query = query.where(and_(*conditions))

                query = query.order_by(RelationshipQuestionQueue.available_at.asc()).limit(limit)

                result = await session.execute(query)
                rows = result.scalars().all()
                return [row.to_dict() for row in rows]
                
        except Exception as e:
            logger.error(f"Error listing relationship question queue: {str(e)}", exc_info=True)
            return []

    async def mark_relationship_question_delivered(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Mark a relationship question as delivered/awaiting response."""

        if not question_id:
            return None

        try:
            question_uuid = uuid.UUID(str(question_id))
        except Exception:
            logger.warning(f"Invalid relationship question ID for delivery mark: {question_id}")
            return None

        try:
            async with get_session(self.db_session) as session:
                result = await session.execute(
                    select(RelationshipQuestionQueue).where(RelationshipQuestionQueue.id == question_uuid)
                )
                question = result.scalar_one_or_none()
                if not question:
                    return None

                if question.status != "answered":
                    question.status = "awaiting_response"
                    question.delivered_at = datetime.now(timezone.utc)
                return question.to_dict()
                
        except Exception as e:
            logger.error(f"Error marking relationship question delivered: {str(e)}", exc_info=True)
            raise

    async def upsert_relationship_audit_state(self, payload: Dict[str, Any]) -> None:
        """Cache the most recent relationship-answer audit data per user."""

        if not isinstance(payload, dict):
            return

        user_id = payload.get("user_id")
        if not user_id:
            return

        # For now, store in memory cache; can be replaced with persistent storage later.
        sanitized = {
            "user_id": user_id,
            "analyzed_at": payload.get("analyzed_at"),
            "total_memories": payload.get("total_memories", 0),
            "duplicate_tags": list(payload.get("duplicate_tags", [])),
            "tag_histogram": dict(payload.get("tag_histogram", {})),
        }
        self._relationship_answer_audit_cache[user_id] = sanitized

    async def get_relationship_audit_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Return the cached audit snapshot for a user if present."""

        if not user_id:
            return None
        return self._relationship_answer_audit_cache.get(user_id)

    async def get_relationship_question_awaiting_response(
        self,
        user_profile_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the oldest relationship question awaiting a user response."""

        if not user_profile_id:
            return None

        try:
            async with get_session(self.db_session) as session:
                query = (
                    select(RelationshipQuestionQueue)
                    .where(RelationshipQuestionQueue.user_profile_id == user_profile_id)
                    .where(RelationshipQuestionQueue.status == "awaiting_response")
                    .order_by(
                        RelationshipQuestionQueue.delivered_at.asc().nulls_last(),
                        RelationshipQuestionQueue.available_at.asc(),
                    )
                    .limit(1)
                )
                result = await session.execute(query)
                question = result.scalar_one_or_none()
                return question.to_dict() if question else None
        except Exception as e:
            logger.error(f"Error retrieving awaiting relationship question: {str(e)}", exc_info=True)
            return None

    async def mark_relationship_question_answered(
        self,
        question_id: str,
        answer_text: str,
        memory_id: Optional[str] = None,
        answer_tags: Optional[List[str]] = None,
        importance_score: Optional[int] = None,
        confidence_score: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Mark a relationship question as answered and attach memory linkage."""

        if not question_id:
            return None

        try:
            question_uuid = uuid.UUID(str(question_id))
        except Exception:
            logger.warning(f"Invalid relationship question ID for answer mark: {question_id}")
            return None

        memory_uuid: Optional[uuid.UUID] = None
        if memory_id:
            try:
                memory_uuid = uuid.UUID(str(memory_id))
            except Exception:
                logger.warning(f"Invalid memory ID for relationship question answer link: {memory_id}")
                memory_uuid = None

        try:
            async with get_session(self.db_session) as session:
                result = await session.execute(
                    select(RelationshipQuestionQueue).where(RelationshipQuestionQueue.id == question_uuid)
                )
                question = result.scalar_one_or_none()
                if not question:
                    return None

                question.status = "answered"
                question.answered_at = datetime.now(timezone.utc)
                if memory_uuid:
                    question.answer_memory_id = memory_uuid
                if answer_tags is not None:
                    try:
                        question.answer_tags = list(answer_tags)
                    except Exception:
                        question.answer_tags = answer_tags
                if importance_score is not None:
                    question.answer_importance_score = importance_score
                if confidence_score is not None:
                    question.answer_confidence_score = confidence_score

                try:
                    payload = question.raw_payload or {}
                    if isinstance(payload, dict):
                        payload["answer_text"] = answer_text
                        if answer_tags is not None:
                            payload["answer_tags"] = list(answer_tags)
                        if importance_score is not None:
                            payload["answer_importance_score"] = importance_score
                        if confidence_score is not None:
                            payload["answer_confidence_score"] = confidence_score
                        question.raw_payload = payload
                except Exception:
                    pass

                return question.to_dict()
        except Exception as e:
            logger.error(f"Error marking relationship question answered: {str(e)}", exc_info=True)
            raise
