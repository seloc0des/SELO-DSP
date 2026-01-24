"""
Conversation Repository for SELO AI Persistent Memory

Manages persistent storage and retrieval of conversations, messages, and memories
for SELO's long-term continuity and growth.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, desc, asc, func
from sqlalchemy.orm import selectinload

from ..models.conversation import Conversation, ConversationMessage, Memory
from ..session import get_session

logger = logging.getLogger(__name__)

class ConversationRepository:
    """
    Repository for managing conversations, messages, and memories in SELO AI.
    
    Provides persistent storage for all of SELO's interactions and experiences,
    enabling long-term memory and continuity across sessions.
    """
    
    def __init__(self):
        """Initialize the conversation repository."""
        self.logger = logger
    
    async def get_or_create_conversation(self, session_id: str, user_id: str, 
                                       session: Optional[AsyncSession] = None) -> Conversation:
        """
        Get existing conversation or create a new one for the session.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            session: Optional database session
            
        Returns:
            Conversation: The conversation object
        """
        async with get_session(session) as db:
            return await self._get_or_create_conversation_impl(session_id, user_id, db)
    
    async def _get_or_create_conversation_impl(self, session_id: str, user_id: str, 
                                             session: AsyncSession) -> Conversation:
        """Implementation of get_or_create_conversation."""
        try:
            # Try to get existing active conversation for this session
            result = await session.execute(
                select(Conversation)
                .where(Conversation.session_id == session_id)
                .where(Conversation.is_active == True)
                .order_by(desc(Conversation.last_message_at))
                .limit(1)
            )
            conversation = result.scalar_one_or_none()
            
            if conversation is None:
                # Create new conversation
                conversation = Conversation(
                    session_id=session_id,
                    user_id=user_id,
                    title=f"Conversation {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                    is_active=True,
                    message_count=0
                )
                session.add(conversation)
                await session.flush()  # Flush to get auto-generated ID without full commit
                self.logger.info(f"Created new conversation: {conversation.id}")
            
            return conversation
            
        except Exception as e:
            self.logger.error(f"Error getting/creating conversation: {str(e)}", exc_info=True)
            raise
    
    async def add_message(self, conversation_id: str, role: str, content: str, 
                         model_used: Optional[str] = None, processing_time: Optional[int] = None,
                         reflection_triggered: bool = False, session: Optional[AsyncSession] = None) -> ConversationMessage:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            role: Message role ('user' or 'assistant')
            content: Message content
            model_used: LLM model used (for assistant messages)
            processing_time: Processing time in milliseconds
            reflection_triggered: Whether this message triggered a reflection
            session: Optional database session
            
        Returns:
            ConversationMessage: The created message
        """
        async with get_session(session) as db:
            return await self._add_message_impl(
                conversation_id, role, content, model_used, processing_time, reflection_triggered, db
            )
    
    async def _add_message_impl(self, conversation_id: str, role: str, content: str,
                               model_used: Optional[str], processing_time: Optional[int],
                               reflection_triggered: bool, session: AsyncSession) -> ConversationMessage:
        """Implementation of add_message."""
        try:
            # Lock the conversation row to prevent race conditions when computing message_index.
            # This ensures concurrent add_message calls serialize on the same conversation.
            from sqlalchemy import text
            await session.execute(
                text("SELECT id FROM conversations WHERE id = :conv_id FOR UPDATE"),
                {"conv_id": str(conversation_id)}
            )
            
            # Get current message count for this conversation (now safe under lock)
            result = await session.execute(
                select(func.count(ConversationMessage.id))
                .where(ConversationMessage.conversation_id == conversation_id)
            )
            message_count = result.scalar() or 0
            
            # Create new message
            message = ConversationMessage(
                conversation_id=conversation_id,
                message_index=message_count,
                role=role,
                content=content,
                model_used=model_used,
                processing_time=processing_time,
                reflection_triggered=reflection_triggered,
                timestamp=datetime.now(timezone.utc)
            )
            session.add(message)
            
            # Update conversation metadata
            await session.execute(
                update(Conversation)
                .where(Conversation.id == conversation_id)
                .values(
                    last_message_at=datetime.now(timezone.utc),
                    message_count=message_count + 1
                )
            )
            
            await session.flush()  # Flush to get auto-generated ID without full commit
            
            self.logger.info(f"Added message to conversation {conversation_id}: {role}")
            return message
            
        except Exception as e:
            self.logger.error(f"Error adding message: {str(e)}", exc_info=True)
            raise
    
    async def get_conversation_history(self, session_id: str, limit: int = 50, 
                                     session: Optional[AsyncSession] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            session: Optional database session
            
        Returns:
            List of message dictionaries in chronological order
        """
        async with get_session(session) as db:
            return await self._get_conversation_history_impl(session_id, limit, db)
    
    async def _get_conversation_history_impl(self, session_id: str, limit: int, 
                                           session: AsyncSession) -> List[Dict[str, Any]]:
        """Implementation of get_conversation_history."""
        try:
            # Get the most recent conversation for this session
            conv_result = await session.execute(
                select(Conversation)
                .where(Conversation.session_id == session_id)
                .where(Conversation.is_active == True)
                .order_by(desc(Conversation.last_message_at))
                .limit(1)
            )
            conversation = conv_result.scalar_one_or_none()
            
            if not conversation:
                return []
            
            # Get messages for this conversation
            result = await session.execute(
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == conversation.id)
                .order_by(asc(ConversationMessage.message_index))
                .limit(limit)
            )
            messages = result.scalars().all()
            
            # Convert to the format expected by the chat system
            history = []
            for msg in messages:
                history.append({
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": self._iso_utc(msg.timestamp),
                    "model_used": msg.model_used,
                    "reflection_triggered": msg.reflection_triggered
                })
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error getting conversation history: {str(e)}", exc_info=True)
            return []
    
    async def create_memory(self, user_id: str, memory_type: str, content: str,
                           importance_score: int = 5, confidence_score: int = 5,
                           source_conversation_id: Optional[str] = None,
                           source_message_id: Optional[str] = None,
                           tags: Optional[List[str]] = None,
                           session: Optional[AsyncSession] = None) -> Memory:
        """
        Create a new memory for SELO.
        
        Args:
            user_id: User identifier
            memory_type: Type of memory ('fact', 'preference', 'experience', etc.)
            content: Memory content
            importance_score: Importance score (1-10)
            confidence_score: Confidence score (1-10)
            source_conversation_id: Source conversation ID
            source_message_id: Source message ID
            tags: Optional tags for categorization
            session: Optional database session
            
        Returns:
            Memory: The created memory
        """
        async with get_session(session) as db:
            return await self._create_memory_impl(
                user_id, memory_type, content, importance_score, confidence_score,
                source_conversation_id, source_message_id, tags, db
            )
    
    async def _create_memory_impl(self, user_id: str, memory_type: str, content: str,
                                 importance_score: int, confidence_score: int,
                                 source_conversation_id: Optional[str],
                                 source_message_id: Optional[str],
                                 tags: Optional[List[str]], session: AsyncSession) -> Memory:
        """Implementation of create_memory."""
        try:
            memory = Memory(
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                summary=content[:500] if len(content) > 500 else content,  # Auto-generate summary
                importance_score=importance_score,
                confidence_score=confidence_score,
                source_conversation_id=source_conversation_id,
                source_message_id=source_message_id,
                tags=tags or [],
                is_active=True,
                is_validated=False
            )
            session.add(memory)
            await session.flush()  # Flush to get auto-generated ID without full commit

            self.logger.info(f"Created memory: {memory.id} (type: {memory_type})")
            return memory

        except Exception as e:
            self.logger.error(f"Error creating memory: {str(e)}", exc_info=True)
            raise

    async def get_memories(self, user_id: str, memory_type: Optional[str] = None,
                          importance_threshold: int = 3, limit: int = 100,
                          session: Optional[AsyncSession] = None) -> List[Memory]:
        """
        Get memories for a user.
        
        Args:
            user_id: User identifier
            memory_type: Optional filter by memory type
            importance_threshold: Minimum importance score
            limit: Maximum number of memories to return
            session: Optional database session
            
        Returns:
            List of Memory objects
        """
        async with get_session(session) as db:
            return await self._get_memories_impl(user_id, memory_type, importance_threshold, limit, db)
    
    async def _get_memories_impl(self, user_id: str, memory_type: Optional[str],
                                importance_threshold: int, limit: int, session: AsyncSession) -> List[Memory]:
        """Implementation of get_memories."""
        try:
            query = select(Memory).where(Memory.user_id == user_id).where(Memory.is_active == True)
            
            if memory_type:
                query = query.where(Memory.memory_type == memory_type)
            
            query = query.where(Memory.importance_score >= importance_threshold)
            query = query.order_by(desc(Memory.importance_score), desc(Memory.last_accessed))
            query = query.limit(limit)
            
            result = await session.execute(query)
            memories = result.scalars().all()
            
            # Update access timestamps - keep UUIDs as-is (don't convert to string)
            memory_ids = [memory.id for memory in memories]
            if memory_ids:
                await session.execute(
                    update(Memory)
                    .where(Memory.id.in_(memory_ids))
                    .values(
                        last_accessed=datetime.now(timezone.utc),
                        access_count=Memory.access_count + 1
                    )
                )
                # Note: get_session context manager handles commit automatically
            
            return list(memories)
            
        except Exception as e:
            self.logger.error(f"Error getting memories: {str(e)}", exc_info=True)
            return []

    async def has_user_messages(self, user_id: str, session: Optional[AsyncSession] = None) -> bool:
        """Check if any user-authored messages exist for the given user."""
        async with get_session(session) as db:
            return await self._has_user_messages_impl(user_id, db)

    async def _has_user_messages_impl(self, user_id: str, session: AsyncSession) -> bool:
        try:
            result = await session.execute(
                select(func.count(ConversationMessage.id))
                .join(Conversation, ConversationMessage.conversation_id == Conversation.id)
                .where(Conversation.user_id == user_id)
                .where(ConversationMessage.role == "user")
                .limit(1)
            )
            count = result.scalar() or 0
            return count > 0
        except Exception as e:
            self.logger.error(f"Error checking user messages for user {user_id}: {str(e)}", exc_info=True)
            return False

    @staticmethod
    def _iso_utc(value: Optional[datetime]) -> Optional[str]:
        if not value:
            return None
        coerced = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return coerced.isoformat().replace("+00:00", "Z")
    
    async def update_conversation_summary(self, conversation_id: str, summary: str,
                                        topics: Optional[List[str]] = None,
                                        sentiment: Optional[Dict[str, Any]] = None,
                                        session: Optional[AsyncSession] = None) -> bool:
        """
        Update conversation summary and metadata.
        
        Args:
            conversation_id: Conversation ID
            summary: Conversation summary
            topics: Extracted topics
            sentiment: Sentiment analysis
            session: Optional database session
            
        Returns:
            bool: True if updated successfully
        """
        async with get_session(session) as db:
            return await self._update_conversation_summary_impl(conversation_id, summary, topics, sentiment, db)
    
    async def _update_conversation_summary_impl(self, conversation_id: str, summary: str,
                                              topics: Optional[List[str]], sentiment: Optional[Dict[str, Any]],
                                              session: AsyncSession) -> bool:
        """Implementation of update_conversation_summary."""
        try:
            await session.execute(
                update(Conversation)
                .where(Conversation.id == conversation_id)
                .values(
                    summary=summary,
                    topics=topics or [],
                    sentiment=sentiment or {}
                )
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating conversation summary: {str(e)}", exc_info=True)
            return False

    async def count_conversations(self, user_id: str, session: Optional[AsyncSession] = None) -> int:
        """
        Count total conversations for a user.
        
        Args:
            user_id: User identifier
            session: Optional database session
            
        Returns:
            int: Total count of conversations
        """
        async with get_session(session) as db:
            try:
                return await self._count_conversations_impl(user_id, db)
            except Exception as e:
                self.logger.error(f"Error counting conversations: {e}")
                return 0

    async def _count_conversations_impl(self, user_id: str, session: AsyncSession) -> int:
        """Implementation of count_conversations."""
        try:
            result = await session.execute(
                select(func.count(Conversation.id))
                .where(Conversation.user_id == user_id)
            )
            return result.scalar() or 0
        except Exception as e:
            self.logger.error(f"Error in count_conversations_impl: {str(e)}", exc_info=True)
            return 0

    async def list_conversations(self, user_id: str, limit: int = 50, offset: int = 0, 
                               sort_by: str = "updated_at", sort_order: str = "desc",
                               session: Optional[AsyncSession] = None) -> List[Dict[str, Any]]:
        """
        List conversations for a user with pagination.
        
        Args:
            user_id: User identifier
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            session: Optional database session
            
        Returns:
            List[Dict]: List of conversation dictionaries
        """
        async with get_session(session) as db:
            try:
                return await self._list_conversations_impl(user_id, limit, offset, sort_by, sort_order, db)
            except Exception as e:
                self.logger.error(f"Error listing conversations: {e}")
                return []

    async def _list_conversations_impl(self, user_id: str, limit: int, offset: int,
                                     sort_by: str, sort_order: str, session: AsyncSession) -> List[Dict[str, Any]]:
        """Implementation of list_conversations."""
        try:
            # Build sort clause
            sort_column = getattr(Conversation, sort_by, Conversation.last_message_at)
            if sort_order.lower() == "asc":
                sort_clause = asc(sort_column)
            else:
                sort_clause = desc(sort_column)

            # Execute query
            result = await session.execute(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(sort_clause)
                .limit(limit)
                .offset(offset)
                .options(selectinload(Conversation.messages))
            )
            
            conversations = result.scalars().all()
            
            # Convert to dictionaries
            conversation_list = []
            for conv in conversations:
                conversation_dict = {
                    "id": conv.id,
                    "user_id": conv.user_id,
                    "session_id": conv.session_id,
                    "title": conv.title,
                    "summary": conv.summary,
                    "topics": conv.topics or [],
                    "sentiment": conv.sentiment or {},
                    "created_at": conv.started_at.isoformat() if conv.started_at else None,
                    "updated_at": conv.last_message_at.isoformat() if conv.last_message_at else None,
                    "message_count": len(conv.messages) if conv.messages else 0
                }
                conversation_list.append(conversation_dict)
            
            return conversation_list
            
        except Exception as e:
            self.logger.error(f"Error in list_conversations_impl: {str(e)}", exc_info=True)
            return []
    
    async def get_recent_messages(
        self,
        user_id: str,
        limit: int = 50,
        hours: Optional[int] = None,
        session: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent messages for a user across all conversations.
        
        Args:
            user_id: User identifier
            limit: Maximum number of messages to return
            hours: Optional filter to only include messages from last N hours
            session: Optional database session
            
        Returns:
            List of recent messages ordered by timestamp descending
        """
        async with get_session(session) as db:
            return await self._get_recent_messages_impl(user_id, limit, hours, db)
    
    async def _get_recent_messages_impl(
        self,
        user_id: str,
        limit: int,
        hours: Optional[int],
        session: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Implementation of get_recent_messages."""
        try:
            # Build query to get messages from user's conversations
            query = (
                select(ConversationMessage)
                .join(Conversation, ConversationMessage.conversation_id == Conversation.id)
                .where(Conversation.user_id == user_id)
            )
            
            # Apply time filter if specified
            if hours is not None:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
                query = query.where(ConversationMessage.timestamp >= cutoff_time)
            
            # Order by most recent first and apply limit
            query = query.order_by(desc(ConversationMessage.timestamp)).limit(limit)
            
            result = await session.execute(query)
            messages = result.scalars().all()
            
            # Convert to dictionaries
            message_list = []
            for msg in messages:
                message_list.append({
                    "id": str(msg.id),
                    "conversation_id": msg.conversation_id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": self._iso_utc(msg.timestamp),
                    "model_used": msg.model_used,
                    "processing_time": msg.processing_time,
                    "reflection_triggered": msg.reflection_triggered,
                    "message_index": msg.message_index
                })
            
            return message_list
            
        except Exception as e:
            self.logger.error(f"Error getting recent messages for {user_id}: {str(e)}", exc_info=True)
            return []
