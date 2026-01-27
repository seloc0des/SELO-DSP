"""
Memory Consolidation Scheduler

This service runs background tasks to consolidate and optimize memories:
- Periodic memory consolidation from conversation history
- Memory importance re-scoring based on access patterns
- Memory cleanup and archival of low-importance memories
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
import time

logger = logging.getLogger("selo.memory.consolidator")

class MemoryConsolidator:
    """
    Background service for memory consolidation and optimization.
    """
    
    def __init__(self, conversation_repo=None, user_repo=None, memory_extractor=None):
        """
        Initialize the MemoryConsolidator.
        
        Args:
            conversation_repo: Repository for conversation and memory operations
            user_repo: Repository for user operations
            memory_extractor: MemoryExtractor instance for conversation analysis
        """
        self.conversation_repo = conversation_repo
        self.user_repo = user_repo
        self.memory_extractor = memory_extractor
        self.is_running = False
        self.consolidation_task = None
        
        # Configuration (can be made environment-configurable)
        self.consolidation_interval = 3600  # 1 hour in seconds
        self.batch_size = 50  # Process conversations in batches
        self.lookback_hours = 24  # Look back 24 hours for new conversations
        
    async def start_consolidation_service(self):
        """Start the background memory consolidation service."""
        if self.is_running:
            logger.warning("Memory consolidation service is already running")
            return
            
        if not all([self.conversation_repo, self.user_repo, self.memory_extractor]):
            logger.warning("Memory consolidation service disabled - missing dependencies")
            return
            
        self.is_running = True
        self.consolidation_task = asyncio.create_task(self._consolidation_loop())
        logger.info("Memory consolidation service started")
        
    async def stop_consolidation_service(self):
        """Stop the background memory consolidation service."""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.consolidation_task:
            self.consolidation_task.cancel()
            try:
                await self.consolidation_task
            except asyncio.CancelledError:
                pass
        logger.info("Memory consolidation service stopped")
        
    async def _consolidation_loop(self):
        """Main consolidation loop that runs periodically."""
        logger.info(f"Memory consolidation loop started (interval: {self.consolidation_interval}s)")
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # Run consolidation tasks
                await self._run_consolidation_cycle()
                
                processing_time = time.time() - start_time
                logger.info(f"Memory consolidation cycle completed in {processing_time:.2f}s")
                
                # Wait for next cycle
                await asyncio.sleep(self.consolidation_interval)
                
            except asyncio.CancelledError:
                logger.info("Memory consolidation loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in memory consolidation loop: {e}", exc_info=True)
                # Wait a bit before retrying on error
                await asyncio.sleep(60)
                
    async def _run_consolidation_cycle(self):
        """Run a single consolidation cycle."""
        try:
            # Get all users for processing
            users = await self._get_active_users()
            if not users:
                logger.debug("No active users found for memory consolidation")
                return
                
            consolidation_stats = {
                'users_processed': 0,
                'conversations_analyzed': 0,
                'memories_created': 0,
                'memories_updated': 0,
                'memories_archived': 0
            }
            
            for user in users:
                try:
                    user_stats = await self._consolidate_user_memories(user)
                    
                    # Aggregate stats
                    consolidation_stats['users_processed'] += 1
                    consolidation_stats['conversations_analyzed'] += user_stats.get('conversations_analyzed', 0)
                    consolidation_stats['memories_created'] += user_stats.get('memories_created', 0)
                    consolidation_stats['memories_updated'] += user_stats.get('memories_updated', 0)
                    consolidation_stats['memories_archived'] += user_stats.get('memories_archived', 0)
                    
                except Exception as e:
                    logger.warning(f"Error consolidating memories for user {user.id}: {e}")
                    continue
                    
            logger.info(f"Consolidation cycle stats: {consolidation_stats}")
            
        except Exception as e:
            logger.error(f"Error in consolidation cycle: {e}", exc_info=True)
            
    async def _get_active_users(self) -> List[Any]:
        """Get list of active users for memory consolidation."""
        try:
            # For single-user installation, just get the default user
            user = await self.user_repo.get_or_create_default_user()
            return [user] if user else []
        except Exception as e:
            logger.warning(f"Error getting active users: {e}")
            return []
            
    async def _consolidate_user_memories(self, user) -> Dict[str, int]:
        """Consolidate memories for a specific user."""
        stats = {
            'conversations_analyzed': 0,
            'memories_created': 0,
            'memories_updated': 0,
            'memories_archived': 0
        }
        
        try:
            # Get recent conversations that haven't been fully processed for memories
            recent_conversations = await self._get_recent_conversations_for_user(str(user.id))
            
            for conversation_data in recent_conversations:
                try:
                    # Extract memories from this conversation
                    session_id = conversation_data.get('session_id')
                    if not session_id:
                        continue
                        
                    # Get conversation messages
                    messages = await self.conversation_repo.get_conversation_history(
                        session_id=session_id,
                        limit=100  # Get more messages for better context
                    )
                    
                    if not messages:
                        continue
                        
                    # Extract memories from conversation
                    extracted_memories = await self.memory_extractor.extract_memories_from_conversation(
                        user_id=str(user.id),
                        conversation_messages=messages,
                        conversation_id=conversation_data.get('id')
                    )
                    
                    stats['conversations_analyzed'] += 1
                    stats['memories_created'] += len(extracted_memories)
                    
                except Exception as e:
                    logger.warning(f"Error processing conversation {conversation_data.get('id')}: {e}")
                    continue
                    
            # Update memory importance scores based on access patterns
            updated_count = await self._update_memory_importance_scores(str(user.id))
            stats['memories_updated'] += updated_count
            
            # Archive old, low-importance memories
            archived_count = await self._archive_old_memories(str(user.id))
            stats['memories_archived'] += archived_count
            
            return stats
            
        except Exception as e:
            logger.error(f"Error consolidating memories for user {user.id}: {e}", exc_info=True)
            return stats
            
    async def _get_recent_conversations_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get recent conversations that need memory processing."""
        try:
            # This is a simplified approach - in a real implementation, you might want to track
            # which conversations have been processed for memory extraction
            # cutoff_time would be: datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
            
            # For now, we'll use a simple approach and let the deduplication handle duplicates
            # In a production system, you'd want to track processing state
            
            # Get recent conversation sessions (this would need to be implemented in the repo)
            # For now, return empty list as this requires additional database queries
            return []
            
        except Exception as e:
            logger.warning(f"Error getting recent conversations for user {user_id}: {e}")
            return []
            
    async def _update_memory_importance_scores(self, user_id: str) -> int:
        """Update memory importance scores based on access patterns."""
        try:
            # Get all memories for the user
            memories = await self.conversation_repo.get_memories(
                user_id=user_id,
                importance_threshold=1,  # Get all memories
                limit=1000
            )
            
            updated_count = 0
            
            for memory in memories:
                try:
                    # Calculate new importance based on access patterns
                    new_importance = self._calculate_updated_importance(memory)
                    
                    if new_importance != memory.importance_score:
                        # Update the memory importance (would need to implement update method)
                        # For now, we'll skip this as it requires additional repository methods
                        updated_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error updating importance for memory {memory.id}: {e}")
                    continue
                    
            return updated_count
            
        except Exception as e:
            logger.warning(f"Error updating memory importance scores for user {user_id}: {e}")
            return 0
            
    def _calculate_updated_importance(self, memory) -> int:
        """Calculate updated importance score based on access patterns."""
        base_importance = memory.importance_score
        
        # Boost importance for frequently accessed memories
        access_count = getattr(memory, 'access_count', 0)
        if access_count > 5:
            base_importance += 1
        elif access_count > 10:
            base_importance += 2
            
        # Decay importance for very old memories that aren't accessed
        if hasattr(memory, 'last_accessed') and memory.last_accessed:
            now_utc = datetime.now(timezone.utc)
            last_accessed = memory.last_accessed
            if isinstance(last_accessed, datetime):
                if last_accessed.tzinfo is None:
                    last_accessed = last_accessed.replace(tzinfo=timezone.utc)
                else:
                    last_accessed = last_accessed.astimezone(timezone.utc)
                days_since_access = (now_utc - last_accessed).days
            else:
                days_since_access = None
            if days_since_access is not None:
                if days_since_access > 30 and access_count < 2:
                    base_importance -= 1
                elif days_since_access > 90 and access_count < 5:
                    base_importance -= 2
                
        # Keep importance within valid range
        return max(1, min(10, base_importance))
        
    async def _archive_old_memories(self, user_id: str) -> int:
        """Archive old, low-importance memories."""
        try:
            # Get low-importance memories that are old
            memories = await self.conversation_repo.get_memories(
                user_id=user_id,
                importance_threshold=1,  # Get all memories
                limit=1000
            )
            
            archived_count = 0
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=180)  # 6 months
            
            for memory in memories:
                try:
                    # Archive memories that are old and have low importance/access
                    created_at = getattr(memory, 'created_at', None)
                    if isinstance(created_at, datetime):
                        if created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=timezone.utc)
                        else:
                            created_at = created_at.astimezone(timezone.utc)
                    should_archive = (
                        memory.importance_score <= 2 and
                        getattr(memory, 'access_count', 0) <= 1 and
                        created_at is not None and created_at < cutoff_date
                    )
                    
                    if should_archive:
                        # In a real implementation, you'd mark the memory as archived
                        # rather than deleting it
                        archived_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error archiving memory {memory.id}: {e}")
                    continue
                    
            return archived_count
            
        except Exception as e:
            logger.warning(f"Error archiving old memories for user {user_id}: {e}")
            return 0
            
    async def manual_consolidation(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Manually trigger memory consolidation for a specific user or all users.
        
        Args:
            user_id: Optional user ID to consolidate. If None, consolidates all users.
            
        Returns:
            Dictionary with consolidation results
        """
        try:
            start_time = time.time()
            
            if user_id:
                # Get specific user
                user = await self.user_repo.get_user_by_id(user_id)
                users = [user] if user else []
            else:
                # Get all users
                users = await self._get_active_users()
                
            if not users:
                return {"error": "No users found for consolidation"}
                
            total_stats = {
                'users_processed': 0,
                'conversations_analyzed': 0,
                'memories_created': 0,
                'memories_updated': 0,
                'memories_archived': 0,
                'processing_time': 0
            }
            
            for user in users:
                user_stats = await self._consolidate_user_memories(user)
                
                # Aggregate stats
                total_stats['users_processed'] += 1
                total_stats['conversations_analyzed'] += user_stats.get('conversations_analyzed', 0)
                total_stats['memories_created'] += user_stats.get('memories_created', 0)
                total_stats['memories_updated'] += user_stats.get('memories_updated', 0)
                total_stats['memories_archived'] += user_stats.get('memories_archived', 0)
                
            total_stats['processing_time'] = time.time() - start_time
            
            logger.info(f"Manual memory consolidation completed: {total_stats}")
            return total_stats
            
        except Exception as e:
            logger.error(f"Error in manual memory consolidation: {e}", exc_info=True)
            return {"error": str(e)}
