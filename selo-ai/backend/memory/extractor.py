"""
Memory Extraction Service

This service analyzes conversations and extracts meaningful memories with importance scoring.
Includes deduplication logic to prevent redundant memory creation.
"""

import re
import hashlib
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger("selo.memory.extractor")

class MemoryExtractor:
    """
    Extracts and manages memories from conversations with deduplication.
    """
    
    def __init__(self, conversation_repo=None, llm_controller=None):
        """
        Initialize the MemoryExtractor.
        
        Args:
            conversation_repo: Repository for conversation and memory operations
            llm_controller: LLM controller for semantic analysis (optional)
        """
        self.conversation_repo = conversation_repo
        self.llm_controller = llm_controller
        
    async def extract_memories_from_conversation(self, 
                                               user_id: str,
                                               conversation_messages: List[Dict[str, Any]],
                                               conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract memories from a conversation with deduplication.
        
        Args:
            user_id: User ID for memory ownership
            conversation_messages: List of conversation messages
            conversation_id: Optional conversation ID for source tracking
            
        Returns:
            List of created memory dictionaries
        """
        if not conversation_messages:
            return []
            
        try:
            # Extract potential memories using pattern matching
            potential_memories = await self._extract_potential_memories(conversation_messages)
            
            # Score importance for each potential memory
            scored_memories = await self._score_memory_importance(potential_memories, conversation_messages)
            
            # Filter by importance threshold (>=3)
            significant_memories = [mem for mem in scored_memories if mem.get('importance_score', 0) >= 3]
            
            if not significant_memories:
                logger.debug(f"No significant memories found in conversation for user {user_id}")
                return []
            
            # Deduplicate against existing memories
            deduplicated_memories = await self._deduplicate_memories(user_id, significant_memories)
            
            if not deduplicated_memories:
                logger.debug(f"All potential memories were duplicates for user {user_id}")
                return []
            
            # Create memory objects in database
            created_memories = []
            for memory_data in deduplicated_memories:
                try:
                    memory = await self.conversation_repo.create_memory(
                        user_id=user_id,
                        memory_type=memory_data.get('type', 'conversation'),
                        content=memory_data['content'],
                        importance_score=memory_data['importance_score'],
                        confidence_score=memory_data.get('confidence_score', 7),
                        source_conversation_id=conversation_id,
                        tags=memory_data.get('tags', [])
                    )
                    created_memories.append({
                        'id': str(memory.id),
                        'content': memory.content,
                        'type': memory.memory_type,
                        'importance': memory.importance_score,
                        'confidence': memory.confidence_score
                    })
                except Exception as e:
                    logger.warning(f"Failed to create memory: {e}")
                    continue
            
            logger.info(f"Created {len(created_memories)} new memories for user {user_id}")
            return created_memories
            
        except Exception as e:
            logger.error(f"Error extracting memories from conversation: {e}", exc_info=True)
            return []
    
    async def _extract_potential_memories(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract potential memories using pattern matching and heuristics.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            List of potential memory dictionaries
        """
        potential_memories = []
        
        # Combine user messages for analysis
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        assistant_messages = [msg for msg in messages if msg.get('role') == 'assistant']
        
        # Extract different types of memories
        potential_memories.extend(self._extract_personal_info(user_messages))
        potential_memories.extend(self._extract_preferences(user_messages, assistant_messages))
        potential_memories.extend(self._extract_goals_and_tasks(user_messages))
        potential_memories.extend(self._extract_context_and_topics(user_messages))
        potential_memories.extend(self._extract_emotional_context(user_messages, assistant_messages))
        
        return potential_memories
    
    def _extract_personal_info(self, user_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract personal information from user messages."""
        memories = []
        
        # Patterns for personal information
        personal_patterns = [
            (r'\bmy name is (\w+)', 'personal_info', 'User name: {}', 8),
            (r'\bi am (\d+) years old', 'personal_info', 'User age: {}', 7),
            (r'\bi work (?:as|at) ([^.!?]+)', 'personal_info', 'User occupation: {}', 7),
            (r'\bi live in ([^.!?]+)', 'personal_info', 'User location: {}', 6),
            (r'\bi have (?:a |an )?([^.!?]+) degree', 'personal_info', 'User education: {} degree', 6),
            (r'\bmy (?:hobby|hobbies) (?:is|are) ([^.!?]+)', 'personal_info', 'User hobbies: {}', 5),
        ]
        
        for msg in user_messages:
            content = msg.get('content', '')
            for pattern, mem_type, template, importance in personal_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    memories.append({
                        'content': template.format(match.strip()),
                        'type': mem_type,
                        'base_importance': importance,
                        'tags': ['personal', 'user_info']
                    })
        
        return memories
    
    def _extract_preferences(self, user_messages: List[Dict[str, Any]], assistant_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract user preferences and likes/dislikes."""
        memories = []
        
        # Preference patterns
        preference_patterns = [
            (r'\bi (?:love|like|enjoy|prefer) ([^.!?]+)', 'preference', 'User likes: {}', 5),
            (r'\bi (?:hate|dislike|don\'t like) ([^.!?]+)', 'preference', 'User dislikes: {}', 5),
            (r'\bmy favorite ([^.!?]+) is ([^.!?]+)', 'preference', 'User\'s favorite {}: {}', 6),
            (r'\bi usually ([^.!?]+)', 'preference', 'User habit: {}', 4),
            (r'\bi always ([^.!?]+)', 'preference', 'User always: {}', 5),
            (r'\bi never ([^.!?]+)', 'preference', 'User never: {}', 5),
        ]
        
        for msg in user_messages:
            content = msg.get('content', '')
            for pattern, mem_type, template, importance in preference_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        memories.append({
                            'content': template.format(*match),
                            'type': mem_type,
                            'base_importance': importance,
                            'tags': ['preference', 'user_behavior']
                        })
                    else:
                        memories.append({
                            'content': template.format(match.strip()),
                            'type': mem_type,
                            'base_importance': importance,
                            'tags': ['preference', 'user_behavior']
                        })
        
        return memories
    
    def _extract_goals_and_tasks(self, user_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract user goals, tasks, and objectives."""
        memories = []
        
        # Goal and task patterns
        goal_patterns = [
            (r'\bi want to ([^.!?]+)', 'goal', 'User goal: {}', 6),
            (r'\bi need to ([^.!?]+)', 'task', 'User task: {}', 7),
            (r'\bi\'m trying to ([^.!?]+)', 'goal', 'User attempting: {}', 6),
            (r'\bmy goal is to ([^.!?]+)', 'goal', 'User goal: {}', 7),
            (r'\bi plan to ([^.!?]+)', 'goal', 'User plans: {}', 5),
            (r'\bi\'m working on ([^.!?]+)', 'task', 'User working on: {}', 6),
        ]
        
        for msg in user_messages:
            content = msg.get('content', '')
            for pattern, mem_type, template, importance in goal_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    memories.append({
                        'content': template.format(match.strip()),
                        'type': mem_type,
                        'base_importance': importance,
                        'tags': ['goal', 'task', 'objective']
                    })
        
        return memories
    
    def _extract_context_and_topics(self, user_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract contextual information and topics of interest."""
        memories = []
        
        # Extract topics mentioned frequently
        all_content = ' '.join([msg.get('content', '') for msg in user_messages])
        
        # Technical topics
        tech_patterns = [
            r'\b(python|javascript|react|node|sql|api|database|machine learning|ai|artificial intelligence)\b',
            r'\b(programming|coding|development|software|web development|data science)\b',
            r'\b(github|git|docker|kubernetes|aws|cloud|server|backend|frontend)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, all_content, re.IGNORECASE)
            unique_matches = list(set([match.lower() for match in matches]))
            for match in unique_matches:
                if len([msg for msg in user_messages if match in msg.get('content', '').lower()]) >= 2:
                    memories.append({
                        'content': f'User shows interest in: {match}',
                        'type': 'interest',
                        'base_importance': 4,
                        'tags': ['topic', 'interest', 'technology']
                    })
        
        return memories
    
    def _extract_emotional_context(self, user_messages: List[Dict[str, Any]], assistant_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract emotional context and sentiment."""
        memories = []
        
        # Emotional expressions
        emotion_patterns = [
            (r'\bi (?:feel|am feeling) ([^.!?]+)', 'emotion', 'User felt: {}', 5),
            (r'\bi\'m (?:excited|happy|sad|frustrated|angry|worried) about ([^.!?]+)', 'emotion', 'User emotional about: {}', 6),
            (r'\bthat makes me ([^.!?]+)', 'emotion', 'User reaction: {}', 4),
        ]
        
        for msg in user_messages:
            content = msg.get('content', '')
            for pattern, mem_type, template, importance in emotion_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    memories.append({
                        'content': template.format(match.strip()),
                        'type': mem_type,
                        'base_importance': importance,
                        'tags': ['emotion', 'sentiment', 'user_state']
                    })
        
        return memories
    
    async def _score_memory_importance(self, potential_memories: List[Dict[str, Any]], 
                                     conversation_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score the importance of potential memories.
        
        Args:
            potential_memories: List of potential memory dictionaries
            conversation_messages: Full conversation context
            
        Returns:
            List of memories with importance scores
        """
        scored_memories = []
        
        for memory in potential_memories:
            base_importance = memory.get('base_importance', 3)
            
            # Adjust importance based on context
            importance_score = base_importance
            
            # Boost importance for repeated mentions
            content_lower = memory['content'].lower()
            mention_count = sum(1 for msg in conversation_messages 
                              if any(word in msg.get('content', '').lower() 
                                   for word in content_lower.split()[:3]))
            if mention_count > 1:
                importance_score += min(2, mention_count - 1)
            
            # Boost importance for personal information
            if memory.get('type') == 'personal_info':
                importance_score += 1
            
            # Boost importance for goals and tasks
            if memory.get('type') in ['goal', 'task']:
                importance_score += 1
            
            # Cap importance at 10
            importance_score = min(10, importance_score)
            
            # Calculate confidence score (7-9 range for pattern-based extraction)
            confidence_score = 7
            if memory.get('type') == 'personal_info':
                confidence_score = 8
            elif mention_count > 1:
                confidence_score = 8
            
            scored_memories.append({
                **memory,
                'importance_score': importance_score,
                'confidence_score': confidence_score
            })
        
        return scored_memories
    
    async def _deduplicate_memories(self, user_id: str, new_memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate memories against existing memories for the user.
        
        Args:
            user_id: User ID to check against
            new_memories: List of new memory candidates
            
        Returns:
            List of deduplicated memories
        """
        if not self.conversation_repo:
            return new_memories
        
        try:
            # Get existing memories for the user
            existing_memories = await self.conversation_repo.get_memories(
                user_id=user_id,
                importance_threshold=1,  # Get all memories for deduplication
                limit=1000
            )
            
            # Create content hashes for existing memories
            existing_hashes = set()
            existing_contents = set()
            
            for existing_memory in existing_memories:
                # Hash-based deduplication
                content_hash = self._generate_content_hash(existing_memory.content)
                existing_hashes.add(content_hash)
                
                # Content similarity deduplication
                existing_contents.add(existing_memory.content.lower().strip())
            
            # Filter out duplicates
            deduplicated = []
            for memory in new_memories:
                content = memory['content']
                content_hash = self._generate_content_hash(content)
                content_lower = content.lower().strip()
                
                # Check for exact hash match
                if content_hash in existing_hashes:
                    logger.debug(f"Skipping duplicate memory (hash): {content[:50]}...")
                    continue
                
                # Check for similar content
                if content_lower in existing_contents:
                    logger.debug(f"Skipping duplicate memory (content): {content[:50]}...")
                    continue
                
                # Check for semantic similarity (simple keyword overlap)
                is_similar = False
                for existing_content in existing_contents:
                    if self._calculate_content_similarity(content_lower, existing_content) > 0.8:
                        logger.debug(f"Skipping similar memory: {content[:50]}...")
                        is_similar = True
                        break
                
                if not is_similar:
                    deduplicated.append(memory)
            
            logger.info(f"Deduplicated {len(new_memories)} -> {len(deduplicated)} memories for user {user_id}")
            return deduplicated
            
        except Exception as e:
            logger.warning(f"Error during memory deduplication: {e}")
            # Return original memories if deduplication fails
            return new_memories
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate a hash for memory content."""
        # Normalize content for hashing
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        # Simple word overlap similarity
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def extract_memory_from_single_message(self, 
                                               user_id: str,
                                               message: Dict[str, Any],
                                               conversation_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Extract memory from a single high-value message.
        
        Args:
            user_id: User ID for memory ownership
            message: Single message dictionary
            conversation_id: Optional conversation ID for source tracking
            
        Returns:
            Created memory dictionary or None
        """
        if message.get('role') != 'user':
            return None
        
        content = message.get('content', '')
        if len(content.strip()) < 10:  # Skip very short messages
            return None
        
        # Quick importance check
        high_value_indicators = [
            r'\bmy name is\b', r'\bi work\b', r'\bi live\b', r'\bi want to\b', r'\bi need to\b',
            r'\bmy goal\b', r'\bi\'m trying to\b', r'\bi love\b', r'\bi hate\b', r'\bmy favorite\b'
        ]
        
        has_high_value = any(re.search(pattern, content, re.IGNORECASE) for pattern in high_value_indicators)
        
        if not has_high_value:
            return None
        
        # Extract memories from this single message
        memories = await self.extract_memories_from_conversation(
            user_id=user_id,
            conversation_messages=[message],
            conversation_id=conversation_id
        )
        
        return memories[0] if memories else None
