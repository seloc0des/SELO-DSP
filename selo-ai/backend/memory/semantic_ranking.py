"""
Semantic Memory Ranking System

Ranks and selects memories based on semantic relevance, recency,
importance, and available token budget.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math

logger = logging.getLogger("selo.memory.semantic_ranking")


@dataclass
class MemoryScore:
    """Score components for a memory."""
    memory_id: str
    semantic_similarity: float  # 0.0-1.0 (cosine similarity)
    recency_score: float  # 0.0-1.0 (newer = higher)
    importance_score: float  # 0.0-1.0 (user-defined or inferred)
    access_frequency: float  # 0.0-1.0 (normalized access count)
    emotional_relevance: float  # 0.0-1.0 (emotional state alignment)
    composite_score: float = 0.0
    estimated_tokens: int = 0
    
    def __post_init__(self):
        """Calculate composite score."""
        if self.composite_score == 0.0:
            self.composite_score = self.calculate_composite()
    
    def calculate_composite(self) -> float:
        """
        Calculate composite relevance score.
        
        Weights:
        - Semantic similarity: 40%
        - Recency: 20%
        - Importance: 20%
        - Access frequency: 10%
        - Emotional relevance: 10%
        """
        return (
            self.semantic_similarity * 0.40 +
            self.recency_score * 0.20 +
            self.importance_score * 0.20 +
            self.access_frequency * 0.10 +
            self.emotional_relevance * 0.10
        )


@dataclass
class RankedMemorySet:
    """A set of ranked memories that fit within token budget."""
    memories: List[Dict[str, Any]]
    scores: List[MemoryScore]
    total_tokens: int
    budget_tokens: int
    utilization: float
    excluded_count: int
    ranking_metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticMemoryRanker:
    """
    Ranks memories by semantic relevance and selects optimal subset
    within token budget.
    """
    
    def __init__(self):
        """Initialize the ranker."""
        self.recency_decay_days = 30  # Half-life for recency score
        self.min_similarity_threshold = 0.1  # Minimum similarity to consider
    
    def estimate_memory_tokens(self, memory: Dict[str, Any]) -> int:
        """
        Estimate token count for a memory.
        
        Args:
            memory: Memory dictionary
            
        Returns:
            Estimated token count
        """
        content = memory.get("content", "") or memory.get("text", "")
        metadata_str = str(memory.get("metadata", {}))
        
        # Rough estimate: ~4 chars per token
        total_chars = len(content) + len(metadata_str)
        return total_chars // 4
    
    def calculate_recency_score(self, timestamp: datetime) -> float:
        """
        Calculate recency score using exponential decay.
        
        Args:
            timestamp: Memory creation/update timestamp
            
        Returns:
            Recency score 0.0-1.0
        """
        try:
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            
            age_days = (datetime.now(timezone.utc) - timestamp).days
            
            # Exponential decay with half-life
            decay_rate = math.log(2) / self.recency_decay_days
            score = math.exp(-decay_rate * age_days)
            
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            logger.warning(f"Error calculating recency score: {e}")
            return 0.5  # Default
    
    def calculate_access_frequency_score(
        self,
        access_count: int,
        max_access_count: int
    ) -> float:
        """
        Calculate normalized access frequency score.
        
        Args:
            access_count: Number of times memory was accessed
            max_access_count: Maximum access count in dataset
            
        Returns:
            Normalized score 0.0-1.0
        """
        if max_access_count == 0:
            return 0.0
        
        # Logarithmic scaling to prevent over-weighting frequent memories
        normalized = math.log1p(access_count) / math.log1p(max_access_count)
        return min(normalized, 1.0)
    
    def calculate_emotional_relevance(
        self,
        memory_emotion: Optional[Dict[str, Any]],
        current_emotion: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate emotional relevance between memory and current state.
        
        Args:
            memory_emotion: Emotional state when memory was created
            current_emotion: Current emotional state
            
        Returns:
            Relevance score 0.0-1.0
        """
        if not memory_emotion or not current_emotion:
            return 0.5  # Neutral if no emotional data
        
        try:
            # Extract primary emotions
            mem_primary = memory_emotion.get("primary", "")
            curr_primary = current_emotion.get("primary", "")
            
            # Exact match
            if mem_primary == curr_primary:
                return 1.0
            
            # Related emotions (simple heuristic)
            emotion_groups = {
                "positive": {"joy", "excitement", "contentment", "hope", "gratitude"},
                "negative": {"sadness", "anger", "fear", "anxiety", "frustration"},
                "neutral": {"calm", "focused", "curious", "contemplative"}
            }
            
            for group, emotions in emotion_groups.items():
                if mem_primary in emotions and curr_primary in emotions:
                    return 0.7  # Related emotions
            
            return 0.3  # Different emotion types
            
        except Exception as e:
            logger.warning(f"Error calculating emotional relevance: {e}")
            return 0.5
    
    def score_memory(
        self,
        memory: Dict[str, Any],
        query_embedding: Optional[List[float]],
        current_emotion: Optional[Dict[str, Any]],
        max_access_count: int
    ) -> MemoryScore:
        """
        Calculate comprehensive score for a memory.
        
        Args:
            memory: Memory dictionary
            query_embedding: Query embedding for semantic similarity
            current_emotion: Current emotional state
            max_access_count: Max access count in dataset for normalization
            
        Returns:
            MemoryScore object
        """
        memory_id = memory.get("id", "")
        
        # Semantic similarity
        if query_embedding and "embedding" in memory:
            # Cosine similarity (assuming embeddings are normalized)
            mem_embedding = memory["embedding"]
            if len(query_embedding) == len(mem_embedding):
                similarity = sum(a * b for a, b in zip(query_embedding, mem_embedding))
                semantic_similarity = max(min(similarity, 1.0), 0.0)
            else:
                semantic_similarity = 0.5  # Fallback
        else:
            semantic_similarity = 0.5  # Default when no embeddings
        
        # Recency
        timestamp_str = memory.get("created_at") or memory.get("timestamp")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                recency_score = self.calculate_recency_score(timestamp)
            except (ValueError, AttributeError):
                recency_score = 0.5
        else:
            recency_score = 0.5
        
        # Importance (user-defined or inferred from metadata)
        importance_score = memory.get("importance", 0.5)
        if "metadata" in memory:
            meta_importance = memory["metadata"].get("importance")
            if meta_importance is not None:
                importance_score = float(meta_importance)
        
        # Access frequency
        access_count = memory.get("access_count", 0)
        access_frequency = self.calculate_access_frequency_score(
            access_count,
            max_access_count
        )
        
        # Emotional relevance
        memory_emotion = memory.get("emotional_context") or memory.get("emotion")
        emotional_relevance = self.calculate_emotional_relevance(
            memory_emotion,
            current_emotion
        )
        
        # Estimate tokens
        estimated_tokens = self.estimate_memory_tokens(memory)
        
        return MemoryScore(
            memory_id=memory_id,
            semantic_similarity=semantic_similarity,
            recency_score=recency_score,
            importance_score=importance_score,
            access_frequency=access_frequency,
            emotional_relevance=emotional_relevance,
            estimated_tokens=estimated_tokens
        )
    
    def rank_and_select(
        self,
        memories: List[Dict[str, Any]],
        query_embedding: Optional[List[float]] = None,
        current_emotion: Optional[Dict[str, Any]] = None,
        token_budget: int = 1000,
        min_memories: int = 3,
        max_memories: int = 20
    ) -> RankedMemorySet:
        """
        Rank memories and select optimal subset within token budget.
        
        Args:
            memories: List of memory dictionaries
            query_embedding: Optional query embedding for semantic ranking
            current_emotion: Optional current emotional state
            token_budget: Maximum tokens to allocate to memories
            min_memories: Minimum number of memories to include (if available)
            max_memories: Maximum number of memories to include
            
        Returns:
            RankedMemorySet with selected memories
        """
        if not memories:
            return RankedMemorySet(
                memories=[],
                scores=[],
                total_tokens=0,
                budget_tokens=token_budget,
                utilization=0.0,
                excluded_count=0
            )
        
        # Calculate max access count for normalization
        max_access_count = max(
            (m.get("access_count", 0) for m in memories),
            default=1
        )
        
        # Score all memories
        scores = [
            self.score_memory(memory, query_embedding, current_emotion, max_access_count)
            for memory in memories
        ]
        
        # Filter by minimum similarity threshold
        filtered_pairs = [
            (memory, score) for memory, score in zip(memories, scores)
            if score.semantic_similarity >= self.min_similarity_threshold
        ]
        
        if not filtered_pairs:
            # No memories meet threshold, return empty
            return RankedMemorySet(
                memories=[],
                scores=[],
                total_tokens=0,
                budget_tokens=token_budget,
                utilization=0.0,
                excluded_count=len(memories)
            )
        
        # Sort by composite score (descending)
        filtered_pairs.sort(key=lambda x: x[1].composite_score, reverse=True)
        
        # Select memories within token budget using greedy approach
        selected_memories = []
        selected_scores = []
        accumulated_tokens = 0
        
        for memory, score in filtered_pairs:
            # Check if adding this memory exceeds budget
            if accumulated_tokens + score.estimated_tokens <= token_budget:
                selected_memories.append(memory)
                selected_scores.append(score)
                accumulated_tokens += score.estimated_tokens
                
                # Stop if we hit max_memories
                if len(selected_memories) >= max_memories:
                    break
        
        # Ensure minimum memories if available and within budget
        if len(selected_memories) < min_memories:
            # Try to add more memories even if budget tight
            for memory, score in filtered_pairs[len(selected_memories):]:
                if len(selected_memories) >= min_memories:
                    break
                # Be more lenient about budget for minimum
                if len(selected_memories) < min_memories:
                    selected_memories.append(memory)
                    selected_scores.append(score)
                    accumulated_tokens += score.estimated_tokens
        
        utilization = accumulated_tokens / token_budget if token_budget > 0 else 0.0
        excluded_count = len(memories) - len(selected_memories)
        
        logger.info(
            f"Ranked {len(memories)} memories, selected {len(selected_memories)} "
            f"({accumulated_tokens}/{token_budget} tokens, {utilization:.1%} utilization), "
            f"excluded {excluded_count}"
        )
        
        return RankedMemorySet(
            memories=selected_memories,
            scores=selected_scores,
            total_tokens=accumulated_tokens,
            budget_tokens=token_budget,
            utilization=utilization,
            excluded_count=excluded_count,
            ranking_metadata={
                "query_embedding_used": query_embedding is not None,
                "emotion_aware": current_emotion is not None,
                "avg_score": sum(s.composite_score for s in selected_scores) / len(selected_scores) if selected_scores else 0.0,
                "min_score": min((s.composite_score for s in selected_scores), default=0.0),
                "max_score": max((s.composite_score for s in selected_scores), default=0.0)
            }
        )
    
    def explain_ranking(self, score: MemoryScore) -> str:
        """
        Generate human-readable explanation of memory ranking.
        
        Args:
            score: MemoryScore to explain
            
        Returns:
            Explanation string
        """
        parts = [
            f"Composite: {score.composite_score:.2f}",
            f"Semantic: {score.semantic_similarity:.2f}",
            f"Recency: {score.recency_score:.2f}",
            f"Importance: {score.importance_score:.2f}",
            f"Access: {score.access_frequency:.2f}",
            f"Emotion: {score.emotional_relevance:.2f}",
            f"Tokens: {score.estimated_tokens}"
        ]
        return " | ".join(parts)


# Global ranker instance
_global_ranker: Optional[SemanticMemoryRanker] = None


def get_memory_ranker() -> SemanticMemoryRanker:
    """Get or create the global memory ranker instance."""
    global _global_ranker
    if _global_ranker is None:
        _global_ranker = SemanticMemoryRanker()
    return _global_ranker


def rank_memories(
    memories: List[Dict[str, Any]],
    query_embedding: Optional[List[float]] = None,
    current_emotion: Optional[Dict[str, Any]] = None,
    token_budget: int = 1000
) -> RankedMemorySet:
    """
    Convenience function to rank and select memories.
    
    Args:
        memories: List of memory dictionaries
        query_embedding: Optional query embedding
        current_emotion: Optional current emotional state
        token_budget: Maximum tokens to allocate
        
    Returns:
        RankedMemorySet with selected memories
    """
    ranker = get_memory_ranker()
    return ranker.rank_and_select(
        memories=memories,
        query_embedding=query_embedding,
        current_emotion=current_emotion,
        token_budget=token_budget
    )
