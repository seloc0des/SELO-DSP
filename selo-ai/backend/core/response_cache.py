"""
Response Caching System for LLM Performance Optimization

Implements intelligent caching of LLM responses to reduce computation time
for similar requests without compromising content generation quality.
"""

import hashlib
import json
import logging
import time
from typing import Dict, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass, asdict

logger = logging.getLogger("selo.core.response_cache")

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    response: str
    timestamp: float
    model: str
    prompt_hash: str
    token_count: int
    generation_time: float
    hit_count: int = 0
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if cache entry is expired."""
        return time.time() - self.timestamp > ttl_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

class ResponseCache:
    """
    Intelligent LLM response cache with semantic similarity and performance optimization.
    
    Features:
    - Semantic prompt similarity detection
    - TTL-based expiration
    - Memory usage limits
    - Performance metrics
    - Model-specific caching
    """
    
    def __init__(self, 
                 max_entries: int = 1000,
                 default_ttl: int = 3600,  # 1 hour
                 similarity_threshold: float = 0.85):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_time_saved": 0.0
        }
        self._lock = asyncio.Lock()
        
        logger.info(f"Response cache initialized: max_entries={max_entries}, ttl={default_ttl}s")
    
    def _generate_prompt_hash(self, prompt: str, model: str, context: Dict[str, Any] = None) -> str:
        """Generate a hash for the prompt and context."""
        # Normalize prompt (remove extra whitespace, lowercase for similarity)
        normalized_prompt = " ".join(prompt.lower().split())
        
        # Include relevant context in hash
        context_str = ""
        if context:
            # Only include stable context elements that affect response
            stable_context = {
                k: v for k, v in context.items() 
                if k in ["temperature", "max_tokens", "system_prompt", "persona_id"]
            }
            context_str = json.dumps(stable_context, sort_keys=True)
        
        hash_input = f"{model}:{normalized_prompt}:{context_str}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _calculate_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate semantic similarity between prompts (simple implementation)."""
        # Simple word-based similarity - could be enhanced with embeddings
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def get(self, prompt: str, model: str, context: Dict[str, Any] = None) -> Optional[str]:
        """
        Get cached response if available and valid.
        
        Args:
            prompt: The input prompt
            model: Model name
            context: Additional context for caching
            
        Returns:
            Cached response if found, None otherwise
        """
        async with self._lock:
            prompt_hash = self._generate_prompt_hash(prompt, model, context)
            
            # Direct hash match
            if prompt_hash in self.cache:
                entry = self.cache[prompt_hash]
                if not entry.is_expired(self.default_ttl):
                    entry.hit_count += 1
                    self.stats["hits"] += 1
                    self.stats["total_time_saved"] += entry.generation_time
                    logger.debug(f"Cache HIT: {prompt_hash[:8]}... (saved {entry.generation_time:.1f}s)")
                    return entry.response
                else:
                    # Remove expired entry
                    del self.cache[prompt_hash]
            
            # Check for similar prompts (semantic similarity)
            for cached_hash, entry in list(self.cache.items()):
                if entry.model == model and not entry.is_expired(self.default_ttl):
                    # Reconstruct original prompt for similarity check (simplified)
                    similarity = self._calculate_similarity(prompt, entry.response[:200])  # Use response prefix as proxy
                    
                    if similarity >= self.similarity_threshold:
                        entry.hit_count += 1
                        self.stats["hits"] += 1
                        self.stats["total_time_saved"] += entry.generation_time * 0.8  # Partial time saving
                        logger.debug(f"Cache SIMILAR HIT: {cached_hash[:8]}... (similarity: {similarity:.2f})")
                        return entry.response
            
            self.stats["misses"] += 1
            return None
    
    async def put(self, prompt: str, model: str, response: str, 
                  generation_time: float, context: Dict[str, Any] = None):
        """
        Store response in cache.
        
        Args:
            prompt: The input prompt
            model: Model name
            response: Generated response
            generation_time: Time taken to generate response
            context: Additional context
        """
        async with self._lock:
            prompt_hash = self._generate_prompt_hash(prompt, model, context)
            
            # Evict old entries if at capacity
            if len(self.cache) >= self.max_entries:
                await self._evict_oldest()
            
            # Store new entry
            entry = CacheEntry(
                response=response,
                timestamp=time.time(),
                model=model,
                prompt_hash=prompt_hash,
                token_count=len(response.split()),
                generation_time=generation_time
            )
            
            self.cache[prompt_hash] = entry
            logger.debug(f"Cache STORE: {prompt_hash[:8]}... ({generation_time:.1f}s, {entry.token_count} tokens)")
    
    async def _evict_oldest(self):
        """Evict oldest cache entries."""
        if not self.cache:
            return
        
        # Sort by timestamp and remove oldest 10%
        sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].timestamp)
        evict_count = max(1, len(sorted_entries) // 10)
        
        for i in range(evict_count):
            hash_key, _ = sorted_entries[i]
            del self.cache[hash_key]
            self.stats["evictions"] += 1
        
        logger.debug(f"Cache evicted {evict_count} oldest entries")
    
    async def clear_expired(self):
        """Remove all expired entries."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired(self.default_ttl)
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                logger.info(f"Cleared {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_size": len(self.cache),
            "max_entries": self.max_entries,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "time_saved_seconds": self.stats["total_time_saved"],
            "evictions": self.stats["evictions"],
            **self.stats
        }
    
    async def warm_cache(self, common_prompts: list[Tuple[str, str]]):
        """Pre-warm cache with common prompt-response pairs."""
        logger.info(f"Warming cache with {len(common_prompts)} common prompts")
        
        for prompt, expected_response in common_prompts:
            # This would typically involve actual LLM calls, but for warming
            # we can use pre-computed responses
            await self.put(prompt, "default", expected_response, 0.1)

# Global cache instance
_response_cache: Optional[ResponseCache] = None

def get_response_cache() -> ResponseCache:
    """Get global response cache instance."""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache

async def clear_cache():
    """Clear the global cache."""
    if _response_cache:
        _response_cache.cache.clear()
        logger.info("Response cache cleared")
