"""Emotion vector index service for fast similarity search and optimization."""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import json

logger = logging.getLogger("selo.agent.emotion_index")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, falling back to numpy similarity search")


class EmotionIndexService:
    """Manages emotion vector indexing and similarity search for optimization."""

    def __init__(self, dimension: int = 2048, affective_state_repo=None):
        """
        Initialize the emotion index service.
        
        Args:
            dimension: Embedding dimension (default 2048 for nomic-embed-text)
            affective_state_repo: Repository for fetching historical affective states
        """
        self.dimension = dimension
        self._affective_state_repo = affective_state_repo
        self._index = None
        self._vectors: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []
        self._persona_id: Optional[str] = None
        
        # Emotional cluster definitions (can be refined with actual clustering)
        self._emotional_clusters = {
            "stressed": {
                "adjustments": {"energy": -0.1, "stress": 0.0, "confidence": -0.05},
                "keywords": ["anxious", "worried", "stressed", "overwhelmed"]
            },
            "joyful": {
                "adjustments": {"energy": 0.15, "stress": -0.1, "confidence": 0.1},
                "keywords": ["joy", "happy", "excited", "delighted"]
            },
            "calm": {
                "adjustments": {"energy": 0.0, "stress": -0.15, "confidence": 0.05},
                "keywords": ["calm", "peaceful", "serene", "content"]
            },
            "angry": {
                "adjustments": {"energy": 0.1, "stress": 0.15, "confidence": 0.0},
                "keywords": ["angry", "furious", "rage", "irritated"]
            },
            "sad": {
                "adjustments": {"energy": -0.15, "stress": 0.05, "confidence": -0.1},
                "keywords": ["sad", "depressed", "melancholy", "grief"]
            },
            "neutral": {
                "adjustments": {"energy": 0.0, "stress": 0.0, "confidence": 0.0},
                "keywords": ["neutral", "balanced", "stable"]
            }
        }

    async def initialize_index(self, persona_id: str, lookback_limit: int = 100) -> None:
        """
        Initialize or rebuild the FAISS index from historical affective states.
        
        Args:
            persona_id: Persona to build index for
            lookback_limit: Number of historical states to index
        """
        self._persona_id = persona_id
        
        if not self._affective_state_repo:
            logger.warning("No affective state repo available, index will be empty")
            return
        
        try:
            states = await self._affective_state_repo.get_state_history(
                persona_id=persona_id,
                limit=lookback_limit
            )
            
            vectors = []
            metadata = []
            
            for state in states:
                state_meta = state.get("state_metadata") or {}
                cache = state_meta.get("emotion_vector_cache")
                
                if not cache or not cache.get("vector"):
                    continue
                
                vec = cache.get("vector")
                if not isinstance(vec, list) or len(vec) != self.dimension:
                    continue
                
                vectors.append(np.array(vec, dtype=np.float32))
                metadata.append({
                    "state_id": state.get("id"),
                    "signature": cache.get("signature"),
                    "timestamp": state.get("last_update"),
                    "core_emotions": state.get("core_emotions"),
                    "dominant_emotion": state.get("dominant_emotion"),
                    "energy": state.get("energy"),
                    "stress": state.get("stress"),
                    "confidence": state.get("confidence")
                })
            
            if not vectors:
                logger.info("No emotion vectors found in history for persona %s", persona_id)
                return
            
            self._vectors = vectors
            self._metadata = metadata
            
            if FAISS_AVAILABLE and len(vectors) > 0:
                vector_matrix = np.vstack(vectors).astype(np.float32)
                self._index = faiss.IndexFlatIP(self.dimension)
                faiss.normalize_L2(vector_matrix)
                self._index.add(vector_matrix)
                logger.info("Built FAISS index with %d emotion vectors for persona %s", 
                           len(vectors), persona_id)
            else:
                logger.info("Using numpy fallback for %d emotion vectors", len(vectors))
                
        except Exception as exc:
            logger.error("Failed to initialize emotion index: %s", exc, exc_info=True)

    def add_vector(
        self,
        vector: List[float],
        signature: str,
        state_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        **extra_metadata
    ) -> None:
        """
        Add a new emotion vector to the index.
        
        Args:
            vector: Emotion embedding vector
            signature: Human-readable emotion signature
            state_id: Optional affective state ID
            timestamp: Optional timestamp
            **extra_metadata: Additional metadata to store
        """
        if len(vector) != self.dimension:
            logger.warning("Vector dimension mismatch: expected %d, got %d", 
                          self.dimension, len(vector))
            return
        
        vec_array = np.array(vector, dtype=np.float32)
        self._vectors.append(vec_array)
        
        meta = {
            "state_id": state_id,
            "signature": signature,
            "timestamp": timestamp or datetime.now(timezone.utc),
            **extra_metadata
        }
        self._metadata.append(meta)
        
        if FAISS_AVAILABLE and self._index is not None:
            vec_normalized = vec_array.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(vec_normalized)
            self._index.add(vec_normalized)

    def find_similar_states(
        self,
        query_vector: List[float],
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find similar emotional states using vector similarity.
        
        Args:
            query_vector: Query emotion vector
            top_k: Number of similar states to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of similar states with metadata and similarity scores
        """
        if not self._vectors:
            return []
        
        if len(query_vector) != self.dimension:
            logger.warning("Query vector dimension mismatch")
            return []
        
        query_array = np.array(query_vector, dtype=np.float32)
        
        if FAISS_AVAILABLE and self._index is not None:
            query_normalized = query_array.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_normalized)
            
            k = min(top_k, len(self._vectors))
            distances, indices = self._index.search(query_normalized, k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self._metadata):
                    continue
                if dist < threshold:
                    continue
                
                results.append({
                    "similarity": float(dist),
                    "metadata": self._metadata[idx]
                })
            
            return results
        else:
            query_norm = query_array / (np.linalg.norm(query_array) + 1e-8)
            similarities = []
            
            for i, vec in enumerate(self._vectors):
                vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
                sim = float(np.dot(query_norm, vec_norm))
                
                if sim >= threshold:
                    similarities.append((sim, i))
            
            similarities.sort(reverse=True, key=lambda x: x[0])
            
            results = []
            for sim, idx in similarities[:top_k]:
                results.append({
                    "similarity": sim,
                    "metadata": self._metadata[idx]
                })
            
            return results

    def should_skip_processing(
        self,
        new_vector: List[float],
        last_vector: Optional[List[float]],
        similarity_threshold: float = 0.95
    ) -> bool:
        """
        Determine if processing can be skipped due to similar emotional state.
        
        Args:
            new_vector: Current emotion vector
            last_vector: Previous emotion vector
            similarity_threshold: Minimum similarity to skip (default 0.95)
            
        Returns:
            True if vectors are very similar and processing can be skipped
        """
        if not last_vector or len(new_vector) != len(last_vector):
            return False
        
        try:
            new_arr = np.array(new_vector, dtype=np.float32)
            last_arr = np.array(last_vector, dtype=np.float32)
            
            new_norm = new_arr / (np.linalg.norm(new_arr) + 1e-8)
            last_norm = last_arr / (np.linalg.norm(last_arr) + 1e-8)
            
            similarity = float(np.dot(new_norm, last_norm))
            
            return similarity >= similarity_threshold
        except Exception as exc:
            logger.debug("Error computing vector similarity: %s", exc)
            return False

    def suggest_adjustments(
        self,
        vector: List[float],
        signature: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Suggest affective state adjustments based on emotion vector/signature.
        
        Args:
            vector: Emotion vector
            signature: Optional emotion signature for keyword matching
            
        Returns:
            Dictionary of suggested adjustments for energy, stress, confidence
        """
        cluster = self._identify_cluster(vector, signature)
        
        if cluster in self._emotional_clusters:
            return self._emotional_clusters[cluster]["adjustments"].copy()
        
        return {"energy": 0.0, "stress": 0.0, "confidence": 0.0}

    def _identify_cluster(
        self,
        vector: List[float],
        signature: Optional[str] = None
    ) -> str:
        """
        Identify which emotional cluster the vector belongs to.
        
        Args:
            vector: Emotion vector
            signature: Optional signature for keyword matching
            
        Returns:
            Cluster name (stressed, joyful, calm, angry, sad, neutral)
        """
        if signature:
            sig_lower = signature.lower()
            
            for cluster_name, cluster_info in self._emotional_clusters.items():
                keywords = cluster_info["keywords"]
                if any(kw in sig_lower for kw in keywords):
                    return cluster_name
        
        return "neutral"

    def get_memory_weights(
        self,
        current_vector: List[float],
        memory_vectors: List[List[float]]
    ) -> List[float]:
        """
        Compute similarity weights for memory retrieval.
        
        Args:
            current_vector: Current emotion vector
            memory_vectors: List of emotion vectors from memories
            
        Returns:
            List of weight multipliers (0-2) for each memory
        """
        if not memory_vectors:
            return []
        
        try:
            current_arr = np.array(current_vector, dtype=np.float32)
            current_norm = current_arr / (np.linalg.norm(current_arr) + 1e-8)
            
            weights = []
            for mem_vec in memory_vectors:
                if not mem_vec or len(mem_vec) != self.dimension:
                    weights.append(1.0)
                    continue
                
                mem_arr = np.array(mem_vec, dtype=np.float32)
                mem_norm = mem_arr / (np.linalg.norm(mem_arr) + 1e-8)
                
                similarity = float(np.dot(current_norm, mem_norm))
                weight = 1.0 + similarity
                
                weights.append(weight)
            
            return weights
        except Exception as exc:
            logger.debug("Error computing memory weights: %s", exc)
            return [1.0] * len(memory_vectors)

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the emotion index."""
        return {
            "dimension": self.dimension,
            "vector_count": len(self._vectors),
            "persona_id": self._persona_id,
            "faiss_available": FAISS_AVAILABLE,
            "using_faiss": self._index is not None,
            "clusters": list(self._emotional_clusters.keys())
        }
