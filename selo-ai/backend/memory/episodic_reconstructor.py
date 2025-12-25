"""
Episodic Memory Reconstructor

Reconstructs episodic memories as coherent narratives
rather than isolated facts.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict

logger = logging.getLogger("selo.memory.episodic")


class EpisodicMemoryReconstructor:
    """
    Reconstructs episodic memories as coherent narratives
    rather than isolated facts.
    
    Features:
    - Temporal clustering of memory fragments
    - Causal connection inference
    - Emotional arc extraction
    - Narrative reconstruction
    """
    
    def __init__(
        self,
        llm_router,
        memory_repo,
        conversation_repo
    ):
        """
        Initialize the episodic memory reconstructor.
        
        Args:
            llm_router: LLM router for narrative generation
            memory_repo: Memory repository
            conversation_repo: Conversation repository
        """
        self.llm_router = llm_router
        self.memory_repo = memory_repo
        self.conversation_repo = conversation_repo
    
    async def reconstruct_episode(
        self,
        user_id: str,
        query: str,
        timeframe: Optional[Tuple[datetime, datetime]] = None,
        max_fragments: int = 20
    ) -> Dict[str, Any]:
        """
        Reconstruct a coherent episode from fragmented memories.
        
        Returns a narrative reconstruction with:
        - Temporal sequence
        - Causal connections
        - Emotional arc
        - Key participants and entities
        
        Args:
            user_id: User ID
            query: Query describing the episode to reconstruct
            timeframe: Optional (start, end) datetime tuple
            max_fragments: Maximum memory fragments to retrieve
        
        Returns:
            Reconstructed episode dictionary
        """
        try:
            # Retrieve relevant memory fragments
            fragments = await self._retrieve_memory_fragments(
                user_id,
                query,
                timeframe,
                max_fragments
            )
            
            if not fragments or len(fragments) < 2:
                return {
                    "status": "insufficient_data",
                    "message": "Not enough memory fragments to reconstruct episode",
                    "fragments_found": len(fragments) if fragments else 0
                }
            
            # Cluster by temporal proximity
            temporal_clusters = self._cluster_by_time(fragments)
            
            # Build causal connections
            causal_graph = self._build_causal_connections(temporal_clusters)
            
            # Extract emotional arc
            emotional_arc = self._extract_emotional_trajectory(temporal_clusters)
            
            # Extract key entities
            entities = self._extract_entities(fragments)
            
            # Generate narrative reconstruction
            narrative = await self._generate_narrative_reconstruction(
                query,
                temporal_clusters,
                causal_graph,
                emotional_arc,
                entities
            )
            
            # Calculate reconstruction confidence
            confidence = self._calculate_reconstruction_confidence(
                fragments,
                temporal_clusters,
                causal_graph
            )
            
            result = {
                "status": "success",
                "query": query,
                "narrative": narrative,
                "timeline": self._format_timeline(temporal_clusters),
                "emotional_arc": emotional_arc,
                "key_entities": entities,
                "causal_connections": len(causal_graph),
                "fragments_used": len(fragments),
                "confidence": round(confidence, 3),
                "timeframe": {
                    "start": fragments[0].get("timestamp") if fragments else None,
                    "end": fragments[-1].get("timestamp") if fragments else None
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(
                f"Reconstructed episode for user {user_id}: "
                f"{len(fragments)} fragments, confidence={confidence:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error reconstructing episode: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }
    
    async def _retrieve_memory_fragments(
        self,
        user_id: str,
        query: str,
        timeframe: Optional[Tuple[datetime, datetime]],
        max_fragments: int
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memory fragments."""
        try:
            # Get memories matching query
            memories = await self.memory_repo.search_memories(
                user_id=user_id,
                query=query,
                limit=max_fragments
            )
            
            if not memories:
                # Fallback to recent memories
                memories = await self.memory_repo.get_memories(
                    user_id=user_id,
                    importance_threshold=5,
                    limit=max_fragments
                )
            
            # Filter by timeframe if provided
            if timeframe and memories:
                start_time, end_time = timeframe
                filtered = []
                for memory in memories:
                    timestamp = memory.get("timestamp") or memory.get("created_at")
                    if timestamp:
                        if isinstance(timestamp, str):
                            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        if start_time <= timestamp <= end_time:
                            filtered.append(memory)
                memories = filtered
            
            # Sort by timestamp
            memories.sort(key=lambda m: m.get("timestamp") or m.get("created_at") or datetime.min)
            
            return memories
            
        except Exception as e:
            logger.warning(f"Error retrieving memory fragments: {e}")
            return []
    
    def _cluster_by_time(
        self,
        fragments: List[Dict[str, Any]],
        max_gap_hours: int = 6
    ) -> List[List[Dict[str, Any]]]:
        """
        Cluster memory fragments by temporal proximity.
        
        Args:
            fragments: Memory fragments
            max_gap_hours: Maximum time gap to consider same cluster
        
        Returns:
            List of temporal clusters
        """
        if not fragments:
            return []
        
        clusters = []
        current_cluster = [fragments[0]]
        
        for i in range(1, len(fragments)):
            prev_fragment = fragments[i - 1]
            curr_fragment = fragments[i]
            
            # Get timestamps
            prev_time = prev_fragment.get("timestamp") or prev_fragment.get("created_at")
            curr_time = curr_fragment.get("timestamp") or curr_fragment.get("created_at")
            
            if prev_time and curr_time:
                if isinstance(prev_time, str):
                    prev_time = datetime.fromisoformat(prev_time.replace("Z", "+00:00"))
                if isinstance(curr_time, str):
                    curr_time = datetime.fromisoformat(curr_time.replace("Z", "+00:00"))
                
                time_gap = curr_time - prev_time
                
                # If gap is small, add to current cluster
                if time_gap <= timedelta(hours=max_gap_hours):
                    current_cluster.append(curr_fragment)
                else:
                    # Start new cluster
                    clusters.append(current_cluster)
                    current_cluster = [curr_fragment]
            else:
                # No timestamp info, add to current cluster
                current_cluster.append(curr_fragment)
        
        # Add final cluster
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def _build_causal_connections(
        self,
        temporal_clusters: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Infer causal connections between memory fragments.
        
        Returns list of causal connections.
        """
        connections = []
        
        # Simple heuristic: temporal proximity + keyword overlap suggests causality
        for cluster_idx, cluster in enumerate(temporal_clusters):
            for i in range(len(cluster) - 1):
                fragment_a = cluster[i]
                fragment_b = cluster[i + 1]
                
                # Extract keywords from both fragments
                keywords_a = self._extract_keywords(fragment_a.get("content", ""))
                keywords_b = self._extract_keywords(fragment_b.get("content", ""))
                
                # Calculate keyword overlap
                overlap = len(keywords_a & keywords_b)
                
                if overlap > 0:
                    connections.append({
                        "from": fragment_a.get("id"),
                        "to": fragment_b.get("id"),
                        "strength": min(1.0, overlap / 5.0),  # Normalize to 0-1
                        "type": "temporal_proximity",
                        "cluster": cluster_idx
                    })
        
        return connections
    
    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = text.lower().split()
        # Filter out common words
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = {word.strip(".,!?;:") for word in words if len(word) > 3 and word not in stopwords}
        return keywords
    
    def _extract_emotional_trajectory(
        self,
        temporal_clusters: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Extract emotional arc across the episode.
        
        Returns list of emotional states over time.
        """
        emotional_arc = []
        
        for cluster_idx, cluster in enumerate(temporal_clusters):
            # Aggregate emotions in this cluster
            emotions = []
            
            for fragment in cluster:
                # Try to extract emotional indicators
                content = fragment.get("content", "").lower()
                
                # Simple emotion detection
                if any(word in content for word in ["happy", "joy", "excited", "pleased"]):
                    emotions.append("positive")
                elif any(word in content for word in ["sad", "disappointed", "upset", "frustrated"]):
                    emotions.append("negative")
                elif any(word in content for word in ["worried", "anxious", "nervous", "concerned"]):
                    emotions.append("anxious")
                else:
                    emotions.append("neutral")
            
            # Determine dominant emotion for cluster
            if emotions:
                from collections import Counter
                emotion_counts = Counter(emotions)
                dominant_emotion = emotion_counts.most_common(1)[0][0]
                
                emotional_arc.append({
                    "cluster": cluster_idx,
                    "dominant_emotion": dominant_emotion,
                    "intensity": len([e for e in emotions if e == dominant_emotion]) / len(emotions),
                    "fragment_count": len(cluster)
                })
        
        return emotional_arc
    
    def _extract_entities(
        self,
        fragments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract key entities (people, places, things) from fragments."""
        entities = defaultdict(int)
        
        for fragment in fragments:
            content = fragment.get("content", "")
            
            # Simple entity extraction (capitalized words)
            words = content.split()
            for word in words:
                if word and word[0].isupper() and len(word) > 1:
                    # Clean punctuation
                    entity = word.strip(".,!?;:")
                    if len(entity) > 1:
                        entities[entity] += 1
        
        # Convert to list and sort by frequency
        entity_list = [
            {"entity": entity, "frequency": count}
            for entity, count in entities.items()
        ]
        entity_list.sort(key=lambda x: x["frequency"], reverse=True)
        
        return entity_list[:10]  # Top 10 entities
    
    async def _generate_narrative_reconstruction(
        self,
        query: str,
        temporal_clusters: List[List[Dict[str, Any]]],
        causal_graph: List[Dict[str, Any]],
        emotional_arc: List[Dict[str, Any]],
        entities: List[Dict[str, Any]]
    ) -> str:
        """Generate coherent narrative from fragments."""
        try:
            # Build context for LLM
            cluster_summaries = []
            for idx, cluster in enumerate(temporal_clusters):
                fragments_text = []
                for fragment in cluster[:3]:  # Max 3 fragments per cluster
                    content = fragment.get("content", "")
                    if content:
                        fragments_text.append(content[:200])
                
                cluster_summaries.append(f"Period {idx + 1}: {' | '.join(fragments_text)}")
            
            emotional_summary = ", ".join([
                f"{arc['dominant_emotion']} ({arc['intensity']:.0%})"
                for arc in emotional_arc
            ])
            
            key_entities = ", ".join([e["entity"] for e in entities[:5]])
            
            prompt = f"""Reconstruct a coherent narrative from these memory fragments.

Query: {query}

Memory fragments organized by time period:
{chr(10).join(cluster_summaries)}

Emotional arc: {emotional_summary}

Key entities: {key_entities}

Causal connections: {len(causal_graph)} identified

Generate a coherent 3-4 paragraph narrative that:
1. Tells the story chronologically
2. Shows causal connections between events
3. Reflects the emotional journey
4. Mentions key entities naturally
5. Fills small gaps with reasonable inferences

Write in past tense, third person perspective.
"""
            
            response = await self.llm_router.generate(
                prompt=prompt,
                model_type="analytical",
                max_tokens=500,
                temperature=0.7
            )
            
            return response.strip()
            
        except Exception as e:
            logger.warning(f"Error generating narrative: {e}")
            return "Unable to generate narrative reconstruction."
    
    def _calculate_reconstruction_confidence(
        self,
        fragments: List[Dict[str, Any]],
        clusters: List[List[Dict[str, Any]]],
        causal_graph: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in the reconstruction."""
        # Base confidence on fragment count
        fragment_confidence = min(1.0, len(fragments) / 10.0)
        
        # Factor in temporal clustering quality
        cluster_confidence = 0.5
        if clusters:
            avg_cluster_size = sum(len(c) for c in clusters) / len(clusters)
            cluster_confidence = min(1.0, avg_cluster_size / 3.0)
        
        # Factor in causal connections
        causal_confidence = 0.5
        if fragments:
            causal_density = len(causal_graph) / len(fragments)
            causal_confidence = min(1.0, causal_density)
        
        # Weighted average
        overall = (
            fragment_confidence * 0.4 +
            cluster_confidence * 0.3 +
            causal_confidence * 0.3
        )
        
        return overall
    
    def _format_timeline(
        self,
        temporal_clusters: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Format timeline for output."""
        timeline = []
        
        for idx, cluster in enumerate(temporal_clusters):
            # Get time range for cluster
            timestamps = []
            for fragment in cluster:
                ts = fragment.get("timestamp") or fragment.get("created_at")
                if ts:
                    timestamps.append(ts)
            
            start_time = min(timestamps) if timestamps else None
            end_time = max(timestamps) if timestamps else None
            
            timeline.append({
                "period": idx + 1,
                "start_time": start_time.isoformat() if isinstance(start_time, datetime) else start_time,
                "end_time": end_time.isoformat() if isinstance(end_time, datetime) else end_time,
                "fragment_count": len(cluster),
                "summary": cluster[0].get("content", "")[:100] if cluster else ""
            })
        
        return timeline
