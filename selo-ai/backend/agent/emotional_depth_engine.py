"""
Emotional Depth Engine

Manages rich, multi-dimensional emotional state with:
- Core emotions (Plutchik's wheel)
- Blended emotions (complex feelings)
- Emotional memory (how past events shape current feelings)
- Emotional momentum (emotions persist and transition gradually)
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone, timedelta

from .emotion_vector_cache import EmotionVectorCache
from .emotion_index_service import EmotionIndexService

logger = logging.getLogger("selo.agent.emotional_depth")


class EmotionalDepthEngine:
    """
    Manages deep, multi-dimensional emotional processing.
    
    Based on Plutchik's emotion wheel with extensions for
    complex human-like emotional experiences.
    """
    
    # Plutchik's 8 core emotions
    CORE_EMOTIONS = [
        "joy", "sadness", "anger", "fear",
        "surprise", "disgust", "trust", "anticipation"
    ]
    
    # Blended/complex emotions (combinations of core emotions)
    BLENDED_EMOTIONS = {
        "nostalgia": {"joy": 0.4, "sadness": 0.6},
        "bittersweet": {"joy": 0.5, "sadness": 0.5},
        "awe": {"surprise": 0.6, "fear": 0.2, "joy": 0.2},
        "anxiety": {"fear": 0.7, "anticipation": 0.3},
        "hope": {"joy": 0.4, "anticipation": 0.6},
        "despair": {"sadness": 0.7, "fear": 0.3},
        "guilt": {"sadness": 0.5, "fear": 0.3, "disgust": 0.2},
        "shame": {"sadness": 0.4, "fear": 0.4, "disgust": 0.2},
        "pride": {"joy": 0.6, "trust": 0.4},
        "gratitude": {"joy": 0.5, "trust": 0.5},
        "envy": {"sadness": 0.4, "anger": 0.4, "disgust": 0.2},
        "jealousy": {"fear": 0.4, "anger": 0.4, "sadness": 0.2},
        "relief": {"joy": 0.6, "surprise": 0.4},
        "disappointment": {"sadness": 0.6, "surprise": 0.4},
        "curiosity": {"anticipation": 0.6, "surprise": 0.4},
        "confusion": {"surprise": 0.5, "fear": 0.3, "anticipation": 0.2},
        "contentment": {"joy": 0.7, "trust": 0.3},
        "loneliness": {"sadness": 0.8, "fear": 0.2},
        "overwhelmed": {"fear": 0.5, "sadness": 0.3, "anger": 0.2},
        "frustrated": {"anger": 0.6, "sadness": 0.4},
        "excited": {"joy": 0.6, "anticipation": 0.4},
        "calm": {"trust": 0.6, "joy": 0.4},
        "stressed": {"fear": 0.5, "anger": 0.3, "sadness": 0.2},
    }
    
    def __init__(self, persona_repo, affective_state_repo, memory_repo=None, llm_router=None, emotion_index_service=None):
        """
        Initialize the emotional depth engine.
        
        Args:
            persona_repo: PersonaRepository instance
            affective_state_repo: AffectiveStateRepository instance
            memory_repo: Optional memory repository for emotional memories
            llm_router: Optional LLM router for embeddings
            emotion_index_service: Optional emotion index service for optimization
        """
        self.persona_repo = persona_repo
        self.affective_state_repo = affective_state_repo
        self.memory_repo = memory_repo
        self._emotion_cache = EmotionVectorCache(llm_router=llm_router)
        self._emotion_index = emotion_index_service
    
    async def process_emotional_experience(
        self,
        persona_id: str,
        trigger_event: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an emotional experience with full depth.
        
        Considers:
        - Current emotional state
        - Emotional memory (similar past experiences)
        - Personality traits (emotional reactivity)
        - Temporal momentum (emotions persist)
        
        Args:
            persona_id: Persona ID
            trigger_event: Event that triggered the emotion
            context: Additional context
        
        Returns:
            Rich emotional state dictionary
        """
        try:
            # Get current emotional state
            current_state = await self._get_current_emotional_state(persona_id)
            
            # Retrieve emotionally similar past experiences
            emotional_memories = []
            if self.memory_repo:
                emotional_memories = await self._retrieve_emotional_memories(
                    persona_id,
                    trigger_event,
                    similarity_threshold=0.7
                )
            
            # Calculate raw emotional response to trigger
            raw_emotions = await self._calculate_raw_emotional_response(
                trigger_event,
                context
            )
            
            # Apply emotional momentum (emotions don't switch instantly)
            momentum_factor = 0.3  # 30% persistence of previous state
            blended_emotions = {}
            
            for emotion in self.CORE_EMOTIONS:
                previous = current_state.get(emotion, 0.0)
                new = raw_emotions.get(emotion, 0.0)
                blended_emotions[emotion] = (
                    previous * momentum_factor +
                    new * (1 - momentum_factor)
                )
            
            # Get persona for personality modulation
            persona = await self.persona_repo.get_persona(
                persona_id=persona_id,
                include_traits=True
            )
            
            # Apply personality modulation
            modulated_emotions = self._apply_personality_modulation(
                blended_emotions,
                persona
            )
            
            # Detect active blended emotions
            active_blends = self._detect_blended_emotions(modulated_emotions)
            
            # Calculate emotional intensity
            total_intensity = sum(modulated_emotions.values())
            avg_intensity = total_intensity / len(modulated_emotions) if modulated_emotions else 0.0
            
            # Determine dominant emotion
            dominant_emotion = max(modulated_emotions, key=modulated_emotions.get) if modulated_emotions else "neutral"
            dominant_intensity = modulated_emotions.get(dominant_emotion, 0.0)
            
            # Store emotional memory if significant
            if dominant_intensity > 0.6:
                await self._store_emotional_memory(
                    persona_id,
                    trigger_event,
                    modulated_emotions,
                    active_blends
                )
            
            result = {
                "core_emotions": modulated_emotions,
                "blended_emotions": active_blends,
                "dominant_emotion": dominant_emotion,
                "dominant_intensity": round(dominant_intensity, 3),
                "emotional_intensity": round(avg_intensity, 3),
                "influenced_by_memories": len(emotional_memories),
                "momentum_applied": momentum_factor,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Update affective state repository
            await self._update_affective_state(persona_id, result)
            
            logger.debug(
                f"Processed emotional experience for persona {persona_id}: "
                f"dominant={dominant_emotion} ({dominant_intensity:.2f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing emotional experience: {e}", exc_info=True)
            return {
                "error": str(e),
                "core_emotions": {emotion: 0.0 for emotion in self.CORE_EMOTIONS},
                "dominant_emotion": "neutral"
            }
    
    async def _get_current_emotional_state(self, persona_id: str) -> Dict[str, float]:
        """Get current emotional state from affective state repository."""
        try:
            state = await self.affective_state_repo.get_latest_state(persona_id)
            if not state:
                return {emotion: 0.0 for emotion in self.CORE_EMOTIONS}
            
            # Extract core emotions if stored
            emotions = state.get("core_emotions", {})
            if not emotions:
                # Initialize from mood vector if available
                mood_vector = state.get("mood_vector", {})
                valence = mood_vector.get("valence", 0.0)
                arousal = mood_vector.get("arousal", 0.0)
                
                # Map mood vector to core emotions
                emotions = self._mood_vector_to_emotions(valence, arousal)
            
            return emotions
            
        except Exception as e:
            logger.warning(f"Error getting current emotional state: {e}")
            return {emotion: 0.0 for emotion in self.CORE_EMOTIONS}
    
    def _mood_vector_to_emotions(self, valence: float, arousal: float) -> Dict[str, float]:
        """Convert mood vector (valence/arousal) to core emotions."""
        emotions = {emotion: 0.0 for emotion in self.CORE_EMOTIONS}
        
        # Positive valence
        if valence > 0:
            if arousal > 0:
                emotions["joy"] = min(1.0, valence * arousal)
                emotions["anticipation"] = min(1.0, valence * arousal * 0.5)
            else:
                emotions["trust"] = min(1.0, valence * abs(arousal))
                emotions["joy"] = min(1.0, valence * 0.5)
        
        # Negative valence
        elif valence < 0:
            if arousal > 0:
                emotions["anger"] = min(1.0, abs(valence) * arousal * 0.7)
                emotions["fear"] = min(1.0, abs(valence) * arousal * 0.5)
            else:
                emotions["sadness"] = min(1.0, abs(valence) * abs(arousal))
        
        return emotions
    
    async def _retrieve_emotional_memories(
        self,
        persona_id: str,
        trigger_event: Dict[str, Any],
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Retrieve emotionally similar past experiences."""
        # Placeholder - would query memory repository for similar emotional experiences
        # This would use semantic similarity on event descriptions + emotional signatures
        return []
    
    async def _calculate_raw_emotional_response(
        self,
        trigger_event: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate raw emotional response to trigger event.
        
        Maps event characteristics to core emotions.
        """
        emotions = {emotion: 0.0 for emotion in self.CORE_EMOTIONS}
        
        # Extract event characteristics
        event_type = trigger_event.get("type", "")
        event_valence = trigger_event.get("valence", 0.0)  # -1 to 1
        event_intensity = trigger_event.get("intensity", 0.5)  # 0 to 1
        
        # Extract from emotional_state if present (from reflection)
        if "emotional_state" in trigger_event:
            emotional_state = trigger_event["emotional_state"]
            primary = emotional_state.get("primary", "")
            intensity = emotional_state.get("intensity", 0.5)
            
            # Map primary emotion to core emotions
            if primary in self.CORE_EMOTIONS:
                emotions[primary] = intensity
            elif primary in self.BLENDED_EMOTIONS:
                # Decompose blended emotion
                blend = self.BLENDED_EMOTIONS[primary]
                for core_emotion, weight in blend.items():
                    emotions[core_emotion] = intensity * weight
        
        # Event type mappings
        elif event_type == "reflection":
            # Reflections tend to be contemplative
            emotions["anticipation"] = 0.4
            emotions["trust"] = 0.3
        
        elif event_type == "learning":
            emotions["surprise"] = 0.3
            emotions["joy"] = 0.4
            emotions["anticipation"] = 0.3
        
        elif event_type == "goal_completion":
            emotions["joy"] = 0.7
            emotions["trust"] = 0.5
        
        elif event_type == "goal_failure":
            emotions["sadness"] = 0.5
            emotions["anger"] = 0.3
        
        elif event_type == "user_interaction":
            # Positive interaction
            if event_valence > 0:
                emotions["joy"] = event_valence * event_intensity
                emotions["trust"] = event_valence * event_intensity * 0.7
            # Negative interaction
            elif event_valence < 0:
                emotions["sadness"] = abs(event_valence) * event_intensity
                emotions["fear"] = abs(event_valence) * event_intensity * 0.5
        
        return emotions
    
    def _apply_personality_modulation(
        self,
        emotions: Dict[str, float],
        persona
    ) -> Dict[str, float]:
        """
        Modulate emotions based on personality traits.
        
        Different personalities experience emotions differently.
        """
        if not persona:
            return emotions
        
        modulated = emotions.copy()
        
        try:
            # Get personality dimensions
            personality = getattr(persona, "personality", {}) or {}
            
            # Neuroticism amplifies negative emotions
            neuroticism = personality.get("neuroticism", 0.3)
            if neuroticism > 0.5:
                amplification = 1.0 + (neuroticism - 0.5)
                modulated["fear"] *= amplification
                modulated["sadness"] *= amplification
                modulated["anger"] *= amplification
            
            # Extraversion amplifies joy and dampens fear
            extraversion = personality.get("extraversion", 0.5)
            if extraversion > 0.5:
                modulated["joy"] *= (1.0 + (extraversion - 0.5) * 0.5)
                modulated["fear"] *= (1.0 - (extraversion - 0.5) * 0.3)
            
            # Agreeableness amplifies trust and dampens anger
            agreeableness = personality.get("agreeableness", 0.7)
            if agreeableness > 0.5:
                modulated["trust"] *= (1.0 + (agreeableness - 0.5) * 0.5)
                modulated["anger"] *= (1.0 - (agreeableness - 0.5) * 0.5)
            
            # Openness amplifies surprise and anticipation
            openness = personality.get("openness", 0.7)
            if openness > 0.5:
                modulated["surprise"] *= (1.0 + (openness - 0.5) * 0.4)
                modulated["anticipation"] *= (1.0 + (openness - 0.5) * 0.4)
            
            # Clamp all values to [0, 1]
            for emotion in modulated:
                modulated[emotion] = max(0.0, min(1.0, modulated[emotion]))
            
        except Exception as e:
            logger.warning(f"Error applying personality modulation: {e}")
        
        return modulated
    
    def _detect_blended_emotions(
        self,
        core_emotions: Dict[str, float],
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Detect active blended emotions from core emotion state.
        
        Args:
            core_emotions: Current core emotion values
            threshold: Minimum match score to consider blend active
        
        Returns:
            List of active blended emotions with scores
        """
        active_blends = []
        
        for blend_name, blend_recipe in self.BLENDED_EMOTIONS.items():
            # Calculate match score
            match_score = 0.0
            total_weight = 0.0
            
            for core_emotion, weight in blend_recipe.items():
                if core_emotion in core_emotions:
                    match_score += core_emotions[core_emotion] * weight
                    total_weight += weight
            
            if total_weight > 0:
                match_score /= total_weight
            
            # If match score exceeds threshold, this blend is active
            if match_score >= threshold:
                active_blends.append({
                    "name": blend_name,
                    "intensity": round(match_score, 3),
                    "components": {
                        emotion: round(core_emotions.get(emotion, 0.0), 3)
                        for emotion in blend_recipe.keys()
                    }
                })
        
        # Sort by intensity
        active_blends.sort(key=lambda x: x["intensity"], reverse=True)
        
        return active_blends
    
    async def _store_emotional_memory(
        self,
        persona_id: str,
        trigger_event: Dict[str, Any],
        emotions: Dict[str, float],
        blends: List[Dict[str, Any]]
    ):
        """Store significant emotional experience as memory."""
        # Placeholder - would store in memory repository with emotional tags
        # This enables retrieval of similar emotional experiences later
        pass
    
    async def _update_affective_state(
        self,
        persona_id: str,
        emotional_result: Dict[str, Any]
    ):
        """Update affective state repository with new emotional state."""
        try:
            # Get current state
            current = await self.affective_state_repo.get_latest_state(persona_id)
            
            if not current:
                return
            
            # Update with core emotions
            update_data = {
                "id": current.get("id"),
                "persona_id": persona_id,
                "user_id": current.get("user_id"),
                "core_emotions": emotional_result.get("core_emotions", {}),
                "dominant_emotion": emotional_result.get("dominant_emotion"),
                "emotional_intensity": emotional_result.get("emotional_intensity"),
                "blended_emotions": emotional_result.get("blended_emotions", []),
                "last_update": datetime.now(timezone.utc),
                "state_metadata": {
                    **current.get("state_metadata", {}),
                    "last_emotional_update": datetime.now(timezone.utc).isoformat(),
                    "emotional_depth_version": "1.0"
                }
            }

            try:
                # Get previous cache for potential reuse
                previous_cache = current.get("state_metadata", {}).get("emotion_vector_cache")
                
                cache_payload = await self._emotion_cache.build_cache_payload(
                    emotional_result,
                    previous_cache=previous_cache
                )
                update_data["state_metadata"]["emotion_vector_cache"] = cache_payload
                
                # Add to emotion index if available
                if self._emotion_index and cache_payload.get("vector"):
                    self._emotion_index.add_vector(
                        vector=cache_payload["vector"],
                        signature=cache_payload["signature"],
                        state_id=current.get("id"),
                        timestamp=datetime.now(timezone.utc),
                        core_emotions=emotional_result.get("core_emotions"),
                        dominant_emotion=emotional_result.get("dominant_emotion"),
                        energy=current.get("energy"),
                        stress=current.get("stress"),
                        confidence=current.get("confidence")
                    )
            except Exception as cache_exc:
                logger.debug("Emotion vector cache update failed: %s", cache_exc)
            
            # Preserve existing fields
            for field in ["energy", "stress", "confidence", "mood_vector", "homeostasis_active"]:
                if field in current:
                    update_data[field] = current[field]
            
            await self.affective_state_repo.upsert_state(update_data)
            
        except Exception as e:
            logger.warning(f"Error updating affective state: {e}")
    
    def get_emotional_summary(self, emotional_state: Dict[str, Any]) -> str:
        """
        Generate human-readable emotional summary.
        
        Args:
            emotional_state: Emotional state dictionary
        
        Returns:
            Natural language summary
        """
        dominant = emotional_state.get("dominant_emotion", "neutral")
        intensity = emotional_state.get("dominant_intensity", 0.0)
        blends = emotional_state.get("blended_emotions", [])
        
        intensity_word = "slightly"
        if intensity > 0.7:
            intensity_word = "strongly"
        elif intensity > 0.5:
            intensity_word = "moderately"
        
        summary = f"{intensity_word} {dominant}"
        
        if blends and len(blends) > 0:
            top_blend = blends[0]
            summary += f" (with {top_blend['name']})"
        
        return summary
