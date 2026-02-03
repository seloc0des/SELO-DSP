"""Emotion vector cache utilities."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("selo.agent.emotion_vector")


class EmotionVectorCache:
    """Generate and cache emotion embeddings in affective state metadata."""

    def __init__(self, llm_router: Optional[Any] = None) -> None:
        self._llm_router = llm_router

    async def build_embedding(
        self,
        emotion_signature: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate embedding payload for the emotion signature."""
        if not emotion_signature:
            return {
                "embedding": [],
                "dim": 0,
                "model": model,
                "error": "empty_signature",
            }

        if not self._llm_router:
            return {
                "embedding": [],
                "dim": 0,
                "model": model,
                "error": "llm_router_unavailable",
            }

        try:
            routed = await self._llm_router.route(
                task_type="embedding",
                prompt=emotion_signature,
                model=model,
            )
        except Exception as exc:
            logger.warning("Emotion embedding route failed: %s", exc)
            return {
                "embedding": [],
                "dim": 0,
                "model": model,
                "error": f"embedding_failed: {exc}",
            }

        if isinstance(routed, dict):
            vec = routed.get("embedding") or routed.get("vector") or routed.get("data")
            if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
                return {
                    "embedding": [float(x) for x in vec],
                    "dim": len(vec),
                    "model": routed.get("model") or model,
                }

        if isinstance(routed, list) and all(isinstance(x, (int, float)) for x in routed):
            return {"embedding": [float(x) for x in routed], "dim": len(routed), "model": model}

        return {
            "embedding": [],
            "dim": 0,
            "model": model,
            "error": "invalid_embedding_response",
        }

    def summarize_emotions(self, emotional_result: Dict[str, Any]) -> str:
        """Build a compact signature string for emotion embedding."""
        core_emotions = emotional_result.get("core_emotions") or {}
        dominant = emotional_result.get("dominant_emotion") or "neutral"
        dominant_intensity = emotional_result.get("dominant_intensity")
        blended = emotional_result.get("blended_emotions") or []

        pieces: List[str] = []
        pieces.append(f"dominant:{dominant}")
        if dominant_intensity is not None:
            try:
                pieces.append(f"dominant_intensity:{float(dominant_intensity):.3f}")
            except (TypeError, ValueError):
                pass

        if isinstance(core_emotions, dict) and core_emotions:
            sorted_core = sorted(core_emotions.items(), key=lambda item: item[1], reverse=True)
            core_parts = [f"{name}:{float(val):.3f}" for name, val in sorted_core[:6]]
            pieces.append("core=" + ",".join(core_parts))

        if blended:
            blend_parts = []
            for blend in blended[:3]:
                name = blend.get("name")
                intensity = blend.get("intensity")
                if not name:
                    continue
                try:
                    intensity_val = float(intensity) if intensity is not None else None
                except (TypeError, ValueError):
                    intensity_val = None
                if intensity_val is None:
                    blend_parts.append(str(name))
                else:
                    blend_parts.append(f"{name}:{intensity_val:.3f}")
            if blend_parts:
                pieces.append("blends=" + ",".join(blend_parts))

        return " | ".join(pieces)

    async def build_cache_payload(
        self,
        emotional_result: Dict[str, Any],
        model: Optional[str] = None,
        previous_cache: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create cache payload containing signature and vector.
        
        Args:
            emotional_result: Emotional processing result
            model: Optional embedding model name
            previous_cache: Previous cache payload to check for reuse
            
        Returns:
            Cache payload with signature, vector, and metadata
        """
        signature = self.summarize_emotions(emotional_result)
        
        # Check if we can reuse the previous embedding
        if previous_cache and previous_cache.get("signature") == signature:
            prev_vector = previous_cache.get("vector")
            if prev_vector and len(prev_vector) > 0:
                logger.debug("Reusing cached emotion vector (signature unchanged)")
                return {
                    "signature": signature,
                    "vector": prev_vector,
                    "dim": previous_cache.get("dim", len(prev_vector)),
                    "model": previous_cache.get("model"),
                    "error": previous_cache.get("error"),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "reused": True,
                }
        
        # Generate new embedding
        embedding_payload = await self.build_embedding(signature, model=model)
        return {
            "signature": signature,
            "vector": embedding_payload.get("embedding", []),
            "dim": embedding_payload.get("dim", 0),
            "model": embedding_payload.get("model"),
            "error": embedding_payload.get("error"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "reused": False,
        }
