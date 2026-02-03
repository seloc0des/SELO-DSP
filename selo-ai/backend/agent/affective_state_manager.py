"""Affective state management service for emergent agent roadmap."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..config.thresholds import SystemThresholds
from ..db.repositories.agent_state import AffectiveStateRepository
from ..db.repositories.persona import PersonaRepository

logger = logging.getLogger("selo.agent.affective")


class AffectiveStateManager:
    """Coordinates persistent affective state updates for a persona."""

    def __init__(
        self,
        state_repo: AffectiveStateRepository,
        persona_repo: PersonaRepository,
        emotion_index_service=None,
    ) -> None:
        self._state_repo = state_repo
        self._persona_repo = persona_repo
        self._emotion_index = emotion_index_service

    async def seed_baseline_state(
        self,
        persona_id: str,
        user_id: str,
        traits_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create an initial affective baseline anchored to persona traits."""
        if not traits_snapshot:
            traits_snapshot = await self._collect_trait_baseline(persona_id)
        baseline_payload = {
            "persona_id": persona_id,
            "user_id": user_id,
            "mood_vector": {
                "valence": traits_snapshot.get("empathy", 0.0),
                "arousal": traits_snapshot.get("energy", 0.0),
            },
            "energy": traits_snapshot.get("energy", 0.5),
            "stress": 0.4,
            "confidence": traits_snapshot.get("confidence", 0.6),
            "state_metadata": {
                "seeded_at": datetime.now(timezone.utc).isoformat(),
                "source": "baseline_seed",
            },
        }
        state = await self._state_repo.upsert_state(baseline_payload)
        logger.info("Seeded affective baseline for persona %s", persona_id)
        return state

    async def apply_reflection_adjustment(
        self,
        persona_id: str,
        adjustment: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Fold reflection sentiment data into the affective state."""
        latest = await self._state_repo.get_latest_state(persona_id)
        if not latest:
            logger.warning("No affective state found for persona %s", persona_id)
            return None

        # Start from latest snapshot
        # FIXED: Keep only the fields needed for update, don't pass read-only timestamp fields
        updated_payload = {
            "id": latest.get("id"),
            "persona_id": latest.get("persona_id"),
            "user_id": latest.get("user_id"),
            "mood_vector": latest.get("mood_vector", {}),
            "energy": latest.get("energy", 0.5),
            "stress": latest.get("stress", 0.4),
            "confidence": latest.get("confidence", 0.6),
            "state_metadata": latest.get("state_metadata", {}),
            "homeostasis_active": latest.get("homeostasis_active", True),
            "last_update": datetime.now(timezone.utc),
        }
        baseline_values = {
            "energy": float(latest.get("energy", 0.5) or 0.5),
            "stress": float(latest.get("stress", 0.4) or 0.4),
            "confidence": float(latest.get("confidence", 0.6) or 0.6),
        }

        delta_log = {}

        for key in ("energy", "stress", "confidence"):
            if key in adjustment:
                raw_delta = float(adjustment[key] or 0.0)
                # FIXED: Use centralized threshold configuration
                max_delta = getattr(
                    SystemThresholds, 
                    f"AFFECTIVE_{key.upper()}_DELTA_MAX",
                    0.3  # fallback
                )
                clamped_delta = max(-max_delta, min(max_delta, raw_delta))
                updated_payload[key] = max(0.0, min(1.0, baseline_values[key] + clamped_delta))
                delta_log[key] = round(updated_payload[key] - baseline_values[key], 4)

        if "mood_vector" in adjustment:
            mood_vector = latest.get("mood_vector", {}) or {}
            updated = dict(mood_vector)
            for axis, raw in (adjustment["mood_vector"] or {}).items():
                try:
                    raw_val = float(raw)
                except (TypeError, ValueError):
                    continue
                baseline = float(mood_vector.get(axis, 0.0) or 0.0)
                shift = max(-0.4, min(0.4, raw_val - baseline))
                updated[axis] = max(-1.0, min(1.0, baseline + shift))
                delta_log[f"mood_{axis}"] = round(updated[axis] - baseline, 4)
            updated_payload["mood_vector"] = updated
        metadata = updated_payload.setdefault("state_metadata", {})
        metadata.update(
            {
                "last_adjustment_source": adjustment.get("source", "reflection"),
                "last_adjustment_time": datetime.now(timezone.utc).isoformat(),
                "last_adjustment_deltas": delta_log,
            }
        )
        state = await self._state_repo.upsert_state(updated_payload)
        logger.debug("Updated affective state for persona %s", persona_id)
        return state

    async def apply_emotion_based_adjustment(
        self,
        persona_id: str,
        emotion_vector: list,
        emotion_signature: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Apply fast emotion-based adjustments using emotion index clustering.
        
        Args:
            persona_id: Persona ID
            emotion_vector: Current emotion vector
            emotion_signature: Human-readable emotion signature
            
        Returns:
            Updated affective state or None
        """
        if not self._emotion_index:
            return None
        
        try:
            # Get suggested adjustments from emotion index
            adjustments = self._emotion_index.suggest_adjustments(
                vector=emotion_vector,
                signature=emotion_signature
            )
            
            if not adjustments or all(v == 0.0 for v in adjustments.values()):
                return None
            
            # Apply the adjustments
            adjustment_payload = {
                "source": "emotion_clustering",
                **adjustments
            }
            
            return await self.apply_reflection_adjustment(persona_id, adjustment_payload)
            
        except Exception as exc:
            logger.debug("Emotion-based adjustment failed: %s", exc)
            return None

    async def run_homeostasis_decay(self, persona_id: str) -> Optional[Dict[str, Any]]:
        """Gentle decay toward baseline when agent loop runs."""
        latest = await self._state_repo.get_latest_state(persona_id)
        if not latest:
            return None
        
        # FIXED: Use centralized decay factor configuration
        decay_factor = SystemThresholds.HOMEOSTASIS_DECAY_FACTOR
        
        # FIXED: Build payload with only necessary fields
        updated_payload = {
            "id": latest.get("id"),
            "persona_id": latest.get("persona_id"),
            "user_id": latest.get("user_id"),
            "mood_vector": latest.get("mood_vector", {}),
            "energy": latest.get("energy", 0.5) * (1 - decay_factor) + 0.5 * decay_factor,
            "stress": latest.get("stress", 0.5) * (1 - decay_factor) + 0.5 * decay_factor,
            "confidence": latest.get("confidence", 0.6) * (1 - decay_factor) + 0.6 * decay_factor,
            "state_metadata": latest.get("state_metadata", {}),
            "homeostasis_active": latest.get("homeostasis_active", True),
            "last_update": datetime.now(timezone.utc),
        }

        state = await self._state_repo.upsert_state(updated_payload)
        logger.debug("Applied homeostasis decay for persona %s", persona_id)
        return state

    async def ensure_state_available(
        self,
        persona_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        return await self._state_repo.ensure_state(
            persona_id=persona_id,
            user_id=user_id,
            baseline=await self._collect_trait_baseline(persona_id),
        )

    async def _collect_trait_baseline(self, persona_id: str) -> Dict[str, Any]:
        """
        Collect trait baseline for affective state initialization.
        
        Returns dict with empathy, confidence, energy values. Falls back to
        sensible defaults if traits cannot be loaded.
        """
        # FIXED: Use centralized baseline configuration
        default_baseline = {
            "empathy": 0.5,  # No specific config, use default
            "confidence": SystemThresholds.AFFECTIVE_CONFIDENCE_BASELINE,
            "energy": SystemThresholds.AFFECTIVE_ENERGY_BASELINE,
        }
        
        try:
            persona = await self._persona_repo.get_persona(
                persona_id=persona_id,
                include_traits=True,
            )
            
            if not persona:
                logger.warning(
                    "Persona %s not found when collecting trait baseline, using defaults",
                    persona_id
                )
                return default_baseline
            
            traits = getattr(persona, "traits", None)
            if not traits:
                logger.info(
                    "Persona %s has no traits yet, using default baseline",
                    persona_id
                )
                return default_baseline
            
            # Extract trait values
            if isinstance(traits, list):
                values = {t.name: t.value for t in traits if hasattr(t, "name") and hasattr(t, "value")}
            elif isinstance(traits, dict):
                values = traits
            else:
                logger.warning(
                    "Persona %s has unexpected traits type %s, using defaults",
                    persona_id, type(traits).__name__
                )
                return default_baseline
            
            # Build baseline from traits with fallback to defaults
            baseline = {
                "empathy": float(values.get("empathy", default_baseline["empathy"]) or default_baseline["empathy"]),
                "confidence": float(values.get("confidence", default_baseline["confidence"]) or default_baseline["confidence"]),
                "energy": float(values.get("vitality", values.get("energy", default_baseline["energy"])) or default_baseline["energy"]),
            }
            
            logger.debug(
                "Collected trait baseline for persona %s: empathy=%.2f, confidence=%.2f, energy=%.2f",
                persona_id, baseline["empathy"], baseline["confidence"], baseline["energy"]
            )
            
            return baseline
            
        except Exception as exc:
            logger.warning(
                "Failed to collect baseline traits for %s: %s. Using defaults.",
                persona_id, exc, exc_info=True
            )
            return default_baseline
