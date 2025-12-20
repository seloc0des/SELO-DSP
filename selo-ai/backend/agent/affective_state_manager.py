"""Affective state management service for emergent agent roadmap."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..db.repositories.agent_state import AffectiveStateRepository
from ..db.repositories.persona import PersonaRepository

logger = logging.getLogger("selo.agent.affective")


class AffectiveStateManager:
    """Coordinates persistent affective state updates for a persona."""

    def __init__(
        self,
        state_repo: AffectiveStateRepository,
        persona_repo: PersonaRepository,
    ) -> None:
        self._state_repo = state_repo
        self._persona_repo = persona_repo

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

        # Start from latest snapshot but normalize temporal fields for DB update
        updated_payload = {**latest}
        # Ensure last_update is a proper datetime; let created_at/updated_at be managed by the ORM
        updated_payload.pop("created_at", None)
        updated_payload.pop("updated_at", None)
        updated_payload["last_update"] = datetime.now(timezone.utc)
        baseline_values = {
            "energy": float(latest.get("energy", 0.5) or 0.5),
            "stress": float(latest.get("stress", 0.4) or 0.4),
            "confidence": float(latest.get("confidence", 0.6) or 0.6),
        }

        delta_log = {}

        for key in ("energy", "stress", "confidence"):
            if key in adjustment:
                try:
                    target = float(adjustment[key])
                except (TypeError, ValueError):
                    continue
                current = baseline_values[key]
                # Treat the supplied value as a +/- relative shift around the current level
                # with gentle easing toward the proposed target.
                delta = max(-0.3, min(0.3, target - 0.5))
                blended = current + delta
                updated_payload[key] = max(0.0, min(1.0, blended))
                delta_log[key] = round(updated_payload[key] - current, 4)

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

    async def run_homeostasis_decay(self, persona_id: str) -> Optional[Dict[str, Any]]:
        """Gentle decay toward baseline when agent loop runs."""
        latest = await self._state_repo.get_latest_state(persona_id)
        if not latest:
            return None
        decay_factor = 0.1
        updated_payload = {**latest}
        # Strip ISO timestamp fields coming from to_dict and set a fresh last_update
        updated_payload.pop("created_at", None)
        updated_payload.pop("updated_at", None)
        updated_payload["last_update"] = datetime.now(timezone.utc)

        updated_payload["energy"] = latest.get("energy", 0.5) * (1 - decay_factor) + 0.5 * decay_factor
        updated_payload["stress"] = latest.get("stress", 0.5) * (1 - decay_factor) + 0.5 * decay_factor
        updated_payload["confidence"] = latest.get("confidence", 0.6) * (1 - decay_factor) + 0.6 * decay_factor

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
        try:
            persona = await self._persona_repo.get_persona(
                persona_id=persona_id,
                include_traits=True,
            )
            if not persona or not getattr(persona, "traits", None):
                return {}
            traits = getattr(persona, "traits")
            if isinstance(traits, list):
                values = {t.name: t.value for t in traits if hasattr(t, "name")}
            elif isinstance(traits, dict):
                values = traits
            else:
                values = {}
            return {
                "empathy": float(values.get("empathy", 0.0) or 0.0),
                "confidence": float(values.get("confidence", 0.6) or 0.6),
                "energy": float(values.get("vitality", 0.5) or values.get("energy", 0.5) or 0.5),
            }
        except Exception as exc:
            logger.debug("Failed to collect baseline traits for %s: %s", persona_id, exc)
            return {}
