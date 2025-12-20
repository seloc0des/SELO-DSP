"""Episode builder pipeline for autobiographical memories."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..db.repositories.agent_state import AutobiographicalEpisodeRepository

logger = logging.getLogger("selo.agent.episodes")


class EpisodeBuilder:
    """Constructs autobiographical episodes from reflections and memories."""

    def __init__(self, episode_repo: AutobiographicalEpisodeRepository) -> None:
        self._episode_repo = episode_repo

    async def build_episode(
        self,
        persona_id: str,
        user_id: str,
        artifacts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build and persist an autobiographical episode from preformatted artifacts."""

        episode_payload = {
            "persona_id": persona_id,
            "user_id": user_id,
            "title": artifacts.get("title", "Recent shared moment"),
            "narrative_text": artifacts.get("narrative", ""),
            "summary": artifacts.get("summary"),
            "emotion_tags": artifacts.get("emotion_tags", []),
            "participants": artifacts.get("participants", ["User", "SELO"]),
            "linked_memory_ids": artifacts.get("linked_memory_ids", []),
            "importance": float(artifacts.get("importance", 0.6)),
            "metadata": {
                "source": artifacts.get("source", "prototype_pipeline"),
                "artifacts_used": list(artifacts.keys()),
            },
        }
        if "start_time" in artifacts:
            episode_payload["start_time"] = artifacts["start_time"]
        if "end_time" in artifacts:
            episode_payload["end_time"] = artifacts["end_time"]
        episode = await self._episode_repo.create_episode(episode_payload)
        logger.info("Created autobiographical episode %s", episode["id"])
        return episode

    async def persist_episode_payload(
        self,
        *,
        persona_id: str,
        user_id: str,
        episode_data: Dict[str, Any],
        trigger_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist an episode payload produced by the LLM-driven synthesis pipeline."""

        title = (episode_data.get("title") or "Shared moment").strip()
        narrative_text = (episode_data.get("narrative_text") or "").strip()
        summary = (episode_data.get("summary") or "").strip() or None

        emotion_tags = episode_data.get("emotion_tags") or []
        participants = episode_data.get("participants") or ["User", "SELO"]
        linked_memory_ids = episode_data.get("linked_memory_ids") or []
        importance = episode_data.get("importance", 0.6)

        try:
            importance_value = float(importance)
        except (TypeError, ValueError):
            importance_value = 0.6
        importance_value = max(0.0, min(1.0, importance_value))

        metadata = episode_data.get("metadata") or {}
        if trigger_reason:
            metadata.setdefault("trigger_reason", trigger_reason)
        metadata.setdefault("source", metadata.get("source", "autobiographical_episode_llm"))

        episode_payload: Dict[str, Any] = {
            "persona_id": persona_id,
            "user_id": user_id,
            "title": title or "Shared moment",
            "narrative_text": narrative_text,
            "summary": summary,
            "emotion_tags": emotion_tags,
            "participants": participants,
            "linked_memory_ids": linked_memory_ids,
            "importance": importance_value,
            "metadata": metadata,
        }

        start_time = episode_data.get("start_time")
        end_time = episode_data.get("end_time")
        if isinstance(start_time, datetime):
            episode_payload["start_time"] = _ensure_utc(start_time)
        if isinstance(end_time, datetime):
            episode_payload["end_time"] = _ensure_utc(end_time)

        episode = await self._episode_repo.create_episode(episode_payload)
        logger.info("Persisted autobiographical episode %s", episode.get("id"))
        return episode

    async def list_recent_episodes(
        self,
        persona_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        return await self._episode_repo.list_recent_episodes(persona_id, limit=limit)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
