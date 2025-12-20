"""Meta reflection processor for generating follow-up directives from reflections."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from ..db.repositories.reflection import ReflectionRepository
from ..db.repositories.persona import PersonaRepository
from ..db.repositories.user import UserRepository
from .goal_manager import GoalManager

logger = logging.getLogger("selo.agent.meta_reflection")


class MetaReflectionProcessor:
    """Derives meta directives from reflection output and recent history."""

    def __init__(
        self,
        reflection_repo: ReflectionRepository,
        goal_manager: GoalManager,
        persona_repo: PersonaRepository,
        user_repo: UserRepository,
    ) -> None:
        self._reflection_repo = reflection_repo
        self._goal_manager = goal_manager
        self._persona_repo = persona_repo
        self._user_repo = user_repo

    async def process_reflection(self, reflection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyse a stored reflection and emit meta directives where warranted."""
        if not reflection:
            return []

        reflection_id = reflection.get("id") or reflection.get("reflection_id")
        result_blob = reflection.get("result") or {}
        if not isinstance(result_blob, dict):
            logger.debug("Meta reflection skipped: result blob missing or invalid for %s", reflection_id)
            return []

        user_profile_id = reflection.get("user_profile_id")
        # Prefer canonical installation user_id from metadata when present
        try:
            meta = reflection.get("metadata") or {}
            canonical_user_id = meta.get("user_id") or user_profile_id
        except Exception:
            canonical_user_id = user_profile_id

        if not canonical_user_id:
            logger.debug("Meta reflection skipped: no user id on reflection %s", reflection_id)
            return []

        persona = await self._persona_repo.get_persona_by_user(
            user_id=canonical_user_id,
            is_default=True,
            include_traits=True,
        )
        if not persona:
            logger.debug("Meta reflection skipped: persona not found for user %s", canonical_user_id)
            return []

        persona_id = getattr(persona, "id", None)
        if not persona_id:
            return []

        directives_created: List[Dict[str, Any]] = []
        try:
            existing_active = await self._goal_manager.list_meta_directives(
                persona_id,
                statuses=["pending", "in_progress"],
                limit=50,
            )
            existing_texts = {d.get("directive_text", "").strip().lower() for d in existing_active}
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Unable to fetch existing meta directives: %s", exc)
            existing_texts = set()

        source_ids = set(reflection.get("source_reflection_ids") or [])
        if reflection_id:
            source_ids.add(str(reflection_id))
        source_reflection_ids = sorted(source_ids)

        actions = result_blob.get("actions") or []
        if isinstance(actions, list):
            for raw_action in actions:
                action_text = (raw_action or "").strip()
                if not action_text:
                    continue
                normalized = action_text.lower()
                if normalized in existing_texts:
                    continue

                due_time = datetime.now(timezone.utc) + timedelta(hours=20)
                directive_payload = {
                    "directive_text": action_text,
                    "priority": 0.6,
                    "status": "pending",
                    "due_time": due_time,
                    "source_reflection_ids": source_reflection_ids,
                    # Use canonical installation user id so directives respect FK to users.id
                    "user_id": canonical_user_id,
                    "extra_metadata": {
                        "source": "reflection_action",
                        "reflection_id": reflection_id,
                    },
                }
                directive = await self._goal_manager.attach_meta_directive(persona_id, directive_payload)
                existing_texts.add(normalized)
                directives_created.append(directive)

        emotional_state = result_blob.get("emotional_state")
        if isinstance(emotional_state, dict):
            primary = str(emotional_state.get("primary", "")).strip()
            try:
                intensity = float(emotional_state.get("intensity", 0.0) or 0.0)
            except (TypeError, ValueError):
                intensity = 0.0
            if intensity >= 0.75 and primary:
                directive_text = f"Plan a dedicated follow-up about the user's {primary} within 24 hours."
                normalized = directive_text.lower()
                if normalized not in existing_texts:
                    due_time = datetime.now(timezone.utc) + timedelta(hours=12)
                    directive_payload = {
                        "directive_text": directive_text,
                        "priority": 0.8,
                        "status": "pending",
                        "due_time": due_time,
                        "source_reflection_ids": source_reflection_ids,
                        "user_id": canonical_user_id,
                        "extra_metadata": {
                            "source": "emotional_state",
                            "primary_emotion": primary,
                            "intensity": intensity,
                            "reflection_id": reflection_id,
                        },
                    }
                    directive = await self._goal_manager.attach_meta_directive(persona_id, directive_payload)
                    existing_texts.add(normalized)
                    directives_created.append(directive)

        # Detect repeated themes over recent reflections to nudge meta attention
        themes = result_blob.get("themes") or []
        recent_theme_counts: Dict[str, int] = {}
        if themes:
            try:
                recent_reflections = await self._reflection_repo.list_reflections(
                    user_profile_id=canonical_user_id,
                    limit=5,
                )
                for item in recent_reflections:
                    item_result = item.get("result") or {}
                    for theme in item_result.get("themes") or []:
                        theme_str = str(theme).strip().lower()
                        if theme_str:
                            recent_theme_counts[theme_str] = recent_theme_counts.get(theme_str, 0) + 1
            except Exception as exc:  # pragma: no cover
                logger.debug("Unable to gather recent reflections for meta analysis: %s", exc)

        for theme in themes:
            normalized_theme = str(theme or "").strip().lower()
            if not normalized_theme:
                continue
            if recent_theme_counts.get(normalized_theme, 0) < 2:
                continue
            directive_text = f"Synthesize learnings around recurring theme '{theme}' into a narrative episode."
            normalized = directive_text.lower()
            if normalized in existing_texts:
                continue
            due_time = datetime.now(timezone.utc) + timedelta(days=1)
            directive_payload = {
                "directive_text": directive_text,
                "priority": 0.55,
                "status": "pending",
                "due_time": due_time,
                "source_reflection_ids": source_reflection_ids,
                # Use canonical installation user id to satisfy FK and align with persona
                "user_id": canonical_user_id,
                "extra_metadata": {
                    "source": "recurring_theme",
                    "theme": theme,
                    "reflection_id": reflection_id,
                },
            }
            directive = await self._goal_manager.attach_meta_directive(persona_id, directive_payload)
            existing_texts.add(normalized)
            directives_created.append(directive)

        if directives_created:
            logger.info(
                "Meta reflection created %s directive(s) for persona %s",
                len(directives_created),
                persona_id,
            )
        return directives_created
