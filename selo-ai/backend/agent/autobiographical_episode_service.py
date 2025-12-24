"""LLM-driven autobiographical episode synthesis service."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from ..prompt.builder import PromptBuilder
from ..llm.router import LLMRouter
from ..db.repositories.agent_state import AutobiographicalEpisodeRepository
from .episode_builder import EpisodeBuilder

logger = logging.getLogger("selo.agent.autobio")


class AutobiographicalEpisodeService:
    """Coordinates context aggregation, LLM synthesis, and persistence for episodes."""

    def __init__(
        self,
        *,
        prompt_builder: PromptBuilder,
        llm_router: LLMRouter,
        episode_builder: EpisodeBuilder,
        episode_repo: AutobiographicalEpisodeRepository,
        persona_repo,
        goal_manager,
        affective_state_manager,
        reflection_repo,
        conversation_repo,
    ) -> None:
        self._prompt_builder = prompt_builder
        self._llm_router = llm_router
        self._episode_builder = episode_builder
        self._episode_repo = episode_repo
        self._persona_repo = persona_repo
        self._goal_manager = goal_manager
        self._affective_state_manager = affective_state_manager
        self._reflection_repo = reflection_repo
        self._conversation_repo = conversation_repo

    async def generate_episode(
        self,
        *,
        persona_id: str,
        user_id: str,
        trigger_reason: str,
        plan_steps: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Orchestrate full episode generation and persistence."""

        try:
            context = await self._build_context(
                persona_id=persona_id,
                user_id=user_id,
                plan_steps=plan_steps or [],
                trigger_reason=trigger_reason,
            )
        except Exception as exc:
            logger.error("Failed to construct autobiographical episode context: %s", exc, exc_info=True)
            return None

        persona_name = context.get("persona_name", "")
        try:
            prompt = await self._prompt_builder.build_prompt(
                "autobiographical_episode",
                context,
                persona_name=persona_name,
            )
        except Exception as exc:
            logger.error("Failed to build autobiographical episode prompt: %s", exc, exc_info=True)
            return None

        raw_response = await self._invoke_llm(prompt)
        if not raw_response:
            return None

        episode_payload = self._parse_response(raw_response)
        if not episode_payload:
            logger.warning("Autobiographical episode synthesis returned unusable payload.")
            return None

        metadata = episode_payload.setdefault("metadata", {}) or {}
        metadata.setdefault("source", "autobiographical_episode_llm")
        evidence_refs: List[str] = list(metadata.get("evidence_refs") or [])
        for step in plan_steps or []:
            step_id = (step or {}).get("id")
            if step_id:
                evidence_refs.append(f"plan_step:{step_id}")
        metadata["evidence_refs"] = sorted(set(evidence_refs))
        metadata.setdefault("trigger_reason", trigger_reason)
        episode_payload["metadata"] = metadata

        # Provide sensible defaults for missing scalar fields
        episode_payload.setdefault("emotion_tags", [])
        episode_payload.setdefault("participants", ["User", "SELO"])
        episode_payload.setdefault("linked_memory_ids", [])
        episode_payload.setdefault("importance", 0.6)

        try:
            return await self._episode_builder.persist_episode_payload(
                persona_id=persona_id,
                user_id=user_id,
                episode_data=episode_payload,
                trigger_reason=trigger_reason,
            )
        except Exception as exc:
            logger.error("Failed to persist autobiographical episode: %s", exc, exc_info=True)
            return None

    async def list_recent_episodes(
        self,
        persona_id: str,
        *,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return recent episodes for the given persona."""

        try:
            return await self._episode_repo.list_recent_episodes(persona_id, limit=limit)
        except Exception as exc:
            logger.error("Failed to list episodes for persona %s: %s", persona_id, exc, exc_info=True)
            return []

    async def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single episode by identifier."""

        if not episode_id:
            return None
        try:
            return await self._episode_repo.get_episode(episode_id)
        except Exception as exc:
            logger.error("Failed to load episode %s: %s", episode_id, exc, exc_info=True)
            return None

    async def _invoke_llm(self, prompt: str, *, max_tokens: int = 720) -> Optional[str]:
        try:
            response = await self._llm_router.route(
                task_type="analytical",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.4,
            )
        except Exception as exc:
            logger.error("Autobiographical episode LLM call failed: %s", exc, exc_info=True)
            return None

        if not response:
            return None
        return (response.get("content") or response.get("completion") or "").strip()

    def _parse_response(self, raw: str) -> Optional[Dict[str, Any]]:
        candidate = raw.strip()
        if not candidate:
            return None
        if candidate.startswith("```"):
            lines = [line for line in candidate.splitlines() if not line.strip().startswith("```")]
            candidate = "".join(lines).strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            logger.warning("LLM output is not valid JSON: %s", candidate[:150])
            return None

        required_fields = {"title", "narrative_text", "summary"}
        if not required_fields.issubset(payload.keys()):
            logger.warning("Episode payload missing required fields: %s", required_fields - payload.keys())
            return None

        narrative_text = (payload.get("narrative_text") or "").strip()
        if not narrative_text:
            logger.warning("Episode payload has empty narrative text.")
            return None
        
        # FIXED: Validate word count and reject episodes outside acceptable range
        word_count = len(narrative_text.split())
        if word_count < 120:
            logger.warning("Episode narrative too short (%d words, minimum 120). Rejecting.", word_count)
            return None
        if word_count > 360:
            logger.warning("Episode narrative too long (%d words, maximum 360). Rejecting.", word_count)
            return None

        try:
            payload["importance"] = float(payload.get("importance", 0.6))
        except (TypeError, ValueError):
            payload["importance"] = 0.6

        return payload

    async def _build_context(
        self,
        *,
        persona_id: str,
        user_id: str,
        plan_steps: Sequence[Dict[str, Any]],
        trigger_reason: str,
    ) -> Dict[str, Any]:
        persona = await self._persona_repo.get_persona(persona_id, include_traits=True)
        persona_name = getattr(persona, "name", "").strip() if persona else ""
        persona_overview = self._format_persona(persona)

        affective_snapshot = await self._collect_affective_state(persona_id=persona_id, user_id=user_id)
        active_goals = await self._collect_active_goals(persona_id)
        meta_directives = await self._collect_meta_directives(persona_id)
        recent_reflections = await self._collect_recent_reflections(user_id)
        significant_memories = await self._collect_memories(user_id)
        plan_steps_block = self._format_plan_steps(plan_steps, fallback_steps=await self._collect_plan_steps(persona_id))

        return {
            "persona_overview": persona_overview,
            "persona_name": persona_name,
            "trigger_reason": trigger_reason or "unspecified",
            "affective_state": affective_snapshot,
            "plan_steps": plan_steps_block,
            "meta_directives": meta_directives,
            "recent_reflections": recent_reflections,
            "significant_memories": significant_memories,
        }

    async def _collect_affective_state(self, *, persona_id: str, user_id: str) -> str:
        try:
            state = await self._affective_state_manager.ensure_state_available(
                persona_id=persona_id,
                user_id=user_id,
            )
        except Exception as exc:
            logger.debug("Unable to collect affective state for persona %s: %s", persona_id, exc)
            return "Affective snapshot unavailable."

        if not isinstance(state, dict):
            return "Affective snapshot unavailable."

        lines = ["- energy: {energy:.2f}".format(energy=float(state.get("energy", 0.5) or 0.5))]
        lines.append("- stress: {stress:.2f}".format(stress=float(state.get("stress", 0.4) or 0.4)))
        lines.append("- confidence: {confidence:.2f}".format(confidence=float(state.get("confidence", 0.6) or 0.6)))
        mood_vector = state.get("mood_vector") or {}
        if isinstance(mood_vector, dict) and mood_vector:
            lines.append(f"- mood vector: {json.dumps(mood_vector, ensure_ascii=False)}")
        return "\n".join(lines)

    async def _collect_active_goals(self, persona_id: str) -> str:
        try:
            goals = await self._goal_manager.list_active_goals(persona_id)
        except Exception as exc:
            logger.debug("Unable to collect active goals: %s", exc)
            return "No active goals captured."

        if not goals:
            return "No active goals tracked."

        lines: List[str] = []
        for goal in goals[:4]:
            title = goal.get("title") or goal.get("description") or "Goal"
            progress = float(goal.get("progress", 0.0) or 0.0)
            priority = float(goal.get("priority", 0.5) or 0.5)
            lines.append(f"- {title} (progress {progress:.0%}, priority {priority:.2f})")
        if len(goals) > 4:
            lines.append(f"- (+{len(goals) - 4} additional goals)")
        return "\n".join(lines)

    async def _collect_plan_steps(self, persona_id: str) -> Sequence[Dict[str, Any]]:
        try:
            return await self._goal_manager.list_pending_steps(persona_id)
        except Exception as exc:
            logger.debug("Unable to collect existing plan steps: %s", exc)
            return []

    async def _collect_meta_directives(self, persona_id: str) -> str:
        try:
            directives = await self._goal_manager.list_meta_directives(
                persona_id,
                statuses=["pending", "in_progress"],
                limit=6,
            )
        except Exception as exc:
            logger.debug("Unable to collect meta directives: %s", exc)
            return "No meta directives in play."

        if not directives:
            return "No meta directives in play."

        lines: List[str] = []
        for directive in directives[:4]:
            text = directive.get("directive_text", "(directive)")
            priority = float(directive.get("priority", 0.5) or 0.5)
            due = directive.get("due_time") or directive.get("target_time")
            suffix = f" priority {priority:.2f}"
            if due:
                suffix += f", due {due}"
            lines.append(f"- {text}{suffix}")
        if len(directives) > 4:
            lines.append(f"- (+{len(directives) - 4} additional directives)")
        return "\n".join(lines)

    async def _collect_recent_reflections(self, user_id: str) -> str:
        if not self._reflection_repo:
            return "No reflections available."
        try:
            reflections = await self._reflection_repo.list_reflections(
                user_profile_id=str(user_id),
                limit=3,
            )
        except Exception as exc:
            logger.debug("Unable to collect reflections for user %s: %s", user_id, exc)
            return "No reflections available."

        if not reflections:
            return "No reflections available."

        lines: List[str] = []
        for reflection in reflections:
            created_at = reflection.get("created_at") or "recent"
            result = reflection.get("result") or {}
            highlight = result.get("content") or result.get("insights") or reflection.get("content")
            if isinstance(highlight, list):
                highlight = highlight[0] if highlight else ""
            snippet = (highlight or "").strip()
            if len(snippet) > 160:
                snippet = snippet[:157].rstrip() + "…"
            lines.append(f"- {created_at}: {snippet or 'Reflection recorded.'}")
        return "\n".join(lines)

    async def _collect_memories(self, user_id: str) -> str:
        if not self._conversation_repo:
            return "No salient memories recorded."
        try:
            memories = await self._conversation_repo.get_memories(
                user_id=str(user_id),
                importance_threshold=4,
                limit=6,
            )
        except Exception as exc:
            logger.debug("Unable to collect memories for user %s: %s", user_id, exc)
            return "No salient memories recorded."

        if not memories:
            return "No salient memories recorded."

        lines: List[str] = []
        for memory in memories[:5]:
            if isinstance(memory, dict):
                content = memory.get("content") or memory.get("summary") or "Memory"
            else:
                content = getattr(memory, "summary", None) or getattr(memory, "content", "Memory")
            snippet = " ".join(str(content).split())
            if len(snippet) > 150:
                snippet = snippet[:147].rstrip() + "…"
            lines.append(f"- {snippet}")
        if len(memories) > 5:
            lines.append(f"- (+{len(memories) - 5} additional memories)")
        return "\n".join(lines)

    def _format_plan_steps(
        self,
        plan_steps: Sequence[Dict[str, Any]],
        *,
        fallback_steps: Sequence[Dict[str, Any]],
    ) -> str:
        steps: List[Dict[str, Any]] = list(plan_steps) or list(fallback_steps) or []
        if not steps:
            return "No pending plan steps."

        lines: List[str] = []
        for step in steps[:5]:
            description = step.get("description", "(step)")
            status = step.get("status", "pending")
            priority = step.get("priority")
            detail = f"- {description} [{status}]"
            if priority is not None:
                try:
                    detail += f" (priority {float(priority):.2f})"
                except (TypeError, ValueError):
                    pass
            lines.append(detail)
        if len(steps) > 5:
            lines.append(f"- (+{len(steps) - 5} additional steps)")
        return "\n".join(lines)

    def _format_persona(self, persona) -> str:
        if not persona:
            return "Persona snapshot unavailable."

        name = getattr(persona, "name", "SELO").strip() or "SELO"
        description = getattr(persona, "description", "").strip()
        lines = [f"Name: {name}"]
        if description:
            lines.append(f"Essence: {description}")

        traits = getattr(persona, "traits", []) or []
        trait_pairs: List[tuple[str, float]] = []
        for trait in traits:
            try:
                trait_name = getattr(trait, "name", None)
                value = float(getattr(trait, "value", 0.0) or 0.0)
                if trait_name:
                    trait_pairs.append((trait_name, value))
            except Exception:
                continue
        trait_pairs.sort(key=lambda item: item[1], reverse=True)
        if trait_pairs:
            formatted_traits = [f"{name} ({value:.2f})" for name, value in trait_pairs[:5]]
            lines.append("Traits: " + ", ".join(formatted_traits))

        return "\n".join(lines) or "Persona snapshot unavailable."
