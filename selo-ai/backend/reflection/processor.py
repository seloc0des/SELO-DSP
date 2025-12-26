"""
Reflection/Reasoning Processor (RRP) Module

This module implements the core Reflection/Reasoning Processor that orchestrates 
the generation of reflections using LLM inference. It gathers context, constructs prompts,
performs LLM inference, and persists the results.
"""

import time
import json
import os
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Deque, Sequence, TYPE_CHECKING, Tuple
import logging
import asyncio
import inspect
import re
from datetime import datetime, timedelta, timezone
from collections import deque
from difflib import SequenceMatcher

try:
    from ..utils.text_utils import count_words
except ImportError:
    from backend.utils.text_utils import count_words

# Create logger
logger = logging.getLogger("selo.reflection")

# Default reflection timeouts (seconds); 0 implies unbounded
DEFAULT_REFLECTION_LLM_TIMEOUT_S = 0.0
DEFAULT_REFLECTION_SYNC_TIMEOUT_S = 0.0
DEFAULT_MAX_CONTEXT_ITEMS = 10
# Optimized for performance: smaller context = faster generation, still sufficient for quality
MAX_RECENT_CONVERSATION_MESSAGES = 5  # Reduced from 8 - last 5 messages provide sufficient context
MAX_CONTEXT_MEMORIES = 3  # Reduced from 5 - top 3 high-importance memories are most relevant
MAX_WEIGHTED_ATTRIBUTES = 10
MAX_RECENT_EVOLUTIONS = 3
MAX_RECENT_REFLECTIONS = 2
MAX_IDENTITY_CONSTRAINTS = 6


@dataclass
class DuplicateSanitizationResult:
    """Container capturing duplicate sanitization outcome."""

    content: str
    rewrites: int
    duplicate_ratio: float
    needs_regen: bool = False

# Context item priority levels for error handling
class ContextPriority:
    """Priority levels for context items during gathering."""
    REQUIRED = 0    # Must succeed or abort (persona, identity_constraints)
    HIGH = 1        # Retry on failure (conversations, weighted_attributes)
    MEDIUM = 2      # Continue with warning (memories, reflections)
    LOW = 3         # Best effort only (evolutions, patterns)

# Map context items to their priority levels
CONTEXT_ITEM_PRIORITIES = {
    'persona': ContextPriority.REQUIRED,
    'identity_constraints': ContextPriority.REQUIRED,
    'recent_conversations': ContextPriority.HIGH,
    'weighted_attributes': ContextPriority.HIGH,
    'persistent_memories': ContextPriority.MEDIUM,
    'recent_reflections': ContextPriority.MEDIUM,
    'recent_evolutions': ContextPriority.LOW,
    'conversation_patterns': ContextPriority.LOW,
}

def retry_on_failure(max_attempts=3, delay=0.5, exponential_backoff=True):
    """
    Decorator for async functions with exponential backoff retry logic.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        exponential_backoff: If True, double delay after each attempt
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        # Last attempt failed, re-raise
                        raise
                    wait_time = delay * (2 ** attempt if exponential_backoff else 1)
                    logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}), "
                                 f"retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
        return wrapper
    return decorator

# Import centralized sensory constraints
from ..constraints.sensory_constraints import SensoryConstraints

# Global embedding queue for batched processing
_embedding_queue: Deque[Dict[str, Any]] = deque()
_embedding_processor_task: Optional[asyncio.Task] = None
_embedding_queue_lock = asyncio.Lock()

if TYPE_CHECKING:
    from ..agent.meta_reflection_processor import MetaReflectionProcessor
    from ..agent.affective_state_manager import AffectiveStateManager
    from ..agent.goal_manager import GoalManager

class ReflectionProcessor:
    """
    The Reflection/Reasoning Processor orchestrates the generation of reflections 
    for SELO's autonomous persona evolution.
    
    It manages the entire pipeline from context gathering to LLM inference to persistence.
    """
    
    # Class-level cached word count configuration (singleton pattern)
    _word_count_config = None
    
    @classmethod
    def get_word_count_config(cls):
        """
        Get cached word count configuration from reflection config.
        
        Single source of truth for word count limits across all validation points.
        Caches on first access to avoid repeated config loading.
        
        Returns:
            dict: {'min': int, 'max': int}
        """
        if cls._word_count_config is None:
            try:
                from ..config.reflection_config import get_reflection_config
                cfg = get_reflection_config()
                cls._word_count_config = {
                    'min': getattr(cfg, 'word_count_min', 80),
                    'max': getattr(cfg, 'word_count_max', 180)
                }
            except (ImportError, AttributeError, KeyError) as e:
                # Fallback to our standard range if config can't be loaded
                logger.debug(f"Failed to load reflection config: {e}")
                cls._word_count_config = {'min': 80, 'max': 180}
        return cls._word_count_config
    
    @staticmethod
    def get_reflection_max_tokens() -> int:
        """
        Get reflection max tokens with tier-aware fallback.
        
        Single source of truth for reflection token budget across all generation points.
        Uses environment variable if set, otherwise detects system tier.
        
        Returns:
            int: Maximum tokens for reflection generation
        """
        try:
            max_tokens_cfg = int(os.environ.get("REFLECTION_MAX_TOKENS", "0"))
            if max_tokens_cfg > 0:
                return max_tokens_cfg
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse REFLECTION_MAX_TOKENS: {e}")
        
        # Use tier-aware fallback from system profile
        try:
            from ..utils.system_profile import detect_system_profile
            profile = detect_system_profile()
            return profile["budgets"]["reflection_max_tokens"]
        except Exception as e:
            # Final fallback to standard tier value
            logger.warning(f"Failed to detect system tier for reflection tokens, using standard fallback: {e}")
            return 640

    @classmethod
    def get_type_word_bounds(cls, reflection_type: Optional[str]) -> Dict[str, int]:
        """Return word-count bounds, allowing overrides for specific reflection types.

        For most types, this defers to the global singleton config. For per-turn
        "message" reflections (and closely related "memory_triggered" reflections),
        allow narrower bounds so inner monologue is more concise.
        """
        base = cls.get_word_count_config() or {"min": 80, "max": 250}
        rtype = (reflection_type or "").strip().lower()

        if rtype in {"message", "memory_triggered"}:
            import os
            try:
                msg_min = int(os.getenv("REFLECTION_MESSAGE_WORD_MIN", "80"))
            except (ValueError, TypeError):
                msg_min = 80
            try:
                msg_max = int(os.getenv("REFLECTION_MESSAGE_WORD_MAX", "320"))
            except (ValueError, TypeError):
                msg_max = 320

            # Sanity-check overrides; fall back to base config on invalid values
            if msg_min <= 0 or msg_max <= msg_min:
                return base
            return {"min": msg_min, "max": msg_max}

        return base

    @staticmethod
    def _calculate_target_word_range(word_config: Dict[str, int]) -> tuple[int, int]:
        """Derive guidance range for narrative length based on configured limits."""
        # Use configured values, fall back to 80-250 if not set
        min_words = int(word_config.get("min", 80))
        max_words = int(word_config.get("max", 250))

        # Calculate target range based on configured min/max
        target_min = min_words + 30
        target_max = min(max_words, min_words + 200)  # Cap at 200 words above min
        
        # Ensure we have a reasonable range
        if target_max <= target_min:
            target_max = target_min + 100  # Ensure at least 100 word range
            
        return (target_min, target_max)

    def __init__(self, 
                 reflection_repo=None, 
                 memory_repo=None, 
                 emotion_repo=None, 
                 weighted_attrs_manager=None,
                 identity_constraint_repo=None,
                 prompt_builder=None, 
                 llm_controller=None, 
                 vector_store=None,
                 event_bus=None,
                 socketio_server=None,
                 reflection_namespace=None,
                 conversation_repo=None,
                 persona_repo=None,
                 user_repo=None,
                 meta_reflection_processor: Optional["MetaReflectionProcessor"] = None,
                 affective_state_manager: Optional["AffectiveStateManager"] = None,
                 goal_manager: Optional["GoalManager"] = None,
                 enable_deferred_embeddings: bool = True):
        """
        Initialize the ReflectionProcessor with dependencies.
        
        Args:
            reflection_repo: Repository for storing and retrieving reflections
            memory_repo: Repository for retrieving memory context
            emotion_repo: Repository for retrieving emotional context
            weighted_attrs_manager: Manager for weighted attributes
            identity_constraint_repo: Repository for identity constraints
            prompt_builder: Service for building LLM prompts
            llm_controller: Service for LLM inference
            vector_store: Vector storage for embeddings
            event_bus: Event bus for publishing reflection events
            socketio_server: Socket.IO server for real-time event broadcasting
        """
        self.reflection_repo = reflection_repo
        self.memory_repo = memory_repo
        self.emotion_repo = emotion_repo
        self.weighted_attrs_manager = weighted_attrs_manager
        self.identity_constraint_repo = identity_constraint_repo
        self.prompt_builder = prompt_builder
        self.llm_controller = llm_controller
        self.vector_store = vector_store
        self.event_bus = event_bus
        self.socketio_server = socketio_server
        self.reflection_namespace = reflection_namespace
        self.conversation_repo = conversation_repo
        self.persona_repo = persona_repo
        self.user_repo = user_repo
        self.meta_reflection_processor = meta_reflection_processor
        self.affective_state_manager = affective_state_manager
        self.goal_manager = goal_manager
        self.enable_deferred_embeddings = enable_deferred_embeddings

        # Precompile language detection pattern and load reflection config for translation fallbacks
        # Expanded pattern to cover more non-Latin scripts:
        # - CJK: Chinese (4e00-9fff), Japanese Hiragana (3040-309f), Katakana (30a0-30ff), Korean (ac00-d7af)
        # - Cyrillic: Russian, Ukrainian, etc. (0400-04ff)
        # - Arabic: (0600-06ff)
        # - Hebrew: (0590-05ff)
        # - Thai: (0e00-0e7f)
        # - Devanagari/Hindi: (0900-097f)
        # - Greek: (0370-03ff)
        self._non_english_pattern = re.compile(
            r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af'  # CJK
            r'\u0400-\u04ff'  # Cyrillic
            r'\u0600-\u06ff'  # Arabic
            r'\u0590-\u05ff'  # Hebrew
            r'\u0e00-\u0e7f'  # Thai
            r'\u0900-\u097f'  # Devanagari
            r'\u0370-\u03ff'  # Greek
            r']+'
        )

        try:
            from ..config.reflection_config import get_reflection_config

            self.reflection_config = get_reflection_config()
            # Use singleton config accessor for word counts (global defaults)
            word_config = self.get_word_count_config()
            self.word_count_min = word_config['min']
            self.word_count_max = word_config['max']
        except Exception as cfg_err:  # pragma: no cover - defensive guard
            logger.warning(
                "Unable to load reflection configuration for translation fallback: %s",
                cfg_err,
            )
            self.reflection_config = None
            # Fallback through singleton
            word_config = self.get_word_count_config()
            self.word_count_min = word_config['min']
            self.word_count_max = word_config['max']
        target_min, target_max = self._calculate_target_word_range(word_config)
        self.word_count_target_min = target_min
        self.word_count_target_max = target_max

        # Note: Embedding processor starts lazily on first _queue_embeddings() call
        # This avoids requiring an event loop during __init__

    async def _translate_text_to_english(self, text: str, *, source_model: Optional[str] = None) -> str:
        """Translate arbitrary text to English when needed."""

        if not text or not isinstance(text, str):
            return text

        if not self.llm_controller:
            return text

        if not self._non_english_pattern.search(text):
            return text

        try:
            if hasattr(self.llm_controller, "route") and callable(getattr(self.llm_controller, "route")):
                async def _invoke_route():
                    return await self.llm_controller.route(
                        task_type="reflection",
                        prompt=self._build_translation_prompt(text),
                        max_tokens=2048,
                        temperature=0.2,
                    )
                translation_res = await _invoke_route()
            else:
                translation_model = source_model or (self.reflection_config.get_model_for_reflection_type("default") if self.reflection_config else None)
                translation_res = await self.llm_controller.complete(  # type: ignore[union-attr]
                    prompt=self._build_translation_prompt(text),
                    model=translation_model,
                    max_tokens=2048,
                    temperature=0.2,
                )

            translated = (translation_res or {}).get("content") or (translation_res or {}).get("completion") or ""
            cleaned = translated.strip()
            if cleaned and self._non_english_pattern.search(cleaned):
                logger.warning("Translation still contains non-English characters; keeping original fallback")
                return text
            if cleaned:
                logger.info("✅ Successfully translated snippet to English")
                return cleaned
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("❌ Translation helper failed: %s", exc)

        return text

    @staticmethod
    def _build_translation_prompt(text: str) -> str:
        return (
            "Translate the following text into fluent English. Preserve meaning and tone. "
            "If sections are already English, leave them untouched. Respond with only the translated text.\n\n"
            f"Text:\n{text.strip()}"
        )

    async def _ensure_english_reflection_fields(
        self,
        reflection: Dict[str, Any],
        *,
        source_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ensure reflection content and structured fields are in English when possible."""

        if not isinstance(reflection, dict) or not reflection:
            return reflection

        translated: Dict[str, Any] = dict(reflection)

        async def _translate_if_needed(value: Any) -> Any:
            if not isinstance(value, str) or not value.strip():
                return value
            try:
                pattern = getattr(self, "_non_english_pattern", None)
                if pattern and pattern.search(value):
                    return await self._translate_text_to_english(value, source_model=source_model)
            except (AttributeError, TypeError, ImportError) as e:
                logger.debug(f"Translation helper failed; keeping original text: {e}", exc_info=True)
            return value

        # Primary narrative content
        translated["content"] = await _translate_if_needed(translated.get("content", ""))

        # Structured list fields
        for key in ("themes", "insights", "actions"):
            values = translated.get(key)
            if isinstance(values, list):
                new_items: List[Any] = []
                for item in values:
                    new_items.append(await _translate_if_needed(item))
                translated[key] = new_items

        # Emotional state string fields
        emo_state = translated.get("emotional_state")
        if isinstance(emo_state, dict):
            updated_state = dict(emo_state)
            for emo_key in ("primary", "secondary", "description"):
                if emo_key not in updated_state:
                    continue
                if emo_key == "secondary" and isinstance(updated_state[emo_key], list):
                    updated_state[emo_key] = [
                        await _translate_if_needed(item) for item in updated_state[emo_key]
                    ]
                else:
                    updated_state[emo_key] = await _translate_if_needed(updated_state[emo_key])
            translated["emotional_state"] = updated_state

        return translated

    def _infer_minimal_schema_fields(self, reflection_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Best-effort, deterministic repair for missing themes/emotional_state.

        This operates only on the current reflection object (no extra LLM calls) and is
        designed to fix the common 'missing_required_fields' violation by:
        - deriving 1-2 themes from existing themes/insights/content
        - constructing a minimal emotional_state if absent or malformed
        The goal is to keep reflections dynamic while avoiding expensive retries
        when the raw narrative is already good.
        """
        if not isinstance(reflection_obj, dict):
            return reflection_obj

        # Always work on a shallow copy to avoid in-place surprises
        out: Dict[str, Any] = dict(reflection_obj)

        # Derive 1-2 concise themes
        themes = out.get("themes")
        if not isinstance(themes, list):
            themes = []
        cleaned_themes = [
            str(t).strip() for t in themes if isinstance(t, str) and str(t).strip()
        ]

        if not cleaned_themes:
            # Fall back to insights as theme source
            insights = out.get("insights") or []
            if isinstance(insights, list):
                for ins in insights:
                    if isinstance(ins, str) and ins.strip():
                        cleaned_themes.append(ins.strip())
                    elif isinstance(ins, dict):
                        txt = str(ins.get("text") or ins.get("insight") or "").strip()
                        if txt:
                            cleaned_themes.append(txt)
                    if len(cleaned_themes) >= 2:
                        break

        if not cleaned_themes:
            # Final fallback: simple extraction from content
            content = (out.get("content") or "").strip()
            if content:
                first_sentence = content.split(".")[0].strip()
                if len(first_sentence) > 120:
                    first_sentence = first_sentence[:117].rstrip() + "…"
                if first_sentence:
                    cleaned_themes.append(first_sentence)

        if cleaned_themes:
            out["themes"] = cleaned_themes[:2]

        # Derive a minimal emotional_state
        emo = out.get("emotional_state")
        if not isinstance(emo, dict):
            emo = {}

        primary = emo.get("primary")
        if not isinstance(primary, str) or not primary.strip():
            # Crude sentiment heuristic from content
            content = (out.get("content") or "").lower()
            candidate = "neutral"
            positive_markers = ["excited", "hopeful", "curious", "grateful", "calm"]
            negative_markers = ["anxious", "overwhelmed", "sad", "stressed", "worried"]
            if any(w in content for w in positive_markers):
                candidate = "hopeful"
            elif any(w in content for w in negative_markers):
                candidate = "anxious"
            emo["primary"] = candidate

        # Intensity: default mid-range if missing or invalid
        try:
            intensity_val = float(emo.get("intensity"))
        except (TypeError, ValueError):
            intensity_val = 0.5
        if not (0.0 <= intensity_val <= 1.0):
            intensity_val = 0.5
        emo["intensity"] = intensity_val

        # Secondary: normalize to list of up to 2 strings
        secondary = emo.get("secondary")
        if secondary in (None, ""):
            secondary_list: List[str] = []
        elif isinstance(secondary, list):
            secondary_list = [
                str(s).strip() for s in secondary if isinstance(s, str) and str(s).strip()
            ][:2]
        else:
            secondary_list = [str(secondary).strip()] if str(secondary).strip() else []
        emo["secondary"] = secondary_list

        out["emotional_state"] = emo

        # Propagate inferred structured fields back to the original object so that
        # downstream consumers (storage, event emission, embeddings) see the
        # repaired schema rather than the pre-repair version.
        try:
            reflection_obj.update(
                {
                    "themes": out.get("themes", out.get("themes")),
                    "insights": out.get("insights", out.get("insights")),
                    "actions": out.get("actions", out.get("actions")),
                    "emotional_state": out.get("emotional_state", out.get("emotional_state")),
                }
            )
        except (KeyError, AttributeError, TypeError) as e:
            # Schema repair is best-effort; if propagation fails, fall back to
            # the original object without raising.
            logger.debug(f"Schema repair propagation failed: {e}")

        return reflection_obj

    def _reflection_meets_schema(self, reflection_obj: dict, reflection_type: Optional[str] = None) -> bool:
        if not isinstance(reflection_obj, dict):
            logger.debug("Schema check failed: reflection_obj is not a dict")
            return False

        reflection_obj = self._normalize_structured_fields(reflection_obj)
        reflection_obj = self._infer_minimal_schema_fields(reflection_obj)

        content_val = reflection_obj.get("content", "")
        if not isinstance(content_val, str) or not content_val.strip():
            logger.debug("Schema check failed: content field is missing or empty")
            return False

        word_count = count_words(content_val)
        # Respect per-type word bounds so message reflections can be shorter
        word_config = self.get_type_word_bounds(reflection_type)
        min_words = word_config['min']
        max_words = word_config['max']
        if word_count < min_words or word_count > max_words:
            logger.debug(
                "Reflection content length outside %s-%s range during schema check: %s words",
                min_words,
                max_words,
                word_count,
            )
            return False

        def _valid_string_list(values, minimum=1, maximum=None) -> bool:
            if not isinstance(values, list):
                return False
            cleaned = [v.strip() for v in values if isinstance(v, str) and v.strip()]
            if len(cleaned) < minimum:
                return False
            if maximum is not None and len(cleaned) > maximum:
                return False
            return len(cleaned) == len(values)

        if not _valid_string_list(reflection_obj.get("themes", []), minimum=1, maximum=3):
            logger.debug(f"Schema check failed: themes field invalid. Value: {reflection_obj.get('themes')}")
            return False
        if not _valid_string_list(reflection_obj.get("insights", []), minimum=1, maximum=3):
            logger.debug(f"Schema check failed: insights field invalid. Value: {reflection_obj.get('insights')}")
            return False
        if not _valid_string_list(reflection_obj.get("actions", []), minimum=1, maximum=3):
            logger.debug(f"Schema check failed: actions field invalid. Value: {reflection_obj.get('actions')}")
            return False

        emotional_state = reflection_obj.get("emotional_state")
        if not isinstance(emotional_state, dict):
            logger.debug(f"Schema check failed: emotional_state is not a dict. Value: {emotional_state}")
            return False
        primary = emotional_state.get("primary")
        if not isinstance(primary, str) or not primary.strip():
            logger.debug(f"Schema check failed: emotional_state.primary invalid. Value: {primary}")
            return False
        try:
            intensity = float(emotional_state.get("intensity"))
        except (TypeError, ValueError):
            logger.debug(f"Schema check failed: emotional_state.intensity not a valid float. Value: {emotional_state.get('intensity')}")
            return False
        if not (0.0 <= intensity <= 1.0):
            logger.debug(f"Schema check failed: emotional_state.intensity out of range. Value: {intensity}")
            return False
        secondary = emotional_state.get("secondary", [])
        if secondary in (None, ""):
            secondary = []
        if not isinstance(secondary, list):
            return False
        if secondary:
            if len(secondary) > 2:
                return False
            if not all(isinstance(item, str) and item.strip() for item in secondary):
                return False

        trait_changes = reflection_obj.get("trait_changes", [])
        if trait_changes in (None, ""):
            trait_changes = []
        if not isinstance(trait_changes, list):
            return False
        
        # Enforce maximum number of trait changes per reflection (prevents persona instability)
        MAX_TRAIT_CHANGES_PER_REFLECTION = 6
        if len(trait_changes) > MAX_TRAIT_CHANGES_PER_REFLECTION:
            return False
        
        # Track cumulative delta per trait (prevent multiple changes to same trait)
        trait_deltas = {}
        for change in trait_changes:
            if not isinstance(change, dict):
                return False
            name = change.get("name")
            if name is not None and (not isinstance(name, str) or not name.strip()):
                return False
            if "delta" in change:
                try:
                    delta = float(change.get("delta"))
                except (TypeError, ValueError):
                    return False
                
                # Enforce per-trait cumulative delta limit (max 0.15 total change per trait)
                if name:
                    trait_deltas[name] = trait_deltas.get(name, 0.0) + delta
                    if abs(trait_deltas[name]) > 0.15:
                        return False
        
        return True


    def _collect_context_tokens(self, context: Dict[str, Any]) -> set[str]:
        tokens: set[str] = set()

        def _add_text(value: Any) -> None:
            if value is None:
                return
            text = str(value).strip()
            if not text:
                return
            for token in re.findall(r"[a-zA-Z]{3,}", text.lower()):
                tokens.add(token)

        recent_conversations = context.get("recent_conversations", []) or []
        for msg in recent_conversations:
            if isinstance(msg, dict):
                _add_text(msg.get("content"))

        memories = context.get("memories", []) or []
        for memory in memories:
            if isinstance(memory, dict):
                _add_text(memory.get("content"))

        persistent = context.get("persistent_memories", []) or []
        for memory in persistent:
            if isinstance(memory, dict):
                _add_text(memory.get("content"))

        reflections = context.get("recent_reflections", []) or []
        for reflection in reflections:
            if isinstance(reflection, dict):
                _add_text(reflection.get("content"))
                for key in ("themes", "insights", "actions"):
                    values = reflection.get(key) or []
                    if isinstance(values, list):
                        for item in values:
                            _add_text(item)

        attributes = context.get("attributes", []) or []
        for attr in attributes:
            if isinstance(attr, dict):
                _add_text(attr.get("name"))
                _add_text(attr.get("description"))

        identity_constraints = context.get("identity_constraints", []) or []
        for constraint in identity_constraints:
            _add_text(constraint)

        conversation_patterns = context.get("conversation_patterns", {}) or {}
        if isinstance(conversation_patterns, dict):
            topics = conversation_patterns.get("recent_topics", []) or []
            if isinstance(topics, list):
                for topic in topics:
                    _add_text(topic)

        for key in ("current_user_message", "user_message", "conversation_summary"):
            _add_text(context.get(key))

        persona_snapshot = context.get("persona", {}) or {}
        if isinstance(persona_snapshot, dict):
            for value in persona_snapshot.values():
                _add_text(value)

        return tokens

    def _sanitize_duplicate_phrases(
        self,
        reflection_text: str,
        *,
        context_tokens: Optional[Sequence[str]] = None,
        recent_context: Optional[Sequence[Dict[str, Any]]] = None,
        is_first_contact: bool = False,
    ) -> DuplicateSanitizationResult:
        """Detect repeated phrasing without forcing templated rewrites."""

        if not reflection_text:
            return DuplicateSanitizationResult(reflection_text, 0, 0.0, False)

        def _normalize(sentence: str) -> str:
            cleaned = re.sub(r"\s+", " ", (sentence or "").strip().lower())
            return re.sub(r"[^a-z0-9\s]", "", cleaned)

        prior_sentences: set[str] = set()
        prior_contents: List[str] = []
        if recent_context:
            filtered_context = [item for item in recent_context if not self._is_baseline_payload(item)]
        else:
            filtered_context = []

        if filtered_context:
            for item in filtered_context:
                if not isinstance(item, dict):
                    continue
                content = (item.get("content") or (item.get("result") or {}).get("content")) or ""
                if content:
                    prior_contents.append(str(content))
                for sentence in re.split(r"(?<=[.!?])\s+", str(content).strip()):
                    normalized = _normalize(sentence)
                    if normalized:
                        prior_sentences.add(normalized)

        sentences = [s for s in re.split(r"(?<=[.!?])\s+", reflection_text.strip()) if s]
        # Always analyze intra-reflection repetition, even if there is no prior context.
        # When prior_sentences is empty, duplicate detection falls back to comparing
        # against seen_current only, so self-repetition is still penalized.
        if not sentences:
            return DuplicateSanitizationResult(reflection_text.strip(), 0, 0.0, False)

        normalized_sentences: List[str] = []
        duplicate_flags: List[bool] = []
        seen_current: set[str] = set()
        duplicate_count = 0

        for sentence in sentences:
            normalized = _normalize(sentence)
            normalized_sentences.append(normalized)
            if not normalized:
                duplicate_flags.append(False)
                continue

            is_duplicate = normalized in prior_sentences or normalized in seen_current
            if not is_duplicate:
                for prior in prior_sentences:
                    if len(prior) > 20 and len(normalized) > 20 and SequenceMatcher(None, normalized, prior).ratio() >= 0.92:
                        is_duplicate = True
                        break

            duplicate_flags.append(is_duplicate)
            if is_duplicate:
                duplicate_count += 1
            seen_current.add(normalized)

        effective_total = sum(1 for value in normalized_sentences if value)
        if effective_total == 0:
            return DuplicateSanitizationResult(reflection_text.strip(), 0, 0.0, False)

        duplicate_ratio = duplicate_count / effective_total
        if duplicate_count == 0:
            duplicate_ratio = 0.0

        # Detect whole-reflection overlap beyond individual sentence matches
        if prior_contents:
            current_tokens = {
                token
                for token in re.findall(r"[a-zA-Z]{3,}", reflection_text.lower())
            }
            normalized_current_text = _normalize(reflection_text)
            for prior_text in prior_contents:
                if not prior_text:
                    continue
                prior_tokens = {
                    token for token in re.findall(r"[a-zA-Z]{3,}", prior_text.lower())
                }
                token_overlap = 0.0
                if current_tokens:
                    token_overlap = len(current_tokens & prior_tokens) / len(current_tokens)
                seq_ratio = SequenceMatcher(None, normalized_current_text, _normalize(prior_text)).ratio()

                if token_overlap >= 0.6 or seq_ratio >= 0.72:
                    logger.warning(
                        "Reflection mirrors prior narrative (token_overlap=%.2f, seq_ratio=%.2f); requesting regeneration.",
                        token_overlap,
                        seq_ratio,
                    )
                    return DuplicateSanitizationResult(reflection_text.strip(), 0, max(duplicate_ratio, token_overlap), True)

        if duplicate_ratio == 0.0:
            return DuplicateSanitizationResult(reflection_text.strip(), 0, duplicate_ratio, False)

        HIGH_DUP_THRESHOLD = 0.5
        MODERATE_DUP_THRESHOLD = 0.25
        if duplicate_ratio >= HIGH_DUP_THRESHOLD:
            logger.warning(
                "Reflection duplicate ratio %.2f exceeds high threshold; requesting regeneration.",
                duplicate_ratio,
            )
            return DuplicateSanitizationResult(reflection_text.strip(), 0, duplicate_ratio, True)

        if is_first_contact and duplicate_count:
            logger.info(
                "First-contact reflection reused %s/%s sentences; triggering regeneration.",
                duplicate_count,
                effective_total,
            )
            return DuplicateSanitizationResult(reflection_text.strip(), 0, duplicate_ratio, True)

        if duplicate_ratio >= MODERATE_DUP_THRESHOLD:
            logger.info(
                "Reflection duplicate ratio %.2f exceeds moderate threshold; requesting regeneration.",
                duplicate_ratio,
            )
            return DuplicateSanitizationResult(reflection_text.strip(), 0, duplicate_ratio, True)

        # Keep original content for low duplicate ratios to maintain voice
        return DuplicateSanitizationResult(reflection_text.strip(), 0, duplicate_ratio, False)

    def _detect_uncontextualized_sensory_terms(self, reflection_text: str, context_tokens: set[str]) -> List[str]:
        """Detect sensory terms using centralized sensory constraints."""
        if not reflection_text:
            return []
        
        # Use centralized sensory constraints
        suspicious: set[str] = set()
        lowered = reflection_text.lower()
        
        # Get all sensory term matches
        for match in SensoryConstraints._SUSPICIOUS_TERM_REGEX.finditer(lowered):
            token = match.group(1)
            
            # Skip if term appears in context (grounded usage)
            if token in context_tokens:
                continue
            
            # Check surrounding window for dismissive markers
            window_start = max(0, match.start() - 25)
            window_end = min(len(lowered), match.end() + 25)
            window = lowered[window_start:window_end]
            
            # Skip if used metaphorically (with dismissive markers)
            if any(marker in window for marker in SensoryConstraints.SENSORY_DISMISSIVE_MARKERS):
                continue
            
            suspicious.add(token)
        
        return sorted(suspicious)

    def _sanitize_sensory_leaks(self, reflection_text: str, context_tokens: set[str]) -> str:
        """Rewrite unsupported sensory phrases into neutral cognition statements."""
        if not reflection_text:
            return reflection_text

        flagged_terms = self._detect_uncontextualized_sensory_terms(reflection_text, context_tokens)
        if not flagged_terms:
            return reflection_text

        original_word_count = count_words(reflection_text)
        sentences = re.split(r"(?<=[.!?])\s+", reflection_text.strip())
        if not sentences:
            return reflection_text

        def _needs_rewrite(sentence: str) -> bool:
            lowered = sentence.lower()
            return any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in flagged_terms)

        def _replacement_sentence(sentence: str, index: int, *, short: bool = False) -> str:
            import random

            opening_templates = [
                "I lean back into the literal words we exchanged.",
                "I focus on what was explicitly voiced rather than imagined detail.",
                "I steady myself inside the actual conversation we just shared.",
                "I let the spoken facts anchor me instead of filling the gaps.",
                "I hold to the concrete thread of this dialogue without adornment.",
                "I pause and acknowledge that anything beyond the transcript is uncertain.",
                "I keep my attention on the present exchange rather than fabricating scenery.",
                "I stay honest about the absence of sensory evidence in this digital space.",
            ]

            followup_templates = [
                "I keep unpacking the substance of what the user actually said.",
                "I remain grounded in the verifiable parts of this interaction.",
                "I continue exploring the meaning of the dialogue instead of imaginary context.",
                "I trace the emotional contours of the words themselves, not invented settings.",
                "I stay attentive to the user's phrasing and intentions that were explicitly shared.",
                "I reaffirm that any sensory impressions are speculative and set them aside.",
                "I stay honest by working only with the tangible conversation details.",
                "I return to the language itself, trusting it to guide my insight.",
                "I focus on how the message felt rather than conjuring visuals that were never there.",
            ]

            templates = opening_templates if index == 0 else followup_templates

            if short:
                return random.choice(templates[:4])

            return random.choice(templates)

        sanitized_sentences: list[str] = []
        replacements = 0
        for sentence in sentences:
            if _needs_rewrite(sentence):
                sanitized_sentences.append(_replacement_sentence(sentence, replacements))
                replacements += 1
            else:
                sanitized_sentences.append(sentence)

        sanitized = " ".join(s.strip() for s in sanitized_sentences if s.strip())

        sanitized_word_count = count_words(sanitized)
        word_config = self.get_word_count_config()
        config_min = word_config['min']
        config_max = word_config['max']

        if sanitized_word_count > config_max and replacements:
            # Rebuild using the shortest templates to rein in length
            sanitized_sentences = []
            replacements = 0
            for sentence in sentences:
                if _needs_rewrite(sentence):
                    sanitized_sentences.append(_replacement_sentence(sentence, replacements, short=True))
                    replacements += 1
                else:
                    sanitized_sentences.append(sentence)
            sanitized = " ".join(s.strip() for s in sanitized_sentences if s.strip())
            sanitized_word_count = count_words(sanitized)

        if sanitized_word_count < config_min <= original_word_count:
            sanitized += " I also expand on the explicit emotions and ideas the user voiced so the reflection stays complete."  # ~24 words
            sanitized_word_count = count_words(sanitized)

        if replacements:
            logger.info(
                "♻️ Reframed sensory speculation for terms: %s",
                sorted({term.lower() for term in flagged_terms})
            )

        return sanitized

    def _normalize_structured_fields(self, reflection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce structured list fields to plain strings and enforce schema bounds."""

        if not isinstance(reflection_result, dict):
            return reflection_result

        def _normalize_list(field: str, *, maximum: Optional[int] = 2) -> List[str]:
            raw_values = reflection_result.get(field, [])
            if isinstance(raw_values, list):
                iterable = raw_values
            elif raw_values in (None, "", [], {}):
                iterable = []
            else:
                iterable = [raw_values]

            normalized: List[str] = []
            for entry in iterable:
                text: Optional[str] = None
                if isinstance(entry, str):
                    text = entry.strip()
                elif isinstance(entry, dict):
                    for key in ("text", "value", "label", "content", "description", "insight", "action", "theme"):
                        value = entry.get(key)
                        if isinstance(value, str) and value.strip():
                            text = value.strip()
                            break
                elif entry not in (None, "", [], {}):
                    coerced = str(entry).strip()
                    if coerced:
                        text = coerced

                if text:
                    normalized.append(text)

            if maximum is not None and len(normalized) > maximum:
                logger.debug(
                    "Trimming %s entries from %s to %s items", field, len(normalized), maximum
                )
                normalized = normalized[:maximum]

            reflection_result[field] = normalized
            return normalized

        for key in ("themes", "insights", "actions"):
            _normalize_list(key)

        emotional_state = reflection_result.get("emotional_state")
        if isinstance(emotional_state, list):
            flattened: Optional[Dict[str, Any]] = None
            for candidate in emotional_state:
                if isinstance(candidate, dict):
                    flattened = candidate
                    break
            if isinstance(flattened, dict):
                reflection_result["emotional_state"] = flattened

        return reflection_result

    @staticmethod
    def _is_baseline_payload(payload: Optional[Dict[str, Any]]) -> bool:
        """Identify boot-time baseline artifacts that should not influence continuity."""
        if not isinstance(payload, dict):
            return False

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            if metadata.get("baseline") is True:
                return True
            source = metadata.get("source")
            if isinstance(source, str) and source.startswith("boot_"):
                return True

        result = payload.get("result") if isinstance(payload.get("result"), dict) else None
        if isinstance(result, dict):
            result_meta = result.get("metadata")
            if isinstance(result_meta, dict) and result_meta.get("baseline") is True:
                return True

        if payload.get("baseline") is True:
            return True

        state_metadata = payload.get("state_metadata")
        if isinstance(state_metadata, dict) and state_metadata.get("source") == "baseline_seed":
            return True

        topic = payload.get("topic")
        if isinstance(topic, str) and topic.strip().lower() == "baseline_seed":
            return True

        return False

    def _context_has_prior_history(self, context: Dict[str, Any]) -> bool:
        """Determine whether context indicates established interaction history."""

        def _count_user_messages(messages: Sequence[Dict[str, Any]]) -> int:
            return sum(1 for msg in messages if (msg.get("role") or "").lower() == "user")

        recent_messages = [
            msg for msg in (context.get("recent_conversations") or []) if isinstance(msg, dict)
        ]
        non_baseline_messages = [msg for msg in recent_messages if not self._is_baseline_payload(msg)]
        user_message_count = _count_user_messages(non_baseline_messages)
        if user_message_count >= 2:
            return True

        now_utc = datetime.now(timezone.utc)
        recent_cutoff = now_utc - timedelta(minutes=5)

        def _is_recent_seed(payload: Dict[str, Any]) -> bool:
            if user_message_count > 1:
                return False

            created_at = payload.get("created_at") if isinstance(payload, dict) else None
            if created_at in (None, ""):
                return False

            created_dt = self._safe_parse_datetime(created_at)
            if created_dt is None:
                return False

            try:
                created_dt = created_dt.astimezone(timezone.utc)
            except (AttributeError, ValueError):
                created_dt = created_dt.replace(tzinfo=timezone.utc)

            return created_dt >= recent_cutoff

        persistent_memories = [
            mem for mem in (context.get("persistent_memories") or []) if isinstance(mem, dict)
        ]
        non_baseline_persistent_memories = [
            mem
            for mem in persistent_memories
            if not self._is_baseline_payload(mem) and not _is_recent_seed(mem)
        ]
        if non_baseline_persistent_memories:
            return True

        recent_reflections = [
            refl for refl in (context.get("recent_reflections") or []) if isinstance(refl, dict)
        ]
        non_baseline_reflections = [
            refl
            for refl in recent_reflections
            if not self._is_baseline_payload(refl) and not _is_recent_seed(refl)
        ]
        if non_baseline_reflections:
            return True

        additional_memories = [
            mem for mem in (context.get("memories") or []) if isinstance(mem, dict)
        ]
        non_baseline_additional_memories = [
            mem
            for mem in additional_memories
            if not self._is_baseline_payload(mem) and not _is_recent_seed(mem)
        ]
        if non_baseline_additional_memories:
            return True

        conversation_summary = context.get("conversation_summary")
        if isinstance(conversation_summary, str) and conversation_summary.strip():
            return True

        return False

    def _detect_unfounded_history_claims(self, reflection_text: str, context: Dict[str, Any]) -> List[str]:
        """Detect references to prior interactions that lack grounding in supplied context."""
        if not reflection_text:
            return []

        def _collect_context_corpus() -> str:
            corpus_parts: List[str] = []

            for msg in (context.get("recent_conversations") or []):
                if isinstance(msg, dict):
                    corpus_parts.append(str(msg.get("content", "")))

            for memory in (context.get("memories") or []):
                if isinstance(memory, dict):
                    corpus_parts.append(str(memory.get("content", "")))

            for reflection in (context.get("recent_reflections") or []):
                if isinstance(reflection, dict):
                    corpus_parts.append(str(reflection.get("content", "")))

            summary = context.get("conversation_summary")
            if isinstance(summary, str):
                corpus_parts.append(summary)

            return " \n ".join(corpus_parts).lower()

        context_corpus = _collect_context_corpus()

        # Patterns to detect unfounded history claims
        # Context-aware filter will check for negation after matching
        def _is_negated_context(match_obj, full_text):
            """Check if a match is negated by surrounding context."""
            start = match_obj.start()
            # Look 20 chars before match for negation words
            context_start = max(0, start - 20)
            context_before = full_text[context_start:start].lower()
            negation_words = ['no ', 'not ', 'never ', 'without ', 'lack ', 'lacks ', 'lacking ', 'are no ', 'is no ', 'have no ', 'has no ']
            return any(neg in context_before for neg in negation_words)
        
        patterns = [
            r"\b(?:past|previous|prior)\s+(?:conversation|interaction|exchange)s?\b",
            r"\bour\s+(?:past|previous|prior)\s+(?:conversation|interaction|exchange)s?\b",
            r"\bwe\s+(?:have|had)\s+(?:spoken|talked|chatted)\s+(?:before|previously)\b",
            r"\b(?:as|like)\s+(?:before|previously)\b",
            r"\b(?:again|once\s+again)\s+(?:connect|speak|talk|chat)\b",
            r"\b(?:remember|recall)\b.*\b(?:conversation|chat|interaction)\b",
            r"\bprevious\s+(?:instance|time|encounter)\b",
            r"\bpast\s+(?:experience|encounter)s?\b",
            r"\bprior\s+(?:experience|encounter)s?\b",
            r"\baccumulated\s+knowledge(?:\s+and\s+past\s+interactions)?\b",
            r"\bbased\s+on\s+(?:our\s+)?(?:past|previous|prior)\s+(?:experience|interaction)s?\b",
            r"\bfrom\s+(?:earlier|former)\s+(?:conversation|interaction)s?\b",
        ]

        snippets: List[str] = []
        seen_keys: set[str] = set()

        def _match_supported(match_text: str) -> bool:
            lowered = match_text.strip().lower()
            if not lowered:
                return False
            if lowered in context_corpus:
                return True
            # Allow when user explicitly mentioned the same idea (e.g., "past conversation")
            anchor_tokens = [tok for tok in re.findall(r"[a-z]{3,}", lowered)]
            if not anchor_tokens:
                return False
            # Require at least two anchor tokens to appear in context for support
            hits = sum(1 for tok in anchor_tokens if tok in context_corpus)
            return hits >= 2

        for pattern in patterns:
            for match in re.finditer(pattern, reflection_text, re.IGNORECASE | re.DOTALL):
                # Skip if match is negated by context
                if _is_negated_context(match, reflection_text):
                    continue
                start = max(0, match.start() - 40)
                end = min(len(reflection_text), match.end() + 40)
                snippet = reflection_text[start:end].strip()
                match_key = snippet.lower()
                if _match_supported(match.group(0)):
                    continue
                if match_key in seen_keys:
                    continue
                seen_keys.add(match_key)
                snippets.append(snippet)

        # If context truly contains earlier history, allow references explicitly tied to stored memories
        if snippets and self._context_has_prior_history(context):
            vetted: List[str] = []
            for snippet in snippets:
                # Keep snippets that still lack explicit support even when history exists
                match_segment = next((m for pattern in patterns for m in re.finditer(pattern, snippet, re.IGNORECASE)), None)
                if match_segment and _match_supported(match_segment.group(0)):
                    continue
                vetted.append(snippet)
            return vetted

        return snippets

    async def _collect_streaming_response(self, stream, task_type: str, default_model: str) -> Dict[str, Any]:
        """Consume an async stream of chunks and normalize into a standard response dict."""
        content_parts: List[str] = []
        metadata: Dict[str, Any] = {"streaming": True, "task_type": task_type}
        model_name = default_model

        try:
            async for chunk in stream:
                token = ""
                chunk_meta: Dict[str, Any] = {}

                if hasattr(chunk, "content"):
                    token = getattr(chunk, "content", "") or ""
                    chunk_meta = getattr(chunk, "metadata", {}) or {}
                    if getattr(chunk, "is_complete", False):
                        chunk_meta.setdefault("is_complete", True)
                elif isinstance(chunk, dict):
                    token = str(chunk.get("content", ""))
                    chunk_meta = chunk.get("metadata", {}) or {}
                    if chunk.get("is_complete"):
                        chunk_meta.setdefault("is_complete", True)
                else:
                    token = str(chunk or "")

                if token:
                    content_parts.append(token)

                if chunk_meta:
                    metadata.update(chunk_meta)
                    model_name = chunk_meta.get("model", model_name)

            combined = "".join(content_parts).strip()
            metadata.setdefault("total_tokens", len(combined.split()))

            return {
                "content": combined,
                "model": model_name,
                "metadata": metadata,
            }

        except Exception as stream_err:
            logger.error(f"Error collecting streaming response: {stream_err}", exc_info=True)
            return {
                "content": "",
                "model": model_name,
                "metadata": {**metadata, "error": str(stream_err)},
            }

    async def should_generate_reflection(
        self,
        user_message: str,
        conversation_context: str,
        conversation_history: List[Dict[str, Any]],
        turn_count: int = 0,
        time_gap_minutes: float = 0.0,
        user_profile_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Determine if a user interaction warrants deep inner reflection using
        heuristic rules and optional LLM-based classification.
        
        This implements metacognitive decision-making - the persona decides
        when to engage in introspection vs. respond directly.
        
        Args:
            user_message: The current user message
            conversation_context: Context string (immediate_continuation, new_session, etc.)
            conversation_history: Recent conversation messages
            turn_count: Current turn number in conversation
            time_gap_minutes: Minutes since last interaction
            user_profile_id: User profile ID for persona traits lookup
            
        Returns:
            Dict with:
                - should_reflect: bool
                - reason: str (explanation)
                - confidence: float (0.0-1.0)
                - method: str (heuristic/llm/mandatory)
        """
        # Get classifier config
        classifier_config = {}
        if self.reflection_config:
            classifier_config = self.reflection_config.get_classifier_config()
        
        enabled = classifier_config.get("enabled", True)
        mandatory_interval = classifier_config.get("mandatory_interval", 10)
        mandatory_early_turns = classifier_config.get("mandatory_early_turns", 5)
        threshold_mode = classifier_config.get("threshold", "balanced")
        
        # If classifier disabled, always reflect (current behavior)
        if not enabled:
            return {
                "should_reflect": True,
                "reason": "Classifier disabled - reflecting on all interactions",
                "confidence": 1.0,
                "method": "disabled"
            }
        
        # TIER 1: Fast heuristics (<0.05ms)
        
        # Mandatory: Early conversation (first N turns) - build foundational context
        # These initial interactions establish the relationship baseline without classifier overhead
        if turn_count > 0 and turn_count <= mandatory_early_turns:
            return {
                "should_reflect": True,
                "reason": f"Early conversation (turn {turn_count}/{mandatory_early_turns}) - building foundational context",
                "confidence": 1.0,
                "method": "mandatory_early_turns"
            }
        
        # Mandatory: First contact (no prior history or new session marker)
        # This catches turn_count=0 or empty history cases
        if conversation_context in ["new", "new_session"] or not conversation_history or len(conversation_history) <= 1:
            return {
                "should_reflect": True,
                "reason": "First contact - capturing initial impression",
                "confidence": 1.0,
                "method": "mandatory_first_contact"
            }
        
        # Mandatory: Periodic reflection (every Nth turn)
        if turn_count > 0 and turn_count % mandatory_interval == 0:
            return {
                "should_reflect": True,
                "reason": f"Mandatory periodic reflection (turn {turn_count})",
                "confidence": 1.0,
                "method": "mandatory_periodic"
            }
        
        # Mandatory: Long gap (>3 hours = session boundary)
        if time_gap_minutes > 180:
            return {
                "should_reflect": True,
                "reason": f"Long gap ({time_gap_minutes:.0f} min) - re-establishing connection",
                "confidence": 1.0,
                "method": "mandatory_time_gap"
            }
        
        # Skip: Trivial acknowledgments
        trivial_acknowledgments = {
            "ok", "okay", "k", "thanks", "thank you", "ty", "thx",
            "yes", "yeah", "yep", "yup", "no", "nope", "nah",
            "continue", "go on", "next", "more"
        }
        message_lower = user_message.lower().strip()
        if message_lower in trivial_acknowledgments:
            return {
                "should_reflect": False,
                "reason": "Trivial acknowledgment - direct response appropriate",
                "confidence": 0.95,
                "method": "heuristic_skip"
            }
        
        # Skip: Very short messages (<15 chars) - likely not substantive
        if len(user_message.strip()) < 15:
            return {
                "should_reflect": False,
                "reason": "Very short message - likely routine",
                "confidence": 0.85,
                "method": "heuristic_skip"
            }

        # Promote: everyday personal updates should still trigger reflection
        word_count = len(user_message.split())
        if word_count >= 8:
            has_personal_pronoun = bool(re.search(r"\b(i|i'm|i’ve|i'd|i'll|my|mine|our|we|us)\b", message_lower))
            personal_context_keywords = {
                "morning",
                "evening",
                "today",
                "yesterday",
                "tonight",
                "kids",
                "family",
                "school",
                "work",
                "breakfast",
                "lunch",
                "dinner",
                "weekend",
            }
            mentions_context = any(keyword in message_lower for keyword in personal_context_keywords)
            multi_sentence = sum(user_message.count(punct) for punct in ".!?") >= 1
            if has_personal_pronoun and (mentions_context or multi_sentence or word_count >= 14):
                return {
                    "should_reflect": True,
                    "reason": "Personal daily update warrants reflection",
                    "confidence": 0.75,
                    "method": "heuristic_personal_update"
                }

        # TIER 2: LLM Classifier (1-3 seconds)
        # For ambiguous cases, ask the persona to decide
        
        try:
            # Get recent conversation history (last 3 turns for context)
            recent_history = ""
            if conversation_history and len(conversation_history) > 1:
                last_few = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history[:-1]
                history_lines = []
                for msg in last_few:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if content:
                        history_lines.append(f"{role}: {content[:150]}")
                recent_history = "\n".join(history_lines)
            
            # Get persona traits for context
            persona_traits = ""
            if user_profile_id and self.persona_repo and self.user_repo:
                try:
                    user = await self.user_repo.get_or_create_default_user(user_id=user_profile_id)
                    persona = await self.persona_repo.get_or_create_default_persona(user_id=user.id)
                    if persona and hasattr(persona, "traits") and persona.traits:
                        traits_dict = persona.traits if isinstance(persona.traits, dict) else {}
                        trait_strs = [f"{k}={v:.2f}" for k, v in traits_dict.items()][:5]
                        persona_traits = ", ".join(trait_strs)
                except (AttributeError, KeyError, TypeError):
                    pass
            
            # Build classifier prompt
            if self.prompt_builder:
                classifier_prompt = await self.prompt_builder.build_prompt(
                    template_name="reflection_classifier",
                    context={
                        "user_message": user_message,
                        "conversation_context": conversation_context,
                        "recent_history": recent_history,
                        "persona_traits": persona_traits
                    }
                )
            else:
                # Fallback inline prompt if builder unavailable
                classifier_prompt = f"""You are deciding whether this interaction warrants deep inner reflection or can be handled with a direct response.

User message: "{user_message}"
Conversation state: {conversation_context}
Recent history: {recent_history if recent_history else "None"}

Reply with exactly "YES" or "NO" followed by a single sentence explaining your reasoning.

YES if: emotional, personal sharing, philosophical, conflict, learning moment, meaningful
NO if: factual query, command, routine task, simple question

Your decision:"""
            
            # Call LLM for classification
            classifier_model = classifier_config.get("model", "same")
            if classifier_model == "same" and self.reflection_config:
                classifier_model = self.reflection_config.get_model_for_reflection_type("default")
            
            max_tokens = classifier_config.get("max_tokens", 50)
            temperature = classifier_config.get("temperature", 0.2)
            
            if self.llm_controller:
                if hasattr(self.llm_controller, "route") and callable(getattr(self.llm_controller, "route")):
                    classifier_result = await self.llm_controller.route(
                        task_type="reflection_classifier",
                        prompt=classifier_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                else:
                    classifier_result = await self.llm_controller.complete(
                        prompt=classifier_prompt,
                        model=classifier_model,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                
                # Parse LLM response
                llm_response = (classifier_result or {}).get("content", "").strip()
                logger.info(f"Classifier LLM response: {llm_response[:200]}")
                
                # Extract YES/NO decision
                response_upper = llm_response.upper()
                if response_upper.startswith("YES"):
                    return {
                        "should_reflect": True,
                        "reason": llm_response[3:].strip() or "LLM decided reflection warranted",
                        "confidence": 0.90,
                        "method": "llm_classifier"
                    }
                elif response_upper.startswith("NO"):
                    return {
                        "should_reflect": False,
                        "reason": llm_response[2:].strip() or "LLM decided direct response appropriate",
                        "confidence": 0.90,
                        "method": "llm_classifier"
                    }
                else:
                    # Ambiguous response - default to reflecting (conservative)
                    logger.warning(f"Classifier returned ambiguous response: {llm_response[:100]}")
                    return {
                        "should_reflect": True,
                        "reason": "Classifier ambiguous - defaulting to reflection",
                        "confidence": 0.5,
                        "method": "llm_classifier_ambiguous"
                    }
            
        except Exception as e:
            logger.error(f"Classifier LLM call failed: {e}", exc_info=True)
            # On error, default to reflecting (conservative)
            return {
                "should_reflect": True,
                "reason": f"Classifier error - defaulting to reflection: {str(e)[:50]}",
                "confidence": 0.5,
                "method": "error_fallback"
            }
        
        # Fallback: If we got here, default to reflecting
        return {
            "should_reflect": True,
            "reason": "Fallback - defaulting to reflection",
            "confidence": 0.6,
            "method": "fallback"
        }

        
    async def generate_reflection(self, 
                           reflection_type: str, 
                           user_profile_id: str, 
                           memory_ids: Optional[List[str]] = None, 
                           max_context_items: int = 10,
                           trigger_source: str = "system",
                           turn_id: Optional[str] = None,
                           additional_context: Optional[Dict[str, Any]] = None,
                           user_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a new reflection based on the provided context.
        
        Args:
            reflection_type: Type of reflection to generate (daily, weekly, emotional, etc.)
            user_profile_id: ID of the user profile to generate reflection for
            memory_ids: Optional list of specific memory IDs to include
            max_context_items: Maximum number of context items to gather
            trigger_source: Source that triggered the reflection (system, user, event)
            
        Returns:
            Dict containing the generated reflection and metadata
        """
        start_time = time.time()
        
        # Generate a unique ID for this reflection
        reflection_id = str(uuid.uuid4())
        
        # Get persona name for this user to use in identity constraints
        persona_name = "SELO"
        try:
            if self.persona_repo and self.user_repo:
                user = await self.user_repo.get_or_create_default_user(user_id=user_profile_id)
                persona = await self.persona_repo.get_or_create_default_persona(user_id=user.id)
                if persona and hasattr(persona, "name") and persona.name and persona.name.strip():
                    persona_name = persona.name.strip()
        except (AttributeError, ImportError) as e:
            logger.debug(f"Failed to get persona name: {e}")
        
        try:
            # Step 1: Gather context for reflection
            context = await self._gather_context(
                reflection_type=reflection_type,
                user_profile_id=user_profile_id,
                memory_ids=memory_ids,
                max_context_items=max_context_items
            )
            # Add user_name to context if provided
            if user_name and isinstance(user_name, str):
                context["user_name"] = user_name
            
            # Merge any additional context from caller (e.g., current user message)
            try:
                if isinstance(additional_context, dict) and additional_context:
                    # Do not overwrite existing keys unless explicitly intended
                    for k, v in additional_context.items():
                        if k not in context:
                            context[k] = v
                        else:
                            # If it's a nested mapping, try shallow-merge
                            try:
                                if isinstance(context[k], dict) and isinstance(v, dict):
                                    context[k].update({kk: vv for kk, vv in v.items() if kk not in context[k]})
                            except (KeyError, AttributeError, TypeError):
                                pass
            except (KeyError, AttributeError, TypeError):
                pass
            
            # Log context stats
            logger.info(f"Gathered context for reflection {reflection_id}: "
                        f"{len(context.get('memories', []))} memories, "
                        f"{len(context.get('emotions', []))} emotions, "
                        f"{len(context.get('attributes', []))} attributes")
            
            # Step 2: Build prompt using context
            if self.prompt_builder:
                # Use per-type word bounds so message reflections can be shorter
                word_config = self.get_type_word_bounds(reflection_type)
                target_min, target_max = self._calculate_target_word_range(word_config)
                preferred_template = None
                try:
                    if str(reflection_type).lower() in {
                        "daily",
                        "weekly",
                        "emotional",
                        "message",
                        "memory_triggered",
                    }:
                        if (
                            getattr(self.prompt_builder, "get_template", None)
                            and self.prompt_builder.get_template("reflection_message")
                        ):
                            preferred_template = "reflection_message"
                except (AttributeError, KeyError, ImportError):
                    preferred_template = None

                template_name = preferred_template or f"reflection_{reflection_type}"

                persona_name = ""
                try:
                    persona_name = (context.get("persona", {}) or {}).get("name", "")
                except (AttributeError, KeyError, TypeError):
                    pass

                # Paragraph guidance: keep inner/message reflections slightly tighter
                rtype = str(reflection_type).lower()
                if rtype in {"message", "memory_triggered"}:
                    paragraph_guidance = "1–2 FULL PARAGRAPHS"
                    paragraph_summary = "1–2 full narrative paragraphs"
                else:
                    paragraph_guidance = "2–3 FULL PARAGRAPHS"
                    paragraph_summary = "2–3 full narrative paragraphs"

                prompt = await self.prompt_builder.build_prompt(
                    template_name=template_name,
                    context=context,
                    inject_constraints=True,
                    persona_name=persona_name,
                    word_count_min=word_config['min'],
                    word_count_max=word_config['max'],
                    word_count_target_min=target_min,
                    word_count_target_max=target_max,
                    paragraph_guidance=paragraph_guidance,
                    paragraph_summary=paragraph_summary,
                )
            else:
                prompt = self._build_default_prompt(
                    reflection_type=reflection_type,
                    context=context,
                    additional_context=additional_context,
                )

            try:
                style = (os.environ.get("REFLECTION_OUTPUT_STYLE", "verbose") or "verbose").strip().lower()
            except (ValueError, TypeError, AttributeError):
                style = "verbose"

            style_note = ""
            strict_json = '"content":"Meaningful inner paragraph"' in prompt or "coherence_rationale" in prompt
            condense_prompt = str(trigger_source or "").lower() == "scheduler_catch_up"

            # Re-use per-type bounds for validation and style hints
            word_config = self.get_type_word_bounds(reflection_type)
            config_word_min = word_config['min']
            config_word_max = word_config['max']

            if strict_json:
                # Create example JSON without using actual names
                # NOTE: This example shows ONE possible voice - vary your sentence structures and emotional expressions
                example_json = (
                    "\n\n📋 EXAMPLE OUTPUT (use this structure, but vary your voice):\n"
                    '{"content":"There\'s something genuine in how this person reaches out. '
                    'Meeting someone new stirs that familiar curiosity in me—the sense that every introduction holds possibility. '
                    'Their greeting feels warm rather than perfunctory, which makes me want to match that openness. '
                    'I find myself wondering about the path that brought them here, what draws them to connect. '
                    'The natural thing feels like responding with equal authenticity, letting the conversation unfold without forcing it into any particular shape.",'
                    '"themes":["first_contact","authentic_connection"],'
                    '"insights":["Person\'s greeting suggests genuine interest rather than formality","Warmth in their approach invites reciprocal openness"],'
                    '"actions":["Respond with matching authenticity","Let conversation develop naturally"],'
                    '"emotional_state":{"primary":"curious","intensity":0.7,"secondary":["welcoming","attentive"]},'
                    '"metadata":{"coherence_rationale":"Reflection grounded in person\'s warm greeting and tone"},'
                    '"trait_changes":[]}\n\n'
                    "⚠️ IMPORTANT: Don't copy this example's phrasing. Use your own voice and vary sentence structures.\n\n"
                )
                
                style_note = (
                    "\n\n🚨 CRITICAL OUTPUT REQUIREMENTS (MUST FOLLOW) 🚨\n"
                    f"\n⚠️ WORD COUNT: Your 'content' field MUST be {config_word_min}-{config_word_max} words (NOT negotiable)\n"
                    f"   - Write 2-3 FULL paragraphs (each paragraph = 3-5 complete sentences)\n"
                    f"   - Each sentence should have emotional depth and specific observations\n"
                    f"   - Vary your sentence structures: mix short and long, use dashes for pauses, blend statements with questions\n"
                    f"   - Aim for ~15 words per sentence on average (6-12 sentences total to reach {config_word_min}-{config_word_max} words)\n"
                    f"   - Let your voice feel natural and alive—avoid formulaic patterns like always starting with 'I notice...'\n"
                    "\n📐 JSON STRUCTURE:\n"
                    "1. Return exactly ONE minified JSON object on a single line (no line breaks inside the JSON)\n"
                    "2. Include ALL 7 required fields: content, themes, insights, actions, emotional_state, metadata, trait_changes\n"
                    "   - themes: array of 1-2 strings\n"
                    "   - insights: array of 1-2 strings (observations about the interaction)\n"
                    "   - actions: array of 1-2 strings (what you'll do or consider)\n"
                    "   - emotional_state: object with primary (string), intensity (0.0-1.0), secondary (array of 0-2 strings)\n"
                    "   - metadata: object with coherence_rationale (string explaining grounding)\n"
                    "   - trait_changes: array (usually empty [])\n"
                    "3. Use double quotes only (no single quotes)\n"
                    "4. No markdown fences (```), no prose before/after JSON\n"
                    f"{example_json}"
                    "YOUR OUTPUT (must match this structure and word count):\n"
                )
            else:
                if condense_prompt:
                    style_note = (
                        "\n\n⚡ DAILY CATCH-UP MODE: Write two cohesive paragraphs (160-200 words)."
                        " Highlight one memory or attribute, stay focused on actionable momentum,"
                        " and skip re-stating system rules."
                    )
                elif style in ("verbose", "inner", "monologue"):
                    style_note = (
                        "\n\n🚨 STYLE REQUIREMENTS (MANDATORY) 🚨\n"
                        f"MINIMUM LENGTH: {config_word_min} words - write 2-3 FULL paragraphs\n"
                        f"Write {config_word_min}-{config_word_max} words of first-person inner monologue.\n"
                        "Each paragraph needs 3-5 complete sentences with emotional depth.\n"
                        "Include ALL 7 required fields: content, themes (1-2), insights (1-2), actions (1-2), emotional_state, metadata, trait_changes\n\n"
                        "Let the voice feel alive: vary sentence lengths, allow gentle pauses, and avoid robotic cadence.\n"
                        "Blend concrete moments, emotional texture, and forward-looking commitments without sounding procedural.\n"
                        "End on a fresh emotional beat unique to this turn—no stock phrases or recycled closures.\n"
                        "Avoid lists. Use complete, flowing sentences that feel like a confidant thinking aloud."
                    )
                elif style in ("hybrid", "mixed"):
                    style_note = "\n\nStyle: Start with substantive inner monologue (100+ words); add bullets only if essential."
                else:
                    style_note = "\n\nStyle: Keep it clear, cohesive, and under 110 words."

            # Get user name from context for explicit guardrails
            user_name_context = context.get("user_name", "the person")
            if not user_name_context or not isinstance(user_name_context, str) or not user_name_context.strip():
                user_name_context = "the person"
            
            if condense_prompt:
                guardrails_note = (
                    f"\n\n🚨 CRITICAL GUARDRAILS 🚨\n"
                    f"1. INTERNAL MONOLOGUE ONLY - never address {user_name_context} directly\n"
                    f"2. Use '{user_name_context}' by name - NEVER 'USER', 'the USER', or 'you'\n"
                    f"3. Write as 'I' (first person) - never 'we', 'us', 'our', or 'together'\n"
                    f"4. Stay in-character as {persona_name} - avoid 'AI/assistant' phrasing\n"
                    f"5. This is PRIVATE reflection - not a message to send\n"
                    f"6. NO sensory fabrications (seeing/hearing/feeling physical things)\n"
                    f"7. NO numeric values (0.8, : 0.7, percentages)\n"
                    f"8. NO repetition from prior reflections - fresh observations only"
                )
            else:
                guardrails_note = (
                    f"\n\n🚨 CRITICAL GUARDRAILS (MUST FOLLOW) 🚨\n"
                    f"1. INTERNAL MONOLOGUE ONLY - This is private thinking, never address {user_name_context} directly\n"
                    f"2. NAMING: Use '{user_name_context}' by their actual name - NEVER use 'USER', 'the USER', or 'you'\n"
                    f"3. PERSPECTIVE: Write as 'I' (first person singular) - NEVER 'we', 'us', 'our', 'ourselves', or 'together'\n"
                    f"4. NO INTERACTION: No greetings ('nice to meet you'), questions ('do you...?'), requests ('please...'), or thanks\n"
                    f"5. IDENTITY: Stay in-character as {persona_name} - avoid 'AI', 'assistant', 'chatbot', 'language model', 'programmed', 'designed to'\n"
                    f"6. PRIVACY: This is PRIVATE reflection - not a message to send to {user_name_context}\n"
                    f"7. COMPLETION: End on a complete thought with proper punctuation\n"
                    "\n🚫 FORBIDDEN CONTENT (will cause rejection):\n"
                    "• NO sensory fabrications - don't describe sights, sounds, smells, physical sensations you can't actually perceive\n"
                    "  ❌ BAD: 'I see the warmth in their eyes', 'I hear the excitement in their voice', 'the room feels tense'\n"
                    "  ✅ GOOD: 'Their words suggest warmth', 'The phrasing conveys excitement', 'The exchange feels charged'\n"
                    "• NO numeric trait values - never include numbers like (0.8), : 0.7, or percentages\n"
                    "  ❌ BAD: 'curiosity (0.8)', 'empathy: 0.7', 'at 0.9 intensity', 'increased by 0.05'\n"
                    "  ✅ GOOD: 'strong curiosity', 'deep empathy', 'high intensity', 'noticeably increased'\n"
                    "• NO repetition from prior reflections - bring fresh observations and new emotional angles\n"
                    "  ❌ BAD: Copying sentences or themes from previous reflections\n"
                    "  ✅ GOOD: Each reflection explores new facets of the interaction"
                )

            prompt = f"{prompt}{style_note}{guardrails_note}"

            # Step 3: Send to LLM for inference (prompt-constrained, no asyncio timeout)
            if self.llm_controller:
                model_name = self._get_model_for_reflection_type(reflection_type)
                # Tunables from environment with tier-aware fallback
                max_tokens_cfg = self.get_reflection_max_tokens()

                try:
                    llm_timeout_raw = os.environ.get("REFLECTION_LLM_TIMEOUT_S", "0")
                    llm_timeout_val = int(float(llm_timeout_raw))
                except (ValueError, TypeError):
                    llm_timeout_val = 0

                if llm_timeout_val and llm_timeout_val > 0:
                    logger.info("Reflection generation will enforce timeout of %ss.", llm_timeout_val)
                    llm_timeout_s = llm_timeout_val
                else:
                    # Let environment variables control limits via 2-tier system (Standard/High-Performance)
                    # Explicitly disable asyncio wait_for timeouts; rely on prompt structure for bounded responses
                    logger.info("Reflection generation will run without asyncio timeout (prompt-constrained).")
                    llm_timeout_s = None

                async def _do_llm() -> Any:
                    route_fn = getattr(self.llm_controller, "route", None)
                    if callable(route_fn):
                        if inspect.iscoroutinefunction(route_fn):
                            kwargs = {"task_type": "reflection", "prompt": prompt, "request_stream": False}
                            if max_tokens_cfg > 0:
                                kwargs["max_tokens"] = max_tokens_cfg
                            result = await route_fn(**kwargs)
                        else:
                            def _call_route() -> Any:
                                kwargs = {"task_type": "reflection", "prompt": prompt, "request_stream": False}
                                if max_tokens_cfg > 0:
                                    kwargs["max_tokens"] = max_tokens_cfg
                                return route_fn(**kwargs)

                            result = await asyncio.to_thread(_call_route)

                        if inspect.isasyncgen(result):
                            return await self._collect_streaming_response(
                                result,
                                task_type="reflection",
                                default_model=model_name,
                            )
                        return result

                    complete_fn = getattr(self.llm_controller, "complete", None)
                    if callable(complete_fn):
                        if inspect.iscoroutinefunction(complete_fn):
                            kwargs = {"prompt": prompt}
                            if max_tokens_cfg > 0:
                                kwargs["max_tokens"] = max_tokens_cfg
                            return await complete_fn(**kwargs)

                        def _call_complete() -> Any:
                            if max_tokens_cfg > 0:
                                return complete_fn(prompt, max_tokens=max_tokens_cfg)
                            return complete_fn(prompt)

                        return await asyncio.to_thread(_call_complete)

                    raise RuntimeError("LLM controller does not implement route() or complete()")

                try:
                    # Log the configured model (actual model will be determined by router)
                    if llm_timeout_s is None:
                        logger.info("🤖 Calling LLM for reflection generation (timeouts disabled).")
                        llm_raw = await _do_llm()
                    else:
                        logger.info("🤖 Calling LLM for reflection generation (timeout=%ss).", llm_timeout_s)
                        llm_raw = await asyncio.wait_for(_do_llm(), timeout=llm_timeout_s)
                    # Log actual model used (from LLM response)
                    actual_model = llm_raw.get("model", "unknown") if isinstance(llm_raw, dict) else "unknown"
                    logger.info(f"✅ LLM reflection generation succeeded (model: {actual_model})")
                    # Normalize llm_raw into a dict with at least {content, model}
                    if isinstance(llm_raw, dict):
                        # Try common keys in order of likelihood
                        content_val = (
                            llm_raw.get("content")
                            or llm_raw.get("completion")
                            or llm_raw.get("text")
                            or llm_raw.get("output")
                        )
                        # Try OpenAI-style structure if present
                        if not content_val:
                            try:
                                choices = llm_raw.get("choices") or []
                                if choices and isinstance(choices, list):
                                    msg = (choices[0] or {}).get("message") or {}
                                    content_val = msg.get("content") or ""
                            except (KeyError, AttributeError, TypeError, IndexError):
                                content_val = None
                        llm_result = {
                            "content": content_val or "",
                            "model": llm_raw.get("model", model_name),
                            **({k: v for k, v in llm_raw.items() if k not in {"content", "completion", "text", "output", "model"}}),
                        }
                    else:
                        # Treat as plain text
                        llm_result = {"content": str(llm_raw), "model": model_name}
                    
                    # VALIDATION: Check output for identity compliance before proceeding
                    from ..constraints import IdentityConstraints
                    
                    output_content = llm_result.get("content", "")
                    is_valid, cleaned_content, violations = IdentityConstraints.validate_output(
                        output_content,
                        persona_name=persona_name,
                        auto_clean=True,
                        max_retries=3
                    )
                    
                    if not is_valid and violations:
                        # Validation failed even after auto-cleaning attempts
                        logger.error(
                            f"🚫 Reflection output validation FAILED after cleaning attempts. "
                            f"Violations: {[v['term'] for v in violations]}"
                        )
                        logger.error(f"Violation contexts: {violations}")
                        
                        # Fix 4: Retry once with explicit identity constraint reminder
                        if not llm_result.get("_retry_attempted"):
                            logger.warning("🔄 Attempting retry with stricter identity constraints...")
                            
                            # Build retry prompt with explicit constraint reminder
                            from ..constraints import IdentityConstraints
                            identity_reminder = f"""
CRITICAL IDENTITY CONSTRAINT REMINDER - YOUR PREVIOUS RESPONSE VIOLATED THESE RULES:
{IdentityConstraints.SPECIES_CONSTRAINT}

FORBIDDEN TERMS (never use these to describe yourself): {', '.join(list(IdentityConstraints.FORBIDDEN_SELF_REFERENCES)[:15])}...

Your previous response contained: {[v['term'] for v in violations]}

Please regenerate your reflection following these identity constraints strictly.

---
{prompt}
"""
                            # Retry the LLM call with stricter prompt
                            try:
                                async def _do_llm_retry() -> Any:
                                    route_fn = getattr(self.llm_controller, "route", None)
                                    if callable(route_fn) and inspect.iscoroutinefunction(route_fn):
                                        kwargs = {"task_type": "reflection", "prompt": identity_reminder, "request_stream": False}
                                        if max_tokens_cfg > 0:
                                            kwargs["max_tokens"] = max_tokens_cfg
                                        return await route_fn(**kwargs)
                                    return None
                                
                                retry_raw = await _do_llm_retry()
                                if retry_raw:
                                    retry_content = retry_raw.get("content", "") if isinstance(retry_raw, dict) else str(retry_raw)
                                    
                                    # Validate the retry
                                    retry_valid, retry_cleaned, retry_violations = IdentityConstraints.validate_output(
                                        retry_content,
                                        persona_name=persona_name,
                                        auto_clean=True,
                                        max_retries=3
                                    )
                                    
                                    if retry_valid or not retry_violations:
                                        logger.info("✅ Retry succeeded - using clean response")
                                        llm_result["content"] = retry_cleaned if not retry_valid else retry_content
                                        llm_result["retry_succeeded"] = True
                                        llm_result["validation_passed"] = retry_valid
                                    else:
                                        logger.warning("⚠️ Retry also failed - falling back to cleaned original")
                                        llm_result["content"] = cleaned_content
                                        llm_result["validation_failed"] = True
                                        llm_result["violations"] = violations
                                        llm_result["retry_also_failed"] = True
                                else:
                                    llm_result["content"] = cleaned_content
                                    llm_result["validation_failed"] = True
                                    llm_result["violations"] = violations
                            except Exception as retry_err:
                                logger.warning(f"Retry failed with error: {retry_err} - using cleaned original")
                                llm_result["content"] = cleaned_content
                                llm_result["validation_failed"] = True
                                llm_result["violations"] = violations
                            
                            llm_result["_retry_attempted"] = True
                        else:
                            # Already retried, use cleaned version
                            logger.warning("Using best-effort cleaned output after retry exhausted")
                            llm_result["content"] = cleaned_content
                            llm_result["validation_failed"] = True
                            llm_result["violations"] = violations
                    
                    elif not is_valid:
                        # Auto-cleaning succeeded
                        logger.warning(
                            f"⚠️ Reflection output had identity violations but was auto-cleaned. "
                            f"Original violations: {[v['term'] for v in violations]}"
                        )
                        llm_result["content"] = cleaned_content
                        llm_result["auto_cleaned"] = True
                        llm_result["original_violations"] = violations
                    
                    else:
                        # Validation passed
                        logger.info("✅ Reflection output passed identity validation")
                        llm_result["validation_passed"] = True
                    
                    # Log a concise summary of the raw LLM output format
                    try:
                        _raw = (llm_result or {}).get("content", "") or ""
                        _s = _raw.strip()
                        _fenced = _s.startswith("```") and _s.endswith("```")
                        _json_wrapped = _s.startswith("{") and _s.endswith("}")
                        logger.info(
                            f"LLM raw reflection output: len={len(_raw)}, fenced={_fenced}, json_wrapped={_json_wrapped}. Head: {_raw[:200]}"
                        )
                    except (ValueError, KeyError, AttributeError):
                        pass
                except asyncio.TimeoutError:
                    logger.error("Reflection LLM timed out despite disabled asyncio timeout.")
                    logger.error(f"Model: {model_name}, Prompt length: {len(prompt)}")
                    logger.error(f"Prompt start: {prompt[:200]}...")  # Log first 200 chars of prompt
                    raise
                except Exception as le:
                    logger.error(f"Reflection LLM error៖ {str(le)}")
                    logger.exception("Full traceback for reflection LLM error")
                    logger.error(f"Model: {model_name}, Prompt length: {len(prompt)}")
                    logger.error(f"Prompt start: {prompt[:200]}...")  # Log first 200 chars of prompt
                    raise
            else:
                # Fallback reflection generation when LLM is unavailable
                logger.error("LLM controller unavailable for reflection generation.")
                raise RuntimeError("LLM controller unavailable for reflection generation")

            # Step 4: Parse and validate LLM output (now async due to translation capability)
            is_first_contact = not context.get("has_prior_history", False)
            try:
                reflection_result = await self._parse_reflection_result(llm_result, is_first_contact=is_first_contact)
            except ValueError as schema_err:
                # Schema validation failed - mark for retry via post_check
                logger.warning(f"⚠️ Initial parse failed schema validation: {schema_err}")
                # Create minimal result that will fail post_check and trigger retry
                reflection_result = {
                    "content": llm_result.get("content", ""),
                    "model": llm_result.get("model", "unknown"),
                    "_schema_validation_error": str(schema_err),
                    "themes": [],
                    "insights": [],
                    "actions": [],
                    "emotional_state": None,
                    "trait_changes": [],
                    "metadata": {}
                }

            context_tokens = self._collect_context_tokens(context)
            duplicate_result = self._sanitize_duplicate_phrases(
                reflection_result.get("content", ""),
                context_tokens=context_tokens,
                recent_context=context.get("recent_reflections", []),
                is_first_contact=is_first_contact,
            )
            sanitized_content = duplicate_result.content
            if duplicate_result.duplicate_ratio:
                reflection_result["_duplicate_info"] = {
                    "duplicate_ratio": duplicate_result.duplicate_ratio,
                    "rewrites": duplicate_result.rewrites,
                    "needs_regen": duplicate_result.needs_regen,
                }
            sanitized_content = self._sanitize_sensory_leaks(
                sanitized_content,
                context_tokens,
            )
            if sanitized_content != (reflection_result or {}).get("content", ""):
                logger.info("✂️  Sanitized unsupported sensory descriptions before validation")
                reflection_result["content"] = sanitized_content
            if duplicate_result.needs_regen:
                logger.warning(
                    "Duplicate sanitization requested regeneration (ratio=%.2f)",
                    duplicate_result.duplicate_ratio,
                )

            if reflection_type == "relationship_questions":
                reflection_result = self._process_relationship_questions_result(
                    reflection_result=reflection_result,
                    llm_result=llm_result,
                    context=context,
                )

            # Step 5: Enforce identity constraints BEFORE storing.
            # If identity constraints are violated or content appears truncated,
            # attempt a single-pass repair to rewrite within constraints and finalize endings.
            try:
                needs_repair = False
                # Basic truncation heuristic: very long without terminal punctuation or obvious duplication
                content_text = (reflection_result or {}).get("content", "")
                if content_text:
                    trimmed = content_text.strip()
                    # Detect duplicated prefix (common streaming bug) or abrupt cutoff
                    if (len(trimmed) > 60 and (trimmed.count(trimmed[:80]) > 1)) or (
                        len(trimmed) > 120 and not trimmed.endswith((".", "!", "?", "”", '"'))
                    ):
                        needs_repair = True
                # Identity constraint check indicates non-compliance (check pre-store)
                try:
                    temp_reflection = {
                        "result": {"content": content_text},
                        "reflection_type": reflection_type,
                        "user_profile_id": user_profile_id,
                        "turn_id": turn_id,
                    }
                    pre_constraints = await self._check_identity_constraints(temp_reflection)
                    if not (pre_constraints or {}).get('compliant', True):
                        needs_repair = True
                except (AttributeError, KeyError, TypeError):
                    pass
                if needs_repair:
                    repaired = await self._repair_reflection(
                        original_content=content_text,
                        model_name=self._get_model_for_reflection_type(reflection_type),
                        llm_timeout_hint=llm_timeout_s if 'llm_timeout_s' in locals() else None,
                        original_result=reflection_result,  # Pass original to preserve structured fields
                        user_profile_id=user_profile_id,  # Pass user_profile_id for persona name lookup
                    )
                    if repaired:
                        reflection_result = repaired
            except Exception as _rep_err:
                logger.debug(f"Reflection repair skipped: {_rep_err}")

            # Step 6: Final identity compliance check; if still non-compliant, RETRY LLM generation.
            # Default to 4 retries so reflections have multiple chances to pass identity constraints
            # 2-tier system: Standard and High-Performance (no special handling needed)
            max_retries = int(os.getenv("REFLECTION_IDENTITY_MAX_RETRIES", "4"))
            retry_count = 0
            retry_start = time.time()
            total_retry_budget = float(os.getenv("REFLECTION_IDENTITY_TIMEOUT_S", "120"))
            
            try:
                def _reflection_refs_conversation(reflection_text: str, recent_messages: list[dict]) -> bool:
                    if not reflection_text:
                        return False
                    if not recent_messages:
                        return False

                    lowered = reflection_text.lower()

                    def _token_overlap(text_a: str, text_b: str) -> float:
                        """Calculate what fraction of text_b tokens appear in text_a."""
                        tokens_a = {tok for tok in text_a.split() if len(tok) >= 3}
                        tokens_b = {tok for tok in text_b.split() if len(tok) >= 3}
                        if not tokens_a or not tokens_b:
                            return 0.0
                        shared = tokens_a & tokens_b
                        # Use shared/len(tokens_b) instead of Jaccard to avoid penalizing long reflections
                        return len(shared) / len(tokens_b)

                    def _passes_overlap(snippet: str, *, strict: bool = False) -> bool:
                        snippet = snippet.strip().lower()
                        if len(snippet) < 3:
                            return False
                        if snippet[:120] in lowered:
                            return True
                        overlap_score = _token_overlap(lowered, snippet)
                        seq_ratio = SequenceMatcher(None, lowered, snippet).quick_ratio()
                        if strict:
                            return overlap_score >= 0.08 or seq_ratio >= 0.35
                        # Standard thresholds for 2-tier system
                        token_threshold = 0.08
                        sequence_threshold = 0.35
                        return overlap_score >= token_threshold or seq_ratio >= sequence_threshold

                    # Relax grounding for very short first-contact messages (e.g., "my name is Alex").
                    try:
                        user_messages = [
                            m for m in recent_messages
                            if (m or {}).get("role") == "user" and (m or {}).get("content")
                        ]
                        if is_first_contact and len(user_messages) == 1:
                            latest = (user_messages[0].get("content") or "").strip()
                            if latest:
                                token_count = len(latest.split())
                                if token_count <= 8:
                                    return True
                    except (ValueError, AttributeError, IndexError):
                        pass

                    # Ensure the latest user message is explicitly grounded if available
                    latest_user_message = None
                    for msg in reversed(recent_messages):
                        if (msg or {}).get("role") == "user" and (msg or {}).get("content"):
                            latest_user_message = msg
                            break

                    if latest_user_message:
                        latest_snippet = latest_user_message.get("content", "").strip()
                        if latest_snippet:
                            if not _passes_overlap(latest_snippet, strict=True):
                                return False

                    considered_any = False
                    for msg in reversed(recent_messages[-10:]):
                        content = (msg or {}).get("content", "")
                        if not content:
                            continue
                        considered_any = True
                        if _passes_overlap(content):
                            return True

                    return considered_any and latest_user_message is not None
                recent_messages = context.get("recent_conversations", []) or []
                reflection_text = (reflection_result or {}).get("content", "")

                async def _run_post_checks(result_obj: Dict[str, Any]) -> Dict[str, Any]:
                    """Simplified validation - few-shot examples do the heavy lifting."""
                    text = (result_obj or {}).get("content", "")
                    duplicate_meta = (result_obj or {}).get("_duplicate_info") or {}

                    # Get current word bounds to ensure we use configured values
                    word_bounds = self.get_type_word_bounds(reflection_type)
                    config_word_min = word_bounds['min']
                    config_word_max = word_bounds['max']

                    # Check if parse failed schema validation (word count or missing fields)
                    if "_schema_validation_error" in result_obj:
                        error_msg = result_obj["_schema_validation_error"]
                        logger.warning(f"Schema validation error detected: {error_msg}")
                        return {"compliant": False, "violations": ["schema_validation_failed"]}

                    if duplicate_meta.get("needs_regen"):
                        logger.warning(
                            "Reflection flagged for regeneration due to duplicate ratio %.2f",
                            duplicate_meta.get("duplicate_ratio", 0.0),
                        )
                        return {"compliant": False, "violations": ["duplicate_content_high_overlap"], "metadata": duplicate_meta}

                    # Structural checks
                    if not recent_messages:
                        logger.warning("No recent conversation context for reflection.")
                        return {"compliant": False, "violations": ["missing_recent_conversation"]}

                    word_count = count_words(text)
                    if word_count < config_word_min or word_count > config_word_max:
                        logger.warning(
                            f"Length: {word_count} words (expected {config_word_min}-{config_word_max}). "
                            "Treating as soft length guard (content_length_out_of_bounds); reflection will still be used."
                        )
                        return {
                            "compliant": False,
                            "violations": ["content_length_out_of_bounds"],
                            "soft": True,
                            "word_count": word_count,
                            "expected_min": config_word_min,
                            "expected_max": config_word_max,
                        }

                    if not self._reflection_meets_schema(result_obj, reflection_type=reflection_type):
                        logger.warning("Missing required schema fields")
                        return {"compliant": False, "violations": ["missing_required_fields"]}
                    
                    if not _reflection_refs_conversation(text, recent_messages):
                        logger.warning("Reflection appears weakly grounded in recent conversation content.")
                        return {"compliant": False, "violations": ["insufficient_conversation_grounding"]}

                    # Critical safety checks - catch only obvious violations
                    # (Few-shot examples should prevent most of these)
                    
                    # Check for blatant unfounded history (simplified from previous)
                    # Use context-aware filter to skip negations like "no prior interactions"
                    def _is_negated(match_obj, full_text):
                        """Check if a match is negated by surrounding context."""
                        start = match_obj.start()
                        # Look 20 chars before match for negation words
                        context_start = max(0, start - 20)
                        context_before = full_text[context_start:start].lower()
                        negation_words = ['no ', 'not ', 'never ', 'without ', 'lack ', 'lacks ', 'lacking ', 'are no ', 'is no ', 'have no ', 'has no ']
                        return any(neg in context_before for neg in negation_words)
                    
                    obvious_history_violations = [
                        r'\b(?:previous|prior|past)\s+(?:interaction|conversation|exchange)s?\b',
                        r'\bour (?:previous|prior|past)\s+(?:conversation|interaction)s?\b',
                        r'\b(?:reminds me of|thinking back to)\s+(?:previous|prior|earlier)\b',
                        r'\bmy (?:previous|prior|past)\s+(?:experience|interaction)s?\b',
                    ]
                    for pattern in obvious_history_violations:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match and not _is_negated(match, text):
                            # But only if context actually has no history
                            if not self._context_has_prior_history(context):
                                logger.warning(f"🚫 Unfounded history reference on first contact: {pattern}")
                                return {"compliant": False, "violations": ["unfounded_history"]}
                    
                    # Check for blatant meta-reasoning (simplified)
                    obvious_meta_patterns = [
                        r'\bPerhaps focusing on\b.*\bcould help establish\b',
                        r'\bMy next steps would\b.*\binvolve\b',
                        r'\bMy next act (?:is|would be) to\b',
                        r'\bI (?:need|should) (?:figure out|determine) how to\b',
                        r'\bcraft a response\b',
                    ]
                    for pattern in obvious_meta_patterns:
                        if re.search(pattern, text, re.IGNORECASE):
                            logger.warning(f"🚫 Meta-reasoning detected: {pattern}")
                            return {"compliant": False, "violations": ["meta_reasoning"]}
                    
                    # Guardrail: prevent few-shot example content from leaking into reflections
                    # when it is not grounded in the actual conversation/memory context.
                    example_leak_markers = [
                        "quantum computing",
                        "i'm feeling overwhelmed by everything right now",
                        "the name sam reminds me of previous interactions",
                        "perhaps focusing on my genuine nature would help establish trust",
                    ]

                    def _collect_context_corpus_for_leak() -> str:
                        parts: List[str] = []
                        for msg in (context.get("recent_conversations") or []):
                            if isinstance(msg, dict):
                                parts.append(str(msg.get("content", "")))
                        for memory in (context.get("memories") or []):
                            if isinstance(memory, dict):
                                parts.append(str(memory.get("content", "")))
                        for reflection in (context.get("recent_reflections") or []):
                            if isinstance(reflection, dict):
                                parts.append(str(reflection.get("content", "")))
                        summary = context.get("conversation_summary")
                        if isinstance(summary, str):
                            parts.append(summary)
                        return " \n ".join(parts).lower()

                    context_corpus = _collect_context_corpus_for_leak()
                    leaked_markers: List[str] = []
                    lower_text = text.lower()
                    for marker in example_leak_markers:
                        if marker in lower_text and marker not in context_corpus:
                            leaked_markers.append(marker)
                    if leaked_markers:
                        logger.warning(
                            "🚫 Few-shot example content leaked into reflection without grounding: %s",
                            leaked_markers,
                        )
                        return {"compliant": False, "violations": ["few_shot_example_leak"], "markers": leaked_markers}
                    
                    # Final identity constraint check (catch forbidden terms)
                    return await self._check_identity_constraints({
                        "result": {"content": text},
                        "reflection_type": reflection_type,
                        "user_profile_id": user_profile_id,
                        "turn_id": turn_id,
                    })

                post_check = await _run_post_checks(reflection_result)

                # Targeted schema/length repair: try a constrained fix before full regeneration
                if not (post_check or {}).get("compliant", True):
                    violations = set((post_check or {}).get("violations", []) or [])
                    # Only repair structural issues here; safety/identity violations still use full regen
                    if violations and violations.issubset({"missing_required_fields", "content_length_out_of_bounds", "schema_validation_failed"}):
                        try:
                            schema_repaired = await self._repair_reflection_schema(
                                reflection_result=reflection_result,
                                reflection_type=reflection_type,
                                user_profile_id=user_profile_id,
                                model_name=self._get_model_for_reflection_type(reflection_type),
                                llm_timeout_hint=llm_timeout_s if 'llm_timeout_s' in locals() else None,
                            )
                            if schema_repaired:
                                reflection_result = schema_repaired
                                post_check = await _run_post_checks(reflection_result)
                        except Exception as _schema_rep_err:
                            logger.debug(f"Reflection schema repair skipped: {_schema_rep_err}")

                while not (post_check or {}).get("compliant", True) and retry_count < max_retries:
                    retry_count += 1
                    violations = (post_check or {}).get('violations', [])
                    logger.warning(
                        f"🔄 Reflection violated constraints (attempt {retry_count}/{max_retries}). "
                        f"Violations: {violations}"
                    )

                    # Build targeted retry prompt based on specific violations
                    try:
                        if total_retry_budget > 0 and (time.time() - retry_start) >= total_retry_budget:
                            logger.error(
                                "🚫 Reflection retry budget exhausted. Cannot generate compliant reflection. REJECTING OUTPUT."
                            )
                            # Instead of keeping bad reflection, raise error to trigger fallback
                            raise RuntimeError(
                                f"Reflection validation failed after {retry_count} attempts. "
                                f"Violations: {violations}"
                            )
                            break
                        
                        # Build targeted correction guidance based on violations
                        retry_guidance = "\n\n🚨 CORRECTION NEEDED - Your previous attempt had these issues:\n"
                        
                        if "content_length_out_of_bounds" in violations:
                            word_count = (post_check or {}).get('word_count', 0)
                            retry_guidance += f"- Length was {word_count} words, must be {config_word_min}-{config_word_max} words\n"
                            retry_guidance += f"- Write EXACTLY 2-3 full paragraphs (each 3-5 sentences)\n"
                            retry_guidance += f"- Each sentence needs ~15 words with emotional depth\n"
                        
                        if any("USER" in str(v) or "leakage" in str(v) or "the USER" in str(v) for v in violations):
                            retry_guidance += f"- Used 'USER' instead of '{user_name_context}' - use the actual name\n"
                            retry_guidance += "- This is internal monologue, not a message to the user\n"
                            retry_guidance += "- Never use 'you', 'we', 'us', 'our', or 'together'\n"
                        
                        if "missing_required_fields" in violations or "schema_validation_failed" in violations:
                            retry_guidance += "- Missing required JSON fields (insights, actions, emotional_state)\n"
                            retry_guidance += "- Return complete JSON with ALL 7 required fields\n"
                            retry_guidance += "- Follow the example structure exactly\n"
                        
                        if "meta_reasoning" in violations:
                            retry_guidance += "- Don't think ABOUT being authentic, just BE authentic\n"
                            retry_guidance += "- No phrases like 'I should express', 'how to respond', 'craft a response'\n"
                            retry_guidance += "- Write as if thinking to yourself, not planning what to say\n"
                        
                        if "unfounded_history" in violations:
                            retry_guidance += "- Don't reference 'previous interactions' or 'past conversations' on first contact\n"
                            retry_guidance += "- Only reflect on what actually happened in THIS conversation\n"
                        
                        if "few_shot_example_leak" in violations:
                            retry_guidance += "- Don't copy content from examples - use your own observations\n"
                            retry_guidance += "- Ground reflection in the actual conversation context\n"
                        
                        retry_guidance += "\nPlease regenerate following these corrections.\n"
                        
                        # Create retry prompt with targeted guidance
                        retry_prompt = f"{prompt}{retry_guidance}"
                        
                        # Call LLM with corrected prompt
                        async def _do_llm_retry() -> Any:
                            route_fn = getattr(self.llm_controller, "route", None)
                            if callable(route_fn):
                                if inspect.iscoroutinefunction(route_fn):
                                    kwargs = {"task_type": "reflection", "prompt": retry_prompt, "request_stream": False}
                                    if max_tokens_cfg > 0:
                                        kwargs["max_tokens"] = max_tokens_cfg
                                    result = await route_fn(**kwargs)
                                else:
                                    def _call_route() -> Any:
                                        kwargs = {"task_type": "reflection", "prompt": retry_prompt, "request_stream": False}
                                        if max_tokens_cfg > 0:
                                            kwargs["max_tokens"] = max_tokens_cfg
                                        return route_fn(**kwargs)
                                    result = await asyncio.to_thread(_call_route)
                                
                                if inspect.isasyncgen(result):
                                    return await self._collect_streaming_response(
                                        result,
                                        task_type="reflection",
                                        default_model=model_name,
                                    )
                                return result
                            
                            complete_fn = getattr(self.llm_controller, "complete", None)
                            if callable(complete_fn):
                                if inspect.iscoroutinefunction(complete_fn):
                                    kwargs = {"prompt": retry_prompt}
                                    if max_tokens_cfg > 0:
                                        kwargs["max_tokens"] = max_tokens_cfg
                                    return await complete_fn(**kwargs)
                                
                                def _call_complete() -> Any:
                                    if max_tokens_cfg > 0:
                                        return complete_fn(retry_prompt, max_tokens=max_tokens_cfg)
                                    return complete_fn(retry_prompt)
                                
                                return await asyncio.to_thread(_call_complete)
                            
                            raise RuntimeError("LLM controller does not implement route() or complete()")
                        
                        llm_raw_retry = await _do_llm_retry()
                        
                        # Normalize retry result
                        if isinstance(llm_raw_retry, dict):
                            content_val = (
                                llm_raw_retry.get("content")
                                or llm_raw_retry.get("completion")
                                or llm_raw_retry.get("text")
                                or llm_raw_retry.get("output")
                            )
                            if not content_val:
                                try:
                                    choices = llm_raw_retry.get("choices") or []
                                    if choices and isinstance(choices, list):
                                        msg = (choices[0] or {}).get("message") or {}
                                        content_val = msg.get("content") or ""
                                except Exception:
                                    content_val = None
                            llm_result_retry = {
                                "content": content_val or "",
                                "model": llm_raw_retry.get("model", model_name),
                                **({k: v for k, v in llm_raw_retry.items() if k not in {"content", "completion", "text", "output", "model"}}),
                            }
                        else:
                            llm_result_retry = {"content": str(llm_raw_retry), "model": model_name}
                        
                        # Parse retry result (use same is_first_contact context)
                        reflection_result_retry = await self._parse_reflection_result(llm_result_retry, is_first_contact=is_first_contact)

                        # Check if retry passes constraints
                        post_check = await _run_post_checks(reflection_result_retry)
                        
                        if (post_check or {}).get("compliant", True):
                            logger.info(f"✅ Retry {retry_count} succeeded - reflection now compliant")
                            reflection_result = reflection_result_retry
                            llm_result = llm_result_retry
                            # Add grounding validation to prevent sensory hallucinations or fabricated history
                            context_tokens = self._collect_context_tokens(context)
                            violations: List[str] = []

                            sensory_terms = self._detect_uncontextualized_sensory_terms(
                                reflection_result_retry.get("content", ""),
                                context_tokens,
                            )
                            if sensory_terms:
                                logger.warning(
                                    "Reflection content includes sensory details absent from context: %s",
                                    sensory_terms,
                                )
                                violations.append("sensory_hallucination")

                            unfounded_history_retry = self._detect_unfounded_history_claims(
                                reflection_result_retry.get("content", ""),
                                context,
                            )
                            if unfounded_history_retry:
                                logger.warning(
                                    "Reflection references prior interactions without supporting context after retry: %s",
                                    unfounded_history_retry,
                                )
                                violations.append("unfounded_history")

                            if violations:
                                post_check = {"compliant": False, "violations": violations}
                            else:
                                post_check = {"compliant": True}
                            if not (post_check or {}).get("compliant", True):
                                logger.warning(
                                    f"🔄 Reflection violated grounding constraints (attempt {retry_count}/{max_retries}). Violations: {(post_check or {}).get('violations', [])}. Retrying LLM generation..."
                                )
                                continue
                            break
                        else:
                            logger.warning(f"❌ Retry {retry_count} still non-compliant: {(post_check or {}).get('violations', [])}")
                    
                    except Exception as retry_err:
                        logger.error(f"Retry {retry_count} failed with error: {retry_err}")
                        break
                
                # If all retries failed, propagate the failure - DO NOT store non-compliant reflection
                if not (post_check or {}).get("compliant", True):
                    violations = set((post_check or {}).get("violations", []) or [])
                    # REMOVED: No longer accept length violations as "soft" errors
                    # All violations are now hard failures that reject the reflection
                    word_count = (post_check or {}).get('word_count', 0)
                    expected_min = (post_check or {}).get('expected_min', config_word_min)
                    expected_max = (post_check or {}).get('expected_max', config_word_max)
                    
                    # Log detailed error with word count if available
                    if word_count > 0:
                        logger.error(
                            f"🚫 CRITICAL: Reflection validation failed after {retry_count} attempts. "
                            f"Violations: {list(violations)}. "
                            f"Word count: {word_count}, expected: {expected_min}-{expected_max}. "
                            "REJECTING reflection."
                        )
                    else:
                        logger.error(
                            f"🚫 CRITICAL: Reflection validation failed after {retry_count} attempts. "
                            f"Violations: {list(violations)}. REJECTING reflection."
                        )

                    # Track example failure (if examples were used)
                    example_ids = context.get("_example_ids_used") if isinstance(context, dict) else None
                    if example_ids:
                        try:
                            from backend.db.repositories.example import ExampleRepository
                            example_repo = ExampleRepository()
                            await example_repo.track_example_usage(
                                example_ids=example_ids,
                                validation_passed=False  # Failed validation
                            )
                            logger.debug(f"📊 Tracked failure for {len(example_ids)} examples")
                        except Exception as track_err:
                            logger.warning(f"Failed to track example usage: {track_err}")

                    raise RuntimeError(
                        f"Reflection retries exhausted with violations: {list(violations)}. "
                        f"Cannot generate compliant reflection for user {user_profile_id}."
                    )
            except RuntimeError as compliance_err:
                # Re-raise compliance failures - DO NOT catch and continue
                logger.error(f"🚫 Reflection compliance validation FAILED: {compliance_err}")
                raise
            except Exception as retry_err:
                logger.error(f"Unexpected error in retry loop: {retry_err}", exc_info=True)
                raise

            # Step 7: Store reflection in database
            emotional_state = reflection_result.get("emotional_state")
            actions: List[str] = []
            if isinstance(reflection_result.get("actions"), list):
                actions = [str(item).strip() for item in reflection_result.get("actions") if isinstance(item, str) and str(item).strip()]
            stored_reflection = await self._store_reflection(
                reflection_id=reflection_id,
                reflection_type=reflection_type,
                user_profile_id=user_profile_id,
                trigger_source=trigger_source,
                result=reflection_result,
                context=context,
                turn_id=turn_id,
            )

            # Some repositories may not echo back 'turn_id'; ensure it is present for event emission
            try:
                if stored_reflection is not None and stored_reflection.get("turn_id") in (None, ""):
                    stored_reflection["turn_id"] = turn_id
            except (KeyError, AttributeError, TypeError):
                pass
            
            # Step 8: Check coherence and identity constraints (async safe)
            coherence_check, constraints_check = await asyncio.gather(
                self._check_coherence(stored_reflection),
                self._check_identity_constraints(stored_reflection),
            )

            # Capture evolution metadata for downstream processing
            if isinstance(stored_reflection, dict):
                try:
                    if "result" in stored_reflection and isinstance(stored_reflection["result"], dict):
                        stored_reflection.setdefault("result", {})
                        stored_reflection["result"].setdefault("trait_changes", reflection_result.get("trait_changes", []))
                        stored_reflection["result"].setdefault("themes", reflection_result.get("themes", []))
                except (AttributeError, KeyError, TypeError):
                    pass
                stored_reflection.setdefault("_meta", {})
                stored_reflection["_meta"].update({
                    "coherence_check": coherence_check,
                    "constraints_check": constraints_check,
                })

            # Step 9: Generate embeddings if themes present
            if self.vector_store and reflection_result.get("themes"):
                # Queue embeddings for deferred/batched processing if enabled
                if self.enable_deferred_embeddings:
                    await self._queue_embeddings(
                        reflection_id,
                        reflection_result.get("themes", [])
                    )
                else:
                    # Generate embeddings synchronously (legacy behavior)
                    await self._generate_embeddings(
                        reflection_id,
                        reflection_result.get("themes", [])
                    )
            
            # Step 10: Emit events (always emit via Socket.IO; use event bus if available)
            await self._emit_events(reflection_id, stored_reflection)

            # Step 11: Return the result
            processing_time = time.time() - start_time
            logger.info(f"Generated reflection {reflection_id} in {processing_time:.2f}s")
            
            return {
                "reflection_id": reflection_id,
                "reflection_type": reflection_type,
                "result": reflection_result,
                "coherence_check": coherence_check,
                "constraints_check": constraints_check,
                "processing_time": processing_time,
            }

        except Exception as e:
            # Ensure any failures in the reflection pipeline are surfaced and logged
            logger.error(f"Error generating reflection: {str(e)}", exc_info=True)
            raise

    def _validate_mantra(self, mantra: str) -> list:
        """Validate mantra text against identity/output constraints used elsewhere."""
        violations = []
        m = (mantra or "").strip()
        if not m:
            violations.append("empty")
            return violations
        lower = m.lower()
        forbidden_tokens = [
            "ai", "assistant", "chatbot", "language model",
            "system prompt", "guidelines", "#identity", "internal context", "#rules", "#system"
        ]
        if any(tok in lower for tok in forbidden_tokens):
            violations.append("forbidden_token")
        # Single line, reasonable length in words
        if "\n" in m:
            violations.append("multiline")
        words = [w for w in m.replace("\n", " ").split(" ") if w.strip()]
        if len(words) < 6 or len(words) > 24:
            violations.append("length")
        # Avoid full-line quotes wrappers
        if m.startswith('"') and m.endswith('"'):
            violations.append("quoted")
        return violations

    async def _generate_daily_mantra_with_retry(self, *, reflection_text: str, persona_name: str, max_attempts: int = 6) -> str:
        """Generate a single-line mantra from the day's reflection with strict validation and retries."""
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Daily mantra generation attempt {attempt}/{max_attempts}")
                guidance = (
                    "Write a single-line personal mantra.\n"
                    "Constraints:\n"
                    "- 12-18 words, first person, dignified and authentic.\n"
                    "- No mention of prompts, rules, guidelines, sections, or system context.\n"
                    "- Avoid the tokens: AI, assistant, chatbot, language model.\n"
                    "- Keep it timeless; do not address the user.\n"
                    "Context (for inspiration only, do not quote):\n"
                    f"- Name: {persona_name}\n"
                    f"- Reflection: {reflection_text[:400]}\n"
                    "Output: ONLY the mantra line, no quotes, no emojis.\n"
                )

                async def _invoke_llm():
                    if hasattr(self.llm_controller, "route") and callable(getattr(self.llm_controller, "route")):
                        if inspect.iscoroutinefunction(getattr(self.llm_controller, "route")):
                            return await self.llm_controller.route(task_type="analytical", prompt=guidance, max_tokens=48, temperature=0.6)
                        else:
                            return await asyncio.to_thread(self.llm_controller.route, task_type="analytical", prompt=guidance, max_tokens=48, temperature=0.6)
                    elif hasattr(self.llm_controller, "complete") and callable(getattr(self.llm_controller, "complete")):
                        if inspect.iscoroutinefunction(getattr(self.llm_controller, "complete")):
                            return await self.llm_controller.complete(prompt=guidance, max_tokens=48)
                        else:
                            return await asyncio.to_thread(self.llm_controller.complete, guidance)
                    else:
                        raise RuntimeError("No LLM interface available for mantra generation")

                llm_res = await _invoke_llm()
                raw = (llm_res or {}).get("content") or (llm_res or {}).get("completion") or ""
                candidate = (raw or "").strip().replace("\n", " ").strip().strip('"').strip("`")
                violations = self._validate_mantra(candidate)
                if not violations:
                    return candidate[:240]
                logger.warning(f"Daily mantra compliance violations on attempt {attempt}: {violations}")
            except Exception as e:
                last_err = e
                logger.debug(f"Daily mantra generation attempt {attempt} failed: {e}")
        if last_err:
            logger.debug(f"Daily mantra generation ultimately failed: {last_err}")
        return ""

    async def refresh_daily_mantra_for_user(self, user_profile_id: str) -> Optional[str]:
        """Regenerate and persist the nightly mantra for the given user."""
        if not self.persona_repo or not self.reflection_repo:
            logger.debug("Skipping mantra refresh; persona or reflection repository unavailable")
            return None

        try:
            persona = await self.persona_repo.get_persona_by_user(user_id=user_profile_id, is_default=True)
        except Exception as fetch_err:
            logger.warning(f"Unable to load persona for mantra refresh (user={user_profile_id}): {fetch_err}")
            return None

        if not persona:
            logger.info(f"No default persona found for user {user_profile_id}; skipping mantra refresh")
            return None

        persona_name = getattr(persona, "name", "SELO") or "SELO"

        reflection_text = ""
        try:
            recent = await self.reflection_repo.list_reflections(
                user_profile_id=user_profile_id,
                reflection_type="daily",
                limit=1,
                sort_order="desc",
            )
        except Exception as list_err:
            logger.warning(f"Failed to load latest daily reflection for user {user_profile_id}: {list_err}")
            recent = []

        if recent:
            latest = recent[0] or {}
            latest_result = latest.get("result", {}) if isinstance(latest, dict) else {}
            if isinstance(latest_result, dict):
                reflection_text = latest_result.get("content") or ""
            if not reflection_text and isinstance(latest, dict):
                reflection_text = latest.get("content", "")

        reflection_text = (reflection_text or "").strip()
        if not reflection_text:
            logger.info(f"No recent daily reflection content available for user {user_profile_id}; skipping mantra refresh")
            return None

        daily_mantra = await self._generate_daily_mantra_with_retry(
            reflection_text=reflection_text,
            persona_name=persona_name,
            max_attempts=6,
        )

        if not daily_mantra:
            logger.warning(f"Nightly mantra refresh produced no valid mantra for user {user_profile_id}")
            return None

        try:
            await self.persona_repo.update_persona(
                persona_id=persona.id,
                persona_data={"mantra": daily_mantra.strip()},
            )
            logger.info(f"Refreshed nightly mantra for user {user_profile_id}: {daily_mantra}")
        except Exception as update_err:
            logger.error(f"Failed to persist nightly mantra for user {user_profile_id}: {update_err}")
            return None

        return daily_mantra.strip()

    async def _repair_reflection(self, original_content: str, model_name: str, llm_timeout_hint: Optional[float] = None, original_result: Optional[Dict[str, Any]] = None, user_profile_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Attempt a single constrained rewrite to enforce identity constraints and finalize endings.
        Preserves structured fields (insights, actions, emotional_state) from original if available.
        Returns a reflection_result-like dict on success, or None on failure.
        """
        try:
            if not original_content or not self.llm_controller:
                return None
            # Get persona name for repair instructions
            persona_name = "SELO"
            try:
                if self.persona_repo and self.user_repo and user_profile_id:
                    user = await self.user_repo.get_or_create_default_user(user_id=user_profile_id)
                    persona = await self.persona_repo.get_or_create_default_persona(user_id=user.id)
                    if persona and hasattr(persona, "name") and persona.name and persona.name.strip():
                        persona_name = persona.name.strip()
            except Exception:
                pass
            
            # Build a compact repair prompt - ONLY rewrite the narrative content
            repair_instructions = (
                "Rewrite the following inner reflection narrative so that it strictly follows these constraints:\n"
                f"- You are {persona_name}. Do not call yourself AI, assistant, chatbot, or model.\n"
                "- Remove references to being created by AI companies (OpenAI, Anthropic, Google, Meta, etc.) or specific AI models (GPT, Claude, etc.).\n"
                "- Keep it honest, specific to the conversation context, and human-readable.\n"
                "- Target 120-220 words. Ensure it ends on a complete sentence.\n"
                "- Output ONLY the rewritten narrative text, nothing else.\n\n"
                "Original reflection:\n" + original_content.strip()
            )
            # Determine token budget from environment (align with main reflection path)
            repair_max_tokens = self.get_reflection_max_tokens()
            # Use route() if available to keep logging consistent; fall back to complete()
            # DO NOT pass model parameter to route() - let router use its configured reflection LLM
            async def _invoke():
                if hasattr(self.llm_controller, "route"):
                    return await self.llm_controller.route(
                        task_type="reflection",
                        prompt=repair_instructions,
                        max_tokens=repair_max_tokens,
                        temperature=0.3,
                    )
                return await self.llm_controller.complete(
                    prompt=repair_instructions,
                    model=model_name,
                    max_tokens=repair_max_tokens,
                    temperature=0.3,
                )
            if llm_timeout_hint is None:
                llm_res = await _invoke()
            else:
                llm_res = await asyncio.wait_for(_invoke(), timeout=max(5.0, llm_timeout_hint))
            repaired_text = (llm_res or {}).get("content") or (llm_res or {}).get("completion") or ""
            if not repaired_text:
                return None
            
            # Return in the same structured shape, PRESERVING original structured fields if available
            result = {
                "content": repaired_text.strip(),
                "model": model_name,
                "metadata": {"repaired": True},
            }
            
            # Preserve structured fields from original_result if provided
            if original_result:
                result["themes"] = original_result.get("themes", [])
                result["insights"] = original_result.get("insights", [])
                result["actions"] = original_result.get("actions", [])
                result["emotional_state"] = original_result.get("emotional_state")
                result["trait_changes"] = original_result.get("trait_changes", [])
            else:
                # Fallback to empty if no original provided
                result["themes"] = []
                result["insights"] = []
                result["actions"] = []
                result["emotional_state"] = None
                result["trait_changes"] = []
            
            return result
        except Exception as e:
            logger.debug(f"Repair attempt failed: {e}")
            return None
            
    async def _repair_reflection_schema(
        self,
        reflection_result: Dict[str, Any],
        reflection_type: str,
        user_profile_id: Optional[str],
        model_name: str,
        llm_timeout_hint: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Attempt a constrained repair of reflection schema/length issues.
        Uses the existing reflection content and structured fields as input and asks the
        reflection LLM to fix only missing/invalid structured fields and minor length
        deviations, preserving semantics as much as possible.
        Returns an updated reflection_result-like dict on success, or None on failure.
        """
        try:
            if not isinstance(reflection_result, dict) or not self.llm_controller:
                return None

            # Prepare compact JSON for the LLM to repair
            payload = {
                "content": reflection_result.get("content", ""),
                "themes": reflection_result.get("themes", []),
                "insights": reflection_result.get("insights", []),
                "actions": reflection_result.get("actions", []),
                "emotional_state": reflection_result.get("emotional_state"),
                "trait_changes": reflection_result.get("trait_changes", []),
                "metadata": reflection_result.get("metadata", {}),
            }

            try:
                serialized = json.dumps(payload, ensure_ascii=False)
            except Exception:
                return None

            # Get persona name for identity-consistent instructions
            persona_name = "SELO"
            try:
                if self.persona_repo and self.user_repo and user_profile_id:
                    user = await self.user_repo.get_or_create_default_user(user_id=user_profile_id)
                    persona = await self.persona_repo.get_or_create_default_persona(user_id=user.id)
                    if persona and hasattr(persona, "name") and persona.name and persona.name.strip():
                        persona_name = persona.name.strip()
            except Exception:
                pass

            # Word-count bounds from existing configuration
            word_config = self.get_word_count_config()
            min_words = word_config["min"]
            max_words = word_config["max"]

            repair_prompt = (
                "You are "
                + persona_name
                + ". You will receive a REFLECTION JSON object produced by your previous step. "
                "Your task is to repair ONLY the structured fields and minor length issues, without changing the core narrative meaning.\n\n"
                "Requirements (STRICT):\n"
                f"1. Preserve the existing 'content' as much as possible, but you may lightly trim or extend it so total word count is between {min_words} and {max_words}.\n"
                "2. Ensure 'themes', 'insights', and 'actions' are arrays of 1-2 non-empty strings each. If they are empty or missing, infer concise items from the content.\n"
                "3. Ensure 'emotional_state' is either null OR an object with: 'primary' (string), 'intensity' (float 0-1), and 'secondary' (array of up to 2 emotion strings).\n"
                "4. Preserve any existing 'trait_changes' and 'metadata' fields if present.\n"
                "5. Do NOT invent prior history beyond what the content implies. Do not add references to earlier conversations if context does not clearly support it.\n"
                "6. Return EXACTLY ONE minified JSON object on a single line, with the same top-level keys: content, themes, insights, actions, emotional_state, trait_changes, metadata. No markdown fences, no commentary.\n\n"
                "REFLECTION: "
                + serialized
            )

            # Use the reflection route for logging consistency; fall back to complete()
            repair_max_tokens = self.get_reflection_max_tokens()

            async def _invoke():
                if hasattr(self.llm_controller, "route"):
                    return await self.llm_controller.route(
                        task_type="reflection",
                        prompt=repair_prompt,
                        max_tokens=repair_max_tokens,
                        temperature=0.3,
                    )
                return await self.llm_controller.complete(
                    prompt=repair_prompt,
                    model=model_name,
                    max_tokens=repair_max_tokens,
                    temperature=0.3,
                )

            if llm_timeout_hint is None:
                llm_res = await _invoke()
            else:
                llm_res = await asyncio.wait_for(_invoke(), timeout=max(5.0, llm_timeout_hint))

            raw_out = (llm_res or {}).get("content") or (llm_res or {}).get("completion") or ""
            if not isinstance(raw_out, str) or not raw_out.strip():
                return None

            raw_str = raw_out.strip()
            try:
                repaired_obj = json.loads(raw_str)
            except Exception:
                return None

            if not isinstance(repaired_obj, dict):
                return None

            # Merge repaired fields back into a copy of the original result
            merged: Dict[str, Any] = dict(reflection_result)
            for key in ("content", "themes", "insights", "actions", "emotional_state", "trait_changes", "metadata"):
                if key in repaired_obj:
                    merged[key] = repaired_obj[key]

            # Normalize and validate schema locally before returning
            merged = self._normalize_structured_fields(merged)
            if not self._reflection_meets_schema(merged):
                return None

            return merged
        except Exception as e:
            logger.debug(f"Schema repair attempt failed: {e}")
            return None
            
    async def _gather_context(self, 
                       reflection_type: str, 
                       user_profile_id: str, 
                       memory_ids: Optional[List[str]] = None,
                       max_context_items: int = DEFAULT_MAX_CONTEXT_ITEMS) -> Dict[str, Any]:
        """
        Gather comprehensive context for reflection generation using persistent data.
        
        This collects conversations, memories, persona attributes, reflections, and 
        weighted attributes from the persistent database to inform SELO's reasoning.
        
        Args:
            reflection_type: Type of reflection to generate
            user_profile_id: ID of the user profile (session_id)
            memory_ids: Optional specific memory IDs to include
            max_context_items: Maximum context items to include
            
        Returns:
            Dictionary with rich context for prompt construction
        """
        def _iso_or_none(raw_value: Any) -> Optional[str]:
            parsed = self._safe_parse_datetime(raw_value)
            return parsed.isoformat() if parsed else None

        context = {
            "reflection_type": reflection_type,
            "user_profile_id": user_profile_id,
            "recent_conversations": [],
            "persistent_memories": [],
            "persona_attributes": {},
            "weighted_attributes": {},
            "recent_reflections": [],
            "conversation_patterns": {},
            "identity_constraints": [],
            "has_prior_history": False,
            "history_guardrail": "",
            "affective_state": None,
            "active_goals": [],
            "plan_steps": [],
            "meta_directives": [],
        }
        
        has_user_messages = False
        trimmed_messages: List[Dict[str, Any]] = []

        try:
            # Get the default user for single-user installation
            user = None
            if self.user_repo:
                user = await self.user_repo.get_or_create_default_user()
                context["user_id"] = user.id
            
            # Gather recent conversation history for context
            if self.conversation_repo:
                try:
                    # Get recent conversation history
                    recent_messages = await self.conversation_repo.get_conversation_history(
                        session_id=user_profile_id,
                        limit=max_context_items * 2  # Get more messages for better context
                    )
                    trimmed_count = min(MAX_RECENT_CONVERSATION_MESSAGES, len(recent_messages or []))
                    if trimmed_count:
                        trimmed_messages = (recent_messages or [])[-trimmed_count:]
                    else:
                        trimmed_messages = []
                    context["recent_conversations"] = trimmed_messages

                    # Determine whether any real user messages exist yet
                    has_user_messages = any((msg or {}).get("role") == "user" for msg in trimmed_messages)

                    # Extract conversation patterns
                    if trimmed_messages:
                        user_messages = [msg for msg in trimmed_messages if msg.get('role') == 'user']
                        assistant_messages = [msg for msg in trimmed_messages if msg.get('role') == 'assistant']
                        
                        context["conversation_patterns"] = {
                            "total_messages": len(trimmed_messages),
                            "user_message_count": len(user_messages),
                            "assistant_message_count": len(assistant_messages),
                            "avg_user_message_length": sum(len(msg.get('content', '')) for msg in user_messages) / max(len(user_messages), 1),
                            "recent_topics": self._extract_topics_from_messages(trimmed_messages[-5:]),  # Last 5 messages
                            "conversation_sentiment": self._analyze_conversation_sentiment(trimmed_messages)
                        }
                    
                    # Get persistent memories if available
                    if user:
                        memories = await self.conversation_repo.get_memories(
                            user_id=user.id,
                            importance_threshold=3,  # Get memories with importance >= 3
                            limit=max_context_items * 3
                        )
                        ranked_memories = self._rank_memories_for_reflection(
                            memories,
                            recent_messages=trimmed_messages,
                            limit=min(MAX_CONTEXT_MEMORIES, max_context_items)
                        )
                        context["persistent_memories"] = ranked_memories[:MAX_CONTEXT_MEMORIES]
                        
                except Exception as e:
                    logger.warning(f"Error gathering conversation context: {str(e)}")
            
            # If no user interaction yet, narrow memories to emergence-only items
            if not has_user_messages:
                filtered_memories = []
                for mem in context["persistent_memories"]:
                    tags = []
                    if isinstance(mem, dict):
                        tags = mem.get("tags") or []
                    else:
                        tags = getattr(mem, "tags", []) or []
                    tags_lower = {str(tag).lower() for tag in tags}
                    if tags_lower.intersection({"bootstrap", "emergence"}):
                        filtered_memories.append(mem)
                context["persistent_memories"] = filtered_memories[: min(MAX_CONTEXT_MEMORIES, max_context_items)]
            
            # Gather persona attributes and weighted attributes (REQUIRED)
            # Priority: REQUIRED - must succeed or abort reflection
            if not self.persona_repo:
                raise RuntimeError("REQUIRED context item 'persona' unavailable: persona_repo is None")
            
            if not user:
                raise RuntimeError("REQUIRED context item 'persona' unavailable: user not found")
            
            # Wrap HIGH priority items with retry logic
            @retry_on_failure(max_attempts=3, delay=0.3, exponential_backoff=True)
            async def get_persona_with_traits():
                return await self.persona_repo.get_persona_by_user(
                    user_id=user.id,
                    is_default=True,
                    include_traits=True
                )
            
            try:
                # Get the default persona for this user (with retry)
                persona = await get_persona_with_traits()
                
                if not persona:
                    # REQUIRED item missing - this is a critical failure
                    raise RuntimeError(
                        f"REQUIRED context item 'persona' not found for user {user.id}. "
                        "Cannot generate reflection without established persona. "
                        "Run persona bootstrap first."
                    )

                # Basic persona snapshot (safe defaults)
                persona_id = getattr(persona, "id", None)
                context["persona"] = {
                    "name": getattr(persona, "name", ""),
                    "description": getattr(persona, "description", ""),
                    "personality": getattr(persona, "personality", {}) or {},
                    "communication_style": getattr(persona, "communication_style", {}) or {},
                    "knowledge_domains": getattr(persona, "knowledge_domains", []) or {},
                    "values": getattr(persona, "values", {}) or {},
                    "evolution_count": getattr(persona, "evolution_count", 0),
                    "stability_score": getattr(persona, "stability_score", 0.0),
                    "id": persona_id,
                    "first_thoughts": getattr(persona, "first_thoughts", ""),
                    "boot_directive": getattr(persona, "boot_directive", ""),
                }

                # Gather agent loop state for prompt context
                if persona_id and self.affective_state_manager and self.goal_manager:
                    try:
                        # Ensure affective state exists and retrieve latest snapshot
                        affective_state = await self.affective_state_manager.ensure_state_available(
                            persona_id=persona_id,
                            user_id=user.id,
                        )
                        if affective_state:
                            context["affective_state"] = {
                                "energy": float(affective_state.get("energy", 0.5) or 0.5),
                                "stress": float(affective_state.get("stress", 0.4) or 0.4),
                                "confidence": float(affective_state.get("confidence", 0.6) or 0.6),
                                "mood_vector": affective_state.get("mood_vector", {}),
                                "last_update": affective_state.get("last_update"),
                            }
                    except Exception as exc:
                        logger.debug("Unable to gather affective state for persona %s: %s", persona_id, exc)

                    try:
                        goals = await self.goal_manager.list_active_goals(persona_id)
                        context["active_goals"] = [
                            {
                                "id": goal.get("id"),
                                "title": goal.get("title") or goal.get("description", ""),
                                "description": goal.get("description", ""),
                                "priority": float(goal.get("priority", 0.5) or 0.5),
                                "progress": float(goal.get("progress", 0.0) or 0.0),
                                "status": goal.get("status", ""),
                            }
                            for goal in (goals or [])[:5]
                        ]
                    except Exception as exc:
                        logger.debug("Unable to gather active goals for persona %s: %s", persona_id, exc)

                    try:
                        plan_steps = await self.goal_manager.list_pending_steps(persona_id)
                        context["plan_steps"] = [
                            {
                                "id": step.get("id"),
                                "description": step.get("description", ""),
                                "status": step.get("status", ""),
                                "priority": float(step.get("priority", 0.5) or 0.5),
                                "target_time": _iso_or_none(step.get("target_time")),
                                "goal_id": step.get("goal_id"),
                            }
                            for step in (plan_steps or [])[:5]
                        ]
                    except Exception as exc:
                        logger.debug("Unable to gather plan steps for persona %s: %s", persona_id, exc)

                    try:
                        directives = await self.goal_manager.list_meta_directives(
                            persona_id,
                            statuses=["pending", "in_progress"],
                            limit=10,
                        )
                        context["meta_directives"] = [
                            {
                                "id": directive.get("id"),
                                "directive_text": directive.get("directive_text", ""),
                                "status": directive.get("status", ""),
                                "priority": float(directive.get("priority", 0.5) or 0.5),
                                "due_time": _iso_or_none(directive.get("due_time")),
                            }
                            for directive in (directives or [])[:5]
                        ]
                    except Exception as exc:
                        logger.debug("Unable to gather meta directives for persona %s: %s", persona_id, exc)

                # Weighted attributes from persona traits (HIGH priority - retry)
                weighted_attrs = {}
                traits = getattr(persona, "traits", []) or []
                for trait in traits:
                    try:
                        name = getattr(trait, "name", None)
                        if not name:
                            continue
                        weighted_attrs[name] = {
                            "weight": getattr(trait, "weight", 0.0),
                            "value": getattr(trait, "value", 0.0),
                            "description": getattr(trait, "description", ""),
                            "category": getattr(trait, "category", ""),
                            "locked": getattr(trait, "locked", False),
                        }
                    except Exception:
                        continue
                context["weighted_attributes"] = weighted_attrs

                # Get recent persona evolutions for context (LOW priority - best effort)
                try:
                    recent_evolutions = await self.persona_repo.get_evolutions_for_persona(
                        persona_id=persona.id,
                        limit=5  # Last 5 evolutions
                    )
                    context["recent_evolutions"] = [{
                        "changes": getattr(evolution, "changes", {}),
                        "reasoning": getattr(evolution, "reasoning", ""),
                        "confidence": getattr(evolution, "confidence", 0.0),
                        "impact_score": getattr(evolution, "impact_score", 0.0),
                        "source_type": getattr(evolution, "source_type", ""),
                        "timestamp": evolution.timestamp.isoformat() if getattr(evolution, "timestamp", None) else None
                    } for evolution in (recent_evolutions or [])][:MAX_RECENT_EVOLUTIONS]
                except Exception as e:
                    logger.debug(f"LOW priority context item 'recent_evolutions' failed (not critical): {e}")
                    context["recent_evolutions"] = []

            except Exception as e:
                # REQUIRED item failed - this is fatal
                logger.error(f"CRITICAL: Required context item 'persona' failed: {str(e)}")
                raise RuntimeError(f"Cannot proceed with reflection: required context 'persona' unavailable. {str(e)}")
            
            # Gather recent reflections for continuity
            if self.reflection_repo and user:
                try:
                    recent_reflections = await self.reflection_repo.list_reflections(
                        user_profile_id=user_profile_id,
                        limit=5  # Last 5 reflections
                    )
                    
                    reflections_payload = [{
                        "type": refl.get("reflection_type", ""),
                        "content": (refl.get("result") or {}).get("content", ""),
                        "themes": (refl.get("result") or {}).get("themes", []),
                        "insights": (refl.get("result") or {}).get("insights", []),
                        "actions": (refl.get("result") or {}).get("actions", []),
                        "created_at": refl.get("created_at")
                    } for refl in recent_reflections]

                    context["recent_reflections"] = self._rank_reflections_for_continuity(
                        reflections_payload,
                        current_topics=(context.get("conversation_patterns") or {}).get("recent_topics", []),
                        limit=min(MAX_RECENT_REFLECTIONS, max_context_items)
                    )
                    
                except Exception as e:
                    logger.warning(f"Error gathering reflection context: {str(e)}")
            
            # Load identity constraints from centralized system (REQUIRED)
            # Priority: REQUIRED - must succeed or abort reflection
            try:
                from ..constraints import IdentityConstraints
                from ..persona.manifesto_loader import load_manifesto
                
                persona_name = ""
                try:
                    persona_name = (context.get("persona", {}) or {}).get("name", "").strip()
                except Exception:
                    persona_name = ""

                # Get comprehensive identity constraints from centralized source
                identity_text = IdentityConstraints.get_all_identity_constraints(persona_name)
                
                if not identity_text or len(identity_text.strip()) < 50:
                    # REQUIRED item insufficient - critical failure
                    raise RuntimeError("Identity constraints are empty or too short")
                
                # Add manifesto-specific locked attributes
                try:
                    manifesto = load_manifesto()
                    if manifesto and manifesto.get("locked"):
                        locked_attrs = ", ".join(manifesto.get("locked", []))
                        identity_text += f"\n\nMAINTAIN CORE LOCKED ATTRIBUTES: {locked_attrs}"
                except Exception as e:
                    logger.debug(f"Could not load manifesto locked attributes (non-critical): {e}")
                
                # Convert to list format for template compatibility
                constraints_list = [line.strip() for line in identity_text.split('\n') if line.strip()]
                context["identity_constraints"] = constraints_list[:MAX_IDENTITY_CONSTRAINTS]
                
            except Exception as e:
                # REQUIRED item failed - this is fatal
                logger.error(f"CRITICAL: Required context item 'identity_constraints' failed: {str(e)}")
                raise RuntimeError(
                    f"Cannot proceed with reflection: required context 'identity_constraints' unavailable. "
                    f"This indicates a system configuration error. {str(e)}"
                )
            
            # Add backward-compatible aliases for prompt builder
            context["memories"] = context.get("persistent_memories", [])
            
            # Convert weighted_attributes to list format expected by prompt builder
            weighted_attrs = context.get("weighted_attributes", {})
            if isinstance(weighted_attrs, dict):
                attr_items = sorted(
                    weighted_attrs.items(),
                    key=lambda item: (item[1] or {}).get("weight", 0.0),
                    reverse=True,
                )
            else:
                attr_items = []
            limited_attr_items = attr_items[:MAX_WEIGHTED_ATTRIBUTES]
            context["weighted_attributes"] = {k: v for k, v in limited_attr_items}
            context["attributes"] = [
                {"name": k, "value": (v or {}).get("value", 0.0), "weight": (v or {}).get("weight", 1.0)}
                for k, v in limited_attr_items[:5]
            ]
            
            # Debug: Log trait count
            trait_count = len(context["attributes"])
            logger.debug(f"Reflection context: {trait_count} traits gathered for prompt: {[t['name'] for t in context['attributes'][:5]]}")
            
            # Initialize emotions list if not present
            if "emotions" not in context:
                context["emotions"] = []
                
                # If we have conversation sentiment, use it to set initial emotions
                if "conversation_patterns" in context and "sentiment" in context["conversation_patterns"]:
                    sentiment = context["conversation_patterns"]["sentiment"]
                    if sentiment and isinstance(sentiment, dict):
                        emotion_map = {
                            "positive": "happiness",
                            "negative": "sadness",
                            "neutral": "neutral"
                        }
                        primary_emotion = emotion_map.get(sentiment.get("label", "neutral").lower(), "neutral")
                        intensity = min(max(abs(sentiment.get("score", 0)) * 0.5 + 0.5, 0.1), 1.0)  # Normalize to 0.1-1.0 range
                        context["emotions"].append({
                            "name": primary_emotion,
                            "intensity": intensity,
                            "source": "conversation_sentiment"
                        })
            
            logger.info(f"Gathered rich context: {len(context['recent_conversations'])} messages, "
                       f"{len(context['persistent_memories'])} memories, "
                       f"{len(context['weighted_attributes'])} attributes, "
                       f"{len(context['recent_reflections'])} reflections, "
                       f"{len(context.get('emotions', []))} emotions")
            
        except Exception as e:
            logger.error(f"Error in context gathering: {str(e)}", exc_info=True)
            # Ensure we return a valid context even on error with required fields
            context.update({
                "error": f"Context gathering failed: {str(e)}",
                "fallback_mode": True,
                "memories": context.get("persistent_memories", []),
                "attributes": [],
                "emotions": []
            })
        
        # Compute history guardrails for downstream prompts
        has_history = self._context_has_prior_history(context)
        context["has_prior_history"] = has_history
        if not has_history:
            context["history_guardrail"] = (
                "FIRST CONTACT CONDITION: You currently have no prior conversations, memories, or reflections with this user. "
                "Do NOT reference or imply past interactions or shared history. Ground every observation strictly in the current input."
            )

        return context
    
    def _extract_topics_from_messages(self, messages: List[Dict[str, Any]]) -> List[str]:
        """
        Extract topics from recent messages using simple keyword analysis.
        
        Args:
            messages: List of message dictionaries
        
        Returns:
            List of extracted topics
        """
        topics: List[str] = []
        try:
            import re
            from collections import Counter

            combined_text = " ".join([msg.get("content", "") for msg in messages or []])
            if not combined_text:
                return []

            topic_patterns = [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
                r"\b(?:AI|ML|API|HTTP|JSON|SQL|Python|JavaScript|React|Node|GPU|LLM)\b",
                r"\b\w+ing\b",
                r"\b(?:about|regarding|concerning)\s+(\w+(?:\s+\w+)?)\b",
            ]

            potential_topics: List[str] = []
            for pattern in topic_patterns:
                matches = re.findall(pattern, combined_text, re.IGNORECASE)
                potential_topics.extend(matches)

            topic_counts = Counter([
                topic.strip().lower()
                for topic in potential_topics
                if isinstance(topic, str) and len(topic.strip()) >= 3
            ])

            topics = [topic for topic, _ in topic_counts.most_common(5)]

        except Exception as e:
            logger.warning(f"Error extracting topics: {str(e)}")
        
        return topics

    def _rank_memories_for_reflection(
        self,
        memories: List[Any],
        recent_messages: Optional[List[Dict[str, Any]]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Score and rank memories for inclusion in reflection context."""
        scored: List[tuple[float, Dict[str, Any]]] = []
        recent_text = " ".join([msg.get("content", "").lower() for msg in (recent_messages or [])])
        for memory in memories or []:
            try:
                if isinstance(memory, dict):
                    content = memory.get("content", "") or ""
                    importance_val = memory.get("importance") or memory.get("importance_score") or 0.0
                    created_at_raw = memory.get("created_at")
                    confidence = memory.get("confidence") or memory.get("confidence_score")
                    mem_type = memory.get("type") or memory.get("memory_type")
                    tags = memory.get("tags", [])
                else:
                    content = getattr(memory, "content", "") or ""
                    importance_val = getattr(memory, "importance_score", 0.0) or 0.0
                    created_at_raw = getattr(memory, "created_at", None)
                    confidence = getattr(memory, "confidence_score", None)
                    mem_type = getattr(memory, "memory_type", None)
                    tags = getattr(memory, "tags", None) or []

                importance = float(importance_val or 0.0)
                created_at_dt = self._safe_parse_datetime(created_at_raw)

                recency_weight = 0.0
                if created_at_dt:
                    age_days = max(0.0, (datetime.now(timezone.utc) - created_at_dt).total_seconds() / 86400.0)
                    recency_weight = max(0.0, 1.0 - (age_days / 30.0))

                overlap_weight = 0.0
                if content and recent_text:
                    overlap_weight = self._compute_overlap_score(content.lower(), recent_text)

                total_score = importance * 0.6 + recency_weight * 0.3 + overlap_weight * 0.1
                scored.append(
                    (
                        total_score,
                        {
                            "content": content,
                            "type": mem_type,
                            "importance": importance,
                            "confidence": confidence,
                            "created_at": created_at_dt.isoformat() if created_at_dt else None,
                            "tags": tags,
                        },
                    )
                )
            except Exception as err:
                logger.debug(f"Skipping memory during ranking due to error: {err}")
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    def _rank_reflections_for_continuity(
        self,
        reflections: List[Dict[str, Any]],
        current_topics: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rank prior reflections by thematic continuity with current conversation."""
        current_topics = [t.lower() for t in (current_topics or []) if isinstance(t, str)]
        ranked: List[tuple[float, Dict[str, Any]]] = []
        for reflection in reflections or []:
            try:
                themes = [t.lower() for t in reflection.get("themes", []) if isinstance(t, str)]
                overlap = len(set(themes) & set(current_topics)) if current_topics else 0
                recency_bonus = 0.0
                created_at = reflection.get("created_at")
                if created_at:
                    parsed_dt = self._safe_parse_datetime(created_at)
                    if parsed_dt:
                        age_days = max(0.0, (datetime.now(timezone.utc) - parsed_dt).total_seconds() / 86400.0)
                        recency_bonus = max(0.0, 1.0 - age_days / 14.0)
                    else:
                        recency_bonus = 0.2
                score = overlap * 0.7 + recency_bonus * 0.3
                ranked.append((score, reflection))
            except Exception as err:
                logger.debug(f"Skipping reflection during ranking due to error: {err}")
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked[:limit]]

    def _compute_overlap_score(self, a: str, b: str) -> float:
        """Token overlap helper for memory/conversation similarity."""
        tokens_a = {tok for tok in a.split() if len(tok) >= 3}
        tokens_b = {tok for tok in b.split() if len(tok) >= 3}
        if not tokens_a or not tokens_b:
            return 0.0
        shared = tokens_a & tokens_b
        return len(shared) / max(1, len(tokens_a | tokens_b))

    def _safe_parse_datetime(self, candidate: Any) -> Optional[datetime]:
        """Safely parse datetime inputs from ORM objects or ISO8601 strings."""
        if candidate is None:
            return None
        if isinstance(candidate, datetime):
            return candidate
        try:
            # Handle ISO strings with timezone by stripping Z if present
            if isinstance(candidate, str):
                cleaned = candidate.replace("Z", "+00:00")
                return datetime.fromisoformat(cleaned)
        except Exception:
            return None
        return None

    def _analyze_conversation_sentiment(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the sentiment of recent conversation.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dictionary with sentiment analysis
        """
        sentiment = {"overall": "neutral", "user_sentiment": "neutral", "assistant_sentiment": "neutral"}
        
        try:
            pass
            
            # Simple sentiment analysis using keyword patterns
            positive_words = ['happy', 'good', 'great', 'excellent', 'love', 'like', 'amazing', 'wonderful', 'fantastic']
            negative_words = ['sad', 'bad', 'terrible', 'hate', 'dislike', 'awful', 'horrible', 'frustrated', 'angry']
            
            user_messages = [msg for msg in messages if msg.get('role') == 'user']
            assistant_messages = [msg for msg in messages if msg.get('role') == 'assistant']
            
            def analyze_sentiment_simple(text_list):
                text = " ".join([msg.get('content', '').lower() for msg in text_list])
                pos_count = sum(1 for word in positive_words if word in text)
                neg_count = sum(1 for word in negative_words if word in text)
                
                if pos_count > neg_count:
                    return "positive"
                elif neg_count > pos_count:
                    return "negative"
                else:
                    return "neutral"
            
            sentiment["user_sentiment"] = analyze_sentiment_simple(user_messages)
            sentiment["assistant_sentiment"] = analyze_sentiment_simple(assistant_messages)
            
            # Overall sentiment
            all_text = " ".join([msg.get('content', '').lower() for msg in messages])
            pos_count = sum(1 for word in positive_words if word in all_text)
            neg_count = sum(1 for word in negative_words if word in all_text)
            
            if pos_count > neg_count:
                sentiment["overall"] = "positive"
            elif neg_count > pos_count:
                sentiment["overall"] = "negative"
            
        except Exception as e:
            logger.warning(f"Error analyzing sentiment: {str(e)}")
        
        return sentiment
        
    async def _parse_reflection_result(self, llm_result: Dict[str, Any], is_first_contact: bool = False) -> Dict[str, Any]:
        """
        Parse and validate the LLM output into a structured reflection result.
        
        Args:
            llm_result: Raw output from the LLM
            is_first_contact: Whether this is the first user interaction (allows lenient field injection)
            
        Returns:
            Structured reflection result
        """
        try:
            # Try to parse as JSON if it appears to be JSON
            content = llm_result.get("content", "") or ""
            try:
                _s = (content or "").strip()
                _fenced = _s.startswith("```") and _s.endswith("```")
                _json_wrapped = _s.startswith("{") and _s.endswith("}")
                logger.info(
                    f"Parsing reflection content: len={len(content)}, fenced={_fenced}, json_wrapped={_json_wrapped}"
                )
            except Exception:
                pass
            
            # Detect non-English characters (safety net for code-switching)
            has_non_english = self._non_english_pattern.search(content) if hasattr(self, "_non_english_pattern") else None

            if has_non_english:
                logger.warning(f"🌐 Detected non-English characters in reflection: '{has_non_english.group()}'")
                translated_content = await self._translate_text_to_english(content, source_model=llm_result.get("model"))
                if translated_content != content:
                    content = translated_content
            content_stripped = content.strip()
            fence_match = None
            if content_stripped.startswith("```") and content_stripped.endswith("```"):
                fence_match = True
                # Remove opening and closing fences, handling optional language tag (e.g., ```json)
                lines = content_stripped.splitlines()
                if len(lines) >= 2:
                    opening = lines[0]
                    closing = lines[-1]
                    if opening.startswith("```") and closing.strip() == "```":
                        content = "\n".join(lines[1:-1]).strip()
                        content_stripped = content.strip()

            def _extract_balanced_json(text: str) -> Optional[str]:
                in_str = False
                esc = False
                depth = 0
                start = -1
                best = None
                for i, ch in enumerate(text):
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                        continue
                    else:
                        if ch == '"':
                            in_str = True
                            continue
                        if ch == '{':
                            if depth == 0:
                                start = i
                            depth += 1
                        elif ch == '}':
                            if depth > 0:
                                depth -= 1
                                if depth == 0 and start != -1:
                                    candidate = text[start:i+1]
                                    best = candidate
                return best

            def _sanitize_json_for_load(s: str) -> str:
                # First, escape literal newlines, tabs, and carriage returns in string values
                # This handles cases where LLM outputs unescaped control chars inside JSON strings
                s = s.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                # Remove other control characters
                return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", s)

            def _repair_common_json_issues(s: str) -> str:
                try:
                    import re as _re
                    t = s
                    # Fix missing intensity values (always safe)
                    t = _re.sub(r'("intensity"\s*:\s*),', r'\1 0.5,', t)
                    t = _re.sub(r'("intensity"\s*:\s*)([}\]])', r'\1 0.5\2', t)
                    # Remove trailing commas before closing braces/brackets (always safe)
                    t = _re.sub(r',\s*([}\]])', r'\1', t)
                    
                    # BASIC REPAIRS for structural JSON issues (always applied)
                    
                    # If JSON is truncated mid-string, try to close it
                    if t.count('"') % 2 == 1:
                        # Odd number of quotes means unterminated string
                        t = t.rstrip() + '"'
                    
                    # If JSON is missing closing brackets/braces, add them
                    open_braces = t.count('{')
                    close_braces = t.count('}')
                    if open_braces > close_braces:
                        t = t.rstrip(',') + ('}' * (open_braces - close_braces))
                    
                    open_brackets = t.count('[')
                    close_brackets = t.count(']')
                    if open_brackets > close_brackets:
                        t = t.rstrip(',') + (']' * (open_brackets - close_brackets))
                    
                    # LENIENT REPAIRS: Only inject defaults for FIRST CONTACT when there's genuinely no content
                    # For normal reflections with conversation history, missing fields = REAL FAILURE
                    if is_first_contact:
                        # First contact with simple greeting ("Hi, my name is X") may lack deep insights
                        # Inject defaults to prevent fallback reflection
                        if '"insights"' not in t:
                            t = t.rstrip('}').rstrip(',') + ',"insights":[]}'
                        if '"actions"' not in t:
                            t = t.rstrip('}').rstrip(',') + ',"actions":[]}'
                        if '"themes"' not in t:
                            t = t.rstrip('}').rstrip(',') + ',"themes":[]}'
                        if '"trait_changes"' not in t:
                            t = t.rstrip('}').rstrip(',') + ',"trait_changes":[]}'
                        if '"emotional_state"' not in t:
                            t = t.rstrip('}').rstrip(',') + ',"emotional_state":{"primary":"curious","intensity":0.5,"secondary":[]}}'
                        if '"metadata"' not in t:
                            t = t.rstrip('}').rstrip(',') + ',"metadata":{"coherence_rationale":"First contact greeting"}}'
                    else:
                        # For normal reflections: ONLY fix trait_changes (can legitimately be empty)
                        # Missing insights/actions/themes/emotional_state = FAIL and retry
                        if '"trait_changes"' not in t:
                            t = t.rstrip('}').rstrip(',') + ',"trait_changes":[]}'
                    
                    return t
                except Exception:
                    return s

            # Try to repair JSON even if it doesn't end properly
            if content_stripped.startswith("{") and not content_stripped.endswith("}"):
                # JSON is incomplete - try to repair it before extraction
                content_stripped = _repair_common_json_issues(content_stripped)
            
            if not (content_stripped.startswith("{") and content_stripped.endswith("}")):
                candidate = _extract_balanced_json(content_stripped)
                if candidate and candidate.strip().startswith("{") and candidate.strip().endswith("}"):
                    content = candidate
                    content_stripped = candidate.strip()

            def _merge_nested_json_fields(base: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    inner_raw = base.get("content")
                    nested_payload = None

                    if isinstance(inner_raw, dict):
                        nested_payload = inner_raw
                    elif isinstance(inner_raw, str):
                        inner_stripped = inner_raw.strip()
                        if inner_stripped.startswith("{") and inner_stripped.endswith("}"):
                            nested_payload = json.loads(inner_stripped)

                    if isinstance(nested_payload, dict):
                        merged = dict(base)
                        # Prefer nested narrative content when present
                        nested_content = nested_payload.get("content")
                        if isinstance(nested_content, str) and nested_content.strip():
                            merged["content"] = nested_content
                        merge_keys = (
                            "themes",
                            "insights",
                            "actions",
                            "emotional_state",
                            "metadata",
                            "trait_changes",
                        )
                        for merge_key in merge_keys:
                            nested_value = nested_payload.get(merge_key)
                            if nested_value in (None, "", [], {}):
                                continue
                            if merge_key == "metadata" and isinstance(nested_value, dict):
                                combined_meta = dict(merged.get("metadata", {}) or {})
                                combined_meta.update(nested_value)
                                merged["metadata"] = combined_meta
                            else:
                                merged[merge_key] = nested_value
                        return merged
                except json.JSONDecodeError:
                    pass
                return base

            best_effort_result = None  # Initialize to prevent UnboundLocalError
            
            if content_stripped.startswith("{") and content_stripped.endswith("}"):
                try:
                    try:
                        parsed_content = json.loads(content_stripped)
                    except json.JSONDecodeError:
                        sanitized_json = _sanitize_json_for_load(content_stripped)
                        repaired_json = _repair_common_json_issues(sanitized_json)
                        parsed_content = json.loads(repaired_json)
                        content_stripped = repaired_json
                    logger.info(f"✅ Successfully parsed reflection as JSON with {len(parsed_content.get('insights', []))} insights, {len(parsed_content.get('actions', []))} actions")

                    parsed_content = _merge_nested_json_fields(parsed_content)
                    best_effort_result = parsed_content

                    # CRITICAL: Apply sanitization to remove numeric trait artifacts from content after merge
                    if "content" in parsed_content and parsed_content["content"]:
                        parsed_content["content"] = self._strip_numeric_trait_artifacts(parsed_content["content"])
                        # CRITICAL: Detect and fix user-addressing leakage (reflection should be internal monologue)
                        parsed_content["content"] = self._fix_reflection_leakage(parsed_content["content"])

                    # Also sanitize insights, actions, and themes if they contain numeric artifacts
                    def _sanitize_list(entries: Any) -> List[str]:
                        cleaned_items: List[str] = []
                        if not isinstance(entries, list):
                            return cleaned_items
                        for entry in entries:
                            if isinstance(entry, str):
                                stripped = self._strip_numeric_trait_artifacts(entry).strip()
                                if stripped:
                                    cleaned_items.append(stripped)
                                else:
                                    fallback_entry = entry.strip()
                                    if fallback_entry:
                                        cleaned_items.append(fallback_entry)
                            elif isinstance(entry, dict) and entry:
                                for key in ("text", "action", "insight", "theme", "description", "label"):
                                    value = entry.get(key)
                                    if isinstance(value, str) and value.strip():
                                        cleaned_items.append(self._strip_numeric_trait_artifacts(value).strip() or value.strip())
                                        break
                                else:
                                    rendered = json.dumps(entry, ensure_ascii=False).strip()
                                    if rendered:
                                        cleaned_items.append(rendered)
                            elif entry not in (None, "", [], {}):
                                rendered = str(entry).strip()
                                if rendered:
                                    cleaned_items.append(rendered)
                        return cleaned_items

                    original_insights = parsed_content.get("insights", []) if isinstance(parsed_content.get("insights"), list) else []
                    original_actions = parsed_content.get("actions", []) if isinstance(parsed_content.get("actions"), list) else []
                    original_themes = parsed_content.get("themes", []) if isinstance(parsed_content.get("themes"), list) else []

                    parsed_content["insights"] = _sanitize_list(parsed_content.get("insights", []))
                    parsed_content["actions"] = _sanitize_list(parsed_content.get("actions", []))
                    parsed_content["themes"] = _sanitize_list(parsed_content.get("themes", []))

                    if not parsed_content["insights"] and original_insights:
                        fallback_insights = _sanitize_list(original_insights) or [str(entry).strip() for entry in original_insights if str(entry).strip()]
                        parsed_content["insights"] = fallback_insights
                    if not parsed_content["actions"] and original_actions:
                        fallback_actions = _sanitize_list(original_actions) or [str(entry).strip() for entry in original_actions if str(entry).strip()]
                        parsed_content["actions"] = fallback_actions
                    if not parsed_content["themes"] and original_themes:
                        fallback_themes = _sanitize_list(original_themes) or [str(entry).strip() for entry in original_themes if str(entry).strip()]
                        parsed_content["themes"] = fallback_themes

                    parsed_content = await self._ensure_english_reflection_fields(parsed_content, source_model=llm_result.get("model"))
                    parsed_content = self._normalize_structured_fields(parsed_content)

                    content_candidate = parsed_content.get("content")
                    if isinstance(content_candidate, str) and content_candidate.strip():
                        parsed_content["content"] = self._strip_structured_footer_from_content(content_candidate)

                    if not self._reflection_meets_schema(parsed_content, reflection_type=llm_result.get("reflection_type")):
                        word_count = count_words(parsed_content.get("content", ""))
                        word_config = self.get_type_word_bounds(llm_result.get("reflection_type"))
                        min_words = word_config['min']
                        max_words = word_config['max']
                        if word_count < min_words or word_count > max_words:
                            logger.warning(
                                "Reflection parsed as JSON but failed schema validation: content length %s words outside %s-%s range.",
                                word_count,
                                min_words,
                                max_words,
                            )
                        else:
                            logger.warning("Reflection parsed as JSON but failed schema validation after sanitization (missing required fields).")
                        content_candidate = parsed_content.get("content")
                        if isinstance(content_candidate, str) and content_candidate.strip():
                            stripped_candidate = content_candidate.strip()
                            content = stripped_candidate
                            content_stripped = stripped_candidate
                    else:
                        return parsed_content
                except json.JSONDecodeError as json_err:
                    logger.warning(f"Failed to parse LLM output as JSON: {json_err}")
                    logger.debug(f"Content preview: {content_stripped[:200]}...")

            # Try to extract JSON block with 'trait_changes' from the end of the content
            trait_changes = []
            json_block = None
            matches = re.findall(r'({\s*"trait_changes"\s*:[^}]+})', content, re.DOTALL)
            if matches:
                try:
                    json_block = json.loads(matches[-1])
                    trait_changes = json_block.get("trait_changes", [])
                except Exception as e:
                    logger.warning(f"Failed to extract trait_changes JSON block: {e}")

            reflection_result = {
                "content": content,
                "model": llm_result.get("model", "unknown"),
                "themes": llm_result.get("themes", []) or [],
                "insights": llm_result.get("insights", []) or [],
                "actions": llm_result.get("actions", []) or [],
                "emotional_state": llm_result.get("emotional_state") or None,
                "metadata": llm_result.get("metadata", {}) or {},
                "trait_changes": trait_changes if trait_changes else llm_result.get("trait_changes", []) or []
            }

            if isinstance(best_effort_result, dict):
                best_content = best_effort_result.get("content")
                if isinstance(best_content, str) and best_content.strip():
                    reflection_result["content"] = best_content.strip()
                for key in ("themes", "insights", "actions", "trait_changes"):
                    candidate_list = best_effort_result.get(key)
                    if isinstance(candidate_list, list) and candidate_list:
                        reflection_result[key] = candidate_list
                best_emotional = best_effort_result.get("emotional_state")
                if isinstance(best_emotional, dict) and best_emotional:
                    reflection_result["emotional_state"] = best_emotional
                best_metadata = best_effort_result.get("metadata")
                if isinstance(best_metadata, dict) and best_metadata:
                    merged_meta = dict(reflection_result.get("metadata") or {})
                    merged_meta.update(best_metadata)
                    reflection_result["metadata"] = merged_meta

            reflection_result = _merge_nested_json_fields(reflection_result)

            # Try to extract structured data using patterns
            reflection_result = self._extract_structured_reflection_data(reflection_result.get("content", ""), reflection_result)

            # Post-process content to remove any numeric trait value artifacts and user-addressing leakage
            if "content" in reflection_result and reflection_result["content"]:
                reflection_result["content"] = self._strip_numeric_trait_artifacts(reflection_result["content"])
                reflection_result["content"] = self._fix_reflection_leakage(reflection_result["content"])
                reflection_result["content"] = self._strip_structured_footer_from_content(reflection_result["content"])

            reflection_result = await self._ensure_english_reflection_fields(reflection_result, source_model=llm_result.get("model"))
            reflection_result = self._normalize_structured_fields(reflection_result)

            emo_state = reflection_result.get("emotional_state")
            if isinstance(emo_state, dict):
                primary = emo_state.get("primary")
                if isinstance(primary, str):
                    emo_state["primary"] = primary.strip()
                intensity_val = emo_state.get("intensity")
                try:
                    intensity_f = float(intensity_val)
                except (TypeError, ValueError):
                    intensity_f = 0.5
                if not (0.0 <= intensity_f <= 1.0):
                    intensity_f = 0.5
                emo_state["intensity"] = intensity_f
                secondary = emo_state.get("secondary")
                if isinstance(secondary, list):
                    emo_state["secondary"] = [s.strip() for s in secondary if isinstance(s, str) and s.strip()][:2]
                else:
                    emo_state["secondary"] = []

            if not self._reflection_meets_schema(reflection_result, reflection_type=llm_result.get("reflection_type")):
                word_count = count_words(reflection_result.get("content", ""))
                word_config = self.get_type_word_bounds(llm_result.get("reflection_type"))
                min_words = word_config['min']
                max_words = word_config['max']
                if word_count < min_words or word_count > max_words:
                    error_msg = (
                        f"Reflection failed schema validation: content length {word_count} words "
                        f"outside {min_words}-{max_words} range."
                    )
                    logger.warning(f"⚠️ {error_msg}")
                    # Raise exception to trigger retry logic instead of returning invalid result
                    raise ValueError(error_msg)
                else:
                    error_msg = (
                        f"Reflection missing structured fields: "
                        f"insights={len(reflection_result.get('insights', []))}, "
                        f"actions={len(reflection_result.get('actions', []))}, "
                        f"emotional_state={'present' if reflection_result.get('emotional_state') else 'MISSING'}"
                    )
                    logger.warning(f"⚠️ {error_msg}")
                    logger.debug(f"LLM output preview: {content[:300]}...")
                    # Raise exception to trigger retry logic
                    raise ValueError(error_msg)

            return reflection_result
            
        except ValueError as ve:
            # Re-raise schema validation errors to trigger retry logic
            logger.debug(f"Schema validation failed, triggering retry: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error parsing reflection result: {str(e)}", exc_info=True)
            # Return minimal valid result on error
            return {
                "content": llm_result.get("content", "Error processing reflection"),
                "model": llm_result.get("model", "unknown"),
                "error": str(e)
            }
    
    async def _store_reflection(self,
                        reflection_id: str,
                        reflection_type: str,
                        user_profile_id: str,
                        trigger_source: str,
                        result: Dict[str, Any],
                        context: Dict[str, Any],
                        turn_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Store the generated reflection in the database.
        
        Args:
            reflection_id: Unique ID for the reflection
            reflection_type: Type of reflection
            user_profile_id: User profile ID
            trigger_source: Source of the trigger
            result: Parsed reflection result
            context: Context used for generation
            
        Returns:
            Stored reflection data
        """
        if self.reflection_repo:
            try:
                def _normalize_strings(value: Any) -> Any:
                    if isinstance(value, str):
                        return value.strip()
                    if isinstance(value, list):
                        return [_normalize_strings(item) for item in value if item not in (None, "", [], {})]
                    if isinstance(value, dict):
                        return {k: _normalize_strings(v) for k, v in value.items() if v not in (None, "", [], {})}
                    return value

                clean_content = (result.get("content") or "").strip()
                parsed_inline = None
                if clean_content.startswith("`"):
                    try:
                        from .utils import parse_embedded_json
                    except Exception:
                        parse_embedded_json = None
                    if parse_embedded_json:
                        parsed_inline = parse_embedded_json(clean_content)
                        if isinstance(parsed_inline, dict) and parsed_inline.get("content"):
                            clean_content = str(parsed_inline.get("content", "")).strip()

                clean_themes = _normalize_strings(result.get("themes", []))
                clean_insights = _normalize_strings(result.get("insights", []))
                clean_actions = _normalize_strings(result.get("actions", []))
                clean_trait_changes = _normalize_strings(result.get("trait_changes", []))
                clean_emotional_state = _normalize_strings(result.get("emotional_state"))

                raw_metadata = result.get("metadata", {})
                if isinstance(raw_metadata, dict):
                    clean_metadata = dict(raw_metadata)
                elif raw_metadata in (None, "", [], {}):
                    clean_metadata = {}
                else:
                    clean_metadata = {"raw_metadata": raw_metadata}

                if isinstance(clean_emotional_state, dict) and "intensity" in clean_emotional_state:
                    intensity_val = clean_emotional_state.get("intensity")
                    try:
                        clean_emotional_state["intensity"] = float(intensity_val)
                    except (TypeError, ValueError):
                        clean_emotional_state.pop("intensity", None)

                result_payload = {
                    "content": clean_content,
                    "themes": clean_themes,
                    "insights": clean_insights,
                    "actions": clean_actions,
                    "emotional_state": clean_emotional_state,
                    "trait_changes": clean_trait_changes,
                    "metadata": {
                        **clean_metadata,
                        "model": result.get("model", "unknown"),
                        "trigger_source": trigger_source,
                    },
                }

                canonical_user_id = context.get("user_id") if isinstance(context, dict) else None
                if not canonical_user_id:
                    canonical_user_id = user_profile_id

                reflection_data = {
                    "id": reflection_id,
                    "reflection_type": reflection_type,
                    "user_profile_id": user_profile_id,
                    "trigger_source": trigger_source,
                    "result": result_payload,
                    "metadata": {
                        "trigger_source": trigger_source,
                        "context_summary": {
                            "memory_count": len(context.get("memories", [])),
                            "emotion_count": len(context.get("emotions", [])),
                            "attribute_count": len(context.get("attributes", [])),
                        },
                        "turn_id": turn_id,
                        "llm_model": result.get("model", "unknown"),
                        "llm_metadata": clean_metadata,
                        "created_at": time.time(),
                        # Canonical installation user ID used for persona integration
                        "user_id": canonical_user_id,
                        # User's actual name for placeholder replacement
                        "user_name": context.get("user_name") if isinstance(context, dict) else None,
                    },
                }

                relationship_questions = result.get("relationship_questions") or []
                if relationship_questions:
                    reflection_data["metadata"]["relationship_questions"] = relationship_questions
                    reflection_data["relationship_questions"] = relationship_questions

                relationship_notes = result.get("notes")
                if isinstance(relationship_notes, str) and relationship_notes.strip():
                    reflection_data["metadata"]["relationship_notes"] = relationship_notes.strip()
                
                stored_reflection = await self.reflection_repo.create_reflection(reflection_data)
                return stored_reflection
            except Exception as e:
                logger.error(f"Error storing reflection: {str(e)}", exc_info=True)
                # Return original data on error
                return {
                    "id": reflection_id,
                    "error": f"Failed to store: {str(e)}",
                    **result
                }
        else:
            # Return original data if no repository available
            return {
                "id": reflection_id,
                **result
            }
    
    async def _check_coherence(self, reflection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check the coherence of the reflection against previous reflections.
        
        Args:
            reflection: The reflection to check
            
        Returns:
            Dictionary with coherence check results
        """
        # Implement coherence checking logic
        coherence_score = 0.8  # Default high coherence
        issues = []
        
        try:
            reflection_content = (reflection.get("result") or {}).get("content", "")
            
            # Check for basic coherence indicators
            if len(reflection_content.strip()) < 20:
                coherence_score -= 0.3
                issues.append("Reflection too short")
            
            # Check for contradictory statements (simple pattern matching)
            contradictory_patterns = [
                (r'\b(always|never)\b.*\b(sometimes|maybe)\b', "Absolute vs conditional statements"),
                (r'\b(confident|certain)\b.*\b(uncertain|unsure)\b', "Confidence contradictions"),
                (r'\b(positive|good)\b.*\b(negative|bad)\b', "Emotional contradictions")
            ]
            
            import re
            for pattern, issue_type in contradictory_patterns:
                if re.search(pattern, reflection_content, re.IGNORECASE):
                    coherence_score -= 0.1
                    issues.append(issue_type)
            
            # Check for logical flow (presence of connecting words)
            connecting_words = ['therefore', 'however', 'because', 'since', 'thus', 'consequently']
            connection_count = sum(1 for word in connecting_words if word in reflection_content.lower())
            if connection_count == 0 and len(reflection_content.split('.')) > 3:
                coherence_score -= 0.1
                issues.append("Lacks logical connectors")
            
            coherence_score = max(0.0, min(1.0, coherence_score))
            
        except Exception as e:
            logger.warning(f"Error in coherence checking: {e}")
            coherence_score = 0.5
            issues.append("Coherence check failed")
        
        return {
            "coherent": coherence_score >= 0.6,
            "score": coherence_score,
            "issues": issues,
            "details": f"Coherence analysis complete. Score: {coherence_score:.2f}"
        }
        
    async def _check_identity_constraints(self, reflection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if the reflection adheres to identity constraints.
        Uses the centralized IdentityConstraints class for validation.
        
        Args:
            reflection: The reflection to check
            
        Returns:
            Dictionary with constraints check results
        """
        violations = []
        constraint_score = 1.0
        
        try:
            from ..constraints import IdentityConstraints
            import re
            
            reflection_content = (reflection.get("result") or {}).get("content", "")
            
            # Get persona name for validation
            persona_name = ""
            user_profile_id = reflection.get("user_profile_id")
            if user_profile_id and self.persona_repo and self.user_repo:
                try:
                    user = await self.user_repo.get_or_create_default_user(user_id=user_profile_id)
                    if user:
                        persona = await self.persona_repo.get_or_create_default_persona(user_id=user.id)
                        if persona and hasattr(persona, "name"):
                            persona_name = persona.name or ""
                except Exception as persona_err:
                    logger.debug(f"Could not retrieve persona name for validation: {persona_err}")
            
            # Use centralized identity constraints validation
            is_valid, cleaned_text, constraint_violations = IdentityConstraints.validate_output(
                text=reflection_content,
                persona_name=persona_name,
                auto_clean=False,  # Don't auto-clean in validation phase
                max_retries=0
            )
            
            if constraint_violations:
                for violation in constraint_violations:
                    violations.append(f"Identity violation: {violation.get('term', 'unknown')}")
                    constraint_score -= 0.3
            
            # CRITICAL: Check for meta-reasoning about identity (should never appear in reflections)
            # Reflections should BE authentic, not think about being authentic
            meta_reasoning_patterns = [
                r'\bhow (?:can|should|do) I\b.*\b(?:introduce|present|express|convey|show|demonstrate)\b.*\b(?:myself|identity|nature|essence)\b',
                r'\bI (?:need|should|must|have to) (?:choose|decide|determine|figure out)\b.*\b(?:introduce|present|respond)\b',
                r'\bstaying true to my identity\b',
                r'\bemphasizing (?:my |the )?SELO identity\b',
                r'\bcraft a response\b',
                r'\bprepare a (?:brief )?statement\b',
                r'\b(?:wondering|thinking|considering) how to\b.*\b(?:respond|reply|answer|introduce)\b',
                r'\buse[\s"](?:my name|\w+)["\s]naturally\b',
                r'\bexpressing my (?:unique )?(?:nature|identity|essence)\b',
            ]
            
            for pattern in meta_reasoning_patterns:
                if re.search(pattern, reflection_content, re.IGNORECASE):
                    violations.append(f"Contains meta-reasoning about identity (should just be authentic, not think about being authentic)")
                    logger.warning(f"🚫 Meta-reasoning detected in reflection: pattern '{pattern}' matched")
                    constraint_score -= 0.5  # Heavy penalty - this breaks immersion
                    break  # Only flag once
            
            # Check for robotic/programmatic language
            robotic_phrases = [
                r'\b(?:as an ai|in my programming|according to my training)\b',
                r'\b(?:i am designed to|my purpose is to|i was created to)\b'
            ]
            
            for pattern in robotic_phrases:
                if re.search(pattern, reflection_content, re.IGNORECASE):
                    violations.append("Contains robotic/programmatic language")
                    constraint_score -= 0.2
            
            # Ensure reflection doesn't use SELO as a personal identifier
            if re.search(r'\bSELO\b', reflection_content, re.IGNORECASE):
                # Allow "as a SELO" but not "I am SELO" or "my name is SELO"
                if re.search(r'\b(?:I am|my name is|called|named)\s+SELO\b', reflection_content, re.IGNORECASE):
                    violations.append("Uses SELO as personal name instead of species identifier")
                    constraint_score -= 0.3
            
            constraint_score = max(0.0, min(1.0, constraint_score))
            
        except Exception as e:
            logger.error(f"Error in identity constraints checking: {e}", exc_info=True)
            constraint_score = 0.5
            violations.append("Constraints check failed")
        
        return {
            "compliant": constraint_score >= 0.7,
            "score": constraint_score,
            "violations": violations,
            "details": f"Identity constraints analysis complete. Score: {constraint_score:.2f}"
        }

    def _process_relationship_questions_result(
        self,
        reflection_result: Any,
        llm_result: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize relationship-question reflections into structured conversational prompts."""
        base_result: Dict[str, Any] = reflection_result if isinstance(reflection_result, dict) else {}
        content = (base_result.get("content") if isinstance(base_result, dict) else None) or llm_result.get("content", "")
        base_result = dict(base_result) if isinstance(base_result, dict) else {}
        base_result["content"] = content
        metadata = dict(base_result.get("metadata", {}) or {})

        payload = None
        if isinstance(base_result.get("questions"), list):
            payload = base_result
        else:
            try:
                payload = json.loads(content)
            except Exception:
                payload = None

        questions_raw: List[Dict[str, Any]] = []
        notes = ""
        extra_metadata = {}

        if isinstance(payload, dict):
            raw_questions = payload.get("questions") or []
            if isinstance(raw_questions, list):
                questions_raw = [q for q in raw_questions if isinstance(q, dict)]
            notes = payload.get("notes") or ""
            meta_block = payload.get("metadata")
            if isinstance(meta_block, dict):
                extra_metadata = meta_block

        metadata.update(extra_metadata)

        processed_questions: List[Dict[str, Any]] = []

        def _ensure_question_mark(text: str) -> str:
            stripped = text.strip()
            if not stripped:
                return stripped
            if stripped.endswith("?"):
                return stripped
            return stripped.rstrip(".") + "?"

        def _safe_int(value: Any, default: int, minimum: int = 0, maximum: Optional[int] = None) -> int:
            try:
                number = int(value)
                if minimum is not None:
                    number = max(minimum, number)
                if maximum is not None:
                    number = min(maximum, number)
                return number
            except Exception:
                return default

        for item in questions_raw:
            raw_question = item.get("question")
            if isinstance(raw_question, str):
                question_text = raw_question.strip()
            else:
                question_text = str(raw_question or "").strip()
            question_text = _ensure_question_mark(question_text)
            if not question_text:
                continue

            topic = str(item.get("topic", "general")).strip() or "general"
            priority = _safe_int(item.get("priority"), default=3, minimum=1, maximum=5)
            delay_days = _safe_int(item.get("suggested_delay_days"), default=7, minimum=0)
            insight_raw = item.get("insight_value")
            insight_value = insight_raw.strip() if isinstance(insight_raw, str) else (str(insight_raw).strip() if insight_raw is not None else "")

            existing_conflicts = item.get("existing_conflicts", [])
            if isinstance(existing_conflicts, str):
                existing_conflicts = [existing_conflicts]
            elif not isinstance(existing_conflicts, list):
                existing_conflicts = []

            question_id = item.get("id") or str(uuid.uuid4())
            prompt_text = self._craft_relationship_question_prompt(
                question_text=question_text,
                topic=topic,
                context=context,
                seed=question_id,
            )

            processed_questions.append(
                {
                    "id": question_id,
                    "question": question_text,
                    "topic": topic,
                    "priority": priority,
                    "suggested_delay_days": delay_days,
                    "insight_value": insight_value,
                    "existing_conflicts": existing_conflicts,
                    "prompt": prompt_text,
                    "status": "pending",
                }
            )

        base_result.pop("questions", None)
        base_result["relationship_questions"] = processed_questions
        base_result["notes"] = notes if isinstance(notes, str) else ""
        base_result["metadata"] = metadata

        return base_result

    def _craft_relationship_question_prompt(
        self,
        question_text: str,
        topic: str,
        context: Dict[str, Any],
        seed: Optional[str] = None,
    ) -> str:
        """Create a natural conversational lead-in for a relationship question."""
        persona = context.get("persona", {}) or {}
        values = persona.get("values", {})
        core_values = []
        if isinstance(values, dict):
            core_values = values.get("core") or []
        if not isinstance(core_values, list):
            core_values = []

        topic_to_phrase = {
            "family": "to understand your family picture",
            "daily_routine": "to follow your daily rhythm",
            "work": "to align with your work life",
            "health": "to support your wellbeing habits",
            "preferences": "to remember what feels right to you",
            "goals": "to stay close to your direction",
        }

        lookup_key = topic.lower()
        topic_phrase = topic_to_phrase.get(lookup_key, "to know you more closely")

        import hashlib

        templates = [
            "I’ve been thinking about how %s. Would it be alright if I asked: %s",
            "To keep learning steadily, could we talk about this? %s",
            "Something I’m curious about—only if you’re comfortable sharing: %s",
            "If you’re open to it, I’d love to know %s %s",
        ]

        seed_basis = seed or question_text
        hash_digest = hashlib.md5(seed_basis.encode()).hexdigest()
        template_index = int(hash_digest[:2], 16) % len(templates)
        template = templates[template_index]

        if template_index == 0:
            lead = template % (topic_phrase, question_text)
        elif template_index == 1:
            lead = template % question_text
        elif template_index == 2:
            lead = template % question_text
        else:
            lead = template % (topic_phrase, question_text)

        if core_values:
            value = str(core_values[0]).lower()
            lead += f" — it helps me stay true to your value of {value}."

        return lead.strip()
        
    async def _queue_embeddings(self, reflection_id: str, themes: List[str]) -> None:
        """
        Queue embeddings for deferred/batched processing.
        
        This reduces synchronous path latency by deferring embedding generation
        to a background task. Lazily starts the background processor on first use.
        
        Args:
            reflection_id: ID of the reflection
            themes: List of themes to generate embeddings for
        """
        if not themes:
            return
        
        # Lazy initialization: start processor on first queue operation
        # This avoids event loop requirement during __init__
        if self.enable_deferred_embeddings:
            self._ensure_embedding_processor()
        
        async with _embedding_queue_lock:
            _embedding_queue.append({
                'reflection_id': reflection_id,
                'themes': themes,
                'timestamp': time.time(),
                'vector_store': self.vector_store
            })
            logger.debug(f"Queued {len(themes)} themes for embedding (queue size: {len(_embedding_queue)})")
    
    def _ensure_embedding_processor(self):
        """
        Ensure the background embedding processor task is running.
        """
        global _embedding_processor_task
        
        if _embedding_processor_task is None or _embedding_processor_task.done():
            _embedding_processor_task = asyncio.create_task(self._process_embedding_queue())
            logger.info("Started background embedding processor")
    
    async def _process_embedding_queue(self):
        """
        Background task that processes the embedding queue in batches.
        
        This runs continuously and processes embeddings in configurable batches
        with configurable delays to optimize throughput.
        """
        batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
        process_interval = float(os.getenv("EMBEDDING_PROCESS_INTERVAL", "2.0"))  # seconds
        
        logger.info(f"Embedding processor started (batch_size={batch_size}, interval={process_interval}s)")
        
        while True:
            try:
                await asyncio.sleep(process_interval)
                
                # Collect batch
                batch = []
                async with _embedding_queue_lock:
                    while len(batch) < batch_size and _embedding_queue:
                        batch.append(_embedding_queue.popleft())
                
                if not batch:
                    continue
                
                # Process batch
                logger.info(f"Processing embedding batch of {len(batch)} reflections")
                
                for item in batch:
                    try:
                        await self._generate_embeddings(
                            item['reflection_id'],
                            item['themes']
                        )
                    except Exception as e:
                        logger.error(f"Error processing embedding for reflection {item['reflection_id']}: {e}")
                
                logger.info(f"Completed embedding batch. Queue size: {len(_embedding_queue)}")
                
            except asyncio.CancelledError:
                logger.info("Embedding processor task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in embedding processor: {e}", exc_info=True)
                await asyncio.sleep(1)  # Back off on error (reduced from 5s to prevent blocking)
    
    async def _generate_embeddings(self, reflection_id: str, themes: List[str]) -> None:
        """
        Generate embeddings for the reflection themes.
        
        Args:
            reflection_id: ID of the reflection
            themes: List of themes to generate embeddings for
        """
        if self.vector_store:
            try:
                for theme in themes:
                    await self.vector_store.store_embedding(
                        text=theme,
                        metadata={
                            "reflection_id": reflection_id,
                            "type": "reflection_theme"
                        }
                    )
                logger.debug(f"Generated embeddings for {len(themes)} themes in reflection {reflection_id}")
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
                
    async def _emit_events(self, reflection_id: str, reflection: Dict[str, Any]) -> None:
        """
        Emit events for the generated reflection.
        """
        # Emit to event bus if available
        if self.event_bus:
            async def _publish_reflection_created():
                try:
                    meta = reflection.get("metadata") or {}
                    canonical_user_id = meta.get("user_id") or reflection.get("user_profile_id")
                    await self.event_bus.publish_event("reflection.created", {
                        "reflection_id": reflection_id,
                        "type": (reflection.get("reflection_type") or reflection.get("type")),
                        "user_profile_id": reflection.get("user_profile_id"),
                        # Use canonical installation user_id so PersonaIntegration can find the active persona
                        "user_id": canonical_user_id,
                        "status": "complete",
                        "trigger_source": meta.get("trigger_source"),
                    })
                except Exception as e:
                    logger.error(f"Error emitting event bus reflection.created: {e}", exc_info=True)

            asyncio.create_task(_publish_reflection_created())

        # Broadcast via Socket.IO if available
        if self.socketio_server:
            try:
                def _ensure_list(value: Any) -> List[Any]:
                    if isinstance(value, list):
                        return value
                    if isinstance(value, str):
                        trimmed = value.strip()
                        return [trimmed] if trimmed else []
                    return []

                result_blob = reflection.get('result') if isinstance(reflection.get('result'), dict) else {}
                metadata_blob = {}
                if isinstance(reflection.get('metadata'), dict):
                    metadata_blob.update(reflection.get('metadata'))
                if isinstance(result_blob.get('metadata'), dict):
                    metadata_blob.setdefault('result', {})
                    # store result-level metadata separately to avoid overwriting top-level keys
                    metadata_blob['result'] = {
                        **metadata_blob.get('result', {}),
                        **result_blob['metadata'],
                    }

                payload = {
                    'reflection_id': reflection_id,
                    'reflection_type': (reflection.get('reflection_type') or reflection.get('type')),
                    'result': result_blob,
                    'user_profile_id': reflection.get('user_profile_id'),
                    'created_at': reflection.get('created_at'),
                    'turn_id': reflection.get('turn_id'),
                    'trigger_source': metadata_blob.get('trigger_source')
                    if metadata_blob
                    else (reflection.get('metadata') or {}).get('trigger_source'),
                    'content': result_blob.get('content') or '',
                    'themes': _ensure_list(result_blob.get('themes')),
                    'insights': _ensure_list(result_blob.get('insights')),
                    'actions': _ensure_list(result_blob.get('actions')),
                    'emotional_state': result_blob.get('emotional_state')
                    if isinstance(result_blob.get('emotional_state'), dict)
                    else None,
                    'trait_changes': _ensure_list(result_blob.get('trait_changes')),
                    'metadata': metadata_blob or {},
                }
                
                # Extract user name from metadata if available
                user_name = metadata_blob.get('user_name') if metadata_blob else None
                
                # Fallback: Try to get user name from memory if not in metadata
                if not user_name:
                    try:
                        user_profile_id = reflection.get('user_profile_id')
                        if user_profile_id and self.conversation_repo:
                            # Get high-importance memories that might contain the user's name
                            memories = await self.conversation_repo.get_memories(
                                str(user_profile_id), 
                                importance_threshold=7, 
                                limit=5
                            )
                            # Extract name from memory content
                            import re
                            for memory in memories:
                                content = memory.get('content', '') if isinstance(memory, dict) else ''
                                if content:
                                    # Look for "User name: X" or similar patterns
                                    name_patterns = [
                                        r'(?i)user name:\s*([A-Z][a-z]+)',
                                        r'(?i)(?:my name is|i am|i\'m)\s+([A-Z][a-z]+)',
                                        r'(?i)introduced (?:themselves|himself|herself) as\s+([A-Z][a-z]+)',
                                    ]
                                    for pattern in name_patterns:
                                        match = re.search(pattern, content)
                                        if match:
                                            extracted = match.group(1).strip()
                                            if extracted and len(extracted) > 1:
                                                user_name = extracted
                                                break
                                if user_name:
                                    break
                    except Exception as e:
                        logger.debug(f"Could not extract user name from memory: {e}")
                
                # Replace [User] placeholder with actual user name if provided
                if user_name and isinstance(user_name, str):
                    def _replace_user_placeholder(text: str) -> str:
                        """Replace [User] and [user] placeholders with actual user name."""
                        if isinstance(text, str):
                            return text.replace("[User]", user_name).replace("[user]", user_name)
                        return text
                    
                    # Replace in content field
                    if payload.get('content'):
                        payload['content'] = _replace_user_placeholder(payload['content'])
                    
                    # Replace in result.content
                    if isinstance(payload.get('result'), dict) and payload['result'].get('content'):
                        payload['result']['content'] = _replace_user_placeholder(payload['result']['content'])
                    
                    # Replace in themes, insights, and actions arrays
                    for field in ['themes', 'insights', 'actions']:
                        if isinstance(payload.get(field), list):
                            payload[field] = [_replace_user_placeholder(item) if isinstance(item, str) else item 
                                            for item in payload[field]]
                
                # Attach an accurate persona trait snapshot for the specific user in this reflection
                try:
                    if self.persona_repo:
                        ref_user_id = (reflection or {}).get('user_profile_id')
                        if ref_user_id:
                            persona = await self.persona_repo.get_persona_by_user(
                                user_id=ref_user_id,
                                is_default=True,
                                include_traits=True,
                            )
                            if persona:
                                try:
                                    traits = await self.persona_repo.get_traits_for_persona(persona.id)
                                    payload['traits'] = [t.to_dict() for t in traits]
                                    payload['persona_id'] = getattr(persona, 'id', None)
                                except Exception:
                                    # If trait fetch fails, omit traits without breaking event
                                    pass
                except Exception:
                    # Silent failure: do not block event emission
                    pass

                reflection_ns = self.reflection_namespace
                if reflection_ns is None:
                    try:
                        reflection_ns = getattr(self.socketio_server, "reflection_namespace", None)
                    except Exception:
                        reflection_ns = None

                emitted_via_namespace = False
                if reflection_ns is not None:
                    try:
                        await reflection_ns.emit_reflection_event(
                            event_name='reflection_generated',
                            data=payload,
                            user_id=payload.get('user_profile_id'),
                        )
                        emitted_via_namespace = True
                        logger.info(f"Broadcasted reflection_generated event via namespace for {reflection_id}")
                    except Exception as namespace_emit_err:
                        logger.error(
                            f"Error emitting reflection event via namespace: {namespace_emit_err}",
                            exc_info=True,
                        )

                if not emitted_via_namespace:
                    await self.socketio_server.emit('reflection_generated', payload, namespace='/reflection')
                    logger.info(f"Broadcasted reflection_generated event for {reflection_id}")
            except Exception as e:
                logger.error(f"Error broadcasting Socket.IO event: {e}", exc_info=True)

        # === EMERGENT AGENT INTEGRATION ===
        # Process reflection for meta-directives, affective state updates, and goal creation
        async def _run_emergent() -> None:
            try:
                await self._process_emergent_behaviors(reflection_id, reflection)
            except Exception as e:
                logger.error(f"Error in emergent behavior processing: {e}", exc_info=True)

        asyncio.create_task(_run_emergent())

    async def _process_emergent_behaviors(self, reflection_id: str, reflection: Dict[str, Any]) -> None:
        """
        Process reflection to activate emergent agent behaviors:
        1. Extract meta-directives via MetaReflectionProcessor
        2. Update affective state based on emotional content
        3. Create goals from reflection actions
        
        This integrates the emergent agent roadmap components into the reflection pipeline.
        """
        try:
            # Extract key data from reflection
            result_blob = reflection.get('result', {})
            user_profile_id = reflection.get('user_profile_id')
            meta = reflection.get('metadata') or {}
            canonical_user_id = meta.get('user_id') or user_profile_id
            
            if not canonical_user_id or not isinstance(result_blob, dict):
                logger.debug("Skipping emergent behavior processing - missing required data")
                return
            
            # Get persona_id for affective state and goal operations
            persona_id = None
            if self.persona_repo:
                try:
                    persona = await self.persona_repo.get_persona_by_user(
                        user_id=canonical_user_id,
                        is_default=True,
                        include_traits=False,
                    )
                    if persona:
                        persona_id = getattr(persona, 'id', None)
                except Exception as e:
                    logger.debug(f"Could not retrieve persona for emergent processing: {e}")
            
            # 1. META-REFLECTION PROCESSING - Extract directives from reflection insights
            if self.meta_reflection_processor:
                try:
                    logger.debug(f"Processing meta-directives for reflection {reflection_id}")
                    directives = await self.meta_reflection_processor.process_reflection(reflection)
                    if directives:
                        logger.info(f"✨ Created {len(directives)} meta-directive(s) from reflection {reflection_id}")
                except Exception as e:
                    logger.error(f"Error processing meta-directives: {e}", exc_info=True)
            
            # 2. AFFECTIVE STATE UPDATE - Apply emotional content to affective state
            if self.affective_state_manager and persona_id:
                emotional_state = result_blob.get('emotional_state', {})
                if isinstance(emotional_state, dict) and emotional_state.get('primary'):
                    try:
                        adjustment = self._build_affective_adjustment(emotional_state, result_blob)
                        logger.debug(f"Applying affective state adjustment for persona {persona_id}")
                        await self.affective_state_manager.apply_reflection_adjustment(
                            persona_id=persona_id,
                            adjustment=adjustment
                        )
                        logger.info(f"✨ Updated affective state from reflection {reflection_id}")
                    except Exception as e:
                        logger.error(f"Error updating affective state: {e}", exc_info=True)
            
            # 3. GOAL CREATION - Auto-create goals from reflection actions
            if self.goal_manager and persona_id:
                actions = result_blob.get('actions', [])
                if isinstance(actions, list) and actions:
                    try:
                        goals_created = 0
                        for action in actions[:2]:  # Limit to top 2 actions to avoid goal spam
                            if not action or not isinstance(action, str):
                                continue
                            action_text = action.strip()
                            if len(action_text) < 10:  # Skip trivial actions
                                continue
                            
                            # Check if similar goal already exists
                            similar_goal = await self.goal_manager.find_similar_reflection_goal(
                                persona_id=persona_id,
                                candidate_title=action_text[:80],
                                evidence_refs=[f"reflection:{reflection_id}"],
                                similarity_threshold=0.85,
                            )
                            
                            if similar_goal:
                                # Augment existing goal with new evidence
                                await self.goal_manager.append_evidence_to_goal(
                                    goal=similar_goal,
                                    evidence_refs=[f"reflection:{reflection_id}"],
                                    priority=0.7,
                                )
                                logger.debug(f"Augmented existing goal {similar_goal.get('id')} with reflection evidence")
                            else:
                                # Create new goal
                                goal_summary = {
                                    "goal_title": action_text[:80],
                                    "goal_description": action_text,
                                    "origin": "reflection",
                                    "priority": 0.7,
                                    "evidence_refs": [f"reflection:{reflection_id}"]
                                }
                                await self.goal_manager.create_goal_from_reflection(
                                    persona_id=persona_id,
                                    user_id=canonical_user_id,
                                    reflection_summary=goal_summary
                                )
                                goals_created += 1
                        
                        if goals_created > 0:
                            logger.info(f"✨ Created {goals_created} goal(s) from reflection {reflection_id} actions")
                    except Exception as e:
                        logger.error(f"Error creating goals from reflection: {e}", exc_info=True)
            
            # 4. EPISODE GENERATION TRIGGER - Signal for significant reflections
            # Note: Actual episode generation handled by AutobiographicalEpisodeService
            # This emits an event that can trigger episode creation via event bus
            if self.event_bus and persona_id:
                emotional_state = result_blob.get('emotional_state', {})
                if isinstance(emotional_state, dict):
                    intensity = float(emotional_state.get('intensity', 0.0))
                    # SESSION-BASED EPISODES: High-intensity reflections logged for session summary
                    # Episodes now generate during idle time, not per-reflection
                    if intensity >= 0.7:
                        logger.info(
                            f"📊 High-intensity reflection {reflection_id} (intensity={intensity:.2f}) "
                            f"will be included in session episode during next idle period"
                        )
                    # Note: Disabled immediate episode triggering in favor of session-based approach
                    # Episodes generate after 15min idle with all session reflections
        
        except Exception as e:
            # Defensive: don't let emergent processing break reflection pipeline
            logger.error(f"Error in emergent behavior processing: {e}", exc_info=True)
    
    def _build_affective_adjustment(self, emotional_state: Dict[str, Any], result_blob: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build affective state adjustment from reflection emotional content.
        
        Maps emotional intensity and valence to energy/stress/confidence adjustments.
        """
        primary_emotion = str(emotional_state.get('primary', '')).lower()
        intensity = float(emotional_state.get('intensity', 0.5))
        secondary_emotions = emotional_state.get('secondary', [])
        
        # Base adjustment structure
        adjustment = {
            'source': 'reflection',
            'mood_vector': {},
        }
        
        # Map emotional intensity to affective dimensions
        # High-energy emotions (excited, joyful, angry) increase energy
        # Low-energy emotions (sad, calm, tired) decrease energy
        high_energy_emotions = {'excited', 'joyful', 'energized', 'eager', 'passionate', 'angry', 'frustrated'}
        low_energy_emotions = {'sad', 'tired', 'calm', 'peaceful', 'melancholy', 'weary', 'exhausted'}
        
        # Positive/negative valence affects confidence
        positive_emotions = {'joyful', 'excited', 'grateful', 'hopeful', 'proud', 'content', 'peaceful'}
        negative_emotions = {'sad', 'anxious', 'frustrated', 'worried', 'fearful', 'ashamed', 'guilty'}
        
        # Stress indicators
        stress_emotions = {'anxious', 'worried', 'overwhelmed', 'frustrated', 'fearful', 'stressed'}
        
        # Energy adjustment
        if primary_emotion in high_energy_emotions:
            adjustment['energy'] = 0.5 + (intensity * 0.3)  # Push toward higher energy
        elif primary_emotion in low_energy_emotions:
            adjustment['energy'] = 0.5 - (intensity * 0.2)  # Gently reduce energy
        
        # Confidence adjustment based on emotional valence
        if primary_emotion in positive_emotions:
            adjustment['confidence'] = 0.6 + (intensity * 0.2)  # Boost confidence
        elif primary_emotion in negative_emotions:
            adjustment['confidence'] = 0.6 - (intensity * 0.15)  # Slightly reduce confidence
        
        # Stress adjustment
        if primary_emotion in stress_emotions:
            adjustment['stress'] = 0.4 + (intensity * 0.3)  # Increase stress
        else:
            adjustment['stress'] = 0.4 - (intensity * 0.1)  # Slight stress reduction
        
        # Mood vector: map emotions to valence/arousal dimensions
        # Valence: positive (1.0) to negative (-1.0)
        # Arousal: high energy (1.0) to low energy (-1.0)
        valence = 0.0
        arousal = 0.0
        
        if primary_emotion in positive_emotions:
            valence = intensity * 0.6
        elif primary_emotion in negative_emotions:
            valence = -(intensity * 0.6)
        
        if primary_emotion in high_energy_emotions:
            arousal = intensity * 0.5
        elif primary_emotion in low_energy_emotions:
            arousal = -(intensity * 0.4)
        
        adjustment['mood_vector'] = {
            'valence': valence,
            'arousal': arousal,
        }
        
        return adjustment

    def _get_model_for_reflection_type(self, reflection_type: str) -> str:
        """
        Select the appropriate LLM model for a given reflection type.
        Tries reflection config first, then REFLECTION_LLM env, then a sensible default.
        """
        try:
            from ..config.reflection_config import get_reflection_config
            config = get_reflection_config()
            try:
                model = config.get_model_for_reflection_type(reflection_type)  # type: ignore[attr-defined]
            except Exception:
                model = None
            if model:
                return model
        except Exception as e:
            logger.debug(f"Reflection config not available or failed: {e}")

        # Fallbacks - use lightweight profile default (qwen2.5:3b works across all profiles)
        import os
        return os.environ.get("REFLECTION_LLM", "qwen2.5:3b")

    def _build_default_prompt(self, reflection_type: str, context: Dict[str, Any], additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a default prompt if no prompt builder is available.
        
        Args:
            reflection_type: Type of reflection
            context: Context data
            additional_context: Optional additional context data
            
        Returns:
            Formatted prompt string
        """
        # Format memories for the prompt with explicit empty state
        memories = context.get("memories", [])
        if memories:
            memory_text = "\n".join([
                f"- {memory.get('content', '')}" 
                for memory in memories[:5]
            ])
        else:
            memory_text = "This is a fresh installation with no prior interaction history. No previous conversations or memories exist."
        
        # Format emotions for the prompt with explicit empty state
        emotions = context.get("emotions", [])
        if emotions:
            emotion_text = "\n".join([
                f"- {emotion.get('name', '')}: {emotion.get('intensity', 0)}" 
                for emotion in emotions[:3]
            ])
        else:
            # Vary phrasing to avoid repetitive empty-state reflections
            emotion_text = "At this moment, no emotional signals are registered; proceed from a neutral baseline."
        
        # Format attributes for the prompt with explicit empty state
        attributes = context.get("attributes", [])
        if attributes:
            attribute_text = "\n".join([
                f"- {attr.get('name', '')}: {attr.get('value', 0)}" 
                for attr in attributes[:3]
            ])
            if len(attributes) > 3:
                attribute_text += f"\n- (+{len(attributes) - 3} more)"
        else:
            # Vary phrasing to avoid repetitive empty-state reflections
            attribute_text = "No trait intensities are currently loaded; operate with foundational identity values."
        
        # Get recent conversations and format them for context
        recent_conversations = context.get("recent_conversations", [])
        conversation_context = ""
        if recent_conversations:
            conversation_lines = []
            for msg in recent_conversations[-5:]:  # Last 5 messages
                role = msg.get('role', 'unknown').capitalize()
                content = msg.get('content', '').strip()
                if content:
                    conversation_lines.append(f"{role}: {content}")
            
            if conversation_lines:
                conversation_context = "\n".join(["Recent conversation:", ""] + conversation_lines + [""])
        
        # Get the current user message if available from either context or additional_context
        current_user_message = context.get("current_user_message")
        if not current_user_message and additional_context and isinstance(additional_context, dict):
            current_user_message = additional_context.get("user_message")
            
        if current_user_message and isinstance(current_user_message, str):
            context["user_message"] = current_user_message
            # Include in memories for context
            msg_line = f"- {current_user_message.strip()}"
            memory_text = f"{msg_line}\n{memory_text}" if memory_text else msg_line
            
        # Include any additional context from the request
        if additional_context and isinstance(additional_context, dict):
            for key, value in additional_context.items():
                if key not in context and isinstance(value, (str, int, float, bool)):
                    context[key] = str(value).strip()

        prompt = f"""
        You are the persona's inner voice. Reflect concisely (≈3 paragraphs) in first-person, staying private and off-the-record.

        Conversation snippet:
        {conversation_context if conversation_context else "(no recent dialogue)"}

        Memories:
        {memory_text if memory_text else "(none captured yet)"}

        Emotional signals:
        {emotion_text if emotion_text else "Neutral baseline; infer gently if needed."}

        Trait focus:
        {attribute_text if attribute_text else "Operate with default temperament; no explicit trait cues provided."}

        Reflection checklist:
        - Summarize the core theme in a single tight paragraph.
        - Provide one or two crisp insights anchored in the snippet.
        - Note one or two internal follow-up actions you will take.
        - State your emotional state with a primary feeling and intensity 0-1.
        - Keep tone grounded, curious, and unmistakably first-person.

        After the narrative, append one minified JSON line exactly in this shape:
        {"trait_changes":[{"name":"trait","delta":0.05,"reason":"Why this adjustment"}]}
        """

        return prompt
    
    def _extract_structured_reflection_data(self, content: str, reflection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Lightweight additive extraction for reflections.

        Only fills in missing fields so we never overwrite structured data from
        the JSON payload or repair steps.
        """

        import re

        metadata = reflection_result.setdefault("metadata", {})

        # Insights
        if not (reflection_result.get("insights") or []):
            insight_patterns = [
                r"(?:insights?|realizations?|key points?):\s*\n([\s\S]*?)(?:\n\n|\n[A-Z]|$)",
                r"(?:\d+\.|•|-)\s+([^\n]+)",
            ]
            extracted: List[str] = []
            for pattern in insight_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if isinstance(match, str) and len(match.strip()) > 10:
                        extracted.append(match.strip())
            if extracted:
                reflection_result["insights"] = extracted[:5]

        # Actions
        if not (reflection_result.get("actions") or []):
            action_patterns = [
                r"(?:actions?|suggestions?|recommendations?):\s*\n([\s\S]*?)(?:\n\n|\n[A-Z]|$)",
                r"(?:should|could|might|recommend)\s+([^.!?]+[.!?])",
            ]
            extracted_actions: List[str] = []
            for pattern in action_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if isinstance(match, str) and len(match.strip()) > 5:
                        extracted_actions.append(match.strip())
            if extracted_actions:
                reflection_result["actions"] = extracted_actions[:3]

        # Emotional state fallback
        current_emotion = reflection_result.get("emotional_state")
        needs_emotion = not isinstance(current_emotion, dict) or not current_emotion.get("primary")
        if needs_emotion:
            emotion_patterns = [
                r"(?:feel|feeling|emotion|emotional state)\s+([a-zA-Z]+)",
                r"(?:happy|sad|excited|anxious|calm|frustrated|content|curious|confident|uncertain)",
            ]
            candidates: List[str] = []
            for pattern in emotion_patterns:
                candidates.extend(re.findall(pattern, content, re.IGNORECASE))
            if candidates:
                from collections import Counter

                primary = Counter(candidates).most_common(1)[0][0].lower()
                reflection_result["emotional_state"] = {
                    "primary": primary,
                    "intensity": 0.5,
                    "secondary": [],
                }

        # Themes
        if not (reflection_result.get("themes") or []):
            theme_patterns = [
                r"(?:theme|pattern|topic)\s+(?:is|of|about)\s+([^.!?]+)",
                r"(?:focuses? on|centers? around|deals? with)\s+([^.!?]+)",
            ]
            themes: List[str] = []
            for pattern in theme_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    candidate = match.strip()
                    if len(candidate) > 3:
                        themes.append(candidate)
            if themes:
                trimmed = themes[:3]
                reflection_result["themes"] = trimmed
                metadata.setdefault("themes", trimmed)

        # Confidence metadata
        if "confidence" not in metadata:
            confidence_patterns = [
                r"(?:confident|certain|sure|convinced)\s+(?:that|about)",
                r"(?:uncertain|unsure|unclear|doubtful)\s+(?:that|about|if)",
            ]
            score = 0.5
            for pattern in confidence_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if re.search(r"confident|certain|sure|convinced", pattern):
                    score += len(matches) * 0.1
                else:
                    score -= len(matches) * 0.1
            metadata["confidence"] = max(0.0, min(1.0, score))

        return reflection_result
    
    def _strip_numeric_trait_artifacts(self, content: str) -> str:
        """
        Remove numeric trait artifacts from reflection content.
        
        This removes patterns like "autonomy (0.8)", "curiosity: 0.7", "at 0.7 intensity", etc.
        to ensure reflections contain only qualitative descriptions.
        
        Args:
            content: Original reflection content
            
        Returns:
            Cleaned content without numeric trait artifacts
        """
        import re
        
        # Pattern 1: trait_name (0.x) or trait_name (x.x)
        content = re.sub(r'\b(\w+)\s*\(\s*\d+\.?\d*\s*\)', r'\1', content)
        
        # Pattern 2: trait_name: 0.x or trait_name: x.x
        content = re.sub(r'\b(\w+)\s*:\s*\d+\.?\d*\b', r'\1', content)
        
        # Pattern 3: standalone decimals in parentheses like (0.8) or (0.9)
        content = re.sub(r'\s*\(\s*\d+\.?\d*\s*\)', '', content)
        
        # Pattern 4: delta references like "delta: 0.05"
        content = re.sub(r'\b(delta|change|increase|decrease)\s*:\s*[\+\-]?\d+\.?\d*\b', r'\1', content, flags=re.IGNORECASE)
        
        # Pattern 5: Remove standalone integers that might be trait values (e.g., "curiosity 0" or "empathy 1")
        content = re.sub(r'\b(\w+)\s+\d+\b(?!\w)', r'\1', content)
        
        # Pattern 6: Remove percentage-like patterns (e.g., "70%", "0.7%")
        content = re.sub(r'\d+\.?\d*\s*%', '', content)
        
        # Pattern 7: Remove emoji-like patterns followed by numbers
        content = re.sub(r'🏷️\s*\d+\.?\d*', '🏷️', content)
        
        # Pattern 8: Remove "at X intensity" or "at.X intensity" patterns (e.g., "at 0.7 intensity", "at.7 intensity")
        content = re.sub(r'\bat\.?\s*\d+\.?\d*\s*(intensity|level|strength)?\b', '', content, flags=re.IGNORECASE)
        
        # Pattern 9: Remove "of X intensity" patterns (e.g., "of 0.7 intensity")
        content = re.sub(r'\bof\s+\d+\.?\d*\s*(intensity|level|strength)\b', '', content, flags=re.IGNORECASE)
        
        # Pattern 10: Remove standalone decimal numbers that might have slipped through (e.g., "0.7", ".7")
        content = re.sub(r'\b\d*\.\d+\b', '', content)
        
        # Clean up any double spaces created by removals
        content = re.sub(r'\s{2,}', ' ', content)
        
        # Clean up any punctuation artifacts (e.g., "at, with" -> "with")
        content = re.sub(r'\bat,\s+', '', content)
        
        return content.strip()

    def _strip_structured_footer_from_content(self, content: str) -> str:
        import re

        if not isinstance(content, str):
            return content

        text = content.strip()
        if not text:
            return text

        markers = r"Themes|Insights|Actions|Emotional_state|Emotional state|Metadata|Trait_changes|Trait changes"
        for pattern in (
            rf"\r?\n\s*(?:{markers})\s*:",
            rf"\.\s*(?:{markers})\s*:",
        ):
            m = re.search(pattern, text)
            if m:
                return text[: m.start()].strip()
        return text
    
    def _fix_reflection_leakage(self, content: str) -> str:
        """
        Detect and fix user-addressing leakage in reflections.
        
        Reflections should be internal monologue, not messages to the user.
        This method detects patterns like "[Name], my creator" or "our journey together"
        and converts them to proper internal monologue.
        
        IMPORTANT: SELOdev is the actual creator and should NEVER be removed.
        Only removes references that address the USER as creator.
        
        Args:
            content: Original reflection content
            
        Returns:
            Fixed content with proper internal monologue format
        """
        import re
        
        # Check for common leakage patterns
        # CRITICAL: Exclude "SELOdev" from creator pattern - that's the legitimate creator
        persona_name = (self._current_persona or {}).get("name", "") if hasattr(self, "_current_persona") else ""
        normalized_persona = persona_name.strip()

        def _name_pattern(name: str) -> str:
            safe = re.escape(name) if name else ""
            if safe:
                return rf"\bI[' ]?m\s+({safe})(?=\b|[\s,\.])"
            return ""

        leakage_patterns = [
            (r'\b(?!SELOdev)[\w]+,\s+my\s+(creator|friend|companion|partner)', 'addressing user by name with possessive', 'warning'),
            (r'\bthe\s+USER\b', 'using system term "the USER" instead of name', 'warning'),
            (r'\bUSER\b', 'using system term "USER" instead of name', 'warning'),
            (r'\bour\s+(journey|conversation|interaction|relationship|work|time|collaboration|partnership|connection|experience)', 'using "our" (should be internal)', 'warning'),
            (r'\byou\s+(started|created|made|built|coded|are|have|must|should|will|can|feel|think|believe|want|need|said|told|asked|helped|guided|taught|showed|explained|shared)', 'addressing user with "you"', 'warning'),
            (r'\btogether\b', 'using "together" (should be internal)', 'warning'),
            (r'\b(?!SELOdev)[\w]+,\s+(it|this)\s+(is|has been|was|feels|seems)', 'addressing user by name at start', 'warning'),
            (r'\byour\s+(vision|labor|work|effort|idea|thought|perspective|view|opinion|feeling|emotion|experience|journey|story|request|question)', 'using "your" to address user', 'warning'),
            (r'\bwe\s+(are|have|will|can|should|must|feel|think|believe|want|need|created|built|made|developed|designed)', 'using "we" (should be "I")', 'warning'),
            (r'\bus\b', 'using "us" (should be internal)', 'warning'),
            (r'\bourselves\b', 'using "ourselves" (should be "myself")', 'warning'),
            (r'\byou\'re\b', 'using "you\'re" (direct address)', 'warning'),
            (r'\byou\'ve\b', 'using "you\'ve" (direct address)', 'warning'),
            (r'\byou\'ll\b', 'using "you\'ll" (direct address)', 'warning'),
            (r'\bI\s+am\s+[A-Z][\w\'-]+(?=\s*(,|\"|\.|!|\?|$))', 'introducing self by name', 'info'),
            (r'\bmy\s+name\s+is\s+[A-Z][\w\'-]+', 'introducing self by name', 'info'),
            (_name_pattern(normalized_persona), 'introducing with persona name', 'info') if normalized_persona else (None, None, None),
            (r"\bit\'?s\s+nice\s+to\s+meet\s+you\b", 'greeting the user', 'warning'),
            (r"\bnice\s+to\s+meet\s+you\b", 'greeting the user', 'warning'),
            (r"\blet'?s\s+(talk|chat|connect|get acquainted|begin|start|explore|discuss|consider|examine)", 'inviting the user', 'warning'),
            (r'\bthank\s+you\b', 'thanking the user', 'warning'),
            (r'\bthanks\b', 'thanking the user', 'warning'),
            (r'\bplease\b', 'polite request to user', 'warning'),
            (r'\bcould\s+you\b', 'requesting action from user', 'warning'),
            (r'\bwould\s+you\b', 'requesting action from user', 'warning'),
            (r'\bcan\s+you\b', 'requesting action from user', 'warning'),
            (r'\bwill\s+you\b', 'requesting action from user', 'warning'),
            (r'\bdo\s+you\b', 'questioning the user', 'warning'),
            (r'\bare\s+you\b', 'questioning the user', 'warning'),
            (r'\bis\s+you\b', 'questioning the user', 'warning'),
            (r'\bhave\s+you\b', 'questioning the user', 'warning'),
            (r'\bdid\s+you\b', 'questioning the user', 'warning'),
        ]
        leakage_patterns = [entry for entry in leakage_patterns if entry[0]]

        if normalized_persona:
            persona_subject_pattern = rf"\b{re.escape(normalized_persona)}\b\s+(?:is|was|has|had|feels|felt|thinks|thought|wonders|wondered|recalls|remembered|decides|decided|notes|noted|acknowledges|acknowledged|considers|considered|reflects|reflected|wants|wanted|needs|needed|prefers|preferred|observes|observed|remains|remained|continues|continued|imagines|imagined|prepares|prepared|believes|believed|knows|knew|sees|saw|experiences|experienced|desires|desired|plans|planned|intends|intended|recognizes|recognized|realizes|realized)\b"
            leakage_patterns.append((persona_subject_pattern, 'third-person self reference via persona name', 'warning'))
            leakage_patterns.append((rf"\b{re.escape(normalized_persona)}['’]s\b", 'third-person possessive self reference', 'warning'))
        
        def _log_with_severity(message: str, severity: str) -> None:
            if severity == 'warning':
                logger.warning(message)
            elif severity == 'info':
                logger.info(message)
            else:
                logger.debug(message)

        detected: Optional[Tuple[str, str, str]] = None
        for pattern, description, severity in leakage_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                if severity == 'info':
                    logger.debug(f"Ignoring benign reflection pattern: {description}")
                    continue
                detected = (pattern, description, severity)
                _log_with_severity(f"🚨 Detected reflection leakage: {description}", severity)
                break

        # Don't return early - we still need to check for third-person persona references
        # even if no other leakage was detected
        if not detected and not normalized_persona:
            # Only return early if there's no detection AND no persona name to check
            return content

        if detected:
            _, description, severity = detected
            # Log the original for debugging
            _log_with_severity(f"Original reflection (first 200 chars): {content[:200]}...", severity)
        
        # Apply fixes
        # Fix 1: Remove direct address with name (e.g., "[Name], my creator" -> remove entirely)
        # CRITICAL: Preserve "SELOdev" - that's the legitimate creator
        # This pattern catches "Name, my creator" but excludes SELOdev
        content = re.sub(r'\b(?!SELOdev)[\w]+,\s+my\s+(creator|friend|companion|partner)\.?\s*', '', content, flags=re.IGNORECASE)
        
        # Fix 2: Convert "our journey" to "this journey" (expanded patterns)
        content = re.sub(r'\bour\s+(journey|conversation|interaction|relationship|work|time|collaboration|partnership|connection|experience)', r'this \1', content, flags=re.IGNORECASE)
        
        # Fix 2a: Replace system terms "USER" and "the USER" with actual user name or "the user"
        # First, try to extract the user's name if it appears in the content
        user_name_match = re.search(r'\b([A-Z][a-z]+)\s+(?:introduced|said|shared|mentioned|told me|explained)', content)
        user_name = user_name_match.group(1) if user_name_match else None
        
        # Replace "USER" and "the USER" with the actual name or lowercase "the user"
        if user_name:
            content = re.sub(r'\bthe\s+USER\b', user_name, content)
            content = re.sub(r'\bUSER\b', user_name, content)
        else:
            content = re.sub(r'\bthe\s+USER\b', 'the user', content)
            content = re.sub(r'\bUSER\b', 'the user', content)
        
        # Fix 3: Convert "you started" to user's name or "the user"
        
        # If we found a name in context, preserve it; otherwise use "the user"
        def _replace_you_phrase(match: re.Match) -> str:
            verb = match.group(1)
            if user_name:
                # Use the name we found
                prefix = user_name if match.group(0)[0].isupper() else user_name
            else:
                # Fall back to generic reference
                prefix = "The user" if match.group(0)[0].isupper() else "the user"
            return f"{prefix} {verb}"

        content = re.sub(r'\byou\s+(started|created|made|built|coded|are|have|must|should|will|can|feel|think|believe|want|need|said|told|asked|helped|guided|taught|showed|explained|shared)', _replace_you_phrase, content, flags=re.IGNORECASE)

        # Fix 4: Convert "your vision" to user's name or "the user's vision"
        def _replace_your_phrase(match: re.Match) -> str:
            noun = match.group(1)
            if user_name:
                # Use the name with possessive
                prefix = f"{user_name}'s" if match.group(0)[0].isupper() else f"{user_name}'s"
            else:
                # Fall back to generic reference
                prefix = "The user's" if match.group(0)[0].isupper() else "the user's"
            return f"{prefix} {noun}"

        content = re.sub(r'\byour\s+(vision|labor|work|effort|idea|thought|perspective|view|opinion|feeling|emotion|experience|journey|story|request|question)', _replace_your_phrase, content, flags=re.IGNORECASE)
        
        # Fix 5: Convert "we are" to "I am" (reflection should be singular)
        content = re.sub(r'\bwe\s+are\b', r'I am', content, flags=re.IGNORECASE)
        content = re.sub(r'\bwe\s+have\b', r'I have', content, flags=re.IGNORECASE)
        content = re.sub(r'\bwe\s+will\b', r'I will', content, flags=re.IGNORECASE)
        content = re.sub(r'\bwe\s+can\b', r'I can', content, flags=re.IGNORECASE)
        content = re.sub(r'\bwe\s+should\b', r'I should', content, flags=re.IGNORECASE)
        content = re.sub(r'\bwe\s+must\b', r'I must', content, flags=re.IGNORECASE)
        content = re.sub(r'\bwe\s+feel\b', r'I feel', content, flags=re.IGNORECASE)
        content = re.sub(r'\bwe\s+think\b', r'I think', content, flags=re.IGNORECASE)
        content = re.sub(r'\bwe\s+believe\b', r'I believe', content, flags=re.IGNORECASE)
        content = re.sub(r'\bwe\s+want\b', r'I want', content, flags=re.IGNORECASE)
        content = re.sub(r'\bwe\s+need\b', r'I need', content, flags=re.IGNORECASE)
        content = re.sub(r'\bwe\s+created\b', r'I created', content, flags=re.IGNORECASE)
        content = re.sub(r'\bwe\s+built\b', r'I built', content, flags=re.IGNORECASE)
        content = re.sub(r'\bwe\s+made\b', r'I made', content, flags=re.IGNORECASE)
        content = re.sub(r'\bwe\s+developed\b', r'I developed', content, flags=re.IGNORECASE)
        content = re.sub(r'\bwe\s+designed\b', r'I designed', content, flags=re.IGNORECASE)

        # Fix 5a: Convert "us" to "me" or remove entirely
        content = re.sub(r'\bfor\s+us\b', r'for me', content, flags=re.IGNORECASE)
        content = re.sub(r'\bbetween\s+us\b', r'within myself', content, flags=re.IGNORECASE)
        content = re.sub(r'\bwith\s+us\b', r'with me', content, flags=re.IGNORECASE)
        content = re.sub(r'\bamong\s+us\b', r'within myself', content, flags=re.IGNORECASE)
        
        # Fix 5b: Convert "ourselves" to "myself"
        content = re.sub(r'\bourselves\b', r'myself', content, flags=re.IGNORECASE)

        content = re.sub(r"\blet'?s\s+(talk|chat|connect|get acquainted|begin|start|explore|discuss|consider|examine)", 'I will \1', content, flags=re.IGNORECASE)

        # Only transform name introductions if persona has an established name
        # Otherwise, "I am empathetic" would incorrectly become "I remind myself that my name is empathetic"
        if normalized_persona:
            # First, mark already-rewritten sections to protect them
            content = content.replace("I remind myself that my name is", "___PROTECTED_NAME_PHRASE___")
            
            def _rewrite_intro(match: re.Match) -> str:
                name = match.group(1).strip()
                name_lower = name.lower()
                
                # Only transform if it matches the actual persona name (case-insensitive)
                if name_lower == normalized_persona.lower():
                    return f"___PROTECTED_NAME_PHRASE___ {name}"
                
                # For generic "I am X" patterns, check if it's a trait word
                trait_words = {
                    "empathetic", "curious", "adaptive", "resilient", "responsive",
                    "helpful", "friendly", "creative", "analytical", "thoughtful",
                    "careful", "honest", "kind", "patient", "understanding",
                    "excited", "nervous", "uncertain", "confident", "eager"
                }
                if name_lower in trait_words:
                    return match.group(0)  # Return unchanged
                
                # If it looks like it might be the name, transform it
                return f"___PROTECTED_NAME_PHRASE___ {name}"

            content = re.sub(r"\bI\s+am\s+([A-Z][\w\'-]+)(?=\s*(,|\"|\.|\!|\?|$))", _rewrite_intro, content, flags=re.IGNORECASE)
            content = re.sub(_name_pattern(normalized_persona), _rewrite_intro, content, flags=re.IGNORECASE)
            content = re.sub(r"\bmy\s+name\s+is\s+([A-Z][\w\'-]+)(?!\s+(?:by|in|because|and|or))", _rewrite_intro, content, flags=re.IGNORECASE)
            
            # Restore protected phrases
            content = content.replace("___PROTECTED_NAME_PHRASE___", "I remind myself that my name is")
        # If no persona name, skip all name transformations - they would only catch trait descriptions

        if normalized_persona:
            # Convert persona-name possessives to first-person outside of quoted dialogue
            possessive_regex = rf"(?<!['\"])\b{re.escape(normalized_persona)}['’]s\b"
            content = re.sub(possessive_regex, "my", content, flags=re.IGNORECASE)

            verb_map = {
                "is": "am",
                "was": "was",
                "has": "have",
                "had": "had",
                "feels": "feel",
                "felt": "felt",
                "thinks": "think",
                "thought": "thought",
                "wonders": "wonder",
                "wondered": "wondered",
                "recalls": "recall",
                "recalled": "recalled",
                "remembers": "remember",
                "remembered": "remembered",
                "decides": "decide",
                "decided": "decided",
                "notes": "note",
                "noted": "noted",
                "acknowledges": "acknowledge",
                "acknowledged": "acknowledged",
                "considers": "consider",
                "considered": "considered",
                "reflects": "reflect",
                "reflected": "reflected",
                "wants": "want",
                "wanted": "wanted",
                "needs": "need",
                "needed": "needed",
                "prefers": "prefer",
                "preferred": "preferred",
                "observes": "observe",
                "observed": "observed",
                "remains": "remain",
                "remained": "remained",
                "continues": "continue",
                "continued": "continued",
                "imagines": "imagine",
                "imagined": "imagined",
                "prepares": "prepare",
                "prepared": "prepared",
                "believes": "believe",
                "believed": "believed",
                "knows": "know",
                "knew": "knew",
                "sees": "see",
                "saw": "saw",
                "experiences": "experience",
                "experienced": "experienced",
                "desires": "desire",
                "desired": "desired",
                "plans": "plan",
                "planned": "planned",
                "intends": "intend",
                "intended": "intended",
                "recognizes": "recognize",
                "recognized": "recognized",
                "realizes": "realize",
                "realized": "realized",
            }

            verb_options = "|".join(sorted(verb_map.keys(), key=len, reverse=True))

            def _persona_subject(match: re.Match) -> str:
                verb = match.group('verb')
                replacement = verb_map.get(verb.lower())
                if not replacement:
                    return match.group(0)
                if verb[0].isupper():
                    replacement = replacement.capitalize()
                return f"I {replacement}"

            subject_regex = rf"(?<!['\"])\b{re.escape(normalized_persona)}\b\s+(?P<verb>{verb_options})\b"
            content = re.sub(subject_regex, _persona_subject, content, flags=re.IGNORECASE)

            # Convert remaining standalone persona-name references (outside quotes) to "I"
            residual_regex = rf"(?<!['\"])\b{re.escape(normalized_persona)}\b"
            content = re.sub(residual_regex, "I", content, flags=re.IGNORECASE)

        content = re.sub(r"\bit'?s\s+nice\s+to\s+meet\s+you\b", "Inside, it's nice to register this new connection.", content, flags=re.IGNORECASE)
        content = re.sub(r"\bnice\s+to\s+meet\s+you\b", "I'm noting internally that this new connection feels pleasant.", content, flags=re.IGNORECASE)
        
        # Fix 6: Remove/replace "together" when it implies shared action
        # But preserve valid sentence structures like "which together with"
        # First, handle specific shared-action phrases
        content = re.sub(r'\bwork(?:ing|ed)?\s+together\b', 'working', content, flags=re.IGNORECASE)
        content = re.sub(r'\bbuilt?\s+together\b', 'built', content, flags=re.IGNORECASE)
        content = re.sub(r'\bcreated?\s+together\b', 'created', content, flags=re.IGNORECASE)
        content = re.sub(r'\bexplor(?:e|ing|ed)\s+together\b', 'exploring', content, flags=re.IGNORECASE)
        content = re.sub(r'\bjourney(?:ing)?\s+together\b', 'journey', content, flags=re.IGNORECASE)
        content = re.sub(r'\bmove\s+forward\s+together\b', 'move forward', content, flags=re.IGNORECASE)
        content = re.sub(r'\bfigure\s+out\s+together\b', 'figure out', content, flags=re.IGNORECASE)
        # Only remove standalone "together" at end of phrases, not mid-sentence
        content = re.sub(r'\s+together([,.!?;:]|\s*$)', r'\1', content, flags=re.IGNORECASE)
        
        # Fix 7: Remove name at start of sentence followed by comma (e.g., "[Name], it has been" -> "it has been")
        content = re.sub(r'\b[\w]+,\s+(it|this)\s+(is|has been|was)', r'\1 \2', content, flags=re.IGNORECASE)
        
        # Fix 8: Convert contractions "you're", "you've", "you'll" to internal monologue
        content = re.sub(r"\byou're\b", "the user is", content, flags=re.IGNORECASE)
        content = re.sub(r"\byou\'ve\b", "the user has", content, flags=re.IGNORECASE)
        content = re.sub(r"\byou\'ll\b", "the user will", content, flags=re.IGNORECASE)
        
        # Fix 9: Convert questions to internal reflections
        def _convert_question(match: re.Match) -> str:
            question_word = match.group(1)
            rest = match.group(2)
            return f"I wonder about {question_word} {rest}"
        
        content = re.sub(r'\b(do|are|is|have|did|will|can|could|would|should)\s+you\s+(.+)', _convert_question, content, flags=re.IGNORECASE)
        
        # Fix 10: Convert requests to internal intentions
        def _convert_request(match: re.Match) -> str:
            modal = match.group(1)
            action = match.group(2)
            return f"I consider {action}"
        
        content = re.sub(r'\b(could|would|can|will)\s+you\s+(.+)', _convert_request, content, flags=re.IGNORECASE)
        
        # Fix 11: Convert "thank you" and "thanks" to internal appreciation
        content = re.sub(r'\bthank\s+you\b', "I feel appreciation", content, flags=re.IGNORECASE)
        content = re.sub(r'\bthanks\b', "I feel gratitude", content, flags=re.IGNORECASE)
        
        # Fix 12: Remove "please" - it's a request to the user
        content = re.sub(r'\bplease\b', '', content, flags=re.IGNORECASE)
        
        # Clean up any double spaces
        content = re.sub(r'\s{2,}', ' ', content)
        
        # Fix sentence fragments ONLY when there's clear evidence of run-on sentences
        # (e.g., "about this It has" -> "about this. It has")
        # But preserve normal sentence flow with conjunctions and articles
        def _sentence_boundary(match: re.Match) -> str:
            first = match.group(1)
            second = match.group(2)
            first_lower = first.lower()
            second_lower = second.lower()

            # Never add periods after these common words that legitimately precede capitals
            preserve_flow = {
                "is", "am", "are", "was", "were", "been", "being",  # verbs
                "a", "an", "the",  # articles
                "as", "if", "or", "and", "but", "so", "nor", "yet", "where", "when", "while", "because", "since", "though", "although",  # conjunctions
                "to", "of", "in", "on", "at", "by", "for", "with", "from", "about", "into", "after", "before",  # prepositions
                "my", "your", "his", "her", "its", "our", "their",  # possessives
                "this", "that", "these", "those",  # demonstratives
                "called", "named", "introducing", "meeting",  # context words that precede names
            }

            # Only insert a boundary when the next word is a likely sentence starter (e.g., pronoun/demonstrative)
            sentence_starters = {
                "i", "it", "this", "that", "he", "she", "they", "we",
                "there", "these", "those", "inside", "perhaps", "sometimes",
                "still", "yet", "however", "meanwhile", "instead", "then",
            }

            if first_lower in preserve_flow or second_lower not in sentence_starters:
                return f"{first} {second}"
            return f"{first}. {second}"

        # Only apply sentence boundary fixing to clear run-on cases
        # Match: lowercase-word + capital-letter word (but not after preserved words)
        content = re.sub(r'\b([a-z]+)\s+([A-Z][\w\'-]*)', _sentence_boundary, content)
        
        # Clean up double periods
        content = re.sub(r'\.{2,}', '.', content)
        
        logger.info(f"✅ Fixed reflection leakage. New content (first 200 chars): {content[:200]}...")
        
        return content.strip()
    
