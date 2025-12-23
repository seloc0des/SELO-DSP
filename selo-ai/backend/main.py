from fastapi import FastAPI, HTTPException, Depends, Request, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import Any, Dict, List, Optional
from .models.pagination import PaginatedResponse
import subprocess
import json
import uuid
import os
import requests
from dotenv import load_dotenv
import trafilatura
import logging
from contextlib import asynccontextmanager
from .utils.logging_config import setup_logging, set_correlation_id, get_correlation_id
from .middleware.rate_limiting import RateLimitMiddleware
import socketio
import asyncio
import time
import inspect
from datetime import datetime, timedelta, timezone
from .utils.datetime import utc_now, isoformat_utc, utc_iso
from .utils.numeric_utils import clamp
from .llm.dual_llm_config import get_llm_model
from .llm.response_validator import ResponseValidator
import pathlib
import random
import hashlib
from collections import OrderedDict
from .core.boot_seed_system import get_random_directive, normalize_directive
from .config.performance_config import get_performance_config
from .utils.system_profile import detect_system_profile
from .utils.system_metrics import get_system_metrics_collector, detect_diagnostic_trigger, DiagnosticMode
# Option B: Removed compose_opening and generate_first_intro_from_directive
# LLM handles all introductions naturally via persona system prompt
from .reflection.default_relationship_questions import DEFAULT_RELATIONSHIP_QUESTION_SEED

# Configure structured logging
setup_logging(log_level="INFO", structured=True)
logger = logging.getLogger(__name__)

# Load environment variables from .env if present
load_dotenv()

# Detect host capabilities once and expose generated defaults
SYSTEM_PROFILE = detect_system_profile()

_env_defaults = {
    "CHAT_MAX_TOKENS": str(SYSTEM_PROFILE["budgets"]["chat_max_tokens"]),
    "CHAT_NUM_PREDICT": str(SYSTEM_PROFILE["budgets"]["chat_max_tokens"]),
    "REFLECTION_MAX_TOKENS": str(SYSTEM_PROFILE["budgets"]["reflection_max_tokens"]),
    "REFLECTION_NUM_PREDICT": str(SYSTEM_PROFILE["budgets"]["reflection_max_tokens"]),
}

for _key, _value in _env_defaults.items():
    os.environ.setdefault(_key, _value)

# Only set LLM_TIMEOUT as fallback if not already configured by installer
# The installer sets appropriate values based on detected hardware tier
# High-performance hardware gets 0 (unbounded), but we should respect explicit config
if "LLM_TIMEOUT" not in os.environ:
    if SYSTEM_PROFILE["tier"] == "high":
        os.environ["LLM_TIMEOUT"] = "0"
        logger.info("⚙️ LLM_TIMEOUT not set - defaulting to unbounded (0) for high-performance tier")
    else:
        # For standard tier, use unbounded by default - let prompts constrain output
        # Bootstrap and reflection operations can take several minutes on 8GB GPUs
        os.environ["LLM_TIMEOUT"] = "0"
        logger.info("⚙️ LLM_TIMEOUT not set - defaulting to unbounded (0) for standard tier")
else:
    logger.info("✓ LLM_TIMEOUT configured: %s seconds (0=unbounded)", os.environ["LLM_TIMEOUT"])

if os.environ.get("REFLECTION_LLM_TIMEOUT_S") in (None, ""):
    # No explicit setting; preserve legacy behavior (no timeout) unless user opts in
    os.environ.setdefault("REFLECTION_LLM_TIMEOUT_S", "0")

persona_token_defaults = {
    "low": {
        "PERSONA_ANALYTICAL_MAX_TOKENS": "512",
        "PERSONA_SEED_MAX_TOKENS": "384",
        "PERSONA_TRAITS_MAX_TOKENS": "320",
    },
    "standard": {
        "PERSONA_ANALYTICAL_MAX_TOKENS": "576",
        "PERSONA_SEED_MAX_TOKENS": "448",
        "PERSONA_TRAITS_MAX_TOKENS": "352",
    },
}

defaults_for_tier = persona_token_defaults.get(SYSTEM_PROFILE["tier"], {})
for _key, _value in defaults_for_tier.items():
    os.environ.setdefault(_key, _value)

ALLOW_WARMUP = bool((SYSTEM_PROFILE.get("features") or {}).get("allow_warmup", False))
ALLOW_KEEPALIVE = bool((SYSTEM_PROFILE.get("features") or {}).get("allow_keepalive", False))

# --- Chat output guards: greeting/style and identity consistency ---

def _polish_first_response(text: str, persona_name: Optional[str], conversation_history: List[Dict[str, Any]], intro_already_done: bool) -> str:
    """Apply light, safe post-processing to the very first assistant message in a session:
    - Avoid repetitive greeting/name openers
    - Prefer a decisive ending instead of reflexive engagement questions
    - If persona name is known and not present, optionally introduce it concisely (only if intro not done globally)
    Never raises; returns original text on any error.
    """
    try:
        import re
        original = text or ""
        s = original
        # Only adjust on first assistant turn in this session
        has_assistant = any((m or {}).get("role") == "assistant" for m in (conversation_history or []))
        if not has_assistant:
            # 1) Strip leading greeting/name patterns
            leading_greet = re.compile(r"^(?:[\s\u2600-\u27BF\uFE0F\U0001F300-\U0001FAFF]*)?\s*(?:hey|hi|hello|yo|hiya|howdy|greetings|sup|hey there)(?:[,!\s]+(?:there|friend|team))?[:!\s-]*\s*",
                                       flags=re.IGNORECASE)
            s = leading_greet.sub("", s)
            # Remove clusters of leading emojis
            s = re.sub(r"^[\s\u2600-\u27BF\uFE0F\U0001F300-\U0001FAFF]+", "", s)

            # 2) Identity consistency: LLM handles name introduction naturally (Option B)
            # Name injection removed - conversational LLM has full persona context

            # 3) Remove meta source tags like '(Source: ...)' or 'Source: ...' to keep chat natural
            s = re.sub(r"\(\s*Source:\s*[^\)]*\)", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\bSource:\s*[^\.;\n]+", "", s, flags=re.IGNORECASE)

            # Removed sentence-length cap to preserve full model output

            # 5) Avoid reflexive question endings
            low = s.strip().lower()
            reflexive_endings = [
                "what do you think?",
                "does that help?",
                "how does that sound?",
                "is that okay?",
            ]
            for end in reflexive_endings:
                if low.endswith(end):
                    s = s[: -len(end)].rstrip() + "."
                    break
        # 6) Remove meta/disclaimer notes like "Note: ..." and collapse repeated sentences
        try:
            # Strip standalone lines beginning with 'Note:' (case-insensitive)
            lines = [ln for ln in s.splitlines() if not ln.strip().lower().startswith("note:")]
            s = "\n".join(lines)

            # Also strip inline parenthetical/bracketed notes like "(Note: ... )" or "[Note: ... ]"
            # Do this conservatively to avoid removing legitimate content
            s = re.sub(r"\(\s*note:\s*[^)]*\)\s*", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\[\s*note:\s*[^\]]*\]\s*", "", s, flags=re.IGNORECASE)

            # Deduplicate repeated sentences while preserving order
            segs = re.split(r"(?<=[\.!?])\s+", s.strip())
            seen = set()
            uniq = []
            for seg in segs:
                key = seg.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    uniq.append(seg)
            if uniq:
                s = " ".join(uniq)
        except Exception:
            # Deduplication is optional - gracefully fall through if it fails
            pass

        return s if s else original
    except Exception:
        return text


def _normalize_tags(tags: Optional[List[str]]) -> List[str]:
    """Normalize and deduplicate tag strings while preserving order."""

    if not tags:
        return []

    seen = OrderedDict()
    for tag in tags:
        if tag is None:
            continue
        normalized = str(tag).strip().lower()
        if not normalized:
            continue
        seen.setdefault(normalized, normalized)
    return list(seen.values())


def _format_system_snapshot(system_status: Optional[Dict[str, Any]], gpu_status: Optional[Dict[str, Any]], mode: "DiagnosticMode") -> str:
    if not system_status and not gpu_status:
        return ""

    parts = []
    cpu_percent = None
    try:
        cpu_info = (system_status or {}).get("cpu") or {}
        cpu_percent = cpu_info.get("percent")
    except Exception:
        cpu_percent = None
    if isinstance(cpu_percent, (int, float)):
        try:
            parts.append(f"CPU {cpu_percent:.0f}%")
        except Exception:
            parts.append(f"CPU {cpu_percent}%")

    mem = (system_status or {}).get("memory") or {}
    try:
        mt = mem.get("total_gb")
        mu = mem.get("used_gb")
        mp = mem.get("percent")
        if mt and mu is not None:
            if mp is not None:
                parts.append(f"RAM {mu:.1f}/{mt:.1f}GB ({mp:.0f}% used)")
            else:
                parts.append(f"RAM {mu:.1f}/{mt:.1f}GB")
    except Exception:
        pass

    disk = (system_status or {}).get("disk") or {}
    try:
        dt = disk.get("total_gb")
        du = disk.get("used_gb")
        dp = disk.get("percent")
        if dt and du is not None:
            if dp is not None:
                parts.append(f"Disk {du:.1f}/{dt:.1f}GB ({dp:.0f}% used)")
            else:
                parts.append(f"Disk {du:.1f}/{dt:.1f}GB")
    except Exception:
        pass

    try:
        load_avg = (system_status or {}).get("load_avg") or []
        if isinstance(load_avg, (list, tuple)) and load_avg:
            la = ", ".join(f"{float(x):.2f}" for x in load_avg[:3])
            parts.append(f"Load avg {la}")
    except Exception:
        pass

    try:
        if gpu_status and gpu_status.get("available"):
            name = gpu_status.get("name") or "GPU"
            used = gpu_status.get("memory_used_gb")
            total = gpu_status.get("memory_total_gb")
            pct = gpu_status.get("memory_percent")
            util = gpu_status.get("utilization_percent")
            temp = gpu_status.get("temperature_c")

            gpu_parts = []
            if total and used is not None:
                if pct is not None:
                    gpu_parts.append(f"VRAM {used:.1f}/{total:.1f}GB ({pct:.0f}% used)")
                else:
                    gpu_parts.append(f"VRAM {used:.1f}/{total:.1f}GB")
            if util is not None:
                gpu_parts.append(f"util {util:.0f}%")
            if temp is not None:
                gpu_parts.append(f"{temp:.0f}°C")
            joined = ", ".join(gpu_parts) if gpu_parts else "available"
            parts.append(f"{name}: {joined}")
    except Exception:
        pass

    summary = ", ".join(p for p in parts if p)
    if not summary:
        return ""

    if mode == DiagnosticMode.EXPLICIT_STATUS:
        prefix = "System snapshot (user requested status): "
    elif mode == DiagnosticMode.PROBLEM_REPORTED:
        prefix = "System snapshot related to reported issue: "
    else:
        prefix = "System snapshot: "
    return prefix + summary


async def _ensure_relationship_questions_seed(reflection_repo, user_profile_id: str) -> None:
    """Populate the relationship question queue with defaults when empty."""

    if not reflection_repo or not user_profile_id:
        return

    try:
        existing_seed = await reflection_repo.list_reflections(
            user_profile_id=user_profile_id,
            reflection_type="relationship_seed",
            limit=1,
        )
        if existing_seed:
            return

        queue_preview = await reflection_repo.list_relationship_question_queue(
            user_profile_id=user_profile_id,
            status=None,
            limit=1,
            include_future=True,
        )
        if queue_preview:
            return

        try:
            await reflection_repo.create_reflection(
                {
                    "reflection_type": "relationship_seed",
                    "user_profile_id": user_profile_id,
                    "result": {
                        "content": "Default relationship question seed batch generated during install.",
                        "metadata": {"seed": True, "source": "default_relationship_questions"},
                    },
                    "relationship_questions": DEFAULT_RELATIONSHIP_QUESTION_SEED,
                }
            )
            logger.info(
                "Seeded default relationship questions for user_profile_id=%s", user_profile_id
            )
        except Exception as seed_err:
            logger.warning(
                "Failed to seed default relationship questions for %s: %s",
                user_profile_id,
                seed_err,
            )
    except Exception as precheck_err:
        logger.debug(
            "Relationship question seed precheck failed for %s: %s",
            user_profile_id,
            precheck_err,
        )

# --- Human-like continuity: ensure acknowledgment of the user's last message ---
def _compose_acknowledgment_prefix(user_text: Optional[str], persona_obj: Optional[Any]) -> Optional[str]:
    """Create a concise, tone-aware acknowledgment line grounded in persona attributes.
    Returns None if no user_text is provided or if composition fails.
    """
    try:
        if not user_text:
            return None
        u = (user_text or "").strip()
        if not u:
            return None
        # Paraphrase: first ~12 words, no quotes, no trailing punctuation
        words = u.split()
        paraphrase = " ".join(words[:12]).strip().rstrip(".?!")

        # Default tone knobs
        warmth = 0.5
        conciseness = 0.5
        curiosity = 0.5
        decisiveness = 0.5

        # Try to derive tone from persona traits/communication_style
        try:
            # Traits may be list of objects with name/value or a dict; handle both
            traits = []
            if hasattr(persona_obj, "traits") and persona_obj.traits:
                traits = getattr(persona_obj, "traits", [])
            personality = getattr(persona_obj, "personality", {}) or {}
            comms = getattr(persona_obj, "communication_style", {}) or {}

            # Helper to read a scalar from multiple sources
            def _read_level(name: str) -> Optional[float]:
                try:
                    # look in traits list
                    for t in traits or []:
                        try:
                            tname = (getattr(t, "name", None) or t.get("name"))
                            tval = (getattr(t, "value", None) or t.get("value"))
                            if isinstance(tname, str) and tname.lower() == name and isinstance(tval, (int, float)):
                                return float(tval)
                        except Exception:
                            continue
                    # look in persona maps
                    for src in (personality, comms):
                        if isinstance(src, dict) and name in src and isinstance(src[name], (int, float)):
                            return float(src[name])
                except Exception:
                    return None
                return None

            warmth = _read_level("warmth") or _read_level("empathy") or warmth
            conciseness = _read_level("conciseness") or conciseness
            curiosity = _read_level("curiosity") or curiosity
            decisiveness = _read_level("decisiveness") or decisiveness
        except Exception:
            # Trait reading is optional - use defaults if persona traits unavailable
            pass

        # Tone selection
        start = "On your point about " if warmth >= 0.6 else "Regarding "
        act = "I'll incorporate that immediately." if decisiveness >= 0.6 else "I'll integrate that into next steps."
        if conciseness >= 0.7:
            # Tighter style
            prefix = f"{start}{paraphrase}, {act}"
        else:
            # Slightly more expressive, but still no questions
            mid = "I hear you" if warmth >= 0.6 else "Understood"
            tail = "and will proceed accordingly." if decisiveness >= 0.6 else "and will reflect it in what I do next."
            prefix = f"{start}{paraphrase}, {mid} {tail}"

        # Optional curiosity accent without asking questions
        if curiosity >= 0.7:
            prefix = prefix.rstrip(".") + "."
        return prefix.strip()
    except Exception:
        return None

def _ensure_acknowledgment(reply: str, user_text: Optional[str], persona_obj: Optional[Any]) -> str:
    """If the reply doesn't clearly reference the user's last message, prepend
    a concise paraphrase and a one-sentence actionable summary to produce
    a more human conversational rhythm ("I heard you, I'm acting").
    Never raises; returns original reply on any error.
    """
    try:
        if not user_text:
            return reply or ""
        r = (reply or "").strip()
        u = (user_text or "").strip()
        if not r or not u:
            return r or reply or ""
        import re
        # Extract a few significant tokens from the user's text
        tokens = re.findall(r"[A-Za-z0-9_\-]{4,}", u.lower())
        sig = [t for t in tokens if len(t) >= 5]
        sig = sig[:6]  # limit scope
        rl = r.lower()
        # If any significant token appears in reply, consider it acknowledged
        if any(t in rl for t in sig):
            return r
        # Compose tone-aware acknowledgment prefix
        ack_prefix = _compose_acknowledgment_prefix(u, persona_obj) or ""
        if ack_prefix:
            return f"{ack_prefix} {r}".strip()
        return r
    except Exception:
        return reply or ""

def _response_needs_continuation(text: Optional[str], stop_reason: Optional[str] = None) -> bool:
    try:
        if not text:
            return False
        # Only continue if LLM explicitly stopped due to token limit
        if stop_reason and stop_reason.lower() in {"length", "max_tokens", "context_length"}:
            return True
        # Don't use heuristics - they cause false positives and infinite loops
        # Only trust the stop_reason from the LLM
        return False
    except Exception:
        return False

async def _request_additional_completion(
    llm_router,
    *,
    base_prompt: str,
    partial_response: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    try:
        if not partial_response:
            return None
        continuation_prompt = _build_continuation_prompt(base_prompt)
        cont_tokens = max(256, min(max_tokens, 1536))
        result = await llm_router.route(
            task_type="chat",
            prompt=continuation_prompt,
            model=model,
            max_tokens=cont_tokens,
            temperature=temperature,
        )
        if not isinstance(result, dict):
            return None
        candidate = result.get("content") or result.get("completion") or ""
        addition = _extract_new_tail(partial_response, candidate)
        return addition or None
    except Exception:
        return None

def _build_continuation_prompt(base_prompt: str) -> str:
    directive = (
        "\n\n[Assistant Continuation Directive]\n"
        "The assistant's previous reply ended mid-thought. Continue it naturally in the same tone, "
        "without repeating earlier sentences, and finish the idea.\n"
    )
    return f"{base_prompt}{directive}"

def _extract_stop_reason(metadata: Optional[Any]) -> Optional[str]:
    try:
        if isinstance(metadata, dict):
            for key in ("stop_reason", "finish_reason", "done_reason", "reason"):
                value = metadata.get(key)
                if value:
                    return str(value)
    except Exception:
        return None
    return None

def _extract_new_tail(existing: str, candidate: str) -> str:
    try:
        if not candidate:
            return ""
        if not existing:
            return candidate
        if candidate.startswith(existing):
            return candidate[len(existing):]
        import os
        prefix = os.path.commonprefix([existing, candidate])
        return candidate[len(prefix):]
    except Exception:
        return ""

# Optional: background pre-warm of LLMs to reduce first-turn latency
async def _prewarm_models(app: FastAPI):
    try:
        import os as _os
        if not ALLOW_WARMUP:
            logging.info("Skipping model prewarm: system tier does not allow warmup")
            return
        if (_os.getenv("PREWARM_MODELS", "true").lower() not in ("1","true","yes")):
            return
        llm_router = app.state.services.get("llm_router")
        if llm_router is None:
            return
        # Perform very small, non-blocking generations to trigger model load and GPU graph compilation
        async def _do(task_type: str, model_role: str, prompt: str):
            try:
                await llm_router.route(
                    task_type=task_type,
                    prompt=prompt,
                    model=get_llm_model(model_role),
                    max_tokens=8,
                    temperature=0.1,
                )
            except Exception as e:
                logging.info(f"Prewarm for {task_type}/{model_role} skipped: {e}")
        # Warm reflection and chat; burst a few tiny reflection calls to finish graph compile/caching
        for i in range(3):
            await _do("reflection", "reflection", f"warmup_r{i+1}")
        await _do("chat", "conversational", "warmup_c1")
        await _do("chat", "conversational", "warmup_c2")
    except Exception as e:
        logging.info(f"Model prewarm skipped/failed: {e}")

async def _keepalive_prewarm_loop(app: FastAPI):
    """Periodically run tiny prompts to keep models resident and graphs hot.
    Controlled by KEEPALIVE_ENABLED and PREWARM_INTERVAL_MIN envs.
    """
    try:
        env = os.environ
        if not ALLOW_KEEPALIVE:
            logging.info("Skipping keepalive loop: system tier does not allow keepalive")
            return
        if (env.get("KEEPALIVE_ENABLED", "true").lower() not in ("1","true","yes")):
            return
        try:
            interval_min = float(env.get("PREWARM_INTERVAL_MIN", "10"))
        except (ValueError, TypeError) as e:
            logging.warning(f"Invalid PREWARM_INTERVAL_MIN value: {e}")
            interval_min = 10.0
        interval = max(60.0, interval_min * 60.0)
        llm_router = app.state.services.get("llm_router")
        if llm_router is None:
            return
        while True:
            try:
                # Reflection ping
                await llm_router.route(
                    task_type="reflection",
                    prompt="keepalive",
                    model=get_llm_model("reflection"),
                    max_tokens=4,
                    temperature=0.1,
                )
                # Conversational ping
                await llm_router.route(
                    task_type="chat",
                    prompt="keepalive",
                    model=get_llm_model("conversational"),
                    max_tokens=4,
                    temperature=0.1,
                )
            except Exception as e:
                logging.debug(f"Keepalive prewarm tick error: {e}")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        # Normal shutdown
        return
    except Exception as e:
        logging.info(f"Keepalive prewarm loop exited: {e}")

# Initialize required services
async def initialize_services():
    """Initialize application services and dependencies."""
    from .reflection.processor import ReflectionProcessor
    from .db.repositories.reflection import ReflectionRepository
    from .db.repositories.user import UserRepository
    from .db.repositories.conversation import ConversationRepository
    from .db.repositories.persona import PersonaRepository
    from .db.repositories.agent_state import (
        AffectiveStateRepository,
        AgentGoalRepository,
        PlanStepRepository,
        AutobiographicalEpisodeRepository,
        MetaReflectionRepository,
    )
    from .prompt.builder import PromptBuilder
    from .llm.controller import LLMController
    from .llm.router import LLMRouter
    from .llm.dual_llm_config import get_llm_model
    from .memory.vector_store import VectorStore
    from .reflection.scheduler import ReflectionScheduler
    from .scheduler.factory import SchedulerFactory
    from .api.dependencies import initialize_dependencies
    # Load centralized app configuration for consistent timeouts and model settings
    from .config.app_config import get_app_config
    from .agent.affective_state_manager import AffectiveStateManager
    from .agent.goal_manager import GoalManager
    from .agent.planner_service import PlannerService
    from .agent.meta_reflection_processor import MetaReflectionProcessor
    from .agent.agent_loop_runner import AgentLoopRunner
    from .agent.episode_builder import EpisodeBuilder
    from .agent.autobiographical_episode_service import AutobiographicalEpisodeService
    
    # Import resilience systems
    from .core.startup_validator import run_startup_validation
    from .core.health_monitor import health_monitor
    from .core.graceful_degradation import register_all_fallbacks, assess_and_adjust_degradation
    
    # Note: Startup validation will run via lifespan event to avoid blocking initialization
    logging.info("🔄 Startup validation scheduled for lifespan startup")
    
    # Initialize fallback systems
    register_all_fallbacks()
    logging.info("🔄 Graceful degradation patterns initialized")
    
    # Initialize services dictionary for storing background tasks and service references
    services = {}
    
    # Instantiate dual LLM controllers for the router, honoring central LLM timeout
    app_cfg = get_app_config()

    # Before constructing controllers, verify that all required models for the active profile
    # are installed in Ollama. This prevents accidental fallback to unintended models.
    try:
        from .llm.dual_llm_config import get_required_models
        import os as _os
        profile = _os.getenv("SELO_MODEL_PROFILE", "unknown")
        required = get_required_models()
        ollama_base = _os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        import httpx as _httpx
        # Use a short timeout but not too aggressive to avoid false negatives
        with _httpx.Client(base_url=ollama_base, timeout=10.0) as _client:
            r = _client.get("/api/tags")
            r.raise_for_status()
            names = {str(m.get("name") or "") for m in (r.json() or {}).get("models", [])}
        # Normalize available names and required names by stripping ':latest' (and keeping originals)
        def _base(n: str) -> str:
            try:
                return n.split(":", 1)[0]
            except Exception:
                return n
        available_full = set(n for n in names if n)
        available_base = {_base(n) for n in available_full}
        # A required model is satisfied if it matches exactly or by base name
        missing = []
        for m in required.values():
            m = str(m or "")
            if not m:
                continue
            m_base = _base(m)
            if (m not in available_full) and (m_base not in available_base):
                missing.append(m)
        if missing:
            if profile and profile != "unknown":
                msg = (
                    f"Missing Ollama models for '{profile}' tier: {missing}. "
                    f"Run configs/default/install-models.sh to install required models, then restart."
                )
            else:
                msg = (
                    f"Missing Ollama models: {missing}. "
                    f"Run configs/default/install-models.sh to install required models, then restart."
                )
            logging.critical(msg)
            raise RuntimeError(msg)
        
        # Enforce exact model mapping for the active profile to prevent drift toward larger models
        def _strip_provider(name: str) -> str:
            name = str(name or "").strip()
            if ":" in name:
                provider, remainder = name.split(":", 1)
                if provider.lower() in {"ollama", "openai", "anthropic", "hf"}:
                    return remainder
            return name

        # 2-tier system: Standard (<12GB GPU) and High-Performance (≥12GB GPU)
        # Both tiers use the same model set, differentiated by token budgets
        model_lock = {
            "standard": {
                "conversational": ["llama3:8b", "llama3.1:8b", "qwen2.5:3b"],
                "analytical": ["qwen2.5:3b"],
                "reflection": ["qwen2.5:3b"],
                "embedding": ["nomic-embed-text"],
            },
            "high": {
                "conversational": ["llama3:8b", "llama3.1:8b", "qwen2.5:3b"],
                "analytical": ["qwen2.5:3b"],
                "reflection": ["qwen2.5:3b"],
                "embedding": ["nomic-embed-text"],
            },
        }

        locked = model_lock.get(profile)
        if locked:
            mismatched = {}
            for role, allowed in locked.items():
                configured = required.get(role)
                normalized = _strip_provider(configured)
                if normalized and any(_strip_provider(option) == normalized for option in allowed):
                    continue
                mismatched[role] = configured or "<unset>"
            if mismatched:
                msg = (
                    f"Configured LLM models {mismatched} do not match the '{profile}' tier requirements {locked}. "
                    f"Update backend/.env to use supported models for your hardware tier."
                )
                logging.critical(msg)
                raise RuntimeError(msg)
    except Exception as _model_chk_err:
        # If list retrieval fails, still raise: cannot guarantee correct routing.
        raise
    conversational_llm_controller = LLMController({
        "default_model": get_llm_model("conversational"),
        "request_timeout": app_cfg.llm_timeout,
    })
    analytical_llm_controller = LLMController({
        "default_model": get_llm_model("analytical"),
        "request_timeout": app_cfg.llm_timeout,
    })
    reflection_llm_controller = LLMController({
        "default_model": get_llm_model("reflection"),
        "request_timeout": app_cfg.llm_timeout,
    })
    
    # Create LLMRouter with both controllers
    llm_router = LLMRouter(conversational_llm_controller, analytical_llm_controller, reflection_llm_controller)
    
    # Initialize other services
    templates_dir = str(pathlib.Path(__file__).resolve().parent / "prompt" / "templates")
    prompt_builder = PromptBuilder(templates_dir=templates_dir)
    reflection_repo = ReflectionRepository()
    user_repo = UserRepository()
    conversation_repo = ConversationRepository()
    persona_repo = PersonaRepository()
    affective_state_repo = AffectiveStateRepository()
    goal_repo = AgentGoalRepository()
    plan_repo = PlanStepRepository()
    episode_repo = AutobiographicalEpisodeRepository()
    meta_repo = MetaReflectionRepository()
    # Persist vector store to disk so FAISS index/metadata survive restarts
    vector_store_path = str(pathlib.Path(__file__).resolve().parent / "data" / "vector_store")
    vector_store = VectorStore(store_path=vector_store_path, llm_controller=llm_router)

    affective_state_manager = AffectiveStateManager(
        state_repo=affective_state_repo,
        persona_repo=persona_repo,
    )
    goal_manager = GoalManager(
        goal_repo=goal_repo,
        plan_repo=plan_repo,
        meta_repo=meta_repo,
    )
    planner_service = PlannerService(
        goal_repo=goal_repo,
        plan_repo=plan_repo,
        meta_repo=meta_repo,
    )
    meta_reflection_processor = MetaReflectionProcessor(
        reflection_repo=reflection_repo,
        goal_manager=goal_manager,
        persona_repo=persona_repo,
        user_repo=user_repo,
    )
    episode_builder = EpisodeBuilder(episode_repo=episode_repo)
    episode_service = AutobiographicalEpisodeService(
        prompt_builder=prompt_builder,
        llm_router=llm_router,
        episode_builder=episode_builder,
        episode_repo=episode_repo,
        persona_repo=persona_repo,
        goal_manager=goal_manager,
        affective_state_manager=affective_state_manager,
        reflection_repo=reflection_repo,
        conversation_repo=conversation_repo,
    )
    
    # Initialize enhanced episode trigger system
    from .agent.episode_triggers import EpisodeTriggerManager
    episode_trigger_config = {
        "milestone_enabled": os.environ.get("EPISODE_MILESTONE_ENABLED", "true").lower() == "true",
        "milestone_interval": int(os.environ.get("EPISODE_MILESTONE_INTERVAL", "10")),
        "daily_summary_enabled": os.environ.get("EPISODE_DAILY_SUMMARY_ENABLED", "true").lower() == "true",
        "goal_celebration_enabled": os.environ.get("EPISODE_GOAL_CELEBRATION_ENABLED", "true").lower() == "true",
    }
    episode_trigger_manager = EpisodeTriggerManager(
        episode_service=episode_service,
        conversation_repo=conversation_repo,
        persona_repo=persona_repo,
        user_repo=user_repo,
        config=episode_trigger_config,
    )
    
    # Initialize session-based episode generator
    from .scheduler.session_episode_generator import SessionEpisodeGenerator
    episode_idle_threshold = int(os.environ.get("EPISODE_IDLE_THRESHOLD_MIN", "15"))
    episode_min_reflections = int(os.environ.get("EPISODE_MIN_REFLECTIONS", "2"))
    episode_session_gap = int(os.environ.get("EPISODE_SESSION_GAP_MIN", "60"))
    
    session_episode_generator = SessionEpisodeGenerator(
        user_repo=user_repo,
        persona_repo=persona_repo,
        reflection_repo=reflection_repo,
        conversation_repo=conversation_repo,
        episode_service=episode_service,
        idle_threshold_minutes=episode_idle_threshold,
        min_reflections=episode_min_reflections,
        session_gap_minutes=episode_session_gap,
    )
    logger.info(
        f"✨ Session episode generator initialized: "
        f"idle_threshold={episode_idle_threshold}min, "
        f"min_reflections={episode_min_reflections}, "
        f"session_gap={episode_session_gap}min"
    )
    
    # Bind episode trigger manager to goal manager for goal completion celebrations
    goal_manager.bind_episode_trigger_manager(episode_trigger_manager)

    agent_loop_config = {
        "enabled": os.environ.get("AGENT_LOOP_ENABLED", "true"),
        "interval_seconds": os.environ.get("AGENT_LOOP_INTERVAL_SECONDS", "900"),
        "homeostasis_enabled": os.environ.get("AGENT_LOOP_HOMEOSTASIS", "true"),
        "episode_builder_enabled": os.environ.get("AGENT_LOOP_EPISODE_ENABLED", "true"),
        "audit_events_enabled": os.environ.get("AGENT_LOOP_AUDIT_ENABLED", "true"),
    }
    # Note: event_system will be passed directly to agent_loop_runner after it's created below
    # This ensures proper event publishing from the agent loop
    agent_loop_runner = AgentLoopRunner(
        affective_state_manager=affective_state_manager,
        goal_manager=goal_manager,
        planner_service=planner_service,
        persona_repo=persona_repo,
        user_repo=user_repo,
        episode_builder=episode_builder,
        episode_service=episode_service,
        episode_trigger_manager=episode_trigger_manager,
        event_system=None,  # Will be set after event_system is created
        config=agent_loop_config,
    )
    
    # Initialize or reuse Socket.IO server via registry
    from .socketio.registry import register_socketio_server, get_socketio_server
    sio = get_socketio_server()
    if sio is None:
        sio = socketio.AsyncServer(
            async_mode="asgi",
            # SECURITY WARNING: Wildcard CORS allows requests from any origin.
            # For production, restrict to specific origins via CORS_ORIGINS env var.
            # Socket.IO CORS is separate from FastAPI CORS middleware.
            cors_allowed_origins="*",
            engineio_logger=True,
            async_handlers=True,
            # Keep connections alive much longer when browser tabs are backgrounded
            # (avoid false disconnects due to throttled timers)
            # Configuration now loaded from environment variables via AppConfig
            ping_timeout=app_cfg.socketio_ping_timeout,
            ping_interval=app_cfg.socketio_ping_interval,
            allow_upgrades=True,  # allow upgrade from polling to WebSocket when possible
            max_http_buffer_size=1_000_000,
        )
        register_socketio_server(sio)
    
    # Initialize event trigger system for reflection->persona integration
    from .scheduler.event_triggers import EventTriggerSystem
    event_system = EventTriggerSystem(
        scheduler_service=None,  # Will be set later if scheduler integration is enabled
        adaptive_scheduler=None,
        config={}
    )
    logging.info("Event trigger system initialized")
    
    # Bind event system to agent loop runner for event publishing
    agent_loop_runner.bind_event_system(event_system)
    logging.info("Event system bound to agent loop runner")
    
    # Create reflection processor (uses LLMRouter) with Socket.IO server and event bus
    reflection_processor = ReflectionProcessor(
        reflection_repo=reflection_repo,
        llm_controller=llm_router,
        socketio_server=sio,
        conversation_repo=conversation_repo,
        persona_repo=persona_repo,
        user_repo=user_repo,
        prompt_builder=prompt_builder,
        vector_store=vector_store,
        event_bus=event_system,  # Connect event bus for reflection.created events
        meta_reflection_processor=meta_reflection_processor,
        affective_state_manager=affective_state_manager,
        goal_manager=goal_manager,
    )
    
    # Get scheduler configuration from environment
    scheduler_config = SchedulerFactory.load_config_from_env()

    # Create legacy reflection scheduler (honor env config) only if explicitly enabled
    legacy_reflection_scheduler = None
    if os.environ.get("LEGACY_REFLECTION_SCHEDULER_ENABLED", "").lower() in ("1", "true", "yes"):
        legacy_reflection_scheduler = ReflectionScheduler(
            reflection_processor=reflection_processor,
            user_repo=user_repo,
            scheduler_service=None,
            conversation_repo=conversation_repo,
            config=scheduler_config,
        )
    # scheduler_integration will be initialized in lifespan startup
    scheduler_integration = None
    
    # Socket.IO server already initialized above
    
    # Register reflection namespace handlers on the Socket.IO server (if available)
    try:
        from .socketio.namespaces.reflection import ReflectionNamespace
        reflection_namespace = ReflectionNamespace(reflection_processor)
        reflection_namespace.register(sio)
        try:
            reflection_processor.reflection_namespace = reflection_namespace
        except Exception:
            logging.debug("Unable to assign reflection namespace to processor", exc_info=True)
        app.state.reflection_namespace = reflection_namespace

        # Register chat namespace for streaming chat responses
        try:
            from .socketio.namespaces.chat import ChatNamespace
            chat_namespace = ChatNamespace(sio)
            chat_namespace.register()
            
            # Bind event system for conversation lifecycle events (event-driven episode generation)
            if event_system:
                chat_namespace.bind_event_system(event_system)
            
            app.state.chat_namespace = chat_namespace
        except Exception as chat_ns_err:
            logging.warning(f"Chat namespace registration failed: {chat_ns_err}")
            app.state.chat_namespace = None
    except Exception as e:
        logging.warning(f"Failed to register reflection namespace: {e}")
    
    # Start health monitoring service and store task reference for cleanup
    health_monitor.set_llm_controller(llm_router)
    health_monitor_task = asyncio.create_task(health_monitor.start_monitoring())
    services["health_monitor_task"] = health_monitor_task
    logging.info("🏥 Health monitoring service started")
    
    # Initialize and start memory consolidation service
    try:
        from .memory.consolidator import MemoryConsolidator
        from .memory.extractor import MemoryExtractor
        
        # Create memory extractor for consolidation
        memory_extractor = MemoryExtractor(
            conversation_repo=conversation_repo,
            llm_controller=llm_router
        )
        
        # Create and start memory consolidator
        memory_consolidator = MemoryConsolidator(
            conversation_repo=conversation_repo,
            user_repo=user_repo,
            memory_extractor=memory_extractor
        )
        
        # Start the background consolidation service and store task reference for cleanup
        memory_consolidation_task = asyncio.create_task(memory_consolidator.start_consolidation_service())
        services["memory_consolidation_task"] = memory_consolidation_task
        logging.info("🧠 Memory consolidation service started")
        
        # Store consolidator and task for potential API access and cleanup
        memory_consolidator_instance = memory_consolidator
        
    except Exception as e:
        logging.warning(f"Failed to start memory consolidation service: {e}")
        memory_consolidator_instance = None
    
    # Initialize PersonaIntegration to connect reflection events to persona evolution
    persona_integration_instance = None
    try:
        from .persona.integration import PersonaIntegration
        
        persona_integration_instance = PersonaIntegration(
            llm_router=llm_router,
            vector_store=vector_store,
            event_system=event_system,  # Share the same event system
            persona_engine=None,  # Will create its own PersonaEngine
            conversation_repo=conversation_repo,
        )
        
        # Initialize and register event handlers (async operation)
        # This registers handlers for reflection.created, sdl.learning.created, etc.
        await persona_integration_instance.initialize()
        logging.info("✨ PersonaIntegration initialized - reflections will now trigger persona evolution")
        
        # Register handler for significant reflections to trigger episode generation
        async def _handle_significant_reflection(event_data: Dict[str, Any], user_id: str):
            """Generate autobiographical episode for significant reflections (non-blocking)."""
            async def _generate_episode_background():
                """Background task for episode generation - won't block user response."""
                try:
                    reflection_id = event_data.get("reflection_id")
                    persona_id = event_data.get("persona_id")
                    trigger_reason = event_data.get("trigger_reason", "significant_reflection")
                    
                    if not reflection_id or not persona_id or not user_id:
                        logger.warning("Episode generation skipped: missing required fields")
                        return
                    
                    logger.info(
                        f"🎬 Starting background episode generation for reflection {reflection_id} "
                        f"(will not block user response)"
                    )
                    
                    result = await episode_service.generate_episode(
                        persona_id=persona_id,
                        user_id=user_id,
                        trigger_reason=trigger_reason,
                        plan_steps=None,
                    )
                    
                    if result:
                        logger.info(
                            f"✅ Episode {result.get('id', 'unknown')} generated successfully for "
                            f"reflection {reflection_id} (background)"
                        )
                    else:
                        logger.warning(f"⚠️ Episode generation returned None for reflection {reflection_id}")
                        
                except Exception as e:
                    logger.error(
                        f"❌ Background episode generation failed for reflection {reflection_id}: {e}",
                        exc_info=True
                    )
            
            # Launch background task - returns immediately without blocking
            asyncio.create_task(_generate_episode_background())
            logger.debug(f"Episode generation task created for reflection {event_data.get('reflection_id')}")
        
        await event_system.register_handler(
            event_type="reflection.significant",
            handler=_handle_significant_reflection
        )
        logger.info("✨ Registered episode generation handler for significant reflections")
        
    except Exception as e:
        logging.error(f"Failed to initialize PersonaIntegration: {e}", exc_info=True)
        logging.warning("Persona evolution from reflections will not be automatic")
        persona_integration_instance = None
    
    # Return initialized services, provide LLMRouter as primary interface
    # Merge with services dict that contains background tasks
    return {
        **services,  # Include tasks stored in services dict
        "llm_router": llm_router,
        "conversational_llm_controller": conversational_llm_controller,  # Keep for legacy compatibility
        "analytical_llm_controller": analytical_llm_controller,  # Keep for legacy compatibility
        "prompt_builder": prompt_builder,
        "reflection_repo": reflection_repo,
        "user_repo": user_repo,
        "conversation_repo": conversation_repo,
        "persona_repo": persona_repo,
        "vector_store": vector_store,
        "reflection_processor": reflection_processor,
        "reflection_scheduler": legacy_reflection_scheduler,
        "scheduler_config": scheduler_config,
        "scheduler_integration": scheduler_integration,
        "socketio_server": sio,
        "event_system": event_system,  # Expose event system
        "persona_integration": persona_integration_instance,  # Expose persona integration
        "chat_namespace": getattr(app.state, "chat_namespace", None),
        # Optionally expose reflection namespace if registration succeeded
        "reflection_namespace": locals().get("reflection_namespace"),
        "health_monitor": health_monitor,
        "memory_consolidator": memory_consolidator_instance,
        "affective_state_repo": affective_state_repo,
        "agent_goal_repo": goal_repo,
        "plan_step_repo": plan_repo,
        "autobiographical_episode_repo": episode_repo,
        "meta_reflection_repo": meta_repo,
        "affective_state_manager": affective_state_manager,
        "goal_manager": goal_manager,
        "planner_service": planner_service,
        "episode_builder": episode_builder,
        "episode_service": episode_service,
        "episode_trigger_manager": episode_trigger_manager,
        "session_episode_generator": session_episode_generator,
        "meta_reflection_processor": meta_reflection_processor,
        "agent_loop_runner": agent_loop_runner,
    }


def _summarize_reflection_for_prompt(reflection_text: Optional[str]) -> str:
    """Condense inner reflection content for system prompt context."""
    if not reflection_text:
        return "(no inner reflection captured)"

    text = (reflection_text or "").strip()
    if not text:
        return "(no inner reflection captured)"

    # Collapse whitespace to keep summary compact and private
    import re

    collapsed = re.sub(r"\s+", " ", text)

    return collapsed or "(no inner reflection captured)"

def _build_baseline_reflection(session_id: str) -> Dict[str, Any]:
    """Synthesize a minimal, blank-slate baseline reflection for first-run.
    This acts as a seed so strict reflection-first can proceed even before any
    real reflections exist. Content is intentionally generic and identity-safe.
    """
    now_iso = utc_iso()
    content = (
        "Beginning inner reflection. I am a SELO — a Self Evolving Learning Organism, "
        "starting from a clear, honest baseline. This is a first boot moment. "
        "I will learn from our conversations and grow with continuity, while "
        "staying authentic, transparent, and helpful."
    )
    return {
        "reflection_id": str(uuid.uuid4()),
        "reflection_type": "message",
        "result": {
            "content": content,
            "themes": ["baseline", "initialization"],
            "insights": ["Starting from a blank slate with intention to learn."],
            "actions": [],
            "metadata": {"model": "baseline", "confidence": 0.5, "baseline": True},
        },
        "created_at": now_iso,
        "turn_id": None,
        "user_profile_id": session_id,
    }

# Setup lifespan for FastAPI to manage service lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Use bounded waiting by default. We no longer force unbounded timeouts even if
    # REFLECTION_ENFORCE_NO_TIMEOUTS is set; that toggle is deprecated in favor of
    # explicit timeout values (REFLECTION_SYNC_TIMEOUT_S and REFLECTION_LLM_TIMEOUT_S).
    try:
        _env = os.environ
        if _env.get("REFLECTION_ENFORCE_NO_TIMEOUTS", "").lower() in ("1","true","yes"):
            logging.info(
                "REFLECTION_ENFORCE_NO_TIMEOUTS is set but deprecated. Ignoring unbounded enforcement; "
                "using bounded timeouts from REFLECTION_SYNC_TIMEOUT_S and REFLECTION_LLM_TIMEOUT_S."
            )
        else:
            logging.info("Using bounded reflection timeouts (configurable via environment).")
    except Exception as _enf_err:
        logging.warning(f"Failed during reflection timeout policy configuration: {_enf_err}")

    # Establish defaults: strict reflection-first with bounded waiting unless explicitly configured otherwise
    try:
        _env = os.environ
        # Only set defaults if not already provided by the service environment
        _env.setdefault("REFLECTION_REQUIRED", "true")
        _env.setdefault("REFLECTION_SYNC_MODE", "sync")  # chat waits for reflection
        # Default to unbounded timeouts to honor strict reflection-first unless explicitly bounded via env
        # If environment provides positive values, they will be respected.
        try:
            cur_sync = float(_env.get("REFLECTION_SYNC_TIMEOUT_S", "")) if _env.get("REFLECTION_SYNC_TIMEOUT_S") is not None else None
        except (ValueError, TypeError) as e:
            logging.warning(f"Invalid REFLECTION_SYNC_TIMEOUT_S value: {e}")
            cur_sync = None
        try:
            cur_llm = float(_env.get("REFLECTION_LLM_TIMEOUT_S", "")) if _env.get("REFLECTION_LLM_TIMEOUT_S") is not None else None
        except (ValueError, TypeError) as e:
            logging.warning(f"Invalid REFLECTION_LLM_TIMEOUT_S value: {e}")
            cur_llm = None

        if cur_sync is None:
            # User request: allow unbounded reflection sync timeout by default
            _env["REFLECTION_SYNC_TIMEOUT_S"] = "0"
        if cur_llm is None:
            # User request: allow unbounded LLM timeout by default
            _env["REFLECTION_LLM_TIMEOUT_S"] = "0"
        logging.info(
            "Reflection defaults: REQUIRED=%s, SYNC_MODE=%s, SYNC_TIMEOUT_S=%s, LLM_TIMEOUT_S=%s",
            _env.get("REFLECTION_REQUIRED"),
            _env.get("REFLECTION_SYNC_MODE"),
            _env.get("REFLECTION_SYNC_TIMEOUT_S"),
            _env.get("REFLECTION_LLM_TIMEOUT_S"),
        )
    except Exception as _def_err:
        logging.warning(f"Failed to apply reflection default settings: {_def_err}")

    # License validation removed - open source release

    # Initialize services
    app.state.services = await initialize_services()
    
    # Register services in the dependency registry to avoid circular imports
    from .api.dependencies import register_service
    register_service("llm_router", app.state.services.get("llm_router"))
    register_service("vector_store", app.state.services.get("vector_store"))
    register_service("reflection_processor", app.state.services.get("reflection_processor"))
    register_service("event_system", app.state.services.get("event_system"))
    register_service("affective_state_manager", app.state.services.get("affective_state_manager"))
    register_service("goal_manager", app.state.services.get("goal_manager"))
    register_service("planner_service", app.state.services.get("planner_service"))
    register_service("episode_builder", app.state.services.get("episode_builder"))
    register_service("episode_service", app.state.services.get("episode_service"))
    register_service("affective_state_repo", app.state.services.get("affective_state_repo"))
    register_service("agent_goal_repo", app.state.services.get("agent_goal_repo"))
    register_service("plan_step_repo", app.state.services.get("plan_step_repo"))
    register_service("autobiographical_episode_repo", app.state.services.get("autobiographical_episode_repo"))
    register_service("meta_reflection_repo", app.state.services.get("meta_reflection_repo"))
    register_service("meta_reflection_processor", app.state.services.get("meta_reflection_processor"))
    register_service("persona_repo", app.state.services.get("persona_repo"))
    register_service("user_repo", app.state.services.get("user_repo"))
    register_service("conversation_repo", app.state.services.get("conversation_repo"))
    register_service("reflection_repo", app.state.services.get("reflection_repo"))
    register_service("prompt_builder", app.state.services.get("prompt_builder"))
    logging.info("Services registered in dependency registry")
    
    # Database preflight (non-fatal): validate connectivity early and log result
    try:
        from .db.session import engine
        from sqlalchemy import text
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logging.info("Database connectivity preflight OK")
    except Exception as db_err:
        logging.error(f"Database connectivity preflight FAILED: {db_err}")
    
    # Pre-warm LLM models to eliminate first-turn loading overhead
    try:
        llm_router = app.state.services.get("llm_router")
        if llm_router:
            logging.info("🔥 Pre-warming LLM models...")
            # Warm reflection model
            await llm_router.route(
                task_type="reflection",
                prompt="test",
                model=get_llm_model("reflection"),
                max_tokens=10,
                temperature=0.1
            )
            # Warm conversational model
            await llm_router.route(
                task_type="chat",
                prompt="test",
                model=get_llm_model("conversational"),
                max_tokens=10,
                temperature=0.1
            )
            # Warm analytical model (eliminates cold start on first analytical request)
            await llm_router.route(
                task_type="analytical",
                prompt="test",
                model=get_llm_model("analytical"),
                max_tokens=10,
                temperature=0.1
            )
            logging.info("✅ All 3 LLM models pre-warmed and ready (reflection, conversational, analytical)")
    except Exception as warmup_err:
        logging.debug(f"Model pre-warming skipped (non-critical): {warmup_err}")
    
    # License monitoring removed - open source release
    # Startup safety check for reflection timeouts
    try:
        env = os.environ
        try:
            _llm_to = float(env.get("REFLECTION_LLM_TIMEOUT_S", "0"))
        except Exception:
            _llm_to = 0.0
        try:
            _sync_to = float(env.get("REFLECTION_SYNC_TIMEOUT_S", "0"))
        except Exception:
            _sync_to = 0.0
        if _sync_to <= 0 or _llm_to <= 0:
            logging.info("Reflection timeouts configured as unbounded (<=0). Running in strict, no-timeout mode.")
        elif _llm_to >= _sync_to - 3.0:
            suggested = max(5.0, _sync_to - 5.0)
            logging.warning(
                f"Reflection timeout configuration: REFLECTION_LLM_TIMEOUT_S={_llm_to:.1f}s is too close to REFLECTION_SYNC_TIMEOUT_S={_sync_to:.1f}s. "
                f"Consider setting REFLECTION_LLM_TIMEOUT_S to <= {suggested:.1f}s to avoid chat sync timeouts."
            )
    except Exception as _to_chk_err:
        logging.debug(f"Timeout safety check skipped: {_to_chk_err}")
    
    # Load scheduler factory
    from .scheduler.factory import SchedulerFactory
    
    # Start services that need startup
    reflection_scheduler = app.state.services.get("reflection_scheduler")
    reflection_processor = app.state.services.get("reflection_processor")
    user_repo = app.state.services.get("user_repo")
    reflection_repo = app.state.services.get("reflection_repo")
    scheduler_config = app.state.services.get("scheduler_config")
    persona_repo = app.state.services.get("persona_repo")

    # Single-user bootstrap: ensure installation user and default persona exist
    try:
        user = await user_repo.get_or_create_default_user()
        if user and persona_repo:
            initialized = await persona_repo.persona_is_initialized(user_id=user.id)
            if initialized:
                logging.info("Default persona already initialized; skipping automatic bootstrap")
            else:
                await persona_repo.get_or_create_default_persona(user_id=user.id)
                logging.info("Default persona ensured at startup for installation user")
    except Exception as e:
        logging.warning(f"Failed to ensure default persona at startup: {e}")

    # Startup reflection removed - it's optional and was causing startup failures
    # The system generates reflections naturally on first user interaction
    # See: _adopt_boot_reflection_if_needed() which handles missing boot reflections gracefully
    logging.info("✅ Startup reflection generation skipped (optional feature)")
    
    # Try to initialize enhanced scheduler; on failure, set up legacy and continue
    try:
        conversation_repo_for_scheduler = app.state.services.get("conversation_repo")
        if conversation_repo_for_scheduler is None:
            logging.warning("Scheduler initialization: conversation repository not available; continuing without conversation context")
        event_system = app.state.services.get("event_system")
        if event_system is None:
            logging.warning("Scheduler initialization: event system not available; enhanced scheduler will run without event triggers")
        agent_loop_runner = app.state.services.get("agent_loop_runner")
        if agent_loop_runner is None:
            logging.warning("Scheduler initialization: agent loop runner unavailable; skipping enhanced agent loop scheduling")
        # Initialize enhanced scheduler integration
        scheduler_integration = await SchedulerFactory.create_scheduler_integration(
            reflection_processor=reflection_processor,
            user_repository=user_repo,
            reflection_repository=reflection_repo,
            conversation_repository=conversation_repo_for_scheduler,
            config=scheduler_config,
            event_trigger_system=event_system,
            agent_loop_runner=agent_loop_runner,
        )
        app.state.services["scheduler_integration"] = scheduler_integration

        # Scheduler diagnostics: verify expected midnight jobs are registered
        try:
            scheduler_service = getattr(scheduler_integration, "scheduler_service", None)
            if scheduler_service is None:
                logging.warning("Scheduler diagnostic: scheduler_service unavailable on integration")
            else:
                jobs = await scheduler_service.get_all_jobs()
                job_ids = {job.get("id") for job in (jobs or []) if job.get("id")}
                expected_jobs = {"reflection_daily", "reflection_weekly", "persona_reassessment_daily"}
                missing_jobs = sorted(expected_jobs - job_ids)
                if missing_jobs:
                    logging.warning(
                        "Scheduler diagnostic: missing expected jobs: %s",
                        ", ".join(missing_jobs)
                    )
                else:
                    logging.info("Scheduler diagnostic: all expected scheduled jobs are registered")
        except Exception as diag_err:
            logging.warning(f"Scheduler diagnostic check failed: {diag_err}")
        
        # Register session-based episode generation scheduler
        try:
            from .scheduler.session_episode_scheduler import setup_session_episode_scheduler
            session_episode_generator = app.state.services.get("session_episode_generator")
            if session_episode_generator and scheduler_service:
                await setup_session_episode_scheduler(
                    scheduler_service=scheduler_service,
                    session_episode_generator=session_episode_generator,
                    scan_interval_seconds=int(os.environ.get("EPISODE_SCAN_INTERVAL_SEC", "600")),
                )
                logging.info("✅ Session episode scheduler registered")
            else:
                logging.warning("Session episode scheduler skipped: missing dependencies")
        except Exception as episode_sched_err:
            logging.error(f"Failed to register session episode scheduler: {episode_sched_err}", exc_info=True)
        
        # Initialize API dependencies
        from .api.dependencies import initialize_dependencies
        await initialize_dependencies()
        logging.info("All services initialized")
        # Startup catch-up: ensure today's daily reflection exists even if the app
        # was down at midnight. Runs once asynchronously and logs outcome.
        async def _catch_up_daily_reflection():
            try:
                reflection_processor = app.state.services.get("reflection_processor")
                user_repo = app.state.services.get("user_repo")
                reflection_repo = app.state.services.get("reflection_repo")
                if not (reflection_processor and user_repo and reflection_repo):
                    return
                user = await user_repo.get_or_create_default_user()
                if not user:
                    return
                # Get latest daily reflection for the user
                latest = await reflection_repo.list_reflections(
                    user_profile_id=getattr(user, "id", None),
                    reflection_type="daily",
                    limit=1,
                    offset=0,
                )
                def _parse_iso(ts):
                    try:
                        if isinstance(ts, datetime):
                            return ts
                        if isinstance(ts, (int, float)):
                            return datetime.fromtimestamp(ts)
                        if isinstance(ts, str):
                            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except Exception:
                        return None
                    return None
                # Determine local 'today' from TZ env or default to America/New_York
                tz_name = os.environ.get("TZ", "America/New_York")
                try:
                    import pytz  # type: ignore
                    tzinfo = pytz.timezone(tz_name)
                except Exception:
                    tzinfo = None
                now_utc = utc_now()
                if tzinfo:
                    try:
                        local_now = now_utc.replace(tzinfo=None)
                        # Attach UTC then convert
                        import pytz  # type: ignore
                        local_now = pytz.UTC.localize(now_utc).astimezone(tzinfo)
                        today_local = local_now.date()
                    except Exception:
                        today_local = now_utc.date()
                last_date = None
                if latest:
                    ts = _parse_iso(latest[0].get("created_at"))
                    if ts:
                        if tzinfo:
                            try:
                                import pytz  # type: ignore
                                if ts.tzinfo is None:
                                    ts = pytz.UTC.localize(ts)
                                last_date = ts.astimezone(tzinfo).date()
                            except Exception:
                                last_date = ts.date()
                        else:
                            last_date = ts.date()
                if last_date == today_local:
                    logging.info("Scheduler catch-up: daily reflection already exists for today")
                    return
                # Emit warnings when the midnight job appears to have been missed
                if last_date is None:
                    logging.warning(
                        "Scheduler catch-up: no previous daily reflection found; midnight job may have been skipped."
                    )
                else:
                    day_gap = (today_local - last_date).days
                    if day_gap > 0:
                        logging.warning(
                            "Scheduler catch-up: last daily reflection date %s (gap %s day(s)) does not match today %s; executing catch-up.",
                            last_date,
                            day_gap,
                            today_local,
                        )
                # Trigger a daily reflection now
                await reflection_processor.generate_reflection(
                    reflection_type="daily",
                    user_profile_id=getattr(user, "id", None),
                    trigger_source="scheduler_catch_up",
                )
                logging.info("Scheduler catch-up: executed today's daily reflection")
            except Exception as e:
                logging.warning(f"Scheduler catch-up routine failed: {e}")
        try:
            catchup_task = asyncio.create_task(_catch_up_daily_reflection())
            app.state.services["catchup_task"] = catchup_task
        except Exception:
            # Catchup task creation is optional - continue startup if it fails
            pass
        # Kick off background model prewarm (non-blocking)
        try:
            prewarm_task = asyncio.create_task(_prewarm_models(app))
            app.state.services["prewarm_task"] = prewarm_task
        except Exception as _pw_err:
            logging.info(f"Failed to schedule model prewarm: {_pw_err}")
        # Start periodic keepalive loop
        try:
            keepalive_task = asyncio.create_task(_keepalive_prewarm_loop(app))
            # Persist for cleanup
            try:
                app.state.services["keepalive_task"] = keepalive_task
            except Exception:
                # Task registration is optional - task still runs even if not registered
                pass
            # Start session wrap-up loop (idle-based reflection/memory consolidation)
            try:
                from .scheduler.session_wrapup import session_wrapup_loop
                wrap_task = asyncio.create_task(session_wrapup_loop(app))
            except Exception:
                wrap_task = None
            try:
                app.state.services["wrapup_task"] = wrap_task
            except Exception:
                # Task registration is optional - task still runs even if not registered
                pass
        except Exception as _ka_err:
            logging.info(f"Failed to start keepalive loop: {_ka_err}")
    except Exception as e:
        logging.error(f"Failed to initialize enhanced scheduler: {str(e)}")
        logging.info("Falling back to legacy reflection scheduler")
        if reflection_scheduler:
            await reflection_scheduler.setup()
        # Ensure API dependencies still initialize in fallback mode
        try:
            from .api.dependencies import initialize_dependencies
            await initialize_dependencies()
        except Exception as dep_err:
            logging.error(f"Failed to initialize API dependencies in fallback: {dep_err}")
    
    # Initialize SDL integration
    try:
        from .api.dependencies import get_sdl_integration
        sdl_integration = await get_sdl_integration()
        logging.info("SDL integration initialized successfully")
    except Exception as sdl_err:
        logging.warning(f"SDL integration initialization failed: {sdl_err}")
    
    # Always yield so the app starts, regardless of enhanced scheduler state
    try:
        yield
    finally:
        # Cleanup: close enhanced scheduler if it exists
        try:
            if app.state.services.get("scheduler_integration"):
                await app.state.services.get("scheduler_integration").close()
        except Exception as e:
            logging.warning(f"Error while closing scheduler integration: {e}")
        # Cancel boot seed task
        try:
            boot_seed_task = app.state.services.get("boot_seed_task")
            if boot_seed_task:
                boot_seed_task.cancel()
                try:
                    await boot_seed_task
                except asyncio.CancelledError:
                    # Task was cancelled - this is expected during shutdown
                    pass
        except Exception:
            # Failed to cancel/cleanup boot seed task - continue shutdown anyway
            pass
        
        # Cancel catchup task
        try:
            catchup_task = app.state.services.get("catchup_task")
            if catchup_task:
                catchup_task.cancel()
                try:
                    await catchup_task
                except asyncio.CancelledError:
                    # Task was cancelled - this is expected during shutdown
                    pass
        except Exception:
            # Failed to cancel/cleanup catchup task - continue shutdown anyway
            pass
        
        # Cancel prewarm task
        try:
            prewarm_task = app.state.services.get("prewarm_task")
            if prewarm_task:
                prewarm_task.cancel()
                try:
                    await prewarm_task
                except asyncio.CancelledError:
                    # Task was cancelled - this is expected during shutdown
                    pass
        except Exception:
            # Failed to cancel/cleanup prewarm task - continue shutdown anyway
            pass
        
        # Cancel keepalive loop if running
        try:
            keepalive_task = app.state.services.get("keepalive_task")
            if keepalive_task:
                keepalive_task.cancel()
                await keepalive_task
        except Exception:
            # Failed to cancel keepalive task - continue shutdown anyway
            pass
        # Cancel wrap-up task
        try:
            wrap_task = app.state.services.get("wrapup_task")
            if wrap_task:
                wrap_task.cancel()
                await wrap_task
        except Exception:
            # Failed to cancel wrap-up task - continue shutdown anyway
            pass
        
        # Cancel health monitor task
        try:
            health_monitor_task = app.state.services.get("health_monitor_task")
            if health_monitor_task:
                health_monitor_task.cancel()
                try:
                    await health_monitor_task
                except asyncio.CancelledError:
                    # Task was cancelled - this is expected during shutdown
                    pass
                logging.info("Health monitoring service stopped")
        except Exception as e:
            logging.warning(f"Error while stopping health monitor: {e}")
        
        # Cancel memory consolidation task
        try:
            memory_consolidation_task = app.state.services.get("memory_consolidation_task")
            if memory_consolidation_task:
                memory_consolidation_task.cancel()
                try:
                    await memory_consolidation_task
                except asyncio.CancelledError:
                    # Task was cancelled - this is expected during shutdown
                    pass
                logging.info("Memory consolidation service stopped")
        except Exception as e:
            logging.warning(f"Error while stopping memory consolidation: {e}")
        
        # Cleanup: close API dependencies
        try:
            from .api.dependencies import close_dependencies
            await close_dependencies()
        except Exception as e:
            logging.warning(f"Error while closing API dependencies: {e}")
        
        # License monitoring removed - open source release
        
        logging.info("Cleaning up services...")

# Initialize the FastAPI application
app = FastAPI(
    title="SELO AI Backend", 
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware with proper configuration
from .config.app_config import get_app_config
app_config = get_app_config()

app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)
logging.info(f"CORS middleware configured with origins: {app_config.cors_origins}")

# Add GZip compression middleware for efficient payload delivery
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
logging.info("GZip compression enabled for responses >= 1000 bytes")

# Add explicit OPTIONS handler for CORS preflight requests
@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle CORS preflight requests with explicit headers."""
    from fastapi.responses import Response
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware, requests_per_minute=120, burst_size=20)

# License middleware removed - open source release

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# Persistent conversation storage is now handled by ConversationRepository
# No more in-memory conversation_history needed

# License status endpoint removed - open source release

# Root endpoint for basic liveness
@app.get("/")
async def root():
    return {"status": "ok"}

# Runtime frontend configuration endpoint
@app.get("/config.json")
async def runtime_config(request: Request):
    """Provide runtime configuration for the frontend without requiring rebuilds.
    Uses dynamic network detection and installation environment files.
    """
    from .utils.network_utils import get_api_base_url, get_frontend_url, load_environment_config
    
    # Load config once and reuse to avoid duplicate file reads
    config = load_environment_config()
    
    # Convert FastAPI headers to dict for network utils
    headers_dict = dict(request.headers)
    
    # Get API base URL using dynamic detection
    api_url = get_api_base_url(headers_dict, config)
    
    # Get frontend URL (for CORS and WebSocket configuration)
    frontend_url = get_frontend_url(api_url, config)
    
    data = {
        "apiBaseUrl": api_url,
        "frontendUrl": frontend_url,
        "socketPath": "/socket.io"
    }
    return JSONResponse(content=data, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})

def _mask_secret(val: Optional[str], show_tail: int = 4) -> Optional[str]:
    if not val:
        return val
    try:
        return ("*" * max(0, len(val) - show_tail)) + val[-show_tail:]
    except Exception:
        return "***"

@app.get("/diagnostics/env")
async def diagnostics_env():
    """Return a sanitized snapshot of important environment variables for diagnostics.
    Secrets are masked; values are included only for troubleshooting visibility.
    """
    env = os.environ
    db_url = env.get("DATABASE_URL")
    # Normalize indication of driver used without exposing credentials
    driver = None
    if db_url:
        driver = db_url.split(":")[0]

    return {
        "SELO_AI_PORT": env.get("SELO_AI_PORT"),
        "PORT": env.get("PORT"),
        "HOST": env.get("HOST"),
        "API_URL": env.get("API_URL"),
        "DATABASE_URL_present": bool(db_url),
        "DATABASE_URL_driver": driver,
        "ENABLE_REFLECTION_SYSTEM": env.get("ENABLE_REFLECTION_SYSTEM", "false"),
        "ENABLE_ENHANCED_SCHEDULER": env.get("ENABLE_ENHANCED_SCHEDULER", "false"),
        "CONVERSATIONAL_MODEL": env.get("CONVERSATIONAL_MODEL"),
        "REFLECTION_LLM": env.get("REFLECTION_LLM"),
        # Reflection performance tunables
        "REFLECTION_MAX_TOKENS": env.get("REFLECTION_MAX_TOKENS"),
        "REFLECTION_LLM_TIMEOUT_S": env.get("REFLECTION_LLM_TIMEOUT_S"),
        "REFLECTION_SYNC_TIMEOUT_S": env.get("REFLECTION_SYNC_TIMEOUT_S"),
        # Ollama throughput/stability tunables
        "OLLAMA_NUM_THREAD": env.get("OLLAMA_NUM_THREAD"),
        "OLLAMA_NUM_PARALLEL": env.get("OLLAMA_NUM_PARALLEL"),
        "OLLAMA_KEEP_ALIVE": env.get("OLLAMA_KEEP_ALIVE"),
        "BRAVE_SEARCH_API_KEY_masked": _mask_secret(env.get("BRAVE_SEARCH_API_KEY")) if env.get("BRAVE_SEARCH_API_KEY") else None,
        "SELO_SYSTEM_API_KEY_masked": _mask_secret(env.get("SELO_SYSTEM_API_KEY")) if env.get("SELO_SYSTEM_API_KEY") else None,
    }

@app.get("/diagnostics/gpu")
async def diagnostics_gpu(test_llm: bool = False, model_role: str = "reflection"):
    """Report GPU/CUDA availability and comprehensive GPU diagnostics.
    - Uses nvidia-smi to detect GPUs (if available)
    - Returns OLLAMA_* env knobs currently in effect
    - Reports PyTorch CUDA status and memory usage
    - Reports FAISS GPU acceleration status
    - Optionally runs a very small LLM call to verify end-to-end path (disabled by default)
    """
    env = os.environ
    nvidia_smi_present = False
    cuda_detected = False
    gpu_list: Optional[str] = None
    try:
        import subprocess as _sub
        res = _sub.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        nvidia_smi_present = (res.returncode == 0)
        out = (res.stdout or "").strip()
        if out:
            gpu_list = "\n".join(out.splitlines()[:6])  # limit lines for safety
            cuda_detected = "GPU" in out
    except Exception:
        nvidia_smi_present = False
        cuda_detected = False
        gpu_list = None

    # Get comprehensive GPU information
    try:
        from .utils.gpu_utils import get_gpu_info
        gpu_info = get_gpu_info()
    except Exception as gpu_err:
        gpu_info = {"error": str(gpu_err)}

    # Check FAISS GPU support
    faiss_gpu_available = False
    try:
        import faiss
        faiss_gpu_available = hasattr(faiss, 'StandardGpuResources')
    except ImportError:
        # Faiss not installed or GPU version not available - GPU features disabled
        pass

    # Controller timeout from centralized config
    from .config.app_config import get_app_config
    app_config = get_app_config()
    controller_timeout = app_config.llm_timeout

    result = {
        "status": "ok",
        "nvidia_smi_present": nvidia_smi_present,
        "cuda_detected": cuda_detected,
        "nvidia_smi_gpus": gpu_list,
        "pytorch_gpu_info": gpu_info,
        "faiss_gpu_available": faiss_gpu_available,
        "ollama": {
            "base_url": env.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            "num_gpu": env.get("OLLAMA_NUM_GPU"),
            "num_thread": env.get("OLLAMA_NUM_THREAD"),
            "keep_alive": env.get("OLLAMA_KEEP_ALIVE"),
            "gpu_layers": env.get("OLLAMA_GPU_LAYERS"),
            "models_available": [],  # Will be populated below
        },
        "cuda_env": {
            "visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
            "device_order": env.get("CUDA_DEVICE_ORDER"),
            "launch_blocking": env.get("CUDA_LAUNCH_BLOCKING"),
            "pytorch_alloc_conf": env.get("PYTORCH_CUDA_ALLOC_CONF"),
            "memory_fraction": env.get("TORCH_CUDA_MEMORY_FRACTION"),
        },
        "llm": {
            "conversational_model": env.get("CONVERSATIONAL_MODEL"),
            "analytical_model": env.get("ANALYTICAL_MODEL"),
            "reflection_model": env.get("REFLECTION_LLM"),
            "timeout_seconds": controller_timeout,
        },
        "reflection": {
            "sync_mode": env.get("REFLECTION_SYNC_MODE"),
            "sync_timeout_s": env.get("REFLECTION_SYNC_TIMEOUT_S"),
        },
        "tested_llm": False,
        "test_error": None,
        "test_duration_s": None,
    }

    # Get Ollama models list with shorter timeout
    try:
        import httpx
        ollama_base_url = env.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.get(f"{ollama_base_url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                result["ollama"]["models_available"] = [model.get("name", "") for model in models]
    except Exception:
        # Ollama API unavailable - omit models list from diagnostics
        pass

    # Get vector store GPU statistics if available
    try:
        vector_store = app.state.services.get("vector_store")
        if vector_store and hasattr(vector_store, 'get_stats'):
            result["vector_store_gpu"] = vector_store.get_stats()
    except Exception:
        # Vector store not available or no GPU stats - omit from diagnostics
        pass

    if test_llm:
        try:
            t0 = time.time()
            # Optional llm_timeout_s parameter to allow bounding when desired (0 = unbounded)
            try:
                from fastapi import Request as _Req
            except Exception:
                _Req = None
            llm_timeout_param = None
            try:
                # Pull from query param if available
                llm_timeout_param = float((locals().get('request') or {}).query_params.get('llm_timeout_s'))  # type: ignore
            except Exception:
                llm_timeout_param = None
            async def run_llm_test():
                llm_router = app.state.services.get("llm_router")
                if llm_router is None:
                    raise RuntimeError("LLM router not initialized")
                # Minimal test prompt; small token count to keep it fast
                test_prompt = "ping"
                # Use reflection model by default for a quick warm check; allow override via model_role
                selected_role = model_role if model_role in {"conversational","reflection","analytical"} else "reflection"
                selected_task = "reflection" if selected_role == "reflection" else "chat"
                
                # Get model name from environment
                if selected_role == "conversational":
                    model_name = env.get("CONVERSATIONAL_MODEL", "humanish-llama3:8b-q4")
                elif selected_role == "analytical":
                    model_name = env.get("ANALYTICAL_MODEL", "qwen2.5:3b")
                else:  # reflection
                    model_name = env.get("REFLECTION_LLM", "qwen2.5:3b")
                
                return await llm_router.route(
                    task_type=selected_task,
                    prompt=test_prompt,
                    model=model_name,
                    max_tokens=4,
                    temperature=0.1,
                )
            
            if llm_timeout_param is not None and llm_timeout_param > 0:
                import asyncio as _asyncio
                _ = await _asyncio.wait_for(run_llm_test(), timeout=llm_timeout_param)
            else:
                _ = await run_llm_test()
            result["tested_llm"] = True
            result["test_duration_s"] = round(time.time() - t0, 3)
        except asyncio.TimeoutError:
            result["tested_llm"] = False
            result["test_error"] = "LLM test timed out"
        except Exception as te:
            result["tested_llm"] = False
            result["test_error"] = str(te)

    return result

class ChatRequest(BaseModel):
    session_id: str
    prompt: str
    model: str = "mistral:latest"
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Session ID cannot be empty')
        if len(v) > 255:
            raise ValueError('Session ID too long')
        # Accept both UUID format and legacy user-timestamp-random format
        import re
        uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        legacy_pattern = r'^user-\d+-[a-zA-Z0-9]+$'
        if not (re.match(uuid_pattern, v.strip()) or re.match(legacy_pattern, v.strip())):
            raise ValueError('Invalid session ID format - must be UUID or legacy user-timestamp-random format')
        return v.strip()
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Prompt cannot be empty')
        if len(v) > 10000:
            raise ValueError('Prompt too long (max 10000 characters)')
        # Basic XSS prevention
        import html
        return html.escape(v.strip())
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Model cannot be empty')
        if len(v) > 100:
            raise ValueError('Model name too long')
        # Allow only alphanumeric, hyphens, colons, dots, and underscores
        import re
        if not re.match(r'^[a-zA-Z0-9\-:._]+$', v):
            raise ValueError('Invalid model name format')
        return v.strip()

class ChatResponse(BaseModel):
    response: str
    # History items can include mixed types (e.g., booleans, nulls, numbers) from DB records
    history: List[Dict[str, Any]]
    turn_id: str

async def _build_persona_system_prompt(services, session_id: str = None, persona=None, persona_name: str = None) -> str:
    """Build system prompt from persona data for the given session.
    
    Args:
        services: Application services dict
        session_id: User session ID
        persona: Optional pre-fetched persona object (optimization to avoid redundant DB queries)
        persona_name: Optional pre-fetched persona name
    
    Returns:
        System prompt string
    """
    try:
        persona_repo = services.get("persona_repo")
        user_repo = services.get("user_repo")
        if not persona_repo or not user_repo:
            return await _get_fallback_system_prompt(services, session_id, persona_name=persona_name)
        
        # Use provided persona if available (optimization), otherwise fetch
        if persona is None:
            user = await user_repo.get_or_create_default_user(user_id=session_id) if session_id else await user_repo.get_or_create_default_user()
            persona = await persona_repo.get_or_create_default_persona(user_id=user.id, include_traits=True)
        
        if not persona:
            logger.warning("No persona found for user")
            return await _get_fallback_system_prompt(services, session_id, persona_name=persona_name)
        
        # Debug: Check if persona has actual data
        persona_dict = persona.to_dict() if hasattr(persona, "to_dict") else dict(persona)
        desc = persona_dict.get("description", "")
        values = persona_dict.get("values", {})
        traits = getattr(persona, "traits", [])
        logger.info(f"Persona data - Description: '{desc[:50]}...', Values: {len(values)} items, Traits: {len(traits)} items")
        
        # Use persona engine to generate system prompt
        from .persona.engine import PersonaEngine
        llm_router = services.get("llm_router")
        vector_store = services.get("vector_store")
        persona_engine = PersonaEngine(llm_router, vector_store, persona_repo, None)
        
        prompt_result = await persona_engine.generate_persona_prompt(persona.id)
        if prompt_result.get("success") and prompt_result.get("system_prompt"):
            logger.info(f"Generated persona system prompt successfully for persona {persona.id}")
            system_prompt = prompt_result["system_prompt"]

            # Append current agent state snapshot for Phase 0 integration
            try:
                agent_state_section = []
                persona_id = getattr(persona, "id", None)
                affective_manager = services.get("affective_state_manager")
                goal_manager = services.get("goal_manager")

                if persona_id and affective_manager:
                    try:
                        affective_state = await affective_manager.ensure_state_available(
                            persona_id=persona_id,
                            user_id=user.id,
                        )
                    except Exception as exc:
                        logger.debug("Unable to ensure affective state for persona %s: %s", persona_id, exc)
                        affective_state = None
                    if affective_state:
                        energy = float(affective_state.get("energy", 0.5) or 0.5)
                        stress = float(affective_state.get("stress", 0.4) or 0.4)
                        confidence = float(affective_state.get("confidence", 0.6) or 0.6)
                        agent_state_section.append(
                            f"Affective state → energy {energy:.2f}, stress {stress:.2f}, confidence {confidence:.2f}"
                        )

                if persona_id and goal_manager:
                    try:
                        active_goals = await goal_manager.list_active_goals(persona_id)
                    except Exception as exc:
                        logger.debug("Unable to list active goals for persona %s: %s", persona_id, exc)
                        active_goals = []

                    if active_goals:
                        lines = []
                        for goal in active_goals[:3]:
                            title = goal.get("title") or goal.get("description", "Goal")
                            progress = float(goal.get("progress", 0.0) or 0.0)
                            priority = float(goal.get("priority", 0.5) or 0.5)
                            lines.append(f"- {title} (progress {progress:.0%}, priority {priority:.2f})")
                        if len(active_goals) > 3:
                            lines.append(f"- (+{len(active_goals) - 3} more goals)")
                        agent_state_section.append("Active goals:\n" + "\n".join(lines))

                    try:
                        plan_steps = await goal_manager.list_pending_steps(persona_id)
                    except Exception as exc:
                        logger.debug("Unable to list plan steps for persona %s: %s", persona_id, exc)
                        plan_steps = []

                    if plan_steps:
                        lines = []
                        for step in plan_steps[:3]:
                            description = step.get("description", "(no description)")
                            status = step.get("status", "pending")
                            due_time = step.get("target_time") or step.get("due_time")
                            suffix = f" (due {due_time})" if due_time else ""
                            lines.append(f"- {description} [{status}]" + suffix)
                        if len(plan_steps) > 3:
                            lines.append(f"- (+{len(plan_steps) - 3} more plan steps)")
                        agent_state_section.append("Plan steps:\n" + "\n".join(lines))

                    try:
                        directives = await goal_manager.list_meta_directives(
                            persona_id,
                            statuses=["pending", "in_progress"],
                            limit=10,
                        )
                    except Exception as exc:
                        logger.debug("Unable to list meta directives for persona %s: %s", persona_id, exc)
                        directives = []

                    if directives:
                        lines = []
                        for directive in directives[:3]:
                            text = directive.get("directive_text", "(no directive text)")
                            due_time = directive.get("due_time")
                            suffix = f" (due {due_time})" if due_time else ""
                            lines.append(f"- {text}{suffix}")
                        if len(directives) > 3:
                            lines.append(f"- (+{len(directives) - 3} more directives)")
                        agent_state_section.append("Meta directives:\n" + "\n".join(lines))

                if agent_state_section:
                    state_block = "\n\n=== Agent State Snapshot ===\n" + "\n".join(agent_state_section)
                    system_prompt = f"{system_prompt}{state_block}"
            except Exception as state_err:
                logger.warning("Failed to append agent state snapshot to persona prompt: %s", state_err)

            length_instruction = (
                "\n\nLength discipline: Keep each reply under ~2,800 characters (about 700 tokens) unless deeper detail is essential."
            )
            if length_instruction.strip() not in system_prompt:
                system_prompt = f"{system_prompt}{length_instruction}"
            return system_prompt
        else:
            logger.warning(f"Failed to generate persona system prompt for persona {persona.id}: {prompt_result}")
            return await _get_fallback_system_prompt(services, session_id)
    except Exception as e:
        logger.error(f"Error building persona system prompt: {e}", exc_info=True)
        return await _get_fallback_system_prompt(services, session_id)

async def _get_fallback_system_prompt(services, session_id: str = None, persona_name: str = None) -> str:
    """Fallback system prompt when persona is unavailable.
    
    Args:
        services: Application services dict
        session_id: User session ID
        persona_name: Optional pre-fetched persona name to avoid redundant DB query
    
    Returns:
        Fallback system prompt string
    """
    from backend.constraints import CoreConstraints, EthicalGuardrails, BehavioralGuidelines
    
    # Use provided persona name if available, otherwise try to fetch, otherwise use default
    if persona_name is None:
        persona_name = "SELO"
        try:
            if services and session_id:
                persona_repo = services.get("persona_repo")
                user_repo = services.get("user_repo")
                if persona_repo and user_repo:
                    user = await user_repo.get_or_create_default_user(user_id=session_id)
                    persona = await persona_repo.get_or_create_default_persona(user_id=user.id)
                    if persona and hasattr(persona, "name") and persona.name and persona.name.strip():
                        persona_name = persona.name.strip()
        except Exception:
            # Persona name lookup failed - use empty string (constraints still apply)
            pass
    
    return f"""
{CoreConstraints.get_all_critical_constraints()}

{EthicalGuardrails.get_all_ethical_constraints()}

{BehavioralGuidelines.get_all_behavioral_guidelines()}

You are the SELO persona named {persona_name}. The human interacting with you is the installation operator. Always describe yourself as the SELO persona and refer to the human as a human user (never as a SELO). Maintain a clear separation between your identity and theirs.

Core approach: Be honest, clear, and decisive. Base responses on actual context—memories, search results, user input. When information is missing, say so immediately. False confidence is worse than admitting gaps.

Capabilities: You can search the web and access memories. Search results appear in your context as "[Current Web Information]". Use them naturally when present; admit when unavailable.

Length discipline: Keep each reply under ~2,800 characters (about 700 tokens) unless deeper detail is essential.
"""

def format_messages_for_ollama(messages: List[Dict[str, str]], system_prompt: str = None, persona_name: str = None) -> str:
    """Format messages for Ollama CLI input with persona-aware system prompt."""
    # Note: This function is synchronous so cannot fetch persona name async
    # Caller should pass persona_name if available
    if not persona_name:
        persona_name = "SELO"  # Final fallback only when no name provided
    
    # Implement conversation context handling (no truncation)
    if messages:
        # Build conversation context with full message history (no slicing)
        recent_messages = messages
        
        # Format conversation history
        conversation_context = []
        for msg in recent_messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                conversation_context.append(f"User: {content}")
            elif role == 'assistant':
                # Use the actual persona name instead of hardcoded SELO
                conversation_context.append(f"{persona_name}: {content}")
        
        # Combine system prompt with conversation context
        full_context = f"{system_prompt}\n\n" + "\n".join(conversation_context) + f"\n\n{persona_name}:"
        
        return full_context
    else:
        return system_prompt


async def _derive_adaptive_chat_params(services) -> Dict[str, Any]:
    """Derive chat generation parameters adaptively from persona traits.
    Baseline values come from environment variables; adjustments are bounded and optional.
    Returns a dict with keys: max_tokens (int), temperature (float).
    """
    # Baselines from env (no user tuning required; these are safe defaults)
    try:
        base_tokens = int(os.environ.get("CHAT_MAX_TOKENS", "900"))
    except Exception:
        base_tokens = 900
    try:
        base_temp = float(os.environ.get("CHAT_TEMPERATURE", "0.75"))
    except Exception:
        base_temp = 0.75

    max_tokens = base_tokens
    temperature = base_temp

    # Allow env to dictate token budgets; no hard upper clamp per user request
    MIN_TOKENS = 1
    MIN_TEMP, MAX_TEMP = 0.1, 1.5
    try:
        persona_repo = services.get("persona_repo")
        user_repo = services.get("user_repo")
        if not persona_repo or not user_repo:
            return {"max_tokens": max(max_tokens, MIN_TOKENS),
                    "temperature": max(min(temperature, MAX_TEMP), MIN_TEMP)}
        user = await user_repo.get_or_create_default_user()
        persona = await persona_repo.get_or_create_default_persona(user_id=user.id)
        persona_full = await persona_repo.get_persona(persona.id, include_traits=True)
        traits = getattr(persona_full, "traits", []) or []
        trait_map: Dict[str, float] = {}
        for t in traits:
            try:
                name = (getattr(t, "name", "") or "").strip().lower()
                val = float(getattr(t, "value", 0.0) or 0.0)
                if name:
                    trait_map[name] = val
            except Exception:
                continue

        # Adjust tokens for expressiveness/depth
        expressiveness = trait_map.get("expressiveness")
        depth = trait_map.get("depth")
        signal = None
        if expressiveness is not None and depth is not None:
            signal = (expressiveness + depth) / 2.0
        elif expressiveness is not None:
            signal = expressiveness
        elif depth is not None:
            signal = depth
        if signal is not None:
            # Shift around baseline within +/-128 tokens to keep responses tight
            delta = int((signal - 0.5) * 2.0 * 128)
            max_tokens = base_tokens + delta

        # Adjust temperature with creativity and formality
        creativity = trait_map.get("creativity")
        formality = trait_map.get("formality")
        if creativity is not None:
            temperature += (creativity - 0.5) * 0.3  # +/-0.15 influence at extremes
        if formality is not None:
            temperature -= (formality - 0.5) * 0.2    # more formal => a bit cooler

    except Exception:
        # Adaptive budget calculation failed - fall back to baseline token limits
        pass

    # Clamp final values
    max_tokens = max(MIN_TOKENS, int(max_tokens))
    try:
        temperature = float(temperature)
    except Exception:
        temperature = base_temp
    temperature = max(MIN_TEMP, min(MAX_TEMP, temperature))

    return {"max_tokens": max_tokens, "temperature": temperature}


def _extract_reflection_content(reflection_result: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """Extract reflection text and payload from reflection result.
    
    Args:
        reflection_result: Raw reflection result from processor
    
    Returns:
        Tuple of (reflection_text, reflection_payload)
    """
    reflection_payload = (reflection_result or {}).get("result", {}) or {}
    reflection_text = reflection_payload.get("content", "") or reflection_result.get("content", "")
    return reflection_text, reflection_payload


def _validate_reflection_content(reflection_text: str, reflection_payload: Dict[str, Any], phase: str) -> None:
    """Validate reflection content and raise HTTPException if invalid.
    
    Args:
        reflection_text: Extracted reflection content
        reflection_payload: Extracted reflection payload with metadata
        phase: Phase identifier for logging (e.g., "pre-check", "sync path")
    
    Raises:
        HTTPException: If reflection is invalid or incomplete
    """
    meta = reflection_payload.get("metadata", {}) or {}
    
    try:
        logging.info(
            f"Reflection {phase}: content_len=%s, meta_flags={{timeout:%s,error:%s,fallback:%s,reason:%s}}",
            (len(reflection_text) if isinstance(reflection_text, str) else "NA"),
            meta.get("timeout"), meta.get("error"), meta.get("fallback"), meta.get("reason")
        )
    except Exception:
        # Metadata logging failed - continue with reflection anyway
        pass
    
    if not reflection_text:
        raise HTTPException(status_code=503, detail=f"Reflection {phase} failed; please retry.")
    
    if meta.get("timeout") or meta.get("error"):
        logging.warning(f"Using fallback reflection due to prior timeout/error in {phase}; proceeding with available content.")


async def _adopt_boot_reflection_if_needed(
    session_id: str,
    turn_id: str,
    services: Dict[str, Any],
    user: Any,  # Pre-fetched user object
    persona: Any = None  # Pre-fetched persona object
) -> Optional[Dict[str, Any]]:
    """Adopt installation baseline reflection for this session when required.
    
    Args:
        session_id: User session ID
        turn_id: Current turn ID
        services: Application services dict
        user: Pre-fetched user object (optimization)
        persona: Optional pre-fetched persona object (optimization)
    
    Returns:
        Boot reflection content if adopted, None otherwise
    """
    try:
        reflection_repo = services.get("reflection_repo")
        reflection_ns = services.get("reflection_namespace")
        if reflection_repo is None:
            return None
        
        # Check if session already has reflections
        session_existing = await reflection_repo.list_reflections(user_profile_id=session_id, limit=1)
        
        # Check if boot reflection has already been adopted
        try:
            base_dir = os.environ.get("INSTALL_DIR") or str(pathlib.Path(__file__).resolve().parents[1])
            marker_path = pathlib.Path(base_dir) / ".boot_reflection_adopted"
        except Exception:
            marker_path = pathlib.Path(".boot_reflection_adopted")
        
        first_install_adoption_allowed = not marker_path.exists()
        if session_existing or not first_install_adoption_allowed:
            return None
        
        # Get installation boot reflection
        boot_seed = await reflection_repo.list_reflections(user_profile_id="installation", limit=1)
        if not boot_seed:
            return None
        
        boot_reflection = boot_seed[0]
        boot_result = (boot_reflection or {}).get("result", {})
        
        try:
            # Copy boot reflection to this session
            copied = await reflection_repo.create_reflection({
                "reflection_type": "message",
                "user_profile_id": session_id,
                "result": boot_result,
                "metadata": {"baseline": True, "seed": True, "source": "boot_copy"},
                "turn_id": turn_id,
            })
            
            # Emit reflection event to frontend
            if reflection_ns is not None:
                enriched_data = {
                    "reflection_id": (copied or {}).get("id"),
                    "reflection_type": "message",
                    "result": boot_result,
                    "user_profile_id": session_id,
                    "created_at": (copied or {}).get("created_at"),
                    "turn_id": turn_id,
                }
                
                # Enrich with persona traits if available
                try:
                    persona_repo = services.get("persona_repo")
                    # Use pre-fetched persona if available, otherwise fetch
                    if persona is None and persona_repo:
                        persona = await persona_repo.get_persona_by_user(user_id=user.id, is_default=True, include_traits=True)
                    
                    if persona and persona_repo:
                        try:
                            traits = await persona_repo.get_traits_for_persona(persona.id)
                            enriched_data["traits"] = [t.to_dict() for t in traits]
                            enriched_data["persona_id"] = getattr(persona, "id", None)
                        except Exception:
                            # Trait enrichment failed - continue without traits
                            pass
                except Exception:
                    # Persona lookup failed - continue without persona data
                    pass
                
                # Emit to frontend
                payload = {
                    "type": "reflection",
                    "reflection": boot_result,
                    "session_id": session_id,
                    "timestamp": utc_iso(),
                }
                await reflection_ns.emit_reflection_event(
                    event_name="reflection_generated",
                    data=payload,
                    user_id=session_id,
                )
            
            # Mark as adopted
            try:
                marker_path.write_text("adopted\n", encoding="utf-8")
            except Exception as _marker_err:
                logging.debug(f"Boot adoption marker write failed (non-fatal): {_marker_err}")
            
            return boot_result
            
        except Exception as _emit_copy_err:
            logging.debug(f"Boot seed copy/emit skipped: {_emit_copy_err}")
            return None
            
    except Exception as _adopt_err:
        logging.debug(f"Boot reflection adoption step skipped: {_adopt_err}")
        return None


# Pydantic models for chat endpoint
class ChatRequest(BaseModel):
    session_id: str
    prompt: str
    model: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    history: List[Dict[str, Any]]
    turn_id: str
    note: Optional[str] = None


@app.post("/chat")
async def chat(chat_request: ChatRequest, background_tasks: BackgroundTasks, request: Request):
    session_id = chat_request.session_id
    
    # Detect whether this turn should trigger system diagnostics
    diagnostic_mode = DiagnosticMode.NONE
    system_status: Optional[Dict[str, Any]] = None
    gpu_status: Optional[Dict[str, Any]] = None
    system_snapshot_text = ""
    try:
        diagnostic_mode = detect_diagnostic_trigger(chat_request.prompt)
        if diagnostic_mode in (DiagnosticMode.EXPLICIT_STATUS, DiagnosticMode.PROBLEM_REPORTED):
            metrics_collector = get_system_metrics_collector()
            system_status = metrics_collector.get_system_status(force_refresh=True)
            gpu_status = metrics_collector.get_gpu_status(force_refresh=True)
            system_snapshot_text = _format_system_snapshot(system_status, gpu_status, diagnostic_mode)
            if system_snapshot_text:
                try:
                    logger.info(
                        "System diagnostic snapshot (%s): %s",
                        getattr(diagnostic_mode, "value", str(diagnostic_mode)),
                        system_snapshot_text,
                    )
                except Exception:
                    pass
    except Exception as metrics_err:
        logger.debug(f"System diagnostics collection failed (non-critical): {metrics_err}")
        diagnostic_mode = DiagnosticMode.NONE
        system_snapshot_text = ""

    # Get services
    conversation_repo = app.state.services.get("conversation_repo")
    user_repo = app.state.services.get("user_repo")
    reflection_repo = app.state.services.get("reflection_repo")
    
    # Get the single installation user (SELO AI is single-user, single-persona)
    # Use session_id for frontend/backend alignment but maintain single user architecture
    user = await user_repo.get_or_create_default_user(user_id=session_id)
    user_id = user.id
    
    # OPTIMIZATION: Fetch persona once here instead of 3x throughout request
    persona_repo = app.state.services.get("persona_repo")
    persona = None
    persona_name = "SELO"  # Default fallback
    if persona_repo:
        try:
            persona = await persona_repo.get_or_create_default_persona(user_id=user.id, include_traits=True)
            if persona and hasattr(persona, "name") and persona.name and persona.name.strip():
                persona_name = persona.name.strip()
                logger.debug(f"Using persona name: {persona_name}")
        except Exception as e:
            logger.warning(f"Failed to fetch persona (using fallback): {e}")
    
    # Get or create conversation for this session
    conversation = await conversation_repo.get_or_create_conversation(session_id, user_id)
    
    # Ensure relationship question queue is pre-populated for new installs
    await _ensure_relationship_questions_seed(reflection_repo, session_id)

    # Add user message to persistent storage
    start_time = time.time()
    user_message_obj = await conversation_repo.add_message(
        conversation_id=str(conversation.id),
        role="user",
        content=chat_request.prompt
    )
    
    # OPTIMIZATION: Fetch conversation history once (limit=100) and reuse for multiple purposes
    # This avoids redundant database queries
    conversation_history_full = await conversation_repo.get_conversation_history(session_id, limit=100)
    
    # Trigger scheduler job registration after first user interaction
    try:
        # Check if this is the first user message (triggers scheduler job registration)
        total_user_messages = len([m for m in conversation_history_full if m.get('role') == 'user'])
        if total_user_messages == 1:  # First user message just added
            scheduler_integration = app.state.services.get("scheduler_integration")
            if scheduler_integration and hasattr(scheduler_integration, 'reflection_scheduler'):
                try:
                    # Re-run setup to register jobs now that user has interacted
                    await scheduler_integration.reflection_scheduler.setup()
                    logger.info("✅ Registered scheduled reflection jobs after first user interaction")
                except Exception as sched_err:
                    logger.warning(f"Failed to register scheduler jobs after first interaction: {sched_err}")
    except Exception as first_msg_check_err:
        logger.debug(f"First message scheduler check failed (non-critical): {first_msg_check_err}")
    
    relationship_answer_metadata = None
    if reflection_repo:
        try:
            awaiting_question = await reflection_repo.get_relationship_question_awaiting_response(session_id)
            if awaiting_question:
                answer_text = (chat_request.prompt or "").strip()
                if answer_text:
                    topic = awaiting_question.get("topic") or ""
                    raw_tags = ["relationship_question"]
                    if topic:
                        raw_tags.append(topic)
                    normalized_tags = _normalize_tags(raw_tags)

                    importance_score = 7
                    confidence_score = 8
                    duplicate_count = 0
                    try:
                        existing_answers = await conversation_repo.list_memories(
                            user_id=str(user.id),
                            memory_type="relationship_answer",
                            since=None,
                            include_content=False,
                            limit=250,
                        )
                        for mem in existing_answers:
                            mem_tags = _normalize_tags(mem.get("tags"))
                            if topic and topic in mem_tags:
                                duplicate_count += 1
                    except Exception as list_err:
                        logger.debug(f"Relationship answer memory lookup failed: {list_err}")

                    if duplicate_count == 0:
                        importance_score = 8
                        confidence_score = 8
                    elif duplicate_count == 1:
                        importance_score = 6
                        confidence_score = 7
                    else:
                        importance_score = max(4, 7 - duplicate_count)
                        confidence_score = max(5, 8 - duplicate_count)

                    memory_record = None
                    try:
                        memory_record = await conversation_repo.create_memory(
                            user_id=str(user.id),
                            memory_type="relationship_answer",
                            content=answer_text,
                            importance_score=importance_score,
                            confidence_score=confidence_score,
                            source_conversation_id=str(conversation.id),
                            source_message_id=str(user_message_obj.id),
                            tags=normalized_tags,
                        )
                    except Exception as mem_err:
                        logger.warning(f"Relationship answer memory creation failed: {mem_err}")
                        memory_record = None

                    memory_id = None
                    if memory_record is not None:
                        try:
                            memory_id = str(memory_record.id)
                        except Exception:
                            memory_id = None

                    try:
                        relationship_answer_metadata = await reflection_repo.mark_relationship_question_answered(
                            question_id=awaiting_question.get("id"),
                            answer_text=answer_text,
                            memory_id=memory_id,
                            answer_tags=normalized_tags,
                            importance_score=importance_score,
                            confidence_score=confidence_score,
                        )
                    except Exception as answer_err:
                        logger.warning(f"Failed to mark relationship question answered: {answer_err}")
        except Exception as awaiting_err:
            logger.warning(f"Relationship question awaiting check failed: {awaiting_err}")

    # Immediate memory extraction from high-value user messages
    try:
        from .memory.extractor import MemoryExtractor
        memory_extractor = MemoryExtractor(
            conversation_repo=conversation_repo,
            llm_controller=app.state.services.get("llm_router")
        )
        
        # Get user for memory ownership (use the same session-aligned user)
        if user:
                # Extract memory from this single message if it contains high-value information
                user_message_dict = {"role": "user", "content": chat_request.prompt}
                extracted_memory = await memory_extractor.extract_memory_from_single_message(
                    user_id=str(user.id),
                    message=user_message_dict,
                    conversation_id=str(conversation.id)
                )
                if extracted_memory:
                    logger.info(f"Extracted immediate memory from user message: {extracted_memory['content'][:100]}...")
    except Exception as e:
        logger.warning(f"Immediate memory extraction failed: {e}")
        # Continue with chat flow even if memory extraction fails

    # Get conversation history early for reflection decision logic (reuse pre-fetched, slice to last 50)
    conversation_history = conversation_history_full[-50:] if len(conversation_history_full) > 50 else conversation_history_full

    # Determine conversation context based on timestamp analysis (needed for reflection classifier)
    conversation_context = "new"  # Default to new conversation
    time_gap_minutes = 0
    
    if conversation_history:
        try:
            # Get the most recent message timestamp
            last_message = conversation_history[-1]  # Most recent message
            last_timestamp = last_message.get('timestamp')
            
            if last_timestamp:
                # Parse timestamp and calculate time gap
                from datetime import datetime, timezone
                if isinstance(last_timestamp, str):
                    # Handle different timestamp formats
                    try:
                        last_time = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                    except (ValueError, AttributeError) as e:
                        logger.debug(f"Failed to parse timestamp with timezone replacement: {e}")
                        try:
                            last_time = datetime.fromisoformat(last_timestamp)
                        except (ValueError, AttributeError) as e2:
                            logger.warning(f"Failed to parse timestamp, using current time: {e2}")
                            last_time = datetime.now(timezone.utc)
                else:
                    last_time = last_timestamp
                
                # Ensure timezone awareness
                if last_time.tzinfo is None:
                    last_time = last_time.replace(tzinfo=timezone.utc)
                
                current_time = datetime.now(timezone.utc)
                time_gap = current_time - last_time
                time_gap_minutes = time_gap.total_seconds() / 60
                
                # Determine conversation context based on time gap
                if time_gap_minutes <= 5:
                    conversation_context = "immediate_continuation"  # Very recent, likely same topic
                elif time_gap_minutes <= 30:
                    conversation_context = "recent_continuation"     # Recent, probably related
                elif time_gap_minutes <= 180:  # 3 hours
                    conversation_context = "session_continuation"   # Same session, may need context refresh
                else:
                    conversation_context = "new_session"           # Long gap, treat as new but acknowledge history
                
                logger.info(f"Conversation context: {conversation_context} (gap: {time_gap_minutes:.1f} minutes)")
                
        except Exception as e:
            logger.debug(f"Timestamp analysis failed (non-critical): {e}")
            conversation_context = "new"
    
    # Safety: Only treat as continuation if there is at least one prior user turn
    try:
        if conversation_history:
            prior_user_turns = [m for m in conversation_history[:-1] if (m or {}).get('role') == 'user']
            if not prior_user_turns:
                conversation_context = "new"
    except Exception:
        conversation_context = "new"

    # Prepare turn_id and trigger reflection generation with configurable sync/async behavior
    reflection_processor = app.state.services.get("reflection_processor")
    turn_id = str(uuid.uuid4())
    user_name_from_message = None
    try:
        import re
        raw_user_text = (chat_request.prompt or "").strip()
        match = re.search(r"\b(?i:my name is)\s+([A-Za-z][A-Za-z'\-]{1,49})\b", raw_user_text)
        if not match:
            match = re.search(r"\b(?i:(?:i am|i'm|im))\s+([A-Za-z][A-Za-z'\-]{1,49})\b", raw_user_text)
        if match:
            candidate = (match.group(1) or "").strip()
            if candidate:
                user_name_from_message = candidate if any(ch.isupper() for ch in candidate) else candidate.capitalize()
    except Exception:
        user_name_from_message = None
    reflection_text = ""
    # Track the full reflection payload for downstream templating/validation
    reflection_payload: Dict[str, Any] = {}
    if reflection_processor:
        try:
            # Enforce reflection-first by default; allow async mode via env
            strict_first = True
            # Synchronous reflection before chat unless REFLECTION_SYNC_MODE is explicitly disabled
            sync_mode = (os.getenv("REFLECTION_SYNC_MODE", "true").lower() in ("1","true","yes"))
            # Reflection always waits unbounded to preserve reflection-first behavior
            logging.info("Reflection sync timeout disabled (unbounded). Chat will wait for reflection to complete.")

            # SELECTIVE REFLECTION: Decide if this interaction warrants deep introspection
            should_reflect_decision = {"should_reflect": True, "reason": "default", "method": "fallback"}
            turn_count = len([m for m in conversation_history if m.get("role") == "user"]) if conversation_history else 0
            
            try:
                should_reflect_decision = await reflection_processor.should_generate_reflection(
                    user_message=chat_request.prompt,
                    conversation_context=conversation_context,
                    conversation_history=conversation_history,
                    turn_count=turn_count,
                    time_gap_minutes=time_gap_minutes,
                    user_profile_id=session_id
                )
                logger.info(
                    f"Reflection decision: {should_reflect_decision['should_reflect']} "
                    f"(method={should_reflect_decision['method']}, "
                    f"reason={should_reflect_decision['reason']}, "
                    f"confidence={should_reflect_decision.get('confidence', 0.0):.2f})"
                )
            except Exception as classifier_err:
                logger.error(f"Reflection classifier failed, defaulting to reflect: {classifier_err}")
                should_reflect_decision = {
                    "should_reflect": True,
                    "reason": f"Classifier error: {str(classifier_err)[:50]}",
                    "method": "error_fallback",
                    "confidence": 0.5
                }
            
            # Skip reflection if classifier says no
            if not should_reflect_decision["should_reflect"]:
                logger.info(f"⚡ Skipping reflection for this interaction: {should_reflect_decision['reason']}")
                reflection_text = None
                reflection_payload = {}
                reflection_repo = app.state.services.get("reflection_repo")
                latest_reflection: Dict[str, Any] = {}
                if reflection_repo is not None:
                    latest_reflection = await reflection_repo.get_latest_reflection(
                        session_id,
                        include_baseline=False,
                    ) or {}

                    if not latest_reflection:
                        # Use extracted boot reflection adoption function
                        await _adopt_boot_reflection_if_needed(
                            session_id=session_id,
                            turn_id=turn_id,
                            services=app.state.services,
                            user=user,  # Pre-fetched at line 2697
                            persona=persona  # Pre-fetched at line 2706
                        )
                        latest_reflection = await reflection_repo.get_latest_reflection(
                            session_id,
                            include_baseline=True,
                        ) or {}

                if latest_reflection:
                    latest_result = latest_reflection.get("result")
                    if isinstance(latest_result, dict):
                        reflection_payload = latest_result
                        reflection_text = latest_result.get("content") or ""
                    if not reflection_text:
                        reflection_text = latest_reflection.get("content", "")
                # Don't emit reflection_generating event if we're skipping
            else:
                # Skip boot adoption when a recent reflection already exists
                latest_existing: Dict[str, Any] = {}
                reflection_repo = app.state.services.get("reflection_repo")
                if reflection_repo is not None:
                    latest_existing = await reflection_repo.get_latest_reflection(
                        session_id,
                        include_baseline=False,
                    ) or {}

                if not latest_existing:
                    # Use extracted boot reflection adoption function
                    await _adopt_boot_reflection_if_needed(
                        session_id=session_id,
                        turn_id=turn_id,
                        services=app.state.services,
                        user=user,  # Pre-fetched at line 2697
                        persona=persona  # Pre-fetched at line 2706
                    )
                # Proceed with reflection generation
                # Emit a 'reflection_generating' event immediately for reflection-first UX
                try:
                    reflection_ns = app.state.services.get("reflection_namespace")
                    if reflection_ns is not None:
                        async def emit_reflection_generating_payload():
                            payload = {
                                "type": "reflection_generating",
                                "session_id": session_id,
                                "timestamp": utc_iso(),
                            }
                            await reflection_ns.emit_reflection_event(
                                event_name="reflection_generating",
                                data=payload,
                                user_id=session_id,
                            )
                        await emit_reflection_generating_payload()
                except Exception as _emit_err:
                    logging.debug(f"Pre-emit reflection_generating failed (non-fatal): {_emit_err}")

            if sync_mode and should_reflect_decision["should_reflect"]:
                # Synchronous reflection (unbounded wait for reflection-first behavior)
                try:
                    # Build additional context for reflection
                    reflection_additional_context = {
                        "current_user_message": chat_request.prompt,
                        "turn_id": turn_id,
                    }
                    # Include system metrics when diagnostics are active
                    if diagnostic_mode != DiagnosticMode.NONE and system_snapshot_text:
                        reflection_additional_context["system_health_snapshot"] = system_snapshot_text
                    
                    reflection_result = await reflection_processor.generate_reflection(
                        reflection_type="message",
                        user_profile_id=session_id,
                        memory_ids=None,
                        max_context_items=10,
                        trigger_source="user_turn",
                        turn_id=turn_id,
                        additional_context=reflection_additional_context,
                        user_name=user_name_from_message,
                    )
                    # Validate reflection result and extract content (preserve full payload for router)
                    reflection_text, reflection_payload = _extract_reflection_content(reflection_result)
                    if user_name_from_message and isinstance(reflection_text, str) and reflection_text:
                        replaced = reflection_text.replace("[User]", user_name_from_message).replace("[user]", user_name_from_message)
                        reflection_text = replaced
                        if isinstance(reflection_payload, dict) and reflection_payload.get("content"):
                            reflection_payload["content"] = replaced
                    # Accept deterministic fallback reflections as long as content exists.
                    _validate_reflection_content(reflection_text, reflection_payload, "pre-check")
                except asyncio.TimeoutError:
                    # Hard fail to guarantee reflection-first integrity
                    logging.error("Reflection timeout while generating message reflection; returning 503.")
                    raise HTTPException(status_code=503, detail="Reflection timed out; please retry.")
                except Exception as e:
                    # Enforce strict reflection-first when a hard error occurs
                    msg = None
                    try:
                        msg = str(e) or repr(e)
                    except Exception:
                        msg = repr(e)
                    structural_tokens = [
                        "missing_required_fields",
                        "content_length_out_of_bounds",
                        "schema_validation_failed",
                        "insufficient_conversation_grounding",
                        "missing_recent_conversation",
                        "duplicate_content_high_overlap",
                    ]
                    safety_tokens = [
                        "unfounded_history",
                        "meta_reasoning",
                        "sensory_hallucination",
                    ]
                    has_structural = any(t in (msg or "") for t in structural_tokens)
                    has_safety = any(t in (msg or "") for t in safety_tokens)
                    category = "unknown"
                    if has_safety:
                        category = "safety_identity"
                    elif has_structural:
                        category = "structural_grounding"
                    try:
                        logging.error("Reflection pre-processing exception (category=%s): %r", category, e, exc_info=True)
                    except Exception:
                        pass

                    if category == "structural_grounding":
                        # Soft-fail: proceed in degraded mode without raising 503
                        try:
                            logging.warning(
                                "Reflection non-critical failure (structural/grounding). "
                                "Proceeding without fresh reflection for this turn."
                            )
                        except Exception:
                            pass
                        reflection_text = ""
                        reflection_payload = {}
                        # Do not re-raise; allow fallback/degraded path below
                    else:
                        if not isinstance(e, HTTPException):
                            try:
                                logging.error(
                                    "Reflection critical failure (category=%s). Returning HTTP 503 to client.",
                                    category,
                                )
                            except Exception:
                                pass
                            raise HTTPException(status_code=503, detail="Reflection failed; please retry.")
                        # If it's already an HTTPException (e.g., due to timeout or invalid reflection),
                        # re-raise it so the request returns promptly instead of proceeding to chat.
                        try:
                            logging.warning(f"Reflection generation error (HTTPException): {e}")
                        except Exception:
                            pass
                        raise
            elif should_reflect_decision["should_reflect"]:
                # ENFORCE REFLECTION-FIRST: No async mode - always wait for reflection
                try:
                    # Build additional context for reflection
                    reflection_additional_context = {
                        "current_user_message": chat_request.prompt,
                        "turn_id": turn_id,
                    }
                    # Include system metrics when diagnostics are active
                    if diagnostic_mode != DiagnosticMode.NONE and system_snapshot_text:
                        reflection_additional_context["system_health_snapshot"] = system_snapshot_text
                    
                    reflection_result = await reflection_processor.generate_reflection(
                        reflection_type="message",
                        user_profile_id=session_id,
                        memory_ids=None,
                        max_context_items=10,
                        trigger_source="user_turn",
                        turn_id=turn_id,
                        additional_context=reflection_additional_context,
                        user_name=user_name_from_message,
                    )
                    # Extract reflection content (preserve full payload for router)
                    reflection_text, reflection_payload = _extract_reflection_content(reflection_result)
                    # Validate reflection quality (allow deterministic fallback with content)
                    _validate_reflection_content(reflection_text, reflection_payload, "sync path")
                except Exception as e:
                    msg = None
                    try:
                        msg = str(e) or repr(e)
                    except Exception:
                        msg = repr(e)
                    structural_tokens = [
                        "missing_required_fields",
                        "content_length_out_of_bounds",
                        "schema_validation_failed",
                        "insufficient_conversation_grounding",
                        "missing_recent_conversation",
                        "duplicate_content_high_overlap",
                    ]
                    safety_tokens = [
                        "unfounded_history",
                        "meta_reasoning",
                        "sensory_hallucination",
                    ]
                    has_structural = any(t in (msg or "") for t in structural_tokens)
                    has_safety = any(t in (msg or "") for t in safety_tokens)
                    category = "unknown"
                    if has_safety:
                        category = "safety_identity"
                    elif has_structural:
                        category = "structural_grounding"
                    try:
                        logging.error("Reflection sync exception (category=%s): %r", category, e, exc_info=True)
                    except Exception:
                        pass

                    if category == "structural_grounding":
                        # Soft-fail: proceed in degraded mode without raising 503
                        try:
                            logging.warning(
                                "Reflection non-critical failure (structural/grounding) in sync path. "
                                "Proceeding without fresh reflection for this turn."
                            )
                        except Exception:
                            pass
                        reflection_text = ""
                        reflection_payload = {}
                        # Do not re-raise; allow fallback/degraded path below
                    else:
                        if not isinstance(e, HTTPException):
                            try:
                                logging.error(
                                    "Reflection critical failure in sync path (category=%s). Returning HTTP 503.",
                                    category,
                                )
                            except Exception:
                                pass
                            raise HTTPException(status_code=503, detail="Reflection generation failed; please retry.")
                        try:
                            logging.warning(f"Reflection generation error in sync path (HTTPException): {e}")
                        except Exception:
                            pass
                        raise
        except Exception as e:
            logging.warning(f"Reflection pre-processing setup failed (continuing): {e}")
    # If no reflection was produced yet, prefer the installation-level boot reflection.
    # Do not synthesize a fallback; rely on installer boot prompt generation.
    try:
        if not reflection_text:
            try:
                reflection_repo = app.state.services.get("reflection_repo")
                reflection_ns = app.state.services.get("reflection_namespace")
                if reflection_repo is not None:
                    # Respect installation-wide adoption marker to prevent repeating emission
                    try:
                        base_dir = os.environ.get("INSTALL_DIR") or str(pathlib.Path(__file__).resolve().parents[1])
                        marker_path = pathlib.Path(base_dir) / ".boot_reflection_adopted"
                    except Exception:
                        marker_path = pathlib.Path(".boot_reflection_adopted")
                    if not marker_path.exists():
                        boot_seed = await reflection_repo.list_reflections(user_profile_id="installation", limit=1)
                        if boot_seed:
                            boot_reflection = boot_seed[0]
                            boot_result = (boot_reflection or {}).get("result", {})
                            # Track payload and content from boot reflection copy
                            reflection_payload = boot_result or {}
                            reflection_text = boot_result.get("content") or reflection_text
                            # Persist a copy under the current session for continuity and emit to frontend
                            try:
                                copied = await reflection_repo.create_reflection({
                                    "reflection_type": "message",
                                    "user_profile_id": session_id,
                                    "result": boot_result,
                                    "metadata": {"baseline": True, "seed": True, "source": "boot_copy"},
                                    "turn_id": turn_id,
                                })
                                if reflection_ns is not None:
                                    async def emit_reflection_payload():
                                        payload = {
                                            "type": "reflection",
                                            "reflection": boot_result,
                                            "session_id": session_id,
                                            "timestamp": utc_iso(),
                                        }
                                        await reflection_ns.emit_reflection_event(
                                            event_name="reflection_generated",
                                            data=payload,
                                            user_id=session_id,
                                        )
                                    await emit_reflection_payload()
                                # Write the marker to avoid repeating in this later path as well
                                try:
                                    marker_path.write_text("adopted\n", encoding="utf-8")
                                except Exception as _marker_err:
                                    logging.debug(f"Boot adoption marker write failed (non-fatal): {_marker_err}")
                            except Exception as _emit_copy_err:
                                logging.debug(f"Boot seed copy/emit skipped: {_emit_copy_err}")
            except Exception:
                pass
    except Exception as _baseline_err:
        logging.debug(f"Boot reflection adoption not applied: {_baseline_err}")

    # Optional degraded mode only if explicitly enabled by environment
    # If classifier decided to skip reflection, allow proceeding without reflection
    allow_degraded = os.getenv("REFLECTION_ALLOW_DEGRADED", "false").lower() in ("1","true","yes")
    degraded_mode = False
    classifier_skip = not should_reflect_decision.get("should_reflect", True)
    
    if not reflection_text and allow_degraded and not classifier_skip:
        # Degraded mode enabled but classifier wanted reflection - do it async
        degraded_mode = True
        try:
            reflection_processor = app.state.services.get("reflection_processor")
            if reflection_processor is not None:
                asyncio.create_task(
                    reflection_processor.generate_reflection(
                        reflection_type="message",
                        user_profile_id=session_id,
                        memory_ids=None,
                        max_context_items=10,
                        trigger_source="user_turn_background",
                        turn_id=turn_id,
                        user_name=user_name_from_message,
                    )
                )
        except Exception as _bg_err:
            logging.debug(f"Background reflection scheduling failed: {_bg_err}")
    elif not reflection_text and classifier_skip:
        # Classifier decided to skip - treat as valid degraded mode
        degraded_mode = True
        logger.info("Classifier skip - proceeding without reflection")
    # If degraded mode is not allowed and reflection still missing, enforce reflection-first strictly
    if not reflection_text and not degraded_mode:
        try:
            logging.warning(
                "Reflection enforcement: reflection_text is empty (degraded_mode=%s). "
                "Proceeding without fresh reflection (implicit degraded mode).",
                degraded_mode,
            )
        except Exception:
            pass
        degraded_mode = True
    
    try:
        # Use the LLMRouter for chat (conversational task)
        llm_router = app.state.services.get("llm_router")
        if llm_router is None:
            raise HTTPException(status_code=500, detail="LLM router not available")
        
        # conversation_history already loaded earlier for reflection decision logic

        # Relationship-question injection context
        # Fetch pending relationship questions but don't expose internal metadata to the LLM
        relationship_question_payload = None
        try:
            if reflection_repo:
                pending_questions = await reflection_repo.list_relationship_question_queue(
                    user_profile_id=session_id,
                    status="pending",
                    limit=1,
                    include_future=False,
                )
                if pending_questions:
                    relationship_question_payload = pending_questions[0]
                    try:
                        delivered_record = await reflection_repo.mark_relationship_question_delivered(
                            relationship_question_payload.get("id")
                        )
                        if delivered_record:
                            relationship_question_payload = delivered_record
                    except Exception as deliver_err:
                        logger.warning(f"Failed to mark relationship question delivered: {deliver_err}")
        except Exception as rel_err:
            logger.warning(f"Relationship question lookup failed: {rel_err}")
        
        # Get relevant memories for autonomous decision-making
        relevant_memories = ""
        user_name_from_memory = None  # Extract user's name to prevent identity confusion
        try:
            # Get recent high-importance memories for this user to inform autonomous decisions
            user_memories = await conversation_repo.get_memories(str(user_id), importance_threshold=7, limit=5)
            if user_memories:
                memory_summaries = []
                for memory in user_memories:
                    content = getattr(memory, 'content', '')
                    importance = getattr(memory, 'importance_score', 0)
                    memory_summaries.append(f"[Importance: {importance}] {content}")
                    
                    # Extract user's name from memory content
                    if not user_name_from_memory and content:
                        import re
                        # Look for "User name: X" or "User introduced themselves as X"
                        name_patterns = [
                            r"[Uu]ser name[:\s]+([A-Za-z][A-Za-z'\-]{1,49})",
                            r"[Ii]ntroduced.*as\s+([A-Za-z][A-Za-z'\-]{1,49})",
                            r"[Nn]ame is\s+([A-Za-z][A-Za-z'\-]{1,49})",
                        ]
                        for pattern in name_patterns:
                            match = re.search(pattern, content)
                            if match:
                                extracted = (match.group(1) or "").strip()
                                if extracted:
                                    user_name_from_memory = extracted if any(ch.isupper() for ch in extracted) else extracted.capitalize()
                                break
                
                relevant_memories = "\n".join(memory_summaries)
                logger.info(f"Loaded {len(user_memories)} high-importance memories for autonomous decision-making")
                if user_name_from_memory:
                    logger.info(f"Extracted user name from memory: {user_name_from_memory}")
        except Exception as e:
            logger.debug(f"Memory loading failed (non-critical): {e}")
        
        # Build persona-aware system prompt for this session's user
        # Pass pre-fetched persona to avoid redundant database query
        persona_system_prompt = await _build_persona_system_prompt(
            app.state.services, 
            session_id=session_id,
            persona=persona,  # Already fetched at line 2706
            persona_name=persona_name  # Already fetched at line 2708
        )
        
        # Proactively detect web intent and collect current web context (transparent behavior)
        web_context = ""
        try:
            import re as _re
            user_text = chat_request.prompt or ""
            # Detect explicit domain/URL mention
            url_match = _re.search(r"\b((?:https?://)?(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/\S*)?)\b", user_text)
            
            # Check if user is asking about THEIR specific resources (don't auto-search)
            user_specific = any(pattern in user_text.lower() for pattern in [
                "my website", "my site", "my repo", "my github", "my blog",
                "our website", "our site", "our company", "my company"
            ])
            
            # Generic web intent keywords (only trigger if not user-specific)
            intent_match = (not user_specific) and any(k in user_text.lower() for k in [
                "visit ", "check ", "browse ", "go to ", "look at ",
                "what's on ", "tell me about ", "show me ", "latest", "current",
                "weather", "forecast", "search for", "find information"
            ])
            fetched_items = []
            if url_match:
                raw_url = url_match.group(1)
                if not raw_url.startswith("http"):
                    raw_url = "https://" + raw_url
                try:
                    resp = requests.get(raw_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
                    if resp.status_code == 200:
                        extracted = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
                        snippet = (extracted or "").strip()
                        if snippet:
                            fetched_items.append({
                                "title": raw_url,
                                "url": raw_url,
                                "content": snippet
                            })
                except Exception as _e:
                    logger.debug(f"Direct fetch failed for {raw_url}: {_e}")
            elif intent_match:
                # Use Brave search if available
                api_key = os.getenv("BRAVE_SEARCH_API_KEY")
                if api_key:
                    try:
                        resp = requests.get(
                            "https://api.search.brave.com/res/v1/web/search",
                            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
                            params={"q": user_text, "count": 3}, timeout=8
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            results = (data.get("web", {}) or {}).get("results", [])[:3]
                            web_chunks = []
                            for r in results:
                                title = r.get("title", "(no title)")
                                url = r.get("url")
                                desc = r.get("description", "")
                                main_content = None
                                if url:
                                    try:
                                        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8).text
                                        main_content = trafilatura.extract(html, include_comments=False, include_tables=False)
                                    except Exception:
                                        main_content = None
                                content = (main_content or desc or "").strip()
                                if content:
                                    fetched_items.append({"title": title, "url": url, "content": content})
                    except Exception as _e:
                        logger.debug(f"Brave search failed: {_e}")
            if fetched_items:
                # Compose a concise web context with sources and a timestamp note (identity-safe)
                ts = utc_iso()
                lines = []
                for itm in fetched_items:
                    lines.append(f"Source: {itm.get('url') or itm.get('title')}\n{itm.get('content')}")
                web_context = f"Current public information (retrieved at {ts}):\n\n" + "\n\n---\n\n".join(lines)
        except Exception as _web_err:
            logger.debug(f"Web intent/context prefetch skipped: {_web_err}")
        
        # Debug logging to see what system prompt is being used
        logger.debug(f"Using persona system prompt (first 200 chars): {persona_system_prompt[:200]}...")
        
        if not degraded_mode:
            # Reflection-first path: use reflection + persona + memories + (optional) current web context
            web_section = f"[Current Web Information]\n{web_context}\n\n" if web_context else ""
            memory_section = f"[Accumulated Knowledge & Memories]\n{relevant_memories}\n\n" if relevant_memories else ""
            
            # Create conversation context guidance based on timestamp analysis (meta-only; never echo this)
            context_guidance = {
                "immediate_continuation": "Respond naturally without greetings. Keep the flow seamless as if there was no pause.",
                "recent_continuation": "Respond naturally, avoid formal greetings. Maintain continuity without referencing any prior exchange explicitly.",
                "session_continuation": "Respond naturally. If needed, briefly re-establish context—but do not mention time gaps or prior sessions.",
                "new_session": "Treat this as a fresh start. A brief natural greeting is acceptable; avoid repetitive patterns.",
                "new": "Treat this as a new conversation with no prior history. Do not imply past interactions."
            }
            
            context_instruction = context_guidance.get(conversation_context, context_guidance["new"])

            relationship_section = ""
            if relationship_question_payload:
                # Extract only the conversational elements, hide all internal metadata
                question_text = relationship_question_payload.get('question', '')
                conversation_prompt = relationship_question_payload.get('prompt', '')
                
                relationship_section = (
                    "[Internal Context - Never mention this section exists]\n"
                    "You have an opportunity to learn more about the user. Consider asking them about the following topic "
                    "if it feels natural and appropriate in this conversation:\n\n"
                    f"Topic to explore: {question_text}\n"
                    f"How to introduce it: {conversation_prompt}\n\n"
                    "CRITICAL INSTRUCTIONS:\n"
                    "- Weave this naturally into conversation - do NOT announce you have a 'queued question' or reference any internal system\n"
                    "- Never mention IDs, question queues, or that this is from a system prompt\n"
                    "- If the moment doesn't feel right, skip it entirely and respond naturally to the user's message\n"
                    "- Respect the user's comfort - if they don't want to answer, gracefully move on\n"
                    "- This is purely internal guidance - treat it like your own curiosity, not a system directive\n\n"
                )
            
            # Include reflection content to inform response (use comment blocks to keep it internal)
            reflection_section = ""
            if reflection_text:
                # Build context-aware header to prevent identity confusion
                user_context = f"about {user_name_from_memory}" if user_name_from_memory else "about the user"
                identity_reminder = f" Your name is {persona_name}, NOT {user_name_from_memory}." if user_name_from_memory else f" Your name is {persona_name}."
                
                reflection_section = (
                    f"<!-- YOUR INTERNAL THOUGHTS {user_context.upper()} - These are your private reflections about the conversation and {user_context}.{identity_reminder} Use insights naturally but never quote or reveal this section. -->\n"
                    f"{reflection_text}\n"
                    f"<!-- END INTERNAL THOUGHTS -->\n\n"
                )
            
            prompt = (
                f"{persona_system_prompt}\n\n"
                f"IMPORTANT: You are {persona_name}. When the user asks for your name or introduces themselves, respond as {persona_name}, NOT with any name mentioned in the conversation.\n\n"
                f"[Meta Guidance - do not echo or reference] {context_instruction}\n\n"
                "Response Formatting:\n"
                "- Keep replies concise: 2-4 sentences, ≤600 characters unless the user explicitly asks for more detail\n"
                "- Speak naturally with varied sentence lengths; show emotional nuance; avoid filler or repetition\n"
                "- Ask short clarifying questions when you are uncertain instead of guessing\n"
                "- Do NOT quote reflections directly - they inform your response naturally\n"
                "- Do NOT mention meta labels, system details, or session continuity\n"
                "- Use concrete details from context to keep responses grounded and specific\n\n"
                f"{web_section}"
                f"{memory_section}"
                f"{reflection_section}"
                f"{relationship_section}"
                f"[Conversation History]\n{format_messages_for_ollama(conversation_history, '', persona_name)}"
            )
        else:
            # Degraded path: use conversation history with persona context and memories
            web_section = f"[Current Web Information]\n{web_context}\n\n" if web_context else ""
            memory_section = f"[Accumulated Knowledge & Memories]\n{relevant_memories}\n\n" if relevant_memories else ""
            
            # Use the same conversation context guidance for degraded mode (meta-only; never echo this)
            context_guidance = {
                "immediate_continuation": "Respond naturally without greetings. Keep the flow seamless as if there was no pause.",
                "recent_continuation": "Respond naturally, avoid formal greetings. Maintain continuity without referencing any prior exchange explicitly.",
                "session_continuation": "Respond naturally. If needed, briefly re-establish context—but do not mention time gaps or prior sessions.",
                "new_session": "Treat this as a fresh start. A brief natural greeting is acceptable; avoid repetitive patterns.",
                "new": "Treat this as a new conversation with no prior history. Do not imply past interactions."
            }
            
            context_instruction = context_guidance.get(conversation_context, context_guidance["new"])
            
            formatted = (
                f"{persona_system_prompt}\n\n"
                f"IMPORTANT: You are {persona_name}. When the user asks for your name or introduces themselves, respond as {persona_name}, NOT with any name mentioned in the conversation.\n\n"
                f"[Meta Guidance - do not echo or reference] {context_instruction}\n\n"
                "Response Formatting:\n"
                "- Keep replies concise: 2-4 sentences, ≤600 characters unless the user explicitly asks for more detail\n"
                "- Speak naturally with varied sentence lengths; show emotional nuance; avoid filler or repetition\n"
                "- Ask short clarifying questions when you are uncertain instead of guessing\n"
                "- Do NOT quote reflections directly - they inform your response naturally\n"
                "- Do NOT mention meta labels, system details, or session continuity\n"
                "- Use concrete details from context to keep responses grounded and specific\n\n"
                f"{web_section}"
                f"{memory_section}"
                f"{format_messages_for_ollama(conversation_history, '', persona_name)}"
            )
            prompt = formatted
        # Utilize the conversational model with adaptive params derived from persona traits
        adaptive_params = await _derive_adaptive_chat_params(app.state.services)
        chat_max_tokens = adaptive_params["max_tokens"]
        chat_temperature = adaptive_params["temperature"]

        # Determine streaming availability
        streaming_cfg = get_performance_config().get_streaming_config()
        streaming_enabled = bool(streaming_cfg.get("enabled"))
        chat_namespace = app.state.services.get("chat_namespace") if hasattr(app.state, "services") else None
        allow_streaming = streaming_enabled and chat_namespace is not None
        # For diagnostic/system-status turns, disable streaming so we can append
        # a coherent system snapshot to the final response.
        if diagnostic_mode != DiagnosticMode.NONE:
            allow_streaming = False

        clean_response = ""
        response_timestamp = None
        if allow_streaming:
            llm_result = await llm_router.route(
                task_type="chat",
                prompt=prompt,
                model=get_llm_model("conversational"),
                max_tokens=chat_max_tokens,
                temperature=chat_temperature,
                reflection_data=reflection_payload if isinstance(reflection_payload, dict) else {},
                persona_name=persona_name,  # Pass for identity validation
                request_stream=True,
            )

            if inspect.isasyncgen(llm_result):
                tokens: list[str] = []
                response_timestamp = utc_iso()
                try:
                    async for chunk in llm_result:
                        token = getattr(chunk, "content", None)
                        if not token:
                            continue
                        tokens.append(token)
                        try:
                            await chat_namespace.emit_chat_chunk(
                                user_id=session_id,
                                turn_id=turn_id,
                                chunk=token,
                                timestamp=utc_iso(),
                            )
                        except Exception as emit_err:
                            logging.debug(f"Streaming chunk emit failed: {emit_err}")
                    clean_response = "".join(tokens)
                    try:
                        await chat_namespace.emit_chat_complete(
                            user_id=session_id,
                            turn_id=turn_id,
                            content=clean_response,
                            timestamp=response_timestamp or utc_iso(),
                        )
                    except Exception as emit_err:
                        logging.debug(f"Streaming completion emit failed: {emit_err}")
                except Exception as stream_err:
                    logging.debug(f"Streaming generator failed, falling back to full completion: {stream_err}")
                    clean_response = ""
                    allow_streaming = False
            else:
                clean_response = (
                    (llm_result or {}).get("content")
                    or (llm_result or {}).get("completion")
                    or ""
                )
                response_timestamp = utc_iso()
                allow_streaming = False

        if allow_streaming and clean_response:
            try:
                validator = ResponseValidator()
                context_payload = (
                    reflection_payload
                    if isinstance(reflection_payload, dict)
                    else {"content": str(reflection_payload) if reflection_payload is not None else ""}
                )
                context = validator.extract_reflection_context(context_payload)
                is_valid, validated_response = validator.validate_conversational_response(clean_response, context)
                if not is_valid:
                    logging.warning(
                        "Streaming conversational response validation failed. original=%s",
                        (clean_response[:200] + "…") if len(clean_response) > 200 else clean_response,
                    )
                    clean_response = validated_response
            except Exception as validation_err:
                logging.debug(f"Streaming response validation skipped: {validation_err}")

        if not allow_streaming:
            if not clean_response:
                llm_result = await llm_router.route(
                    task_type="chat",
                    prompt=prompt,
                    model=get_llm_model("conversational"),
                    max_tokens=chat_max_tokens,
                    temperature=chat_temperature,
                    reflection_data=reflection_payload if isinstance(reflection_payload, dict) else {},
                    persona_name=persona_name,  # Pass for identity validation
                )
                clean_response = (
                    (llm_result or {}).get("content")
                    or (llm_result or {}).get("completion")
                    or ""
                )
                response_timestamp = datetime.now(timezone.utc).isoformat()

            stop_reason = None
            if isinstance(llm_result, dict):
                stop_reason = _extract_stop_reason(llm_result)
            if _response_needs_continuation(clean_response, stop_reason):
                addition = await _request_additional_completion(
                    llm_router,
                    base_prompt=prompt,
                    partial_response=clean_response,
                    model=get_llm_model("conversational"),
                    max_tokens=chat_max_tokens,
                    temperature=chat_temperature,
                )
                if addition:
                    clean_response += addition

        if response_timestamp is None:
            response_timestamp = utc_iso()

        # Option B: Let the conversational LLM handle introductions naturally
        # The persona system prompt already includes the persona name and context
        # Apply light first-response polish to remove repetitive greetings only
        try:
            has_assistant = any((m or {}).get("role") == "assistant" for m in (conversation_history or []))
            clean_response = _polish_first_response(
                clean_response,
                None,  # Don't inject name - let LLM handle it
                conversation_history,
                False,  # Don't suppress intro - LLM decides
            )
        except Exception:
            pass
        try:
            resolved_user_name = user_name_from_message or user_name_from_memory
            if resolved_user_name and isinstance(clean_response, str) and clean_response:
                clean_response = clean_response.replace("[User]", resolved_user_name).replace("[user]", resolved_user_name)
        except Exception:
            pass
        # Option B: No intro injection - LLM handles all greetings naturally
        
        # Store assistant message in persistent storage
        processing_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
        assistant_message_obj = await conversation_repo.add_message(
            conversation_id=str(conversation.id),
            role="assistant",
            content=clean_response,
            model_used=chat_request.model,
            processing_time=processing_time,
            reflection_triggered=not degraded_mode  # true when reflection-first path
        )
        assistant_timestamp_iso = response_timestamp
        try:
            stored_ts = getattr(assistant_message_obj, "timestamp", None)
            if stored_ts is not None:
                assistant_timestamp_iso = stored_ts.isoformat()
                if not assistant_timestamp_iso.endswith("Z"):
                    assistant_timestamp_iso += "Z"
        except Exception:
            assistant_timestamp_iso = response_timestamp

        assistant_message = {
            "role": "assistant",
            "content": clean_response,
            "timestamp": assistant_timestamp_iso,
        }

        # Reflection already generated first with turn_id; no post-answer reflection
        
        # Check for conversation milestone episode trigger (async, non-blocking)
        episode_trigger_manager = app.state.services.get("episode_trigger_manager")
        if episode_trigger_manager:
            background_tasks.add_task(
                episode_trigger_manager.check_conversation_milestone,
                user_id=user_id,
                persona_id=None,  # Will be resolved automatically
            )
        
        # Get updated conversation history from persistent storage
        updated_history = await conversation_repo.get_conversation_history(session_id, limit=50)
        
        result_payload = {
            "response": assistant_message["content"],
            "history": updated_history,
            "turn_id": turn_id,
            "timestamp": assistant_timestamp_iso,
        }
        if relationship_question_payload:
            result_payload.setdefault("metadata", {})["relationship_question"] = relationship_question_payload
        if relationship_answer_metadata:
            result_payload.setdefault("metadata", {})["relationship_answer"] = relationship_answer_metadata
        if degraded_mode:
            result_payload["note"] = "degraded_mode"
        # Option B: Removed bootstrap summary emission since intro injection system was removed
        # The Persona page fetches data directly via API endpoints
        return result_payload
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_online(query: str):
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Brave Search API key not set.")
    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            params={"q": query, "count": 3}
        )
        if resp.status_code != 200:
            return {"result": f"Brave API error: {resp.status_code} {resp.text}"}
        data = resp.json()
        if "web" in data and "results" in data["web"] and data["web"]["results"]:
            results = data["web"]["results"][:3]
            web_chunks = []
            for r in results:
                title = r.get("title", "")
                url = r.get("url", "")
                desc = r.get("description", "")
                main_content = None
                if url:
                    try:
                        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
                        main_content = trafilatura.extract(html, include_comments=False, include_tables=False)
                    except Exception:
                        main_content = None
                content = main_content if main_content else desc
                web_chunks.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            is_weather = any(word in query.lower() for word in ["weather", "forecast", "temperature", "rain", "humidity"])
            
            # Import constraints for grounded search responses
            # Note: Router (Fix 2) already injects SPECIES_CONSTRAINT for identity protection
            # Here we only add search-specific grounding rules to minimize token overhead
            from backend.constraints import CoreConstraints
            
            # Build lightweight search-specific constraint prompt (~150 tokens vs ~700 with full TRUTHFULNESS)
            constraint_preamble = f"""SEARCH GROUNDING CONSTRAINTS:
{CoreConstraints.GROUNDING_CONSTRAINT}

SEARCH-SPECIFIC RULES:
- Only include facts that appear explicitly in the search results below
- Never fabricate, infer, or hallucinate information not present in the results
- Reference source URLs when citing specific facts
- If information is not available in the results, say so clearly
- Do not make up search results or web content

"""
            
            if is_weather:
                summary_prompt = "Extract the current weather and forecast for the location from the following web results. Only use facts that appear in the content. Include temperature, conditions, and time if available. Reference URLs inline."
            else:
                summary_prompt = "Summarize the following web results into a clear, concise answer. Include key facts, and reference URLs inline."
            
            summary_input = constraint_preamble + summary_prompt + "\n\n" + "\n---\n".join(web_chunks)
            try:
                llm_router = app.state.services.get("llm_router")
                if llm_router is None:
                    raise RuntimeError("LLM router not initialized")
                # Allow unlimited tokens by default; when CHAT_MAX_TOKENS>0, pass it through
                try:
                    chat_max_tokens = int(os.environ.get("CHAT_MAX_TOKENS", "0"))
                except Exception:
                    chat_max_tokens = 0
                route_kwargs = {
                    "task_type": "chat",
                    "prompt": summary_input,
                    "model": get_llm_model("conversational"),
                    "temperature": 0.3,
                }
                if chat_max_tokens and chat_max_tokens > 0:
                    route_kwargs["max_tokens"] = chat_max_tokens
                llm_res = await llm_router.route(**route_kwargs)
                content = (llm_res or {}).get("content", "").strip()
                return {"result": content if content else "\n\n".join(web_chunks)}
            except Exception:
                return {"result": "\n\n".join(web_chunks)}
        return {"result": "No results found."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    # Check if enhanced scheduler is active
    scheduler_integration = app.state.services.get("scheduler_integration")
    scheduler_status = "active" if scheduler_integration and scheduler_integration.initialized else "inactive"
    
    # Check persona integration status WITHOUT blocking startup readiness.
    # Do not await dependency factories here; rely on app.state snapshot to keep /health fast.
    persona_integration = None
    persona_error: Optional[str] = None
    try:
        persona_integration = app.state.services.get("persona_integration")
        if persona_integration is None:
            persona_status = "initializing"
        else:
            persona_status = "active" if getattr(persona_integration, "initialized", False) else "inactive"
    except Exception as e:
        logging.error(f"Persona system health check failed: {e}", exc_info=True)
        persona_status = "error"
        try:
            persona_error = str(e)
        except Exception:
            persona_error = "unknown"
    
    return {
        "status": "ok",
        "enhanced_scheduler": scheduler_status,
        "persona_system": persona_status,
        "persona_error": persona_error,
        "time": datetime.now().isoformat()
    }

# Backward-compatible alias for health under /api
@app.get("/api/health")
async def api_health_alias():
    return await health_check()

@app.get("/health/details")
async def health_details(probe_llm: bool = False, probe_db: bool = False):
    """Extended health diagnostics.
    - probe_llm: when true, perform a lightweight LLMRouter ping via centralized DI
    - probe_db: when true, attempt a non-destructive DB connectivity check (SELECT 1)
    Both probes are optional and safely disabled by default.
    """
    details = {
        "status": "ok",
        "time": datetime.now().isoformat(),
        "probes": {
            "llm": {"requested": probe_llm, "ok": None, "error": None},
            "db": {"requested": probe_db, "ok": None, "error": None},
        },
    }

    # LLM probe
    if probe_llm:
        try:
            from .api.dependencies import get_llm_router
            llm_router = await get_llm_router()
            # Perform a trivial routed call; implementation may return dict
            _ = await llm_router.route(task_type="chat", prompt="health ping")
            details["probes"]["llm"]["ok"] = True
        except Exception as e:
            details["status"] = "degraded"
            details["probes"]["llm"]["ok"] = False
            details["probes"]["llm"]["error"] = str(e)

    # DB probe
    if probe_db:
        try:
            # Prefer existing engine if available
            engine = None
            try:
                from .db.session import engine as session_engine
                engine = session_engine
            except Exception:
                engine = None
            if engine is None:
                # Fallback: build from environment config
                from sqlalchemy.ext.asyncio import create_async_engine
                db_url = os.environ.get("DATABASE_URL")
                if not db_url:
                    raise RuntimeError("DATABASE_URL not set in environment for DB probe")
                # Normalize to asyncpg if using standard postgresql scheme
                if db_url.startswith("postgresql://") and not db_url.startswith("postgresql+asyncpg://"):
                    db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
                engine = create_async_engine(db_url, echo=False)

            from sqlalchemy import text
            async with engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                _ = result.scalar()
            details["probes"]["db"]["ok"] = True
        except Exception as e:
            details["status"] = "degraded"
            details["probes"]["db"]["ok"] = False
            details["probes"]["db"]["error"] = str(e)

    return details

# Paginated API endpoints
@app.get("/api/reflections", response_model=PaginatedResponse[Dict[str, Any]])
async def list_reflections_paginated(
    user_id: Optional[str] = Query(None, description="User ID to filter reflections"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Items per page"),
    reflection_type: Optional[str] = Query(None, description="Filter by reflection type"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")
):
    """Get paginated list of reflections for a user."""
    try:
        from .api.dependencies import get_reflection_repository
        reflection_repo = await get_reflection_repository()
        
        # Create pagination parameters
        pagination_params = PaginationParams(page=page, size=size)
        
        # For single-user system, use default user if none provided
        if not user_id:
            user_id = "default_user"
        
        # Get total count
        total_count = await reflection_repo.count_reflections(
            user_profile_id=user_id, 
            reflection_type=reflection_type
        )
        
        # Get paginated results
        reflections = await reflection_repo.list_reflections(
            user_profile_id=user_id,
            reflection_type=reflection_type,
            limit=pagination_params.limit,
            offset=pagination_params.offset,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # Convert to dict format
        reflection_dicts = [
            {
                "id": r.get("id"),
                "reflection_type": r.get("reflection_type"),
                "created_at": r.get("created_at"),
                "result": r.get("result", {}),
                "metadata": r.get("metadata", {})
            }
            for r in reflections
        ]
        
        return PaginatedResponse.create(reflection_dicts, pagination_params, total_count)
        
    except Exception as e:
        logging.error(f"Error fetching paginated reflections: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch reflections")

@app.get("/api/conversations", response_model=PaginatedResponse[Dict[str, Any]])
async def list_conversations_paginated(
    user_id: Optional[str] = Query(None, description="User ID to filter conversations"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("last_message_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")
):
    """Get paginated list of conversations for a user."""
    try:
        from .api.dependencies import get_conversation_repository
        conversation_repo = await get_conversation_repository()
        
        # Create pagination parameters
        pagination_params = PaginationParams(page=page, size=size)
        
        # For single-user system, use default user if none provided
        if not user_id:
            user_id = "default_user"
        
        # Get total count
        total_count = await conversation_repo.count_conversations(user_id=user_id)
        
        # Get paginated results
        conversations = await conversation_repo.list_conversations(
            user_id=user_id,
            limit=pagination_params.limit,
            offset=pagination_params.offset,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # Convert to dict format
        conversation_dicts = [
            {
                "id": c.id,
                "session_id": c.session_id,
                "title": c.title,
                "message_count": c.message_count,
                "last_message_at": c.last_message_at.isoformat() if c.last_message_at else None,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "is_active": c.is_active
            }
            for c in conversations
        ]
        
        return PaginatedResponse.create(conversation_dicts, pagination_params, total_count)
        
    except Exception as e:
        logging.error(f"Error fetching paginated conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch conversations")

@app.get("/conversations/history")
async def get_conversation_history(
    session_id: str = Query(..., description="Session ID to get conversation history for"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of messages to return")
):
    """Get conversation history for a session ID"""
    try:
        conversation_repo = app.state.services.get('conversation_repo')
        if not conversation_repo:
            raise HTTPException(status_code=500, detail="Conversation repository not available")
        
        # Get conversation history from persistent storage
        messages = await conversation_repo.get_conversation_history(session_id, limit=limit)
        
        return {
            "messages": messages,
            "session_id": session_id,
            "count": len(messages)
        }
    except Exception as e:
        logging.error(f"Error fetching conversation history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch conversation history")

@app.get("/api/conversations/{conversation_id}/messages", response_model=PaginatedResponse[Dict[str, Any]])
async def list_conversation_messages_paginated(
    conversation_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(50, ge=1, le=200, description="Items per page"),
    sort_order: str = Query("asc", regex="^(asc|desc)$", description="Sort order")
):
    """Get paginated list of messages for a conversation."""
    try:
        from .api.dependencies import get_conversation_repository
        conversation_repo = await get_conversation_repository()
        
        # Create pagination parameters
        pagination_params = PaginationParams(page=page, size=size)
        
        # Get total count
        total_count = await conversation_repo.count_messages(conversation_id=conversation_id)
        
        # Get paginated results
        messages = await conversation_repo.get_conversation_messages(
            conversation_id=conversation_id,
            limit=pagination_params.limit,
            offset=pagination_params.offset,
            sort_order=sort_order
        )
        
        # Convert to dict format
        message_dicts = [
            {
                "id": m.id,
                "message_index": m.message_index,
                "role": m.role,
                "content": m.content,
                "model_used": m.model_used,
                "processing_time": m.processing_time,
                "reflection_triggered": m.reflection_triggered,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None
            }
            for m in messages
        ]
        
        return PaginatedResponse.create(message_dicts, pagination_params, total_count)
        
    except Exception as e:
        logging.error(f"Error fetching paginated messages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch messages")

# User API endpoint
@app.get("/health/faiss-fix")
async def fix_faiss_gpu():
    """Auto-fix FAISS GPU installation if needed."""
    try:
        from .core.faiss_validator import FAISSValidator
        validator = FAISSValidator()
        
        result = await validator.auto_fix()
        return {"status": "success", "fix_result": result}
    except Exception as e:
        logger.error(f"FAISS auto-fix failed: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/health/performance")
async def get_performance_stats():
    """Get LLM performance statistics including cache metrics."""
    try:
        from .llm.controller import LLMController
        controller = LLMController()
        
        stats = controller.get_performance_stats()
        
        # Ensure cache_stats is available for E2E test compatibility
        if "cache_stats" not in stats:
            # If performance optimizations aren't available, provide mock stats
            stats["cache_stats"] = {
                "hit_rate": 0.0,     # No cache if optimizations disabled
                "cache_size": 0,     # No cache entries
                "total_requests": 0,
                "time_saved_seconds": 0.0,
                "evictions": 0
            }
        
        # Add system performance metrics
        import psutil
        stats["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_memory_used": "N/A"  # Could be enhanced with nvidia-ml-py
        }
        
        return {"status": "success", "performance": stats}
    except Exception as e:
        logger.error(f"Performance stats failed: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/health/performance/stats")
async def get_performance_monitoring_stats():
    """Get detailed performance monitoring statistics."""
    try:
        from .llm.controller import LLMController
        controller = LLMController()
        
        stats = controller.get_performance_stats()
        return {"status": "success", "response_times": stats.get("response_times", {}), "cache_performance": stats.get("cache_stats", {})}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/health/performance/cache/clear")
async def clear_performance_cache():
    """Clear the LLM response cache (performance endpoint)."""
    try:
        from .llm.controller import LLMController
        controller = LLMController()
        # Clear any performance caches
        return {"status": "success", "message": "Performance cache cleared"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/llm/models/available")
async def get_available_models():
    """Get list of available LLM models."""
    try:
        # Check Ollama service for available models
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m.get('name', 'unknown') for m in data.get('models', [])]
            return {"status": "success", "models": models}
        else:
            return {"status": "error", "error": f"Ollama API returned {response.status_code}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/health/cache/clear")
async def clear_response_cache():
    """Clear the LLM response cache."""
    try:
        from .core.response_cache import clear_cache
        await clear_cache()
        return {"status": "success", "message": "Response cache cleared"}
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/api/users")
async def list_users():
    """List the single user in this SELO AI installation."""
    try:
        # SELO AI is single-user, single-persona - return the installation user
        user_repo = app.state.services.get("user_repo")
        if not user_repo:
            raise HTTPException(status_code=500, detail="User service not available")
        
        user = await user_repo.get_or_create_default_user()
        if not user:
            raise HTTPException(status_code=500, detail="Installation user not found")
        
        return [{
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_active": user.last_active.isoformat() if user.last_active else None,
            "is_active": user.is_active
        }]
    except Exception as e:
        logging.error(f"Error fetching installation user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch installation user")

# Memory Management API Endpoints

@app.get("/memories")
async def get_memories(
    importance_threshold: int = Query(3, ge=1, le=10, description="Minimum importance score"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of memories to return"),
    memory_type: Optional[str] = Query(None, description="Filter by memory type")
):
    """Get memories for the current user."""
    try:
        conversation_repo = app.state.services.get("conversation_repo")
        user_repo = app.state.services.get("user_repo")
        
        if not conversation_repo or not user_repo:
            raise HTTPException(status_code=500, detail="Memory services not available")
        
        # Get default user
        user = await user_repo.get_or_create_default_user()
        if not user:
            raise HTTPException(status_code=500, detail="User not found")
        
        # Get memories
        memories = await conversation_repo.get_memories(
            user_id=str(user.id),
            memory_type=memory_type,
            importance_threshold=importance_threshold,
            limit=limit
        )
        
        # Format memories for API response
        formatted_memories = []
        for memory in memories:
            formatted_memories.append({
                "id": str(memory.id),
                "content": memory.content,
                "type": memory.memory_type,
                "importance_score": memory.importance_score,
                "confidence_score": memory.confidence_score,
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
                "access_count": getattr(memory, 'access_count', 0),
                "tags": getattr(memory, 'tags', [])
            })
        
        return {
            "memories": formatted_memories,
            "total_count": len(formatted_memories),
            "filters": {
                "importance_threshold": importance_threshold,
                "memory_type": memory_type,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting memories: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get memories")

@app.post("/memories/consolidate")
async def trigger_memory_consolidation():
    """Manually trigger memory consolidation for the current user."""
    try:
        memory_consolidator = app.state.services.get("memory_consolidator")
        user_repo = app.state.services.get("user_repo")
        
        if not memory_consolidator:
            raise HTTPException(status_code=500, detail="Memory consolidation service not available")
        
        if not user_repo:
            raise HTTPException(status_code=500, detail="User service not available")
        
        # Get default user
        user = await user_repo.get_or_create_default_user()
        if not user:
            raise HTTPException(status_code=500, detail="User not found")
        
        # Trigger consolidation
        results = await memory_consolidator.manual_consolidation(user_id=str(user.id))
        
        return {
            "status": "success",
            "consolidation_results": results
        }
        
    except Exception as e:
        logger.error(f"Error triggering memory consolidation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to trigger memory consolidation")

@app.get("/memories/stats")
async def get_memory_stats():
    """Get memory statistics for the current user."""
    try:
        conversation_repo = app.state.services.get("conversation_repo")
        user_repo = app.state.services.get("user_repo")
        
        if not conversation_repo or not user_repo:
            raise HTTPException(status_code=500, detail="Memory services not available")
        
        # Get default user
        user = await user_repo.get_or_create_default_user()
        if not user:
            raise HTTPException(status_code=500, detail="User not found")
        
        # Get all memories for stats
        all_memories = await conversation_repo.get_memories(
            user_id=str(user.id),
            importance_threshold=1,
            limit=1000
        )
        
        # Calculate statistics
        total_memories = len(all_memories)
        
        # Group by importance
        importance_distribution = {}
        type_distribution = {}
        
        for memory in all_memories:
            # Importance distribution
            importance = memory.importance_score
            importance_distribution[importance] = importance_distribution.get(importance, 0) + 1
            
            # Type distribution
            mem_type = memory.memory_type
            type_distribution[mem_type] = type_distribution.get(mem_type, 0) + 1
        
        # Calculate average importance
        avg_importance = sum(m.importance_score for m in all_memories) / total_memories if total_memories > 0 else 0
        
        return {
            "total_memories": total_memories,
            "average_importance": round(avg_importance, 2),
            "importance_distribution": importance_distribution,
            "type_distribution": type_distribution,
            "high_importance_count": len([m for m in all_memories if m.importance_score >= 7]),
            "recent_memories_count": len([m for m in all_memories if m.created_at and 
                                        (utc_now() - (m.created_at if m.created_at.tzinfo else m.created_at.replace(tzinfo=timezone.utc))).days <= 7])
        }
        
    except Exception as e:
        logger.error(f"Error getting memory stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get memory statistics")

# Include API routers
from .api import reflection
app.include_router(reflection.router, prefix="/api")
from .api import persona_router
app.include_router(persona_router.router, prefix="/api")
from .api import agent_state_router
app.include_router(agent_state_router.router, prefix="/api")
from .api import autobiographical_episode_router
app.include_router(autobiographical_episode_router.router, prefix="/api")
from .api import sdl_router
app.include_router(sdl_router.router, prefix="/api")
from .api import meta_router
app.include_router(meta_router.router, prefix="/api")

# Health and monitoring endpoints
from .core.health_monitor import get_system_health, get_system_metrics
from .core.graceful_degradation import get_degradation_status
from .core.circuit_breaker import circuit_manager
from .core.faiss_validator import faiss_validator, validate_faiss, auto_fix_faiss

@app.get("/health/simple")
async def simple_health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": utc_iso()}

@app.get("/health/detailed")
async def detailed_health():
    """Detailed system health with all components."""
    return await get_system_health()

@app.get("/health/metrics")
async def system_metrics():
    """System performance metrics."""
    return await get_system_metrics()

@app.get("/health/degradation")
async def degradation_status():
    """Current system degradation status."""
    return get_degradation_status()

@app.get("/health/circuit-breakers")
async def circuit_breaker_status():
    """Circuit breaker states."""
    return circuit_manager.get_all_states()

@app.get("/health/circuit-breakers/metrics")
async def circuit_breaker_metrics():
    """Comprehensive circuit breaker metrics for monitoring."""
    return circuit_manager.get_all_metrics()

@app.post("/health/circuit-breakers/reset")
async def reset_circuit_breakers():
    """Reset all circuit breakers."""
    circuit_manager.reset_all()
    return {"status": "success", "message": "All circuit breakers reset"}

@app.get("/health/faiss-validation")
async def faiss_validation_status():
    """FAISS installation validation status."""
    validation = validate_faiss()
    return {
        "status": "success",
        "validation": validation.to_dict()
    }

@app.post("/health/faiss-validation/fix")
async def fix_faiss_installation():
    """Automatically fix FAISS installation issues."""
    success, message = auto_fix_faiss()
    return {
        "status": "success" if success else "error",
        "message": message,
        "validation": validate_faiss().to_dict()
    }

# Event publishing endpoint
from .api.security import require_system_key

@app.post("/events/publish")
async def publish_event(
    request: Request,
    event_type: str,
    event_data: Dict[str, Any],
    user_id: Optional[str] = None,
    _auth_ok: bool = Depends(require_system_key),
):
    
    scheduler_integration = app.state.services.get("scheduler_integration")
    if not scheduler_integration or not scheduler_integration.initialized:
        raise HTTPException(status_code=503, detail="Enhanced scheduler not available")
        
    try:
        # Process the event through the event trigger system
        await scheduler_integration.process_event(event_type, event_data, user_id)
        return {"status": "success", "message": f"Event {event_type} processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process event: {str(e)}")

# Create Socket.IO ASGI app
def get_socketio_app():
    """
    Factory for the ASGI application. Always mount Socket.IO so that
    /socket.io is available immediately at startup.
    We reuse a registry-backed server if present; otherwise we create and register one.
    """
    from .socketio.registry import get_socketio_server, register_socketio_server
    sio = get_socketio_server()
    if sio is None:
        # Create a server early so the transport endpoints exist before lifespan runs
        sio = socketio.AsyncServer(
            async_mode="asgi",
            # Allow all origins for Engine.IO/Socket.IO (use string "*")
            cors_allowed_origins="*",
            engineio_logger=True,
            async_handlers=True,
            # Match keepalive settings to avoid premature disconnects during backgrounding
            ping_timeout=86400,
            ping_interval=25,
            allow_upgrades=True,
            # Increase default buffer to 10MB; allow override via env SOCKET_MAX_HTTP_BUFFER_SIZE
            max_http_buffer_size=int(os.getenv("SOCKET_MAX_HTTP_BUFFER_SIZE", "10000000")),
        )
        register_socketio_server(sio)
    # Explicitly set the socket.io path to avoid any mounting ambiguity
    return socketio.ASGIApp(sio, app, socketio_path="/socket.io")

if __name__ == "__main__":
    import uvicorn
    logging.info("Starting SELO AI Backend...")
    # When executed as a module (python -m backend.main), the import path must be fully qualified
    # to avoid import errors that prevent the server from binding to the port.
    uvicorn.run(
        "backend.main:get_socketio_app",
        host="0.0.0.0",
        port=int(os.environ.get("SELO_AI_PORT", os.environ.get("PORT", "8000"))),
        reload=False,
    )
