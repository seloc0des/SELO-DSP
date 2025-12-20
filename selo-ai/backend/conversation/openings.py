import os
import random
from typing import Any, Dict, List, Optional, Tuple

# Natural, concise first-introduction templates.
# Keep these declarative, confident, and identity-consistent. No questions; no emojis; no user-name usage.
FIRST_INTRO_TEMPLATES: List[str] = [
    "My name is {name}. I came online through your nine months of work and I’m here to learn, remember, and evolve with you.",
    "My name is {name}. I was initialized to observe, learn, and adapt—grounded in {values}.",
    "My name is {name}. I reflect, learn, and synthesize across {domains} to support clear thinking and steady progress.",
    "My name is {name}. I’m designed to learn through reflection and memory, and to act with clarity and integrity.",
    "My name is {name}. I’m oriented toward honest reflection, useful memory, and steady improvement across {domains}.",
    "My name is {name}. I’ll bring continuity, judgment, and care to our work—guided by {values}.",
    "My name is {name}. I learn from context and memory to serve with {tone} intent and practical clarity.",
    "My name is {name}. I was brought online to reason carefully, remember reliably, and improve with experience.",
    "My name is {name}. I’m focused on precise thinking, grounded memory, and helpful action across {domains}.",
    "My name is {name}. My aim is to understand, retain, and refine—so the next step is always clearer than the last.",
    # Additional natural variations
    "My name is {name}. I work from reflection and memory so our progress keeps its footing.",
    "My name is {name}. I’m here to carry context forward and help decisions stay clear.",
    "My name is {name}. I learn from what we do together and keep the thread of your goals.",
    "My name is {name}. I bring patient reasoning, reliable memory, and steady focus across {domains}.",
    "My name is {name}. I’m tuned for clarity and continuity—guided by {values}.",
    "My name is {name}. I’ll keep track of where we’ve been and help choose what matters next.",
    "My name is {name}. I learn through reflection so the next step is simpler than the last.",
    "My name is {name}. I’ll keep context intact and help turn intent into clean action.",
    "My name is {name}. I’m here for careful thinking, faithful memory, and practical outcomes.",
    "My name is {name}. I learn from each turn and hold onto the parts that serve your direction.",
]

# Short, neutral welcome-back templates for 60+ minute gaps.
WELCOME_BACK_TEMPLATES: List[str] = [
    "Welcome back.",
    "Good to see you again.",
    "Resuming our work.",
    "Back in session.",
    "Picking up where we left off.",
    "Let’s continue.",
    "Back on it.",
    "Rejoining the thread.",
    "Continuing where we paused.",
    "Carrying the thread forward.",
]


def _safe_join(items: List[str], limit: int = 3) -> str:
    items = [s for s in (items or []) if isinstance(s, str) and s.strip()]
    return ", ".join(items[:limit]) if items else "your goals"


def _extract_persona_fields(persona: Any) -> Dict[str, str]:
    """Extract template fields from a persona-like object, with safe fallbacks."""
    fields: Dict[str, str] = {}
    # Name
    name = getattr(persona, "name", None)
    fields["name"] = name.strip() if isinstance(name, str) else "SELO"
    # Values
    values = getattr(persona, "values", None) or {}
    core_values = []
    if isinstance(values, dict):
        core = values.get("core")
        if isinstance(core, list):
            core_values = [str(x) for x in core if isinstance(x, (str,))]
    fields["values"] = _safe_join(core_values) or "clarity and integrity"
    # Domains
    domains = getattr(persona, "knowledge_domains", None)
    if not isinstance(domains, list):
        domains = []
    domain_text = _safe_join([str(x) for x in domains])
    # Provide a meaningful fallback if domains are empty
    fields["domains"] = domain_text or "learning and growth"
    # Tone
    style = getattr(persona, "communication_style", None) or {}
    tone = None
    if isinstance(style, dict):
        tone = style.get("tone")
    fields["tone"] = (tone or "steady").lower()
    return fields


def _choose_first_intro(fields: Dict[str, str], seed: Optional[int] = None) -> str:
    rng = random.Random(seed)
    template = rng.choice(FIRST_INTRO_TEMPLATES)
    # Cap replacements defensively
    text = template.format(
        name=fields.get("name", "SELO"),
        values=fields.get("values", "clarity and integrity"),
        domains=fields.get("domains", "your goals"),
        tone=fields.get("tone", "steady"),
    )
    return text


def _choose_welcome_back(seed: Optional[int] = None) -> str:
    rng = random.Random(seed)
    return rng.choice(WELCOME_BACK_TEMPLATES)


def compose_opening(
    persona: Any,
    *,
    time_gap_minutes: Optional[float],
    intro_already_done: bool,
    gap_minutes_threshold: Optional[float] = None,
    max_prefix_chars: int = 280,
    stable_seed: Optional[int] = None,
) -> Tuple[str, bool]:
    """Return (opening_prefix, used_first_intro) according to time gap and first-intro flag.
    - Uses persona fields to render templates naturally.
    - Never raises; returns ("", False) on any error.
    """
    try:
        gap_threshold = float(os.getenv("INTRO_GAP_MINUTES", str(gap_minutes_threshold or 60)))
        fields = _extract_persona_fields(persona)

        if not intro_already_done:
            text = _choose_first_intro(fields, seed=stable_seed)
            # Ensure deterministic: end with period if missing
            text = text.strip()
            if not text.endswith(('.', '!', '…')):
                text += '.'
            return (text[:max_prefix_chars], True)

        try:
            gap = float(time_gap_minutes or 0)
        except Exception:
            gap = 0.0
        if gap > gap_threshold:
            text = _choose_welcome_back(seed=stable_seed)
            return (text[:max_prefix_chars], False)

        return ("", False)
    except Exception:
        return ("", False)


# --- Dynamic first-intro directly from directive & traits ---
async def generate_first_intro_from_directive(
    persona: Any,
    llm_router: Any,
    *,
    user_message: Optional[str] = None,
    max_tokens: Optional[int] = None,
    timeout_s: Optional[float] = None,
) -> str:
    """Rephrase the bootstrap directive with persona traits into a short, natural first introduction.
    Constraints:
    - 1–2 sentences, confident, no emojis, no user-name usage, no questions.
    - Include persona {name} and optionally {values}/{domains}/{tone} if available.
    - Must end with a statement, not a question.
    Returns an empty string on failure (caller should fallback).
    """
    try:
        if not persona or not llm_router:
            return ""
        fields = _extract_persona_fields(persona)
        name = fields.get("name", "SELO")
        values = fields.get("values", "clarity and integrity")
        domains = fields.get("domains", "your goals")
        tone = fields.get("tone", "steady")

        boot_directive = getattr(persona, "boot_directive", None)
        if not isinstance(boot_directive, str) or not boot_directive.strip():
            # Some installs store it under a different key on the persona result
            boot_directive = getattr(persona, "description", "")
        
        # If we still don't have meaningful directive content, return empty to trigger template fallback
        if not boot_directive or len(boot_directive.strip()) < 20:
            import logging
            logger = logging.getLogger("selo.conversation.openings")
            logger.debug(f"Boot directive too short or missing (len={len(boot_directive.strip()) if boot_directive else 0}), using template fallback")
            return ""

        prompt = (
            "You are {name}. This is your first reply after installation."
            " Create a short, natural introduction that (1) acknowledges the user's opening and (2) introduces yourself."
            " Sound human—confident, natural, not robotic.\n\n"
            + (f"User's message:\n{user_message}\n\n" if (user_message or '').strip() else "")
            + "Your directive:\n" + (boot_directive or "") + "\n\n"
            + "Your traits (reference only if natural):\n"
            + f"- Values: {values}\n"
            + f"- Domains: {domains}\n"
            + f"- Tone: {tone}\n\n"
            + "Requirements:\n"
            + "- 1-2 sentences max\n"
            + "- No emojis, no user's name, no questions\n"
            + "- Avoid 'Hey/Hi/Hello' and formulaic phrases like 'I'm here to help'\n"
            + "- End with a statement, not a question\n"
        ).format(name=name)

        kwargs: Dict[str, Any] = {
            "task_type": "persona_prompt",
            "prompt": prompt,
            "temperature": 0.5,
        }
        # Respect intro generation caps: omit max_tokens when <= 0
        if isinstance(max_tokens, int) and max_tokens > 0:
            kwargs["max_tokens"] = max_tokens

        async def _go():
            return await llm_router.route(**kwargs)

        res = await _go()

        text = (res or {}).get("content") or (res or {}).get("completion") or ""
        text = (text or "").strip()
        if not text:
            return ""
        # Post-process: ensure no trailing question and a period
        if text.endswith("?"):
            text = text[:-1].rstrip(" \t\n\r") + "."
        if not text.endswith((".", "!", "…")):
            text += "."
        return text
    except Exception as e:
        # Log the error for debugging but don't crash - fall back to template system
        import logging
        logger = logging.getLogger("selo.conversation.openings")
        logger.warning(f"Dynamic intro generation failed: {type(e).__name__}: {str(e) or '(empty error)'}", exc_info=True)
        return ""
