"""
Capabilities Registry

Computes current capabilities and known limitations based on AppConfig and simple runtime facts.
This enables truthful reporting to the UI and safe inclusion in system prompts.
"""
from __future__ import annotations

from typing import Dict, Any, List
from datetime import datetime, timezone

from ..config.app_config import get_app_config


def compute_capabilities() -> Dict[str, Any]:
    """Compute current capabilities and limitations from configuration.

    Returns a dict shape:
    {
      "capabilities": {
        "persona_evolution": {"enabled": bool},
        "reflection": {"enabled": bool, "schedule_enabled": bool},
        "sdl": {"enabled": bool},
        "socketio": {"enabled": bool},
        "llm": {"conversational_model": str, "analytical_model": str},
      },
      "limitations": [str, ...],
      "generated_at": iso8601
    }
    """
    cfg = get_app_config()

    persona_cfg = cfg.get_persona_config()
    reflection_cfg = cfg.get_reflection_config()
    sdl_cfg = cfg.get_sdl_config()
    llm_cfg = cfg.get_llm_config()

    capabilities = {
        "persona_evolution": {"enabled": bool(persona_cfg.get("evolution_enabled", False))},
        "reflection": {
            "enabled": bool(reflection_cfg.get("enabled", False)),
            "schedule_enabled": bool(reflection_cfg.get("schedule_enabled", False)),
        },
        "sdl": {"enabled": bool(sdl_cfg.get("enabled", False))},
        "socketio": {"enabled": True},  # Server normally starts Socket.IO when backend is running
        "llm": {
            "conversational_model": llm_cfg.get("conversational_model"),
            "analytical_model": llm_cfg.get("analytical_model"),
        },
    }

    limitations: List[str] = []
    if not capabilities["persona_evolution"]["enabled"]:
        limitations.append("Persona evolution is disabled (PERSONA_EVOLUTION_ENABLED=false).")
    if not capabilities["reflection"]["enabled"]:
        limitations.append("Reflection is disabled (REFLECTION_ENABLED=false).")
    elif not capabilities["reflection"]["schedule_enabled"]:
        limitations.append("Reflection scheduling is disabled (REFLECTION_SCHEDULE_ENABLED=false).")
    if not capabilities["sdl"]["enabled"]:
        limitations.append("Self-Directed Learning (SDL) is disabled (SDL_ENABLED=false).")

    return {
        "capabilities": capabilities,
        "limitations": limitations,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
