"""
Dual LLM Configuration for SELO AI

Defines which LLM model to use for conversational and analytical/coding tasks.
"""

import os

def _sanitize_model(model: str, default_model: str) -> str:
    """Return a model string compatible with LLMController parsing.

    Keep the model string as-is unless it includes a known provider prefix. We do NOT strip
    Ollama-style tags like ':7b-instruct' because these are often required to select the model
    variant. If empty, return the provided default.
    """
    if not model:
        return default_model
    # If model contains a known provider prefix (e.g., 'ollama:...'), keep as-is
    if ":" in model and model.split(":", 1)[0] in {"ollama", "openai", "anthropic", "hf"}:
        return model
    # Otherwise, return the model unchanged (preserve any tags like ':7b-instruct')
    return model

# Prefer installer-provided names; fall back to legacy SELO_* names; then defaults
# Default models: llama3:8b (conversational), qwen2.5:3b (analytical), qwen2.5:3b (reflection)
CONV_RAW = os.environ.get("CONVERSATIONAL_MODEL") or os.environ.get("SELO_CONVERSATIONAL_LLM") or "llama3:8b"
ANALYTIC_RAW = os.environ.get("ANALYTICAL_MODEL") or os.environ.get("SELO_ANALYTICAL_LLM") or "qwen2.5:3b"
REFLECTION_RAW = os.environ.get("REFLECTION_LLM") or os.environ.get("SELO_REFLECTION_LLM") or "qwen2.5:3b"
EMBEDDING_RAW = os.environ.get("EMBEDDING_MODEL") or os.environ.get("SELO_EMBEDDING_LLM") or "nomic-embed-text"

CONVERSATIONAL_LLM_MODEL = _sanitize_model(CONV_RAW, "llama3:8b")
ANALYTICAL_LLM_MODEL = _sanitize_model(ANALYTIC_RAW, "qwen2.5:3b")
REFLECTION_LLM_MODEL = _sanitize_model(REFLECTION_RAW, "qwen2.5:3b")
EMBEDDING_LLM_MODEL = _sanitize_model(EMBEDDING_RAW, "nomic-embed-text")

LLM_ROLE_MODELS = {
    "conversational": CONVERSATIONAL_LLM_MODEL,
    "analytical": ANALYTICAL_LLM_MODEL,
    "reflection": REFLECTION_LLM_MODEL,
    "embedding": EMBEDDING_LLM_MODEL,
}

def get_llm_model(role: str) -> str:
    """Return the model name for a given LLM role."""
    return LLM_ROLE_MODELS.get(role, ANALYTICAL_LLM_MODEL)

def get_required_models() -> dict:
    """Return a dict of all role->model mappings expected to be present.
    Keys: conversational, analytical, reflection, embedding.
    """
    return dict(LLM_ROLE_MODELS)
