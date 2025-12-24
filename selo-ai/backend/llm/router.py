"""LLM Router Module

Provides dynamic LLM routing and logging for SELO AI subsystems.
"""
import logging
import os
from typing import Literal, Dict, Any
import inspect

from .controller import LLMController
from .response_validator import ResponseValidator
from .streaming_validator import validate_streaming_response
logger = logging.getLogger("selo.llm.router")

def _resolve_reflection_cap() -> int:
    """Resolve reflection token cap with tier-aware fallback."""
    try:
        env_cap = int(os.getenv("REFLECTION_MAX_TOKENS", "0"))
    except Exception:
        env_cap = 0
    
    if env_cap <= 0:
        # Use tier-aware fallback from system profile
        try:
            from ..utils.system_profile import detect_system_profile
            profile = detect_system_profile()
            env_cap = profile["budgets"]["reflection_max_tokens"]
            logger.debug(f"Using tier-aware reflection cap: {env_cap} (tier={profile['tier']})")
        except Exception as e:
            # Final fallback to concise standard value
            env_cap = 480
            logger.warning(f"Failed to detect system tier, using concise fallback: {e}")
    
    return env_cap

MAX_REFLECTION_COMPLETION_TOKENS = _resolve_reflection_cap()

def _resolve_reflection_min_completion() -> int:
    """Resolve minimum reflection completion tokens with tier-aware word max."""
    try:
        word_max = int(os.getenv("REFLECTION_WORD_MAX", "0"))
    except Exception:
        word_max = 0
    
    if word_max <= 0:
        # Use tier-aware fallback from system profile
        try:
            from ..utils.system_profile import detect_system_profile
            profile = detect_system_profile()
            # Concise reflections share the same cap across tiers
            word_max = profile["budgets"].get("reflection_word_cap", 180)
            logger.debug(f"Using tier-aware word max: {word_max} (tier={profile['tier']})")
        except Exception as e:
            # Final fallback to concise value
            word_max = 180
            logger.warning(f"Failed to detect system tier for word max, using concise fallback: {e}")
    
    # Buffer for JSON scaffolding and metadata around the narrative content
    buffer_tokens = int(os.getenv("REFLECTION_COMPLETION_BUFFER", "60"))
    min_tokens = int(os.getenv("REFLECTION_MIN_COMPLETION_TOKENS", "260"))
    return max(min_tokens, word_max + buffer_tokens)


def _resolve_chat_min_completion() -> int:
    try:
        env_min = int(os.getenv("CHAT_MIN_COMPLETION_TOKENS", "0"))
    except Exception:
        env_min = 0
    # Default to ~4 dense sentences worth of tokens when not configured explicitly
    if env_min <= 0:
        env_min = 256
    return env_min


REFLECTION_MIN_COMPLETION_TOKENS = _resolve_reflection_min_completion()
CHAT_MIN_COMPLETION_TOKENS = _resolve_chat_min_completion()

class LLMRouter:
    """
    Central router for selecting and logging LLM usage.
    Supports dynamic routing by task type and analytics.
    """
    def __init__(self, conversational_llm: LLMController, analytical_llm: LLMController, reflection_llm: LLMController | None = None):
        self.conversational_llm = conversational_llm
        self.analytical_llm = analytical_llm
        self.reflection_llm = reflection_llm
        self.usage_log = []  # In-memory log; replace with persistent store as needed

    async def route(self, *, task_type: Literal["chat","persona_prompt","persona_evolve","sdl","reflection","reflection_classifier","embedding"], 
                   **kwargs) -> Dict[str, Any]:
        """
        Route request to the correct LLM and log usage.
        
        Args:
            task_type: The type of task being performed
            kwargs: Arguments for LLMController.complete or get_embedding
                - For 'chat' tasks, can include 'reflection_data' with the reflection content
                - For 'reflection' tasks, should include 'memories', 'emotions', 'attributes', and 'constraints'
                - For 'reflection_classifier' tasks, quick YES/NO decision on whether to reflect
                - persona_name: Optional persona name for identity validation
                - model: Optional model override (e.g., "qwen2.5:1.5b" for traits bootstrap)
                
        Returns:
            LLM response dict with additional metadata
        """
        # Extract reflection data and persona_name if provided (for chat tasks)
        reflection_data = kwargs.pop('reflection_data', None)
        persona_name = kwargs.pop('persona_name', "")
        latest_user_message = kwargs.pop('latest_user_message', "")
        request_stream = bool(kwargs.pop('request_stream', False))
        model_override = kwargs.pop('model', None)

        # Special-case embedding tasks: they don't use token budgets or streaming
        if task_type == "embedding":
            # Use analytical controller for route, but treat this as a dedicated embedding role
            from .dual_llm_config import get_llm_model
            llm = self.analytical_llm
            llm_role = "embedding"

            # Normalize args for embedding contract and force use of dedicated embedding model
            text = kwargs.pop("prompt", None) or kwargs.pop("text", None)
            model = kwargs.get("model") or get_llm_model("embedding") or getattr(llm, "default_model", None)

            log_entry = {
                "llm_role": llm_role,
                "task_type": task_type,
                "prompt": "<embedding>",
                "model": model,
            }
            logger.info(f"Routing task '{task_type}' to {llm_role} LLM: {log_entry['model']}")
            self.usage_log.append(log_entry)

            try:
                result = await llm.get_embedding(text=text or "", model=model)
                # Enforce strict response shape
                if isinstance(result, dict) and "embedding" in result:
                    out = result
                elif isinstance(result, list):
                    out = {"embedding": result, "dim": len(result), "model": model}
                else:
                    out = {"embedding": [], "dim": 0, "model": model, "error": "invalid_embedding_response"}
            except Exception as e:
                logger.error(f"Error getting embedding: {str(e)}", exc_info=True)
                out = {"embedding": [], "dim": 0, "model": model, "error": f"embedding_failed: {str(e)}"}

            if isinstance(out, dict):
                out["llm_role"] = llm_role
                out["task_type"] = task_type
            return out

        if task_type in ("chat", "persona_prompt"):
            llm = self.conversational_llm
            llm_role = "conversational"
                    
        elif task_type == "reflection":
            # Require an explicit reflection LLM; do not silently fall back
            if self.reflection_llm is None:
                raise RuntimeError("Reflection LLM not configured; cannot route reflection task")
            llm = self.reflection_llm
            llm_role = "reflection"
        elif task_type == "reflection_classifier":
            # Use reflection model for quick classification (already optimized for introspection)
            if self.reflection_llm is None:
                raise RuntimeError("Reflection LLM not configured; cannot route reflection_classifier task")
            llm = self.reflection_llm
            llm_role = "reflection"
        else:
            llm = self.analytical_llm
            llm_role = "analytical"

        log_entry = {
            "llm_role": llm_role,
            "task_type": task_type,
            "prompt": kwargs.get("prompt", "<embedding>"),
            "model": getattr(llm, "default_model", None),
        }
        logger.info(f"Routing task '{task_type}' to {llm_role} LLM: {log_entry['model']}")
        self.usage_log.append(log_entry)

        # Inject task-specific defaults for speed/quality balance
        # Use TokenBudgetManager for clean, maintainable budget calculation
        from .token_budget import TokenBudgetManager
        
        prompt = kwargs.get("prompt", "")
        
        # Determine effective task type for budget calculation
        budget_task_type = task_type
        if task_type in ("sdl", "persona_evolve") or llm_role == "analytical":
            budget_task_type = "analytical"
        
        # Calculate token budget
        budget = TokenBudgetManager.calculate_budget(
            task_type=budget_task_type,
            prompt=prompt
        )
        
        # Apply budget recommendations to kwargs
        kwargs.setdefault("max_tokens", budget["max_tokens"])
        kwargs.setdefault("temperature", budget["temperature"])
        
        # Apply model override if provided (e.g., qwen2.5:1.5b for traits bootstrap)
        if model_override:
            kwargs["model"] = model_override
            logger.info(f"Using model override: {model_override} for task_type={task_type}")

        # Route call for non-embedding tasks
        # Make the LLM call
        out = await llm.complete(**kwargs)

        # Check if the response is an async generator (streaming)
        if inspect.isasyncgen(out):
            # Apply streaming validation for chat responses
            if task_type in ("chat", "persona_prompt") and (persona_name or reflection_data):
                logger.debug("Applying streaming validation wrapper")
                return validate_streaming_response(
                    out,
                    persona_name=persona_name,
                    reflection_data=reflection_data
                )
            return out

        # For chat responses, ALWAYS validate identity constraints (non-streaming path only)
        # Fix 3: Removed reflection_data condition - validation now mandatory for all chat/persona_prompt
        if (not request_stream) and task_type in ("chat", "persona_prompt"):
            # Controller returns 'content' key, not 'response'
            response_text = out.get('content', '') if isinstance(out, dict) else str(out)
            
            # Extract context from reflection data for validation (use empty context if no reflection data)
            validator = ResponseValidator()
            if reflection_data:
                context = validator.extract_reflection_context(
                    reflection_data if isinstance(reflection_data, dict) else {'content': str(reflection_data)}
                )
            else:
                # No reflection data available - use minimal context
                # This ensures identity validation still runs even without reflection
                context = {'reflection_content': '', 'themes': [], 'emotional_state': {}}
            
            # Validate the response with identity constraints
            is_valid, validated_response = validator.validate_conversational_response(
                response_text, 
                context,
                persona_name=persona_name
            )
            
            # If the response was invalid, log it and use the sanitized version
            if not is_valid:
                logger.warning(f"Conversational response validation failed. Original: {response_text[:200]}...")
                if isinstance(out, dict):
                    out['content'] = validated_response  # Use 'content' to match controller
                    out['validation'] = {
                        'valid': False,
                        'reason': 'response_contained_hallucinations',
                        'original_response': response_text,
                        'sanitized_response': validated_response
                    }
                else:
                    out = {
                        'content': validated_response,  # Use 'content' to match controller
                        'validation': {
                            'valid': False,
                            'reason': 'response_contained_hallucinations',
                            'original_response': response_text,
                            'sanitized_response': validated_response
                        }
                    }
            else:
                # If response is valid but we're in a dict, add validation info
                if isinstance(out, dict):
                    out['validation'] = {'valid': True}
                else:
                    out = {'content': out, 'validation': {'valid': True}}  # Use 'content' to match controller
        # Attach routing/log info
        if isinstance(out, dict):
            out["llm_role"] = llm_role
            out["task_type"] = task_type
        return out

    def get_usage_log(self):
        """Return the current usage log."""
        return self.usage_log
