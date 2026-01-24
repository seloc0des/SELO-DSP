"""
Token Budget Management for LLM Requests

Provides centralized token budget calculation with adaptive clamping to prevent
context window overflow while ensuring minimum completion quality.

The TokenBudgetManager handles:
1. Prompt token estimation
2. Context window size resolution
3. Available completion token calculation
4. Task-specific clamping (chat, reflection, analytical)
"""

import os
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

def _resolve_chat_max_cap() -> int:
    """
    Resolve the hard ceiling for chat completions.

    Priority:
    1) CHAT_RESPONSE_MAX_TOKENS (explicit cap for responses)
    2) CHAT_MAX_TOKENS legacy env
    3) Safe fallback of 480 tokens
    """
    try:
        cap = int(os.getenv("CHAT_RESPONSE_MAX_TOKENS", "0"))
    except Exception:
        cap = 0
    if cap <= 0:
        try:
            cap = int(os.getenv("CHAT_MAX_TOKENS", "480"))
        except Exception:
            cap = 480
    return max(cap, 1)


# Task-specific minimum completion tokens
# Keep chat expressive but prevent rambling
CHAT_MIN_COMPLETION_TOKENS = 128

# Reflection defaults: tuned for concise yet creative inner monologues.
# Env vars (REFLECTION_NUM_PREDICT, REFLECTION_MAX_TOKENS) can still raise/lower caps.
REFLECTION_MIN_COMPLETION_TOKENS = 220
MAX_REFLECTION_COMPLETION_TOKENS = 480

ANALYTICAL_MIN_COMPLETION_TOKENS = 1024


class TokenBudgetManager:
    """
    Manages token budget calculation with adaptive clamping.
    
    Prevents context overflow by:
    1. Estimating prompt tokens (~4 chars per token)
    2. Calculating available completion budget (90% of context - prompt)
    3. Clamping to safe min/max bounds per task type
    
    Usage:
        manager = TokenBudgetManager()
        budget = manager.calculate_budget(
            task_type="reflection",
            prompt="...",
            config={"num_ctx": 4096}
        )
        max_tokens = budget['max_tokens']
    """
    
    @staticmethod
    def _env_float(name: str, default: float) -> float:
        """Safely parse float from environment variable."""
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default
    
    @staticmethod
    def _env_int(name: str, default: int) -> int:
        """Safely parse int from environment variable."""
        try:
            return int(float(os.getenv(name, str(default))))
        except Exception:
            return default
    
    @classmethod
    def estimate_prompt_tokens(cls, prompt: str) -> int:
        """
        Estimate token count for prompt text.
        
        Uses improved heuristic: ~3.5 characters per token on average.
        This is more accurate for modern tokenizers (GPT-3.5/4, Llama, Qwen)
        and reduces false token budget warnings.
        
        Previous value of 4 chars/token was too conservative, causing
        unnecessary warnings about token budget exhaustion.
        
        Args:
            prompt: Text to estimate
            
        Returns:
            Estimated token count
        """
        prompt_length = len(str(prompt))
        # Use 3.5 chars/token: multiply by 10, divide by 35 (avoids float)
        return (prompt_length * 10) // 35
    
    @classmethod
    def get_context_window_size(cls, task_type: str = "chat") -> int:
        """
        Get context window size for task type.
        
        Args:
            task_type: Type of task (chat, reflection, analytical)
            
        Returns:
            Context window size in tokens
        """
        # All tasks share the same context window (model-level config)
        # qwen2.5:3b supports 8192 tokens natively - use full capacity
        return cls._env_int("CHAT_NUM_CTX", 8192)

    @classmethod
    def _resolve_reflection_max_cap(cls) -> int:
        """
        Resolve the maximum completion tokens for reflections.

        Priority:
        1) REFLECTION_MAX_TOKENS from environment (if > 0)
        2) Tier-aware fallback via detect_system_profile().budgets.reflection_max_tokens
        3) Safe standard fallback = 640
        """
        # 1) Explicit env override
        env_max = cls._env_int("REFLECTION_MAX_TOKENS", 0)
        if env_max > 0:
            return env_max

        # 2) Tier-aware system profile detection
        try:
            # Local import to avoid import-time coupling
            from ..utils.system_profile import detect_system_profile  # type: ignore
            profile = detect_system_profile()
            budgets = (profile or {}).get("budgets") or {}
            tier_cap = int(budgets.get("reflection_max_tokens") or 0)
            if tier_cap > 0:
                return tier_cap
        except Exception:
            
            logger.error("Silent exception caught", exc_info=True)

            # pass
        return 640
    
    @classmethod
    def calculate_available_tokens(
        cls,
        prompt_tokens: int,
        num_ctx: int,
        safety_buffer: float = 0.95
    ) -> int:
        """
        Calculate available tokens for completion.
        
        Reserves safety buffer (default 5%) to prevent overflow.
        With improved token estimation (3.5 chars/token), we can safely
        use 95% of context window instead of 90%.
        
        Args:
            prompt_tokens: Estimated prompt token count
            num_ctx: Context window size
            safety_buffer: Percentage of context to use (0.95 = 95%)
            
        Returns:
            Available tokens for completion
        """
        return int((num_ctx * safety_buffer) - prompt_tokens)
    
    @classmethod
    def get_task_config(cls, task_type: str) -> Dict[str, Any]:
        """
        Get configuration for specific task type.
        
        Args:
            task_type: chat, reflection, reflection_classifier, analytical, persona_prompt, sdl, persona_evolve
            
        Returns:
            dict with base_max_tokens, min_tokens, temperature, env_prefix
        """
        if task_type in ("chat", "persona_prompt"):
            chat_cap = _resolve_chat_max_cap()
            predict_env = cls._env_int("CHAT_NUM_PREDICT", chat_cap)
            base_chat_cap = min(max(predict_env, CHAT_MIN_COMPLETION_TOKENS), chat_cap)
            return {
                "base_max_tokens": base_chat_cap,
                "min_tokens": CHAT_MIN_COMPLETION_TOKENS,
                "temperature": cls._env_float("CHAT_TEMPERATURE", 0.6),
                "env_prefix": "CHAT"
            }
        elif task_type == "reflection":
            # Honor .env and tier-aware max cap instead of static clamp
            max_cap = cls._resolve_reflection_max_cap()
            env_predict = cls._env_int("REFLECTION_NUM_PREDICT", max_cap)
            base_reflection_cap = min(env_predict, max_cap)
            return {
                "base_max_tokens": max(base_reflection_cap, REFLECTION_MIN_COMPLETION_TOKENS),
                "min_tokens": REFLECTION_MIN_COMPLETION_TOKENS,
                "temperature": cls._env_float("REFLECTION_TEMPERATURE", 0.35),
                "env_prefix": "REFLECTION"
            }
        elif task_type == "reflection_classifier":
            # Classifier only needs YES/NO + brief reasoning (50-100 tokens max)
            return {
                "base_max_tokens": 100,
                "min_tokens": 20,
                "temperature": 0.2,
                "env_prefix": "REFLECTION"
            }
        elif task_type in ("sdl", "persona_evolve", "analytical"):
            return {
                "base_max_tokens": cls._env_int("ANALYTICAL_NUM_PREDICT", 1024),
                "min_tokens": ANALYTICAL_MIN_COMPLETION_TOKENS,
                "temperature": cls._env_float("ANALYTICAL_TEMPERATURE", 0.2),
                "env_prefix": "ANALYTICAL"
            }
        else:
            # Default to chat config
            return {
                "base_max_tokens": 1024,
                "min_tokens": 128,
                "temperature": 0.6,
                "env_prefix": "CHAT"
            }
    
    @classmethod
    def calculate_budget(
        cls,
        task_type: str,
        prompt: str = "",
        config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Calculate safe token allocation for task.
        
        Main entry point for token budget calculation. Handles:
        - Prompt token estimation
        - Available token calculation
        - Task-specific clamping
        - Minimum guarantees
        
        Args:
            task_type: Type of task (chat, reflection, analytical, etc.)
            prompt: Prompt text for token estimation
            config: Optional config overrides (num_ctx, etc.)
            
        Returns:
            dict with:
                - max_tokens: int (recommended max_tokens)
                - temperature: float (recommended temperature)
                - num_ctx: int (context window size)
                - estimated_prompt_tokens: int
                - available_tokens: int
                - clamped: bool (True if clamping was applied)
                - min_tokens: int (minimum guarantee)
        """
        config = config or {}
        
        # Step 1: Estimate prompt tokens
        estimated_prompt_tokens = cls.estimate_prompt_tokens(prompt)
        
        # Step 2: Get context window size
        num_ctx = config.get("num_ctx") or cls.get_context_window_size(task_type)
        
        # Step 3: Calculate available tokens
        available_tokens = cls.calculate_available_tokens(estimated_prompt_tokens, num_ctx)
        
        # Step 4: Get task-specific configuration
        task_config = cls.get_task_config(task_type)
        base_max_tokens = task_config["base_max_tokens"]
        min_tokens = task_config["min_tokens"]
        temperature = task_config["temperature"]
        
        # Step 5: Apply clamping logic
        clamped = False
        final_max_tokens = base_max_tokens
        
        # Enforce minimum
        if available_tokens < min_tokens:
            logger.warning(
                "%s prompt leaves only %s tokens, below minimum %s. Forcing minimum.",
                task_type.capitalize(),
                available_tokens,
                min_tokens
            )
            final_max_tokens = min_tokens
            clamped = True
        # Clamp if needed
        elif available_tokens < base_max_tokens:
            final_max_tokens = max(min_tokens, available_tokens)
            logger.info(
                "Clamping %s max_tokens from %s to %s (prompt ~%s tokens, ctx=%s)",
                task_type,
                base_max_tokens,
                final_max_tokens,
                estimated_prompt_tokens,
                num_ctx
            )
            clamped = True
        
        return {
            "max_tokens": final_max_tokens,
            "temperature": temperature,
            "num_ctx": num_ctx,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "available_tokens": available_tokens,
            "clamped": clamped,
            "min_tokens": min_tokens,
            "base_max_tokens": base_max_tokens
        }


# Convenience function for backward compatibility
def calculate_token_budget(task_type: str, prompt: str = "", **config) -> Dict[str, Any]:
    """
    Calculate token budget for task (convenience wrapper).
    
    Args:
        task_type: Type of task
        prompt: Prompt text
        **config: Optional config overrides
        
    Returns:
        Token budget dict
    """
    return TokenBudgetManager.calculate_budget(task_type, prompt, config)
