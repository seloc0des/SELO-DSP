"""
Adaptive Token Budget System

Dynamically adjusts token budgets for constraints and responses based on:
- Available context window
- Prompt complexity
- Task priority
- Historical usage patterns
"""

import logging
import os
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("selo.adaptive_budgets")


@dataclass
class TokenBudget:
    """Represents token allocation for a specific task."""
    total_context_window: int
    prompt_tokens: int
    constraint_tokens: int
    response_tokens: int
    buffer_tokens: int
    utilization: float  # 0.0-1.0
    
    @property
    def allocated_tokens(self) -> int:
        """Total allocated tokens."""
        return self.prompt_tokens + self.constraint_tokens + self.response_tokens + self.buffer_tokens
    
    @property
    def available_tokens(self) -> int:
        """Remaining unallocated tokens."""
        return self.total_context_window - self.allocated_tokens


class AdaptiveTokenBudgetManager:
    """
    Manages adaptive token budget allocation across the system.
    """
    
    def __init__(self):
        """Initialize the budget manager."""
        self.context_window_size = self._get_context_window_size()
        self.safety_buffer_pct = 0.10  # Reserve 10% buffer
        self.min_response_tokens = 256  # Minimum response length
        self.max_constraint_pct = 0.15  # Max 15% for constraints
        
    def _get_context_window_size(self) -> int:
        """Get context window size from environment or defaults."""
        try:
            return int(os.getenv("CHAT_NUM_CTX", "8192"))
        except (ValueError, TypeError):
            return 8192
    
    def estimate_prompt_tokens(self, prompt: str) -> int:
        """
        Estimate token count for a prompt.
        
        Args:
            prompt: The prompt text
            
        Returns:
            Estimated token count (rough: ~4 chars per token)
        """
        # Rough estimation - in production, use proper tokenizer
        return len(prompt) // 4
    
    def calculate_budget(
        self,
        task_type: str,
        prompt: Optional[str] = None,
        prompt_token_estimate: Optional[int] = None,
        priority: str = "normal"
    ) -> TokenBudget:
        """
        Calculate adaptive token budget for a task.
        
        Args:
            task_type: Type of task ("chat", "reflection", "analytical")
            prompt: Optional prompt text for estimation
            prompt_token_estimate: Pre-calculated prompt token count
            priority: Task priority ("low", "normal", "high")
            
        Returns:
            TokenBudget allocation
        """
        # Estimate prompt tokens
        if prompt_token_estimate is not None:
            prompt_tokens = prompt_token_estimate
        elif prompt is not None:
            prompt_tokens = self.estimate_prompt_tokens(prompt)
        else:
            # Default estimates by task type
            defaults = {
                "chat": 1200,
                "reflection": 1500,
                "analytical": 1800,
                "persona_prompt": 1000
            }
            prompt_tokens = defaults.get(task_type, 1000)
        
        # Calculate buffer
        buffer_tokens = int(self.context_window_size * self.safety_buffer_pct)
        
        # Calculate available tokens for response and constraints
        available_for_allocation = self.context_window_size - prompt_tokens - buffer_tokens
        
        # Constraint budget (max % of available or fixed amount)
        max_constraint_tokens = int(available_for_allocation * self.max_constraint_pct)
        
        # Task-specific constraint needs
        constraint_defaults = {
            "chat": 100,          # Minimal (persona prompt has constraints)
            "reflection": 150,    # More constraints for internal generation
            "analytical": 120,
            "persona_prompt": 180  # Comprehensive identity constraints
        }
        base_constraint_tokens = constraint_defaults.get(task_type, 120)
        
        # Adjust based on priority
        priority_multipliers = {
            "low": 0.8,
            "normal": 1.0,
            "high": 1.2
        }
        constraint_multiplier = priority_multipliers.get(priority, 1.0)
        constraint_tokens = min(
            int(base_constraint_tokens * constraint_multiplier),
            max_constraint_tokens
        )
        
        # Response budget (remainder, with minimum)
        response_tokens = available_for_allocation - constraint_tokens
        response_tokens = max(response_tokens, self.min_response_tokens)
        
        # If response tokens are too low, reduce constraint budget
        if response_tokens < self.min_response_tokens:
            logger.warning(
                f"Insufficient tokens for task '{task_type}'. "
                f"Reducing constraint budget to fit minimum response."
            )
            response_tokens = self.min_response_tokens
            constraint_tokens = available_for_allocation - response_tokens
            constraint_tokens = max(constraint_tokens, 50)  # Absolute minimum
        
        # Calculate utilization
        allocated_tokens = prompt_tokens + constraint_tokens + response_tokens + buffer_tokens
        utilization = allocated_tokens / self.context_window_size
        
        budget = TokenBudget(
            total_context_window=self.context_window_size,
            prompt_tokens=prompt_tokens,
            constraint_tokens=constraint_tokens,
            response_tokens=response_tokens,
            buffer_tokens=buffer_tokens,
            utilization=utilization
        )
        
        logger.debug(
            f"Token budget for {task_type}: "
            f"prompt={prompt_tokens}, constraints={constraint_tokens}, "
            f"response={response_tokens}, buffer={buffer_tokens}, "
            f"utilization={utilization:.1%}"
        )
        
        return budget
    
    def optimize_constraint_set(
        self,
        constraint_set: Any,  # ConstraintSet from composition module
        budget: TokenBudget
    ) -> Any:
        """
        Optimize a constraint set to fit within budget.
        
        Args:
            constraint_set: The ConstraintSet to optimize
            budget: The token budget to fit
            
        Returns:
            Optimized ConstraintSet
        """
        # Use the constraint composition framework's optimize method
        return constraint_set.optimize(max_tokens=budget.constraint_tokens)
    
    def adjust_for_degradation(
        self,
        budget: TokenBudget,
        degradation_level: str
    ) -> TokenBudget:
        """
        Adjust budget for degraded operation mode.
        
        Args:
            budget: Original budget
            degradation_level: "minor", "moderate", "severe"
            
        Returns:
            Adjusted budget with reduced allocations
        """
        reductions = {
            "minor": 0.9,      # 10% reduction
            "moderate": 0.75,  # 25% reduction
            "severe": 0.5      # 50% reduction
        }
        
        factor = reductions.get(degradation_level, 1.0)
        
        return TokenBudget(
            total_context_window=budget.total_context_window,
            prompt_tokens=budget.prompt_tokens,
            constraint_tokens=int(budget.constraint_tokens * factor),
            response_tokens=int(budget.response_tokens * factor),
            buffer_tokens=budget.buffer_tokens,
            utilization=budget.utilization * factor
        )
    
    def get_budget_recommendation(
        self,
        task_type: str,
        context_data: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Get recommended token allocations for a task.
        
        Args:
            task_type: Type of task
            context_data: Dictionary with context info (conversation_length, memory_count, etc.)
            
        Returns:
            Dictionary with recommended token allocations
        """
        # Estimate prompt complexity from context
        conversation_length = context_data.get("conversation_length", 5)
        memory_count = context_data.get("memory_count", 0)
        web_context_present = context_data.get("web_context_present", False)
        
        # Base estimate
        base_prompt_tokens = 800
        
        # Add per message (~100 tokens avg)
        base_prompt_tokens += conversation_length * 100
        
        # Add for memories (~50 tokens per memory)
        base_prompt_tokens += memory_count * 50
        
        # Add for web context
        if web_context_present:
            base_prompt_tokens += 300
        
        # Calculate budget
        budget = self.calculate_budget(
            task_type=task_type,
            prompt_token_estimate=base_prompt_tokens
        )
        
        return {
            "max_tokens": budget.response_tokens,
            "constraint_budget": budget.constraint_tokens,
            "estimated_prompt_tokens": budget.prompt_tokens,
            "buffer_tokens": budget.buffer_tokens,
            "utilization": budget.utilization
        }


# Global manager instance
_global_budget_manager: Optional[AdaptiveTokenBudgetManager] = None


def get_budget_manager() -> AdaptiveTokenBudgetManager:
    """Get or create the global budget manager instance."""
    global _global_budget_manager
    if _global_budget_manager is None:
        _global_budget_manager = AdaptiveTokenBudgetManager()
    return _global_budget_manager


def calculate_adaptive_budget(
    task_type: str,
    prompt: Optional[str] = None,
    context_data: Optional[Dict[str, Any]] = None
) -> Dict[str, int]:
    """
    Convenience function to calculate adaptive budget.
    
    Args:
        task_type: Type of task
        prompt: Optional prompt text
        context_data: Optional context information
        
    Returns:
        Budget recommendations
    """
    manager = get_budget_manager()
    
    if context_data:
        return manager.get_budget_recommendation(task_type, context_data)
    else:
        budget = manager.calculate_budget(task_type, prompt=prompt)
        return {
            "max_tokens": budget.response_tokens,
            "constraint_budget": budget.constraint_tokens,
            "estimated_prompt_tokens": budget.prompt_tokens
        }
