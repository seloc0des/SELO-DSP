"""
Model Selection Optimizer

Intelligently selects the optimal model based on request characteristics,
balancing performance, quality, and resource usage.
"""

import logging
import re
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("selo.core.model_optimizer")

class TaskType(Enum):
    """Types of tasks for model selection."""
    CONVERSATIONAL = "conversational"
    ANALYTICAL = "analytical"
    CODING = "coding"
    REFLECTION = "reflection"
    QUICK_RESPONSE = "quick_response"
    CREATIVE = "creative"

@dataclass
class ModelProfile:
    """Profile of a model's capabilities and performance characteristics."""
    name: str
    size_gb: float
    avg_tokens_per_second: float
    quality_score: float  # 0-1 scale
    specialties: list[TaskType]
    max_context: int
    memory_usage_mb: int

class ModelOptimizer:
    """
    Intelligent model selection based on request characteristics and system state.
    
    Optimizes for:
    - Response time vs quality trade-offs
    - System resource availability
    - Task-specific model strengths
    - User preferences and context
    """
    
    def __init__(self):
        # Define available models with their characteristics
        self.models = {
            # Heavyweight models
            "phi3:mini-4k-instruct": ModelProfile(
                name="phi3:mini-4k-instruct",
                size_gb=2.4,
                avg_tokens_per_second=15.0,  # Estimated
                quality_score=0.75,
                specialties=[TaskType.QUICK_RESPONSE, TaskType.CONVERSATIONAL],
                max_context=4096,
                memory_usage_mb=2400
            ),
            "qwen2.5-coder:3b": ModelProfile(
                name="qwen2.5-coder:3b",
                size_gb=1.9,
                avg_tokens_per_second=20.0,
                quality_score=0.85,
                specialties=[TaskType.CODING, TaskType.ANALYTICAL],
                max_context=8192,
                memory_usage_mb=1900
            ),
            "humanish-llama3:8b-q4": ModelProfile(
                name="humanish-llama3:8b-q4",
                size_gb=4.9,
                avg_tokens_per_second=8.0,
                quality_score=0.90,
                specialties=[TaskType.CONVERSATIONAL, TaskType.CREATIVE],
                max_context=8192,
                memory_usage_mb=4900
            ),
            "qwen2.5:7b-instruct": ModelProfile(
                name="qwen2.5:7b-instruct",
                size_gb=4.7,
                avg_tokens_per_second=10.0,
                quality_score=0.88,
                specialties=[TaskType.ANALYTICAL, TaskType.REFLECTION],
                max_context=32768,
                memory_usage_mb=4700
            ),
            # Lightweight models
            "phi3:mini": ModelProfile(
                name="phi3:mini",
                size_gb=2.3,
                avg_tokens_per_second=18.0,
                quality_score=0.72,
                specialties=[TaskType.QUICK_RESPONSE, TaskType.CONVERSATIONAL],
                max_context=4096,
                memory_usage_mb=2300
            ),
            "qwen2.5-coder:1.5b": ModelProfile(
                name="qwen2.5-coder:1.5b",
                size_gb=0.9,
                avg_tokens_per_second=25.0,
                quality_score=0.78,
                specialties=[TaskType.CODING, TaskType.ANALYTICAL],
                max_context=8192,
                memory_usage_mb=900
            ),
            "gemma2:2b": ModelProfile(
                name="gemma2:2b",
                size_gb=1.3,
                avg_tokens_per_second=22.0,
                quality_score=0.75,
                specialties=[TaskType.REFLECTION, TaskType.ANALYTICAL],
                max_context=8192,
                memory_usage_mb=1300
            ),
            "llama3:8b": ModelProfile(
                name="llama3:8b",
                size_gb=4.7,
                avg_tokens_per_second=9.0,
                quality_score=0.88,
                specialties=[TaskType.REFLECTION, TaskType.CONVERSATIONAL],
                max_context=8192,
                memory_usage_mb=4700
            ),
            # Smaller models (for reference only - not used in 2-tier system)
            "smollm2:360m": ModelProfile(
                name="smollm2:360m",
                size_gb=0.72,
                avg_tokens_per_second=35.0,
                quality_score=0.60,
                specialties=[TaskType.QUICK_RESPONSE, TaskType.CONVERSATIONAL],
                max_context=2048,
                memory_usage_mb=720
            ),
            "smollm2:135m": ModelProfile(
                name="smollm2:135m",
                size_gb=0.27,
                avg_tokens_per_second=45.0,
                quality_score=0.55,
                specialties=[TaskType.QUICK_RESPONSE, TaskType.ANALYTICAL],
                max_context=2048,
                memory_usage_mb=270
            ),
            "tinyllama:1.1b": ModelProfile(
                name="tinyllama:1.1b",
                size_gb=0.64,
                avg_tokens_per_second=30.0,
                quality_score=0.62,
                specialties=[TaskType.REFLECTION, TaskType.QUICK_RESPONSE],
                max_context=2048,
                memory_usage_mb=640
            ),
            "all-minilm:22m": ModelProfile(
                name="all-minilm:22m",
                size_gb=0.044,
                avg_tokens_per_second=100.0,
                quality_score=0.70,  # Good for embeddings
                specialties=[TaskType.QUICK_RESPONSE],
                max_context=512,
                memory_usage_mb=44
            )
        }
        
        # Performance preferences
        self.preferences = {
            "prioritize_speed": False,  # Can be configured
            "max_response_time_seconds": 60,
            "min_quality_threshold": 0.7
        }
    
    def classify_task(self, prompt: str, context: Dict[str, Any] = None) -> TaskType:
        """
        Classify the task type based on prompt content and context.
        
        Args:
            prompt: Input prompt text
            context: Additional context information
            
        Returns:
            Classified task type
        """
        prompt_lower = prompt.lower()
        
        # Coding indicators
        coding_patterns = [
            r'\bcode\b', r'\bfunction\b', r'\bclass\b', r'\bimport\b',
            r'\bdef\b', r'\basync\b', r'\breturn\b', r'```',
            r'\bbug\b', r'\berror\b', r'\bdebug\b', r'\bapi\b'
        ]
        
        if any(re.search(pattern, prompt_lower) for pattern in coding_patterns):
            return TaskType.CODING
        
        # Analytical indicators
        analytical_patterns = [
            r'\banalyze\b', r'\bcompare\b', r'\bevaluate\b', r'\bexplain\b',
            r'\breason\b', r'\blogic\b', r'\bproblem\b', r'\bsolve\b',
            r'\bstrategy\b', r'\bplan\b'
        ]
        
        if any(re.search(pattern, prompt_lower) for pattern in analytical_patterns):
            return TaskType.ANALYTICAL
        
        # Reflection indicators
        reflection_patterns = [
            r'\breflect\b', r'\bthink about\b', r'\bconsider\b',
            r'\bintrospect\b', r'\bself\b', r'\bidentity\b'
        ]
        
        if any(re.search(pattern, prompt_lower) for pattern in reflection_patterns):
            return TaskType.REFLECTION
        
        # Quick response indicators (short prompts, simple questions)
        if len(prompt.split()) < 10 and ('?' in prompt or prompt_lower.startswith(('what', 'how', 'when', 'where', 'why'))):
            return TaskType.QUICK_RESPONSE
        
        # Creative indicators
        creative_patterns = [
            r'\bcreate\b', r'\bwrite\b', r'\bstory\b', r'\bpoem\b',
            r'\bimagine\b', r'\binvent\b', r'\bdesign\b'
        ]
        
        if any(re.search(pattern, prompt_lower) for pattern in creative_patterns):
            return TaskType.CREATIVE
        
        # Default to conversational
        return TaskType.CONVERSATIONAL
    
    def estimate_response_time(self, model: ModelProfile, prompt_length: int) -> float:
        """
        Estimate response time based on model characteristics and prompt.
        
        Args:
            model: Model profile
            prompt_length: Length of prompt in characters
            
        Returns:
            Estimated response time in seconds
        """
        # Rough estimation based on model speed and prompt complexity
        estimated_tokens = min(512, prompt_length // 4)  # Rough char to token conversion
        base_time = estimated_tokens / model.avg_tokens_per_second
        
        # Add overhead for model loading and processing
        overhead = 2.0 if model.size_gb > 4.0 else 1.0
        
        return base_time + overhead
    
    def select_optimal_model(self, 
                           prompt: str,
                           context: Dict[str, Any] = None,
                           user_preferences: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Select the optimal model for the given prompt and context.
        
        Args:
            prompt: Input prompt
            context: Additional context
            user_preferences: User-specific preferences
            
        Returns:
            Tuple of (model_name, selection_metadata)
        """
        task_type = self.classify_task(prompt, context)
        prompt_length = len(prompt)
        
        # Score each model
        model_scores = {}
        
        for model_name, model in self.models.items():
            score = 0.0
            
            # Task specialty bonus
            if task_type in model.specialties:
                score += 0.4
            
            # Quality score
            score += model.quality_score * 0.3
            
            # Speed preference
            estimated_time = self.estimate_response_time(model, prompt_length)
            if self.preferences["prioritize_speed"]:
                # Higher score for faster models
                speed_score = min(1.0, 30.0 / estimated_time)  # 30s as baseline
                score += speed_score * 0.3
            else:
                # Penalize very slow models
                if estimated_time > self.preferences["max_response_time_seconds"]:
                    score -= 0.2
            
            # Context length consideration
            if context and context.get("requires_long_context", False):
                if model.max_context >= 16384:
                    score += 0.1
            
            # Resource availability (simplified)
            # In a real implementation, this would check actual GPU memory
            if model.memory_usage_mb < 3000:  # Prefer smaller models under resource pressure
                score += 0.1
            
            model_scores[model_name] = {
                "score": score,
                "estimated_time": estimated_time,
                "task_match": task_type in model.specialties
            }
        
        # Select best model
        best_model = max(model_scores.keys(), key=lambda m: model_scores[m]["score"])
        
        selection_metadata = {
            "task_type": task_type.value,
            "all_scores": model_scores,
            "selection_reason": self._get_selection_reason(best_model, model_scores[best_model], task_type)
        }
        
        logger.info(f"Selected model {best_model} for {task_type.value} task (score: {model_scores[best_model]['score']:.2f})")
        
        return best_model, selection_metadata
    
    def _get_selection_reason(self, model_name: str, score_data: Dict[str, Any], task_type: TaskType) -> str:
        """Generate human-readable selection reason."""
        reasons = []
        
        if score_data["task_match"]:
            reasons.append(f"specialized for {task_type.value} tasks")
        
        if score_data["estimated_time"] < 30:
            reasons.append("fast response time")
        elif score_data["estimated_time"] > 60:
            reasons.append("high quality despite slower speed")
        
        model = self.models[model_name]
        if model.quality_score > 0.85:
            reasons.append("high quality output")
        
        return f"Selected {model_name}: " + ", ".join(reasons)
    
    def update_performance_data(self, model_name: str, actual_time: float, quality_feedback: Optional[float] = None):
        """
        Update model performance data based on actual usage.
        
        Args:
            model_name: Name of the model
            actual_time: Actual response time observed
            quality_feedback: Optional quality feedback (0-1 scale)
        """
        if model_name in self.models:
            model = self.models[model_name]
            
            # Update speed estimate (simple moving average)
            if actual_time > 0:
                # Rough tokens per second calculation (assuming 100 tokens generated)
                estimated_tps = 100 / actual_time
                model.avg_tokens_per_second = (model.avg_tokens_per_second * 0.8 + estimated_tps * 0.2)
            
            # Update quality score if feedback provided
            if quality_feedback is not None:
                model.quality_score = (model.quality_score * 0.9 + quality_feedback * 0.1)
            
            logger.debug(f"Updated performance data for {model_name}: {model.avg_tokens_per_second:.1f} tps")

# Global optimizer instance
_model_optimizer: Optional[ModelOptimizer] = None

def get_model_optimizer() -> ModelOptimizer:
    """Get global model optimizer instance."""
    global _model_optimizer
    if _model_optimizer is None:
        _model_optimizer = ModelOptimizer()
    return _model_optimizer
