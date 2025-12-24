"""
A/B Testing Framework for Constraints

Enables experimentation with different constraint configurations
to measure impact on quality, token usage, and violations.
"""

import logging
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger("selo.constraints.ab_testing")


class ExperimentStatus(Enum):
    """Status of an A/B test experiment."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ConstraintVariant:
    """A variant configuration for A/B testing."""
    variant_id: str
    name: str
    description: str
    constraint_modifications: Dict[str, Any]
    allocation_percentage: float  # 0.0-1.0
    
    def apply_to_constraint_set(self, constraint_set: Any) -> Any:
        """
        Apply this variant's modifications to a constraint set.
        
        Args:
            constraint_set: The ConstraintSet to modify
            
        Returns:
            Modified ConstraintSet
        """
        modifications = self.constraint_modifications
        
        # Apply priority adjustments
        if "priority_adjustments" in modifications:
            for constraint_name, new_priority in modifications["priority_adjustments"].items():
                constraint = constraint_set.get_constraint(constraint_name)
                if constraint:
                    from .composition import ConstraintPriority
                    constraint.priority = ConstraintPriority[new_priority]
        
        # Apply content modifications
        if "content_modifications" in modifications:
            for constraint_name, new_content in modifications["content_modifications"].items():
                constraint = constraint_set.get_constraint(constraint_name)
                if constraint:
                    constraint.content = new_content
        
        # Enable/disable constraints
        if "enabled_state" in modifications:
            for constraint_name, enabled in modifications["enabled_state"].items():
                constraint = constraint_set.get_constraint(constraint_name)
                if constraint:
                    constraint.enabled = enabled
        
        # Add metadata about variant
        constraint_set.metadata["ab_test_variant"] = {
            "variant_id": self.variant_id,
            "name": self.name,
            "applied_at": datetime.now(timezone.utc).isoformat()
        }
        
        return constraint_set


@dataclass
class ExperimentMetrics:
    """Metrics for an A/B test variant."""
    variant_id: str
    sample_size: int = 0
    violation_count: int = 0
    validation_failure_count: int = 0
    avg_validation_time_ms: float = 0.0
    avg_token_usage: float = 0.0
    auto_clean_success_count: int = 0
    auto_clean_failure_count: int = 0
    quality_score: float = 0.0  # Composite quality metric
    
    def calculate_quality_score(self) -> float:
        """
        Calculate composite quality score for this variant.
        
        Higher score = better performance
        """
        if self.sample_size == 0:
            return 0.0
        
        # Violation rate (lower is better)
        violation_rate = self.violation_count / self.sample_size
        violation_score = max(0, 1.0 - violation_rate)
        
        # Validation success rate (higher is better)
        validation_success_rate = 1.0 - (self.validation_failure_count / self.sample_size)
        
        # Auto-clean success rate
        total_cleans = self.auto_clean_success_count + self.auto_clean_failure_count
        if total_cleans > 0:
            clean_success_rate = self.auto_clean_success_count / total_cleans
        else:
            clean_success_rate = 1.0
        
        # Performance score (faster is better, normalized to 0-1)
        # Assume 100ms is "perfect", scale accordingly
        performance_score = max(0, 1.0 - (self.avg_validation_time_ms / 100))
        
        # Token efficiency (fewer tokens is better, assume 200 is "perfect")
        token_score = max(0, 1.0 - (self.avg_token_usage / 200))
        
        # Weighted composite
        quality_score = (
            violation_score * 0.30 +
            validation_success_rate * 0.25 +
            clean_success_rate * 0.20 +
            performance_score * 0.15 +
            token_score * 0.10
        )
        
        return min(quality_score, 1.0)
    
    def update(self) -> None:
        """Recalculate derived metrics."""
        self.quality_score = self.calculate_quality_score()


@dataclass
class ABTestExperiment:
    """An A/B test experiment configuration."""
    experiment_id: str
    name: str
    description: str
    variants: List[ConstraintVariant] = field(default_factory=list)
    control_variant_id: str = ""
    status: ExperimentStatus = ExperimentStatus.DRAFT
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    target_sample_size: int = 1000
    metrics_by_variant: Dict[str, ExperimentMetrics] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize metrics for each variant."""
        for variant in self.variants:
            if variant.variant_id not in self.metrics_by_variant:
                self.metrics_by_variant[variant.variant_id] = ExperimentMetrics(
                    variant_id=variant.variant_id
                )
    
    def assign_variant(self, user_id: str) -> ConstraintVariant:
        """
        Assign a variant to a user using consistent hashing.
        
        Args:
            user_id: User identifier for consistent assignment
            
        Returns:
            Assigned ConstraintVariant
        """
        if not self.variants:
            raise ValueError("No variants defined for experiment")
        
        # Use consistent hashing for stable assignments
        hash_input = f"{self.experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        hash_fraction = (hash_value % 10000) / 10000.0
        
        # Find variant based on allocation percentages
        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.allocation_percentage
            if hash_fraction < cumulative:
                return variant
        
        # Fallback to last variant
        return self.variants[-1]
    
    def record_sample(
        self,
        variant_id: str,
        violated: bool,
        validation_failed: bool,
        validation_time_ms: float,
        token_usage: int,
        auto_clean_success: Optional[bool] = None
    ) -> None:
        """
        Record a sample for a variant.
        
        Args:
            variant_id: ID of the variant
            violated: Whether constraint was violated
            validation_failed: Whether validation failed
            validation_time_ms: Validation duration
            token_usage: Tokens used for constraints
            auto_clean_success: Whether auto-clean succeeded (if applicable)
        """
        if variant_id not in self.metrics_by_variant:
            logger.warning(f"Unknown variant ID: {variant_id}")
            return
        
        metrics = self.metrics_by_variant[variant_id]
        
        # Update running averages
        n = metrics.sample_size
        metrics.sample_size += 1
        
        # Incremental average update
        metrics.avg_validation_time_ms = (
            (metrics.avg_validation_time_ms * n + validation_time_ms) / (n + 1)
        )
        metrics.avg_token_usage = (
            (metrics.avg_token_usage * n + token_usage) / (n + 1)
        )
        
        # Count events
        if violated:
            metrics.violation_count += 1
        if validation_failed:
            metrics.validation_failure_count += 1
        if auto_clean_success is not None:
            if auto_clean_success:
                metrics.auto_clean_success_count += 1
            else:
                metrics.auto_clean_failure_count += 1
        
        # Recalculate quality score
        metrics.update()
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get experiment results and statistical analysis.
        
        Returns:
            Dictionary with results and analysis
        """
        results = {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status.value,
            "variants": []
        }
        
        control_metrics = None
        if self.control_variant_id:
            control_metrics = self.metrics_by_variant.get(self.control_variant_id)
        
        for variant in self.variants:
            metrics = self.metrics_by_variant[variant.variant_id]
            
            variant_result = {
                "variant_id": variant.variant_id,
                "name": variant.name,
                "is_control": variant.variant_id == self.control_variant_id,
                "allocation_percentage": variant.allocation_percentage,
                "sample_size": metrics.sample_size,
                "metrics": {
                    "violation_rate": metrics.violation_count / metrics.sample_size if metrics.sample_size > 0 else 0,
                    "validation_failure_rate": metrics.validation_failure_count / metrics.sample_size if metrics.sample_size > 0 else 0,
                    "avg_validation_time_ms": round(metrics.avg_validation_time_ms, 2),
                    "avg_token_usage": round(metrics.avg_token_usage, 1),
                    "quality_score": round(metrics.quality_score, 3)
                }
            }
            
            # Add comparison to control if applicable
            if control_metrics and variant.variant_id != self.control_variant_id:
                if metrics.sample_size > 0 and control_metrics.sample_size > 0:
                    quality_delta = metrics.quality_score - control_metrics.quality_score
                    quality_pct_change = (quality_delta / control_metrics.quality_score) * 100 if control_metrics.quality_score > 0 else 0
                    
                    # Basic significance indicator based on sample size and delta magnitude
                    # Note: This is a simplified heuristic. For rigorous statistical testing,
                    # implement proper t-test or chi-square test with p-value calculation.
                    min_sample_size = 30  # Minimum for meaningful comparison
                    significant_threshold = 0.05  # 5% improvement threshold
                    
                    has_sufficient_samples = (
                        metrics.sample_size >= min_sample_size and 
                        control_metrics.sample_size >= min_sample_size
                    )
                    has_meaningful_delta = abs(quality_pct_change) >= significant_threshold * 100
                    
                    if has_sufficient_samples and has_meaningful_delta:
                        significance = "likely_significant" if abs(quality_delta) > 0.1 else "possibly_significant"
                    elif has_sufficient_samples:
                        significance = "not_significant"
                    else:
                        significance = "insufficient_data"
                    
                    variant_result["vs_control"] = {
                        "quality_score_delta": round(quality_delta, 3),
                        "quality_pct_change": round(quality_pct_change, 1),
                        "statistical_significance": significance,
                        "note": "Heuristic indicator. For rigorous testing, use proper statistical methods."
                    }
            
            results["variants"].append(variant_result)
        
        # Determine winner
        if len(self.variants) > 1:
            best_variant = max(
                self.variants,
                key=lambda v: self.metrics_by_variant[v.variant_id].quality_score
            )
            results["winner"] = {
                "variant_id": best_variant.variant_id,
                "name": best_variant.name,
                "quality_score": round(self.metrics_by_variant[best_variant.variant_id].quality_score, 3)
            }
        
        return results


class ABTestManager:
    """
    Manages A/B test experiments for constraints.
    """
    
    def __init__(self):
        """Initialize the A/B test manager."""
        self.experiments: Dict[str, ABTestExperiment] = {}
        self.active_experiment_id: Optional[str] = None
    
    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str,
        variants: List[ConstraintVariant],
        control_variant_id: str,
        target_sample_size: int = 1000
    ) -> ABTestExperiment:
        """
        Create a new A/B test experiment.
        
        Args:
            experiment_id: Unique identifier
            name: Human-readable name
            description: Experiment description
            variants: List of constraint variants to test
            control_variant_id: ID of the control variant
            target_sample_size: Target sample size per variant
            
        Returns:
            Created ABTestExperiment
        """
        # Validate allocation percentages sum to 1.0
        total_allocation = sum(v.allocation_percentage for v in variants)
        if not (0.99 <= total_allocation <= 1.01):
            raise ValueError(f"Allocation percentages must sum to 1.0, got {total_allocation}")
        
        experiment = ABTestExperiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            control_variant_id=control_variant_id,
            target_sample_size=target_sample_size
        )
        
        self.experiments[experiment_id] = experiment
        logger.info(f"Created A/B test experiment: {experiment_id} with {len(variants)} variants")
        
        return experiment
    
    def start_experiment(self, experiment_id: str) -> None:
        """Start an experiment."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment.status = ExperimentStatus.ACTIVE
        experiment.start_date = datetime.now(timezone.utc)
        self.active_experiment_id = experiment_id
        
        logger.info(f"Started A/B test experiment: {experiment_id}")
    
    def stop_experiment(self, experiment_id: str) -> None:
        """Stop an experiment."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_date = datetime.now(timezone.utc)
        
        if self.active_experiment_id == experiment_id:
            self.active_experiment_id = None
        
        logger.info(f"Stopped A/B test experiment: {experiment_id}")
    
    def get_active_experiment(self) -> Optional[ABTestExperiment]:
        """Get the currently active experiment."""
        if self.active_experiment_id:
            return self.experiments.get(self.active_experiment_id)
        return None
    
    def apply_experiment_variant(
        self,
        constraint_set: Any,
        user_id: str
    ) -> Any:
        """
        Apply A/B test variant to constraint set if experiment active.
        
        Args:
            constraint_set: The ConstraintSet to modify
            user_id: User identifier for variant assignment
            
        Returns:
            Modified ConstraintSet (or original if no active experiment)
        """
        experiment = self.get_active_experiment()
        if not experiment:
            return constraint_set
        
        # Assign variant
        variant = experiment.assign_variant(user_id)
        
        # Apply variant modifications
        modified_set = variant.apply_to_constraint_set(constraint_set)
        
        logger.debug(
            f"Applied A/B test variant '{variant.name}' for user {user_id}"
        )
        
        return modified_set


# Global manager instance
_global_ab_manager: Optional[ABTestManager] = None


def get_ab_test_manager() -> ABTestManager:
    """Get or create the global A/B test manager."""
    global _global_ab_manager
    if _global_ab_manager is None:
        _global_ab_manager = ABTestManager()
    return _global_ab_manager
