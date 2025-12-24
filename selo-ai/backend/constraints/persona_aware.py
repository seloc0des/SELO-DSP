"""
Persona-Aware Constraint Relaxation

Dynamically adjusts constraint strictness based on persona maturity,
evolution history, and trust metrics.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger("selo.constraints.persona_aware")


@dataclass
class PersonaMaturityMetrics:
    """Metrics that determine a persona's maturity level."""
    evolution_count: int
    stability_score: float  # 0.0-1.0
    age_days: int
    confidence_avg: float  # Average confidence across evolutions
    violation_rate: float  # Constraint violation rate (0.0-1.0)
    
    @property
    def maturity_score(self) -> float:
        """
        Calculate overall maturity score (0.0-1.0).
        
        Higher score = more mature = eligible for relaxed constraints.
        """
        # Weight factors
        evolution_factor = min(self.evolution_count / 50, 1.0) * 0.25  # Max at 50 evolutions
        stability_factor = self.stability_score * 0.25
        age_factor = min(self.age_days / 90, 1.0) * 0.20  # Max at 90 days
        confidence_factor = self.confidence_avg * 0.15
        violation_factor = (1.0 - self.violation_rate) * 0.15  # Fewer violations = better
        
        score = (
            evolution_factor +
            stability_factor +
            age_factor +
            confidence_factor +
            violation_factor
        )
        
        return min(score, 1.0)
    
    @property
    def maturity_level(self) -> str:
        """Get maturity level label."""
        score = self.maturity_score
        if score >= 0.75:
            return "mature"
        elif score >= 0.50:
            return "developing"
        elif score >= 0.25:
            return "emerging"
        else:
            return "new"


class PersonaAwareConstraintManager:
    """
    Manages constraint relaxation based on persona maturity.
    """
    
    def __init__(self):
        """Initialize the manager."""
        self.relaxation_rules = self._initialize_relaxation_rules()
    
    def _initialize_relaxation_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Define relaxation rules for different maturity levels.
        
        Returns:
            Dictionary mapping maturity levels to relaxation settings
        """
        return {
            "new": {
                "meta_reasoning_strictness": 1.0,  # Full strictness
                "grounding_strictness": 1.0,
                "sensory_strictness": 1.0,
                "identity_strictness": 1.0,
                "constraint_priority_threshold": "HIGH",  # Only HIGH+ constraints
                "allow_metaphor": False,
                "description": "New persona - full constraint enforcement"
            },
            "emerging": {
                "meta_reasoning_strictness": 0.9,
                "grounding_strictness": 1.0,  # Keep grounding strict
                "sensory_strictness": 0.85,
                "identity_strictness": 1.0,  # Keep identity strict
                "constraint_priority_threshold": "HIGH",
                "allow_metaphor": False,
                "description": "Emerging persona - slight relaxation on non-critical constraints"
            },
            "developing": {
                "meta_reasoning_strictness": 0.75,  # More relaxed
                "grounding_strictness": 0.9,
                "sensory_strictness": 0.7,
                "identity_strictness": 1.0,  # Always strict
                "constraint_priority_threshold": "MEDIUM",  # Include MEDIUM+ constraints
                "allow_metaphor": True,  # Allow metaphorical language
                "description": "Developing persona - moderate constraint relaxation"
            },
            "mature": {
                "meta_reasoning_strictness": 0.5,  # Significantly relaxed
                "grounding_strictness": 0.8,
                "sensory_strictness": 0.5,
                "identity_strictness": 1.0,  # Never relax identity
                "constraint_priority_threshold": "MEDIUM",
                "allow_metaphor": True,
                "description": "Mature persona - relaxed constraints with maintained identity"
            }
        }
    
    def calculate_persona_maturity(
        self,
        persona_data: Dict[str, Any],
        violation_history: Optional[List[Dict[str, Any]]] = None
    ) -> PersonaMaturityMetrics:
        """
        Calculate maturity metrics for a persona.
        
        Args:
            persona_data: Dictionary with persona attributes
            violation_history: Optional list of recent violations
            
        Returns:
            PersonaMaturityMetrics object
        """
        # Extract data
        evolution_count = persona_data.get("evolution_count", 0)
        stability_score = persona_data.get("stability_score", 1.0)
        
        # Calculate age
        creation_date_str = persona_data.get("creation_date")
        if creation_date_str:
            try:
                creation_date = datetime.fromisoformat(creation_date_str.replace('Z', '+00:00'))
                age_days = (datetime.now(timezone.utc) - creation_date).days
            except (ValueError, AttributeError):
                age_days = 0
        else:
            age_days = 0
        
        # Calculate average confidence from evolutions
        evolutions = persona_data.get("evolutions", [])
        if evolutions:
            confidence_avg = sum(e.get("confidence", 0.5) for e in evolutions) / len(evolutions)
        else:
            confidence_avg = 0.5  # Default
        
        # Calculate violation rate from history
        if violation_history:
            # Count violations in last 100 interactions
            recent_violations = len([v for v in violation_history[-100:] if not v.get("auto_cleaned", False)])
            violation_rate = recent_violations / 100
        else:
            violation_rate = 0.0  # Assume no violations if no history
        
        return PersonaMaturityMetrics(
            evolution_count=evolution_count,
            stability_score=stability_score,
            age_days=age_days,
            confidence_avg=confidence_avg,
            violation_rate=violation_rate
        )
    
    def get_relaxation_settings(
        self,
        maturity: PersonaMaturityMetrics
    ) -> Dict[str, Any]:
        """
        Get constraint relaxation settings for a persona.
        
        Args:
            maturity: PersonaMaturityMetrics object
            
        Returns:
            Dictionary with relaxation settings
        """
        level = maturity.maturity_level
        settings = self.relaxation_rules[level].copy()
        settings["maturity_score"] = maturity.maturity_score
        settings["maturity_level"] = level
        
        logger.debug(
            f"Persona maturity: {level} (score={maturity.maturity_score:.2f}), "
            f"meta_reasoning_strictness={settings['meta_reasoning_strictness']}"
        )
        
        return settings
    
    def should_relax_constraint(
        self,
        constraint_type: str,
        maturity: PersonaMaturityMetrics
    ) -> bool:
        """
        Determine if a specific constraint should be relaxed.
        
        Args:
            constraint_type: Type of constraint ("meta_reasoning", "grounding", etc.)
            maturity: PersonaMaturityMetrics object
            
        Returns:
            True if constraint should be relaxed
        """
        settings = self.get_relaxation_settings(maturity)
        
        # Identity constraints never relax
        if constraint_type == "identity":
            return False
        
        # Check strictness threshold
        strictness_key = f"{constraint_type}_strictness"
        strictness = settings.get(strictness_key, 1.0)
        
        # Relax if strictness < 1.0
        return strictness < 1.0
    
    def apply_relaxation_to_constraint_set(
        self,
        constraint_set: Any,  # ConstraintSet from composition module
        persona_data: Dict[str, Any],
        violation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        """
        Apply persona-aware relaxation to a constraint set.
        
        Args:
            constraint_set: The ConstraintSet to adjust
            persona_data: Persona data dictionary
            violation_history: Optional violation history
            
        Returns:
            Modified ConstraintSet with relaxed constraints
        """
        from .composition import ConstraintPriority
        
        # Calculate maturity
        maturity = self.calculate_persona_maturity(persona_data, violation_history)
        settings = self.get_relaxation_settings(maturity)
        
        # Map threshold to enum
        threshold_map = {
            "CRITICAL": ConstraintPriority.CRITICAL,
            "HIGH": ConstraintPriority.HIGH,
            "MEDIUM": ConstraintPriority.MEDIUM,
            "LOW": ConstraintPriority.LOW
        }
        priority_threshold = threshold_map.get(
            settings["constraint_priority_threshold"],
            ConstraintPriority.HIGH
        )
        
        # Filter constraints by priority threshold
        for constraint in constraint_set.constraints:
            if constraint.priority < priority_threshold:
                # Disable low-priority constraints for mature personas
                constraint.enabled = False
                logger.debug(
                    f"Disabled constraint '{constraint.name}' "
                    f"(priority={constraint.priority}, threshold={priority_threshold})"
                )
        
        # Adjust specific constraint strictness
        for constraint in constraint_set.constraints:
            if not constraint.enabled:
                continue
            
            # Check if this constraint type should be relaxed
            constraint_category = constraint.category.lower()
            
            if "meta" in constraint_category or "reasoning" in constraint_category:
                strictness = settings["meta_reasoning_strictness"]
                if strictness < 1.0:
                    # Could modify constraint content to be less strict
                    # For now, just log
                    logger.debug(
                        f"Meta-reasoning constraint '{constraint.name}' "
                        f"strictness={strictness:.2f}"
                    )
            
            if "sensory" in constraint_category:
                strictness = settings["sensory_strictness"]
                if strictness < 1.0:
                    logger.debug(
                        f"Sensory constraint '{constraint.name}' "
                        f"strictness={strictness:.2f}"
                    )
        
        # Add maturity metadata
        constraint_set.metadata["persona_maturity"] = {
            "score": maturity.maturity_score,
            "level": maturity.maturity_level,
            "relaxation_applied": True,
            "settings": settings
        }
        
        return constraint_set
    
    def get_validator_config(
        self,
        persona_data: Dict[str, Any],
        violation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Get validator configuration adjusted for persona maturity.
        
        Args:
            persona_data: Persona data dictionary
            violation_history: Optional violation history
            
        Returns:
            Configuration dictionary for validators
        """
        maturity = self.calculate_persona_maturity(persona_data, violation_history)
        settings = self.get_relaxation_settings(maturity)
        
        return {
            "allow_metaphor": settings["allow_metaphor"],
            "sensory_strict": settings["sensory_strictness"] >= 0.9,
            "grounding_strict": settings["grounding_strictness"] >= 0.9,
            "meta_reasoning_detection": settings["meta_reasoning_strictness"] >= 0.8,
            "maturity_level": maturity.maturity_level,
            "maturity_score": maturity.maturity_score
        }


# Global manager instance
_global_persona_aware_manager: Optional[PersonaAwareConstraintManager] = None


def get_persona_aware_manager() -> PersonaAwareConstraintManager:
    """Get or create the global persona-aware constraint manager."""
    global _global_persona_aware_manager
    if _global_persona_aware_manager is None:
        _global_persona_aware_manager = PersonaAwareConstraintManager()
    return _global_persona_aware_manager


def apply_persona_relaxation(
    constraint_set: Any,
    persona_data: Dict[str, Any],
    violation_history: Optional[List[Dict[str, Any]]] = None
) -> Any:
    """
    Convenience function to apply persona-aware constraint relaxation.
    
    Args:
        constraint_set: The ConstraintSet to adjust
        persona_data: Persona data dictionary
        violation_history: Optional violation history
        
    Returns:
        Modified ConstraintSet
    """
    manager = get_persona_aware_manager()
    return manager.apply_relaxation_to_constraint_set(
        constraint_set,
        persona_data,
        violation_history
    )


def get_persona_validator_config(
    persona_data: Dict[str, Any],
    violation_history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Convenience function to get validator config for a persona.
    
    Args:
        persona_data: Persona data dictionary
        violation_history: Optional violation history
        
    Returns:
        Validator configuration dictionary
    """
    manager = get_persona_aware_manager()
    return manager.get_validator_config(persona_data, violation_history)
