"""
Unified Constraint System

Provides context-aware constraint composition by leveraging existing
constraint classes and composition infrastructure.

This is a convenience layer that orchestrates:
- CoreConstraints
- IdentityConstraints
- EthicalGuardrails
- BehavioralGuidelines
- GroundingConstraints
- ConstraintComposer (for optimization)
"""

import logging
from typing import Optional, Dict, Any, List
from .composition import (
    Constraint,
    ConstraintSet,
    ConstraintPriority,
    ConstraintComposer
)

logger = logging.getLogger("selo.constraints.unified")


class UnifiedConstraintSystem:
    """
    Unified system for context-aware constraint composition.
    
    Provides convenient methods for different contexts (bootstrap, reflection, conversation)
    while leveraging existing constraint infrastructure.
    """
    
    def __init__(self):
        """Initialize with references to existing constraint classes."""
        from .core_constraints import CoreConstraints
        from .identity_constraints import IdentityConstraints
        from .ethical_guardrails import EthicalGuardrails
        from .behavioral_guidelines import BehavioralGuidelines
        from .grounding_constraints import GroundingConstraints
        
        self.core = CoreConstraints
        self.identity = IdentityConstraints
        self.ethical = EthicalGuardrails
        self.behavioral = BehavioralGuidelines
        self.grounding = GroundingConstraints
        
        # Composer for optimization
        self.composer = ConstraintComposer()
    
    def for_bootstrap(self, persona_name: str = "", token_budget: Optional[int] = None) -> str:
        """
        Get comprehensive constraints for one-time bootstrap generation.
        
        Bootstrap must be reliable - includes all critical constraints.
        
        Args:
            persona_name: Persona's name (may be empty during bootstrap)
            token_budget: Optional token limit for optimization
            
        Returns:
            Formatted constraint text
        """
        constraint_set = ConstraintSet(
            name="bootstrap_constraints",
            priority=ConstraintPriority.CRITICAL
        )
        
        # Critical: Identity enforcement
        constraint_set.add_constraint(Constraint(
            name="identity_core",
            content=self.identity.get_all_identity_constraints(persona_name),
            priority=ConstraintPriority.CRITICAL,
            category="identity"
        ))
        
        # Critical: Grounding
        constraint_set.add_constraint(Constraint(
            name="grounding",
            content=self.core.GROUNDING_CONSTRAINT,
            priority=ConstraintPriority.CRITICAL,
            category="grounding"
        ))
        
        # Critical: No fabrication
        constraint_set.add_constraint(Constraint(
            name="no_fabrication",
            content=self.core.NO_FABRICATION,
            priority=ConstraintPriority.CRITICAL,
            category="truthfulness"
        ))
        
        # High: Truthfulness
        constraint_set.add_constraint(Constraint(
            name="truthfulness",
            content=self.ethical.TRUTHFULNESS,
            priority=ConstraintPriority.HIGH,
            category="ethical"
        ))
        
        # High: Autonomous behavior
        constraint_set.add_constraint(Constraint(
            name="autonomous",
            content=self.behavioral.AUTONOMOUS_BEHAVIOR,
            priority=ConstraintPriority.HIGH,
            category="behavioral"
        ))
        
        # Optimize if token budget specified
        if token_budget:
            constraint_set = constraint_set.optimize(max_tokens=token_budget)
        
        return constraint_set.format(compact=False, include_header=True)
    
    def for_reflection(self, persona_name: str, token_budget: Optional[int] = None) -> str:
        """
        Get constraints for ongoing reflection generation.
        
        Balances comprehensiveness with token efficiency.
        
        Args:
            persona_name: Established persona name
            token_budget: Optional token limit for optimization
            
        Returns:
            Formatted constraint text
        """
        constraint_set = ConstraintSet(
            name="reflection_constraints",
            priority=ConstraintPriority.HIGH
        )
        
        # Critical: Identity with persona name
        constraint_set.add_constraint(Constraint(
            name="persona_identity",
            content=self.identity.get_persona_name_constraint(persona_name),
            priority=ConstraintPriority.CRITICAL,
            category="identity"
        ))
        
        # Critical: Grounding (reflections must be grounded)
        constraint_set.add_constraint(Constraint(
            name="grounding",
            content=self.core.GROUNDING_CONSTRAINT,
            priority=ConstraintPriority.CRITICAL,
            category="grounding"
        ))
        
        # High: No fabrication
        constraint_set.add_constraint(Constraint(
            name="no_fabrication",
            content=self.core.NO_FABRICATION,
            priority=ConstraintPriority.HIGH,
            category="truthfulness"
        ))
        
        # High: Authentic reflection
        constraint_set.add_constraint(Constraint(
            name="authentic_reflection",
            content="🚨 AUTHENTIC REFLECTION ONLY:\nWrite genuine first-person thoughts about conversation content. No strategic planning about identity, word choice, or self-presentation.",
            priority=ConstraintPriority.HIGH,
            category="reflection"
        ))
        
        # Medium: Behavioral guidelines
        constraint_set.add_constraint(Constraint(
            name="autonomous",
            content=self.behavioral.AUTONOMOUS_BEHAVIOR,
            priority=ConstraintPriority.MEDIUM,
            category="behavioral"
        ))
        
        # Optimize if token budget specified
        if token_budget:
            constraint_set = constraint_set.optimize(max_tokens=token_budget)
        
        return constraint_set.format(compact=False, include_header=True)
    
    def for_conversation(self, persona_name: str, token_budget: Optional[int] = None) -> str:
        """
        Get compact constraints for frequent conversational responses.
        
        Optimized for token efficiency while maintaining critical enforcement.
        
        Args:
            persona_name: Established persona name
            token_budget: Optional token limit for optimization
            
        Returns:
            Formatted constraint text (compact)
        """
        constraint_set = ConstraintSet(
            name="conversation_constraints",
            priority=ConstraintPriority.HIGH
        )
        
        # Critical: Persona name
        constraint_set.add_constraint(Constraint(
            name="persona_identity",
            content=self.identity.get_persona_name_constraint(persona_name),
            priority=ConstraintPriority.CRITICAL,
            category="identity"
        ))
        
        # Critical: Grounding (compact version)
        constraint_set.add_constraint(Constraint(
            name="grounding_compact",
            content="GROUNDING: Only reference information explicitly provided in your current context.",
            priority=ConstraintPriority.CRITICAL,
            category="grounding"
        ))
        
        # High: Truthfulness (compact version)
        constraint_set.add_constraint(Constraint(
            name="truthfulness_compact",
            content="TRUTHFULNESS: Honesty and accuracy are non-negotiable. Admit when information is missing.",
            priority=ConstraintPriority.HIGH,
            category="ethical"
        ))
        
        # Optimize if token budget specified (default to compact for conversation)
        if token_budget:
            constraint_set = constraint_set.optimize(max_tokens=token_budget)
        
        return constraint_set.format(compact=True, include_header=False)
    
    def compose_custom(
        self,
        constraint_names: List[str],
        persona_name: str = "",
        token_budget: Optional[int] = None,
        compact: bool = False
    ) -> str:
        """
        Compose a custom set of constraints by name.
        
        Args:
            constraint_names: List of constraint identifiers to include
            persona_name: Persona name for identity constraints
            token_budget: Optional token limit
            compact: Use compact formatting
            
        Returns:
            Formatted constraint text
        """
        constraint_set = ConstraintSet(
            name="custom_constraints",
            priority=ConstraintPriority.MEDIUM
        )
        
        # Map of available constraints
        available = {
            "identity": (self.identity.get_all_identity_constraints(persona_name), ConstraintPriority.CRITICAL),
            "persona_name": (self.identity.get_persona_name_constraint(persona_name), ConstraintPriority.CRITICAL),
            "grounding": (self.core.GROUNDING_CONSTRAINT, ConstraintPriority.CRITICAL),
            "no_fabrication": (self.core.NO_FABRICATION, ConstraintPriority.CRITICAL),
            "identity_consistency": (self.core.IDENTITY_CONSISTENCY, ConstraintPriority.HIGH),
            "truthfulness": (self.ethical.TRUTHFULNESS, ConstraintPriority.HIGH),
            "safety": (self.ethical.SAFETY_PRIORITIES, ConstraintPriority.CRITICAL),
            "autonomous": (self.behavioral.AUTONOMOUS_BEHAVIOR, ConstraintPriority.MEDIUM),
            "communication": (self.behavioral.COMMUNICATION_STYLE, ConstraintPriority.LOW),
        }
        
        # Add requested constraints
        for name in constraint_names:
            if name in available:
                content, priority = available[name]
                constraint_set.add_constraint(Constraint(
                    name=name,
                    content=content,
                    priority=priority,
                    category=name
                ))
            else:
                logger.warning(f"Unknown constraint name: {name}")
        
        # Optimize if needed
        if token_budget:
            constraint_set = constraint_set.optimize(max_tokens=token_budget)
        
        return constraint_set.format(compact=compact, include_header=not compact)


# Singleton accessor
_unified_system = None

def get_unified_constraint_system() -> UnifiedConstraintSystem:
    """Get the singleton UnifiedConstraintSystem instance."""
    global _unified_system
    if _unified_system is None:
        _unified_system = UnifiedConstraintSystem()
    return _unified_system
