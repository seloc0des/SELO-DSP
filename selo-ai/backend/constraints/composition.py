"""
Constraint Composition Framework

Provides declarative constraint composition with conflict resolution,
priority management, and telemetry integration.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger("selo.constraints.composition")


class ConstraintPriority(IntEnum):
    """Priority levels for constraint enforcement."""
    CRITICAL = 100  # Identity, safety - never overridden
    HIGH = 75       # Grounding, truthfulness
    MEDIUM = 50     # Behavioral guidelines, style
    LOW = 25        # Suggestions, preferences
    INFO = 10       # Informational only


@dataclass
class Constraint:
    """
    Represents a single constraint with metadata for composition.
    """
    name: str
    content: str
    priority: ConstraintPriority
    category: str = "general"
    tags: Set[str] = field(default_factory=set)
    enabled: bool = True
    token_cost: Optional[int] = None
    
    def __post_init__(self):
        """Calculate token cost if not provided."""
        if self.token_cost is None:
            # Rough estimate: ~4 chars per token
            self.token_cost = len(self.content) // 4
    
    def format(self, compact: bool = False) -> str:
        """
        Format constraint for inclusion in prompt.
        
        Args:
            compact: If True, use minimal formatting
            
        Returns:
            Formatted constraint text
        """
        if compact:
            return self.content
        else:
            return f"[{self.category.upper()}] {self.content}"


@dataclass
class ConstraintSet:
    """
    A collection of constraints with composition and merging capabilities.
    """
    name: str
    constraints: List[Constraint] = field(default_factory=list)
    priority: ConstraintPriority = ConstraintPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the set."""
        self.constraints.append(constraint)
    
    def remove_constraint(self, name: str) -> bool:
        """Remove a constraint by name. Returns True if found and removed."""
        original_len = len(self.constraints)
        self.constraints = [c for c in self.constraints if c.name != name]
        return len(self.constraints) < original_len
    
    def get_constraint(self, name: str) -> Optional[Constraint]:
        """Get a constraint by name."""
        for constraint in self.constraints:
            if constraint.name == name:
                return constraint
        return None
    
    def filter_by_priority(self, min_priority: ConstraintPriority) -> List[Constraint]:
        """Get all constraints with priority >= min_priority."""
        return [c for c in self.constraints if c.enabled and c.priority >= min_priority]
    
    def filter_by_category(self, category: str) -> List[Constraint]:
        """Get all constraints in a specific category."""
        return [c for c in self.constraints if c.enabled and c.category == category]
    
    def filter_by_tags(self, tags: Set[str], match_all: bool = False) -> List[Constraint]:
        """
        Get constraints matching tags.
        
        Args:
            tags: Set of tags to match
            match_all: If True, constraint must have all tags. If False, any tag matches.
        """
        result = []
        for c in self.constraints:
            if not c.enabled:
                continue
            if match_all:
                if tags.issubset(c.tags):
                    result.append(c)
            else:
                if tags.intersection(c.tags):
                    result.append(c)
        return result
    
    def total_token_cost(self, include_disabled: bool = False) -> int:
        """Calculate total token cost of all constraints."""
        constraints = self.constraints if include_disabled else [c for c in self.constraints if c.enabled]
        return sum(c.token_cost or 0 for c in constraints)
    
    def merge(
        self, 
        other: 'ConstraintSet',
        conflict_resolution: str = "priority"
    ) -> 'ConstraintSet':
        """
        Merge another constraint set into this one.
        
        Args:
            other: The constraint set to merge
            conflict_resolution: How to handle conflicts:
                - "priority": Higher priority constraint wins
                - "first": Keep existing constraint
                - "last": Use new constraint
                - "combine": Include both
                
        Returns:
            New merged constraint set
        """
        merged_name = f"{self.name}+{other.name}"
        merged = ConstraintSet(
            name=merged_name,
            priority=max(self.priority, other.priority),
            metadata={
                "merged_from": [self.name, other.name],
                "conflict_resolution": conflict_resolution
            }
        )
        
        # Index existing constraints by name
        constraint_map: Dict[str, Constraint] = {c.name: c for c in self.constraints}
        
        # Add all from self
        for constraint in self.constraints:
            merged.add_constraint(constraint)
        
        # Add from other, resolving conflicts
        for constraint in other.constraints:
            existing = constraint_map.get(constraint.name)
            
            if existing is None:
                # No conflict, add directly
                merged.add_constraint(constraint)
            else:
                # Conflict - apply resolution strategy
                if conflict_resolution == "priority":
                    if constraint.priority > existing.priority:
                        # Replace with higher priority
                        merged.remove_constraint(existing.name)
                        merged.add_constraint(constraint)
                        logger.debug(
                            f"Constraint conflict: '{constraint.name}' - using higher priority version"
                        )
                elif conflict_resolution == "first":
                    # Keep existing, ignore new
                    pass
                elif conflict_resolution == "last":
                    # Replace with new
                    merged.remove_constraint(existing.name)
                    merged.add_constraint(constraint)
                elif conflict_resolution == "combine":
                    # Add new with modified name
                    combined_constraint = Constraint(
                        name=f"{constraint.name}_alt",
                        content=constraint.content,
                        priority=constraint.priority,
                        category=constraint.category,
                        tags=constraint.tags,
                        enabled=constraint.enabled
                    )
                    merged.add_constraint(combined_constraint)
        
        return merged
    
    def optimize(self, max_tokens: Optional[int] = None) -> 'ConstraintSet':
        """
        Optimize constraint set to fit within token budget.
        
        Args:
            max_tokens: Maximum token budget (None = no limit)
            
        Returns:
            Optimized constraint set (may disable low-priority constraints)
        """
        if max_tokens is None:
            return self
        
        # Sort constraints by priority (descending)
        sorted_constraints = sorted(
            self.constraints,
            key=lambda c: c.priority,
            reverse=True
        )
        
        optimized = ConstraintSet(
            name=f"{self.name}_optimized",
            priority=self.priority,
            metadata={**self.metadata, "optimized": True, "max_tokens": max_tokens}
        )
        
        accumulated_tokens = 0
        for constraint in sorted_constraints:
            if accumulated_tokens + constraint.token_cost <= max_tokens:
                optimized.add_constraint(constraint)
                accumulated_tokens += constraint.token_cost
            else:
                # Token budget exceeded - disable this and lower priority constraints
                logger.info(
                    f"Constraint '{constraint.name}' (priority={constraint.priority}) "
                    f"excluded to fit {max_tokens} token budget"
                )
        
        logger.info(
            f"Optimized constraint set: {len(optimized.constraints)}/{len(self.constraints)} "
            f"constraints, {accumulated_tokens}/{max_tokens} tokens"
        )
        
        return optimized
    
    def format(
        self,
        compact: bool = False,
        priority_threshold: Optional[ConstraintPriority] = None,
        include_header: bool = True
    ) -> str:
        """
        Format the entire constraint set for inclusion in a prompt.
        
        Args:
            compact: Use compact formatting
            priority_threshold: Only include constraints >= this priority
            include_header: Include section header
            
        Returns:
            Formatted constraint text
        """
        if priority_threshold is not None:
            constraints = self.filter_by_priority(priority_threshold)
        else:
            constraints = [c for c in self.constraints if c.enabled]
        
        if not constraints:
            return ""
        
        lines = []
        
        if include_header and not compact:
            lines.append("=" * 80)
            lines.append(f"CONSTRAINTS: {self.name}")
            lines.append("=" * 80)
            lines.append("")
        
        for constraint in constraints:
            lines.append(constraint.format(compact=compact))
            if not compact:
                lines.append("")
        
        if include_header and not compact:
            lines.append("=" * 80)
        
        return "\n".join(lines)


class ConstraintComposer:
    """
    Orchestrates constraint composition across the system.
    """
    
    def __init__(self):
        """Initialize the constraint composer."""
        self.constraint_sets: Dict[str, ConstraintSet] = {}
        self.telemetry: List[Dict[str, Any]] = []
    
    def register_set(self, constraint_set: ConstraintSet) -> None:
        """Register a constraint set for composition."""
        self.constraint_sets[constraint_set.name] = constraint_set
        logger.info(f"Registered constraint set: '{constraint_set.name}' ({len(constraint_set.constraints)} constraints)")
    
    def get_set(self, name: str) -> Optional[ConstraintSet]:
        """Get a registered constraint set by name."""
        return self.constraint_sets.get(name)
    
    def compose(
        self,
        set_names: List[str],
        conflict_resolution: str = "priority",
        max_tokens: Optional[int] = None,
        priority_threshold: Optional[ConstraintPriority] = None
    ) -> ConstraintSet:
        """
        Compose multiple constraint sets into one.
        
        Args:
            set_names: Names of constraint sets to compose
            conflict_resolution: How to resolve conflicts
            max_tokens: Optional token budget
            priority_threshold: Minimum priority to include
            
        Returns:
            Composed constraint set
        """
        if not set_names:
            return ConstraintSet(name="empty")
        
        # Get first set
        composed = self.constraint_sets.get(set_names[0])
        if composed is None:
            logger.warning(f"Constraint set '{set_names[0]}' not found")
            return ConstraintSet(name="empty")
        
        # Make a copy
        composed = ConstraintSet(
            name=composed.name,
            constraints=list(composed.constraints),
            priority=composed.priority,
            metadata=dict(composed.metadata)
        )
        
        # Merge remaining sets
        for name in set_names[1:]:
            constraint_set = self.constraint_sets.get(name)
            if constraint_set is None:
                logger.warning(f"Constraint set '{name}' not found")
                continue
            composed = composed.merge(constraint_set, conflict_resolution=conflict_resolution)
        
        # Apply filters
        if priority_threshold is not None:
            # Disable constraints below threshold
            for constraint in composed.constraints:
                if constraint.priority < priority_threshold:
                    constraint.enabled = False
        
        # Optimize for token budget if specified
        if max_tokens is not None:
            composed = composed.optimize(max_tokens)
        
        # Record telemetry
        self.telemetry.append({
            "action": "compose",
            "sets": set_names,
            "result_name": composed.name,
            "constraint_count": len([c for c in composed.constraints if c.enabled]),
            "total_tokens": composed.total_token_cost(),
            "max_tokens": max_tokens,
            "conflict_resolution": conflict_resolution
        })
        
        return composed
    
    def get_telemetry(self) -> List[Dict[str, Any]]:
        """Get composition telemetry data."""
        return self.telemetry
    
    def clear_telemetry(self) -> None:
        """Clear telemetry data."""
        self.telemetry.clear()


# Global composer instance
_global_composer: Optional[ConstraintComposer] = None


def get_constraint_composer() -> ConstraintComposer:
    """Get or create the global constraint composer instance."""
    global _global_composer
    if _global_composer is None:
        _global_composer = ConstraintComposer()
    return _global_composer
