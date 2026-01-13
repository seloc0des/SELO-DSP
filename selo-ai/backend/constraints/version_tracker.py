"""
Constraint Version Tracking

Tracks which constraint versions were used for each generation,
enabling A/B testing and regression analysis.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger("selo.constraints.version_tracker")


class ConstraintVersionTracker:
    """
    Tracks constraint versions used in LLM generations.
    
    Integrates with ABTestManager for experimentation and telemetry
    for monitoring constraint effectiveness.
    """
    
    def __init__(self):
        """Initialize tracker with references to constraint classes."""
        self._constraint_versions = self._gather_constraint_versions()
    
    def _gather_constraint_versions(self) -> Dict[str, str]:
        """Gather VERSION strings from all constraint classes."""
        versions = {}
        
        try:
            from .core_constraints import CoreConstraints
            versions['core'] = getattr(CoreConstraints, 'VERSION', 'unknown')
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not get CoreConstraints version: {e}")
            versions['core'] = 'unknown'
        
        try:
            from .identity_constraints import IdentityConstraints
            versions['identity'] = getattr(IdentityConstraints, 'VERSION', 'unknown')
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not get IdentityConstraints version: {e}")
            versions['identity'] = 'unknown'
        
        try:
            from .ethical_guardrails import EthicalGuardrails
            versions['ethical'] = getattr(EthicalGuardrails, 'VERSION', 'unknown')
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not get EthicalGuardrails version: {e}")
            versions['ethical'] = 'unknown'
        
        try:
            from .behavioral_guidelines import BehavioralGuidelines
            versions['behavioral'] = getattr(BehavioralGuidelines, 'VERSION', 'unknown')
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not get BehavioralGuidelines version: {e}")
            versions['behavioral'] = 'unknown'
        
        try:
            from .grounding_constraints import GroundingConstraints
            versions['grounding'] = getattr(GroundingConstraints, 'VERSION', '1.0.0')
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not get GroundingConstraints version: {e}")
            versions['grounding'] = '1.0.0'
        
        return versions
    
    def get_constraint_metadata(
        self,
        context_type: str,
        persona_name: Optional[str] = None,
        token_budget: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get metadata about constraints used for a generation.
        
        Args:
            context_type: Type of generation (bootstrap, reflection, conversation)
            persona_name: Persona name if applicable
            token_budget: Token budget if applied
            
        Returns:
            Metadata dictionary to attach to generation record
        """
        metadata = {
            'constraint_versions': self._constraint_versions.copy(),
            'context_type': context_type,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'unified_system_enabled': True  # Phase 3 implementation flag
        }
        
        if persona_name:
            metadata['persona_name'] = persona_name
        
        if token_budget is not None:
            metadata['token_budget'] = token_budget
            metadata['optimization_applied'] = True
        else:
            metadata['optimization_applied'] = False
        
        return metadata
    
    def record_generation(
        self,
        generation_id: str,
        context_type: str,
        persona_name: Optional[str] = None,
        token_budget: Optional[int] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record constraint versions used for a specific generation.
        
        Args:
            generation_id: Unique identifier for the generation
            context_type: Type of generation
            persona_name: Persona name if applicable
            token_budget: Token budget if applied
            additional_metadata: Additional metadata to include
        """
        metadata = self.get_constraint_metadata(
            context_type=context_type,
            persona_name=persona_name,
            token_budget=token_budget
        )
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Log for tracking (could be extended to write to database)
        logger.info(
            f"Constraint version tracking - Generation: {generation_id}, "
            f"Context: {context_type}, Versions: {metadata['constraint_versions']}"
        )
        
        # TODO: Integrate with telemetry system to persist metadata
        try:
            from .telemetry import record_validation
            record_validation(
                validation_type='constraint_version_tracking',
                result=True,
                metadata=metadata
            )
        except Exception as e:
            logger.debug(f"Could not record to telemetry: {e}")
    
    def check_for_updates(self) -> Dict[str, bool]:
        """
        Check if constraint versions have been updated since initialization.
        
        Returns:
            Dictionary mapping constraint class to whether it changed
        """
        current_versions = self._gather_constraint_versions()
        changes = {}
        
        for key, old_version in self._constraint_versions.items():
            new_version = current_versions.get(key, 'unknown')
            changes[key] = (old_version != new_version)
            
            if changes[key]:
                logger.info(
                    f"Constraint version update detected: {key} "
                    f"{old_version} -> {new_version}"
                )
        
        return changes
    
    def refresh_versions(self) -> None:
        """Refresh constraint version cache (call after updating constraint classes)."""
        old_versions = self._constraint_versions.copy()
        self._constraint_versions = self._gather_constraint_versions()
        
        # Log any changes
        for key, new_version in self._constraint_versions.items():
            old_version = old_versions.get(key, 'unknown')
            if old_version != new_version:
                logger.info(
                    f"Constraint version refreshed: {key} "
                    f"{old_version} -> {new_version}"
                )


# Singleton accessor
_version_tracker = None

def get_constraint_version_tracker() -> ConstraintVersionTracker:
    """Get the singleton ConstraintVersionTracker instance."""
    global _version_tracker
    if _version_tracker is None:
        _version_tracker = ConstraintVersionTracker()
    return _version_tracker
