"""
Centralized validation helpers for consistent constraint checking across the codebase.

This module provides reusable validation patterns to eliminate code duplication
and ensure consistent error reporting.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_IdentityConstraints = None
_BoilerplateConstraints = None
_GroundingConstraints = None


def _get_identity_constraints():
    """Lazy load IdentityConstraints to avoid circular imports."""
    global _IdentityConstraints
    if _IdentityConstraints is None:
        try:
            from ..constraints import IdentityConstraints
            _IdentityConstraints = IdentityConstraints
        except ImportError:
            from backend.constraints import IdentityConstraints
            _IdentityConstraints = IdentityConstraints
    return _IdentityConstraints


def _get_boilerplate_constraints():
    """Lazy load BoilerplateConstraints to avoid circular imports."""
    global _BoilerplateConstraints
    if _BoilerplateConstraints is None:
        try:
            from ..constraints.boilerplate_constraints import BoilerplateConstraints
            _BoilerplateConstraints = BoilerplateConstraints
        except ImportError:
            from backend.constraints.boilerplate_constraints import BoilerplateConstraints
            _BoilerplateConstraints = BoilerplateConstraints
    return _BoilerplateConstraints


def _get_grounding_constraints():
    """Lazy load GroundingConstraints to avoid circular imports."""
    global _GroundingConstraints
    if _GroundingConstraints is None:
        try:
            from ..constraints.grounding_constraints import GroundingConstraints
            _GroundingConstraints = GroundingConstraints
        except ImportError:
            from backend.constraints.grounding_constraints import GroundingConstraints
            _GroundingConstraints = GroundingConstraints
    return _GroundingConstraints


class ValidationHelper:
    """Centralized validation helper with standardized error reporting."""
    
    @staticmethod
    def validate_text_compliance(
        text: str,
        context: str = "",
        ignore_persona_name: bool = True,
        persona_name: str = ""
    ) -> Tuple[bool, List[str]]:
        """
        Validate text for identity compliance with standardized error reporting.
        
        Args:
            text: Text to validate
            context: Context label for error messages (e.g., "summary", "trait.description")
            ignore_persona_name: Whether to ignore persona name in validation
            persona_name: Persona name to whitelist if ignore_persona_name=True
            
        Returns:
            Tuple of (is_compliant, list_of_formatted_violations)
            
        Example:
            >>> is_compliant, violations = ValidationHelper.validate_text_compliance(
            ...     text="I am an AI assistant",
            ...     context="persona.core_values"
            ... )
            >>> print(violations)
            ["persona.core_values: identity violations ['AI', 'assistant'] in 'I am an AI assistant...'"]
        """
        if not text or not isinstance(text, str):
            return False, [f"{context}: text must be non-empty string"]
        
        IdentityConstraints = _get_identity_constraints()
        
        is_compliant, violation_terms = IdentityConstraints.check_compliance(
            text=text,
            ignore_persona_name=ignore_persona_name,
            persona_name=persona_name
        )
        
        formatted_violations = []
        if not is_compliant:
            text_preview = text[:50] + "..." if len(text) > 50 else text
            formatted_violations.append(
                f"{context}: identity violations {violation_terms} in '{text_preview}'"
            )
        
        return is_compliant, formatted_violations
    
    @staticmethod
    def validate_text_with_boilerplate(
        text: str,
        stage: str,
        context: str = ""
    ) -> Tuple[bool, List[str]]:
        """
        Validate text for both identity compliance and boilerplate phrases.
        
        Args:
            text: Text to validate
            stage: Stage identifier for boilerplate checking (e.g., "seed", "traits")
            context: Context label for error messages
            
        Returns:
            Tuple of (is_compliant, list_of_formatted_violations)
        """
        if not text or not isinstance(text, str):
            return False, [f"{context}: text must be non-empty string"]
        
        violations = []
        
        # Check identity compliance
        is_identity_compliant, identity_violations = ValidationHelper.validate_text_compliance(
            text=text,
            context=context
        )
        violations.extend(identity_violations)
        
        # Check boilerplate
        BoilerplateConstraints = _get_boilerplate_constraints()
        has_boilerplate, boilerplate_terms = BoilerplateConstraints.check_boilerplate(text, stage)
        
        if has_boilerplate:
            text_preview = text[:50] + "..." if len(text) > 50 else text
            violations.append(
                f"{context}: boilerplate violations {boilerplate_terms} in '{text_preview}'"
            )
        
        is_compliant = is_identity_compliant and not has_boilerplate
        return is_compliant, violations
    
    @staticmethod
    def validate_text_with_fabrication(
        text: str,
        context: str = "",
        has_history: bool = False,
        context_stage: str = "summary"
    ) -> Tuple[bool, List[str]]:
        """
        Validate text for identity compliance and fabrication indicators.
        
        Args:
            text: Text to validate
            context: Context label for error messages
            has_history: Whether there is conversation history
            context_stage: Stage identifier for fabrication checking
            
        Returns:
            Tuple of (is_compliant, list_of_formatted_violations)
        """
        if not text or not isinstance(text, str):
            return False, [f"{context}: text must be non-empty string"]
        
        violations = []
        
        # Check identity compliance
        is_identity_compliant, identity_violations = ValidationHelper.validate_text_compliance(
            text=text,
            context=context
        )
        violations.extend(identity_violations)
        
        # Check fabrication
        GroundingConstraints = _get_grounding_constraints()
        has_fabrication, fabrication_terms = GroundingConstraints.check_fabrication(
            text=text,
            has_history=has_history,
            context_stage=context_stage
        )
        
        if has_fabrication:
            violations.append(
                f"{context}: fabrication detected - {fabrication_terms}"
            )
        
        is_compliant = is_identity_compliant and not has_fabrication
        return is_compliant, violations
    
    @staticmethod
    def validate_dict_recursive(
        data: Dict[str, Any],
        stage: str,
        check_boilerplate: bool = True,
        path: str = ""
    ) -> List[str]:
        """
        Recursively validate dictionary structure for compliance violations.
        
        This is particularly useful for validating bootstrap payloads where
        forbidden terms could appear at any nesting level.
        
        Args:
            data: Dictionary to validate (can be nested)
            stage: Stage identifier for boilerplate checking
            check_boilerplate: Whether to check for boilerplate phrases
            path: Current path in the structure (used internally for recursion)
            
        Returns:
            List of formatted violation messages
            
        Example:
            >>> violations = ValidationHelper.validate_dict_recursive(
            ...     data={"name": "TestBot", "traits": [{"description": "I am an AI"}]},
            ...     stage="bootstrap"
            ... )
            >>> print(violations)
            ["traits[0].description: identity violations ['AI'] in 'I am an AI...'"]
        """
        violations = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                violations.extend(
                    ValidationHelper.validate_dict_recursive(
                        value, stage, check_boilerplate, current_path
                    )
                )
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                violations.extend(
                    ValidationHelper.validate_dict_recursive(
                        item, stage, check_boilerplate, current_path
                    )
                )
        elif isinstance(data, str):
            if check_boilerplate:
                is_compliant, text_violations = ValidationHelper.validate_text_with_boilerplate(
                    text=data,
                    stage=stage,
                    context=path
                )
            else:
                is_compliant, text_violations = ValidationHelper.validate_text_compliance(
                    text=data,
                    context=path
                )
            violations.extend(text_violations)
        
        return violations
    
    @staticmethod
    def validate_score(
        score: Any,
        score_name: str = "score",
        min_value: float = 0.0,
        max_value: float = 1.0,
        allow_none: bool = False
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        """
        Validate that a score is within expected bounds.
        
        Args:
            score: Score value to validate
            score_name: Name of the score for error messages
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
            allow_none: Whether None is acceptable
            
        Returns:
            Tuple of (is_valid, normalized_score, error_message)
            
        Example:
            >>> is_valid, normalized, error = ValidationHelper.validate_score(1.5, "confidence")
            >>> print(error)
            "confidence must be between 0.0 and 1.0, got 1.5"
        """
        if score is None:
            if allow_none:
                return True, None, None
            return False, None, f"{score_name} cannot be None"
        
        try:
            normalized = float(score)
        except (TypeError, ValueError):
            return False, None, f"{score_name} must be numeric, got {type(score).__name__}"
        
        if normalized < min_value or normalized > max_value:
            return False, None, f"{score_name} must be between {min_value} and {max_value}, got {normalized}"
        
        return True, normalized, None
    
    @staticmethod
    def validate_scores_dict(
        data: Dict[str, Any],
        score_fields: List[str],
        context: str = ""
    ) -> Tuple[bool, Dict[str, float], List[str]]:
        """
        Validate multiple score fields in a dictionary.
        
        Args:
            data: Dictionary containing scores
            score_fields: List of field names to validate
            context: Context label for error messages
            
        Returns:
            Tuple of (all_valid, normalized_scores, error_messages)
        """
        normalized = {}
        errors = []
        all_valid = True
        
        for field in score_fields:
            if field in data:
                is_valid, norm_score, error = ValidationHelper.validate_score(
                    data[field],
                    score_name=f"{context}.{field}" if context else field
                )
                if not is_valid:
                    errors.append(error)
                    all_valid = False
                else:
                    normalized[field] = norm_score
        
        return all_valid, normalized, errors
    
    @staticmethod
    def clamp_score(
        score: Any,
        min_value: float = 0.0,
        max_value: float = 1.0,
        default: float = 0.5
    ) -> float:
        """
        Clamp a score to valid range, with fallback to default.
        
        Args:
            score: Score value to clamp
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            default: Default value if score is invalid
            
        Returns:
            Clamped score value
        """
        try:
            value = float(score)
            return max(min_value, min(max_value, value))
        except (TypeError, ValueError):
            return default
    
    @staticmethod
    def log_violations(
        violations: List[str],
        stage: str,
        level: str = "error"
    ) -> None:
        """
        Log validation violations with consistent formatting.
        
        Args:
            violations: List of violation messages
            stage: Stage identifier for context
            level: Log level ("error", "warning", "info")
        """
        if not violations:
            logger.info(f"Validation passed for {stage}")
            return
        
        log_func = getattr(logger, level, logger.error)
        log_func(f"Validation violations found in {stage}:")
        for violation in violations:
            log_func(f"  - {violation}")
