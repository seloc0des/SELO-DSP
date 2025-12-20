"""
Grounding Constraint Validation

Centralizes detection of fabricated history, unfounded experience claims, and
other violations of grounding principles. Ensures personas remain grounded in
actual context and don't hallucinate prior interactions.
"""

import re
from typing import List, Tuple


class GroundingConstraints:
    """
    Centralized grounding validation for persona and reflection generation.
    
    Prevents personas from fabricating history, claiming unfounded experiences,
    or referencing non-existent prior interactions.
    """
    
    # Fabrication indicators - phrases that suggest false history/memory
    FABRICATION_INDICATORS = [
        "i remember",
        "in the past",
        "previously",
        "before",
        "earlier",
        "my experience",
        "i have learned",
        "i used to",
        "when i was",
        "i have seen",
        "i have encountered",
        "from my experience",
        "in my experience",
    ]
    
    # Unfounded history phrases - references to non-existent prior interactions
    UNFOUNDED_HISTORY_PATTERNS = [
        r"\bmy previous interactions?\b",
        r"\bour (past|prior|earlier) (conversations?|discussions?|exchanges?)\b",
        r"\b(we|I) (discussed|talked about|spoke about) (this|that|it) (before|earlier|previously)\b",
        r"\bas (we|I) mentioned (before|earlier|previously)\b",
    ]
    
    # Compiled regex patterns for efficient checking
    _COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in UNFOUNDED_HISTORY_PATTERNS]
    
    @classmethod
    def check_fabrication(
        cls,
        text: str,
        has_history: bool = True,
        context_stage: str = "general"
    ) -> Tuple[bool, List[str]]:
        """
        Check text for fabricated history or unfounded experience claims.
        
        Args:
            text: Text to check for fabrication indicators
            has_history: Whether there is actual prior conversation history
            context_stage: Context stage (e.g., "first_contact", "ongoing")
            
        Returns:
            Tuple of (has_violations, list_of_violations)
        """
        if not isinstance(text, str):
            return False, []
        
        text_lower = text.lower()
        violations = []
        
        # Check fabrication indicators (memory/experience claims)
        for indicator in cls.FABRICATION_INDICATORS:
            if indicator in text_lower:
                violations.append(f"fabrication_indicator:{indicator}")
        
        # Check unfounded history patterns (only if no history exists)
        if not has_history or context_stage == "first_contact":
            for pattern in cls._COMPILED_PATTERNS:
                match = pattern.search(text)
                if match:
                    violations.append(f"unfounded_history:{match.group(0)}")
        
        has_violations = len(violations) > 0
        return has_violations, violations
    
    @classmethod
    def check_unfounded_history(cls, text: str, has_history: bool = False) -> Tuple[bool, List[str]]:
        """
        Specialized check for unfounded history references.
        
        More strict than check_fabrication - only checks for references to
        non-existent prior interactions.
        
        Args:
            text: Text to check
            has_history: Whether there is actual prior history
            
        Returns:
            Tuple of (has_violations, list_of_violations)
        """
        if not isinstance(text, str) or has_history:
            return False, []
        
        violations = []
        
        for pattern in cls._COMPILED_PATTERNS:
            match = pattern.search(text)
            if match:
                violations.append(f"unfounded_history:{match.group(0)}")
        
        has_violations = len(violations) > 0
        return has_violations, violations
    
    @classmethod
    def get_all_indicators(cls) -> List[str]:
        """Get all fabrication indicators for documentation/reference."""
        return list(cls.FABRICATION_INDICATORS)
    
    @classmethod
    def get_all_patterns(cls) -> List[str]:
        """Get all unfounded history patterns for documentation/reference."""
        return list(cls.UNFOUNDED_HISTORY_PATTERNS)
