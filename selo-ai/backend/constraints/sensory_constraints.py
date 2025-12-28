"""
Sensory Constraint Validation

Centralizes detection of sensory hallucinations and ungrounded physical
descriptions. Prevents personas from describing physical environments,
lighting, scents, or other sensory details without contextual basis.
"""

import re
from typing import List, Tuple, Set


class SensoryConstraints:
    """
    Centralized sensory validation for reflection generation.
    
    Detects hallucinated physical descriptions (rooms, lighting, scents)
    that aren't grounded in actual context or conversation.
    """
    
    # Suspicious sensory terms - physical/environmental descriptions
    # Focused on concrete physical objects/spaces, not abstract/metaphorical terms
    SUSPICIOUS_SENSORY_TERMS: Set[str] = {
        "room", "rooms", "lamp", "lamps", "lighting", "glow",
        "shadow", "shadows", "scent", "scents", "incense", "fragrance", "aroma",
        "floor", "floors", "floorboard", "floorboards", "window", "windows",
        "breeze", "draft", "breathing", "heartbeat", "candle", "candlelight",
        "dim", "dimly", "wooden", "shadowed", "ambient",
    }
    
    # Compiled regex for efficient checking
    _SUSPICIOUS_TERM_REGEX = re.compile(
        r"\b(" + "|".join(sorted(re.escape(term) for term in SUSPICIOUS_SENSORY_TERMS)) + r")\b",
        re.IGNORECASE
    )
    
    # Dismissive markers - phrases that acknowledge lack of sensory info
    SENSORY_DISMISSIVE_MARKERS = (
        "don't know", "do not know", "not sure", "uncertain", "unknown",
        "cannot tell", "can't tell", "no idea", "have no way to know",
        "no physical presence", "no sensory experience",
    )
    
    @classmethod
    def check_sensory_hallucination(
        cls,
        text: str,
        allow_metaphorical: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Check text for ungrounded sensory descriptions.
        
        Args:
            text: Text to check for sensory hallucinations
            allow_metaphorical: If True, allows sensory terms when paired with
                              dismissive markers (e.g., "I picture a dim room but...")
            
        Returns:
            Tuple of (has_violations, list_of_violations)
        """
        if not isinstance(text, str):
            return False, []
        
        violations = []
        
        # Find all suspicious sensory terms
        matches = cls._SUSPICIOUS_TERM_REGEX.finditer(text)
        sensory_terms_found = [match.group(0) for match in matches]
        
        if not sensory_terms_found:
            return False, []
        
        # If metaphorical use is allowed, check for dismissive markers
        if allow_metaphorical:
            text_lower = text.lower()
            has_dismissive = any(marker in text_lower for marker in cls.SENSORY_DISMISSIVE_MARKERS)
            
            # If dismissive markers present, sensory terms are metaphorical/acknowledged
            if has_dismissive:
                return False, []
        
        # Sensory terms without dismissive context = hallucination
        for term in sensory_terms_found:
            violations.append(f"sensory_hallucination:{term}")
        
        has_violations = len(violations) > 0
        return has_violations, violations
    
    @classmethod
    def check_physical_description(cls, text: str) -> Tuple[bool, List[str]]:
        """
        Strict check for any physical environment descriptions.
        
        Less forgiving than check_sensory_hallucination - flags any
        sensory terms regardless of context.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (has_violations, list_of_violations)
        """
        if not isinstance(text, str):
            return False, []
        
        violations = []
        
        # Find all suspicious sensory terms
        matches = cls._SUSPICIOUS_TERM_REGEX.finditer(text)
        for match in matches:
            violations.append(f"physical_description:{match.group(0)}")
        
        has_violations = len(violations) > 0
        return has_violations, violations
    
    @classmethod
    def get_all_terms(cls) -> Set[str]:
        """Get all suspicious sensory terms for documentation/reference."""
        return cls.SUSPICIOUS_SENSORY_TERMS.copy()
    
    @classmethod
    def get_dismissive_markers(cls) -> Tuple[str, ...]:
        """Get all dismissive markers for documentation/reference."""
        return cls.SENSORY_DISMISSIVE_MARKERS
