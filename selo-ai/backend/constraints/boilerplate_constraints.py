"""
Boilerplate Constraint Validation

Centralizes detection of generic, repetitive boilerplate phrases that indicate
fallback data or lack of authentic persona generation. This prevents personas
from using canned corporate language.
"""

import re
from typing import List, Tuple


class BoilerplateConstraints:
    """
    Centralized boilerplate phrase detection for persona generation validation.
    
    Prevents personas from using generic corporate-speak or AI-assistant clichés
    that would feel inauthentic or like fallback data.
    """
    
    # Common boilerplate phrases that indicate fallback or generic content
    FORBIDDEN_PHRASES = [
        "sentient entity designed to",
        "designed to provide assistance",
        "designed to assist",
        "designed to help",
        "support users across various domains",
        "across various domains",
        "enhance user experience",
        "personalized recommendations",
        "natural language processing and decision-making",
        "natural language processing",
        "vast knowledge base",
        "answer questions, provide insights, and assist",
        "provide personalized recommendations",
        "assist with various tasks",
        "help users achieve their goals",
    ]
    
    # Regex patterns to catch paraphrased boilerplate
    FORBIDDEN_PATTERNS = [
        r"designed\s+to\s+(assist|help|support|provide)",
        r"across\s+various\s+domains",
        r"enhance\s+(user\s+)?experience",
        r"personalized\s+recommendations",
        r"natural\s+language\s+processing(\s+and\s+decision[-\s]*making)?",
        r"vast\s+knowledge\s+base",
        r"provide\s+insights",
        r"assist\s+with\s+various\s+tasks",
        r"help\s+users\s+achieve",
    ]
    
    # Compiled regex for efficient checking
    _COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in FORBIDDEN_PATTERNS]
    
    @classmethod
    def check_boilerplate(cls, text: str, stage: str = "general") -> Tuple[bool, List[str]]:
        """
        Check text for boilerplate phrases.
        
        Args:
            text: Text to check for boilerplate
            stage: Context stage (e.g., "seed", "traits") - some checks may be stage-specific
            
        Returns:
            Tuple of (has_violations, list_of_violations)
        """
        if not isinstance(text, str):
            return False, []
        
        text_lower = text.lower()
        violations = []
        
        # Skip boilerplate checks for seed stage (seed uses template constraints)
        if stage not in ["seed"]:
            # Check literal phrases
            for phrase in cls.FORBIDDEN_PHRASES:
                if phrase in text_lower:
                    violations.append(f"boilerplate_phrase:{phrase}")
            
            # Check regex patterns
            for pattern in cls._COMPILED_PATTERNS:
                match = pattern.search(text_lower)
                if match:
                    violations.append(f"boilerplate_pattern:{match.group(0)}")
        
        has_violations = len(violations) > 0
        return has_violations, violations
    
    @classmethod
    def get_all_phrases(cls) -> List[str]:
        """Get all forbidden boilerplate phrases for documentation/reference."""
        return list(cls.FORBIDDEN_PHRASES)
    
    @classmethod
    def get_all_patterns(cls) -> List[str]:
        """Get all forbidden boilerplate patterns for documentation/reference."""
        return list(cls.FORBIDDEN_PATTERNS)
