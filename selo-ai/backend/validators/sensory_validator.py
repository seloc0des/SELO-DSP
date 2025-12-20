"""
Sensory Hallucination Detection Validator

Detects when LLM outputs reference physical/sensory details not present
in the provided context (e.g., rooms, lights, sounds, scents).
"""

import re
import logging
from typing import List, Set, Dict, Any

logger = logging.getLogger("selo.validators.sensory")


# Terms that indicate physical/sensory descriptions that could be hallucinated
SUSPICIOUS_SENSORY_TERMS = {
    "room",
    "rooms",
    "lamp",
    "lamps",
    "light",
    "lights",
    "lighting",
    "glow",
    "shadow",
    "shadows",
    "scent",
    "scents",
    "incense",
    "fragrance",
    "aroma",
    "floor",
    "floors",
    "floorboard",
    "floorboards",
    "window",
    "windows",
    "breeze",
    "draft",
    "breathing",
    "heartbeat",
    "candle",
    "candlelight",
    "dim",
    "dimly",
    "wooden",
    "shadowed",
    "ambient",
    "quiet",
    "silence",
    "whisper",
    "murmur",
}

# Compile regex pattern for efficient matching
SUSPICIOUS_TERM_REGEX = re.compile(
    r"\b(" + "|".join(sorted(re.escape(term) for term in SUSPICIOUS_SENSORY_TERMS)) + r")\b"
)

# Markers that indicate the output is explicitly uncertain about sensory details
SENSORY_DISMISSIVE_MARKERS = (
    "don't know",
    "do not know",
    "not sure",
    "uncertain",
    "unknown",
    "cannot tell",
    "can't tell",
    "no idea",
)


class SensoryValidator:
    """
    Validates that outputs don't hallucinate sensory/physical details
    not present in the provided context.
    """
    
    def __init__(self, allow_metaphor: bool = False):
        """
        Initialize the sensory validator.
        
        Args:
            allow_metaphor: If True, be more lenient with sensory terms
                           used metaphorically (e.g., "light at the end of the tunnel")
        """
        self.allow_metaphor = allow_metaphor
    
    def extract_context_tokens(self, context: Dict[str, Any]) -> Set[str]:
        """
        Extract tokens from context that could legitimately contain sensory terms.
        
        Args:
            context: Dictionary containing conversation history, memories, etc.
            
        Returns:
            Set of lowercase tokens found in context
        """
        tokens: Set[str] = set()
        
        def _add_text(value: Any) -> None:
            """Helper to extract text from various data structures."""
            if isinstance(value, str):
                # Extract words (5+ chars for meaningful matching)
                tokens.update(re.findall(r"\b\w{5,}\b", value.lower()))
            elif isinstance(value, list):
                for item in value:
                    _add_text(item)
            elif isinstance(value, dict):
                for v in value.values():
                    _add_text(v)
        
        # Extract from conversation history
        conversation = context.get("conversation", []) or context.get("conversation_history", [])
        if conversation:
            for msg in conversation:
                if isinstance(msg, dict):
                    _add_text(msg.get("content", ""))
                else:
                    _add_text(msg)
        
        # Extract from memories
        memories = context.get("memories", []) or []
        for memory in memories:
            if isinstance(memory, dict):
                _add_text(memory.get("content", ""))
                _add_text(memory.get("text", ""))
            else:
                _add_text(memory)
        
        # Extract from web context
        web_context = context.get("web_context", "") or context.get("web_data", "")
        if web_context:
            _add_text(web_context)
        
        # Extract from persona snapshot
        persona = context.get("persona", {}) or {}
        if isinstance(persona, dict):
            for value in persona.values():
                _add_text(value)
        
        return tokens
    
    def detect_uncontextualized_sensory_terms(
        self, 
        text: str, 
        context_tokens: Set[str]
    ) -> List[str]:
        """
        Detect sensory terms in text that are not supported by context.
        
        Args:
            text: The text to analyze (reflection, response, etc.)
            context_tokens: Set of tokens extracted from context
            
        Returns:
            List of suspicious sensory terms found
        """
        if not text:
            return []
        
        suspicious: Set[str] = set()
        lowered = text.lower()
        
        for match in SUSPICIOUS_TERM_REGEX.finditer(lowered):
            token = match.group(1)
            
            # If the term exists in context, it's legitimate
            if token in context_tokens:
                continue
            
            # Check window around the term for dismissive markers
            # (e.g., "I don't know what the room looks like")
            window_start = max(0, match.start() - 25)
            window_end = min(len(lowered), match.end() + 25)
            window = lowered[window_start:window_end]
            
            # If dismissive marker present, the output is acknowledging uncertainty
            if any(marker in window for marker in SENSORY_DISMISSIVE_MARKERS):
                continue
            
            suspicious.add(token)
        
        return sorted(suspicious)
    
    def validate_output(
        self, 
        text: str, 
        context: Dict[str, Any],
        strict: bool = True
    ) -> tuple[bool, List[str]]:
        """
        Validate that output doesn't hallucinate sensory details.
        
        Args:
            text: The output text to validate
            context: Context dictionary (conversation, memories, etc.)
            strict: If True, any uncontextualized term fails validation
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        context_tokens = self.extract_context_tokens(context)
        violations = self.detect_uncontextualized_sensory_terms(text, context_tokens)
        
        if not violations:
            return True, []
        
        # In non-strict mode, allow a few sensory terms (could be metaphorical)
        if not strict and len(violations) <= 2:
            logger.debug(
                f"Sensory validator: {len(violations)} minor violations allowed in non-strict mode"
            )
            return True, []
        
        logger.warning(
            f"Sensory hallucination detected: {len(violations)} uncontextualized terms: {violations}"
        )
        
        return False, violations
    
    def get_violation_report(
        self, 
        text: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate detailed report on sensory validation.
        
        Args:
            text: The text to analyze
            context: Context dictionary
            
        Returns:
            Dictionary with validation details
        """
        context_tokens = self.extract_context_tokens(context)
        violations = self.detect_uncontextualized_sensory_terms(text, context_tokens)
        is_valid = len(violations) == 0
        
        return {
            "is_valid": is_valid,
            "violations": violations,
            "violation_count": len(violations),
            "context_token_count": len(context_tokens),
            "text_length": len(text),
            "severity": "high" if len(violations) > 5 else "medium" if len(violations) > 2 else "low"
        }


# Singleton instance for shared use
_sensory_validator_instance = None


def get_sensory_validator(allow_metaphor: bool = False) -> SensoryValidator:
    """Get or create the shared sensory validator instance."""
    global _sensory_validator_instance
    if _sensory_validator_instance is None or _sensory_validator_instance.allow_metaphor != allow_metaphor:
        _sensory_validator_instance = SensoryValidator(allow_metaphor=allow_metaphor)
    return _sensory_validator_instance
