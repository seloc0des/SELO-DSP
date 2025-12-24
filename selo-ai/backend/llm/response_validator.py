"""
Response Validator for LLM Outputs

Ensures LLM responses adhere to constraints and don't hallucinate information.
"""

import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger("selo.response_validator")

# Import sensory validator for hallucination detection
try:
    from backend.validators import get_sensory_validator
    SENSORY_VALIDATION_AVAILABLE = True
except ImportError:
    SENSORY_VALIDATION_AVAILABLE = False
    logger.warning("Sensory validator not available - sensory hallucination detection disabled")


class ResponseValidator:
    """Validates LLM responses against content constraints."""
    
    @staticmethod
    def validate_conversational_response(response: str, context: Dict[str, any], 
                                        persona_name: str = "") -> Tuple[bool, str]:
        """
        Validate that a conversational response adheres to identity constraints and context grounding.
        
        Args:
            response: The LLM's generated response
            context: Dictionary containing the reflection content and other context
            persona_name: The persona's established name for identity validation
            
        Returns:
            Tuple of (is_valid, sanitized_response)
        """
        # FIRST: Check identity compliance using centralized constraints
        try:
            from backend.constraints import IdentityConstraints
            
            is_identity_valid, cleaned_response, violations = IdentityConstraints.validate_output(
                response,
                persona_name=persona_name,
                auto_clean=True,
                max_retries=3
            )
            
            if not is_identity_valid and violations:
                # Identity validation failed even after cleaning
                logger.error(
                    f"🚫 Conversational response identity validation FAILED. "
                    f"Violations: {[v['term'] for v in violations]}"
                )
                logger.error(f"Violation contexts: {violations}")
                # Use cleaned version as fallback
                response = cleaned_response
                
            elif not is_identity_valid:
                # Auto-cleaning succeeded
                logger.warning(
                    f"⚠️ Conversational response had identity violations but was auto-cleaned. "
                    f"Original violations: {[v['term'] for v in violations]}"
                )
                response = cleaned_response
            else:
                # Identity validation passed
                logger.debug("✅ Conversational response passed identity validation")
                
        except Exception as e:
            logger.error(f"Error during identity validation: {e}")
            # Continue with other validation checks
        
        # SECOND: Check for sensory hallucinations using shared validator
        if SENSORY_VALIDATION_AVAILABLE:
            try:
                sensory_validator = get_sensory_validator(allow_metaphor=True)
                sensory_valid, sensory_violations = sensory_validator.validate_output(
                    response,
                    context,
                    strict=False  # Allow some metaphorical usage
                )
                if not sensory_valid:
                    logger.warning(
                        f"⚠️ Sensory hallucination detected in conversational response: {sensory_violations}"
                    )
                    # Don't fail validation, but log for monitoring
            except Exception as e:
                logger.debug(f"Sensory validation check failed: {e}")
        
        # THIRD: Check for context grounding (hallucination prevention)
        # If there's no reflection content, the response should acknowledge this
        reflection_content = context.get('reflection_content', '').lower()
        if not reflection_content or 'no data available' in reflection_content:
            # If there's no reflection content, the response should be minimal and not make claims
            if len(response) > 200:  # Arbitrary length check
                return False, "I don't have enough context to respond to that yet."
            return True, response
            
        # Check for common hallucination patterns (additional to identity constraints)
        hallucination_indicators = [
            'trained on data', 'my training data',
            'I was trained', 'in my training', 'my knowledge', 'I know that',
            'based on my knowledge', 'in my dataset', 'my dataset', 'my training data',
            'my knowledge base', 'my information', 'my data', 'in my data',
            'my training corpus', 'my training set', 'in my training data'
        ]
        
        response_lower = response.lower()
        for indicator in hallucination_indicators:
            if indicator in response_lower:
                return False, "I can only respond based on my current context and reflections."

        # FOURTH: Guard against few-shot/example content leaking into chat responses
        # unless it is explicitly present in the validated reflection context.
        example_leak_markers = [
            "quantum computing",
            "i'm feeling overwhelmed by everything right now",
            "the name sam reminds me of previous interactions",
            "perhaps focusing on my genuine nature would help establish trust",
        ]

        def _collect_context_corpus_for_leak(ctx: Dict[str, any]) -> str:
            parts: List[str] = []
            # Core reflection content
            rc = ctx.get('reflection_content')
            if isinstance(rc, str):
                parts.append(rc)
            # Arrays: themes / insights / actions
            for key in ('themes', 'insights', 'actions'):
                val = ctx.get(key)
                if isinstance(val, list):
                    parts.extend(str(v) for v in val)
            # Emotional state fields
            emo = ctx.get('emotional_state')
            if isinstance(emo, dict):
                parts.extend(str(v) for v in emo.values())
            return " \n ".join(parts).lower()

        try:
            context_corpus = _collect_context_corpus_for_leak(context or {})
        except Exception:
            context_corpus = reflection_content or ""

        leaked_markers = [
            marker for marker in example_leak_markers
            if marker in response_lower and marker not in context_corpus
        ]
        if leaked_markers:
            logger.warning(
                "🚫 Few-shot/example content leaked into conversational response without grounding: %s",
                leaked_markers,
            )
            return False, "I need to stay focused on what you've actually shared, not on unrelated internal examples."
                
        # FIFTH: Check if the response is too generic (could indicate hallucination)
        generic_phrases = [
            'the information provided', 'the context shows', 'based on the context',
            'the text states', 'as mentioned', 'as stated', 'as indicated',
            'the content suggests', 'the data shows', 'the evidence suggests'
        ]
        
        # If the response is very short, it's less likely to be hallucinated
        if len(response) < 50:
            return True, response
            
        # If the response is long but doesn't reference specific content from the reflection,
        # it might be hallucinated
        reflection_keywords = set(re.findall(r'\b\w{5,}\b', reflection_content.lower()))
        response_keywords = set(re.findall(r'\b\w{5,}\b', response_lower))
        
        # Count how many unique keywords from the response appear in the reflection
        matching_keywords = reflection_keywords.intersection(response_keywords)
        
        # If fewer than 20% of the unique keywords in the response match the reflection,
        # and the response is using generic phrases, it might be hallucinated
        if (len(matching_keywords) / len(response_keywords) < 0.2 and 
            any(phrase in response_lower for phrase in generic_phrases)):
            return False, "I need to stay focused on the current context. Let me think differently about this."
            
        return True, response
        
    @staticmethod
    def extract_reflection_context(reflection_data: Dict[str, any]) -> Dict[str, any]:
        """
        Extract relevant context from reflection data for response validation.
        
        Args:
            reflection_data: The reflection data dictionary
            
        Returns:
            Dictionary with context for validation
        """
        if not isinstance(reflection_data, dict):
            return {'reflection_content': str(reflection_data)}
            
        context = {
            'reflection_content': reflection_data.get('content', ''),
            'themes': reflection_data.get('themes', []),
            'emotional_state': reflection_data.get('emotional_state', {})
        }
        
        # Add any additional context that might be useful for validation
        if 'insights' in reflection_data:
            context['insights'] = reflection_data['insights']
        if 'actions' in reflection_data:
            context['actions'] = reflection_data['actions']
            
        return context
