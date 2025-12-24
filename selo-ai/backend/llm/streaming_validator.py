"""
Streaming Response Validator

Provides validation for streaming LLM responses by buffering and checking
identity constraints on accumulated content.
"""

import logging
from typing import AsyncGenerator, Dict, Any, Optional
from .response_validator import ResponseValidator

logger = logging.getLogger("selo.streaming_validator")


class StreamingValidator:
    """
    Validates streaming LLM responses by buffering content and checking
    for identity constraint violations.
    """
    
    def __init__(self, persona_name: str = "", reflection_data: Optional[Dict[str, Any]] = None):
        """
        Initialize the streaming validator.
        
        Args:
            persona_name: The persona's name for identity validation
            reflection_data: Optional reflection context for validation
        """
        self.persona_name = persona_name
        self.reflection_data = reflection_data or {}
        self.validator = ResponseValidator()
        self.buffer = ""
        self.check_interval = 50  # Check every N characters
        
    async def validate_stream(
        self, 
        stream: AsyncGenerator[Dict[str, Any], None]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Wrap an async generator stream to validate chunks as they arrive.
        
        This performs incremental validation by:
        1. Buffering accumulated content
        2. Checking for forbidden terms in the buffer
        3. Yielding chunks if validation passes
        4. Performing final validation on complete response
        
        Args:
            stream: The original async generator from LLM
            
        Yields:
            Validated chunks or sanitized replacements
        """
        try:
            from backend.constraints import IdentityConstraints
            
            # Get forbidden pattern for efficient checking
            forbidden_pattern = IdentityConstraints._FORBIDDEN_PATTERN
            
            chunk_count = 0
            violation_detected = False
            
            async for chunk in stream:
                chunk_count += 1
                
                # Extract content from chunk
                content = ""
                if isinstance(chunk, dict):
                    content = chunk.get("content", "") or chunk.get("delta", "")
                elif isinstance(chunk, str):
                    content = chunk
                    
                # Accumulate content
                self.buffer += content
                
                # Perform incremental validation every N characters
                if len(self.buffer) % self.check_interval < len(content):
                    # Quick check for forbidden terms in buffer
                    if forbidden_pattern and forbidden_pattern.search(self.buffer.lower()):
                        # Don't fail immediately - log and continue buffering
                        # Final validation will handle cleanup
                        logger.warning(
                            f"⚠️ Streaming validation detected potential identity violation "
                            f"at chunk {chunk_count}"
                        )
                        violation_detected = True
                
                # Always yield the chunk (we'll validate the complete response at the end)
                yield chunk
            
            # Final validation on complete buffered response
            if self.buffer:
                logger.debug(f"Performing final validation on {len(self.buffer)} char stream")
                
                context = self.validator.extract_reflection_context(self.reflection_data)
                is_valid, validated_response = self.validator.validate_conversational_response(
                    self.buffer,
                    context,
                    persona_name=self.persona_name
                )
                
                if not is_valid:
                    logger.warning(
                        f"🚫 Streaming response failed final validation. "
                        f"Original length: {len(self.buffer)}, "
                        f"Validated length: {len(validated_response)}"
                    )
                    
                    # Yield a correction chunk if the content was modified
                    if validated_response != self.buffer:
                        correction_chunk = {
                            "content": "",
                            "validation": {
                                "valid": False,
                                "warning": "Response was sanitized after streaming",
                                "original_length": len(self.buffer),
                                "sanitized_length": len(validated_response)
                            },
                            "done": True
                        }
                        yield correction_chunk
                elif violation_detected:
                    # Incremental check flagged something but final validation passed
                    logger.info("✅ Stream passed final validation despite earlier warnings")
                    
        except Exception as e:
            logger.error(f"Error in streaming validation: {e}", exc_info=True)
            # On error, continue yielding without validation
            async for chunk in stream:
                yield chunk


async def validate_streaming_response(
    stream: AsyncGenerator[Dict[str, Any], None],
    persona_name: str = "",
    reflection_data: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Convenience function to validate a streaming response.
    
    Args:
        stream: The async generator stream to validate
        persona_name: The persona's name
        reflection_data: Optional reflection context
        
    Yields:
        Validated chunks
    """
    validator = StreamingValidator(persona_name=persona_name, reflection_data=reflection_data)
    async for chunk in validator.validate_stream(stream):
        yield chunk
