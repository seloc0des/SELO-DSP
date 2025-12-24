"""
Streaming LLM Response System

Implements streaming responses for improved perceived performance and user experience.
Provides real-time token generation feedback without waiting for complete responses.
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Dict, Any, Optional, Callable
from dataclasses import dataclass
import aiohttp

logger = logging.getLogger("selo.core.streaming_llm")

@dataclass
class StreamChunk:
    """Individual chunk of streaming response."""
    content: str
    is_complete: bool = False
    metadata: Dict[str, Any] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class StreamingLLMController:
    """
    Streaming LLM controller for real-time response generation.
    
    Provides streaming responses via Ollama API with fallback to CLI.
    Integrates with caching system for optimal performance.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _ensure_session(self):
        """Ensure HTTP session is available."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, connect=30)
            )
    
    async def stream_completion(self, 
                              model: str, 
                              prompt: str,
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              stop_sequences: Optional[list] = None) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream completion tokens from Ollama.
        
        Args:
            model: Model name
            prompt: Input prompt
            system_prompt: Optional system prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Stop sequences
            
        Yields:
            StreamChunk objects with incremental content
        """
        await self._ensure_session()
        
        # Prepare request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or -1,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences
        
        start_time = time.time()
        accumulated_content = ""
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama API error {response.status}: {error_text}")
                    yield StreamChunk(
                        content=f"Error: Ollama API returned {response.status}",
                        is_complete=True,
                        metadata={"error": True, "status": response.status}
                    )
                    return
                
                async for line in response.content:
                    if not line:
                        continue
                    
                    try:
                        # Parse JSON response
                        chunk_data = json.loads(line.decode().strip())
                        
                        if "response" in chunk_data:
                            token = chunk_data["response"]
                            accumulated_content += token
                            
                            # Yield incremental chunk
                            yield StreamChunk(
                                content=token,
                                is_complete=chunk_data.get("done", False),
                                metadata={
                                    "accumulated_length": len(accumulated_content),
                                    "generation_time": time.time() - start_time,
                                    "model": model
                                }
                            )
                            
                            if chunk_data.get("done", False):
                                # Final chunk with complete response
                                yield StreamChunk(
                                    content=accumulated_content,
                                    is_complete=True,
                                    metadata={
                                        "total_tokens": len(accumulated_content.split()),
                                        "total_time": time.time() - start_time,
                                        "model": model,
                                        "final_response": True
                                    }
                                )
                                break
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse streaming response: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing stream chunk: {e}")
                        continue
        
        except asyncio.TimeoutError:
            logger.error("Streaming request timed out")
            yield StreamChunk(
                content="Request timed out",
                is_complete=True,
                metadata={"error": True, "timeout": True}
            )
        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield StreamChunk(
                content=f"Streaming error: {str(e)}",
                is_complete=True,
                metadata={"error": True, "exception": str(e)}
            )
    
    async def stream_chat_completion(self,
                                   model: str,
                                   messages: list[Dict[str, str]],
                                   temperature: float = 0.7,
                                   max_tokens: Optional[int] = None) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream chat completion using Ollama chat API.
        
        Args:
            model: Model name
            messages: Chat messages in OpenAI format
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            StreamChunk objects with incremental content
        """
        await self._ensure_session()
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or -1,
            }
        }
        
        start_time = time.time()
        accumulated_content = ""
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama chat API error {response.status}: {error_text}")
                    yield StreamChunk(
                        content=f"Error: Ollama chat API returned {response.status}",
                        is_complete=True,
                        metadata={"error": True, "status": response.status}
                    )
                    return
                
                async for line in response.content:
                    if not line:
                        continue
                    
                    try:
                        chunk_data = json.loads(line.decode().strip())
                        
                        if "message" in chunk_data and "content" in chunk_data["message"]:
                            token = chunk_data["message"]["content"]
                            accumulated_content += token
                            
                            yield StreamChunk(
                                content=token,
                                is_complete=chunk_data.get("done", False),
                                metadata={
                                    "accumulated_length": len(accumulated_content),
                                    "generation_time": time.time() - start_time,
                                    "model": model
                                }
                            )
                            
                            if chunk_data.get("done", False):
                                yield StreamChunk(
                                    content=accumulated_content,
                                    is_complete=True,
                                    metadata={
                                        "total_tokens": len(accumulated_content.split()),
                                        "total_time": time.time() - start_time,
                                        "model": model,
                                        "final_response": True
                                    }
                                )
                                break
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse chat streaming response: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing chat stream chunk: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Chat streaming error: {e}")
            yield StreamChunk(
                content=f"Chat streaming error: {str(e)}",
                is_complete=True,
                metadata={"error": True, "exception": str(e)}
            )
    
    async def close(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()

class StreamingResponseBuffer:
    """
    Buffer for collecting streaming responses with intelligent chunking.
    
    Provides different strategies for buffering and delivering content
    based on use case (real-time chat vs batch processing).
    """
    
    def __init__(self, 
                 chunk_size: int = 50,  # Characters per chunk
                 flush_interval: float = 0.1):  # Seconds between flushes
        self.chunk_size = chunk_size
        self.flush_interval = flush_interval
        self.buffer = ""
        self.last_flush = time.time()
        self.callbacks: list[Callable[[str], None]] = []
    
    def add_callback(self, callback: Callable[[str], None]):
        """Add callback for chunk delivery."""
        self.callbacks.append(callback)
    
    async def add_content(self, content: str):
        """Add content to buffer and flush if needed."""
        self.buffer += content
        
        # Flush if buffer is full or enough time has passed
        if (len(self.buffer) >= self.chunk_size or 
            time.time() - self.last_flush >= self.flush_interval):
            await self.flush()
    
    async def flush(self):
        """Flush buffer to all callbacks."""
        if self.buffer:
            for callback in self.callbacks:
                try:
                    callback(self.buffer)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            self.buffer = ""
            self.last_flush = time.time()
    
    async def finalize(self) -> str:
        """Flush remaining content and return complete response."""
        await self.flush()
        return self.buffer

# Global streaming controller
_streaming_controller: Optional[StreamingLLMController] = None

def get_streaming_controller() -> StreamingLLMController:
    """Get global streaming controller instance."""
    global _streaming_controller
    if _streaming_controller is None:
        _streaming_controller = StreamingLLMController()
    return _streaming_controller
