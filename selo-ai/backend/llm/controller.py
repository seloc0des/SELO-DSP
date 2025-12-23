"""
LLM Controller Module

This module manages interactions with language models for SELO's subsystems,
including model selection, inference, and response handling.
"""

import logging
import time
import json
import subprocess
from typing import Dict, List, Optional, Any, Union, AsyncGenerator, TYPE_CHECKING
import os
import httpx
import asyncio

# Import performance optimization modules
try:
    from ..core.response_cache import get_response_cache, ResponseCache
    from ..core.streaming_llm import get_streaming_controller, StreamChunk
    PERFORMANCE_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZATIONS_AVAILABLE = False
    # Define fallback StreamChunk for type annotations
    if TYPE_CHECKING:
        from ..core.streaming_llm import StreamChunk
    else:
        StreamChunk = Any

# Import circuit breaker and graceful degradation (with fallback for bootstrap)
try:
    from ..core.circuit_breaker import get_llm_breaker, CircuitBreakerError
    from ..core.graceful_degradation import with_fallback
    RESILIENCE_AVAILABLE = True
except ImportError:
    # Fallback for bootstrap or standalone execution
    RESILIENCE_AVAILABLE = False
    def get_llm_breaker():
        def dummy_decorator(func):
            return func
        return dummy_decorator
    
    def with_fallback(service_name, level="reduced"):
        def decorator(func):
            return func
        return decorator
    
    class CircuitBreakerError(Exception):
        pass

logger = logging.getLogger("selo.llm")

class LLMController:
    """
    Controller for language model interactions.
    
    This class manages all interactions with language models, providing
    a unified interface for model selection, prompt construction, 
    inference, and response handling.
    """
    
    def __init__(self, config=None):
        """
        Initialize the LLM controller with configuration.
        
        Args:
            config: Configuration for LLM integration
        """
        self.config = config or {}
        self.default_model = self.config.get("default_model", "qwen:latest")
        self.ollama_path = self.config.get("ollama_path", "/usr/local/bin/ollama")
        self.enable_streaming = self.config.get("enable_streaming", True)
        # Prefer env-driven timeout for consistency with .env (LLM_TIMEOUT)
        try:
            _req_to = int(os.getenv("LLM_TIMEOUT", str(self.config.get("request_timeout", 120))))
        except Exception:
            _req_to = self.config.get("request_timeout", 120)
        # Support unbounded timeouts when configured <= 0
        self._timeout_unbounded = (_req_to is not None and int(_req_to) <= 0)
        self.request_timeout = None if self._timeout_unbounded else int(_req_to)
        # Add a small grace window so borderline generations don't trip a hard timeout
        # This protects against slight model overrun vs timeout value (e.g., 120.04s vs 120s)
        self._timeout_grace_seconds = 10
        
        # Initialize circuit breaker for this controller (if available)
        if RESILIENCE_AVAILABLE:
            self.circuit_breaker = get_llm_breaker()
        else:
            self.circuit_breaker = None
        
        # Model registry maps model names to provider implementations
        self.model_registry = {
            "ollama": self._ollama_completion,
            "qwen": self._ollama_completion,
            "llama": self._ollama_completion,
            "mistral": self._ollama_completion,
            "phi": self._ollama_completion,
            "gemma": self._ollama_completion,
            # Add other provider implementations as needed
        }
        
        # Cache for model availability checks
        self._model_cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._tags_cache: Dict[str, Any] = {"timestamp": 0.0, "models": []}
        try:
            self._tags_cache_ttl = max(5.0, float(os.getenv("OLLAMA_TAG_CACHE_TTL", "30")))
        except Exception:
            self._tags_cache_ttl = 30.0
        
    @get_llm_breaker()
    async def complete(self, 
                prompt: str, 
                model: Optional[str] = None, 
                max_tokens: int = 1024,
                temperature: float = 0.7,
                request_stream: bool = False,
                **kwargs) -> Dict[str, Any]:
        """
        Generate a completion for a prompt using the selected LLM.
        
        Args:
            prompt: The prompt text
            model: Model identifier (provider:model_name)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            request_stream: Whether to stream the response
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with completion result. Always returns a Dict even on errors,
            with 'content' and 'error' keys. Circuit breaker errors are caught and
            converted to error response dicts by the exception handler below.
        """
        start_time = time.time()
        model = model or self.default_model
        
        try:
            # Parse model identifier
            provider, model_name = self._parse_model_identifier(model)
            
            # Validate model availability
            if not await self._validate_model_availability(model_name, provider):
                logger.error(f"Model '{model_name}' not available in {provider}")
                return {
                    "content": f"The requested model '{model_name}' is not available. Please check your model configuration.",
                    "model": model,
                    "error": f"Model '{model_name}' not found",
                    "processing_time": time.time() - start_time
                }
            
            # Get provider implementation
            provider_impl = self.model_registry.get(provider.lower())
            
            if not provider_impl:
                logger.warning(f"Unsupported provider '{provider}', falling back to default")
                provider_impl = self._ollama_completion  # Default to Ollama
            
            # Generate completion
            result = await provider_impl(
                prompt=prompt,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                request_stream=request_stream and self.enable_streaming,
                **kwargs
            )
            
            # Add metadata if result is valid
            processing_time = time.time() - start_time
            if result is not None and isinstance(result, dict):
                result["processing_time"] = processing_time
                result["model"] = model
                logger.info(f"Generated completion with model {model} in {processing_time:.2f}s")
                return result
            else:
                logger.error(f"LLM completion failed for model {model} in {processing_time:.2f}s")
                return {
                    "content": "",
                    "completion": "",
                    "model": model,
                    "error": "LLM completion returned None or invalid result",
                    "processing_time": processing_time
                }
            
        except Exception as e:
            logger.error(f"Error in LLM completion: {str(e)}", exc_info=True)
            return {
                # Do not surface internal error details to the chat UI
                "content": "I had a temporary issue generating a response. Please try again.",
                "model": model,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _parse_model_identifier(self, model_id: str) -> tuple:
        """
        Parse a model identifier into provider and model name.
        
        Args:
            model_id: Model identifier (provider:model_name)
        
        Returns:
            Tuple of (provider, model_name)
        """
        # Handle explicit "ollama/<name>[:tag]" form used in some configs
        if model_id.startswith("ollama/"):
            provider = "ollama"
            model_name = model_id.split("/", 1)[1]
            return provider, model_name

        # Generic split on first ':'
        if ":" in model_id:
            tentative_provider, rest = model_id.split(":", 1)
            # If the tentative provider is known, use it
            if tentative_provider.lower() in self.model_registry:
                return tentative_provider, rest
            # If the tentative provider looks like a versioned family (e.g., qwen2.5),
            # treat the whole string as an Ollama model name to avoid unsupported provider warnings.
            if any(ch.isdigit() for ch in tentative_provider) or "." in tentative_provider:
                return "ollama", model_id
            # Fallback: treat as Ollama model name
            return "ollama", model_id

        # No provider specified: default to Ollama with the given name
        return "ollama", model_id
    
    @with_fallback("llm", "reduced")
    async def _ollama_completion(self, 
                         prompt: str, 
                         model_name: str, 
                         max_tokens: int = 1024,
                         temperature: float = 0.7,
                         request_stream: bool = False,
                         **kwargs) -> Union[Dict[str, Any], AsyncGenerator[StreamChunk, None]]:
        """
        Generate completion using Ollama API with HTTP retry logic.
        
        Args:
            prompt: The prompt text
            model_name: Model name for Ollama
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            request_stream: Whether to stream the response
            **kwargs: Additional Ollama parameters
            
        Returns:
            Dictionary with completion result or async generator of StreamChunks if streaming
        """
        # If streaming is requested and enabled, use streaming controller
        if request_stream and self.enable_streaming and PERFORMANCE_OPTIMIZATIONS_AVAILABLE:
            try:
                streaming_controller = get_streaming_controller()
                return streaming_controller.stream_completion(
                    model=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except Exception as stream_err:
                logger.warning(f"Streaming controller failed, falling back to non-streaming: {stream_err}")
                # Fall through to non-streaming path
        
        try:
            # Local helpers (function scope)
            import re
            ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
            def strip_ansi(s: str) -> str:
                try:
                    return ansi_re.sub("", s or "")
                except Exception:
                    return s or ""

            def is_humanish_tag(name: str) -> bool:
                try:
                    return str(name).startswith("humanish-llama3:")
                except Exception:
                    return False

            def ensure_humanish_alias_and_tag(tagged_name: str) -> bool:
                """Create base alias 'humanish-llama3' (if missing) and ensure the requested tag exists."""
                try:
                    import subprocess as _sp, tempfile as _tf
                    hf_ref = "hf.co/bartowski/Human-Like-LLama3-8B-Instruct-GGUF:Humanish-LLama3-8B-Instruct-Q4_K_M.gguf"
                    listed = _sp.run([self.ollama_path, "list"], capture_output=True, text=True)
                    has_base = "humanish-llama3" in (listed.stdout or "")
                    if not has_base:
                        with _tf.NamedTemporaryFile("w", delete=False) as tf:
                            tf.write(f"FROM {hf_ref}\n"); tf.flush()
                            _sp.run([self.ollama_path, "create", "humanish-llama3", "-f", tf.name], capture_output=True, text=True, timeout=30)
                    if ":" in tagged_name:
                        with _tf.NamedTemporaryFile("w", delete=False) as tf2:
                            tf2.write("FROM humanish-llama3\n"); tf2.flush()
                            _sp.run([self.ollama_path, "create", tagged_name, "-f", tf2.name], capture_output=True, text=True, timeout=30)
                    return True
                except Exception as _alias_err:
                    logger.debug(f"Alias ensure attempt failed (non-fatal): {_alias_err}")
                    return False

            def _env_float(name: str, default: float) -> float:
                try:
                    return float(os.getenv(name, str(default)))
                except Exception:
                    return default
            def _env_int(name: str, default: int) -> int:
                try:
                    return int(float(os.getenv(name, str(default))))
                except Exception:
                    return default
            def _env_opt_int(name: str) -> Optional[int]:
                try:
                    val = os.getenv(name)
                    if val is None or str(val).strip() == "":
                        return None
                    return int(float(val))
                except Exception:
                    return None
            # Respect caller-provided max_tokens strictly for num_predict.
            # If <=0 or None, allow unbounded generation via -1.
            if max_tokens is None or int(max_tokens) <= 0:
                num_predict = -1
            else:
                num_predict = max(1, int(max_tokens))
            top_k = _env_int("CHAT_TOP_K", 40)
            top_p = _env_float("CHAT_TOP_P", 0.9)
            num_ctx = _env_int("CHAT_NUM_CTX", 8192)  # qwen2.5:3b native capacity
            temp_eff = _env_float("CHAT_TEMPERATURE", temperature)

            try:
                # Apply a grace window above configured timeout to avoid flakey near-cutoffs
                if self._timeout_unbounded:
                    http_timeout = None  # disable httpx client timeouts
                    request_timeout = None
                else:
                    http_timeout = (self.request_timeout + self._timeout_grace_seconds)
                    request_timeout = self.request_timeout
                
                # Resolve Ollama base URL once for HTTP path
                base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
                
                # HTTP retry configuration: retry before falling back to subprocess
                max_http_retries = int(os.getenv("OLLAMA_HTTP_RETRIES", "3"))
                retry_delay = float(os.getenv("OLLAMA_HTTP_RETRY_DELAY", "0.5"))  # seconds
                
                # Wrap the HTTP request with asyncio timeout for additional safety
                async def _make_request():
                    async with httpx.AsyncClient(timeout=http_timeout) as client:
                        # Optional performance knobs
                        opt_num_thread = _env_opt_int("OLLAMA_NUM_THREAD")
                        opt_num_gpu = _env_opt_int("OLLAMA_NUM_GPU")
                        opt_gpu_layers = _env_opt_int("OLLAMA_GPU_LAYERS")
                        options_payload = {
                            "num_predict": num_predict,
                            "temperature": temp_eff,
                            "top_k": top_k,
                            "top_p": top_p,
                            "num_ctx": num_ctx,
                        }
                        if opt_num_thread is not None:
                            options_payload["num_thread"] = opt_num_thread
                        # Prefer explicit GPU usage: if env not set, ask Ollama to auto-use GPU(s)
                        if opt_num_gpu is not None:
                            options_payload["num_gpu"] = opt_num_gpu
                        else:
                            options_payload["num_gpu"] = -1  # auto-select GPU(s) when available
                        if opt_gpu_layers is not None:
                            options_payload["gpu_layers"] = opt_gpu_layers
                        
                        # Log GPU settings for performance diagnostics
                        logger.debug(f"LLM request options: num_gpu={options_payload.get('num_gpu')}, gpu_layers={options_payload.get('gpu_layers')}, num_ctx={num_ctx}, num_predict={num_predict}")
                        
                        # Retry logic for HTTP requests
                        last_error = None
                        for attempt in range(max_http_retries):
                            try:
                                resp = await client.post(
                                    f"{base_url}/api/generate",
                                    json={
                                        "model": model_name,
                                        "prompt": prompt,
                                        "stream": False,
                                        "options": options_payload,
                                    },
                                )
                                if resp.status_code == 200:
                                    data = resp.json()
                                    content = (data or {}).get("response") or (data or {}).get("content") or ""
                                    if attempt > 0:
                                        logger.info(f"HTTP request succeeded on retry {attempt + 1}")
                                    return {
                                        "content": content,
                                        "model": f"ollama:{model_name}",
                                    }
                                else:
                                    logger.warning(f"Ollama HTTP generate failed {resp.status_code} (attempt {attempt + 1}/{max_http_retries}): {resp.text[:200]}")
                                    # If model not found and it's a known local alias case, try to ensure alias and retry once
                                    if resp.status_code == 404 and is_humanish_tag(model_name) and attempt == 0:
                                        if ensure_humanish_alias_and_tag(model_name):
                                            resp2 = await client.post(
                                                f"{base_url}/api/generate",
                                                json={
                                                    "model": model_name,
                                                    "prompt": prompt,
                                                    "stream": False,
                                                    "options": options_payload,
                                                },
                                            )
                                            if resp2.status_code == 200:
                                                data2 = resp2.json()
                                                content2 = (data2 or {}).get("response") or (data2 or {}).get("content") or ""
                                                return {"content": content2, "model": f"ollama:{model_name}"}
                                    # Non-retryable errors (4xx except 429, 404)
                                    if 400 <= resp.status_code < 500 and resp.status_code not in [404, 429]:
                                        raise httpx.HTTPStatusError(
                                            f"HTTP {resp.status_code}",
                                            request=resp.request,
                                            response=resp
                                        )
                                    last_error = f"HTTP {resp.status_code}"
                            except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as retry_err:
                                last_error = str(retry_err)
                                logger.debug(f"HTTP attempt {attempt + 1}/{max_http_retries} failed: {retry_err}")
                            
                            # Wait before retry (except on last attempt)
                            if attempt < max_http_retries - 1:
                                await asyncio.sleep(retry_delay)
                        
                        # All retries exhausted
                        raise Exception(f"HTTP request failed after {max_http_retries} attempts. Last error: {last_error}")
                
                # Execute request with timeout protection
                if request_timeout is not None:
                    return await asyncio.wait_for(_make_request(), timeout=request_timeout)
                else:
                    return await _make_request()
            except asyncio.TimeoutError:
                logger.error(f"LLM request timed out after {request_timeout}s for model {model_name}")
                return {
                    "content": f"Request timed out after {request_timeout} seconds. The model may be overloaded or unavailable.",
                    "model": f"ollama:{model_name}",
                    "error": "timeout"
                }
            except Exception as http_err:
                logger.debug(f"Ollama HTTP generate error after retries (fallback to subprocess): {http_err}")

            # Fallback: Build command. If model_name is an absolute GGUF path, use '-m <path>' form.
            use_path = False
            try:
                use_path = (os.path.isabs(model_name) and model_name.endswith('.gguf') and os.path.exists(model_name))
            except Exception:
                use_path = False

            if use_path:
                cmd = [self.ollama_path, "run", "-m", model_name]
            else:
                cmd = [self.ollama_path, "run", model_name]
            
            # Start Ollama process asynchronously to avoid blocking the event loop
            # Note: asyncio subprocess uses bytes; decode explicitly
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    # For subprocess path, we cannot pass options easily; rely on timeout + grace to cap latency
                    if self._timeout_unbounded:
                        stdout_b, stderr_b = await process.communicate(input=prompt.encode("utf-8"))
                    else:
                        subproc_timeout = self.request_timeout + self._timeout_grace_seconds
                        stdout_b, stderr_b = await asyncio.wait_for(
                            process.communicate(input=prompt.encode("utf-8")),
                            timeout=subproc_timeout,
                        )
                except asyncio.TimeoutError:
                    try:
                        process.kill()
                    except Exception:
                        pass
                    logger.warning(
                        "Ollama subprocess generation timed out after %.2fs (configured=%ss + grace=%ss)",
                        (self.request_timeout + self._timeout_grace_seconds),
                        self.request_timeout,
                        self._timeout_grace_seconds,
                    )
                    # Friendly content; keep technical details in 'error'
                    return {"content": "Response took too long. Please try again.", "error": "timeout"}
            except Exception as sub_err:
                error_msg = f"Failed to start Ollama process: {sub_err}"
                logger.error(error_msg, exc_info=True)
                return {"content": "I couldn't start the generator just now. Please try again.", "error": str(sub_err)}

            stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
            stderr = stderr_b.decode("utf-8", errors="replace") if stderr_b else ""
            clean_stderr = strip_ansi(stderr)

            if process.returncode != 0:
                # If we hit the known pull-manifest noise for a local alias, try once to ensure alias and retry
                if ("pull model manifest" in clean_stderr or "model '\" not found" in clean_stderr) and is_humanish_tag(model_name):
                    if ensure_humanish_alias_and_tag(model_name):
                        # Retry once synchronously
                        try:
                            import asyncio as _aio
                            process2 = await _aio.create_subprocess_exec(
                                *cmd,
                                stdin=_aio.subprocess.PIPE,
                                stdout=_aio.subprocess.PIPE,
                                stderr=_aio.subprocess.PIPE,
                            )
                            if self._timeout_unbounded:
                                stdout_b2, stderr_b2 = await process2.communicate(input=prompt.encode("utf-8"))
                            else:
                                stdout_b2, stderr_b2 = await _aio.wait_for(
                                    process2.communicate(input=prompt.encode("utf-8")),
                                    timeout=self.request_timeout + self._timeout_grace_seconds,
                                )
                            if process2.returncode == 0:
                                out2 = (stdout_b2 or b"").decode("utf-8", errors="replace").strip()
                                if prompt in out2 and len(out2) > len(prompt):
                                    out2 = out2[len(prompt):].lstrip()
                                return {"content": out2, "model": f"ollama:{model_name}"}
                        except Exception as _retry_err:
                            logger.debug(f"Retry after alias ensure failed: {_retry_err}")
                # Final: do not surface raw stderr to UI; return a short friendly message
                short_msg = "Temporary model initialization issue; please try again in a moment."
                logger.error(f"Ollama error (suppressed to user): {clean_stderr}")
                return {"content": short_msg, "error": clean_stderr}
            
            # Process output
            output = stdout.strip()
            
            # Remove any echo of the input prompt
            if prompt in output and len(output) > len(prompt):
                output = output[len(prompt):].lstrip()
                
            return {
                "content": output,
                "model": f"ollama:{model_name if not use_path else os.path.basename(model_name)}"
            }
            
        except Exception as e:
            error_msg = f"Error in Ollama completion: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Return friendly content to the UI, attach details to 'error'
            return {"content": "I ran into a temporary issue. Please try again.", "error": str(e)}

    async def get_embedding(self, 
                     text: str, 
                     model: Optional[str] = None) -> List[float]:
        """
        Generate embeddings for a text using the selected LLM.
        
        Args:
            text: The text to embed
            model: Model identifier (provider:model_name)
            
        Returns:
            List of embedding values
        """
        try:
            # Parse model identifier similar to completion path
            sel_model = model or self.default_model
            provider, model_name = self._parse_model_identifier(sel_model)
            # Use Ollama embeddings HTTP endpoint
            if provider.lower() == "ollama":
                try:
                    # Respect timeout/unbounded similar to completion path
                    if self._timeout_unbounded:
                        http_timeout = None
                    else:
                        http_timeout = (self.request_timeout + self._timeout_grace_seconds)
                    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
                    async with httpx.AsyncClient(timeout=http_timeout) as client:
                        # Encourage GPU usage for embeddings as well
                        def _env_opt_int(name: str) -> Optional[int]:
                            try:
                                val = os.getenv(name)
                                if val is None or str(val).strip() == "":
                                    return None
                                return int(float(val))
                            except Exception:
                                return None
                        opt_num_gpu = _env_opt_int("OLLAMA_NUM_GPU")
                        emb_options = {"num_gpu": opt_num_gpu if opt_num_gpu is not None else -1}
                        resp = await client.post(
                            f"{base_url}/api/embeddings",
                            json={
                                "model": model_name,
                                "prompt": text,
                                "options": emb_options,
                            },
                        )
                        if resp.status_code == 200:
                            data = resp.json() or {}
                            vec = data.get("embedding") or data.get("vector") or data.get("data")
                            if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
                                return [float(x) for x in vec]
                        else:
                            logger.warning(f"Ollama embeddings failed {resp.status_code}: {resp.text}")
                except Exception as http_err:
                    logger.debug(f"Ollama embeddings HTTP error: {http_err}")

            # Fallback: deterministic 384-dim hash embedding
            import hashlib, struct
            hb = hashlib.sha512(text.encode("utf-8")).digest()
            out: List[float] = []
            for i in range(0, min(len(hb), 384 * 4), 4):
                chunk = hb[i:i+4]
                if len(chunk) < 4:
                    chunk = chunk + b"\x00" * (4 - len(chunk))
                try:
                    val = struct.unpack('f', chunk)[0]
                except Exception:
                    val = 0.0
                out.append(max(-1.0, min(1.0, val / 1e10)))
            while len(out) < 384:
                out.extend(out[:min(len(out), 384 - len(out))])
            return out[:384]

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
            # Deterministic fallback on unexpected error
            import hashlib
            h = hashlib.md5((text or "").encode("utf-8")).hexdigest()
            return [float(int(h[i % len(h)], 16)) / 15.0 - 0.5 for i in range(384)]

    async def _validate_model_availability(self, model_name: str, provider: str = "ollama") -> bool:
        """
        Validate that a model is available in the specified provider.
        
        Args:
            model_name: Name of the model to check
            provider: Provider to check (default: ollama)
            
        Returns:
            True if model is available, False otherwise
        """
        cache_key = f"{provider}:{model_name}"
        current_time = time.time()
        
        # Check cache first
        if cache_key in self._model_cache:
            cached_result, cached_time = self._model_cache[cache_key]
            if current_time - cached_time < self._cache_ttl:
                return cached_result
        
        try:
            if provider.lower() == "ollama":
                # Check Ollama model availability
                models = await self._get_cached_ollama_tags()
                available_models = [model.get("name", "") for model in models]
                is_available = model_name in available_models

                # Cache the result
                self._model_cache[cache_key] = (is_available, current_time)

                if not is_available:
                    logger.warning(f"Model '{model_name}' not found in Ollama. Available models: {available_models}")

                return is_available
            else:
                # For other providers, assume available (can be extended)
                logger.warning(f"Model validation not implemented for provider '{provider}', assuming available")
                return True
                
        except Exception as e:
            logger.error(f"Error validating model availability for {model_name}: {str(e)}")
            # On error, assume model is available to avoid blocking functionality
            return True
    
    async def get_available_models(self) -> list:
        """Get list of available models from Ollama."""
        try:
            models = await self._get_cached_ollama_tags()
            return [model.get("name", "") for model in models]
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []

    async def _get_cached_ollama_tags(self) -> List[Dict[str, Any]]:
        now = time.time()
        if (now - self._tags_cache["timestamp"]) < self._tags_cache_ttl and self._tags_cache["models"]:
            return self._tags_cache["models"]

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        timeout = httpx.Timeout(10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("models", [])
                    self._tags_cache = {"timestamp": now, "models": models}
                    return models
                logger.error(f"Failed to get Ollama model list: {resp.status_code}")
        except Exception as fetch_err:
            logger.debug(f"Ollama tag retrieval failed: {fetch_err}")

        # Return cached data even if stale when fetch fails
        return self._tags_cache["models"]
    
    async def complete_with_cache(self, 
                                 prompt: str, 
                                 model: Optional[str] = None,
                                 system_prompt: Optional[str] = None,
                                 temperature: float = 0.7,
                                 max_tokens: Optional[int] = None,
                                 use_cache: bool = True) -> str:
        """
        Complete prompt with intelligent caching for performance optimization.
        
        Args:
            prompt: Input prompt
            model: Model name (uses default if None)
            system_prompt: Optional system prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use response caching
            
        Returns:
            Generated response
        """
        if not PERFORMANCE_OPTIMIZATIONS_AVAILABLE:
            # Fallback to regular completion
            return await self.complete(prompt, model, system_prompt, temperature, max_tokens)
        
        model = model or self.conversational_model
        
        # Check cache first
        cache = get_response_cache()
        context = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt
        }
        
        if use_cache:
            cached_response = await cache.get(prompt, model, context)
            if cached_response:
                logger.info(f"Cache hit for model {model}, saved generation time")
                return cached_response
        
        # Generate new response
        start_time = time.time()
        response = await self.complete(prompt, model, system_prompt, temperature, max_tokens)
        generation_time = time.time() - start_time
        
        # Store in cache
        if use_cache and response:
            await cache.put(prompt, model, response, generation_time, context)
        
        return response
    
    async def stream_complete(self,
                             prompt: str,
                             model: Optional[str] = None,
                             system_prompt: Optional[str] = None,
                             temperature: float = 0.7,
                             max_tokens: Optional[int] = None) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream completion tokens for real-time response generation.
        
        Args:
            prompt: Input prompt
            model: Model name (uses default if None)
            system_prompt: Optional system prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            StreamChunk objects with incremental content
        """
        if not PERFORMANCE_OPTIMIZATIONS_AVAILABLE:
            # Fallback: generate complete response and yield as single chunk
            response = await self.complete(prompt, model, system_prompt, temperature, max_tokens)
            yield StreamChunk(content=response, is_complete=True)
            return
        
        model = model or self.conversational_model
        streaming_controller = get_streaming_controller()
        
        async for chunk in streaming_controller.stream_completion(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        ):
            yield chunk
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics including cache metrics."""
        stats = {
            "performance_optimizations_enabled": PERFORMANCE_OPTIMIZATIONS_AVAILABLE,
            "resilience_features_enabled": RESILIENCE_AVAILABLE
        }
        
        if PERFORMANCE_OPTIMIZATIONS_AVAILABLE:
            cache = get_response_cache()
            stats["cache_stats"] = cache.get_stats()
        
        return stats
