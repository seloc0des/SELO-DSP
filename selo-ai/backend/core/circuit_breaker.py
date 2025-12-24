"""
Circuit Breaker Pattern Implementation for SELO AI

Prevents cascade failures by isolating failing components and providing fallback mechanisms.
Automatically recovers when components become healthy again.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict
from enum import Enum
from dataclasses import dataclass
from functools import wraps
import threading

logger = logging.getLogger("selo.circuit_breaker")

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 60.0      # Seconds before trying recovery
    success_threshold: int = 3          # Successes needed to close
    timeout: float = None               # Request timeout (None = unbounded)
    expected_exception: tuple = (Exception,)  # Exceptions to count as failures

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

class CircuitBreaker:
    """
    Circuit breaker implementation with async support.
    
    Tracks failures and automatically opens/closes circuit based on health.
    Provides fallback mechanisms and graceful degradation.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.lock = threading.RLock()
        
        # Metrics tracking for monitoring
        # Use bounded counters to prevent integer overflow in long-running deployments
        self._metrics_max = 10_000_000  # Cap at 10 million to prevent overflow
        self.metrics = {
            "total_failures": 0,
            "total_successes": 0,
            "total_opens": 0,
            "total_closes": 0,
            "total_half_opens": 0,
            "last_state_change": time.time(),
            "created_at": time.time()
        }
        
        logger.info(f"🔧 Circuit breaker '{name}' initialized")
    
    def __call__(self, func: Callable = None, *, fallback: Callable = None):
        """
        Decorator to wrap functions with circuit breaker protection.
        
        Args:
            func: Function to protect
            fallback: Fallback function to call when circuit is open
        """
        if func is None:
            return lambda f: self.__call__(f, fallback=fallback)
            
        if asyncio.iscoroutinefunction(func):
            return self._async_wrapper(func, fallback)
        else:
            return self._sync_wrapper(func, fallback)
    
    def _async_wrapper(self, func: Callable, fallback: Callable = None):
        """Async wrapper for circuit breaker."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not self._can_execute():
                if fallback:
                    logger.warning(f"🚫 Circuit '{self.name}' open, using fallback")
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    else:
                        return fallback(*args, **kwargs)
                else:
                    raise CircuitBreakerError(f"Circuit breaker '{self.name}' is open")
            
            try:
                # Execute with timeout (if configured)
                if self.config.timeout is not None:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs), 
                        timeout=self.config.timeout
                    )
                else:
                    # Execute without timeout for unbounded operations
                    result = await func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.config.expected_exception as e:
                self._on_failure()
                raise
            except asyncio.TimeoutError:
                self._on_failure()
                raise CircuitBreakerError(f"Circuit breaker '{self.name}' timeout")
                
        return wrapper
    
    def _sync_wrapper(self, func: Callable, fallback: Callable = None):
        """Sync wrapper for circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self._can_execute():
                if fallback:
                    logger.warning(f"🚫 Circuit '{self.name}' open, using fallback")
                    return fallback(*args, **kwargs)
                else:
                    raise CircuitBreakerError(f"Circuit breaker '{self.name}' is open")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.config.expected_exception as e:
                self._on_failure()
                raise
                
        return wrapper
    
    def _can_execute(self) -> bool:
        """Check if request can be executed based on circuit state."""
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    self._increment_metric("total_half_opens")
                    self.metrics["last_state_change"] = time.time()
                    logger.info(f"🔄 Circuit '{self.name}' entering half-open state")
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return True
            
        return False
    
    def _increment_metric(self, key: str) -> None:
        """Increment a metric with bounds checking to prevent overflow."""
        if self.metrics[key] < self._metrics_max:
            self.metrics[key] += 1
        # If at max, counter stays at max (bounded)
    
    def _on_success(self):
        """Handle successful execution."""
        with self.lock:
            self._increment_metric("total_successes")
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self._increment_metric("total_closes")
                    self.metrics["last_state_change"] = time.time()
                    logger.info(f"✅ Circuit '{self.name}' closed - recovered")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self._increment_metric("total_failures")
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self._increment_metric("total_opens")
                self.metrics["last_state_change"] = time.time()
                logger.warning(f"❌ Circuit '{self.name}' opened - recovery failed")
            elif (self.state == CircuitState.CLOSED and 
                  self.failure_count >= self.config.failure_threshold):
                self.state = CircuitState.OPEN
                self._increment_metric("total_opens")
                self.metrics["last_state_change"] = time.time()
                logger.warning(f"❌ Circuit '{self.name}' opened - failure threshold reached")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self.lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time,
                "can_execute": self._can_execute()
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for monitoring and alerting."""
        with self.lock:
            current_time = time.time()
            uptime = current_time - self.metrics["created_at"]
            time_since_last_change = current_time - self.metrics["last_state_change"]
            
            return {
                "name": self.name,
                "state": self.state.value,
                "current_failure_count": self.failure_count,
                "current_success_count": self.success_count,
                "total_failures": self.metrics["total_failures"],
                "total_successes": self.metrics["total_successes"],
                "total_opens": self.metrics["total_opens"],
                "total_closes": self.metrics["total_closes"],
                "total_half_opens": self.metrics["total_half_opens"],
                "last_state_change": self.metrics["last_state_change"],
                "time_since_last_change_seconds": time_since_last_change,
                "uptime_seconds": uptime,
                "failure_rate": (
                    self.metrics["total_failures"] / (self.metrics["total_failures"] + self.metrics["total_successes"])
                    if (self.metrics["total_failures"] + self.metrics["total_successes"]) > 0
                    else 0.0
                ),
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold
                }
            }
    
    def reset(self):
        """Manually reset circuit breaker to closed state."""
        with self.lock:
            if self.state != CircuitState.CLOSED:
                self._increment_metric("total_closes")
                self.metrics["last_state_change"] = time.time()
            
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = 0
            logger.info(f"🔄 Circuit '{self.name}' manually reset")

class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different system components.
    """
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.lock = threading.RLock()
    
    def get_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        with self.lock:
            if name not in self.breakers:
                self.breakers[name] = CircuitBreaker(name, config)
            return self.breakers[name]
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers."""
        with self.lock:
            return {name: breaker.get_state() for name, breaker in self.breakers.items()}
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive metrics for all circuit breakers."""
        with self.lock:
            return {name: breaker.get_metrics() for name, breaker in self.breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self.lock:
            for breaker in self.breakers.values():
                breaker.reset()
            logger.info("🔄 All circuit breakers reset")

# Global circuit breaker manager
circuit_manager = CircuitBreakerManager()

# Pre-configured circuit breakers for SELO AI components
def get_llm_breaker() -> CircuitBreaker:
    """Get circuit breaker for LLM operations."""
    import os
    
    # Respect unbounded timeout settings from environment
    enforce_no_timeouts = os.getenv("REFLECTION_ENFORCE_NO_TIMEOUTS", "false").lower() in ("1", "true", "yes")
    llm_timeout = float(os.getenv("LLM_TIMEOUT", "60"))
    
    # Use unbounded timeout if configured, otherwise use environment setting
    if enforce_no_timeouts or llm_timeout <= 0:
        timeout_value = None  # No timeout
    else:
        timeout_value = llm_timeout
    
    config = CircuitBreakerConfig(
        failure_threshold=5,  # Increased to be more tolerant
        recovery_timeout=30.0,
        timeout=timeout_value,
        expected_exception=(Exception,)
    )
    return circuit_manager.get_breaker("llm_controller", config)

def get_database_breaker() -> CircuitBreaker:
    """Get circuit breaker for database operations."""
    config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=10.0,
        timeout=30.0,
        expected_exception=(Exception,)
    )
    return circuit_manager.get_breaker("database", config)

def get_reflection_breaker() -> CircuitBreaker:
    """Get circuit breaker for reflection operations."""
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60.0,
        timeout=120.0,
        expected_exception=(Exception,)
    )
    return circuit_manager.get_breaker("reflection_processor", config)

def get_vector_store_breaker() -> CircuitBreaker:
    """Get circuit breaker for vector store operations."""
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30.0,
        timeout=45.0,
        expected_exception=(Exception,)
    )
    return circuit_manager.get_breaker("vector_store", config)

# Convenience decorators
llm_circuit = get_llm_breaker()
database_circuit = get_database_breaker()
reflection_circuit = get_reflection_breaker()
vector_store_circuit = get_vector_store_breaker()
