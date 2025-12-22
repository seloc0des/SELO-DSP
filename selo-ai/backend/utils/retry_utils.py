"""
Retry utilities for handling transient failures in external services.

Provides decorators and helper functions for implementing retry logic
with exponential backoff for vector stores, LLMs, and other services.
"""

import asyncio
import logging
from functools import wraps
from typing import Callable, Optional, Tuple, Type, Any

logger = logging.getLogger("selo.utils.retry")


async def retry_with_backoff(
    func: Callable,
    *args,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    **kwargs
) -> Any:
    """
    Execute an async function with exponential backoff retry logic.
    
    Args:
        func: Async function to execute
        *args: Positional arguments for func
        max_attempts: Maximum number of attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        backoff_factor: Multiplier for delay on each retry (default: 2.0)
        exceptions: Tuple of exception types to catch and retry (default: all exceptions)
        on_retry: Optional callback function(attempt_num, exception) called on retry
        **kwargs: Keyword arguments for func
        
    Returns:
        Result from successful function execution
        
    Raises:
        The last exception if all retries are exhausted
    """
    last_exception = None
    delay = initial_delay
    
    for attempt in range(1, max_attempts + 1):
        try:
            result = await func(*args, **kwargs)
            if attempt > 1:
                logger.info(f"Operation succeeded on attempt {attempt}/{max_attempts}")
            return result
            
        except exceptions as e:
            last_exception = e
            
            if attempt == max_attempts:
                logger.error(
                    f"Operation failed after {max_attempts} attempts: {e}",
                    exc_info=True
                )
                raise
            
            if on_retry:
                on_retry(attempt, e)
            
            logger.warning(
                f"Operation failed on attempt {attempt}/{max_attempts}: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            
            await asyncio.sleep(delay)
            delay *= backoff_factor
    
    # This should never be reached, but for type safety
    if last_exception:
        raise last_exception


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator to add retry logic with exponential backoff to async functions.
    
    Usage:
        @with_retry(max_attempts=3, initial_delay=1.0)
        async def my_function(arg1, arg2):
            # ... code that might fail transiently
            pass
    
    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exception types to catch and retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_with_backoff(
                func,
                *args,
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                backoff_factor=backoff_factor,
                exceptions=exceptions,
                **kwargs
            )
        return wrapper
    return decorator


class RetryableOperation:
    """
    Context manager for retryable operations with custom error handling.
    
    Usage:
        async with RetryableOperation("vector_store_add", max_attempts=3) as retry_op:
            result = await retry_op.execute(vector_store.add_embedding, data)
    """
    
    def __init__(
        self,
        operation_name: str,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ):
        self.operation_name = operation_name
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.attempt = 0
        self.last_error: Optional[Exception] = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Suppress exceptions - they're handled in execute()
        return False
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute the function with retry logic.
        
        Args:
            func: Function to execute (can be sync or async)
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result from successful execution
            
        Raises:
            Last exception if all retries exhausted
        """
        delay = self.initial_delay
        
        for attempt in range(1, self.max_attempts + 1):
            self.attempt = attempt
            
            try:
                # Handle both sync and async functions
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                if attempt > 1:
                    logger.info(
                        f"{self.operation_name} succeeded on attempt {attempt}/{self.max_attempts}"
                    )
                
                return result
                
            except Exception as e:
                self.last_error = e
                
                if attempt == self.max_attempts:
                    logger.error(
                        f"{self.operation_name} failed after {self.max_attempts} attempts: {e}",
                        exc_info=True
                    )
                    raise
                
                logger.warning(
                    f"{self.operation_name} failed on attempt {attempt}/{self.max_attempts}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                
                await asyncio.sleep(delay)
                delay *= self.backoff_factor
        
        # This should never be reached
        if self.last_error:
            raise self.last_error
