"""
LLM Retry Utilities

Provides utility functions for retry logic with temperature adjustment,
backoff strategies, and other retry-related helpers.
"""

from typing import Optional


def get_retry_temperature(
    base_temp: float,
    attempt: int,
    step: float = 0.05,
    min_temp: float = 0.05,
    max_temp: Optional[float] = None
) -> float:
    """
    Calculate temperature for a retry attempt with progressive tightening.
    
    Progressive temperature tightening encourages more deterministic responses
    on retries, which is useful when initial attempts fail validation.
    
    Args:
        base_temp: Starting temperature (attempt 0)
        attempt: Current attempt number (0-indexed)
        step: Amount to decrease temperature per attempt (default: 0.05)
        min_temp: Minimum temperature floor (default: 0.05)
        max_temp: Maximum temperature ceiling (optional)
        
    Returns:
        Adjusted temperature for the current attempt
        
    Examples:
        >>> get_retry_temperature(0.2, 0)  # First attempt
        0.2
        >>> get_retry_temperature(0.2, 1)  # Second attempt
        0.15
        >>> get_retry_temperature(0.2, 2)  # Third attempt
        0.1
        >>> get_retry_temperature(0.2, 3)  # Fourth attempt
        0.05
        >>> get_retry_temperature(0.2, 4)  # Fifth attempt (floor)
        0.05
        
    Note:
        Temperature tightening is most effective for:
        - JSON generation (reduces format errors)
        - Constrained text generation (reduces violations)
        - Classification tasks (reduces ambiguity)
        
        Less useful for:
        - Creative generation (tightening reduces quality)
        - Open-ended responses (defeats purpose)
    """
    # Calculate tightened temperature
    tightened_temp = base_temp - (attempt * step)
    
    # Apply floor
    tightened_temp = max(min_temp, tightened_temp)
    
    # Apply ceiling if specified
    if max_temp is not None:
        tightened_temp = min(max_temp, tightened_temp)
    
    return tightened_temp


def get_exponential_backoff_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True
) -> float:
    """
    Calculate exponential backoff delay for retry attempts.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay ceiling in seconds (default: 60.0)
        exponential: Use exponential backoff vs linear (default: True)
        
    Returns:
        Delay in seconds before next retry
        
    Examples:
        >>> get_exponential_backoff_delay(0)  # First retry
        1.0
        >>> get_exponential_backoff_delay(1)  # Second retry
        2.0
        >>> get_exponential_backoff_delay(2)  # Third retry
        4.0
        >>> get_exponential_backoff_delay(3)  # Fourth retry
        8.0
    """
    if exponential:
        delay = base_delay * (2 ** attempt)
    else:
        delay = base_delay * (attempt + 1)
    
    return min(delay, max_delay)


def should_retry(
    attempt: int,
    max_attempts: int,
    error: Exception,
    retryable_errors: Optional[tuple] = None
) -> bool:
    """
    Determine if an operation should be retried.
    
    Args:
        attempt: Current attempt number (0-indexed)
        max_attempts: Maximum number of attempts
        error: The exception that was raised
        retryable_errors: Tuple of error types that are retryable (optional)
        
    Returns:
        True if should retry, False otherwise
        
    Examples:
        >>> should_retry(0, 3, ValueError())
        True
        >>> should_retry(2, 3, ValueError())
        False  # Last attempt
        >>> should_retry(1, 3, KeyboardInterrupt())
        False  # Not retryable
    """
    # Never retry on last attempt
    if attempt >= max_attempts - 1:
        return False
    
    # Never retry certain critical errors
    non_retryable = (KeyboardInterrupt, SystemExit)
    if isinstance(error, non_retryable):
        return False
    
    # If specific retryable errors specified, check them
    if retryable_errors is not None:
        return isinstance(error, retryable_errors)
    
    # By default, retry all other errors
    return True
