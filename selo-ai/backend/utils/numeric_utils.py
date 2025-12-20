"""
Numeric utility functions for common mathematical operations.
"""

def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Clamp a value to a specified range.
    
    Args:
        value: Value to clamp
        min_val: Minimum allowed value (default: 0.0)
        max_val: Maximum allowed value (default: 1.0)
        
    Returns:
        Clamped value within [min_val, max_val]
        
    Examples:
        >>> clamp(1.5)
        1.0
        >>> clamp(-0.5)
        0.0
        >>> clamp(0.5)
        0.5
        >>> clamp(42, 0, 100)
        42
        >>> clamp(150, 0, 100)
        100
    """
    return min(max_val, max(min_val, value))
