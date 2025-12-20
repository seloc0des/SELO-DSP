"""
Text processing utility functions.

This module provides common text processing functions used across the SELO backend.
"""

import re
from typing import Any


def count_words(text: Any) -> int:
    """
    Count words in text using word boundary regex.
    
    This is the standardized word counting function used throughout SELO.
    It handles non-string inputs gracefully and uses consistent word boundaries.
    
    Args:
        text: Text to count words in (can be any type, non-strings return 0)
        
    Returns:
        int: Number of words found using word boundary detection
        
    Examples:
        >>> count_words("Hello world!")
        2
        >>> count_words("  spaced   text  ")
        2
        >>> count_words("")
        0
        >>> count_words(None)
        0
        >>> count_words(123)
        0
    """
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b\w+\b", text))


def count_words_simple(text: Any) -> int:
    """
    Count words using simple whitespace splitting.
    
    This is an alternative word counting method that uses whitespace splitting
    instead of regex word boundaries. May be useful for different counting needs.
    
    Args:
        text: Text to count words in
        
    Returns:
        int: Number of words found using whitespace splitting
    """
    if not isinstance(text, str):
        return 0
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text by collapsing multiple spaces and trimming.
    
    Args:
        text: Text to normalize
        
    Returns:
        str: Text with normalized whitespace
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip())


def truncate_words(text: str, max_words: int, ellipsis: str = "...") -> str:
    """
    Truncate text to a maximum number of words.
    
    Args:
        text: Text to truncate
        max_words: Maximum number of words to keep
        ellipsis: String to append if truncation occurs
        
    Returns:
        str: Truncated text
    """
    if not isinstance(text, str) or max_words <= 0:
        return ""
    
    words = text.split()
    if len(words) <= max_words:
        return text
    
    return " ".join(words[:max_words]) + ellipsis
