"""
Import Helper Utilities

This module provides standardized patterns for handling relative vs absolute imports
in the SELO codebase. These patterns are necessary because the code needs to work in
multiple execution contexts:

1. As an installed package (relative imports work)
2. As standalone scripts (absolute imports needed)
3. In test environments (both patterns may be needed)

The try/except ImportError pattern is INTENTIONAL and should not be removed.
"""

from typing import Any, Callable, TypeVar
import sys
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


def import_with_fallback(
    relative_path: str,
    absolute_path: str,
    item_name: str = None
) -> Any:
    """
    Import a module or item with fallback from relative to absolute import.
    
    This is the standard pattern used throughout the SELO codebase for
    handling imports that need to work in both package and script contexts.
    
    Args:
        relative_path: Relative import path (e.g., "..utils.text_utils")
        absolute_path: Absolute import path (e.g., "backend.utils.text_utils")
        item_name: Specific item to import (e.g., "count_words"), or None for module
        
    Returns:
        The imported module or item
        
    Raises:
        ImportError: If both import attempts fail
        
    Example:
        >>> # Import a specific function
        >>> count_words = import_with_fallback(
        ...     "..utils.text_utils",
        ...     "backend.utils.text_utils",
        ...     "count_words"
        ... )
        
        >>> # Import entire module
        >>> text_utils = import_with_fallback(
        ...     "..utils.text_utils",
        ...     "backend.utils.text_utils"
        ... )
    """
    try:
        # Try relative import first (works when run as package)
        if item_name:
            module = __import__(relative_path, fromlist=[item_name], level=0)
            return getattr(module, item_name)
        else:
            return __import__(relative_path, level=0)
    except (ImportError, ValueError) as e1:
        try:
            # Fall back to absolute import (works when run as script)
            if item_name:
                module = __import__(absolute_path, fromlist=[item_name])
                return getattr(module, item_name)
            else:
                return __import__(absolute_path)
        except ImportError as e2:
            raise ImportError(
                f"Failed to import from both relative ({relative_path}) "
                f"and absolute ({absolute_path}) paths. "
                f"Relative error: {e1}. Absolute error: {e2}"
            )


def safe_import(module_path: str, item_name: str = None) -> Any:
    """
    Safely import a module or item, returning None if import fails.
    
    Useful for optional dependencies or feature detection.
    
    Args:
        module_path: Module path to import
        item_name: Specific item to import, or None for module
        
    Returns:
        The imported module/item, or None if import fails
        
    Example:
        >>> torch = safe_import("torch")
        >>> if torch is not None:
        ...     # Use torch features
        ...     pass
    """
    try:
        if item_name:
            module = __import__(module_path, fromlist=[item_name])
            return getattr(module, item_name)
        else:
            return __import__(module_path)
    except ImportError:
        logger.debug(f"Optional import failed: {module_path}")
        return None


# Standard pattern documentation for developers
IMPORT_PATTERN_DOCS = """
Standard Import Pattern for SELO Codebase
==========================================

When writing code that needs to work in both package and script contexts,
use this pattern:

```python
try:
    from ..module import item  # Relative import (package context)
except ImportError:
    from backend.module import item  # Absolute import (script context)
```

Why This Pattern Exists:
------------------------
1. Persona bootstrap runs as standalone script
2. Tests may run from different directories
3. Development tools may execute files directly
4. Package installation uses relative imports

Alternative Using Helper:
------------------------
```python
from backend.utils.import_helpers import import_with_fallback

item = import_with_fallback("..module", "backend.module", "item")
```

DO NOT "fix" these patterns by removing the try/except blocks.
They are intentional and necessary for the codebase to function.
"""


def get_import_pattern_docs() -> str:
    """Return documentation about the import pattern."""
    return IMPORT_PATTERN_DOCS


if __name__ == "__main__":
    # Print documentation when run directly
    print(IMPORT_PATTERN_DOCS)
