#!/usr/bin/env python3
"""
Add logging to silent exception handlers across the codebase.

This script identifies and fixes silent exception handlers that use:
- except Exception: pass
- except: pass
- except SomeException: pass (without logging)

It adds appropriate logging statements to make debugging possible.
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Patterns to match silent exception handlers
# These handle various formatting styles including comments
PATTERNS = [
    # except Exception: pass (with optional comment)
    (r'(\s+)(except\s+Exception:\s*)(?:#[^\n]*)?(\n\s+)(pass)(?:\s*(?:#[^\n]*)?)$', 
     r'\1\2\3logger.error("Silent exception caught", exc_info=True)\n\3# \4'),
    
    # except: pass (with optional comment)
    (r'(\s+)(except:\s*)(?:#[^\n]*)?(\n\s+)(pass)(?:\s*(?:#[^\n]*)?)$',
     r'\1\2\3logger.error("Silent bare exception caught", exc_info=True)\n\3# \4'),
    
    # except SomeSpecificException: pass (with optional comment)
    (r'(\s+)(except\s+\w+Error:\s*)(?:#[^\n]*)?(\n\s+)(pass)(?:\s*(?:#[^\n]*)?)$',
     r'\1\2\3logger.warning("Exception caught and ignored", exc_info=True)\n\3# \4'),
    
    # except (Multiple, Exceptions): pass (with optional comment)
    (r'(\s+)(except\s+\([^)]+\):\s*)(?:#[^\n]*)?(\n\s+)(pass)(?:\s*(?:#[^\n]*)?)$',
     r'\1\2\3logger.debug("Expected exception caught", exc_info=True)\n\3# \4'),
    
    # except SomeException as e: pass
    (r'(\s+)(except\s+\w+\s+as\s+\w+:\s*)(?:#[^\n]*)?(\n\s+)(pass)(?:\s*(?:#[^\n]*)?)$',
     r'\1\2\3logger.debug("Named exception caught", exc_info=True)\n\3# \4'),
]

def has_logging_import(content: str) -> bool:
    """Check if file already imports logging."""
    return bool(re.search(r'^import logging', content, re.MULTILINE) or 
                re.search(r'^from .* import .*logging', content, re.MULTILINE))

def has_logger_defined(content: str) -> bool:
    """Check if file defines a logger."""
    return bool(re.search(r'logger\s*=\s*logging\.getLogger', content))

def add_logging_setup(content: str, filepath: str) -> str:
    """Add logging import and logger definition if missing."""
    lines = content.split('\n')
    
    # Find where to insert imports (after docstring, before first import)
    insert_idx = 0
    in_docstring = False
    docstring_char = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Handle docstrings
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if not in_docstring:
                in_docstring = True
                docstring_char = stripped[:3]
                if stripped.endswith(docstring_char) and len(stripped) > 3:
                    in_docstring = False
            elif stripped.endswith(docstring_char):
                in_docstring = False
                insert_idx = i + 1
            continue
        
        if in_docstring:
            continue
            
        # Found first import
        if stripped.startswith('import ') or stripped.startswith('from '):
            insert_idx = i
            break
    
    # Add logging import if missing
    if not has_logging_import(content):
        lines.insert(insert_idx, 'import logging')
        insert_idx += 1
    
    # Add logger definition if missing
    if not has_logger_defined(content):
        # Find module-level code (after imports)
        for i in range(insert_idx, len(lines)):
            if lines[i].strip() and not lines[i].strip().startswith('#'):
                if not (lines[i].strip().startswith('import ') or 
                       lines[i].strip().startswith('from ')):
                    # Insert logger before first non-import code
                    lines.insert(i, f'\nlogger = logging.getLogger(__name__)')
                    break
    
    return '\n'.join(lines)

def fix_silent_exceptions(content: str) -> Tuple[str, int]:
    """Fix silent exception handlers by adding logging."""
    fixed_count = 0
    
    for pattern, replacement in PATTERNS:
        new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
        if count > 0:
            content = new_content
            fixed_count += count
    
    return content, fixed_count

def process_file(filepath: Path) -> Tuple[int, bool]:
    """Process a single Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Fix silent exceptions
        fixed_content, fix_count = fix_silent_exceptions(original_content)
        
        if fix_count > 0:
            # Add logging setup if needed
            fixed_content = add_logging_setup(fixed_content, str(filepath))
            
            # Write back
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            return fix_count, True
        
        return 0, False
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return 0, False

def main():
    """Main entry point."""
    # Get backend directory
    script_dir = Path(__file__).parent
    backend_dir = script_dir.parent
    
    total_fixes = 0
    files_modified = 0
    
    print("Adding logging to silent exception handlers...")
    print("=" * 60)
    
    # Process all Python files in backend directory
    all_python_files = list(backend_dir.rglob("*.py"))
    
    # Skip certain directories/files
    skip_patterns = [
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "migrations",
        "alembic",
        "tests",
        "__init__.py",
    ]
    
    for filepath in sorted(all_python_files):
        # Skip files matching skip patterns
        if any(pattern in str(filepath) for pattern in skip_patterns):
            continue
        
        # Skip this script itself
        if filepath.name == "add_exception_logging.py":
            continue
            
        fix_count, modified = process_file(filepath)
        if modified:
            rel_path = filepath.relative_to(backend_dir)
            print(f"✓ {rel_path}: {fix_count} fixes")
            total_fixes += fix_count
            files_modified += 1
    
    print("=" * 60)
    print(f"Summary: {total_fixes} silent exceptions fixed in {files_modified} files")
    print(f"\nNote: This addresses silent 'pass' statements in exception handlers.")
    print(f"Review the changes and ensure logging levels are appropriate for each case.")
    
    return 0 if total_fixes > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
