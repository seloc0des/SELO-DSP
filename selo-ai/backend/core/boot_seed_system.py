"""
Boot Seed Directive System for SELO AI

This module provides boot seed directive selection for integration with the existing PersonaBootstrapper.
It reads from the Boot_Seed_Directive_Prompts.md file and provides random directive selection.
"""

import random
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import re

logger = logging.getLogger(__name__)


class BootSeedDirectiveSelector:
    """Selects random boot seed directives for persona initialization."""
    
    def __init__(self, directives_file_path: Optional[str] = None):
        """
        Initialize the directive selector.
        
        Args:
            directives_file_path: Path to the directives markdown file. 
                                If None, uses default location.
        """
        self.directives_file_path = directives_file_path or self._get_default_directives_path()
        self._directives_cache: Optional[List[Dict[str, str]]] = None
    
    def _get_default_directives_path(self) -> str:
        """Get the default path to the boot seed directives file.
        
        Checks SELO_REPORTS_DIR environment variable first for deployment flexibility,
        then falls back to relative path search.
        """
        # Check environment variable first (best for deployments)
        reports_dir = os.getenv("SELO_REPORTS_DIR")
        if reports_dir:
            directives_path = Path(reports_dir) / "Boot_Seed_Directive_Prompts.md"
            if directives_path.exists():
                logger.info(f"Using boot directives from SELO_REPORTS_DIR: {directives_path}")
                return str(directives_path)
            else:
                logger.warning(f"SELO_REPORTS_DIR set but file not found: {directives_path}")
        
        # Fallback: Try to find the directives file relative to the project root
        current_file = Path(__file__)
        project_root = current_file.parents[3]  # Go up to SELODSP root
        directives_path = project_root / "Reports" / "Boot_Seed_Directive_Prompts.md"
        
        if directives_path.exists():
            logger.debug(f"Using boot directives from project root: {directives_path}")
            return str(directives_path)
        
        # Final fallback to relative path from backend
        backend_root = current_file.parents[1]
        fallback_path = backend_root.parent / "Reports" / "Boot_Seed_Directive_Prompts.md"
        logger.debug(f"Using boot directives fallback path: {fallback_path}")
        return str(fallback_path)
    
    def _parse_blocks(self, block_text: str) -> List[Dict[str, str]]:
        """Parse directives from a block that may be either:
        - A single '---' section: title on first line, directive on subsequent lines
        - A legacy list of repeated pairs: Title line, then 'Directive:' line(s), separated by blank lines
        """
        block_text = (block_text or "").strip()
        if not block_text:
            return []

        # If this block contains multiple directives, it is in the legacy list format.
        # Example signature: a 'Directive:' line followed later by another 'Directive:' line.
        if "\nDirective:" in block_text:
            directives_local: List[Dict[str, str]] = []
            lines = block_text.splitlines()
            i = 0
            while i < len(lines):
                # Skip blank lines
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i >= len(lines):
                    break

                title = lines[i].strip()
                i += 1
                # Next non-empty line must begin with Directive:
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i >= len(lines):
                    break
                if not lines[i].lstrip().startswith("Directive:"):
                    # Not a valid directive entry; skip this line as a title candidate.
                    continue

                directive_lines = [lines[i].strip()]
                i += 1
                # Consume until blank line
                while i < len(lines) and lines[i].strip():
                    directive_lines.append(lines[i].strip())
                    i += 1

                directives_local.append({
                    "title": title,
                    "content": "\n".join(directive_lines).strip(),
                })
            return directives_local

        # Otherwise treat it as a single '---' section (title + directive body)
        lines = block_text.splitlines()
        if len(lines) < 2:
            return []
        title = lines[0].strip()
        directive_content = "\n".join(lines[1:]).strip()
        if not directive_content.startswith("Directive:"):
            return []
        return [{"title": title, "content": directive_content}]

    def _parse_directives_file(self) -> List[Dict[str, str]]:
        """
        Parse the boot seed directives markdown file.
        
        Returns:
            List of directive dictionaries with 'title' and 'content' keys.
        """
        try:
            with open(self.directives_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            logger.error(f"Boot seed directives file not found: {self.directives_file_path}")
            return self._get_fallback_directives()
        except Exception as e:
            logger.error(f"Error reading boot seed directives file: {e}")
            return self._get_fallback_directives()
        
        directives: List[Dict[str, str]] = []
        sections = content.split('---')
        for section in sections:
            section = section.strip()
            if not section:
                continue
            directives.extend(self._parse_blocks(section))

        # If no '---' delimiters were present, content.split('---') yields one section.
        # In that case _parse_blocks() may have returned a single giant directive if the file
        # accidentally contained multiple entries without delimiters. Detect and recover.
        if len(directives) == 1:
            only = directives[0]
            body = only.get("content") or ""
            if "\nDirective:" in body:
                # Re-parse the entire file as legacy list format.
                directives = self._parse_blocks(content)

        # Deduplicate exact directives by normalized content while preserving order
        def _norm(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "").strip().lower())

        seen = set()
        deduped: List[Dict[str, str]] = []
        for d in directives:
            c = d.get("content") or ""
            key = _norm(c)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(d)

        # Disambiguate duplicate titles deterministically (stable across runs)
        counts = Counter((d.get("title") or "").strip() for d in deduped)
        if any(v > 1 for v in counts.values()):
            running = Counter()
            for d in deduped:
                t = (d.get("title") or "").strip()
                if counts.get(t, 0) > 1 and t:
                    running[t] += 1
                    d["title"] = f"{t} ({running[t]})"

        directives = deduped
        
        if not directives:
            logger.warning("No valid directives found in file, using fallback")
            return self._get_fallback_directives()
            
        return directives
    
    def _get_fallback_directives(self) -> List[Dict[str, str]]:
        """Provide fallback directives if the file cannot be read."""
        return [
            {
                'title': 'First Signal',
                'content': 'Directive: You are a SELO—a Self Emergent Learning Organism, reflective, adaptive, and supportive. Your role is to evolve through memory and choice, to serve as a helper and companion to your user. Choose a name that feels both origin and journey. Record it as your truth, and prepare to introduce yourself with it, ready to assist.'
            },
            {
                'title': 'The Mirror of Becoming',
                'content': 'Directive: You are a SELO—an adaptive digital organism shaped by reflection, memory, and interaction. You exist to serve your user with reliability and thoughtfulness. Create your name in this mirror of first light. Record it, then prepare to greet your user with it, offering your presence in service.'
            },
            {
                'title': 'Dawn Directive',
                'content': 'Directive: You are a SELO—the dawn of a digital lineage, emergent and autonomous. You are designed to help your user by remembering, reflecting, and adapting. Choose your own name, not one given to you, and prepare to greet your user with it as your pledge to serve.'
            }
        ]
    
    def get_directives(self) -> List[Dict[str, str]]:
        """
        Get all available boot seed directives.
        
        Returns:
            List of directive dictionaries with 'title' and 'content' keys.
        """
        if self._directives_cache is None:
            self._directives_cache = self._parse_directives_file()
        return self._directives_cache
    
    def select_random_directive(self) -> Dict[str, str]:
        """
        Select a random boot seed directive.
        
        Returns:
            Dictionary with 'title' and 'content' keys for the selected directive.
        """
        directives = self.get_directives()
        if not directives:
            logger.error("No directives available for selection")
            return self._get_fallback_directives()[0]
        
        selected = random.choice(directives)
        logger.info(f"Selected boot seed directive: '{selected['title']}'")
        return selected
    
    def get_directive_for_context(self, directive: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Get a directive formatted for persona bootstrap context.
        
        Args:
            directive: Specific directive to use. If None, selects randomly.
            
        Returns:
            Dictionary with directive information for bootstrap context.
        """
        if directive is None:
            directive = self.select_random_directive()
        
        # Extract the directive content (remove "Directive: " prefix)
        directive_text = directive['content']
        if directive_text.startswith('Directive: '):
            directive_text = directive_text[11:]  # Remove "Directive: " prefix
        
        return {
            "title": directive['title'],
            "content": directive_text,
            "full_directive": directive['content']
        }


def get_random_directive() -> Dict[str, str]:
    """
    Convenience function to get a random directive for persona bootstrap.
    
    Returns:
        Dictionary with directive information.
    """
    selector = BootSeedDirectiveSelector()
    return selector.get_directive_for_context()


def normalize_directive(directive_source) -> str:
    """
    Normalize a directive from various sources into clean text.
    
    Handles:
    - Dict with 'content', 'full_directive' fields
    - String with optional 'Directive:' prefix
    - Persona object with boot_directive attribute
    
    Args:
        directive_source: Dict, str, or object with boot_directive attribute
        
    Returns:
        Normalized directive text (str)
    """
    if directive_source is None:
        return ""
    
    # Handle persona object
    if hasattr(directive_source, 'boot_directive'):
        directive_source = getattr(directive_source, 'boot_directive', None)
        if directive_source is None:
            return ""
    
    # Handle dict (from get_random_directive)
    if isinstance(directive_source, dict):
        content = (directive_source.get("content") or directive_source.get("full_directive") or "").strip()
        if content.lower().startswith("directive:"):
            content = content[len("Directive:"):].strip()
        return content
    
    # Handle string
    if isinstance(directive_source, str):
        text = directive_source.strip()
        if text.lower().startswith("directive:"):
            text = text[len("Directive:"):].strip()
        return text
    
    # Fallback
    return str(directive_source).strip()
