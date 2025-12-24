#!/usr/bin/env python3
"""
Persona & Mantra Bootstrap Script

Runs the PersonaBootstrapper to generate an initial self-authored persona
(including name and mantra) before the service is created.

Usage:
    python -m scripts.bootstrap_persona [--verbose] [--mock]
"""
import argparse
import logging
import sys
from typing import Optional

# Ensure script helpers are importable (same directory)
# Try relative import first (when run as module), then absolute (when run as script)
try:
    from .script_helpers import ScriptContext, setup_script_logging  # type: ignore
except ImportError:
    try:
        from script_helpers import ScriptContext, setup_script_logging  # type: ignore
    except Exception as e:
        import logging
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Failed to import script_helpers: {e}")
        sys.exit(2)


def run_bootstrap(mock: bool = False) -> int:
    """Run persona bootstrap to completion. Returns exit code (0=success)."""
    # Defer imports until context is established to ensure env is loaded from backend/.env
    from persona.bootstrapper import PersonaBootstrapper  # type: ignore
    import asyncio

    with ScriptContext(mock_mode=mock, log_level=logging.getLevelName(logging.getLogger().level)) as services:
        llm_router = services.get("llm_router")
        prompt_builder = services.get("prompt_builder")
        persona_repo = services.get("persona_repo")
        user_repo = services.get("user_repo")

        if not all([llm_router, prompt_builder, persona_repo, user_repo]):
            logging.error("Missing required services in DI context; cannot bootstrap persona")
            return 2

        conversation_repo = services.get("conversation_repo")
        reflection_repo = services.get("reflection_repo")
        bootstrapper = PersonaBootstrapper(
            llm_router=llm_router,
            prompt_builder=prompt_builder,
            persona_repo=persona_repo,
            user_repo=user_repo,
            conversation_repo=conversation_repo,
            reflection_repo=reflection_repo,
        )

        logging.info("=" * 80)
        logging.info("Starting persona bootstrap (NO FALLBACKS MODE)")
        logging.info("All data must be LLM-generated and DB-persisted")
        logging.info("=" * 80)
        
        # Run the async ensure_persona in the event loop
        result: Optional[dict] = asyncio.get_event_loop().run_until_complete(bootstrapper.ensure_persona())
        
        if not result:
            logging.error("=" * 80)
            logging.error("BOOTSTRAP FAILED: ensure_persona() returned None")
            logging.error("This means LLM generation or DB persistence failed")
            logging.error("NO FALLBACK DATA - installation cannot proceed")
            logging.error("=" * 80)
            return 1

        # Verify all required fields are present in result
        name = (result or {}).get("name")
        mantra = (result or {}).get("mantra", "") or ""
        description = (result or {}).get("description", "") or ""
        
        logging.info("=" * 80)
        logging.info("BOOTSTRAP RESULT VALIDATION")
        logging.info("=" * 80)
        
        validation_failures = []
        
        # Validate name
        if not name or not name.strip() or name.strip() in ("SELO", ""):
            validation_failures.append(f"Invalid name: '{name}'")
        else:
            logging.info(f"✓ Valid name: {name}")
        
        # Validate mantra
        if not mantra or not mantra.strip():
            validation_failures.append("Mantra is empty")
        else:
            logging.info(f"✓ Valid mantra: {mantra[:60]}...")
        
        # Validate description
        if not description or len(description.strip()) < 10:
            validation_failures.append(f"Description too short or empty (length: {len(description or '')})")
        else:
            logging.info(f"✓ Valid description: {len(description)} chars")
        
        # Check for other required fields
        if not result.get("values"):
            validation_failures.append("Missing values")
        else:
            logging.info(f"✓ Values present: {list(result['values'].keys()) if isinstance(result['values'], dict) else 'invalid'}")
        
        if validation_failures:
            logging.error("=" * 80)
            logging.error("BOOTSTRAP VALIDATION FAILED")
            logging.error("=" * 80)
            for failure in validation_failures:
                logging.error(f"  ✗ {failure}")
            logging.error("=" * 80)
            logging.error("NO FALLBACK DATA - installation cannot proceed")
            logging.error("=" * 80)
            return 1
        
        logging.info("=" * 80)
        logging.info("✓✓✓ BOOTSTRAP VALIDATION SUCCESSFUL ✓✓✓")
        logging.info("=" * 80)
        logging.info(f"Bootstrap complete: name={name}, mantra={mantra[:80].replace(chr(10), ' ')}")
        logging.info("=" * 80)
        
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap persona (pre-service)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose (DEBUG) logging")
    # Mock mode removed - installation requires real services
    args = parser.parse_args()

    # Configure logging
    setup_script_logging("DEBUG" if args.verbose else "INFO")

    try:
        # Always use production services (mock=False)
        # run_bootstrap is now synchronous - it handles async internally
        return run_bootstrap(mock=False)
    except KeyboardInterrupt:
        logging.error("Bootstrap interrupted by user")
        return 130
    except Exception as e:
        logging.exception("Bootstrap failed with an unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
