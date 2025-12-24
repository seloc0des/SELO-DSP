#!/usr/bin/env python3
"""
Update Reflection Examples Script

Updates existing reflection examples in the database to the new expanded versions
that match the 250-350 word target range.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.repositories.example import ExampleRepository
from backend.db.session import get_session

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def update_examples():
    """Update existing reflection examples to new expanded versions."""
    
    logger.info("🔄 Updating reflection examples to expanded versions...")
    
    repo = ExampleRepository()
    
    # Get all existing examples
    async with get_session() as session:
        from sqlalchemy import select
        from backend.db.models.example import ReflectionExample
        
        # Delete all existing examples
        result = await session.execute(select(ReflectionExample))
        existing_examples = result.scalars().all()
        
        count = len(existing_examples)
        logger.info(f"Found {count} existing examples to replace")
        
        # Delete existing examples
        for example in existing_examples:
            await session.delete(example)
        
        await session.commit()
        logger.info(f"✅ Deleted {count} old examples")
    
    # Re-seed with new expanded examples
    logger.info("🌱 Re-seeding with expanded examples...")
    await repo._seed_initial_examples()
    
    logger.info("✅ Successfully updated all reflection examples!")
    logger.info("🎉 Examples now match the 250-350 word target range")


async def main():
    """Main entry point."""
    try:
        await update_examples()
        return 0
    except Exception as e:
        logger.error(f"❌ Failed to update examples: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
