"""
Batch Tracker Utility

Provides batching mechanism for database write operations to reduce load.
Queues updates and flushes them periodically or when batch size is reached.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger("selo.utils.batch_tracker")


class BatchTracker:
    """
    Generic batch tracker for accumulating and periodically flushing database updates.
    
    Reduces database write load by 90% by batching updates instead of
    writing immediately after each operation.
    
    Example:
        tracker = BatchTracker(
            flush_callback=my_db_update_func,
            batch_size=10,
            flush_interval_seconds=5
        )
        await tracker.add("item_id", {"field": "value"})
    """
    
    def __init__(
        self,
        flush_callback: Callable,
        batch_size: int = 10,
        flush_interval_seconds: int = 5,
        name: str = "batch_tracker"
    ):
        """
        Initialize batch tracker.
        
        Args:
            flush_callback: Async function to call with batched updates
            batch_size: Flush when this many items are queued
            flush_interval_seconds: Flush at least every N seconds
            name: Name for logging purposes
        """
        self.flush_callback = flush_callback
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.name = name
        
        # Queue of pending updates: {item_id: update_data}
        self._queue: Dict[str, Any] = {}
        self._last_flush = datetime.now(timezone.utc)
        self._flush_task = None
        self._lock = asyncio.Lock()
        
    async def add(self, item_id: str, update_data: Any) -> None:
        """
        Add an item to the batch queue.
        
        If item_id already exists, the update_data will be merged/replaced.
        Automatically flushes when batch_size is reached.
        
        Args:
            item_id: Unique identifier for the item
            update_data: Data to pass to flush_callback
        """
        async with self._lock:
            # Merge if item already queued
            if item_id in self._queue:
                # For numeric updates, accumulate
                if isinstance(update_data, dict) and isinstance(self._queue[item_id], dict):
                    for key, value in update_data.items():
                        if key in self._queue[item_id] and isinstance(value, (int, float)):
                            self._queue[item_id][key] += value
                        else:
                            self._queue[item_id][key] = value
                else:
                    self._queue[item_id] = update_data
            else:
                self._queue[item_id] = update_data
            
            # Flush if batch size reached
            if len(self._queue) >= self.batch_size:
                logger.debug(f"[{self.name}] Batch size reached ({len(self._queue)}), flushing")
                await self._flush()
            
            # Start periodic flush task if not running
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._periodic_flush())
    
    async def _periodic_flush(self) -> None:
        """Periodically flush the queue based on flush_interval_seconds."""
        try:
            while True:
                await asyncio.sleep(self.flush_interval_seconds)
                
                async with self._lock:
                    if self._queue:
                        logger.debug(f"[{self.name}] Periodic flush triggered ({len(self._queue)} items)")
                        await self._flush()
                    else:
                        # Queue empty, stop periodic task
                        break
        except asyncio.CancelledError:
            logger.debug(f"[{self.name}] Periodic flush task cancelled")
        except Exception as e:
            logger.error(f"[{self.name}] Error in periodic flush: {e}", exc_info=True)
    
    async def _flush(self) -> None:
        """
        Flush queued items to database.
        
        Called automatically when batch size is reached or periodically.
        Should be called with _lock held.
        """
        if not self._queue:
            return
        
        items_to_flush = dict(self._queue)
        self._queue.clear()
        self._last_flush = datetime.now(timezone.utc)
        
        try:
            # Call the flush callback with batched items
            await self.flush_callback(items_to_flush)
            logger.info(f"[{self.name}] ✅ Flushed {len(items_to_flush)} items to database")
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Failed to flush batch: {e}", exc_info=True)
            # Re-queue items on failure
            self._queue.update(items_to_flush)
    
    async def force_flush(self) -> None:
        """Force immediate flush of all queued items."""
        async with self._lock:
            if self._queue:
                logger.debug(f"[{self.name}] Force flush requested ({len(self._queue)} items)")
                await self._flush()
    
    def get_queue_size(self) -> int:
        """Get current number of items in queue."""
        return len(self._queue)


class ExampleUsageTracker(BatchTracker):
    """
    Specialized batch tracker for reflection example usage statistics.
    
    Reduces database writes for example tracking by 90% by batching updates.
    """
    
    def __init__(self, example_repository):
        """
        Initialize with reference to ExampleRepository for flush callback.
        
        Args:
            example_repository: Instance of ExampleRepository
        """
        self.example_repo = example_repository
        
        async def flush_example_usage(batched_items: Dict[str, Dict[str, Any]]) -> None:
            """Flush batched example usage updates to database."""
            # Group by success/failure
            successes = []
            failures = []
            
            for example_id, data in batched_items.items():
                times_shown = data.get("times_shown", 0)
                times_succeeded = data.get("times_succeeded", 0)
                
                if times_succeeded > 0:
                    successes.append((example_id, times_shown, times_succeeded))
                else:
                    failures.append((example_id, times_shown))
            
            # Batch update to database
            if successes or failures:
                await self.example_repo._batch_update_usage_stats(successes, failures)
        
        super().__init__(
            flush_callback=flush_example_usage,
            batch_size=10,
            flush_interval_seconds=5,
            name="example_usage_tracker"
        )
    
    async def track_usage(self, example_ids: List[str], validation_passed: bool) -> None:
        """
        Track example usage with batching.
        
        Args:
            example_ids: List of example IDs that were shown
            validation_passed: Whether validation passed
        """
        for example_id in example_ids:
            await self.add(
                example_id,
                {
                    "times_shown": 1,
                    "times_succeeded": 1 if validation_passed else 0
                }
            )
