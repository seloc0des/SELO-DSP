"""
Optimized database connection pooling for SELO AI Backend.

Provides efficient connection management with proper pooling,
health checks, performance monitoring, and retry logic.
"""

import logging
import asyncio
from typing import Optional, Callable, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.pool import QueuePool
from sqlalchemy import event, text
from sqlalchemy.exc import DisconnectionError, OperationalError
import os
import time
import random

logger = logging.getLogger(__name__)

class OptimizedConnectionPool:
    """Optimized database connection pool with health monitoring."""
    
    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._stats = {
            'connections_created': 0,
            'connections_closed': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'health_checks': 0,
            'health_failures': 0
        }
    
    def create_engine(self, database_url: str) -> AsyncEngine:
        """Create optimized async engine with connection pooling."""
        
        # Normalize PostgreSQL URL for asyncpg
        if database_url.startswith("postgresql://") and not database_url.startswith("postgresql+asyncpg://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        
        # Optimized pool settings
        pool_size = int(os.getenv("DB_POOL_SIZE", "20"))
        max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "30"))
        pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))  # 1 hour
        
        self.engine = create_async_engine(
            database_url,
            # Connection pool settings
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            pool_pre_ping=True,  # Validate connections before use
            
            # Performance settings
            echo=False,  # Disable SQL logging for performance
            future=True,
            
            # Connection arguments for asyncpg
            connect_args={
                "server_settings": {
                    "application_name": "selo_ai_backend",
                    "jit": "off",  # Disable JIT for faster connection times
                },
                "command_timeout": 60,
                "statement_cache_size": 0,  # Disable prepared statement cache
            }
        )
        
        # Add event listeners for monitoring
        self._setup_event_listeners()
        
        # Start health check task
        self._start_health_check()
        
        logger.info(f"Created optimized connection pool: size={pool_size}, max_overflow={max_overflow}")
        return self.engine
    
    def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners for monitoring."""
        if not self.engine:
            return
        
        @event.listens_for(self.engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            self._stats['connections_created'] += 1
            logger.debug("Database connection created")
        
        @event.listens_for(self.engine.sync_engine, "close")
        def on_close(dbapi_connection, connection_record):
            self._stats['connections_closed'] += 1
            logger.debug("Database connection closed")
        
        @event.listens_for(self.engine.sync_engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            self._stats['pool_hits'] += 1
        
        @event.listens_for(self.engine.sync_engine, "invalid")
        def on_invalid(dbapi_connection, connection_record, exception):
            logger.warning(f"Database connection invalidated: {exception}")
    
    def _start_health_check(self):
        """Start periodic health check task."""
        if self._health_check_task:
            self._health_check_task.cancel()
        
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _health_check_loop(self):
        """Periodic health check for database connections."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _perform_health_check(self):
        """Perform database health check."""
        if not self.engine:
            return
        
        try:
            start_time = time.time()
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            
            duration = time.time() - start_time
            self._stats['health_checks'] += 1
            
            logger.debug(f"Database health check passed in {duration:.3f}s")
            
        except Exception as e:
            self._stats['health_failures'] += 1
            logger.error(f"Database health check failed: {e}")
    
    def get_stats(self) -> dict:
        """Get connection pool statistics."""
        stats = self._stats.copy()
        
        if self.engine:
            pool = self.engine.pool
            stats.update({
                'pool_size': pool.size(),
                'checked_in': pool.checkedin(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
                'invalid': pool.invalid()
            })
        
        return stats
    
    async def execute_with_retry(self, operation: Callable, max_retries: int = 3, base_delay: float = 1.0) -> Any:
        """
        Execute database operation with retry logic for connection failures.
        
        Args:
            operation: Async callable that performs the database operation
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries (exponential backoff)
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await operation()
            except (DisconnectionError, OperationalError, ConnectionError) as e:
                last_exception = e
                
                if attempt == max_retries:
                    logger.error(f"Database operation failed after {max_retries} retries: {e}")
                    raise e
                
                # Calculate exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.2f}s: {e}")
                
                await asyncio.sleep(delay)
                
                # Attempt to refresh the connection pool on connection errors
                if self.engine:
                    try:
                        await self.engine.dispose()
                        # Engine will be recreated on next use
                    except Exception as dispose_err:
                        logger.debug(f"Error disposing engine during retry: {dispose_err}")
            
            except Exception as e:
                # Non-connection errors should not be retried
                logger.error(f"Non-retryable database error: {e}")
                raise e
        
        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
    
    async def close(self):
        """Close the connection pool and cleanup resources."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self.engine:
            await self.engine.dispose()
            logger.info("Connection pool closed")

# Global connection pool instance
connection_pool = OptimizedConnectionPool()

def get_optimized_engine(database_url: str) -> AsyncEngine:
    """Get or create optimized database engine."""
    if connection_pool.engine is None:
        return connection_pool.create_engine(database_url)
    return connection_pool.engine

def get_pool_stats() -> dict:
    """Get connection pool statistics."""
    return connection_pool.get_stats()

async def close_connection_pool():
    """Close the global connection pool."""
    await connection_pool.close()

def get_connection_pool() -> OptimizedConnectionPool:
    """Get the global connection pool instance."""
    return connection_pool
