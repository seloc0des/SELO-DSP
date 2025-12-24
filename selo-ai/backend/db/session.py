"""
DATABASE SESSION MANAGEMENT
===========================

SELO uses async SQLAlchemy with three session access patterns for different use cases.

SESSION USAGE GUIDE
-------------------

**1. FastAPI Routes (Automatic Lifecycle Management):**

Use `get_db()` as a FastAPI dependency for automatic session lifecycle management.
The session is created, committed (or rolled back on error), and closed automatically.

```python
from backend.db.session import get_db
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    return user
    # Session automatically committed and closed by FastAPI
```

**2. Repositories (PREFERRED Pattern - STANDARDIZED):**

✅ **ALL SELO repositories now use this pattern consistently.**
Use `get_session()` as an async context manager in repository classes.
Supports both standalone usage and working with existing sessions.

```python
from backend.db.session import get_session

class UserRepository:
    async def get_user(self, user_id: int, session: AsyncSession = None):
        async with get_session(session) as db:
            result = await db.execute(select(User).where(User.id == user_id))
            return result.scalar_one_or_none()
        # Auto-committed and closed (if new session created)

# Standalone usage:
repo = UserRepository()
user = await repo.get_user(123)  # Creates new session

# Within existing transaction:
async with get_db() as db:
    user = await repo.get_user(123, session=db)  # Reuses session
```

**3. Standalone Scripts (AVOID - Manual Management Required):**

Use `get_db_session()` only when neither FastAPI nor repository pattern applies.
YOU MUST MANUALLY CLOSE THE SESSION to avoid connection leaks.

```python
from backend.db.session import get_db_session

async def migration_script():
    session = await get_db_session()
    try:
        result = await session.execute(select(User))
        users = result.scalars().all()
        # Process users...
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise
    finally:
        await session.close()  # ⚠️ YOU MUST CLOSE!
```

BEST PRACTICES
--------------

✅ **DO:**
- Use `get_db()` in FastAPI endpoint parameters
- Use `get_session()` in all repository methods
- Pass existing sessions to nested repository calls for transactional consistency
- Handle exceptions and let SQLAlchemy rollback automatically

❌ **DON'T:**
- Use `get_db_session()` unless absolutely necessary (e.g., standalone scripts)
- Forget to close sessions created with `get_db_session()`
- Create new sessions inside loops (reuse existing session)
- Mix sync and async SQLAlchemy patterns

CONNECTION POOLING
------------------

The database engine uses connection pooling for efficiency:
- Default: NullPool (safe for async, prevents cross-event-loop issues)
- Configurable via SQL_NULLPOOL, SQL_POOL_SIZE, SQL_MAX_OVERFLOW
- pool_pre_ping=True ensures connections are alive before use

SECURITY NOTE
-------------

Do NOT hardcode credentials. All DB credentials must come from environment variables
or .env file (never committed to version control).

Best practices:
- Use strong, unique passwords for database users
- Restrict database access by network/firewall rules
- Use DATABASE_URL environment variable for connection string
- Never commit .env files containing credentials

ENVIRONMENT VARIABLES
---------------------

- DATABASE_URL: PostgreSQL connection string (postgresql://user:pass@host/dbname)
- SQL_NULLPOOL: Use NullPool (true) or QueuePool (false) - default: true
- SQL_POOL_SIZE: Connection pool size - default: 5
- SQL_MAX_OVERFLOW: Max overflow connections - default: 10
- SQL_POOL_TIMEOUT: Pool acquisition timeout (seconds) - default: 30
- SQL_POOL_RECYCLE: Recycle connections after (seconds) - default: 1800
- SQL_ECHO: Log all SQL statements (true/false) - default: false

REFERENCES
----------

- SQLAlchemy Async: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
- FastAPI Dependencies: https://fastapi.tiangolo.com/tutorial/dependencies/
- Connection Pooling: https://docs.sqlalchemy.org/en/20/core/pooling.html
"""

import os
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Get database URL from environment variable (required)
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Please configure it in backend/.env or the environment before starting SELO DSP."
    )

# Convert standard PostgreSQL URL to async format if needed
if DATABASE_URL.startswith("postgresql://") and not DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Connection pool settings
USE_NULLPOOL = os.getenv("SQL_NULLPOOL", "true").lower() == "true"
POOL_SIZE = int(os.getenv("SQL_POOL_SIZE", 5))
MAX_OVERFLOW = int(os.getenv("SQL_MAX_OVERFLOW", 10))
POOL_TIMEOUT = int(os.getenv("SQL_POOL_TIMEOUT", 30))  # seconds
POOL_RECYCLE = int(os.getenv("SQL_POOL_RECYCLE", 1800))  # seconds

# Create async engine with pooling (default to NullPool to avoid cross-event-loop reuse)
engine_kwargs = {
    "echo": os.getenv("SQL_ECHO", "false").lower() == "true",
    "future": True,
    "pool_pre_ping": True,
}
if USE_NULLPOOL:
    engine_kwargs["poolclass"] = NullPool
else:
    engine_kwargs.update({
        "pool_size": POOL_SIZE,
        "max_overflow": MAX_OVERFLOW,
        "pool_timeout": POOL_TIMEOUT,
        "pool_recycle": POOL_RECYCLE,
    })

engine = create_async_engine(
    DATABASE_URL,
    **engine_kwargs,
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI to get a database session.
    Yields an async SQLAlchemy session and ensures it's closed after use.
    
    Usage:
        @app.get("/items/")
        async def read_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db_session() -> AsyncSession:
    """
    Manual async DB session for scripts/utilities. You MUST close the session when done.
    """
    return AsyncSessionLocal()


@asynccontextmanager
async def get_session(session: Optional[AsyncSession] = None):
    """
    Convenience async context manager used by repositories.
    - If an existing session is provided, yields it without managing lifecycle.
    - If none is provided, creates a session and handles commit/rollback/close.
    """
    if session is not None:
        # Caller manages lifecycle
        yield session
        return
    async with AsyncSessionLocal() as s:
        try:
            yield s
            await s.commit()
        except Exception:
            await s.rollback()
            raise
        finally:
            await s.close()


async def init_db() -> None:
    """
    Backward-compatible DB init noop used by some legacy scripts.
    Performs a simple connectivity check against the configured engine.
    """
    try:
        async with engine.begin() as conn:  # No-op transaction to verify connectivity
            await conn.run_sync(lambda *_: None)
    except Exception:
        # Re-raise so callers can handle/log appropriately
        raise
