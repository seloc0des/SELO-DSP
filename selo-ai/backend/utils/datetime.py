"""UTC datetime utilities used across the backend."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def ensure_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Coerce a datetime to UTC, attaching tzinfo when missing."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def isoformat_utc(value: Optional[datetime]) -> Optional[str]:
    """Return an ISO 8601 string in UTC with a trailing 'Z'."""
    coerced = ensure_utc(value)
    if coerced is None:
        return None
    return coerced.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def isoformat_utc_now() -> str:
    """Convenience helper to return the current UTC time as an ISO string."""
    # isoformat_utc never returns None when provided a datetime
    return isoformat_utc(utc_now())  # type: ignore[return-value]


def add_seconds(value: Optional[datetime], seconds: float) -> datetime:
    """Add seconds to a datetime, working in UTC."""
    base = ensure_utc(value) if value is not None else utc_now()
    return base + timedelta(seconds=seconds)


def utc_iso(dt: Optional[datetime] = None) -> str:
    """
    Convert datetime to ISO string with 'Z' suffix, defaulting to now.
    
    Args:
        dt: Optional datetime to convert. If None, uses current UTC time.
        
    Returns:
        ISO 8601 string with 'Z' suffix (e.g., "2025-11-06T19:51:00.000Z")
    """
    return isoformat_utc(dt if dt else utc_now())
