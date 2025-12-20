"""
Centralized security helpers and FastAPI dependencies for system/admin authentication.
"""
from typing import Optional
import os

from fastapi import Depends, Header, HTTPException, Request


async def get_provided_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
    legacy_api_key: Optional[str] = Header(None, alias="api-key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Optional[str]:
    """Extract API key from standard and legacy headers.
    Falls back to raw headers to cover non-annotated access paths.
    """
    return (
        x_api_key
        or legacy_api_key
        or authorization
        or request.headers.get("X-API-KEY")
        or request.headers.get("api-key")
        or request.headers.get("Authorization")
    )


async def require_system_key(provided_key: Optional[str] = Depends(get_provided_api_key)) -> bool:
    """FastAPI dependency that enforces the presence of the correct system API key.
    Raises HTTP 403 on failure.
    """
    expected_key = os.environ.get("SELO_SYSTEM_API_KEY")
    if not expected_key or provided_key != expected_key:
        raise HTTPException(status_code=403, detail="Unauthorized access to system endpoint")
    return True


async def is_system_or_admin(provided_key: Optional[str] = Depends(get_provided_api_key)) -> bool:
    """FastAPI dependency that returns True if the caller provides a valid system key."""
    expected_key = os.environ.get("SELO_SYSTEM_API_KEY")
    if not expected_key:
        return False
    return provided_key == expected_key
