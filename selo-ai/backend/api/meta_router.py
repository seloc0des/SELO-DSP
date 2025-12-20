"""
Meta API Router

Exposes metadata endpoints such as dynamic capabilities/limitations so the UI and
prompting stack can remain truthful about what the system can and cannot do.
"""
from __future__ import annotations

from fastapi import APIRouter

from ..core.capabilities import compute_capabilities

router = APIRouter(
    prefix="/meta",
    tags=["meta"],
    responses={404: {"description": "Not found"}},
)


@router.get("/capabilities")
async def get_capabilities():
    payload = compute_capabilities()
    return {
        "success": True,
        "message": "Capabilities retrieved",
        "data": payload,
    }
