"""Autobiographical episode API endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query

from .dependencies import get_episode_service

router = APIRouter(prefix="/episodes", tags=["episodes"])


@router.get("", response_model=Dict[str, Any])
async def list_episodes(
    persona_id: str = Query(..., description="Persona identifier"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of episodes to return"),
    episode_service = Depends(get_episode_service),
):
    """Return recent autobiographical episodes for the given persona."""
    episodes = await episode_service.list_recent_episodes(persona_id=persona_id, limit=limit)
    return {"success": True, "data": {"episodes": episodes, "count": len(episodes)}}


@router.get("/{episode_id}", response_model=Dict[str, Any])
async def get_episode(
    episode_id: str,
    episode_service = Depends(get_episode_service),
):
    """Fetch a single autobiographical episode by identifier."""
    episode = await episode_service.get_episode(episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    return {"success": True, "data": {"episode": episode}}
