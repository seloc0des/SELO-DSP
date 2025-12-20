"""Agent state transparency API router."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from .dependencies import (
    get_affective_state_manager,
    get_goal_manager,
    get_episode_service,
    get_persona_integration,
)
from ..db.repositories.user import UserRepository

router = APIRouter(prefix="/agent-state", tags=["agent-state"])


async def _resolve_persona(persona_id: Optional[str]) -> Dict[str, Any]:
    persona_integration = await get_persona_integration()

    if persona_id:
        persona = await persona_integration.persona_repo.get_persona(
            persona_id,
            include_traits=False,
        )
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")
        user_id = getattr(persona, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=500, detail="Persona missing user binding")
        return {"persona_id": persona_id, "user_id": user_id}

    user_repo = UserRepository()
    user = await user_repo.get_or_create_default_user()
    if not user:
        raise HTTPException(status_code=500, detail="Installation user unavailable")
    persona = await persona_integration.persona_repo.get_or_create_default_persona(
        user_id=user.id,
        include_traits=False,
    )
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    return {"persona_id": persona.id, "user_id": user.id}


@router.get("/affective", response_model=Dict[str, Any])
async def get_affective_state(
    persona_id: Optional[str] = Query(None, description="Persona identifier"),
    affective_manager = Depends(get_affective_state_manager),
):
    resolved = await _resolve_persona(persona_id)
    persona = resolved.get("persona_id")
    user_id = resolved.get("user_id")

    state = await affective_manager.ensure_state_available(
        persona_id=persona,
        user_id=user_id,
    )
    if not isinstance(state, dict):
        raise HTTPException(status_code=404, detail="Affective state unavailable")

    return {"success": True, "data": state}


@router.get("/goals", response_model=Dict[str, Any])
async def list_goals(
    persona_id: Optional[str] = Query(None, description="Persona identifier"),
    goal_manager = Depends(get_goal_manager),
):
    resolved = await _resolve_persona(persona_id)
    persona = resolved.get("persona_id")

    goals = await goal_manager.list_active_goals(persona)
    return {"success": True, "data": {"goals": goals, "count": len(goals)}}


@router.get("/plan-steps", response_model=Dict[str, Any])
async def list_plan_steps(
    persona_id: Optional[str] = Query(None, description="Persona identifier"),
    goal_manager = Depends(get_goal_manager),
):
    resolved = await _resolve_persona(persona_id)
    persona = resolved.get("persona_id")

    steps = await goal_manager.list_pending_steps(persona)
    return {"success": True, "data": {"plan_steps": steps, "count": len(steps)}}


@router.get("/meta-directives", response_model=Dict[str, Any])
async def list_meta_directives(
    persona_id: Optional[str] = Query(None, description="Persona identifier"),
    goal_manager = Depends(get_goal_manager),
):
    resolved = await _resolve_persona(persona_id)
    persona = resolved.get("persona_id")

    directives = await goal_manager.list_meta_directives(
        persona,
        statuses=["pending", "in_progress"],
        limit=20,
    )
    return {
        "success": True,
        "data": {"meta_directives": directives, "count": len(directives)},
    }


@router.get("/episodes", response_model=Dict[str, Any])
async def list_episodes(
    persona_id: Optional[str] = Query(None, description="Persona identifier"),
    limit: int = Query(5, ge=1, le=25, description="Maximum number of episodes"),
    episode_service = Depends(get_episode_service),
):
    resolved = await _resolve_persona(persona_id)
    persona = resolved.get("persona_id")

    episodes = await episode_service.list_recent_episodes(persona_id=persona, limit=limit)
    return {"success": True, "data": {"episodes": episodes, "count": len(episodes)}}
