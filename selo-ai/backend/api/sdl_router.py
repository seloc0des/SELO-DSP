"""
SDL API Router for Self-Directed Learning endpoints.

This module provides REST API endpoints for SDL functionality including
learning management, concept exploration, and knowledge search.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from .dependencies import get_sdl_integration
from ..sdl.integration import SDLIntegration

logger = logging.getLogger("selo.api.sdl")

router = APIRouter(prefix="/sdl", tags=["SDL"])


# Request/Response Models
class LearningRequest(BaseModel):
    content: str = Field(..., description="Learning content")
    domain: str = Field(..., description="Learning domain")
    confidence: float = Field(0.7, ge=0.0, le=1.0, description="Confidence score")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance score")
    concepts: List[str] = Field(default_factory=list, description="Related concepts")


class ConceptRequest(BaseModel):
    name: str = Field(..., description="Concept name")
    description: Optional[str] = Field(None, description="Concept description")
    category: Optional[str] = Field(None, description="Concept category")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance score")


class ConnectionRequest(BaseModel):
    source_concept: str = Field(..., description="Source concept name")
    target_concept: str = Field(..., description="Target concept name")
    relation_type: str = Field(..., description="Relationship type")
    strength: float = Field(0.5, ge=0.0, le=1.0, description="Connection strength")
    bidirectional: bool = Field(False, description="Is bidirectional connection")


class LearningResponse(BaseModel):
    id: str
    content: str
    domain: str
    confidence: float
    importance: float
    concepts: List[str]
    created_at: str
    user_id: str


class ConceptResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    category: Optional[str]
    importance: float
    familiarity: float
    learning_count: int
    created_at: str


class KnowledgeStatsResponse(BaseModel):
    total_learnings: int
    total_concepts: int
    domain_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    recent_reflections_count: int
    timestamp: str


# Learning Endpoints
@router.post("/learnings", response_model=LearningResponse)
async def create_learning(
    learning_req: LearningRequest,
    user_id: str = Query(..., description="User ID"),
    sdl_integration: SDLIntegration = Depends(get_sdl_integration)
):
    """Create a new learning entry."""
    try:
        learning_data = {
            "user_id": user_id,
            "content": learning_req.content,
            "domain": learning_req.domain,
            "confidence": learning_req.confidence,
            "importance": learning_req.importance,
            "concepts": learning_req.concepts,
            "source_type": "manual",
            "source_id": "api"
        }
        
        learning = await sdl_integration.learning_repo.create_learning(learning_data)
        
        # Update vector store
        vector_id = await sdl_integration.sdl_engine._update_vector_store(learning)
        if vector_id:
            await sdl_integration.learning_repo.update_learning(
                learning.id, {"vector_id": vector_id}
            )
        
        return LearningResponse(**learning.to_dict())
        
    except Exception as e:
        logger.error(f"Error creating learning: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create learning: {str(e)}")


@router.get("/learnings", response_model=List[LearningResponse])
async def get_learnings(
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    sdl_integration: SDLIntegration = Depends(get_sdl_integration)
):
    """Get learnings for a user."""
    try:
        learnings = await sdl_integration.learning_repo.get_learnings_for_user(
            user_id=user_id,
            limit=limit,
            offset=offset,
            domain=domain,
            source_type=source_type
        )
        
        return [LearningResponse(**learning.to_dict()) for learning in learnings]
        
    except Exception as e:
        logger.error(f"Error getting learnings: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get learnings: {str(e)}")


@router.get("/learnings/{learning_id}", response_model=LearningResponse)
async def get_learning(
    learning_id: str,
    sdl_integration: SDLIntegration = Depends(get_sdl_integration)
):
    """Get a specific learning by ID."""
    try:
        learning = await sdl_integration.learning_repo.get_learning(learning_id)
        if not learning:
            raise HTTPException(status_code=404, detail="Learning not found")
            
        return LearningResponse(**learning.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting learning {learning_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get learning: {str(e)}")


@router.delete("/learnings/{learning_id}")
async def delete_learning(
    learning_id: str,
    sdl_integration: SDLIntegration = Depends(get_sdl_integration)
):
    """Delete a learning by ID."""
    try:
        success = await sdl_integration.learning_repo.delete_learning(learning_id)
        if not success:
            raise HTTPException(status_code=404, detail="Learning not found")
            
        return {"message": "Learning deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting learning {learning_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete learning: {str(e)}")


# Concept Endpoints
@router.post("/concepts", response_model=ConceptResponse)
async def create_concept(
    concept_req: ConceptRequest,
    user_id: str = Query(..., description="User ID"),
    sdl_integration: SDLIntegration = Depends(get_sdl_integration)
):
    """Create a new concept."""
    try:
        concept_data = {
            "user_id": user_id,
            "name": concept_req.name,
            "description": concept_req.description,
            "category": concept_req.category,
            "importance": concept_req.importance
        }
        
        concept = await sdl_integration.learning_repo.create_concept(concept_data)
        return ConceptResponse(**concept.to_dict())
        
    except Exception as e:
        logger.error(f"Error creating concept: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create concept: {str(e)}")


@router.get("/concepts", response_model=List[ConceptResponse])
async def get_concepts(
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    category: Optional[str] = Query(None, description="Filter by category"),
    sdl_integration: SDLIntegration = Depends(get_sdl_integration)
):
    """Get concepts for a user."""
    try:
        concepts = await sdl_integration.learning_repo.get_concepts_for_user(
            user_id=user_id,
            limit=limit,
            offset=offset,
            category=category
        )
        
        return [ConceptResponse(**concept.to_dict()) for concept in concepts]
        
    except Exception as e:
        logger.error(f"Error getting concepts: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get concepts: {str(e)}")


@router.get("/concepts/{concept_id}/related")
async def get_related_concepts(
    concept_id: str,
    user_id: str = Query(..., description="User ID"),
    max_depth: int = Query(2, ge=1, le=5, description="Maximum traversal depth"),
    sdl_integration: SDLIntegration = Depends(get_sdl_integration)
):
    """Get concepts related to a specific concept."""
    try:
        # Get the concept first
        concept = await sdl_integration.learning_repo.get_concept(concept_id)
        if not concept:
            raise HTTPException(status_code=404, detail="Concept not found")
            
        result = await sdl_integration.sdl_engine.get_related_concepts(
            user_id=user_id,
            concept_name=concept.name,
            max_depth=max_depth
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting related concepts: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get related concepts: {str(e)}")


# Knowledge Search and Analysis
@router.get("/search")
async def search_knowledge(
    query: str = Query(..., description="Search query"),
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    sdl_integration: SDLIntegration = Depends(get_sdl_integration)
):
    """Search the knowledge base for relevant learnings."""
    try:
        results = await sdl_integration.sdl_engine.search_knowledge(
            user_id=user_id,
            query=query,
            limit=limit
        )
        
        return {"query": query, "results": results}
        
    except Exception as e:
        logger.error(f"Error searching knowledge: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to search knowledge: {str(e)}")


@router.post("/consolidate")
async def consolidate_learnings(
    user_id: str = Query(..., description="User ID"),
    domain: Optional[str] = Query(None, description="Domain to consolidate"),
    sdl_integration: SDLIntegration = Depends(get_sdl_integration)
):
    """Consolidate learnings into higher-level insights."""
    try:
        result = await sdl_integration.sdl_engine.consolidate_learnings(
            user_id=user_id,
            domain=domain
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error consolidating learnings: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to consolidate learnings: {str(e)}")


@router.post("/meta-learning")
async def generate_meta_learning(
    user_id: str = Query(..., description="User ID"),
    sdl_integration: SDLIntegration = Depends(get_sdl_integration)
):
    """Generate meta-learning insights about the learning process."""
    try:
        result = await sdl_integration.sdl_engine.generate_meta_learning(user_id)
        return result
        
    except Exception as e:
        logger.error(f"Error generating meta-learning: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate meta-learning: {str(e)}")


@router.get("/stats", response_model=KnowledgeStatsResponse)
async def get_knowledge_stats(
    user_id: str = Query(..., description="User ID"),
    sdl_integration: SDLIntegration = Depends(get_sdl_integration)
):
    """Get statistics about the user's knowledge base."""
    try:
        result = await sdl_integration.get_user_knowledge_stats(user_id)
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to get stats"))
            
        return KnowledgeStatsResponse(**result["stats"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting knowledge stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get knowledge stats: {str(e)}")


# Processing Endpoints
@router.post("/process/reflection/{reflection_id}")
async def process_reflection(
    reflection_id: str,
    sdl_integration: SDLIntegration = Depends(get_sdl_integration)
):
    """Process a reflection to extract learnings."""
    try:
        result = await sdl_integration.process_new_reflection(reflection_id)
        return result
        
    except Exception as e:
        logger.error(f"Error processing reflection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process reflection: {str(e)}")


@router.post("/process/conversation/{conversation_id}")
async def process_conversation(
    conversation_id: str,
    messages: List[Dict[str, Any]],
    sdl_integration: SDLIntegration = Depends(get_sdl_integration)
):
    """Process a conversation to extract learnings."""
    try:
        result = await sdl_integration.process_conversation(conversation_id, messages)
        return result
        
    except Exception as e:
        logger.error(f"Error processing conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process conversation: {str(e)}")


@router.post("/reorganize")
async def reorganize_concepts(
    user_id: str = Query(..., description="User ID"),
    sdl_integration: SDLIntegration = Depends(get_sdl_integration)
):
    """Reorganize concepts to improve knowledge graph structure."""
    try:
        result = await sdl_integration.reorganize_user_concepts(user_id)
        return result
        
    except Exception as e:
        logger.error(f"Error reorganizing concepts: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reorganize concepts: {str(e)}")
