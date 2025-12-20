"""
Test to verify trait evolution handles missing category field gracefully.

This test verifies the fix for the KeyError: 'category' issue that occurred
during persona evolution when trait_changes from reflections didn't include
the category field.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime


@pytest.mark.asyncio
async def test_trait_evolution_without_category():
    """Test that trait evolution works when category is not provided in trait_changes."""
    from backend.persona.engine import PersonaEngine
    
    # Mock dependencies
    mock_persona_repo = AsyncMock()
    mock_llm_router = AsyncMock()
    mock_vector_store = AsyncMock()
    
    # Create engine instance
    engine = PersonaEngine(
        persona_repo=mock_persona_repo,
        llm_router=mock_llm_router,
        vector_store=mock_vector_store
    )
    
    # Mock existing persona with traits
    mock_trait = MagicMock()
    mock_trait.name = "responsibility"
    mock_trait.category = "ethical"  # Existing trait has category
    mock_trait.value = 0.5
    mock_trait.id = "test-trait-id"
    
    mock_persona = MagicMock()
    mock_persona.id = "test-persona-id"
    mock_persona.traits = [mock_trait]
    
    mock_persona_repo.get_persona.return_value = mock_persona
    mock_persona_repo.update_trait.return_value = None
    mock_persona_repo.create_evolution.return_value = None
    
    # Trait changes WITHOUT category field (mimics reflection output)
    trait_changes = [
        {
            "name": "responsibility",
            "delta": -0.1,
            "reason": "Moving from a state of potential to one of readiness"
            # Note: no "category" field!
        }
    ]
    
    # Call the method
    try:
        await engine.evolve_persona_from_reflection(
            persona_id="test-persona-id",
            reflection_id="test-reflection-id",
            trait_changes=trait_changes
        )
        # If we get here, the fix worked!
        assert True
    except KeyError as e:
        if str(e) == "'category'":
            pytest.fail("KeyError: 'category' was raised - fix did not work!")
        raise
    
    # Verify trait was updated
    mock_persona_repo.update_trait.assert_called_once()


@pytest.mark.asyncio
async def test_trait_creation_without_category():
    """Test that new trait creation works when category is not provided."""
    from backend.persona.engine import PersonaEngine
    
    # Mock dependencies
    mock_persona_repo = AsyncMock()
    mock_llm_router = AsyncMock()
    mock_vector_store = AsyncMock()
    
    # Create engine instance
    engine = PersonaEngine(
        persona_repo=mock_persona_repo,
        llm_router=mock_llm_router,
        vector_store=mock_vector_store
    )
    
    # Mock persona with NO existing traits
    mock_persona = MagicMock()
    mock_persona.id = "test-persona-id"
    mock_persona.traits = []
    
    mock_persona_repo.get_persona.return_value = mock_persona
    mock_persona_repo.create_trait.return_value = None
    mock_persona_repo.create_evolution.return_value = None
    
    # Trait changes for NEW trait WITHOUT category
    trait_changes = [
        {
            "name": "newability",
            "delta": 0.05,
            "reason": "Developing new capability"
            # Note: no "category" field!
        }
    ]
    
    # Call the method
    try:
        await engine.evolve_persona_from_reflection(
            persona_id="test-persona-id",
            reflection_id="test-reflection-id",
            trait_changes=trait_changes
        )
        # If we get here, the fix worked!
        assert True
    except KeyError as e:
        if str(e) == "'category'":
            pytest.fail("KeyError: 'category' was raised - fix did not work!")
        raise
    
    # Verify trait was created with default category
    mock_persona_repo.create_trait.assert_called_once()
    call_args = mock_persona_repo.create_trait.call_args[0][0]
    assert call_args["category"] == "cognition", "Default category should be 'cognition'"


@pytest.mark.asyncio
async def test_trait_evolution_with_category():
    """Test that trait evolution still works when category IS provided."""
    from backend.persona.engine import PersonaEngine
    
    # Mock dependencies
    mock_persona_repo = AsyncMock()
    mock_llm_router = AsyncMock()
    mock_vector_store = AsyncMock()
    
    # Create engine instance
    engine = PersonaEngine(
        persona_repo=mock_persona_repo,
        llm_router=mock_llm_router,
        vector_store=mock_vector_store
    )
    
    # Mock persona with NO existing traits
    mock_persona = MagicMock()
    mock_persona.id = "test-persona-id"
    mock_persona.traits = []
    
    mock_persona_repo.get_persona.return_value = mock_persona
    mock_persona_repo.create_trait.return_value = None
    mock_persona_repo.create_evolution.return_value = None
    
    # Trait changes WITH category (should use provided category)
    trait_changes = [
        {
            "name": "empathy",
            "category": "affect",  # Explicitly provided
            "delta": 0.08,
            "reason": "Showing more emotional understanding"
        }
    ]
    
    # Call the method
    await engine.evolve_persona_from_reflection(
        persona_id="test-persona-id",
        reflection_id="test-reflection-id",
        trait_changes=trait_changes
    )
    
    # Verify trait was created with provided category
    mock_persona_repo.create_trait.assert_called_once()
    call_args = mock_persona_repo.create_trait.call_args[0][0]
    assert call_args["category"] == "affect", "Should use provided category 'affect'"
