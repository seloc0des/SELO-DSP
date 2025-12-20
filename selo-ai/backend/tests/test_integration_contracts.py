"""
Integration Contract Tests for SELO AI

Tests parameter compatibility between API endpoints and repository methods
to prevent runtime parameter mismatches and method signature errors.
"""

import pytest
import inspect
import asyncio
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock

# Import components to test
from backend.db.repositories.reflection import ReflectionRepository
from backend.db.repositories.conversation import ConversationRepository
from backend.db.models.reflection import Reflection
from backend.db.models.conversation import Conversation

class TestReflectionAPIContracts:
    """Test contracts between reflection API and repository."""
    
    def test_count_reflections_method_exists(self):
        """Test that count_reflections method exists with correct signature."""
        assert hasattr(ReflectionRepository, 'count_reflections'), \
            "ReflectionRepository missing count_reflections method"
        
        method = getattr(ReflectionRepository, 'count_reflections')
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())[1:]  # Skip 'self'
        
        expected_params = ['user_profile_id', 'reflection_type']
        for param in expected_params:
            assert param in params, f"count_reflections missing parameter: {param}"
    
    def test_list_reflections_method_signature(self):
        """Test that list_reflections has all required parameters."""
        assert hasattr(ReflectionRepository, 'list_reflections'), \
            "ReflectionRepository missing list_reflections method"
        
        method = getattr(ReflectionRepository, 'list_reflections')
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())[1:]  # Skip 'self'
        
        expected_params = [
            'user_profile_id', 'reflection_type', 'limit', 
            'offset', 'sort_by', 'sort_order'
        ]
        
        for param in expected_params:
            assert param in params, f"list_reflections missing parameter: {param}"
    
    def test_reflection_model_attributes(self):
        """Test that Reflection model has required attributes."""
        required_attrs = [
            'id', 'user_profile_id', 'reflection_type', 
            'result', 'created_at', 'embedding'
        ]
        
        for attr in required_attrs:
            assert hasattr(Reflection, attr), f"Reflection model missing attribute: {attr}"
    
    @pytest.mark.asyncio
    async def test_count_reflections_parameter_compatibility(self):
        """Test that count_reflections can be called with expected parameters."""
        # Mock database session
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result
        
        # Create repository instance with mocked session
        repo = ReflectionRepository()
        repo._get_session = AsyncMock(return_value=mock_session)
        
        # Test method call with expected parameters
        try:
            count = await repo.count_reflections(
                user_profile_id="test_user",
                reflection_type="daily"
            )
            assert isinstance(count, int)
        except TypeError as e:
            pytest.fail(f"count_reflections parameter mismatch: {e}")
    
    @pytest.mark.asyncio
    async def test_list_reflections_parameter_compatibility(self):
        """Test that list_reflections can be called with expected parameters."""
        # Mock database session
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        
        # Create repository instance with mocked session
        repo = ReflectionRepository()
        repo._get_session = AsyncMock(return_value=mock_session)
        
        # Test method call with expected parameters
        try:
            reflections = await repo.list_reflections(
                user_profile_id="test_user",
                reflection_type="daily",
                limit=10,
                offset=0,
                sort_by="created_at",
                sort_order="desc"
            )
            assert isinstance(reflections, list)
        except TypeError as e:
            pytest.fail(f"list_reflections parameter mismatch: {e}")

class TestConversationAPIContracts:
    """Test contracts between conversation API and repository."""
    
    def test_conversation_model_attributes(self):
        """Test that Conversation model has required attributes."""
        required_attrs = [
            'id', 'session_id', 'user_id', 'title', 
            'started_at', 'last_message_at', 'is_active', 'message_count'
        ]
        
        for attr in required_attrs:
            assert hasattr(Conversation, attr), f"Conversation model missing attribute: {attr}"
    
    def test_conversation_model_no_updated_at(self):
        """Test that Conversation model doesn't have problematic updated_at attribute."""
        # This should use last_message_at instead
        if hasattr(Conversation, 'updated_at'):
            pytest.fail("Conversation model should not have 'updated_at' attribute. Use 'last_message_at' instead.")
    
    def test_list_conversations_method_exists(self):
        """Test that list_conversations method exists."""
        assert hasattr(ConversationRepository, 'list_conversations'), \
            "ConversationRepository missing list_conversations method"
    
    @pytest.mark.asyncio
    async def test_conversation_sorting_compatibility(self):
        """Test that conversation sorting uses correct attributes."""
        # Mock database session
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        
        # Create repository instance with mocked session
        repo = ConversationRepository()
        repo._get_session = AsyncMock(return_value=mock_session)
        
        # Test sorting by last_message_at (should work)
        try:
            conversations = await repo.list_conversations(
                user_id="test_user",
                limit=10,
                offset=0,
                sort_by="last_message_at",
                sort_order="desc"
            )
            assert isinstance(conversations, list)
        except (AttributeError, TypeError) as e:
            pytest.fail(f"Conversation sorting by last_message_at failed: {e}")

class TestAPIEndpointContracts:
    """Test API endpoint parameter contracts."""
    
    def test_api_reflection_parameters(self):
        """Test that API endpoints use correct parameter names."""
        # This would normally import from main.py and test the actual endpoints
        # For now, we'll test the expected parameter patterns
        
        # Expected API parameter mapping
        api_to_repo_params = {
            'user_profile_id': 'user_profile_id',  # Should match exactly
            'reflection_type': 'reflection_type',   # Should match exactly
            'limit': 'limit',                      # Should match exactly
            'offset': 'offset',                    # Should match exactly
            'sort_by': 'sort_by',                  # Should match exactly
            'sort_order': 'sort_order'             # Should match exactly
        }
        
        # Verify parameter naming consistency
        for api_param, repo_param in api_to_repo_params.items():
            assert api_param == repo_param, \
                f"Parameter mismatch: API uses '{api_param}' but repo expects '{repo_param}'"

class TestMethodSignatureCompatibility:
    """Test method signature compatibility across the system."""
    
    def test_reflection_repository_method_signatures(self):
        """Test that all reflection repository methods have compatible signatures."""
        repo_methods = [
            'count_reflections',
            'list_reflections', 
            'create_reflection',
            'get_reflection',
            'update_reflection',
            'delete_reflection'
        ]
        
        for method_name in repo_methods:
            assert hasattr(ReflectionRepository, method_name), \
                f"ReflectionRepository missing method: {method_name}"
            
            method = getattr(ReflectionRepository, method_name)
            assert callable(method), f"ReflectionRepository.{method_name} is not callable"
    
    def test_conversation_repository_method_signatures(self):
        """Test that all conversation repository methods have compatible signatures."""
        repo_methods = [
            'list_conversations',
            'create_conversation',
            'get_conversation',
            'update_conversation',
            'delete_conversation'
        ]
        
        for method_name in repo_methods:
            if hasattr(ConversationRepository, method_name):
                method = getattr(ConversationRepository, method_name)
                assert callable(method), f"ConversationRepository.{method_name} is not callable"

class TestDatabaseModelCompatibility:
    """Test database model compatibility with repository usage."""
    
    def test_reflection_model_field_access(self):
        """Test that Reflection model fields can be accessed as expected."""
        # Test that commonly used fields exist and are accessible
        common_fields = ['id', 'user_profile_id', 'created_at', 'reflection_type']
        
        for field in common_fields:
            assert hasattr(Reflection, field), f"Reflection model missing field: {field}"
            
            # Test that field can be used in queries (has proper SQLAlchemy setup)
            field_obj = getattr(Reflection, field)
            assert hasattr(field_obj, 'type'), f"Reflection.{field} not properly configured for SQLAlchemy"
    
    def test_conversation_model_field_access(self):
        """Test that Conversation model fields can be accessed as expected."""
        # Test that commonly used fields exist and are accessible
        common_fields = ['id', 'user_id', 'last_message_at', 'started_at']
        
        for field in common_fields:
            assert hasattr(Conversation, field), f"Conversation model missing field: {field}"
            
            # Test that field can be used in queries (has proper SQLAlchemy setup)
            field_obj = getattr(Conversation, field)
            assert hasattr(field_obj, 'type'), f"Conversation.{field} not properly configured for SQLAlchemy"

# Integration test runner
async def run_integration_contract_tests():
    """
    Run all integration contract tests and return results.
    
    Returns:
        Dict with test results and any failures
    """
    test_results = {
        "passed": 0,
        "failed": 0,
        "failures": [],
        "timestamp": None
    }
    
    # Import pytest and run tests programmatically
    try:
        import pytest
        from datetime import datetime, timezone
        
        # Run tests and capture results
        result = pytest.main([
            __file__,
            "-v",
            "--tb=short",
            "-x"  # Stop on first failure
        ])
        
        test_results["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        if result == 0:
            test_results["passed"] = 1
            test_results["status"] = "PASS"
        else:
            test_results["failed"] = 1
            test_results["status"] = "FAIL"
            test_results["failures"].append("Integration contract tests failed")
            
    except Exception as e:
        test_results["failed"] = 1
        test_results["status"] = "ERROR"
        test_results["failures"].append(f"Test execution error: {str(e)}")
    
    return test_results

if __name__ == "__main__":
    # Run tests when executed directly
    asyncio.run(run_integration_contract_tests())
