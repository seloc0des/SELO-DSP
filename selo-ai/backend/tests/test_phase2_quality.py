"""
Test Suite for Phase 2: Quality & Optimization

This test suite validates the quality improvements applied in Phase 2:
- Issue #17: Session documentation (documentation only, no tests needed)
- Issue #5: Template randomization
- Issue #6: Word count singleton configuration
- Issue #13: Token budget refactoring

Run with: pytest selo-ai/backend/tests/test_phase2_quality.py -v
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Import backend modules with absolute imports
try:
    from backend.reflection.processor import ReflectionProcessor
    from backend.llm.token_budget import TokenBudgetManager
    from backend.llm.router import LLMRouter
    from backend.db import session
except ImportError:
    # Fallback for different test contexts
    from reflection.processor import ReflectionProcessor
    from llm.token_budget import TokenBudgetManager
    from llm.router import LLMRouter
    from db import session


class TestIssue5TemplateRandomization:
    """Test that sensory sanitization templates are randomized to prevent LLM parroting."""
    
    def test_replacement_sentences_vary(self):
        """Replacement sentences should vary randomly, not be deterministic."""
        
        # Create minimal processor instance (disable async embedding processor for tests)
        processor = ReflectionProcessor(enable_deferred_embeddings=False)
        
        # Get the _replacement_sentence function
        # (it's defined inside _sanitize_sensory_leaks, so we need to test via that method)
        
        # Test data with sensory terms
        text_with_sensory = "I see a bright room with soft lighting. I smell fresh coffee."
        
        # Call sanitization multiple times
        results = []
        for _ in range(10):
            sanitized = processor._sanitize_sensory_leaks(
                text_with_sensory,
                {"user_message": "hello"}
            )
            results.append(sanitized)
        
        # Results should vary (not all identical)
        unique_results = set(results)
        assert len(unique_results) > 1, "Templates should randomize, not always return same result"
    
    def test_no_exact_template_matching(self):
        """Output should not contain exact old template phrases."""
        
        processor = ReflectionProcessor(enable_deferred_embeddings=False)
        
        # Old templates that should no longer appear
        old_templates = [
            "I imagine a loose scene while admitting I have no direct sensory evidence",
            "I picture only an abstract outline and remind myself there is no direct sensory proof",
            "I acknowledge these impressions are speculative and lean on explicit words instead of ambience",
        ]
        
        text_with_sensory = "I see the room clearly with bright lights everywhere."
        
        # Generate 20 sanitized versions
        for _ in range(20):
            sanitized = processor._sanitize_sensory_leaks(
                text_with_sensory,
                {"user_message": "test"}
            )
            
            # None should contain old templates
            for old_template in old_templates:
                assert old_template.lower() not in sanitized.lower(), \
                    f"Old template found in output: {old_template}"
    
    def test_new_templates_present(self):
        """Verify new template structure exists in code."""
        import inspect
        
        # Check that the replacement function uses new templates
        source = inspect.getsource(ReflectionProcessor)
        
        # New template phrases should be in the code
        new_phrases = [
            "focus on the conversation",
            "ground myself",
            "explicit content",
            "work with the information provided",
        ]
        
        # At least 3 of the 4 should be present in source code
        found = sum(1 for phrase in new_phrases if phrase in source)
        assert found >= 3, f"Expected at least 3 new template phrases in code, found {found}"


class TestIssue6WordCountSingleton:
    """Test that word count configuration uses singleton pattern."""
    
    def test_singleton_caching(self):
        """get_word_count_config should cache configuration."""
        
        # Reset class-level cache
        ReflectionProcessor._word_count_config = None
        
        # First call should set cache
        config1 = ReflectionProcessor.get_word_count_config()
        assert config1 is not None
        assert 'min' in config1
        assert 'max' in config1
        assert ReflectionProcessor._word_count_config is not None
        
        # Second call should return same object (cached)
        config2 = ReflectionProcessor.get_word_count_config()
        assert config2 is config1, "Should return cached config object"
    
    def test_config_structure(self):
        """Config should have expected structure."""
        
        ReflectionProcessor._word_count_config = None
        config = ReflectionProcessor.get_word_count_config()
        
        assert isinstance(config, dict)
        assert 'min' in config
        assert 'max' in config
        assert isinstance(config['min'], int)
        assert isinstance(config['max'], int)
        assert config['min'] > 0
        assert config['max'] > config['min']
    
    def test_fallback_when_config_fails(self):
        """Should fallback gracefully if config loading fails."""
        
        ReflectionProcessor._word_count_config = None
        
        # Mock config to raise exception (use correct import path)
        with patch('backend.config.reflection_config.get_reflection_config', side_effect=Exception("Config error")):
            config = ReflectionProcessor.get_word_count_config()
            
            # Should still return valid defaults
            assert config == {'min': 170, 'max': 500}
    
    def test_all_locations_use_singleton(self):
        """Verify all code locations use get_word_count_config()."""
        import inspect
        
        # Read processor source
        source = inspect.getsource(ReflectionProcessor)
        
        # Should have multiple calls to get_word_count_config()
        config_calls = source.count('get_word_count_config()')
        assert config_calls >= 6, f"Expected at least 6 calls to get_word_count_config(), found {config_calls}"
        
        # Should NOT have old patterns like getattr(self, "word_count_min", 170)
        # (except in __init__ where it sets instance variables)
        old_patterns = [
            'getattr(self, "word_count_min"',
            'getattr(self, "word_count_max"',
        ]
        
        # Count occurrences (should only be in __init__)
        for pattern in old_patterns:
            count = source.count(pattern)
            # Allowed in __init__ where we set instance vars, but not elsewhere
            assert count <= 2, f"Found {count} uses of old pattern '{pattern}', should use singleton"


class TestIssue13TokenBudget:
    """Test token budget manager refactoring."""
    
    def test_token_budget_manager_exists(self):
        """TokenBudgetManager class should exist and be importable."""
        
        assert TokenBudgetManager is not None
        assert hasattr(TokenBudgetManager, 'calculate_budget')
        assert hasattr(TokenBudgetManager, 'estimate_prompt_tokens')
        assert hasattr(TokenBudgetManager, 'get_task_config')
    
    def test_prompt_token_estimation(self):
        """Token estimation should use ~4 chars per token heuristic."""
        
        # Empty string
        assert TokenBudgetManager.estimate_prompt_tokens("") == 0
        
        # 400 chars = ~100 tokens
        prompt_400_chars = "x" * 400
        tokens = TokenBudgetManager.estimate_prompt_tokens(prompt_400_chars)
        assert 90 <= tokens <= 110, f"Expected ~100 tokens for 400 chars, got {tokens}"
        
        # 4000 chars = ~1000 tokens
        prompt_4000_chars = "x" * 4000
        tokens = TokenBudgetManager.estimate_prompt_tokens(prompt_4000_chars)
        assert 900 <= tokens <= 1100, f"Expected ~1000 tokens for 4000 chars, got {tokens}"
    
    def test_budget_calculation_chat(self):
        """Budget calculation for chat task."""
        
        budget = TokenBudgetManager.calculate_budget(
            task_type="chat",
            prompt="Hello, how are you?"
        )
        
        assert 'max_tokens' in budget
        assert 'temperature' in budget
        assert 'num_ctx' in budget
        assert 'clamped' in budget
        assert budget['max_tokens'] > 0
        assert budget['temperature'] >= 0
    
    def test_budget_calculation_reflection(self):
        """Budget calculation for reflection task."""
        
        budget = TokenBudgetManager.calculate_budget(
            task_type="reflection",
            prompt="x" * 1000  # 250 token prompt
        )
        
        assert budget['max_tokens'] >= 512, "Reflection should have min 512 tokens"
        assert budget['temperature'] < 0.5, "Reflection should have lower temperature"
    
    def test_budget_calculation_analytical(self):
        """Budget calculation for analytical task."""
        
        budget = TokenBudgetManager.calculate_budget(
            task_type="analytical",
            prompt="Generate structured JSON"
        )
        
        # Analytical has base 1024, but actual max_tokens may be same as base or clamped
        assert budget['max_tokens'] > 0, "Analytical should have positive max_tokens"
        assert budget['min_tokens'] == 1024, "Analytical minimum should be 1024"
        assert budget['temperature'] <= 0.2, "Analytical should have very low temperature"
    
    def test_adaptive_clamping(self):
        """Budget should clamp when prompt is very large."""
        
        # Very large prompt (3000 tokens ~= 12000 chars)
        large_prompt = "x" * 12000
        
        budget = TokenBudgetManager.calculate_budget(
            task_type="chat",
            prompt=large_prompt,
            config={"num_ctx": 4096}
        )
        
        # Should clamp because prompt + completion > context
        assert budget['clamped'] is True, "Should clamp for large prompt"
        assert budget['max_tokens'] < 2000, "Should clamp below normal max"
    
    def test_different_task_types(self):
        """All task types should have valid configs."""
        
        task_types = ["chat", "reflection", "analytical", "persona_prompt", "sdl", "persona_evolve"]
        
        for task_type in task_types:
            budget = TokenBudgetManager.calculate_budget(
                task_type=task_type,
                prompt="test"
            )
            
            assert budget['max_tokens'] > 0, f"{task_type} should have positive max_tokens"
            assert 0 <= budget['temperature'] <= 1.0, f"{task_type} temperature should be 0-1"
    
    def test_router_uses_token_budget(self):
        """LLM router should use TokenBudgetManager."""
        import inspect
        
        source = inspect.getsource(LLMRouter)
        
        # Should import TokenBudgetManager
        assert 'TokenBudgetManager' in source, "Router should import TokenBudgetManager"
        assert 'calculate_budget' in source, "Router should call calculate_budget"
        
        # Should NOT have old inline calculation patterns
        old_patterns = [
            'def _env_int(',
            'def _env_float(',
            'estimated_prompt_tokens = prompt_length // 4'
        ]
        
        for pattern in old_patterns:
            assert pattern not in source, f"Router should not have old pattern: {pattern}"


class TestIssue17SessionDocs:
    """Test session documentation completeness."""
    
    def test_session_module_has_docstring(self):
        """session.py should have comprehensive module docstring."""
        
        assert session.__doc__ is not None, "session.py should have module docstring"
        
        docstring = session.__doc__
        
        # Should document all three patterns
        assert 'FastAPI' in docstring
        assert 'get_db' in docstring
        assert 'get_session' in docstring
        assert 'get_db_session' in docstring
    
    def test_docstring_has_usage_examples(self):
        """Docstring should include usage examples."""
        
        docstring = session.__doc__
        
        # Should have code examples
        assert '```python' in docstring or 'async def' in docstring
        assert 'Depends(get_db)' in docstring
        assert 'async with get_session' in docstring
    
    def test_docstring_has_best_practices(self):
        """Docstring should include best practices guidance."""
        
        docstring = session.__doc__
        
        # Should mention best practices
        assert 'BEST PRACTICES' in docstring or 'best practice' in docstring.lower()
        assert 'AVOID' in docstring or 'avoid' in docstring.lower()
        
        # Should warn about manual session closing
        assert 'MUST CLOSE' in docstring or 'must close' in docstring.lower()


class TestIntegrationPhase2:
    """Integration tests for Phase 2 changes."""
    
    def test_reflection_processor_initializes(self):
        """ReflectionProcessor should initialize with new singleton pattern."""
        
        # Reset cache
        ReflectionProcessor._word_count_config = None
        
        # Should initialize without errors (disable async for tests)
        processor = ReflectionProcessor(enable_deferred_embeddings=False)
        
        assert hasattr(processor, 'word_count_min')
        assert hasattr(processor, 'word_count_max')
        assert processor.word_count_min == 170
        assert processor.word_count_max == 500
    
    def test_token_budget_integration_with_router(self):
        """TokenBudgetManager should integrate cleanly with LLMRouter."""
        
        # Simulate what router does
        prompt = "This is a test prompt for chat"
        budget = TokenBudgetManager.calculate_budget(
            task_type="chat",
            prompt=prompt
        )
        
        # Should return valid budget
        assert budget['max_tokens'] > 0
        assert budget['temperature'] > 0
        
        # Should be usable directly as kwargs
        kwargs = {}
        kwargs.setdefault("max_tokens", budget["max_tokens"])
        kwargs.setdefault("temperature", budget["temperature"])
        
        assert kwargs['max_tokens'] == budget['max_tokens']
        assert kwargs['temperature'] == budget['temperature']


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
