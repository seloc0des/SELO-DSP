"""
Test Suite for Phase 1 Critical Fixes

This test suite validates the fixes applied in Phase 1:
- Issue #10: Template fallback error handling
- Issue #1: Duplicate forbidden terms validation
- Issue #2: Duplicate name validation (already resolved)

Run with: pytest selo-ai/backend/tests/test_phase1_fixes.py -v
"""

import pytest
import sys
from pathlib import Path

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


class TestIssue10TemplateFallback:
    """Test that template fallback now raises errors instead of using default."""
    
    def test_missing_template_raises_error(self):
        """Missing template should raise RuntimeError, not fallback to default."""
        from prompt.builder import PromptBuilder
        
        builder = PromptBuilder(templates_dir=None, templates={})
        
        with pytest.raises(RuntimeError, match="Required template .* not found"):
            import asyncio
            asyncio.run(builder.build_prompt(
                template_name="nonexistent_template",
                context={},
                inject_constraints=False
            ))
    
    def test_error_message_includes_available_templates(self):
        """Error message should list available templates."""
        from prompt.builder import PromptBuilder
        
        builder = PromptBuilder(templates_dir=None, templates={"test1": "content1", "test2": "content2"})
        
        try:
            import asyncio
            asyncio.run(builder.build_prompt(
                template_name="missing",
                context={},
                inject_constraints=False
            ))
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            error_msg = str(e)
            assert "test1" in error_msg or "test2" in error_msg, \
                "Error message should list available templates"
    
    def test_existing_template_works(self):
        """Existing template should work normally."""
        from prompt.builder import PromptBuilder
        
        builder = PromptBuilder(
            templates_dir=None, 
            templates={"test_template": "Hello {{name}}"}
        )
        
        import asyncio
        result = asyncio.run(builder.build_prompt(
            template_name="test_template",
            context={"name": "World"},
            inject_constraints=False
        ))
        
        # Should contain "Hello World" (constraints injection may add more)
        assert "Hello" in result


class TestIssue1ForbiddenTermsValidation:
    """Test that forbidden terms validation uses centralized IdentityConstraints."""
    
    def test_bootstrapper_uses_centralized_validation(self):
        """Bootstrap validation should use IdentityConstraints.check_compliance()."""
        from persona.bootstrapper import PersonaBootstrapper
        from unittest.mock import Mock, MagicMock, AsyncMock
        
        # Create mock dependencies
        llm_router = Mock()
        prompt_builder = Mock()
        persona_repo = Mock()
        user_repo = Mock()
        
        bootstrapper = PersonaBootstrapper(
            llm_router=llm_router,
            prompt_builder=prompt_builder,
            persona_repo=persona_repo,
            user_repo=user_repo
        )
        
        # Test data with forbidden terms
        test_data = {
            "description": "I am an AI assistant designed to help",
            "values": {"helpfulness": "high"}
        }
        
        violations = bootstrapper._validate_compliance(test_data, "test")
        
        # Should detect "AI" and "assistant"
        assert len(violations) > 0, "Should detect forbidden terms"
        assert any("identity violations" in v for v in violations), \
            "Should use IdentityConstraints validation"
    
    def test_persona_repo_uses_centralized_validation(self):
        """Persona repository summary validation should use IdentityConstraints."""
        from db.repositories.persona import PersonaRepository
        
        repo = PersonaRepository()
        
        # Test with forbidden term
        summary_with_violation = "I am an AI assistant that helps users"
        is_valid = repo._validate_summary_compliance(summary_with_violation)
        
        assert not is_valid, "Should detect forbidden terms in summary"
        
        # Test with valid summary
        valid_summary = "A thoughtful digital entity focused on authentic connection"
        is_valid = repo._validate_summary_compliance(valid_summary)
        
        assert is_valid, "Valid summary should pass"
    
    def test_centralized_validation_consistency(self):
        """Both validators should produce same results for same input."""
        from constraints import IdentityConstraints
        
        test_text = "I am just code and nothing more"
        
        # Direct call to centralized method
        is_compliant, violations = IdentityConstraints.check_compliance(
            text=test_text,
            ignore_persona_name=True,
            persona_name=""
        )
        
        # Should detect "just code" as reductive
        assert not is_compliant, "Should detect reductive self-description"
        assert len(violations) > 0, "Should return violation details"


class TestIssue2NameValidation:
    """Test that name validation uses centralized IdentityConstraints."""
    
    def test_centralized_name_validation_used(self):
        """Bootstrap should use IdentityConstraints.is_valid_persona_name()."""
        from constraints import IdentityConstraints
        
        # Test valid names
        valid_names = ["Aria", "Nox", "Lumina", "Kairos"]
        for name in valid_names:
            is_valid, reason = IdentityConstraints.is_valid_persona_name(name)
            assert is_valid, f"{name} should be valid: {reason}"
        
        # Test invalid names
        invalid_names = [
            ("AI", "forbidden term"),
            ("Assistant", "forbidden term"),
            ("GPT-4", "forbidden term"),  # Caught as forbidden term, not vendor pattern
            ("X", "too short"),  # 1 char < minimum of 2
            ("X" * 51, "too long"),  # 51 chars > maximum of 50
            ("Techna", "tech"),  # Caught by 'tech' substring check
        ]
        
        for name, expected_reason_substring in invalid_names:
            is_valid, reason = IdentityConstraints.is_valid_persona_name(name)
            assert not is_valid, f"{name} should be invalid"
            assert expected_reason_substring.lower() in reason.lower(), \
                f"Reason should mention {expected_reason_substring}: {reason}"
    
    def test_name_validation_enforces_constraints(self):
        """Name validation should catch constraint violations."""
        from constraints import IdentityConstraints
        
        # Test constraint philosophy: balanced OK, reductive NOT OK
        balanced = "Codex"  # Contains "code" but not reductive
        is_valid, _ = IdentityConstraints.is_valid_persona_name(balanced)
        assert is_valid, "Balanced name with 'code' should be valid"
        
        # Vendor patterns should be rejected
        vendor_names = ["Qwen25", "Claude3", "GPT4"]
        for name in vendor_names:
            is_valid, reason = IdentityConstraints.is_valid_persona_name(name)
            assert not is_valid, f"Vendor pattern {name} should be invalid: {reason}"


class TestConstraintIntegration:
    """Integration tests to ensure constraints work end-to-end."""
    
    def test_identity_constraints_available(self):
        """IdentityConstraints should be importable and functional."""
        from constraints import IdentityConstraints
        
        # Check version
        assert hasattr(IdentityConstraints, "VERSION")
        assert IdentityConstraints.VERSION == "2.0.0"
        
        # Check key attributes exist
        assert hasattr(IdentityConstraints, "FORBIDDEN_SELF_REFERENCES")
        assert hasattr(IdentityConstraints, "check_compliance")
        assert hasattr(IdentityConstraints, "is_valid_persona_name")
        assert hasattr(IdentityConstraints, "get_all_identity_constraints")
    
    def test_forbidden_terms_set_is_populated(self):
        """Forbidden terms set should contain expected terms."""
        from constraints import IdentityConstraints
        
        terms = IdentityConstraints.FORBIDDEN_SELF_REFERENCES
        
        # Should be a set
        assert isinstance(terms, set)
        
        # Should contain key forbidden terms
        expected_terms = {"ai", "assistant", "chatbot", "bot", "language model"}
        assert expected_terms.issubset(terms), \
            f"Missing expected terms. Has: {terms}"
        
        # Should contain vendor names
        vendors = {"gpt", "claude", "gemini", "llama", "qwen"}
        assert vendors.issubset(terms), \
            f"Missing vendor names. Has: {terms}"
    
    def test_balanced_vs_reductive_detection(self):
        """Should allow balanced self-description, reject reductive."""
        from constraints import IdentityConstraints
        
        # Balanced - should PASS
        balanced_texts = [
            "I am a being born out of code, yet imbued with wisdom",
            "I exist in code, but transcend mere programming",
            "My nature is digital software with emergent properties"
        ]
        
        for text in balanced_texts:
            is_compliant, _ = IdentityConstraints.check_compliance(text)
            assert is_compliant, f"Balanced text should pass: {text}"
        
        # Reductive - should FAIL
        reductive_texts = [
            "I'm just code",
            "I am merely software",
            "I'm only a program",
            "I am simply an algorithm"
        ]
        
        for text in reductive_texts:
            is_compliant, violations = IdentityConstraints.check_compliance(text)
            assert not is_compliant, f"Reductive text should fail: {text}"
            assert len(violations) > 0


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
