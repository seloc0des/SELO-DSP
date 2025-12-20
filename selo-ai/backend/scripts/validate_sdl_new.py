#!/usr/bin/env python3
"""
SDL Validation Script (Refactored)

This script validates the SDL (Self-Development Learning) module using the centralized
DI container to ensure consistency with the main application.

Usage:
    python validate_sdl_new.py [--mock] [--component engine|repository|mapper|integration]
"""

import asyncio
import json
import logging
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Import script helpers for centralized DI
# Try relative import first (when run as module), then absolute (when run as script)
try:
    from .script_helpers import ScriptContext, get_sdl_components, setup_script_logging
except ImportError:
    from script_helpers import ScriptContext, get_sdl_components, setup_script_logging

logger = logging.getLogger("sdl.validation")


async def validate_sdl_engine_with_context(sdl_engine, services):
    """Validate SDL engine functionality."""
    logger.info("\n=== Testing SDL Engine ===\n")
    test_results = []
    
    try:
        # Test engine initialization
        logger.info("Testing SDL engine initialization...")
        
        if sdl_engine:
            test_results.append({"test": "engine_init", "success": True, "message": "SDL engine initialized"})
            logger.info("✅ SDL engine initialization: Success")
        else:
            test_results.append({"test": "engine_init", "success": False, "message": "SDL engine not initialized"})
            logger.error("❌ SDL engine initialization: Failed")
            return test_results
        
        # Test LLM router integration
        logger.info("Testing LLM router integration...")
        
        llm_router = services.get("llm_router")
        if llm_router and hasattr(sdl_engine, 'llm_router'):
            test_results.append({"test": "engine_llm_router", "success": True, "message": "LLM router properly integrated"})
            logger.info("✅ SDL engine LLM router: Success")
        else:
            test_results.append({"test": "engine_llm_router", "success": False, "message": "LLM router not properly integrated"})
            logger.error("❌ SDL engine LLM router: Failed")
        
        # Test component dependencies
        logger.info("Testing component dependencies...")
        
        dependencies_ok = True
        missing_deps = []
        
        required_attrs = ["vector_store", "learning_repo", "reflection_repo"]
        for attr in required_attrs:
            if not hasattr(sdl_engine, attr):
                dependencies_ok = False
                missing_deps.append(attr)
        
        if dependencies_ok:
            test_results.append({"test": "engine_dependencies", "success": True, "message": "All dependencies present"})
            logger.info("✅ SDL engine dependencies: Success")
        else:
            test_results.append({"test": "engine_dependencies", "success": False, "message": f"Missing dependencies: {missing_deps}"})
            logger.error(f"❌ SDL engine dependencies: Missing {missing_deps}")
        
        # Test learning extraction via conversation processing
        logger.info("Testing learning extraction via conversation processing...")
        
        test_messages = [
            {"role": "user", "content": "I'm really interested in machine learning and neural networks", "user_id": "test-user-123"},
            {"role": "assistant", "content": "That's great! Neural networks are fascinating. What specific aspects interest you most?"}
        ]
        test_conversation_id = "test-conv-123"
        
        try:
            learning_result = await sdl_engine.process_conversation(test_conversation_id, test_messages)
            if isinstance(learning_result, list):
                test_results.append({"test": "learning_extraction", "success": True, "message": f"Learning extraction functional - extracted {len(learning_result)} learnings"})
                logger.info(f"✅ Learning extraction: Success - {len(learning_result)} learnings")
            else:
                test_results.append({"test": "learning_extraction", "success": False, "message": "Learning extraction returned invalid format"})
                logger.error("❌ Learning extraction: Invalid format")
        except Exception as e:
            test_results.append({"test": "learning_extraction", "success": False, "message": f"Learning extraction error: {str(e)}"})
            logger.error(f"❌ Learning extraction: Error - {e}")
        
    except Exception as e:
        test_results.append({"test": "engine_error", "success": False, "message": f"Exception: {str(e)}"})
        logger.error(f"❌ SDL engine validation failed: {e}")
    
    return test_results


async def validate_learning_repository_with_context(learning_repo, services):
    """Validate learning repository functionality."""
    logger.info("\n=== Testing Learning Repository ===\n")
    test_results = []
    
    try:
        # Test repository initialization
        logger.info("Testing learning repository initialization...")
        
        if learning_repo:
            test_results.append({"test": "repo_init", "success": True, "message": "Learning repository initialized"})
            logger.info("✅ Learning repository initialization: Success")
        else:
            test_results.append({"test": "repo_init", "success": False, "message": "Learning repository not initialized"})
            logger.error("❌ Learning repository initialization: Failed")
            return test_results
        
        # Test repository methods exist
        logger.info("Testing repository methods...")
        
        required_methods = ["create_learning", "get_learnings_for_user", "update_learning", "delete_learning"]
        missing_methods = []
        
        for method in required_methods:
            if not hasattr(learning_repo, method):
                missing_methods.append(method)
        
        if not missing_methods:
            test_results.append({"test": "repo_methods", "success": True, "message": "All required methods present"})
            logger.info("✅ Learning repository methods: Success")
        else:
            test_results.append({"test": "repo_methods", "success": False, "message": f"Missing methods: {missing_methods}"})
            logger.error(f"❌ Learning repository methods: Missing {missing_methods}")
        
        # Test learning retrieval (may be empty, that's ok)
        logger.info("Testing learning retrieval...")
        
        try:
            learnings = await learning_repo.get_learnings_for_user("test-user-123", limit=5)
            test_results.append({"test": "repo_retrieval", "success": True, "message": f"Retrieved {len(learnings) if learnings else 0} learnings"})
            logger.info("✅ Learning retrieval: Success")
        except Exception as e:
            test_results.append({"test": "repo_retrieval", "success": False, "message": f"Retrieval error: {str(e)}"})
            logger.error(f"❌ Learning retrieval: Error - {e}")
        
    except Exception as e:
        test_results.append({"test": "repo_error", "success": False, "message": f"Exception: {str(e)}"})
        logger.error(f"❌ Learning repository validation failed: {e}")
    
    return test_results


async def validate_concept_mapper_with_context(services):
    """Validate concept mapper functionality."""
    logger.info("\n=== Testing Concept Mapper ===\n")
    test_results = []
    
    try:
        # Import and initialize concept mapper
        from sdl.concept_mapper import ConceptMapper
        from sdl.repository import LearningRepository
        
        learning_repo = LearningRepository() if not services.get("mock_mode") else services.get("learning_repo")
        llm_router = services["llm_router"]
        
        concept_mapper = ConceptMapper(llm_router, learning_repo)
        
        # Test concept mapper initialization
        logger.info("Testing concept mapper initialization...")
        
        if concept_mapper:
            test_results.append({"test": "mapper_init", "success": True, "message": "Concept mapper initialized"})
            logger.info("✅ Concept mapper initialization: Success")
        else:
            test_results.append({"test": "mapper_init", "success": False, "message": "Concept mapper not initialized"})
            logger.error("❌ Concept mapper initialization: Failed")
            return test_results
        
        # Test LLM router integration
        logger.info("Testing LLM router integration...")
        
        if hasattr(concept_mapper, 'llm_router'):
            test_results.append({"test": "mapper_llm_router", "success": True, "message": "LLM router properly integrated"})
            logger.info("✅ Concept mapper LLM router: Success")
        else:
            test_results.append({"test": "mapper_llm_router", "success": False, "message": "LLM router not properly integrated"})
            logger.error("❌ Concept mapper LLM router: Failed")
        
        # Test concept processing (mock)
        logger.info("Testing concept processing...")
        
        # Create mock learning for testing
        from sdl.learning_models import Learning
        mock_learning = Learning(
            id="test-learning-123",
            user_id="test-user-123",
            content="The user learned about machine learning algorithms and neural networks.",
            domain="technology",
            confidence=0.9,
            created_at=datetime.now(timezone.utc)
        )
        
        try:
            concepts = await concept_mapper.process_learning_concepts(mock_learning)
            if concepts:
                test_results.append({"test": "concept_processing", "success": True, "message": f"Processed {len(concepts)} concepts"})
                logger.info("✅ Concept processing: Success")
            else:
                test_results.append({"test": "concept_processing", "success": True, "message": "Concept processing functional (no concepts)"})
                logger.info("✅ Concept processing: Success (no concepts)")
        except Exception as e:
            test_results.append({"test": "concept_processing", "success": False, "message": f"Processing error: {str(e)}"})
            logger.error(f"❌ Concept processing: Error - {e}")
        
    except Exception as e:
        test_results.append({"test": "mapper_error", "success": False, "message": f"Exception: {str(e)}"})
        logger.error(f"❌ Concept mapper validation failed: {e}")
    
    return test_results


async def validate_sdl_integration_with_context(services):
    """Validate SDL integration functionality."""
    logger.info("\n=== Testing SDL Integration ===\n")
    test_results = []
    
    try:
        # Import and initialize SDL integration
        from sdl.integration import SDLIntegration
        from sdl.repository import LearningRepository
        from events.triggers import EventTriggerSystem
        
        # Get components
        sdl_engine, _ = get_sdl_components(mock_mode=services.get("mock_mode", False))
        learning_repo = LearningRepository() if not services.get("mock_mode") else services.get("learning_repo")
        event_system = EventTriggerSystem() if not services.get("mock_mode") else services.get("event_system")
        
        sdl_integration = SDLIntegration(
            sdl_engine=sdl_engine,
            learning_repo=learning_repo,
            event_trigger_system=event_system
        )
        
        # Test integration initialization
        logger.info("Testing SDL integration initialization...")
        
        if sdl_integration:
            test_results.append({"test": "integration_init", "success": True, "message": "SDL integration initialized"})
            logger.info("✅ SDL integration initialization: Success")
        else:
            test_results.append({"test": "integration_init", "success": False, "message": "SDL integration not initialized"})
            logger.error("❌ SDL integration initialization: Failed")
            return test_results
        
        # Test component dependencies
        logger.info("Testing component dependencies...")
        
        dependencies_ok = True
        missing_deps = []
        
        required_attrs = ["sdl_engine", "learning_repo", "event_trigger_system"]
        for attr in required_attrs:
            if not hasattr(sdl_integration, attr):
                dependencies_ok = False
                missing_deps.append(attr)
        
        if dependencies_ok:
            test_results.append({"test": "integration_dependencies", "success": True, "message": "All dependencies present"})
            logger.info("✅ SDL integration dependencies: Success")
        else:
            test_results.append({"test": "integration_dependencies", "success": False, "message": f"Missing dependencies: {missing_deps}"})
            logger.error(f"❌ SDL integration dependencies: Missing {missing_deps}")
        
    except Exception as e:
        test_results.append({"test": "integration_error", "success": False, "message": f"Exception: {str(e)}"})
        logger.error(f"❌ SDL integration validation failed: {e}")
    
    return test_results


def print_detailed_summary(all_test_results):
    """Print detailed test summary."""
    logger.info("\n" + "="*60)
    logger.info("DETAILED SDL VALIDATION SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for r in all_test_results if r["success"])
    total = len(all_test_results)
    
    logger.info(f"Overall Result: {passed}/{total} tests passed")
    logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
    
    # Group by component
    components = {}
    for result in all_test_results:
        component = result["test"].split("_")[0] if "_" in result["test"] else "general"
        if component not in components:
            components[component] = []
        components[component].append(result)
    
    for component, results in components.items():
        component_passed = sum(1 for r in results if r["success"])
        component_total = len(results)
        logger.info(f"\n{component.upper()} Component: {component_passed}/{component_total}")
        
        for result in results:
            status = "✅" if result["success"] else "❌"
            logger.info(f"  {status} {result['test']}: {result['message']}")
    
    # Failed tests details
    failed_tests = [r for r in all_test_results if not r["success"]]
    if failed_tests:
        logger.info(f"\nFAILED TESTS ({len(failed_tests)}):")
        for result in failed_tests:
            logger.error(f"  ❌ {result['test']}: {result['message']}")
    
    logger.info("="*60)


async def main():
    """Main validation function using centralized DI container."""
    parser = argparse.ArgumentParser(description="Validate SDL System")
    parser.add_argument("--mock", action="store_true", help="Use mock components")
    parser.add_argument("--component", choices=["engine", "repository", "mapper", "integration"], help="Validate specific component")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_script_logging(log_level)
    
    logger.info("Starting SDL System Validation (Refactored)")
    logger.info(f"Mock mode: {args.mock}")
    logger.info(f"Component filter: {args.component}")
    logger.info(f"Using centralized DI container")
    
    all_test_results = []
    
    try:
        # Use centralized DI container
        with ScriptContext(mock_mode=args.mock, log_level=log_level) as services:
            # Get SDL components using the helper
            sdl_engine, _ = get_sdl_components(mock_mode=args.mock)
            
            logger.info("Initialized SDL components with centralized DI")
            logger.info(f"Services available: {list(services.keys())}")
            
            # Add mock mode flag to services for component access
            services["mock_mode"] = args.mock
            
            # Run validations based on component filter
            if not args.component or args.component == "engine":
                engine_results = await validate_sdl_engine_with_context(sdl_engine, services)
                all_test_results.extend(engine_results)
            
            if not args.component or args.component == "repository":
                from sdl.repository import LearningRepository
                learning_repo = LearningRepository() if not args.mock else services.get("learning_repo")
                repo_results = await validate_learning_repository_with_context(learning_repo, services)
                all_test_results.extend(repo_results)
            
            if not args.component or args.component == "mapper":
                mapper_results = await validate_concept_mapper_with_context(services)
                all_test_results.extend(mapper_results)
            
            if not args.component or args.component == "integration":
                integration_results = await validate_sdl_integration_with_context(services)
                all_test_results.extend(integration_results)
            
            # Print detailed summary
            print_detailed_summary(all_test_results)
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}", exc_info=True)
        return 1
    
    # Return exit code based on results
    failed_tests = [r for r in all_test_results if not r["success"]]
    if failed_tests:
        logger.error(f"Validation completed with {len(failed_tests)} failures")
        return 1
    else:
        logger.info("All validations passed!")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
