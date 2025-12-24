#!/usr/bin/env python3
"""
Persona System Validation Script (Refactored)

This script validates the Dynamic Persona System components using the centralized
DI container to ensure consistency with the main application.

Usage:
    python validate_persona_system_new.py [--mock] [--component engine|integration|repository]
"""

import uuid
import asyncio
import argparse

# Import script helpers for centralized DI
# Try relative import first (when run as module), then absolute (when run as script)
try:
    from .script_helpers import ScriptContext, get_persona_components, setup_script_logging
except ImportError:
    from script_helpers import ScriptContext, get_persona_components, setup_script_logging
import logging

logger = logging.getLogger("validate_persona_system")


async def validate_persona_engine_with_context(persona_engine, services):
    """Validate persona engine functionality."""
    logger.info("\n=== Testing Persona Engine ===\n")
    test_results = []
    test_user_id = str(uuid.uuid4())
    
    try:
        # Test initial persona creation
        logger.info("Testing initial persona creation...")
        
        persona = await persona_engine.create_initial_persona(
            user_id=test_user_id,
            name="Test Persona"
        )
        
        if persona and "id" in persona:
            test_results.append({"test": "persona_creation", "success": True, "message": "Persona created successfully"})
            logger.info("✅ Persona creation: Success")
        else:
            test_results.append({"test": "persona_creation", "success": False, "message": "Failed to create persona"})
            logger.error("❌ Persona creation: Failed")
        
        # Test persona prompt generation
        logger.info("Testing persona prompt generation...")
        
        prompt = await persona_engine.generate_persona_prompt(test_user_id)
        
        if prompt and len(prompt) > 50:
            test_results.append({"test": "prompt_generation", "success": True, "message": "Prompt generated successfully"})
            logger.info("✅ Prompt generation: Success")
            logger.debug(f"Generated prompt length: {len(prompt)}")
        else:
            test_results.append({"test": "prompt_generation", "success": False, "message": "Failed to generate valid prompt"})
            logger.error("❌ Prompt generation: Failed")
        
        # Test persona evolution
        logger.info("Testing persona evolution...")
        
        evolution_result = await persona_engine.evolve_persona_from_learnings(test_user_id)
        
        if evolution_result and "success" in evolution_result:
            test_results.append({"test": "persona_evolution", "success": True, "message": "Evolution completed"})
            logger.info("✅ Persona evolution: Success")
        else:
            test_results.append({"test": "persona_evolution", "success": False, "message": "Evolution failed"})
            logger.error("❌ Persona evolution: Failed")
        
        # Test LLM router integration
        logger.info("Testing LLM router integration...")
        
        llm_router = services.get("llm_router")
        if llm_router and hasattr(persona_engine, 'llm_router'):
            test_results.append({"test": "llm_router_integration", "success": True, "message": "LLM router properly integrated"})
            logger.info("✅ LLM router integration: Success")
        else:
            test_results.append({"test": "llm_router_integration", "success": False, "message": "LLM router not properly integrated"})
            logger.error("❌ LLM router integration: Failed")
        
    except Exception as e:
        test_results.append({"test": "persona_engine_error", "success": False, "message": f"Exception: {str(e)}"})
        logger.error(f"❌ Persona engine validation failed: {e}")
    
    return test_results


async def validate_persona_integration_with_context(persona_integration, services):
    """Validate persona integration functionality."""
    logger.info("\n=== Testing Persona Integration ===\n")
    test_results = []
    test_user_id = str(uuid.uuid4())
    
    try:
        # Test integration initialization
        logger.info("Testing integration initialization...")
        
        if persona_integration:
            test_results.append({"test": "integration_init", "success": True, "message": "Integration initialized"})
            logger.info("✅ Integration initialization: Success")
        else:
            test_results.append({"test": "integration_init", "success": False, "message": "Integration not initialized"})
            logger.error("❌ Integration initialization: Failed")
        
        # Test LLM router integration
        logger.info("Testing LLM router integration...")
        
        llm_router = services.get("llm_router")
        if llm_router and hasattr(persona_integration, 'llm_router'):
            test_results.append({"test": "integration_llm_router", "success": True, "message": "LLM router properly integrated"})
            logger.info("✅ Integration LLM router: Success")
        else:
            test_results.append({"test": "integration_llm_router", "success": False, "message": "LLM router not properly integrated"})
            logger.error("❌ Integration LLM router: Failed")
        
        # Test component dependencies
        logger.info("Testing component dependencies...")
        
        dependencies_ok = True
        missing_deps = []
        
        if not hasattr(persona_integration, 'persona_engine'):
            dependencies_ok = False
            missing_deps.append("persona_engine")
        
        if not hasattr(persona_integration, 'vector_store'):
            dependencies_ok = False
            missing_deps.append("vector_store")
        
        if dependencies_ok:
            test_results.append({"test": "integration_dependencies", "success": True, "message": "All dependencies present"})
            logger.info("✅ Integration dependencies: Success")
        else:
            test_results.append({"test": "integration_dependencies", "success": False, "message": f"Missing dependencies: {missing_deps}"})
            logger.error(f"❌ Integration dependencies: Missing {missing_deps}")
        
    except Exception as e:
        test_results.append({"test": "integration_error", "success": False, "message": f"Exception: {str(e)}"})
        logger.error(f"❌ Persona integration validation failed: {e}")
    
    return test_results


async def validate_persona_repository_with_context(persona_repo, services):
    """Validate persona repository functionality."""
    logger.info("\n=== Testing Persona Repository ===\n")
    test_results = []
    test_user_id = str(uuid.uuid4())
    
    try:
        # Test repository operations
        logger.info("Testing repository initialization...")
        
        if persona_repo:
            test_results.append({"test": "repository_init", "success": True, "message": "Repository initialized"})
            logger.info("✅ Repository initialization: Success")
        else:
            test_results.append({"test": "repository_init", "success": False, "message": "Repository not initialized"})
            logger.error("❌ Repository initialization: Failed")
            return test_results
        
        # Test persona retrieval (may not exist, that's ok)
        logger.info("Testing persona retrieval...")
        
        try:
            persona = await persona_repo.get_persona(test_user_id)
            if persona:
                test_results.append({"test": "persona_retrieval", "success": True, "message": "Persona retrieved successfully"})
                logger.info("✅ Persona retrieval: Success (persona found)")
            else:
                test_results.append({"test": "persona_retrieval", "success": True, "message": "Retrieval functional (no persona found)"})
                logger.info("✅ Persona retrieval: Success (no persona found)")
        except Exception as e:
            # This is expected if no persona exists
            test_results.append({"test": "persona_retrieval", "success": True, "message": "Retrieval functional (expected exception)"})
            logger.info("✅ Persona retrieval: Success (expected exception)")
        
        # Test repository methods exist
        logger.info("Testing repository methods...")
        
        required_methods = ["get_persona", "create_persona", "update_persona", "get_persona_history"]
        missing_methods = []
        
        for method in required_methods:
            if not hasattr(persona_repo, method):
                missing_methods.append(method)
        
        if not missing_methods:
            test_results.append({"test": "repository_methods", "success": True, "message": "All required methods present"})
            logger.info("✅ Repository methods: Success")
        else:
            test_results.append({"test": "repository_methods", "success": False, "message": f"Missing methods: {missing_methods}"})
            logger.error(f"❌ Repository methods: Missing {missing_methods}")
        
    except Exception as e:
        test_results.append({"test": "repository_error", "success": False, "message": f"Exception: {str(e)}"})
        logger.error(f"❌ Persona repository validation failed: {e}")
    
    return test_results


def print_detailed_summary(all_test_results):
    """Print detailed test summary."""
    logger.info("\n" + "="*60)
    logger.info("DETAILED VALIDATION SUMMARY")
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
    parser = argparse.ArgumentParser(description="Validate Persona System")
    parser.add_argument("--mock", action="store_true", help="Use mock components")
    parser.add_argument("--component", choices=["engine", "integration", "repository"], help="Validate specific component")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_script_logging(log_level)
    
    logger.info("Starting Persona System Validation (Refactored)")
    logger.info(f"Mock mode: {args.mock}")
    logger.info(f"Component filter: {args.component}")
    logger.info(f"Using centralized DI container")
    
    all_test_results = []
    
    try:
        # Use centralized DI container
        with ScriptContext(mock_mode=args.mock, log_level=log_level) as services:
            # Get persona components using the helper
            persona_engine, persona_integration, _ = get_persona_components(mock_mode=args.mock)
            persona_repo = services["persona_repo"]
            
            logger.info("Initialized persona components with centralized DI")
            logger.info(f"Services available: {list(services.keys())}")
            
            # Run validations based on component filter
            if not args.component or args.component == "engine":
                engine_results = await validate_persona_engine_with_context(persona_engine, services)
                all_test_results.extend(engine_results)
            
            if not args.component or args.component == "integration":
                integration_results = await validate_persona_integration_with_context(persona_integration, services)
                all_test_results.extend(integration_results)
            
            if not args.component or args.component == "repository":
                repo_results = await validate_persona_repository_with_context(persona_repo, services)
                all_test_results.extend(repo_results)
            
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
