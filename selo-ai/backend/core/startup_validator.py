"""
Startup Validation System for SELO AI

Validates all critical system components at startup to prevent runtime failures.
Catches parameter mismatches, missing methods, and configuration issues early.
"""

import logging
import asyncio
import inspect
from typing import Dict, Any
from datetime import datetime, timezone
import importlib

logger = logging.getLogger("selo.startup.validator")

class ValidationError(Exception):
    """Custom exception for startup validation failures."""

class StartupValidator:
    """
    Comprehensive startup validation for SELO AI system.
    
    Validates:
    - Repository method signatures and compatibility
    - API endpoint parameter alignment
    - Database model attributes
    - LLM model availability
    - Environment configuration
    """
    
    def __init__(self):
        self.validation_results = []
        self.critical_failures = []
        self.warnings = []
        
    async def validate_all(self) -> Dict[str, Any]:
        """
        Run all validation checks and return comprehensive results.
        
        Returns:
            Dict with validation results, failures, and warnings
        """
        logger.info("🚀 Starting SELO AI startup validation...")
        
        validation_tasks = [
            self._validate_repository_contracts(),
            self._validate_api_compatibility(),
            self._validate_database_models(),
            self._validate_llm_configuration(),
            self._validate_environment_setup(),
            self._validate_dependency_integrity()
        ]
        
        # Run all validations concurrently
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.critical_failures.append({
                    "validator": validation_tasks[i].__name__,
                    "error": str(result),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            else:
                self.validation_results.extend(result.get("results", []))
                self.warnings.extend(result.get("warnings", []))
        
        # Generate summary
        summary = {
            "status": "PASS" if not self.critical_failures else "FAIL",
            "total_checks": len(self.validation_results),
            "passed": len([r for r in self.validation_results if r["status"] == "PASS"]),
            "failed": len([r for r in self.validation_results if r["status"] == "FAIL"]),
            "warnings": len(self.warnings),
            "critical_failures": self.critical_failures,
            "validation_results": self.validation_results,
            "warnings_list": self.warnings,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if self.critical_failures:
            logger.error(f"❌ Startup validation FAILED: {len(self.critical_failures)} critical failures")
            for failure in self.critical_failures:
                logger.error(f"   • {failure['validator']}: {failure['error']}")
        else:
            logger.info(f"✅ Startup validation PASSED: {summary['passed']}/{summary['total_checks']} checks")
            
        if self.warnings:
            logger.warning(f"⚠️  {len(self.warnings)} warnings found")
            
        return summary
    
    async def _validate_repository_contracts(self) -> Dict[str, Any]:
        """Validate repository method signatures and required methods."""
        results = []
        warnings = []
        
        try:
            # Import repositories
            from ..db.repositories.reflection import ReflectionRepository
            from ..db.repositories.conversation import ConversationRepository
            
            # Check ReflectionRepository required methods
            reflection_methods = [
                ("count_reflections", ["user_profile_id", "reflection_type"]),
                ("list_reflections", ["user_profile_id", "reflection_type", "limit", "offset", "sort_by", "sort_order"]),
                ("create_reflection", ["reflection_data"]),
                ("get_reflection", ["reflection_id"]),
                ("update_reflection", ["reflection_id", "update_data"]),
                ("delete_reflection", ["reflection_id"])
            ]
            
            for method_name, expected_params in reflection_methods:
                if hasattr(ReflectionRepository, method_name):
                    method = getattr(ReflectionRepository, method_name)
                    sig = inspect.signature(method)
                    actual_params = list(sig.parameters.keys())[1:]  # Skip 'self'
                    
                    missing_params = [p for p in expected_params if p not in actual_params]
                    if missing_params:
                        results.append({
                            "check": f"ReflectionRepository.{method_name} parameters",
                            "status": "FAIL",
                            "details": f"Missing parameters: {missing_params}",
                            "expected": expected_params,
                            "actual": actual_params
                        })
                    else:
                        results.append({
                            "check": f"ReflectionRepository.{method_name} parameters",
                            "status": "PASS",
                            "details": "All required parameters present"
                        })
                else:
                    results.append({
                        "check": f"ReflectionRepository.{method_name} existence",
                        "status": "FAIL",
                        "details": f"Method {method_name} not found"
                    })
            
            # Check ConversationRepository for attribute usage
            conv_methods = ["list_conversations", "_list_conversations_impl"]
            for method_name in conv_methods:
                if hasattr(ConversationRepository, method_name):
                    results.append({
                        "check": f"ConversationRepository.{method_name} existence",
                        "status": "PASS",
                        "details": "Method exists"
                    })
                else:
                    results.append({
                        "check": f"ConversationRepository.{method_name} existence",
                        "status": "FAIL",
                        "details": f"Method {method_name} not found"
                    })
                    
        except Exception as e:
            results.append({
                "check": "Repository import and validation",
                "status": "FAIL",
                "details": f"Failed to validate repositories: {str(e)}"
            })
            
        return {"results": results, "warnings": warnings}
    
    async def _validate_api_compatibility(self) -> Dict[str, Any]:
        """Validate API endpoint parameter compatibility with repositories."""
        results = []
        warnings = []
        
        try:
            # Check main.py API endpoints
            import sys
            import os
            
            # Add backend to path if not already there
            backend_path = os.path.dirname(os.path.dirname(__file__))
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)
            
            # Import main module
            main_module = importlib.import_module("main")
            
            # Check if list_reflections_paginated exists and uses correct parameters
            if hasattr(main_module, 'list_reflections_paginated'):
                results.append({
                    "check": "API endpoint list_reflections_paginated exists",
                    "status": "PASS",
                    "details": "Endpoint found in main.py"
                })
            else:
                results.append({
                    "check": "API endpoint list_reflections_paginated exists",
                    "status": "FAIL",
                    "details": "Endpoint not found in main.py"
                })
                
        except Exception as e:
            warnings.append({
                "check": "API compatibility validation",
                "details": f"Could not fully validate API compatibility: {str(e)}"
            })
            
        return {"results": results, "warnings": warnings}
    
    async def _validate_database_models(self) -> Dict[str, Any]:
        """Validate database model attributes and relationships."""
        results = []
        warnings = []
        
        try:
            from ..db.models.conversation import Conversation
            from ..db.models.reflection import Reflection
            
            # Check Conversation model attributes
            required_conversation_attrs = [
                "id", "session_id", "user_id", "title", "started_at", 
                "last_message_at", "is_active", "message_count"
            ]
            
            for attr in required_conversation_attrs:
                if hasattr(Conversation, attr):
                    results.append({
                        "check": f"Conversation.{attr} attribute",
                        "status": "PASS",
                        "details": "Attribute exists"
                    })
                else:
                    results.append({
                        "check": f"Conversation.{attr} attribute",
                        "status": "FAIL",
                        "details": f"Missing attribute: {attr}"
                    })
            
            # Check for problematic attributes
            if hasattr(Conversation, 'updated_at'):
                warnings.append({
                    "check": "Conversation.updated_at attribute",
                    "details": "updated_at attribute exists but should use last_message_at instead"
                })
            
            # Check Reflection model attributes
            required_reflection_attrs = [
                "id", "user_profile_id", "reflection_type", "result", 
                "created_at", "embedding"
            ]
            
            for attr in required_reflection_attrs:
                if hasattr(Reflection, attr):
                    results.append({
                        "check": f"Reflection.{attr} attribute",
                        "status": "PASS",
                        "details": "Attribute exists"
                    })
                else:
                    results.append({
                        "check": f"Reflection.{attr} attribute",
                        "status": "FAIL",
                        "details": f"Missing attribute: {attr}"
                    })
                    
        except Exception as e:
            results.append({
                "check": "Database model validation",
                "status": "FAIL",
                "details": f"Failed to validate models: {str(e)}"
            })
            
        return {"results": results, "warnings": warnings}
    
    async def _validate_llm_configuration(self) -> Dict[str, Any]:
        """Validate LLM configuration and model availability."""
        results = []
        warnings = []
        
        try:
            from ..llm.controller import LLMController
            
            # Check for asyncio import issues
            controller_file = inspect.getfile(LLMController)
            with open(controller_file, 'r') as f:
                content = f.read()
                
            # Check for problematic local asyncio imports
            lines = content.split('\n')
            local_asyncio_imports = []
            for i, line in enumerate(lines):
                if 'import asyncio' in line and 'def ' in lines[max(0, i-10):i]:
                    local_asyncio_imports.append(i + 1)
                    
            if local_asyncio_imports:
                results.append({
                    "check": "LLMController asyncio imports",
                    "status": "FAIL",
                    "details": f"Local asyncio imports found at lines: {local_asyncio_imports}"
                })
            else:
                results.append({
                    "check": "LLMController asyncio imports",
                    "status": "PASS",
                    "details": "No problematic local asyncio imports found"
                })
                
        except Exception as e:
            warnings.append({
                "check": "LLM configuration validation",
                "details": f"Could not fully validate LLM config: {str(e)}"
            })
            
        return {"results": results, "warnings": warnings}
    
    async def _validate_environment_setup(self) -> Dict[str, Any]:
        """Validate environment configuration and dependencies."""
        results = []
        warnings = []
        
        try:
            # Check FAISS installation
            try:
                import faiss
                results.append({
                    "check": "FAISS import",
                    "status": "PASS",
                    "details": f"FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}"
                })
                
                # Use new FAISS validator for comprehensive checking
                try:
                    from .faiss_validator import validate_faiss
                    validation = validate_faiss()
                    
                    if validation.is_valid and validation.has_gpu_support:
                        results.append({
                            "check": "FAISS GPU support",
                            "status": "PASS",
                            "details": f"GPU acceleration available ({validation.package_type} v{validation.version})"
                        })
                    elif validation.is_valid and not validation.has_gpu_support:
                        warnings.append({
                            "check": "FAISS GPU support",
                            "details": f"CPU-only configuration ({validation.package_type} v{validation.version})"
                        })
                    else:
                        errors.append({
                            "check": "FAISS GPU support",
                            "details": f"Issues detected: {', '.join(validation.issues)}"
                        })
                        if validation.auto_fix_available:
                            errors[-1]["auto_fix"] = "Available via /health/faiss-validation/fix endpoint"
                            
                except Exception as e:
                    warnings.append({
                        "check": "FAISS GPU support",
                        "details": f"Validation failed: {str(e)}"
                    })
                    
            except ImportError as e:
                results.append({
                    "check": "FAISS import",
                    "status": "FAIL",
                    "details": f"FAISS import failed: {str(e)}"
                })
            
            # Check critical environment variables
            import os
            required_env_vars = [
                "DATABASE_URL", "CONVERSATIONAL_MODEL", "ANALYTICAL_MODEL", 
                "REFLECTION_LLM", "HOST", "PORT"
            ]
            
            for var in required_env_vars:
                if os.getenv(var):
                    results.append({
                        "check": f"Environment variable {var}",
                        "status": "PASS",
                        "details": "Variable is set"
                    })
                else:
                    results.append({
                        "check": f"Environment variable {var}",
                        "status": "FAIL",
                        "details": f"Missing environment variable: {var}"
                    })
                    
        except Exception as e:
            results.append({
                "check": "Environment validation",
                "status": "FAIL",
                "details": f"Failed to validate environment: {str(e)}"
            })
            
        return {"results": results, "warnings": warnings}
    
    async def _validate_dependency_integrity(self) -> Dict[str, Any]:
        """Validate critical dependency imports and versions."""
        results = []
        warnings = []
        
        critical_imports = [
            ("fastapi", "FastAPI framework"),
            ("sqlalchemy", "Database ORM"),
            ("asyncpg", "PostgreSQL driver"),
            ("torch", "PyTorch for ML"),
            ("sentence_transformers", "Sentence embeddings"),
            ("pydantic", "Data validation")
        ]
        
        for module_name, description in critical_imports:
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, '__version__', 'unknown')
                results.append({
                    "check": f"{module_name} import",
                    "status": "PASS",
                    "details": f"{description} - version: {version}"
                })
            except ImportError as e:
                results.append({
                    "check": f"{module_name} import",
                    "status": "FAIL",
                    "details": f"Failed to import {description}: {str(e)}"
                })
                
        return {"results": results, "warnings": warnings}

# Global validator instance
startup_validator = StartupValidator()

async def run_startup_validation() -> Dict[str, Any]:
    """
    Run complete startup validation and return results.
    
    Returns:
        Validation results dictionary
    """
    return await startup_validator.validate_all()

def validate_startup_sync() -> Dict[str, Any]:
    """
    Synchronous wrapper for startup validation.
    
    Returns:
        Validation results dictionary
    """
    return asyncio.run(run_startup_validation())
