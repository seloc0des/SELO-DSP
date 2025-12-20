"""
FAISS GPU Validation and Auto-Fix Module

This module provides automatic detection and fixing of FAISS package conflicts
to ensure proper GPU acceleration is available when hardware supports it.
"""

import logging
import subprocess
import sys
import os
from typing import Dict, Any, Optional, Tuple
import importlib

logger = logging.getLogger("selo.faiss_validator")

class FAISSValidationResult:
    """Result of FAISS validation check."""
    
    def __init__(self, 
                 is_valid: bool, 
                 has_gpu_support: bool, 
                 package_type: str, 
                 version: str = None,
                 issues: list = None,
                 auto_fix_available: bool = False):
        self.is_valid = is_valid
        self.has_gpu_support = has_gpu_support
        self.package_type = package_type  # 'faiss-gpu', 'faiss-cpu', 'faiss', 'none'
        self.version = version
        self.issues = issues or []
        self.auto_fix_available = auto_fix_available
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "is_valid": self.is_valid,
            "has_gpu_support": self.has_gpu_support,
            "package_type": self.package_type,
            "version": self.version,
            "issues": self.issues,
            "auto_fix_available": self.auto_fix_available
        }

class FAISSValidator:
    """Validates and fixes FAISS installation issues."""
    
    def __init__(self):
        self.cuda_available = self._check_cuda_availability()
        
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available via PyTorch."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def validate_faiss_installation(self) -> FAISSValidationResult:
        """Validate current FAISS installation."""
        try:
            # Try to import FAISS
            import faiss
            version = getattr(faiss, '__version__', 'unknown')
            
            # Check GPU support
            has_gpu_resources = hasattr(faiss, 'StandardGpuResources')
            has_gpu_functions = hasattr(faiss, 'index_cpu_to_gpu')
            gpu_support = has_gpu_resources and has_gpu_functions
            
            # Determine package type
            package_type = self._detect_package_type()
            
            # Analyze issues
            issues = []
            auto_fix_available = False
            
            if self.cuda_available and not gpu_support:
                issues.append("CUDA available but FAISS GPU support missing")
                issues.append(f"Current package: {package_type}")
                issues.append("Recommendation: Install faiss-gpu package")
                auto_fix_available = True
            elif not self.cuda_available and gpu_support:
                issues.append("FAISS GPU package installed but CUDA not available")
            elif not gpu_support and package_type == 'faiss':
                issues.append("Generic 'faiss' package detected - may lack GPU support")
                if self.cuda_available:
                    auto_fix_available = True
            
            # Determine if installation is valid
            is_valid = len(issues) == 0 or (not self.cuda_available and not gpu_support)
            
            return FAISSValidationResult(
                is_valid=is_valid,
                has_gpu_support=gpu_support,
                package_type=package_type,
                version=version,
                issues=issues,
                auto_fix_available=auto_fix_available
            )
            
        except ImportError:
            return FAISSValidationResult(
                is_valid=False,
                has_gpu_support=False,
                package_type='none',
                issues=["FAISS not installed"],
                auto_fix_available=True
            )
        except Exception as e:
            return FAISSValidationResult(
                is_valid=False,
                has_gpu_support=False,
                package_type='unknown',
                issues=[f"FAISS validation error: {str(e)}"],
                auto_fix_available=False
            )
    
    def _detect_package_type(self) -> str:
        """Detect which FAISS package is installed."""
        try:
            # Check pip list for installed packages
            result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                output = result.stdout.lower()
                if 'faiss-gpu' in output:
                    return 'faiss-gpu'
                elif 'faiss-cpu' in output:
                    return 'faiss-cpu'
                elif 'faiss' in output:
                    return 'faiss'
            
            return 'unknown'
            
        except Exception:
            return 'unknown'
    
    def auto_fix_faiss_installation(self) -> Tuple[bool, str]:
        """Automatically fix FAISS installation issues."""
        validation = self.validate_faiss_installation()
        
        if not validation.auto_fix_available:
            return False, "Auto-fix not available for current configuration"
        
        try:
            logger.info("Starting automatic FAISS GPU fix...")
            
            # Step 1: Remove conflicting packages
            logger.info("Removing existing FAISS packages...")
            uninstall_cmd = [sys.executable, '-m', 'pip', 'uninstall', '-y', 
                           'faiss', 'faiss-gpu', 'faiss-cpu']
            subprocess.run(uninstall_cmd, capture_output=True, timeout=60)
            
            # Step 2: Ensure NumPy compatibility
            logger.info("Ensuring NumPy compatibility...")
            numpy_cmd = [sys.executable, '-m', 'pip', 'install', 'numpy>=1.24.0,<2.0.0']
            result = subprocess.run(numpy_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                return False, f"NumPy installation failed: {result.stderr}"
            
            # Step 3: Install appropriate FAISS package
            if self.cuda_available:
                logger.info("Installing FAISS GPU package...")
                install_cmd = [sys.executable, '-m', 'pip', 'install', 'faiss-gpu>=1.7.2']
                result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    # Verify GPU functionality
                    if self._verify_gpu_functionality():
                        logger.info("✅ FAISS GPU installation and verification successful")
                        return True, "FAISS GPU package installed and verified"
                    else:
                        logger.warning("FAISS GPU installed but verification failed, installing CPU fallback...")
                        # Fall back to CPU version
                        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'faiss-gpu'], 
                                     capture_output=True, timeout=60)
                        cpu_cmd = [sys.executable, '-m', 'pip', 'install', 'faiss-cpu>=1.7.2']
                        cpu_result = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=300)
                        
                        if cpu_result.returncode == 0:
                            return True, "FAISS GPU failed verification, CPU fallback installed"
                        else:
                            return False, f"Both GPU and CPU installation failed: {cpu_result.stderr}"
                else:
                    logger.warning("FAISS GPU installation failed, trying CPU version...")
                    cpu_cmd = [sys.executable, '-m', 'pip', 'install', 'faiss-cpu>=1.7.2']
                    cpu_result = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=300)
                    
                    if cpu_result.returncode == 0:
                        return True, "FAISS GPU installation failed, CPU version installed"
                    else:
                        return False, f"Both GPU and CPU installation failed: {cpu_result.stderr}"
            else:
                logger.info("CUDA not available, installing FAISS CPU package...")
                install_cmd = [sys.executable, '-m', 'pip', 'install', 'faiss-cpu>=1.7.2']
                result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    return True, "FAISS CPU package installed successfully"
                else:
                    return False, f"FAISS CPU installation failed: {result.stderr}"
                    
        except subprocess.TimeoutExpired:
            return False, "Installation timed out"
        except Exception as e:
            return False, f"Auto-fix failed: {str(e)}"
    
    def _verify_gpu_functionality(self) -> bool:
        """Verify FAISS GPU functionality works correctly."""
        try:
            # Reload FAISS module to get fresh import
            if 'faiss' in sys.modules:
                importlib.reload(sys.modules['faiss'])
            
            import faiss
            import numpy as np
            
            # Check basic GPU support
            if not (hasattr(faiss, 'StandardGpuResources') and hasattr(faiss, 'index_cpu_to_gpu')):
                return False
            
            # Test basic GPU operations with timeout
            index = faiss.IndexFlatL2(64)  # Small dimension for quick test
            gpu_resources = faiss.StandardGpuResources()
            
            # Use threading to prevent hanging
            import threading
            import time
            
            gpu_index_result = [None]
            gpu_error = [None]
            
            def gpu_test():
                try:
                    gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
                    
                    # Test basic operations
                    test_vectors = np.random.random((5, 64)).astype(np.float32)
                    gpu_index.add(test_vectors)
                    
                    query = np.random.random((1, 64)).astype(np.float32)
                    distances, indices = gpu_index.search(query, 3)
                    
                    gpu_index_result[0] = True
                except Exception as e:
                    gpu_error[0] = e
            
            # Run test with 5 second timeout
            test_thread = threading.Thread(target=gpu_test)
            test_thread.daemon = True
            test_thread.start()
            test_thread.join(timeout=5.0)
            
            if test_thread.is_alive():
                logger.warning("GPU functionality test timed out")
                return False
            elif gpu_error[0]:
                logger.warning(f"GPU functionality test failed: {gpu_error[0]}")
                return False
            elif gpu_index_result[0]:
                return True
            else:
                return False
                
        except Exception as e:
            logger.warning(f"GPU functionality verification failed: {e}")
            return False

# Global validator instance
faiss_validator = FAISSValidator()

def validate_faiss() -> FAISSValidationResult:
    """Convenience function to validate FAISS installation."""
    return faiss_validator.validate_faiss_installation()

def auto_fix_faiss() -> Tuple[bool, str]:
    """Convenience function to auto-fix FAISS installation."""
    return faiss_validator.auto_fix_faiss_installation()
