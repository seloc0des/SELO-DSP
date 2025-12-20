#!/usr/bin/env python3
"""
Runtime FAISS GPU Fix Script

This script provides immediate FAISS GPU package replacement for existing installations
without requiring a full reinstall. It can be run while the backend is running.
"""

import subprocess
import sys
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_cuda_availability():
    """Check if CUDA is available."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"CUDA available: {cuda_available}, GPU: {gpu_name}")
        else:
            logger.info("CUDA not available")
        return cuda_available
    except ImportError:
        logger.warning("PyTorch not available - cannot check CUDA")
        return False

def get_current_faiss_info():
    """Get information about currently installed FAISS package."""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            output = result.stdout.lower()
            if 'faiss-gpu' in output:
                # Extract version
                for line in result.stdout.split('\n'):
                    if 'faiss-gpu' in line.lower():
                        parts = line.split()
                        version = parts[1] if len(parts) > 1 else 'unknown'
                        return 'faiss-gpu', version
            elif 'faiss-cpu' in output:
                for line in result.stdout.split('\n'):
                    if 'faiss-cpu' in line.lower():
                        parts = line.split()
                        version = parts[1] if len(parts) > 1 else 'unknown'
                        return 'faiss-cpu', version
            elif 'faiss' in output and 'faiss-' not in output:
                for line in result.stdout.split('\n'):
                    if line.strip().startswith('faiss '):
                        parts = line.split()
                        version = parts[1] if len(parts) > 1 else 'unknown'
                        return 'faiss', version
        
        return None, None
        
    except Exception as e:
        logger.error(f"Failed to get FAISS info: {e}")
        return None, None

def test_faiss_gpu_functionality():
    """Test if FAISS GPU functionality works."""
    try:
        import faiss
        import numpy as np
        
        # Check basic GPU support
        if not (hasattr(faiss, 'StandardGpuResources') and hasattr(faiss, 'index_cpu_to_gpu')):
            return False, "Missing GPU functions"
        
        # Test GPU operations with timeout
        import threading
        import time
        
        test_result = [False]
        test_error = [None]
        
        def gpu_test():
            try:
                # Create small test index
                index = faiss.IndexFlatL2(64)
                gpu_resources = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
                
                # Test basic operations
                test_vectors = np.random.random((5, 64)).astype(np.float32)
                gpu_index.add(test_vectors)
                
                query = np.random.random((1, 64)).astype(np.float32)
                distances, indices = gpu_index.search(query, 3)
                
                test_result[0] = True
            except Exception as e:
                test_error[0] = e
        
        # Run test with timeout
        test_thread = threading.Thread(target=gpu_test)
        test_thread.daemon = True
        test_thread.start()
        test_thread.join(timeout=5.0)
        
        if test_thread.is_alive():
            return False, "GPU test timed out"
        elif test_error[0]:
            return False, f"GPU test failed: {test_error[0]}"
        elif test_result[0]:
            return True, "GPU functionality verified"
        else:
            return False, "GPU test returned no result"
            
    except ImportError:
        return False, "FAISS not installed"
    except Exception as e:
        return False, f"GPU test error: {e}"

def fix_faiss_installation():
    """Fix FAISS installation by replacing with GPU version."""
    logger.info("🔧 Starting FAISS GPU fix...")
    
    # Check current state
    cuda_available = check_cuda_availability()
    package_type, version = get_current_faiss_info()
    
    logger.info(f"Current FAISS package: {package_type} v{version}")
    logger.info(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        logger.warning("CUDA not available - GPU acceleration not possible")
        return False
    
    if package_type == 'faiss-gpu':
        # Test if it actually works
        gpu_works, gpu_msg = test_faiss_gpu_functionality()
        if gpu_works:
            logger.info("✅ FAISS GPU already installed and working")
            return True
        else:
            logger.warning(f"FAISS GPU installed but not working: {gpu_msg}")
            logger.info("Attempting to reinstall...")
    
    try:
        # Step 1: Remove all FAISS packages
        logger.info("📦 Removing existing FAISS packages...")
        uninstall_cmd = [sys.executable, '-m', 'pip', 'uninstall', '-y', 
                        'faiss', 'faiss-gpu', 'faiss-cpu']
        result = subprocess.run(uninstall_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            logger.warning(f"Uninstall warnings: {result.stderr}")
        
        # Step 2: Ensure NumPy compatibility
        logger.info("🔢 Ensuring NumPy compatibility...")
        numpy_cmd = [sys.executable, '-m', 'pip', 'install', 'numpy>=1.24.0,<2.0.0']
        result = subprocess.run(numpy_cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            logger.error(f"NumPy installation failed: {result.stderr}")
            return False
        
        # Step 3: Install FAISS GPU with Python version-appropriate constraints
        if sys.version_info >= (3, 12):
            logger.info("🚀 Installing FAISS GPU 1.8.0+ (Python 3.12+)...")
            install_cmd = [sys.executable, '-m', 'pip', 'install', 'faiss-gpu>=1.8.0']
        else:
            logger.info("🚀 Installing FAISS GPU 1.7.2+ (Python <3.12)...")
            install_cmd = [sys.executable, '-m', 'pip', 'install', 'faiss-gpu>=1.7.2,<1.8.0']
        result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"FAISS GPU installation failed: {result.stderr}")
            
            # Fallback to CPU version
            logger.info("📉 Falling back to FAISS CPU...")
            cpu_cmd = [sys.executable, '-m', 'pip', 'install', 'faiss-cpu>=1.7.2']
            cpu_result = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=300)
            
            if cpu_result.returncode == 0:
                logger.info("✅ FAISS CPU installed as fallback")
                return True
            else:
                logger.error(f"FAISS CPU installation also failed: {cpu_result.stderr}")
                return False
        
        # Step 4: Verify installation
        logger.info("🔍 Verifying FAISS GPU installation...")
        gpu_works, gpu_msg = test_faiss_gpu_functionality()
        
        if gpu_works:
            logger.info("✅ FAISS GPU installation successful and verified!")
            return True
        else:
            logger.warning(f"FAISS GPU installed but verification failed: {gpu_msg}")
            logger.info("This may still work for basic operations")
            return True
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Installation timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Installation failed: {e}")
        return False

def main():
    """Main execution function."""
    print("🔧 SELO DSP FAISS GPU Runtime Fix")
    print("=" * 40)
    
    # Check if we're in the right directory
    backend_dir = Path(__file__).parent.parent / "backend"
    if not backend_dir.exists():
        logger.error("Backend directory not found. Run from selo-ai/scripts/ directory.")
        return 1
    
    # Show current status
    logger.info("📊 Current FAISS Status:")
    package_type, version = get_current_faiss_info()
    cuda_available = check_cuda_availability()
    
    if package_type:
        logger.info(f"  Package: {package_type} v{version}")
    else:
        logger.info("  Package: Not installed")
    
    logger.info(f"  CUDA Available: {cuda_available}")
    
    if package_type:
        gpu_works, gpu_msg = test_faiss_gpu_functionality()
        logger.info(f"  GPU Functionality: {gpu_msg}")
    
    # Determine if fix is needed
    if package_type == 'faiss-gpu' and cuda_available:
        gpu_works, gpu_msg = test_faiss_gpu_functionality()
        if gpu_works:
            logger.info("✅ No fix needed - FAISS GPU working correctly")
            return 0
    
    if not cuda_available:
        logger.info("ℹ️  CUDA not available - GPU acceleration not possible")
        if package_type != 'faiss-cpu':
            logger.info("Installing CPU version for consistency...")
        else:
            logger.info("✅ CPU version already installed")
            return 0
    
    # Perform fix
    logger.info("\n🔧 Starting fix process...")
    success = fix_faiss_installation()
    
    if success:
        logger.info("\n✅ FAISS fix completed successfully!")
        logger.info("🔄 Restart the SELO DSP backend to apply changes:")
        logger.info("   sudo systemctl restart selo-ai-backend")
        return 0
    else:
        logger.error("\n❌ FAISS fix failed!")
        logger.error("Manual intervention may be required.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
