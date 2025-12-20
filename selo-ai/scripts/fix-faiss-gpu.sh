#!/bin/bash
# FAISS GPU Installation Fix Script
# This script fixes the FAISS CPU-only installation by replacing it with faiss-gpu

set -e

echo "=== FAISS GPU Installation Fix ==="
echo "This script will replace the CPU-only FAISS package with faiss-gpu"
echo

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "WARNING: Not in a virtual environment. This may affect system packages."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Check current FAISS installation
echo "--- Checking current FAISS installation ---"
python3 -c "
import sys
try:
    import faiss
    print(f'FAISS version: {faiss.__version__}')
    print(f'StandardGpuResources available: {hasattr(faiss, \"StandardGpuResources\")}')
    print(f'index_cpu_to_gpu available: {hasattr(faiss, \"index_cpu_to_gpu\")}')
    if hasattr(faiss, 'StandardGpuResources'):
        print('✅ GPU support detected')
        sys.exit(0)
    else:
        print('❌ GPU support missing - CPU-only package detected')
        sys.exit(1)
except ImportError:
    print('❌ FAISS not installed')
    sys.exit(2)
"

FAISS_STATUS=$?
if [ $FAISS_STATUS -eq 0 ]; then
    echo "✅ FAISS GPU support already available - no action needed"
    exit 0
elif [ $FAISS_STATUS -eq 2 ]; then
    echo "Installing faiss-gpu from scratch..."
    # Detect Python version for FAISS compatibility
    if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"; then
        echo "Python 3.12+ detected - using FAISS GPU 1.8.0+"
        pip install "faiss-gpu>=1.8.0"
    else
        echo "Python <3.12 detected - using FAISS GPU 1.7.2+"
        pip install "faiss-gpu>=1.7.2,<1.8.0"
    fi
else
    echo "Replacing CPU-only FAISS with GPU version..."
    
    # Remove CPU-only FAISS
    echo "--- Removing CPU-only FAISS package ---"
    pip uninstall -y faiss || true
    
    # Install GPU version with Python version detection
    echo "--- Installing FAISS GPU package ---"
    if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"; then
        echo "Python 3.12+ detected - using FAISS GPU 1.8.0+"
        pip install "faiss-gpu>=1.8.0"
    else
        echo "Python <3.12 detected - using FAISS GPU 1.7.2+"
        pip install "faiss-gpu>=1.7.2,<1.8.0"
    fi
fi

# Verify installation
echo
echo "--- Verifying FAISS GPU installation ---"
python3 -c "
import faiss
print(f'FAISS version: {faiss.__version__}')
print(f'StandardGpuResources available: {hasattr(faiss, \"StandardGpuResources\")}')
print(f'index_cpu_to_gpu available: {hasattr(faiss, \"index_cpu_to_gpu\")}')

if hasattr(faiss, 'StandardGpuResources') and hasattr(faiss, 'index_cpu_to_gpu'):
    print('✅ FAISS GPU support successfully installed')
else:
    print('❌ FAISS GPU support still missing')
    exit(1)
"

echo
echo "--- Testing GPU functionality ---"
python3 -c "
import torch
import faiss
import numpy as np

print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')

# Test basic FAISS GPU operations
try:
    # Create a simple index
    index = faiss.IndexFlatL2(128)
    
    # Test GPU resources creation
    gpu_resources = faiss.StandardGpuResources()
    print('✅ GPU resources created successfully')
    
    # Test index transfer (with small index to avoid hanging)
    gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
    print('✅ CPU to GPU index transfer successful')
    
    # Test basic operations
    test_vectors = np.random.random((10, 128)).astype(np.float32)
    gpu_index.add(test_vectors)
    
    query = np.random.random((1, 128)).astype(np.float32)
    distances, indices = gpu_index.search(query, 5)
    print(f'✅ GPU search successful - found {len(indices[0])} results')
    
    print('🎉 FAISS GPU functionality fully operational!')
    
except Exception as e:
    print(f'❌ FAISS GPU test failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

echo
echo "=== FAISS GPU Installation Complete ==="
echo "✅ FAISS GPU package installed and verified"
echo "✅ GPU acceleration now available for vector operations"
echo
echo "Next steps:"
echo "1. Restart the SELO DSP backend service to use GPU acceleration"
echo "2. Check logs for 'Successfully initialized GPU-accelerated FAISS index' message"
