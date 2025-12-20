#!/bin/bash

# PyTorch Installation Script for SELO DSP
# This script installs PyTorch with CUDA support if available, CPU version otherwise

echo "Installing PyTorch for SELO DSP..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment detected: $VIRTUAL_ENV"
else
    echo "Warning: No virtual environment detected. Consider activating one first."
fi

# Auto-detect CUDA availability
CUDA_AVAILABLE=false
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi -L 2>/dev/null | grep -q "GPU"; then
        CUDA_AVAILABLE=true
        echo "✅ NVIDIA GPU detected"
    fi
fi

# Install appropriate PyTorch version
if $CUDA_AVAILABLE; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch CPU version..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install sentence-transformers as well (core dependency)
echo "Installing sentence-transformers..."
pip install "sentence-transformers>=2.2.2"

# Verify installation
python -c "
import torch
import sentence_transformers
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Sentence-transformers version: {sentence_transformers.__version__}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print('✅ GPU acceleration ready')
else:
    print('ℹ️  Using CPU-only mode')
"

echo "PyTorch and sentence-transformers installation complete!"
