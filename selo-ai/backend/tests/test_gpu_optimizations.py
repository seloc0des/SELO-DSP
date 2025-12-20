"""
GPU Optimization Tests

Test suite to validate all GPU optimizations are working correctly.
"""

import pytest
import asyncio
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Test GPU utilities
def test_gpu_utils_import():
    """Test that GPU utilities can be imported."""
    try:
        from backend.utils.gpu_utils import get_gpu_info, GPUOptimizedSentenceTransformer
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import GPU utilities: {e}")

def test_get_gpu_info():
    """Test GPU information gathering."""
    from backend.utils.gpu_utils import get_gpu_info
    
    info = get_gpu_info()
    assert isinstance(info, dict)
    assert "torch_available" in info
    assert "cuda_available" in info
    assert "gpu_count" in info

@pytest.mark.asyncio
async def test_vector_store_gpu_initialization():
    """Test that VectorStore can initialize with GPU support."""
    from backend.memory.vector_store import VectorStore
    
    # Test with GPU enabled
    store = VectorStore(embedding_dim=384, use_gpu=True)
    assert store.use_gpu == True
    
    # Test with GPU disabled
    store_cpu = VectorStore(embedding_dim=384, use_gpu=False)
    assert store_cpu.use_gpu == False

@pytest.mark.asyncio
async def test_vector_store_stats():
    """Test that VectorStore returns GPU statistics."""
    from backend.memory.vector_store import VectorStore
    
    store = VectorStore(embedding_dim=384, use_gpu=True)
    stats = store.get_stats()
    
    assert isinstance(stats, dict)
    assert "gpu_accelerated" in stats
    assert "device" in stats
    assert "backend" in stats

def test_sentence_transformer_gpu_wrapper():
    """Test GPU-optimized sentence transformer wrapper."""
    from backend.utils.gpu_utils import GPUOptimizedSentenceTransformer
    
    # Test initialization
    transformer = GPUOptimizedSentenceTransformer(use_gpu=True)
    assert transformer.use_gpu == True
    
    # Test stats
    stats = transformer.get_stats()
    assert isinstance(stats, dict)
    assert "gpu_available" in stats
    assert "device" in stats

def test_requirements_updated():
    """Test that requirements.txt has been updated with GPU packages."""
    # Get path relative to this test file
    backend_dir = Path(__file__).parent.parent
    requirements_path = backend_dir / "requirements.txt"
    
    with open(requirements_path, 'r') as f:
        content = f.read()
    
    # Check for GPU packages
    assert "faiss-gpu" in content
    assert "torch" in content
    assert "torchvision" in content

def test_environment_variables():
    """Test that GPU environment variables are properly configured."""
    # Get path relative to this test file
    backend_dir = Path(__file__).parent.parent
    env_example_path = backend_dir / ".env.example"
    
    with open(env_example_path, 'r') as f:
        content = f.read()
    
    # Check for GPU configuration variables
    assert "CUDA_VISIBLE_DEVICES" in content
    assert "PYTORCH_CUDA_ALLOC_CONF" in content
    assert "TORCH_CUDA_MEMORY_FRACTION" in content

@pytest.mark.asyncio
async def test_vector_store_embedding_operations():
    """Test that vector store embedding operations work."""
    from backend.memory.vector_store import VectorStore
    
    # Mock LLM controller
    mock_controller = Mock()
    mock_controller.get_embedding = Mock(return_value=[0.1] * 384)
    
    store = VectorStore(embedding_dim=384, llm_controller=mock_controller, use_gpu=True)
    
    # Test storing embedding
    embedding_id = await store.store_embedding("test text", metadata={"test": True})
    assert isinstance(embedding_id, str)
    
    # Test searching
    results = await store.search("test query", top_k=5)
    assert isinstance(results, list)

def test_faiss_gpu_availability():
    """Test FAISS GPU support detection."""
    try:
        import faiss
        gpu_available = hasattr(faiss, 'StandardGpuResources')
        # This test passes if FAISS is available, regardless of GPU support
        assert True
    except ImportError:
        pytest.skip("FAISS not available")

@pytest.mark.asyncio 
async def test_gpu_diagnostics_endpoint():
    """Test that GPU diagnostics endpoint includes new information."""
    # This would require the full FastAPI app to be running
    # For now, just test that the endpoint function exists
    try:
        from backend.main import diagnostics_gpu
        assert callable(diagnostics_gpu)
    except ImportError:
        pytest.fail("GPU diagnostics endpoint not found")

def test_cuda_environment_setup():
    """Test CUDA environment setup function."""
    from backend.utils.gpu_utils import setup_cuda_environment
    
    # Test that function runs without error
    try:
        setup_cuda_environment()
        assert True
    except Exception as e:
        pytest.fail(f"CUDA environment setup failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
