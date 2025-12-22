"""
GPU Utilities Module

Provides utilities for GPU detection, configuration, and sentence transformer optimization.
"""

import logging
import os
from typing import Optional, Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger("selo.gpu_utils")

class GPUOptimizedSentenceTransformer:
    """
    GPU-optimized wrapper for SentenceTransformer models.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_gpu: bool = True):
        """
        Initialize GPU-optimized sentence transformer.
        
        Args:
            model_name: Name of the sentence transformer model
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.device = self._setup_device()
        self.model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._initialize_model()
        else:
            logger.warning("sentence-transformers not available, GPU optimization disabled")
    
    def _setup_device(self) -> Optional['torch.device']:
        """Setup the compute device."""
        if not TORCH_AVAILABLE:
            return None
            
        if self.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"SentenceTransformer using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("SentenceTransformer using CPU")
            
        return device
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            
            if self.device and self.device.type == 'cuda':
                self.model = self.model.to(self.device)
                logger.info(f"Moved SentenceTransformer model to GPU")
                
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer: {e}")
            self.model = None
    
    def encode(self, texts, **kwargs):
        """
        Encode texts to embeddings using GPU acceleration.
        
        Args:
            texts: Text or list of texts to encode
            **kwargs: Additional arguments for encoding
            
        Returns:
            Embeddings as numpy array or torch tensor
        """
        if not self.model:
            logger.warning("SentenceTransformer model not available")
            return None
            
        try:
            # Ensure we're using the correct device
            if self.device and self.device.type == 'cuda':
                kwargs.setdefault('device', self.device)
                
            embeddings = self.model.encode(texts, **kwargs)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        stats = {
            "model_name": self.model_name,
            "device": str(self.device) if self.device else "N/A",
            "gpu_available": self.device.type == 'cuda' if self.device else False,
            "model_loaded": self.model is not None,
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            stats["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            stats["gpu_memory_reserved"] = torch.cuda.memory_reserved()
            
        return stats

def get_gpu_info() -> Dict[str, Any]:
    """
    Get comprehensive GPU information.
    
    Returns:
        Dictionary with GPU information
    """
    info = {
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": False,
        "gpu_count": 0,
        "current_device": None,
        "device_name": None,
        "memory_info": {},
    }
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        info["cuda_available"] = True
        info["gpu_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(0)
        
        try:
            info["memory_info"] = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated(),
                "max_reserved": torch.cuda.max_memory_reserved(),
            }
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")
    
    return info

def optimize_gpu_memory() -> None:
    """
    Optimize GPU memory usage.
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Set memory fraction if configured
            memory_fraction = os.getenv("TORCH_CUDA_MEMORY_FRACTION")
            if memory_fraction:
                torch.cuda.set_per_process_memory_fraction(float(memory_fraction))
                logger.info(f"Set CUDA memory fraction to {memory_fraction}")
                
        except Exception as e:
            logger.warning(f"Could not optimize GPU memory: {e}")

def setup_cuda_environment() -> None:
    """
    Setup CUDA environment variables and optimizations.
    """
    # Set CUDA device order
    cuda_device_order = os.getenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_DEVICE_ORDER"] = cuda_device_order
    
    # Set visible devices
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        logger.info(f"CUDA visible devices: {cuda_visible_devices}")
    
    # PyTorch CUDA allocator configuration
    pytorch_cuda_alloc_conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF")
    if pytorch_cuda_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = pytorch_cuda_alloc_conf
        logger.info(f"PyTorch CUDA allocator config: {pytorch_cuda_alloc_conf}")
    
    # Optimize memory
    optimize_gpu_memory()
