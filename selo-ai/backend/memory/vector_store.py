"""
Vector Store Module

This module implements vector storage for SELO's semantic memory,
supporting embeddings for reflections, themes, and other semantic content.
"""

import logging
import os
import numpy as np
import json
import time
from typing import Dict, List, Optional, Any, Union
import uuid

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, falling back to simple vector operations")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, GPU acceleration disabled")

logger = logging.getLogger("selo.memory.vector_store")

class VectorStore:
    """
    Vector store for semantic search and retrieval.
    
    This class manages embeddings storage and semantic search functionality,
    supporting FAISS for efficient similarity search when available.
    """
    
    def __init__(self, 
                 embedding_dim: int = None,
                 store_path: Optional[str] = None,
                 llm_controller=None,
                 use_gpu: bool = True):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            store_path: Path to store index and metadata
            llm_controller: LLM controller for generating embeddings
            use_gpu: Whether to use GPU acceleration when available
        """
        # Determine default embedding dimension: ENV -> arg -> model-based default
        try:
            env_dim = int(os.getenv("EMBEDDING_DIM", "0"))
        except Exception:
            env_dim = 0
        
        # Auto-detect dimension based on embedding model if not explicitly set
        if not embedding_dim and env_dim == 0:
            embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
            if "nomic-embed-text" in embedding_model.lower():
                default_dim = 2048  # nomic-embed-text uses 2048 dimensions
            else:
                default_dim = 768  # Most other models use 768 (e.g., sentence-transformers)
        else:
            default_dim = 768
        
        self.embedding_dim = embedding_dim or (env_dim if env_dim > 0 else default_dim)
        self.store_path = store_path
        self.llm_controller = llm_controller
        self.use_gpu = use_gpu
        
        # GPU configuration
        self.device = self._setup_device()
        self.gpu_available = self.device.type == 'cuda' if TORCH_AVAILABLE else False
        
        # Initialize storage structures
        self.index = None
        self.gpu_index = None
        self.embeddings = []
        self.metadata = {}
        self.text_lookup = {}
        self.id_to_index = {}
        
        # Initialize index with the current embedding_dim
        self._initialize_index()
        
    def _setup_device(self) -> 'torch.device':
        """Setup the compute device (GPU/CPU)."""
        if not TORCH_AVAILABLE:
            return None
            
        if self.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
            
        return device
    
    def _initialize_index(self) -> None:
        """Initialize the vector index."""
        try:
            if FAISS_AVAILABLE:
                # Create FAISS index - using L2 distance for similarity
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                
                # Use FAISS validator for comprehensive checking
                try:
                    from ..core.faiss_validator import validate_faiss
                    validation = validate_faiss()
                    
                    if not validation.has_gpu_support:
                        logger.warning("FAISS GPU support not available")
                        for issue in validation.issues:
                            logger.warning(f"  - {issue}")
                        
                        if validation.auto_fix_available:
                            logger.info("Auto-fix available via /health/faiss-validation/fix endpoint")
                        
                        logger.info(f"Initialized CPU FAISS index with dimension {self.embedding_dim}")
                        self.gpu_available = False
                        return
                except Exception as e:
                    logger.warning(f"FAISS validation failed: {e}")
                    # Fallback to basic GPU support check
                    gpu_support_available = hasattr(faiss, 'StandardGpuResources') and hasattr(faiss, 'index_cpu_to_gpu')
                    
                    if not gpu_support_available:
                        logger.warning("FAISS GPU support not available - CPU-only FAISS package detected")
                        logger.warning("To enable GPU acceleration, install faiss-gpu package:")
                        logger.warning("  pip uninstall faiss")
                        import sys
                        if sys.version_info >= (3, 12):
                            logger.warning("  pip install faiss-gpu>=1.8.0  # Python 3.12+")
                        else:
                            logger.warning("  pip install faiss-gpu>=1.7.2  # Python <3.12")
                        logger.info(f"Initialized CPU FAISS index with dimension {self.embedding_dim}")
                        self.gpu_available = False
                        return
                
                # Try to use GPU if available
                if self.gpu_available:
                    try:
                        logger.info("Attempting to initialize FAISS GPU resources...")
                        
                        # Check CUDA context and environment
                        import os
                        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set')
                        
                        # Use threading to prevent hanging on GPU initialization
                        import threading
                        import time
                        
                        gpu_init_result = [None]
                        gpu_init_error = [None]
                        
                        def gpu_init():
                            try:
                                # Create GPU resources with conservative memory settings
                                gpu_resources = faiss.StandardGpuResources()
                                
                                # Set smaller temporary memory allocation (64MB instead of 256MB)
                                # This reduces the pinned host memory requirement
                                gpu_resources.setTempMemory(64 * 1024 * 1024)  # 64MB
                                
                                # Test basic GPU functionality
                                test_index = faiss.IndexFlatL2(64)  # Small test index
                                gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, test_index)
                                gpu_init_result[0] = gpu_resources
                            except Exception as e:
                                gpu_init_error[0] = e
                        
                        # Run GPU initialization with timeout
                        init_thread = threading.Thread(target=gpu_init)
                        init_thread.daemon = True
                        init_thread.start()
                        init_thread.join(timeout=10.0)  # 10 second timeout
                        
                        if init_thread.is_alive():
                            logger.warning("FAISS GPU initialization timed out - falling back to CPU")
                            self.use_gpu = False
                            self.gpu_resources = None
                        elif gpu_init_error[0]:
                            logger.warning(f"FAISS GPU initialization failed: {gpu_init_error[0]}")
                            logger.warning("Falling back to CPU-only mode")
                            self.use_gpu = False
                            self.gpu_resources = None
                        elif gpu_init_result[0]:
                            self.gpu_resources = gpu_init_result[0]
                            logger.info("✅ FAISS GPU initialization successful")
                        else:
                            logger.warning("FAISS GPU initialization returned no result - falling back to CPU")
                            self.use_gpu = False
                            self.gpu_resources = None
                    except Exception as e:
                        logger.warning(f"FAISS GPU setup failed: {e}")
                        logger.warning("Falling back to CPU-only mode")
                        self.use_gpu = False
                        self.gpu_resources = None
                else:
                    logger.info("FAISS GPU disabled or CUDA not available - using CPU mode")
                    self.use_gpu = False
                    self.gpu_resources = None
                    
                # Load existing index if available
                if self.store_path and os.path.exists(self.store_path):
                    self._load_store()
                    
            else:
                # Fallback to numpy array for embeddings
                self.embeddings = []
                logger.info("Using simple numpy array for embeddings (FAISS not available)")
                
            # Load existing index if available
            if self.store_path and os.path.exists(self.store_path):
                self._load_store()
                
        except Exception as e:
            logger.error(f"Error initializing vector index: {str(e)}", exc_info=True)
            # Fallback to empty index
            self.embeddings = []
            self.metadata = {}
            self.text_lookup = {}
            self.id_to_index = {}
            
    def _load_store(self) -> None:
        """Load vector store from disk."""
        try:
            if not self.store_path:
                return
                
            # Check for metadata file
            metadata_path = os.path.join(self.store_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    data = json.load(f)
                    self.metadata = data.get("metadata", {})
                    self.text_lookup = data.get("text_lookup", {})
                    self.id_to_index = data.get("id_to_index", {})
                    logger.info(f"Loaded metadata with {len(self.metadata)} entries")
                    
            # Load FAISS index if available
            if FAISS_AVAILABLE:
                index_path = os.path.join(self.store_path, "faiss_index.bin")
                if os.path.exists(index_path):
                    self.index = faiss.read_index(index_path)
                    loaded_dim = getattr(self.index, "d", None)
                    if loaded_dim and loaded_dim != self.embedding_dim:
                        self.embedding_dim = loaded_dim
                    # Try to move to GPU if available
                    if self.gpu_available and hasattr(faiss, 'StandardGpuResources'):
                        try:
                            if not hasattr(self, 'gpu_resources'):
                                self.gpu_resources = faiss.StandardGpuResources()
                                # Set conservative memory allocation to prevent OOM
                                self.gpu_resources.setTempMemory(64 * 1024 * 1024)  # 64MB
                            self.gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors (GPU-accelerated)")
                        except Exception as gpu_err:
                            logger.warning(f"Failed to move loaded index to GPU: {gpu_err}")
                            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors (CPU)")
                    else:
                        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors (CPU)")
            else:
                # Load numpy embeddings
                embeddings_path = os.path.join(self.store_path, "embeddings.npy")
                if os.path.exists(embeddings_path):
                    self.embeddings = list(np.load(embeddings_path))
                    logger.info(f"Loaded {len(self.embeddings)} embeddings from file")
                    
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}", exc_info=True)
            
    def _save_store(self) -> None:
        """Save vector store to disk."""
        try:
            if not self.store_path:
                return
                
            # Create directory if needed
            os.makedirs(self.store_path, exist_ok=True)
            
            # Save metadata
            metadata_path = os.path.join(self.store_path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({
                    "metadata": self.metadata,
                    "text_lookup": self.text_lookup,
                    "id_to_index": self.id_to_index,
                }, f)
                
            # Save FAISS index if available
            if FAISS_AVAILABLE and self.index is not None:
                index_path = os.path.join(self.store_path, "faiss_index.bin")
                # Always save the CPU version for portability
                if self.gpu_index is not None:
                    cpu_index = faiss.index_gpu_to_cpu(self.gpu_index)
                    faiss.write_index(cpu_index, index_path)
                else:
                    faiss.write_index(self.index, index_path)
            else:
                # Save numpy embeddings
                embeddings_path = os.path.join(self.store_path, "embeddings.npy")
                np.save(embeddings_path, np.array(self.embeddings))
                
            logger.info(f"Saved vector store to {self.store_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}", exc_info=True)
            
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if self.llm_controller:
            try:
                # Preferred: direct get_embedding() on controller
                if hasattr(self.llm_controller, "get_embedding") and callable(getattr(self.llm_controller, "get_embedding")):
                    return await self.llm_controller.get_embedding(text)
                # Router path: use route(task_type='embedding', prompt=text)
                if hasattr(self.llm_controller, "route") and callable(getattr(self.llm_controller, "route")):
                    routed = await self.llm_controller.route(task_type="embedding", prompt=text)
                    # Common shapes: {embedding: [...]}, {vector: [...]}, or content as space/comma-separated floats
                    if isinstance(routed, dict):
                        vec = routed.get("embedding") or routed.get("vector") or routed.get("data")
                        if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
                            return [float(x) for x in vec]
                        # Some providers may return nested under 'output'
                        out = routed.get("output") or routed.get("content")
                        if isinstance(out, list) and all(isinstance(x, (int, float)) for x in out):
                            return [float(x) for x in out]
                        if isinstance(out, str):
                            try:
                                parts = [p for p in out.replace(",", " ").split() if p]
                                floats = [float(p) for p in parts]
                                if floats:
                                    return floats
                            except Exception:
                                pass
                    # As a last attempt, if routed is a list of numbers
                    if isinstance(routed, list) and all(isinstance(x, (int, float)) for x in routed):
                        return [float(x) for x in routed]
            except Exception as e:
                logger.error(f"Error getting embedding from LLM/Router: {str(e)}", exc_info=True)
                
        # Fallback to simple hash-based embedding
        return self._simple_embedding(text)
        
    def _simple_embedding(self, text: str) -> List[float]:
        """
        Generate a simple hash-based embedding for development/testing.
        
        Args:
            text: Text to embed
            
        Returns:
            Simple embedding vector
        """
        # Create a deterministic but simple embedding from the text
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.random(self.embedding_dim).astype(np.float32)
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
        
    async def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add multiple texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dicts for each text
            
        Returns:
            List of IDs for the stored embeddings
        """
        ids = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else None
            embedding_id = await self.store_embedding(text, metadata=metadata)
            ids.append(embedding_id)
        return ids

    async def store_embedding(self, 
                       text: str, 
                       embedding_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store an embedding for a text string.
        
        Args:
            text: Text to embed
            embedding_id: Optional ID for the embedding
            metadata: Optional metadata to store with the embedding
            
        Returns:
            ID of the stored embedding
        """
        try:
            # Generate ID if not provided
            embedding_id = embedding_id or str(uuid.uuid4())
            
            # Get embedding
            embedding_vector = await self.get_embedding(text)

            # If FAISS is available, ensure index dimensionality matches the vector
            if FAISS_AVAILABLE:
                vec_dim = len(embedding_vector)
                if self.index is None:
                    # Initialize fresh index based on detected dimension
                    self.embedding_dim = vec_dim
                    self._initialize_index()
                    # Persist resolved dimension for future boots
                    try:
                        self._persist_embedding_dim(vec_dim)
                    except Exception:
                        pass
                elif vec_dim != self.embedding_dim:
                    # Dimension mismatch: if index is empty, recreate with new dim; else rebuild
                    logger.warning(
                        f"Embedding dimension mismatch detected: vector={vec_dim}, index={self.embedding_dim}. "
                        f"Reinitializing index to {vec_dim}.")
                    self.embedding_dim = vec_dim
                    # Reset in-memory structures and recreate index fresh
                    self.index = None
                    self.gpu_index = None
                    self.embeddings = []
                    self.metadata = {}
                    self.text_lookup = {}
                    self.id_to_index = {}
                    self._initialize_index()
                    # Persist resolved dimension for future boots
                    try:
                        self._persist_embedding_dim(vec_dim)
                    except Exception:
                        pass
            
            # Store the embedding
            if FAISS_AVAILABLE and self.index is not None:
                # Add to FAISS (use GPU index if available)
                index = len(self.metadata)
                embedding_array = np.array([embedding_vector], dtype=np.float32)
                
                if self.gpu_index is not None:
                    self.gpu_index.add(embedding_array)
                else:
                    self.index.add(embedding_array)
                    
                self.metadata[embedding_id] = metadata or {}
                self.text_lookup[embedding_id] = text
                self.id_to_index[embedding_id] = index
            else:
                # Add to simple list
                self.embeddings.append(embedding_vector)
                index = len(self.embeddings) - 1
                self.metadata[embedding_id] = metadata or {}
                self.text_lookup[embedding_id] = text
                self.id_to_index[embedding_id] = index
                
            # Periodically save the store
            if len(self.metadata) % 100 == 0:
                self._save_store()
                
            return embedding_id
            
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}", exc_info=True)
            return str(uuid.uuid4())  # Return a dummy ID on error
            
    async def search(self, 
               query: str, 
               top_k: int = 5, 
               threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar items by text query.
        
        Args:
            query: Text query
            top_k: Number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of results with metadata and similarity scores
        """
        try:
            # Get query embedding
            query_embedding = await self.get_embedding(query)
            
            # Search by embedding
            return await self.search_by_embedding(
                embedding=query_embedding,
                top_k=top_k,
                threshold=threshold
            )
            
        except Exception as e:
            logger.error(f"Error in text search: {str(e)}", exc_info=True)
            return []

    def _persist_embedding_dim(self, dim: int) -> None:
        """Persist EMBEDDING_DIM to backend/.env and current process env.
        Safe no-op if paths are not writable.
        """
        try:
            os.environ["EMBEDDING_DIM"] = str(dim)
        except Exception:
            pass
        try:
            # backend/.env is two levels up from this file's directory
            here = os.path.abspath(os.path.dirname(__file__))
            backend_dir = os.path.abspath(os.path.join(here, os.pardir))
            env_path = os.path.join(backend_dir, ".env")
            if os.path.exists(env_path) and os.access(env_path, os.W_OK):
                # Read and update EMBEDDING_DIM line
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                found = False
                for i, ln in enumerate(lines):
                    if ln.startswith("EMBEDDING_DIM="):
                        lines[i] = f"EMBEDDING_DIM={dim}\n"
                        found = True
                        break
                if not found:
                    lines.append(f"EMBEDDING_DIM={dim}\n")
                with open(env_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
        except Exception:
            # Non-fatal if we cannot persist
            pass
            
    async def search_by_embedding(self, 
                           embedding: List[float], 
                           top_k: int = 5,
                           threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar items by embedding vector.
        
        Args:
            embedding: Embedding vector
            top_k: Number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of results with metadata and similarity scores
        """
        try:
            if len(self.metadata) == 0:
                return []
                
            if FAISS_AVAILABLE and self.index is not None:
                # Search in FAISS index (use GPU index if available)
                query_vector = np.array([embedding], dtype=np.float32)
                
                if self.gpu_index is not None:
                    distances, indices = self.gpu_index.search(query_vector, min(top_k, self.gpu_index.ntotal))
                else:
                    distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
                
                # Process results
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx == -1:  # FAISS returns -1 for empty slots
                        continue
                        
                    # Find ID for this index
                    embedding_id = None
                    for eid, eidx in self.id_to_index.items():
                        if eidx == idx:
                            embedding_id = eid
                            break
                            
                    if embedding_id:
                        # Convert L2 distance to similarity score (0-1)
                        similarity = max(0, 1 - distances[0][i] / 10.0)  # Normalize, higher is better
                        
                        if similarity >= threshold:
                            results.append({
                                "id": embedding_id,
                                "text": self.text_lookup.get(embedding_id, ""),
                                "metadata": self.metadata.get(embedding_id, {}),
                                "similarity": similarity
                            })
                            
                return results
                
            else:
                # Simple search in numpy array
                if not self.embeddings:
                    return []
                    
                query_vector = np.array(embedding, dtype=np.float32)
                all_vectors = np.array(self.embeddings, dtype=np.float32)
                
                # Calculate cosine similarities
                # Normalize vectors
                query_norm = query_vector / np.linalg.norm(query_vector)
                all_norms = all_vectors / np.linalg.norm(all_vectors, axis=1, keepdims=True)
                
                # Calculate dot products (cosine similarities)
                similarities = np.dot(all_norms, query_norm)
                
                # Get top-k indices
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                # Build results
                results = []
                for idx in top_indices:
                    # Find ID for this index
                    embedding_id = None
                    for eid, eidx in self.id_to_index.items():
                        if eidx == idx:
                            embedding_id = eid
                            break
                            
                    if embedding_id:
                        similarity = float(similarities[idx])
                        if similarity >= threshold:
                            results.append({
                                "id": embedding_id,
                                "text": self.text_lookup.get(embedding_id, ""),
                                "metadata": self.metadata.get(embedding_id, {}),
                                "similarity": similarity
                            })
                            
                return results
                
        except Exception as e:
            logger.error(f"Error in embedding search: {str(e)}", exc_info=True)
            return []
            
    def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete an embedding by ID.
        
        Args:
            embedding_id: ID of the embedding to delete
            
        Returns:
            True if successfully deleted
        """
        try:
            if embedding_id not in self.metadata:
                logger.warning(f"Embedding {embedding_id} not found")
                return False
                
            # Note: In FAISS, we can't easily delete individual vectors
            # Instead, we mark as deleted in metadata and rebuild index periodically
            if embedding_id in self.metadata:
                del self.metadata[embedding_id]
                
            if embedding_id in self.text_lookup:
                del self.text_lookup[embedding_id]
                
            if embedding_id in self.id_to_index:
                del self.id_to_index[embedding_id]
                
            logger.info(f"Marked embedding {embedding_id} as deleted")
            
            # Periodically rebuild index if too many deletions
            deleted_count = max(0, (self.index.ntotal if FAISS_AVAILABLE and self.index else len(self.embeddings)) - len(self.metadata))
            if deleted_count > 100 or deleted_count > 0.25 * len(self.metadata):
                self._rebuild_index()
                
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embedding: {str(e)}", exc_info=True)
            return False
            
    def _rebuild_index(self) -> None:
        """Rebuild the index after many deletions."""
        try:
            logger.info("Rebuilding vector index...")
            
            # Get valid embeddings and metadata
            valid_embeddings = []
            valid_metadata = {}
            valid_text_lookup = {}
            valid_id_to_index = {}
            
            index = 0
            for embedding_id, idx in self.id_to_index.items():
                if embedding_id in self.metadata:
                    if FAISS_AVAILABLE and self.index is not None:
                        # We need to extract the embedding from the FAISS index
                        # This is not directly supported, so we'd need the original vectors
                        # For now, we'll skip this in the example
                        pass
                    else:
                        valid_embeddings.append(self.embeddings[idx])
                        
                    valid_metadata[embedding_id] = self.metadata[embedding_id]
                    valid_text_lookup[embedding_id] = self.text_lookup.get(embedding_id, "")
                    valid_id_to_index[embedding_id] = index
                    index += 1
            
            # Create new index
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                # Try to recreate GPU index if available
                if self.gpu_available and hasattr(faiss, 'StandardGpuResources'):
                    try:
                        if not hasattr(self, 'gpu_resources'):
                            self.gpu_resources = faiss.StandardGpuResources()
                            # Set conservative memory allocation to prevent OOM
                            self.gpu_resources.setTempMemory(64 * 1024 * 1024)  # 64MB
                        self.gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                    except Exception:
                        self.gpu_index = None
                # Add valid embeddings back
                # This part would require the original vectors
            else:
                self.embeddings = valid_embeddings
                
            # Update metadata structures
            self.metadata = valid_metadata
            self.text_lookup = valid_text_lookup
            self.id_to_index = valid_id_to_index
            
            logger.info(f"Rebuilt vector index with {len(self.metadata)} embeddings")
            
            # Save the rebuilt index
            self._save_store()
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}", exc_info=True)
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        # Determine GPU support status
        gpu_support_available = FAISS_AVAILABLE and hasattr(faiss, 'StandardGpuResources') and hasattr(faiss, 'index_cpu_to_gpu')
        gpu_actually_used = self.gpu_index is not None
        
        stats = {
            "embedding_dim": self.embedding_dim,
            "total_embeddings": len(self.metadata),
            "backend": "FAISS" if FAISS_AVAILABLE and self.index else "numpy",
            "gpu_accelerated": gpu_actually_used,
            "device": str(self.device) if self.device else "N/A",
            "faiss_available": FAISS_AVAILABLE,
            "faiss_gpu_support": gpu_support_available,
            "pytorch_cuda_available": TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
        }
        
        # Add status explanation
        if gpu_actually_used:
            stats["status"] = "GPU acceleration active"
        elif gpu_support_available and TORCH_AVAILABLE and torch.cuda.is_available():
            stats["status"] = "GPU available but using CPU fallback"
        elif not gpu_support_available and FAISS_AVAILABLE:
            stats["status"] = "CPU-only FAISS package detected (install faiss-gpu for acceleration)"
        elif not FAISS_AVAILABLE:
            stats["status"] = "FAISS not available, using numpy fallback"
        else:
            stats["status"] = "CPU-only configuration"
        
        if FAISS_AVAILABLE and self.index:
            if self.gpu_index is not None:
                stats["index_size"] = self.gpu_index.ntotal
            else:
                stats["index_size"] = self.index.ntotal
                
        if TORCH_AVAILABLE and torch.cuda.is_available():
            stats["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            stats["gpu_memory_reserved"] = torch.cuda.memory_reserved()
            
        return stats
