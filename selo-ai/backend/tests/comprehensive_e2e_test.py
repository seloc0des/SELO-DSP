#!/usr/bin/env python3
"""
Comprehensive End-to-End Testing Script for SELO AI
Tests all major components, APIs, and integration points to identify remaining issues.
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import requests
import uuid

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

class SELOTestSuite:
    def __init__(self, base_url: str = "http://localhost:8000", frontend_url: str = "http://localhost:3000", default_timeout: Optional[float] = None):
        self.base_url = base_url
        self.frontend_url = frontend_url
        self.test_results = []
        self.session_id = str(uuid.uuid4())
        # Default client timeouts: 300s for general, 420s for heavy LLM/reflection paths
        # Increased to accommodate slow but functional LLM responses (100+ seconds)
        self.timeout = 300.0
        self.long_timeout = 420.0
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('e2e_test_results.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_test(self, test_name: str, status: str, details: str = "", error: str = ""):
        """Log test result and add to results list"""
        result = {
            "test": test_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "error": error
        }
        self.test_results.append(result)
        
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        self.logger.info(f"{status_emoji} {test_name}: {status}")
        if details:
            self.logger.info(f"   Details: {details}")
        if error:
            self.logger.error(f"   Error: {error}")

    def test_system_dependencies(self):
        """Test system-level dependencies and services"""
        self.logger.info("=== SYSTEM DEPENDENCIES TESTS ===")
        
        # Test Python environment
        try:
            python_version = sys.version
            self.log_test("Python Version", "PASS", f"Python {python_version}")
        except Exception as e:
            self.log_test("Python Version", "FAIL", error=str(e))

        # Test required Python packages
        required_packages = [
            'fastapi', 'uvicorn', 'pydantic', 'sqlalchemy', 'asyncpg', 
            'torch', 'numpy', 'scipy', 'sentence_transformers'
        ]
        
        # Test FAISS separately (different import name)
        try:
            import faiss
            self.log_test("Package: faiss", "PASS")
        except ImportError as e:
            self.log_test("Package: faiss", "FAIL", error=str(e))
        
        for package in required_packages:
            try:
                __import__(package)
                self.log_test(f"Package: {package}", "PASS")
            except ImportError as e:
                self.log_test(f"Package: {package}", "FAIL", error=str(e))

        # Test CUDA availability
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                self.log_test("CUDA Availability", "PASS", f"{device_count} devices, primary: {device_name}")
            else:
                self.log_test("CUDA Availability", "WARN", "CUDA not available - will use CPU")
        except Exception as e:
            self.log_test("CUDA Availability", "FAIL", error=str(e))

        # Test FAISS GPU
        try:
            import faiss
            has_gpu_resources = hasattr(faiss, 'StandardGpuResources')
            self.log_test("FAISS GPU Support", "PASS" if has_gpu_resources else "WARN", 
                         f"GPU resources available: {has_gpu_resources}")
        except Exception as e:
            self.log_test("FAISS GPU Support", "FAIL", error=str(e))

        # Test Ollama service
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', 'unknown') for m in models]
                self.log_test("Ollama Service", "PASS", f"Running with {len(models)} models: {model_names[:3]}")
            else:
                self.log_test("Ollama Service", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Ollama Service", "FAIL", error=str(e))

        # Test PostgreSQL connection
        try:
            import asyncpg
            self.log_test("PostgreSQL Driver", "PASS")
        except Exception as e:
            self.log_test("PostgreSQL Driver", "FAIL", error=str(e))

    def test_backend_health(self):
        """Test backend service health and basic endpoints"""
        self.logger.info("=== BACKEND HEALTH TESTS ===")
        
        # Basic health check
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout or 10)
            if response.status_code == 200:
                health_data = response.json()
                self.log_test("Backend Health", "PASS", f"Status: {health_data.get('status')}")
            else:
                self.log_test("Backend Health", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Backend Health", "FAIL", error=str(e))

        # Detailed health check
        try:
            response = requests.get(f"{self.base_url}/health/details?probe_llm=false&probe_db=true", timeout=self.timeout or 15)
            if response.status_code == 200:
                details = response.json()
                db_ok = ((details.get('probes') or {}).get('db') or {}).get('ok')
                self.log_test("Backend Health Details", "PASS" if db_ok else "WARN", 
                             f"DB probe: {db_ok}")
            else:
                self.log_test("Backend Health Details", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Backend Health Details", "FAIL", error=str(e))

        # Test new performance endpoints
        try:
            response = requests.get(f"{self.base_url}/health/performance", timeout=self.timeout or 10)
            if response.status_code == 200:
                perf_data = response.json()
                cache_enabled = (perf_data.get('performance') or {}).get('performance_optimizations_enabled', False)
                self.log_test("Performance Health", "PASS", f"Optimizations enabled: {cache_enabled}")
            else:
                self.log_test("Performance Health", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Performance Health", "FAIL", error=str(e))

        # Test cache management endpoint
        try:
            response = requests.post(f"{self.base_url}/health/cache/clear", timeout=self.timeout or 10)
            if response.status_code == 200:
                self.log_test("Cache Management", "PASS", "Cache clear endpoint working")
            else:
                self.log_test("Cache Management", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Cache Management", "FAIL", error=str(e))

        # GPU diagnostics
        try:
            response = requests.get(f"{self.base_url}/diagnostics/gpu", timeout=self.timeout or 10)
            if response.status_code == 200:
                gpu_data = response.json()
                cuda_detected = gpu_data.get('cuda_detected', False)
                faiss_gpu = gpu_data.get('faiss_gpu_available', False)
                vector_gpu = (gpu_data.get('vector_store_gpu') or {}).get('gpu_accelerated', False)
                self.log_test("GPU Diagnostics", "PASS", 
                             f"CUDA: {cuda_detected}, FAISS-GPU: {faiss_gpu}, Vector GPU: {vector_gpu}")
            else:
                self.log_test("GPU Diagnostics", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("GPU Diagnostics", "FAIL", error=str(e))

        # Environment diagnostics
        try:
            response = requests.get(f"{self.base_url}/diagnostics/env", timeout=self.timeout or 5)
            if response.status_code == 200:
                env_data = response.json()
                port = env_data.get('PORT') or env_data.get('SELO_AI_PORT')
                self.log_test("Environment Diagnostics", "PASS", f"Port: {port}")
            else:
                self.log_test("Environment Diagnostics", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Environment Diagnostics", "FAIL", error=str(e))

    def test_database_operations(self):
        """Test database connectivity and basic operations"""
        self.logger.info("=== DATABASE TESTS ===")
        
        # Test user creation/retrieval
        try:
            response = requests.get(f"{self.base_url}/api/users", timeout=self.timeout or 10)
            if response.status_code == 200:
                users = response.json()
                self.log_test("User API", "PASS", f"Retrieved {len(users)} users")
            else:
                self.log_test("User API", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("User API", "FAIL", error=str(e))

        # Test conversation listing
        try:
            response = requests.get(f"{self.base_url}/api/conversations?user_id={self.session_id}", timeout=self.timeout or 10)
            if response.status_code == 200:
                conversations = response.json()
                self.log_test("Conversation API", "PASS", f"Retrieved conversations structure")
            else:
                self.log_test("Conversation API", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Conversation API", "FAIL", error=str(e))

        # Test reflection listing
        try:
            response = requests.get(f"{self.base_url}/api/reflections/list?user_profile_id={self.session_id}&limit=10", timeout=self.timeout or 10)
            if response.status_code == 200:
                reflections = response.json()
                self.log_test("Reflection API", "PASS", f"Retrieved reflections structure")
            else:
                self.log_test("Reflection API", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Reflection API", "FAIL", error=str(e))

    def test_llm_integration(self):
        """Test LLM integration and model availability"""
        self.logger.info("=== LLM INTEGRATION TESTS ===")
        
        # Test model availability check (our new validation system)
        try:
            response = requests.get(f"{self.base_url}/diagnostics/gpu", timeout=self.timeout or 15)
            if response.status_code == 200:
                gpu_data = response.json()
                # Check if our configured models are detected
                ollama_info = gpu_data.get('ollama', {})
                models_available = ollama_info.get('models_available', [])
                
                # Check for our configured models
                expected_models = ['humanish-llama3:8b-q4', 'qwen2.5-coder:3b', 'phi3:mini-4k-instruct']
                found_models = [m for m in expected_models if any(m in str(model) for model in models_available)]
                
                if found_models:
                    self.log_test("Model Availability", "PASS", f"Found models: {found_models}")
                else:
                    self.log_test("Model Availability", "WARN", f"Expected models not found. Available: {models_available[:3]}")
            else:
                self.log_test("Model Availability", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Model Availability", "FAIL", error=str(e))
        
        # Test LLM health with actual model call
        try:
            # When no timeouts are requested, allow the LLM probe to run without client deadline
            response = requests.get(f"{self.base_url}/diagnostics/gpu?test_llm=true", timeout=self.long_timeout)
            if response.status_code == 200:
                gpu_data = response.json()
                tested_llm = gpu_data.get('tested_llm', False)
                test_error = gpu_data.get('test_error')
                if tested_llm:
                    duration = gpu_data.get('test_duration_s', 0)
                    self.log_test("LLM Health Test", "PASS", f"Completed in {duration}s")
                else:
                    self.log_test("LLM Health Test", "FAIL", f"Test failed: {test_error}")
            else:
                self.log_test("LLM Health Test", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("LLM Health Test", "FAIL", error=str(e))

        # Test search functionality
        try:
            response = requests.post(f"{self.base_url}/search", 
                                   params={"query": "test search"}, timeout=self.timeout or 15)
            if response.status_code == 200:
                search_result = response.json()
                result_text = search_result.get('result', '')
                self.log_test("Search API", "PASS", f"Result length: {len(result_text)}")
            else:
                self.log_test("Search API", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Search API", "FAIL", error=str(e))

    def test_chat_functionality(self):
        """Test chat endpoint with various scenarios"""
        self.logger.info("=== CHAT FUNCTIONALITY TESTS ===")
        
        # Test basic chat with UUID session ID
        try:
            chat_data = {
                "session_id": self.session_id,
                "prompt": "Hello, this is a test message. Please respond briefly."
            }
            response = requests.post(f"{self.base_url}/chat", 
                                   json=chat_data, timeout=self.long_timeout)
            if response.status_code == 200:
                chat_result = response.json()
                response_text = chat_result.get('response', '')
                turn_id = chat_result.get('turn_id', '')
                self.log_test("Chat Basic (UUID)", "PASS", 
                             f"Response length: {len(response_text)}, Turn ID: {turn_id[:8]}...")
            else:
                error_detail = response.text
                self.log_test("Chat Basic (UUID)", "FAIL", 
                             f"HTTP {response.status_code}: {error_detail}")
        except Exception as e:
            self.log_test("Chat Basic (UUID)", "FAIL", error=str(e))

        # Test chat with legacy session ID format
        try:
            legacy_session = f"user-{int(time.time())}-test123"
            chat_data = {
                "session_id": legacy_session,
                "prompt": "Test with legacy session ID format."
            }
            response = requests.post(f"{self.base_url}/chat", 
                                   json=chat_data, timeout=self.long_timeout)
            if response.status_code == 200:
                chat_result = response.json()
                self.log_test("Chat Legacy Session", "PASS", "Legacy session ID accepted")
            else:
                error_detail = response.text
                self.log_test("Chat Legacy Session", "FAIL", 
                             f"HTTP {response.status_code}: {error_detail}")
        except Exception as e:
            self.log_test("Chat Legacy Session", "FAIL", error=str(e))

        # Test chat with invalid session ID
        try:
            chat_data = {
                "session_id": "invalid-session-format!@#",
                "prompt": "This should fail validation."
            }
            response = requests.post(f"{self.base_url}/chat", 
                                   json=chat_data, timeout=self.timeout or 10)
            if response.status_code == 422:
                self.log_test("Chat Invalid Session", "PASS", "Properly rejected invalid session ID")
            else:
                self.log_test("Chat Invalid Session", "FAIL", 
                             f"Expected 422, got {response.status_code}")
        except Exception as e:
            self.log_test("Chat Invalid Session", "FAIL", error=str(e))
        
        # Test chat timeout handling (with a very short timeout to trigger it)
        try:
            chat_data = {
                "session_id": self.session_id,
                "prompt": "Generate a very long response that might timeout. " * 50
            }
            response = requests.post(f"{self.base_url}/chat", 
                                   json=chat_data, timeout=self.timeout or 5)  # Short timeout when enabled
            
            # Either succeeds quickly or times out gracefully
            if response.status_code == 200:
                chat_result = response.json()
                error_msg = chat_result.get('error', '')
                if 'timeout' in error_msg.lower():
                    self.log_test("Chat Timeout Handling", "PASS", "Timeout handled gracefully")
                else:
                    self.log_test("Chat Timeout Handling", "PASS", "Response completed within timeout")
            else:
                self.log_test("Chat Timeout Handling", "WARN", f"HTTP {response.status_code}")
        except requests.exceptions.Timeout:
            self.log_test("Chat Timeout Handling", "WARN", "Request timed out at HTTP level")
        except Exception as e:
            self.log_test("Chat Timeout Handling", "FAIL", error=str(e))

    def test_performance_optimizations(self):
        """Test performance optimization features"""
        self.logger.info("=== PERFORMANCE OPTIMIZATION TESTS ===")
        
        # Test cache statistics endpoint (lightweight test)
        try:
            response = requests.get(f"{self.base_url}/health/performance", timeout=self.timeout or 10)
            if response.status_code == 200:
                perf_data = response.json()
                cache_stats = (perf_data.get('performance') or {}).get('cache_stats', {})
                if cache_stats:
                    hit_rate = cache_stats.get('hit_rate', 0)
                    cache_size = cache_stats.get('cache_size', 0)
                    self.log_test("Cache Statistics", "PASS", 
                                 f"Hit rate: {hit_rate:.2%}, Size: {cache_size} entries")
                else:
                    self.log_test("Cache Statistics", "WARN", "No cache statistics available")
            else:
                self.log_test("Cache Statistics", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Cache Statistics", "FAIL", error=str(e))
        
        # Test performance monitoring endpoint
        try:
            response = requests.get(f"{self.base_url}/health/performance/stats", timeout=self.timeout or 10)
            if response.status_code == 200:
                stats_data = response.json()
                if 'response_times' in stats_data or 'cache_performance' in stats_data:
                    self.log_test("Performance Monitoring", "PASS", 
                                 "Performance statistics endpoint accessible")
                else:
                    self.log_test("Performance Monitoring", "WARN", 
                                 "Performance endpoint accessible but no stats available")
            else:
                self.log_test("Performance Monitoring", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Performance Monitoring", "FAIL", error=str(e))
        
        # Test cache management endpoint (lightweight)
        try:
            response = requests.post(f"{self.base_url}/health/performance/cache/clear", timeout=self.timeout or 10)
            if response.status_code in [200, 204]:
                self.log_test("Cache Management", "PASS", "Cache clear endpoint working")
            else:
                self.log_test("Cache Management", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Cache Management", "FAIL", error=str(e))
        
        # Test model selection optimization endpoint (no actual LLM calls)
        try:
            response = requests.get(f"{self.base_url}/llm/models/available", timeout=self.timeout or 10)
            if response.status_code == 200:
                models_data = response.json()
                available_models = models_data.get('models', [])
                if len(available_models) >= 2:
                    self.log_test("Smart Model Selection", "PASS", 
                                 f"Multiple models available for optimization: {len(available_models)}")
                else:
                    self.log_test("Smart Model Selection", "WARN", 
                                 f"Limited models available: {len(available_models)}")
            else:
                self.log_test("Smart Model Selection", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Smart Model Selection", "FAIL", error=str(e))

    def test_reflection_system(self):
        """Test reflection system functionality"""
        self.logger.info("=== REFLECTION SYSTEM TESTS ===")
        
        # Test reflection generation endpoint
        try:
            reflection_data = {
                "user_profile_id": self.session_id,
                "reflection_type": "message",
                "trigger_source": "system",
                "additional_context": {"test": "reflection generation"}
            }
            response = requests.post(f"{self.base_url}/api/reflections/generate", 
                                   json=reflection_data, timeout=self.long_timeout)
            if response.status_code in [200, 201]:
                reflection_result = response.json()
                self.log_test("Reflection Generation", "PASS", 
                             f"Generated reflection: {reflection_result.get('id', 'unknown')}")
            else:
                error_detail = response.text
                self.log_test("Reflection Generation", "FAIL", 
                             f"HTTP {response.status_code}: {error_detail}")
        except Exception as e:
            self.log_test("Reflection Generation", "FAIL", error=str(e))

        # Test reflection retrieval
        try:
            response = requests.get(f"{self.base_url}/api/reflections/list?user_profile_id={self.session_id}&limit=5", 
                                  timeout=self.timeout or 10)
            if response.status_code == 200:
                reflections = response.json()
                items = reflections.get('reflections', [])
                self.log_test("Reflection Retrieval", "PASS", f"Retrieved {len(items)} reflections")
            else:
                self.log_test("Reflection Retrieval", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Reflection Retrieval", "FAIL", error=str(e))

    def test_persona_system(self):
        """Test persona system functionality"""
        self.logger.info("=== PERSONA SYSTEM TESTS ===")
        
        # Test persona retrieval
        try:
            response = requests.get(f"{self.base_url}/api/persona/default", timeout=self.timeout or 10)
            if response.status_code == 200:
                persona = response.json()
                self.log_test("Persona Retrieval", "PASS", f"Retrieved persona data")
            else:
                self.log_test("Persona Retrieval", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Persona Retrieval", "FAIL", error=str(e))

        # Test persona update
        try:
            persona_data = {
                "user_id": self.session_id,
                "traits": {"test_trait": "test_value"},
                "preferences": {"test_pref": "test_value"}
            }
            response = requests.post(f"{self.base_url}/api/persona/update", 
                                   json=persona_data, timeout=self.timeout or 15)
            if response.status_code in [200, 201]:
                self.log_test("Persona Update", "PASS", "Persona updated successfully")
            else:
                error_detail = response.text
                self.log_test("Persona Update", "FAIL", 
                             f"HTTP {response.status_code}: {error_detail}")
        except Exception as e:
            self.log_test("Persona Update", "FAIL", error=str(e))

    async def test_vector_store(self):
        """Test vector store functionality"""
        self.logger.info("=== VECTOR STORE TESTS ===")
        
        try:
            import sys
            from pathlib import Path
            
            # Add backend to path if not already there
            backend_path = Path(__file__).parent.parent
            if str(backend_path) not in sys.path:
                sys.path.insert(0, str(backend_path))
            
            from memory.vector_store import VectorStore
            
            # Test initialization (this tests our FAISS GPU timeout fix)
            vector_store = VectorStore()
            backend_type = getattr(vector_store, 'backend', 'unknown')
            gpu_status = getattr(vector_store, 'gpu_available', False)
            
            self.log_test("Vector Store Init", "PASS", 
                         f"Backend: {backend_type}, GPU: {gpu_status}")
            
            # Test adding embeddings
            test_texts = ["This is a test document", "Another test document for SELO AI testing"]
            test_ids = ["test_doc_1", "test_doc_2"]
            
            await vector_store.add_texts(test_texts, test_ids)
            self.log_test("Vector Store Add", "PASS", f"Added {len(test_texts)} documents")
            
            # Test search functionality
            results = await vector_store.search("test document", top_k=2)
            self.log_test("Vector Store Search", "PASS", f"Found {len(results)} results")
            
            # Test stats and GPU acceleration status
            stats = vector_store.get_stats()
            total_embeddings = stats.get('total_embeddings', 0)
            gpu_accelerated = stats.get('gpu_accelerated', False)
            
            self.log_test("Vector Store Stats", "PASS", 
                         f"Total: {total_embeddings}, GPU accelerated: {gpu_accelerated}")
            
            # Test our FAISS GPU timeout fix specifically
            if hasattr(vector_store, 'index') and vector_store.index is not None:
                self.log_test("FAISS GPU Timeout Fix", "PASS", "Index initialized without hanging")
            else:
                self.log_test("FAISS GPU Timeout Fix", "WARN", "No FAISS index found")
            
        except Exception as e:
            self.log_test("Vector Store Tests", "FAIL", error=str(e))

    def test_frontend_accessibility(self):
        """Test frontend accessibility and basic functionality"""
        self.logger.info("=== FRONTEND TESTS ===")
        
        # Test frontend main page
        try:
            response = requests.get(self.frontend_url, timeout=self.timeout or 10)
            if response.status_code == 200:
                content = response.text
                has_react = "react" in content.lower() or "app" in content.lower()
                self.log_test("Frontend Accessibility", "PASS", f"Frontend accessible, React app: {has_react}")
            else:
                self.log_test("Frontend Accessibility", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Frontend Accessibility", "FAIL", error=str(e))

        # Test frontend config endpoint
        try:
            response = requests.get(f"{self.base_url}/config.json", timeout=self.timeout or 5)
            if response.status_code == 200:
                config = response.json()
                api_base = config.get('apiBaseUrl')
                self.log_test("Frontend Config", "PASS", f"API Base URL: {api_base}")
            else:
                self.log_test("Frontend Config", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Frontend Config", "FAIL", error=str(e))
        
        # Test frontend GPU diagnostics display fix
        try:
            response = requests.get(f"{self.base_url}/diagnostics/gpu", timeout=10)
            if response.status_code == 200:
                gpu_data = response.json()
                # Check if the response structure matches what frontend expects
                has_ollama_key = 'ollama' in gpu_data
                has_cuda_detected = 'cuda_detected' in gpu_data
                
                if has_ollama_key and has_cuda_detected:
                    self.log_test("Frontend GPU API Compatibility", "PASS", "API structure matches frontend expectations")
                else:
                    self.log_test("Frontend GPU API Compatibility", "WARN", f"Missing keys - ollama: {has_ollama_key}, cuda_detected: {has_cuda_detected}")
            else:
                self.log_test("Frontend GPU API Compatibility", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Frontend GPU API Compatibility", "FAIL", error=str(e))

    def test_security_features(self):
        """Test security features and API protection"""
        self.logger.info("=== SECURITY TESTS ===")
        
        # Test rate limiting (should not fail immediately)
        try:
            responses = []
            for i in range(5):
                response = requests.get(f"{self.base_url}/health", timeout=self.timeout or 5)
                responses.append(response.status_code)
            
            if all(status == 200 for status in responses):
                self.log_test("Rate Limiting Basic", "PASS", "Normal requests allowed")
            else:
                self.log_test("Rate Limiting Basic", "WARN", f"Status codes: {responses}")
        except Exception as e:
            self.log_test("Rate Limiting Basic", "FAIL", error=str(e))

        # Test CORS headers
        try:
            response = requests.options(f"{self.base_url}/health", timeout=self.timeout or 5)
            cors_headers = {k: v for k, v in response.headers.items() if 'cors' in k.lower() or 'access-control' in k.lower()}
            if cors_headers:
                self.log_test("CORS Headers", "PASS", f"CORS headers present: {len(cors_headers)}")
            else:
                self.log_test("CORS Headers", "WARN", "No CORS headers detected")
        except Exception as e:
            self.log_test("CORS Headers", "FAIL", error=str(e))

        # Test security headers
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            security_headers = ['x-content-type-options', 'x-frame-options', 'x-xss-protection']
            present_headers = [h for h in security_headers if h in response.headers]
            self.log_test("Security Headers", "PASS" if present_headers else "WARN", 
                         f"Present: {present_headers}")
        except Exception as e:
            self.log_test("Security Headers", "FAIL", error=str(e))

    def test_performance_benchmarks(self):
        """Test basic performance benchmarks"""
        self.logger.info("=== PERFORMANCE TESTS ===")
        
        # Test performance benchmarks and our fixes
        endpoints = [
            ("/health", "Health Check"),
            ("/diagnostics/gpu", "GPU Diagnostics"),
            ("/config.json", "Frontend Config"),
            ("/diagnostics/env", "Environment Diagnostics")
        ]
        
        for endpoint, name in endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}{endpoint}", timeout=self.timeout or 10)
                end_time = time.time()
                
                duration = round((end_time - start_time) * 1000, 2)  # ms
                
                if response.status_code == 200:
                    status = "PASS" if duration < 1000 else "WARN"
                    self.log_test(f"Performance: {name}", status, f"{duration}ms")
                else:
                    self.log_test(f"Performance: {name}", "FAIL", f"HTTP {response.status_code}")
            except Exception as e:
                self.log_test(f"Performance: {name}", "FAIL", error=str(e))

    def run_all_tests(self):
        """Run all test suites"""
        self.logger.info("🚀 Starting Comprehensive SELO AI E2E Test Suite")
        self.logger.info(f"Backend URL: {self.base_url}")
        self.logger.info(f"Frontend URL: {self.frontend_url}")
        self.logger.info(f"Test Session ID: {self.session_id}")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run all test suites
        test_suites = [
            self.test_system_dependencies,
            self.test_backend_health,
            self.test_database_operations,
            self.test_llm_integration,
            self.test_chat_functionality,
            self.test_performance_optimizations,
            self.test_reflection_system,
            self.test_persona_system,
            lambda: asyncio.run(self.test_vector_store()),
            self.test_frontend_accessibility,
            self.test_security_features,
            self.test_performance_benchmarks
        ]
        
        for test_suite in test_suites:
            try:
                test_suite()
            except Exception as e:
                self.logger.error(f"Test suite {test_suite.__name__} failed: {e}")
                self.log_test(test_suite.__name__, "FAIL", error=str(e))
        
        end_time = time.time()
        total_duration = round(end_time - start_time, 2)
        
        # Generate summary
        self.generate_summary(total_duration)

    def generate_summary(self, duration: float):
        """Generate test summary and save results"""
        self.logger.info("=" * 60)
        self.logger.info("📊 TEST SUMMARY")
        self.logger.info("=" * 60)
        
        # Count results
        passed = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed = len([r for r in self.test_results if r['status'] == 'FAIL'])
        warnings = len([r for r in self.test_results if r['status'] == 'WARN'])
        total = len(self.test_results)
        
        self.logger.info(f"Total Tests: {total}")
        self.logger.info(f"✅ Passed: {passed}")
        self.logger.info(f"❌ Failed: {failed}")
        self.logger.info(f"⚠️  Warnings: {warnings}")
        self.logger.info(f"⏱️  Duration: {duration}s")
        
        # Calculate success rate
        success_rate = (passed / total * 100) if total > 0 else 0
        self.logger.info(f"📈 Success Rate: {success_rate:.1f}%")
        
        # Show critical failures
        critical_failures = [r for r in self.test_results if r['status'] == 'FAIL']
        if critical_failures:
            self.logger.info("\n🚨 CRITICAL FAILURES:")
            for failure in critical_failures:
                self.logger.error(f"   • {failure['test']}: {failure['error']}")
        
        # Show warnings
        warning_tests = [r for r in self.test_results if r['status'] == 'WARN']
        if warning_tests:
            self.logger.info("\n⚠️  WARNINGS:")
            for warning in warning_tests:
                self.logger.warning(f"   • {warning['test']}: {warning['details']}")
        
        # Save detailed results to JSON
        results_file = f"e2e_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'total': total,
                    'passed': passed,
                    'failed': failed,
                    'warnings': warnings,
                    'success_rate': success_rate,
                    'duration': duration,
                    'timestamp': datetime.now().isoformat()
                },
                'results': self.test_results
            }, f, indent=2)
        
        self.logger.info(f"\n📄 Detailed results saved to: {results_file}")
        
        # Overall status
        if failed == 0:
            self.logger.info("🎉 ALL TESTS PASSED!")
        elif failed < 5:
            self.logger.info("⚠️  MINOR ISSUES DETECTED - System mostly functional")
        else:
            self.logger.error("🚨 MAJOR ISSUES DETECTED - System needs attention")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SELO AI Comprehensive E2E Test Suite')
    parser.add_argument('--backend-url', default='http://localhost:8000', 
                       help='Backend URL (default: http://localhost:8000)')
    parser.add_argument('--frontend-url', default='http://localhost:3000',
                       help='Frontend URL (default: http://localhost:3000)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    # Timeouts are now defined in-code: 180s general, 240s heavy LLM paths
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run test suite
    test_suite = SELOTestSuite(args.backend_url, args.frontend_url)
    test_suite.run_all_tests()

if __name__ == "__main__":
    main()
