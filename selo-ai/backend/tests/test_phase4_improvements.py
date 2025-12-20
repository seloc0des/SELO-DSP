"""
Test Phase 4 Optional Improvements

Tests for:
1. Lazy embedding processor initialization
2. Manifesto loader dependency injection
3. Circuit breaker metrics tracking
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

# Test 1: Lazy Embedding Processor Initialization
class TestLazyEmbeddingProcessor:
    """Test that embedding processor starts lazily, not during __init__."""
    
    def test_processor_not_started_during_init(self):
        """Verify processor is not started during __init__ (no event loop required)."""
        from backend.reflection.processor import ReflectionProcessor, _embedding_processor_task
        
        # Reset global state
        import backend.reflection.processor as proc_module
        proc_module._embedding_processor_task = None
        
        # Should initialize without errors even without event loop
        processor = ReflectionProcessor(enable_deferred_embeddings=True)
        
        # Processor task should NOT be started during __init__
        assert proc_module._embedding_processor_task is None
        assert processor.enable_deferred_embeddings is True
    
    @pytest.mark.asyncio
    async def test_processor_starts_on_first_queue(self):
        """Verify processor starts lazily on first _queue_embeddings() call."""
        from backend.reflection.processor import ReflectionProcessor
        import backend.reflection.processor as proc_module
        
        # Reset global state
        proc_module._embedding_processor_task = None
        
        # Create processor with mock vector_store
        mock_vector_store = Mock()
        processor = ReflectionProcessor(
            enable_deferred_embeddings=True,
            vector_store=mock_vector_store
        )
        
        # Processor should not be started yet
        assert proc_module._embedding_processor_task is None
        
        # Queue embeddings (first call)
        await processor._queue_embeddings("test_reflection_id", ["theme1", "theme2"])
        
        # Now processor task should be started
        assert proc_module._embedding_processor_task is not None
        assert isinstance(proc_module._embedding_processor_task, asyncio.Task)
        
        # Clean up
        if proc_module._embedding_processor_task and not proc_module._embedding_processor_task.done():
            proc_module._embedding_processor_task.cancel()
            try:
                await proc_module._embedding_processor_task
            except asyncio.CancelledError:
                pass
    
    def test_processor_disabled_when_flag_false(self):
        """Verify processor is not started when enable_deferred_embeddings=False."""
        from backend.reflection.processor import ReflectionProcessor
        import backend.reflection.processor as proc_module
        
        # Reset global state
        proc_module._embedding_processor_task = None
        
        # Create processor with embeddings disabled
        processor = ReflectionProcessor(enable_deferred_embeddings=False)
        
        # Should not start processor
        assert proc_module._embedding_processor_task is None
        assert processor.enable_deferred_embeddings is False


# Test 2: Manifesto Loader Dependency Injection
class TestManifestoLoaderDI:
    """Test dependency injection for manifesto attributes in PersonaEngine."""
    
    def test_default_manifesto_loading(self):
        """Verify default behavior loads from manifesto."""
        from backend.persona.engine import PersonaEngine
        
        # Mock dependencies
        mock_llm = Mock()
        mock_vector_store = Mock()
        
        # Create engine without overrides (should load from manifesto)
        engine = PersonaEngine(
            llm_router=mock_llm,
            vector_store=mock_vector_store
        )
        
        # Should have loaded attributes from manifesto
        assert engine.initial_attributes is not None
        assert len(engine.initial_attributes) > 0
        assert engine.locked_attributes is not None
        assert len(engine.locked_attributes) > 0
        assert len(engine.personality_dimensions) > 0
    
    def test_override_initial_attributes(self):
        """Verify initial_attributes can be overridden for testing."""
        from backend.persona.engine import PersonaEngine
        
        # Mock dependencies
        mock_llm = Mock()
        mock_vector_store = Mock()
        
        # Custom attributes for testing
        test_attributes = [
            {"name": "test_trait_1", "value": 0.5},
            {"name": "test_trait_2", "value": 0.7}
        ]
        test_locked = ["test_trait_1"]
        
        # Create engine with overrides
        engine = PersonaEngine(
            llm_router=mock_llm,
            vector_store=mock_vector_store,
            initial_attributes=test_attributes,
            locked_attributes=test_locked
        )
        
        # Should use provided overrides
        assert engine.initial_attributes == test_attributes
        assert engine.locked_attributes == test_locked
        assert engine.personality_dimensions == ["test_trait_1", "test_trait_2"]
    
    def test_partial_override_loads_missing_from_manifesto(self):
        """Verify partial override loads missing values from manifesto."""
        from backend.persona.engine import PersonaEngine
        
        # Mock dependencies
        mock_llm = Mock()
        mock_vector_store = Mock()
        
        # Override only initial_attributes
        test_attributes = [{"name": "custom_trait", "value": 0.8}]
        
        # Create engine with partial override
        engine = PersonaEngine(
            llm_router=mock_llm,
            vector_store=mock_vector_store,
            initial_attributes=test_attributes,
            locked_attributes=None  # Should load from manifesto
        )
        
        # Should use provided attributes but load locked from manifesto
        assert engine.initial_attributes == test_attributes
        assert engine.locked_attributes is not None  # Loaded from manifesto
        assert len(engine.locked_attributes) > 0


# Test 3: Circuit Breaker Metrics Tracking
class TestCircuitBreakerMetrics:
    """Test comprehensive metrics tracking in circuit breaker."""
    
    def test_initial_metrics(self):
        """Verify initial metrics are zeroed."""
        from backend.core.circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker("test_breaker")
        metrics = breaker.get_metrics()
        
        # Check initial state
        assert metrics["name"] == "test_breaker"
        assert metrics["state"] == "closed"
        assert metrics["total_failures"] == 0
        assert metrics["total_successes"] == 0
        assert metrics["total_opens"] == 0
        assert metrics["total_closes"] == 0
        assert metrics["total_half_opens"] == 0
        assert metrics["failure_rate"] == 0.0
        assert metrics["uptime_seconds"] > 0
    
    @pytest.mark.asyncio
    async def test_success_tracking(self):
        """Verify successes are tracked in metrics."""
        from backend.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("test_breaker", config)
        
        @breaker
        async def test_func():
            return "success"
        
        # Execute successfully 3 times
        for _ in range(3):
            await test_func()
        
        metrics = breaker.get_metrics()
        assert metrics["total_successes"] == 3
        assert metrics["total_failures"] == 0
        assert metrics["failure_rate"] == 0.0
    
    @pytest.mark.asyncio
    async def test_failure_tracking(self):
        """Verify failures are tracked in metrics."""
        from backend.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError
        
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test_breaker", config)
        
        @breaker
        async def test_func():
            raise ValueError("Test error")
        
        # Execute with failures (should open after 2)
        for _ in range(2):
            with pytest.raises(ValueError):
                await test_func()
        
        metrics = breaker.get_metrics()
        assert metrics["total_failures"] == 2
        assert metrics["total_successes"] == 0
        assert metrics["total_opens"] == 1
        assert metrics["state"] == "open"
        assert metrics["failure_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_state_transition_tracking(self):
        """Verify state transitions are tracked."""
        from backend.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,  # Short timeout for testing
            success_threshold=1
        )
        breaker = CircuitBreaker("test_breaker", config)
        
        call_count = 0
        
        @breaker
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("Fail first 2")
            return "success"
        
        # Fail twice to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await test_func()
        
        metrics = breaker.get_metrics()
        assert metrics["total_opens"] == 1
        assert metrics["state"] == "open"
        
        # Wait for recovery timeout
        await asyncio.sleep(0.15)
        
        # Next call should enter half-open
        result = await test_func()
        assert result == "success"
        
        metrics = breaker.get_metrics()
        assert metrics["total_half_opens"] == 1
        assert metrics["total_closes"] == 1
        assert metrics["state"] == "closed"
    
    def test_metrics_calculation(self):
        """Verify metrics calculations are correct."""
        from backend.core.circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker("test_breaker")
        
        # Manually increment metrics for testing
        breaker.metrics["total_successes"] = 7
        breaker.metrics["total_failures"] = 3
        
        metrics = breaker.get_metrics()
        
        # Failure rate should be 3/(3+7) = 0.3
        assert abs(metrics["failure_rate"] - 0.3) < 0.001
        assert metrics["total_successes"] == 7
        assert metrics["total_failures"] == 3
    
    def test_reset_increments_close_counter(self):
        """Verify manual reset increments close counter."""
        from backend.core.circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker("test_breaker")
        
        # Manually set to open state
        breaker.state = breaker.state.__class__.OPEN
        
        initial_closes = breaker.metrics["total_closes"]
        
        # Reset should increment closes
        breaker.reset()
        
        metrics = breaker.get_metrics()
        assert metrics["state"] == "closed"
        assert metrics["total_closes"] == initial_closes + 1


# Test 4: Circuit Breaker Manager Metrics
class TestCircuitBreakerManagerMetrics:
    """Test metrics aggregation in CircuitBreakerManager."""
    
    def test_get_all_metrics(self):
        """Verify manager returns metrics for all breakers."""
        from backend.core.circuit_breaker import CircuitBreakerManager, CircuitBreaker
        
        manager = CircuitBreakerManager()
        
        # Create multiple breakers
        breaker1 = manager.get_breaker("breaker1")
        breaker2 = manager.get_breaker("breaker2")
        
        # Modify metrics
        breaker1.metrics["total_successes"] = 5
        breaker2.metrics["total_failures"] = 3
        
        # Get all metrics
        all_metrics = manager.get_all_metrics()
        
        assert "breaker1" in all_metrics
        assert "breaker2" in all_metrics
        assert all_metrics["breaker1"]["total_successes"] == 5
        assert all_metrics["breaker2"]["total_failures"] == 3


# Test 5: API Endpoint (Integration Test)
class TestCircuitBreakerMetricsEndpoint:
    """Test the new metrics endpoint."""
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint_exists(self):
        """Verify /health/circuit-breakers/metrics endpoint returns data."""
        # This would require a full app integration test
        # For now, we verify the manager method works
        from backend.core.circuit_breaker import circuit_manager
        
        # Should not raise error
        metrics = circuit_manager.get_all_metrics()
        
        # Should return dict
        assert isinstance(metrics, dict)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
