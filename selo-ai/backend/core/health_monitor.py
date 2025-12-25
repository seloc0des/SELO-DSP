"""
Structured Health Monitoring System for SELO AI

Provides comprehensive health checks, metrics collection, and system status monitoring.
Integrates with circuit breakers and provides early warning for system issues.
"""

import asyncio
import logging
import os
import time
import psutil
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque

from ..utils.datetime import utc_now, isoformat_utc, ensure_utc

logger = logging.getLogger("selo.health.monitor")

class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['status'] = self.status.value
        result['timestamp'] = isoformat_utc(self.timestamp)
        return result

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_connections: int
    process_count: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = isoformat_utc(self.timestamp)
        return result

class HealthMonitor:
    """
    Comprehensive health monitoring system.
    
    Monitors:
    - Database connectivity
    - LLM service availability
    - System resources
    - API endpoint health
    - Circuit breaker states
    """
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = self._resolve_check_interval(check_interval)
        self.health_checks: Dict[str, HealthCheck] = {}
        self.metrics_history: deque = deque(maxlen=100)  # Keep last 100 metrics
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.lock = threading.RLock()
        self.llm_controller = None
        self._fallback_llm_controller = None
        
        # Health check functions
        self.check_functions: Dict[str, Callable] = {}
        
        # Register default health checks
        self._register_default_checks()
        
    def _resolve_check_interval(self, default: int) -> int:
        env_value = os.getenv("HEALTH_MONITOR_INTERVAL")
        if env_value is not None:
            try:
                resolved = int(float(env_value))
            except Exception:
                resolved = default
            return max(5, resolved)
        return default

    def set_llm_controller(self, controller):
        self.llm_controller = controller

    def _register_default_checks(self):
        """Register default health check functions."""
        self.register_check("database", self._check_database_health)
        self.register_check("llm_service", self._check_llm_health)
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("circuit_breakers", self._check_circuit_breakers)
        self.register_check("memory_usage", self._check_memory_health)
        
    def register_check(self, name: str, check_func: Callable):
        """Register a custom health check function."""
        self.check_functions[name] = check_func
        logger.info(f"🏥 Registered health check: {name}")
        
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            logger.warning("Health monitoring already running")
            return
            
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("🏥 Health monitoring started")
        
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("🏥 Health monitoring stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await self.run_all_checks()
                await self._collect_system_metrics()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(2)  # Short delay on error (reduced from 5s)
                
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        # Run checks concurrently
        tasks = []
        for name, check_func in self.check_functions.items():
            task = asyncio.create_task(self._run_single_check(name, check_func))
            tasks.append(task)
            
        # Wait for all checks to complete
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(check_results):
            check_name = list(self.check_functions.keys())[i]
            if isinstance(result, Exception):
                results[check_name] = HealthCheck(
                    name=check_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Check failed: {str(result)}",
                    timestamp=utc_now(),
                    response_time_ms=0.0
                )
            else:
                results[check_name] = result
                
        # Update stored results
        with self.lock:
            self.health_checks.update(results)
            
        return results
        
    async def _run_single_check(self, name: str, check_func: Callable) -> HealthCheck:
        """Run a single health check with timing."""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
                
            response_time = (time.time() - start_time) * 1000
            
            if isinstance(result, HealthCheck):
                result.response_time_ms = response_time
                return result
            else:
                # Convert simple result to HealthCheck
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.CRITICAL,
                    message="Check completed",
                    timestamp=utc_now(),
                    response_time_ms=response_time
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Check failed: {str(e)}",
                timestamp=utc_now(),
                response_time_ms=response_time
            )
            
    async def _check_database_health(self) -> HealthCheck:
        """Check database connectivity and performance."""
        try:
            from ..db.connection_pool import get_connection_pool
            
            pool = get_connection_pool()
            start_time = time.time()
            
            # Test connection with simple query
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                
            response_time = (time.time() - start_time) * 1000
            
            # Check pool status
            pool_size = pool.get_size()
            pool_free = pool.get_idle_size()
            
            if pool_free == 0:
                status = HealthStatus.WARNING
                message = f"Database pool exhausted (0/{pool_size} free)"
            elif response_time > 1000:
                status = HealthStatus.WARNING
                message = f"Slow database response ({response_time:.1f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Database healthy ({pool_free}/{pool_size} free, {response_time:.1f}ms)"
                
            return HealthCheck(
                name="database",
                status=status,
                message=message,
                timestamp=utc_now(),
                response_time_ms=response_time,
                details={
                    "pool_size": pool_size,
                    "pool_free": pool_free,
                    "response_time_ms": response_time
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {str(e)}",
                timestamp=utc_now(),
                response_time_ms=0.0
            )
            
    async def _check_llm_health(self) -> HealthCheck:
        """Check LLM service availability."""
        try:
            from ..llm.controller import LLMController
            
            controller = self.llm_controller
            if controller is None:
                if self._fallback_llm_controller is None:
                    self._fallback_llm_controller = LLMController()
                controller = self._fallback_llm_controller
            start_time = time.time()
            
            # Test model availability
            available_models = await controller.get_available_models()
            response_time = (time.time() - start_time) * 1000
            
            if not available_models:
                status = HealthStatus.CRITICAL
                message = "No LLM models available"
            elif response_time > 5000:
                status = HealthStatus.WARNING
                message = f"Slow LLM response ({response_time:.1f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = f"LLM service healthy ({len(available_models)} models)"
                
            return HealthCheck(
                name="llm_service",
                status=status,
                message=message,
                timestamp=utc_now(),
                response_time_ms=response_time,
                details={
                    "available_models": available_models,
                    "model_count": len(available_models)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="llm_service",
                status=HealthStatus.CRITICAL,
                message=f"LLM service check failed: {str(e)}",
                timestamp=utc_now(),
                response_time_ms=0.0
            )
            
    def _check_system_resources(self) -> HealthCheck:
        """Check system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on resource usage
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                status = HealthStatus.CRITICAL
                message = "Critical resource usage"
            elif cpu_percent > 75 or memory.percent > 75 or disk.percent > 80:
                status = HealthStatus.WARNING
                message = "High resource usage"
            else:
                status = HealthStatus.HEALTHY
                message = "Resource usage normal"
                
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                timestamp=utc_now(),
                response_time_ms=0.0,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Resource check failed: {str(e)}",
                timestamp=utc_now(),
                response_time_ms=0.0
            )
            
    def _check_circuit_breakers(self) -> HealthCheck:
        """Check circuit breaker states."""
        try:
            from .circuit_breaker import circuit_manager
            
            breaker_states = circuit_manager.get_all_states()
            
            open_breakers = [name for name, state in breaker_states.items() 
                           if state['state'] == 'open']
            half_open_breakers = [name for name, state in breaker_states.items() 
                                if state['state'] == 'half_open']
            
            if open_breakers:
                status = HealthStatus.CRITICAL
                message = f"Circuit breakers open: {', '.join(open_breakers)}"
            elif half_open_breakers:
                status = HealthStatus.WARNING
                message = f"Circuit breakers recovering: {', '.join(half_open_breakers)}"
            else:
                status = HealthStatus.HEALTHY
                message = f"All circuit breakers closed ({len(breaker_states)} total)"
                
            return HealthCheck(
                name="circuit_breakers",
                status=status,
                message=message,
                timestamp=utc_now(),
                response_time_ms=0.0,
                details={
                    "breaker_states": breaker_states,
                    "open_count": len(open_breakers),
                    "half_open_count": len(half_open_breakers)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="circuit_breakers",
                status=HealthStatus.WARNING,
                message=f"Circuit breaker check failed: {str(e)}",
                timestamp=utc_now(),
                response_time_ms=0.0
            )
            
    def _check_memory_health(self) -> HealthCheck:
        """Check memory usage and potential leaks."""
        try:
            import gc
            
            # Force garbage collection
            gc.collect()
            
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Check for memory growth patterns
            memory_mb = process_memory.rss / (1024 * 1024)
            
            if memory_mb > 2048:  # 2GB
                status = HealthStatus.WARNING
                message = f"High process memory usage: {memory_mb:.1f}MB"
            elif memory_mb > 4096:  # 4GB
                status = HealthStatus.CRITICAL
                message = f"Critical process memory usage: {memory_mb:.1f}MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_mb:.1f}MB"
                
            return HealthCheck(
                name="memory_usage",
                status=status,
                message=message,
                timestamp=utc_now(),
                response_time_ms=0.0,
                details={
                    "process_memory_mb": memory_mb,
                    "system_memory_percent": memory.percent,
                    "gc_objects": len(gc.get_objects())
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="memory_usage",
                status=HealthStatus.WARNING,
                message=f"Memory check failed: {str(e)}",
                timestamp=utc_now(),
                response_time_ms=0.0
            )
            
    async def _collect_system_metrics(self):
        """Collect and store system performance metrics."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_connections = len(psutil.net_connections())
            
            metric = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024**3),
                network_connections=net_connections,
                process_count=len(psutil.pids()),
                timestamp=utc_now()
            )
            
            with self.lock:
                self.metrics_history.append(metric)
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        with self.lock:
            if not self.health_checks:
                return {
                    "status": "unknown",
                    "message": "No health checks available",
                    "timestamp": isoformat_utc(utc_now())
                }
                
            # Determine overall status
            statuses = [check.status for check in self.health_checks.values()]
            
            if HealthStatus.CRITICAL in statuses:
                overall_status = "critical"
            elif HealthStatus.WARNING in statuses:
                overall_status = "warning"
            elif HealthStatus.UNKNOWN in statuses:
                overall_status = "unknown"
            else:
                overall_status = "healthy"
                
            # Count status types
            status_counts = {}
            for status in HealthStatus:
                status_counts[status.value] = sum(1 for s in statuses if s == status)
                
            return {
                "status": overall_status,
                "timestamp": isoformat_utc(utc_now()),
                "checks": {name: check.to_dict() for name, check in self.health_checks.items()},
                "status_counts": status_counts,
                "total_checks": len(self.health_checks)
            }
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get system metrics summary."""
        with self.lock:
            if not self.metrics_history:
                return {"message": "No metrics available"}
                
            latest = self.metrics_history[-1]
            
            # Calculate averages over last 10 minutes
            now = utc_now()
            recent_metrics = [
                m for m in self.metrics_history
                if (now - ensure_utc(m.timestamp)).total_seconds() < 600
            ]
            
            if recent_metrics:
                avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
                avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            else:
                avg_cpu = latest.cpu_percent
                avg_memory = latest.memory_percent
                
            return {
                "current": latest.to_dict(),
                "averages_10min": {
                    "cpu_percent": round(avg_cpu, 2),
                    "memory_percent": round(avg_memory, 2)
                },
                "history_count": len(self.metrics_history)
            }

# Global health monitor instance
health_monitor = HealthMonitor()

async def get_system_health() -> Dict[str, Any]:
    """Get current system health status."""
    return health_monitor.get_health_summary()

async def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics."""
    return health_monitor.get_metrics_summary()
