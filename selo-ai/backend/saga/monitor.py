"""
Saga monitoring and recovery service.

Provides monitoring, alerting, and automatic recovery for saga executions.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from .orchestrator import SagaOrchestrator
from ..db.repositories.saga import SagaRepository

logger = logging.getLogger("selo.saga.monitor")


class SagaMonitor:
    """
    Monitors saga executions and provides recovery capabilities.
    
    Features:
    - Detects stuck sagas
    - Automatic retry of failed sagas
    - Health metrics collection
    - Alert generation for failures
    """
    
    def __init__(
        self,
        saga_repo: Optional[SagaRepository] = None,
        orchestrator: Optional[SagaOrchestrator] = None
    ):
        self.saga_repo = saga_repo or SagaRepository()
        self.orchestrator = orchestrator or SagaOrchestrator(self.saga_repo)
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(
        self,
        check_interval_seconds: int = 60,
        stuck_threshold_minutes: int = 30
    ):
        """
        Start background monitoring of sagas.
        
        Args:
            check_interval_seconds: How often to check saga health
            stuck_threshold_minutes: Minutes before considering saga stuck
        """
        if self._monitoring:
            logger.warning("Saga monitoring already running")
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(check_interval_seconds, stuck_threshold_minutes)
        )
        logger.info("Started saga monitoring")
    
    async def stop_monitoring(self):
        """Stop background monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped saga monitoring")
    
    async def _monitor_loop(
        self,
        check_interval: int,
        stuck_threshold_minutes: int
    ):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                await self._check_saga_health(stuck_threshold_minutes)
                await asyncio.sleep(check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in saga monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(check_interval)
    
    async def _check_saga_health(self, stuck_threshold_minutes: int):
        """Check health of all active sagas."""
        active_sagas = await self.saga_repo.list_active_sagas()
        
        if not active_sagas:
            return
        
        logger.debug(f"Checking health of {len(active_sagas)} active sagas")
        
        stuck_threshold = datetime.now(timezone.utc) - timedelta(minutes=stuck_threshold_minutes)
        
        for saga in active_sagas:
            try:
                # Check if saga is stuck
                last_updated = datetime.fromisoformat(saga['last_updated'].replace('Z', '+00:00'))
                
                if last_updated < stuck_threshold:
                    logger.warning(
                        f"Saga {saga['id']} appears stuck (last updated {last_updated})"
                    )
                    await self._handle_stuck_saga(saga)
            
            except Exception as e:
                logger.error(f"Error checking saga {saga['id']}: {e}")
    
    async def _handle_stuck_saga(self, saga: Dict[str, Any]):
        """
        Handle a stuck saga.
        
        Args:
            saga: Stuck saga data
        """
        saga_id = saga['id']
        
        # Check if saga has exceeded max retries
        if saga['retry_count'] >= saga['max_retries']:
            logger.error(
                f"Saga {saga_id} is stuck and has exceeded max retries. "
                f"Manual intervention required."
            )
            # Could trigger alert here
            return
        
        # Attempt automatic retry
        logger.info(f"Attempting automatic retry of stuck saga {saga_id}")
        
        try:
            await self.orchestrator.retry_failed_saga(saga_id)
            logger.info(f"Successfully retried stuck saga {saga_id}")
        except Exception as e:
            logger.error(f"Failed to retry stuck saga {saga_id}: {e}")
    
    async def get_saga_metrics(
        self,
        saga_type: Optional[str] = None,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get saga execution metrics.
        
        Args:
            saga_type: Optional filter by saga type
            time_window_hours: Time window for metrics
            
        Returns:
            Metrics dictionary
        """
        # Get all sagas in time window
        # This would need additional repository methods for time-based queries
        
        active = await self.saga_repo.list_active_sagas(saga_type=saga_type)
        failed = await self.saga_repo.list_failed_sagas(saga_type=saga_type)
        
        metrics = {
            "saga_type": saga_type or "all",
            "time_window_hours": time_window_hours,
            "active_count": len(active),
            "failed_count": len(failed),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Calculate success rate if we have historical data
        if active or failed:
            total = len(active) + len(failed)
            metrics["failure_rate"] = len(failed) / total if total > 0 else 0.0
        
        # Group failures by error type
        error_types = {}
        for saga in failed:
            error_data = saga.get('error_data', {})
            error_type = error_data.get('type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        metrics["error_types"] = error_types
        
        return metrics
    
    async def recover_failed_sagas(
        self,
        saga_type: Optional[str] = None,
        max_to_recover: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Attempt to recover failed sagas.
        
        Args:
            saga_type: Optional filter by saga type
            max_to_recover: Maximum number to attempt recovery
            
        Returns:
            List of recovery results
        """
        failed_sagas = await self.saga_repo.list_failed_sagas(
            saga_type=saga_type,
            limit=max_to_recover
        )
        
        if not failed_sagas:
            logger.info("No failed sagas to recover")
            return []
        
        logger.info(f"Attempting to recover {len(failed_sagas)} failed sagas")
        
        results = []
        
        for saga in failed_sagas:
            saga_id = saga['id']
            
            # Skip if already exceeded max retries
            if saga['retry_count'] >= saga['max_retries']:
                logger.debug(f"Skipping saga {saga_id} - max retries exceeded")
                results.append({
                    "saga_id": saga_id,
                    "status": "skipped",
                    "reason": "max_retries_exceeded"
                })
                continue
            
            try:
                result = await self.orchestrator.retry_failed_saga(saga_id)
                results.append({
                    "saga_id": saga_id,
                    "status": "retried",
                    "result": result['status']
                })
                logger.info(f"Retried saga {saga_id}, result: {result['status']}")
            
            except Exception as e:
                logger.error(f"Failed to retry saga {saga_id}: {e}")
                results.append({
                    "saga_id": saga_id,
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    async def get_saga_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive saga health report.
        
        Returns:
            Health report with metrics and recommendations
        """
        active = await self.saga_repo.list_active_sagas()
        failed = await self.saga_repo.list_failed_sagas()
        
        # Calculate health metrics
        total_sagas = len(active) + len(failed)
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_sagas": total_sagas,
            "active_sagas": len(active),
            "failed_sagas": len(failed),
            "health_status": "healthy",
            "issues": [],
            "recommendations": []
        }
        
        # Check for issues
        if len(failed) > 0:
            failure_rate = len(failed) / total_sagas if total_sagas > 0 else 0
            
            if failure_rate > 0.2:
                report["health_status"] = "critical"
                report["issues"].append(
                    f"High failure rate: {failure_rate:.1%} of sagas failed"
                )
                report["recommendations"].append(
                    "Investigate common failure patterns and consider system health check"
                )
            elif failure_rate > 0.1:
                report["health_status"] = "warning"
                report["issues"].append(
                    f"Elevated failure rate: {failure_rate:.1%} of sagas failed"
                )
                report["recommendations"].append(
                    "Monitor saga failures and review error logs"
                )
        
        # Check for stuck sagas
        stuck_threshold = datetime.now(timezone.utc) - timedelta(minutes=30)
        stuck_count = 0
        
        for saga in active:
            last_updated = datetime.fromisoformat(saga['last_updated'].replace('Z', '+00:00'))
            if last_updated < stuck_threshold:
                stuck_count += 1
        
        if stuck_count > 0:
            report["issues"].append(f"{stuck_count} sagas appear stuck")
            report["recommendations"].append(
                "Review stuck sagas and consider manual intervention or retry"
            )
            if report["health_status"] == "healthy":
                report["health_status"] = "warning"
        
        # Group sagas by type
        saga_types = {}
        for saga in active + failed:
            saga_type = saga['saga_type']
            if saga_type not in saga_types:
                saga_types[saga_type] = {"active": 0, "failed": 0}
            
            if saga in active:
                saga_types[saga_type]["active"] += 1
            else:
                saga_types[saga_type]["failed"] += 1
        
        report["saga_types"] = saga_types
        
        return report
