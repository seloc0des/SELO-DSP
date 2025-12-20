"""
Adaptive Scheduler Module

This module implements an intelligent scheduler that adapts job frequency
based on system usage patterns, event importance, and resource availability.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
import statistics

from .scheduler_service import SchedulerService
from ..utils.datetime import utc_now, ensure_utc, add_seconds

logger = logging.getLogger("selo.scheduler.adaptive")

class AdaptiveScheduler:
    """
    Adaptive Scheduler for intelligent scheduling adjustments.
    
    This class extends the basic scheduler functionality by adjusting schedules
    based on usage patterns, event significance, and resource availability.
    """
    
    def __init__(self, 
                 scheduler_service: SchedulerService,
                 config: Dict[str, Any] = None):
        """
        Initialize the adaptive scheduler.
        
        Args:
            scheduler_service: Base scheduler service
            config: Configuration options
        """
        self.scheduler_service = scheduler_service
        self.config = config or {}
        
        # Default configuration
        self.min_interval = self.config.get("min_interval_seconds", 300)  # 5 minutes
        self.max_interval = self.config.get("max_interval_seconds", 86400 * 2)  # 2 days
        self.default_interval = self.config.get("default_interval_seconds", 86400)  # 1 day
        
        # Adaptive metrics
        self.user_activity_history = {}
        self.job_importance_scores = {}
        self.resource_usage_history = []
        self.adaptive_schedules = {}
        
        # Adaptation parameters
        self.adaptation_rate = self.config.get("adaptation_rate", 0.2)  # How quickly to adapt
        self.activity_weight = self.config.get("activity_weight", 0.6)
        self.importance_weight = self.config.get("importance_weight", 0.3)
        self.resource_weight = self.config.get("resource_weight", 0.1)
        
        logger.info("Adaptive scheduler initialized")
        
    async def register_job(self, 
                          job_id: str, 
                          job_type: str,
                          user_id: Optional[str] = None, 
                          initial_interval: int = None,
                          importance: float = 0.5):
        """
        Register a job for adaptive scheduling.
        
        Args:
            job_id: ID of the job
            job_type: Type of job (reflection, learning, etc.)
            user_id: Associated user ID if applicable
            initial_interval: Starting interval in seconds
            importance: Initial importance score (0-1)
            
        Returns:
            True if registered successfully
        """
        if not initial_interval:
            initial_interval = self.default_interval
            
        self.adaptive_schedules[job_id] = {
            "job_id": job_id,
            "job_type": job_type,
            "user_id": user_id,
            "current_interval": initial_interval,
            "next_run": add_seconds(None, initial_interval),
            "last_run": None,
            "runs": 0,
            "importance": importance
        }
        
        self.job_importance_scores[job_id] = importance
        
        logger.info(f"Registered adaptive job {job_id} of type {job_type} with initial interval {initial_interval}s")
        return True
        
    async def update_user_activity(self, user_id: str, activity_level: float):
        """
        Update user activity metrics.
        
        Args:
            user_id: User ID
            activity_level: Activity level (0-1)
            
        This affects scheduling of user-related jobs.
        """
        if user_id not in self.user_activity_history:
            self.user_activity_history[user_id] = []
            
        self.user_activity_history[user_id].append({
            "timestamp": utc_now(),
            "level": activity_level
        })
        
        # Limit history size
        if len(self.user_activity_history[user_id]) > 100:
            self.user_activity_history[user_id] = self.user_activity_history[user_id][-100:]
            
        # Adjust schedules based on new activity
        await self._adapt_schedules_for_user(user_id)
        
    async def update_resource_usage(self, cpu_percent: float, memory_percent: float):
        """
        Update resource usage metrics.
        
        Args:
            cpu_percent: CPU usage percentage (0-100)
            memory_percent: Memory usage percentage (0-100)
        """
        self.resource_usage_history.append({
            "timestamp": utc_now(),
            "cpu": cpu_percent,
            "memory": memory_percent
        })
        
        # Limit history size
        if len(self.resource_usage_history) > 100:
            self.resource_usage_history = self.resource_usage_history[-100:]
            
        # Adapt to resource constraints if necessary
        high_usage = cpu_percent > 80 or memory_percent > 80
        if high_usage:
            await self._adapt_for_resource_constraints()
            
    async def update_job_importance(self, job_id: str, importance: float):
        """
        Update the importance score of a job.
        
        Args:
            job_id: Job ID
            importance: New importance score (0-1)
        """
        if job_id not in self.job_importance_scores:
            logger.warning(f"Job {job_id} not registered for adaptive scheduling")
            return False
            
        self.job_importance_scores[job_id] = importance
        
        if job_id in self.adaptive_schedules:
            self.adaptive_schedules[job_id]["importance"] = importance
            
        # Adapt based on new importance
        await self._adapt_schedule(job_id)
        return True
        
    async def get_next_run_time(self, job_id: str) -> Optional[datetime]:
        """
        Get the next time a job should run.
        
        Args:
            job_id: Job ID
            
        Returns:
            Datetime of next run
        """
        if job_id not in self.adaptive_schedules:
            logger.warning(f"Job {job_id} not registered for adaptive scheduling")
            return None
            
        return self.adaptive_schedules[job_id]["next_run"]
        
    async def record_job_run(self, job_id: str, success: bool = True):
        """
        Record that a job has been run.
        
        Args:
            job_id: Job ID
            success: Whether the job ran successfully
        """
        if job_id not in self.adaptive_schedules:
            logger.warning(f"Job {job_id} not registered for adaptive scheduling")
            return
            
        job_info = self.adaptive_schedules[job_id]
        job_info["last_run"] = utc_now()
        job_info["runs"] += 1
        
        # Calculate next run based on current interval
        job_info["next_run"] = add_seconds(None, job_info["current_interval"])
        
        # Adapt the schedule after each run
        await self._adapt_schedule(job_id)
        
    async def _adapt_schedules_for_user(self, user_id: str):
        """
        Adapt all schedules associated with a user.
        
        Args:
            user_id: User ID
        """
        for job_id, job_info in self.adaptive_schedules.items():
            if job_info["user_id"] == user_id:
                await self._adapt_schedule(job_id)
                
    async def _adapt_for_resource_constraints(self):
        """Spread out jobs when system resources are constrained."""
        # Sort jobs by importance
        jobs_by_importance = sorted(
            self.adaptive_schedules.items(),
            key=lambda x: x[1]["importance"],
            reverse=True
        )
        
        # Adapt intervals - less important jobs get longer intervals
        for i, (job_id, _) in enumerate(jobs_by_importance):
            stretch_factor = 1.0 + (i / len(jobs_by_importance))
            await self._stretch_job_interval(job_id, stretch_factor)
            
    async def _stretch_job_interval(self, job_id: str, factor: float):
        """
        Stretch a job's interval by a factor.
        
        Args:
            job_id: Job ID
            factor: Stretch factor (>1.0)
        """
        if job_id not in self.adaptive_schedules:
            return
            
        job_info = self.adaptive_schedules[job_id]
        new_interval = min(
            job_info["current_interval"] * factor,
            self.max_interval
        )
        
        # Only update if significantly different
        if abs(new_interval - job_info["current_interval"]) > (job_info["current_interval"] * 0.1):
            job_info["current_interval"] = new_interval
            job_info["next_run"] = add_seconds(None, new_interval)
            logger.info(f"Stretched job {job_id} interval to {new_interval}s due to resource constraints")
            
    async def _adapt_schedule(self, job_id: str):
        """
        Adapt a job's schedule based on all factors.
        
        Args:
            job_id: Job ID
        """
        if job_id not in self.adaptive_schedules:
            return
            
        job_info = self.adaptive_schedules[job_id]
        user_id = job_info["user_id"]
        
        # Calculate adaptation factors
        activity_factor = await self._calculate_activity_factor(user_id)
        importance_factor = job_info["importance"]
        resource_factor = await self._calculate_resource_factor()
        
        # Compute weighted adaptation factor
        # Lower value = run more frequently
        adaptation_factor = (
            (activity_factor * self.activity_weight) +
            ((1 - importance_factor) * self.importance_weight) +
            (resource_factor * self.resource_weight)
        )
        
        # Map to interval range
        interval_range = self.max_interval - self.min_interval
        new_interval = self.min_interval + (interval_range * adaptation_factor)
        
        # Apply gradual adaptation
        current = job_info["current_interval"]
        adapted = current + ((new_interval - current) * self.adaptation_rate)
        
        # Update the job schedule
        job_info["current_interval"] = adapted
        job_info["next_run"] = add_seconds(job_info["last_run"], adapted) if job_info["last_run"] else \
                               add_seconds(None, adapted)
                               
        logger.debug(f"Adapted job {job_id} interval to {adapted:.1f}s (activity={activity_factor:.2f}, "
                    f"importance={importance_factor:.2f}, resource={resource_factor:.2f})")
        
    async def _calculate_activity_factor(self, user_id: Optional[str]) -> float:
        """
        Calculate activity factor based on user activity.
        
        Args:
            user_id: User ID
            
        Returns:
            Activity factor (0-1, lower = more active)
        """
        if not user_id or user_id not in self.user_activity_history:
            return 0.5  # Default middle value
            
        # Get recent activity (last 24 hours)
        cutoff = utc_now() - timedelta(hours=24)
        recent = [
            entry["level"] for entry in self.user_activity_history[user_id]
            if ensure_utc(entry["timestamp"]) >= cutoff
        ]
        
        if not recent:
            return 0.5
            
        # Calculate weighted average, more recent = higher weight
        weighted_sum = 0
        weight_sum = 0
        for i, level in enumerate(recent):
            weight = i + 1  # More recent = higher weight
            weighted_sum += level * weight
            weight_sum += weight
            
        avg_activity = weighted_sum / weight_sum if weight_sum > 0 else 0.5
        
        # Invert so higher activity = lower factor = more frequent jobs
        return 1.0 - avg_activity
        
    async def _calculate_resource_factor(self) -> float:
        """
        Calculate resource factor based on system load.
        
        Returns:
            Resource factor (0-1, higher = more constrained)
        """
        if not self.resource_usage_history:
            return 0.2  # Default to low constraint
            
        # Get recent history (last hour)
        cutoff = utc_now() - timedelta(hours=1)
        recent = [
            (entry["cpu"] / 100.0, entry["memory"] / 100.0)  # Normalize to 0-1
            for entry in self.resource_usage_history
            if ensure_utc(entry["timestamp"]) >= cutoff
        ]
        
        if not recent:
            return 0.2
            
        # Calculate 75th percentile (more conservative than average)
        cpu_values = [cpu for cpu, _ in recent]
        mem_values = [mem for _, mem in recent]
        
        try:
            cpu_factor = statistics.quantiles(cpu_values, n=4)[2] if cpu_values else 0.2
            mem_factor = statistics.quantiles(mem_values, n=4)[2] if mem_values else 0.2
        except statistics.StatisticsError:
            # Not enough data points
            if cpu_values:
                cpu_factor = max(cpu_values)
            else:
                cpu_factor = 0.2
                
            if mem_values:
                mem_factor = max(mem_values)
            else:
                mem_factor = 0.2
        
        # Use the higher of CPU and memory constraint
        return max(cpu_factor, mem_factor)
