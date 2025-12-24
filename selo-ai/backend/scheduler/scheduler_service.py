"""
Scheduler Service Module

This module implements a centralized job scheduling service based on APScheduler.
It provides a unified interface for all scheduled tasks in the application.
"""

import logging
from typing import Dict, List, Any, Callable, Optional, Union
import time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.jobstores.memory import MemoryJobStore
import pytz
import os

logger = logging.getLogger("selo.scheduler.service")

class SchedulerService:
    """
    Central scheduler service for managing all scheduled jobs.
    
    This class provides a unified interface for scheduling jobs across the application,
    with support for persistent job storage, resource management, and job prioritization.
    """
    
    def __init__(self, db_url: Optional[str] = None, config: Dict[str, Any] = None):
        """
        Initialize the scheduler service.
        
        Args:
            db_url: Database URL for job persistence
            config: Configuration options
        """
        self.config = config or {}
        self.jobs = {}
        self.active = False
        self.db_url = db_url or os.environ.get("DATABASE_URL", None)

        # Track whether we have a persistent job store and a dedicated volatile store
        self.has_persistent_jobstore = False
        self.has_volatile_jobstore = False

        # Initialize job stores
        jobstores: Dict[str, Any] = {}
        if self.db_url:
            # Convert async driver URLs (e.g., postgresql+asyncpg) to a sync driver for APScheduler
            # APScheduler's SQLAlchemyJobStore uses sync SQLAlchemy engines.
            sync_url = self.db_url
            try:
                if "+asyncpg" in sync_url:
                    sync_url = sync_url.replace("+asyncpg", "+psycopg2")
                # Also handle common async sqlite URLs if ever used
                if sync_url.startswith("sqlite+aiosqlite"):
                    sync_url = sync_url.replace("sqlite+aiosqlite", "sqlite")
            except Exception:
                # Best effort only; keep original
                sync_url = self.db_url

            try:
                # Log which driver/dialect will be used without leaking credentials
                try:
                    url_driver = sync_url.split(":", 1)[0]  # e.g., 'postgresql+psycopg2' or 'sqlite'
                except Exception:
                    url_driver = "unknown"
                logger.info(f"APScheduler job store using SQLAlchemy URL driver: {url_driver}")
                # Default store is persistent SQLAlchemy-backed
                jobstores["default"] = SQLAlchemyJobStore(url=sync_url)
                self.has_persistent_jobstore = True
                # Add a dedicated in-memory store for non-serializable jobs (closures, bound methods)
                jobstores["volatile"] = MemoryJobStore()
                self.has_volatile_jobstore = True
                logger.info("Using SQLAlchemy job store for persistence with a volatile in-memory job store for non-serializable tasks")
            except Exception as e:
                logger.error(f"Failed to initialize SQLAlchemy job store: {str(e)}")
                logger.info("Falling back to in-memory job store only")
                jobstores = {}

        # Initialize scheduler with job stores (empty dict lets APScheduler create a default in-memory store)
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            timezone=pytz.UTC,
            job_defaults={
                "coalesce": True,  # Combine missed executions into a single execution
                "max_instances": 1  # Only allow one instance of each job to run at a time
            }
        )

        # Job execution metrics for adaptive scheduling
        self.job_metrics: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Scheduler service initialized")
        
    async def start(self):
        """Start the scheduler service."""
        if self.active:
            logger.warning("Scheduler already running")
            return
            
        try:
            self.scheduler.start()
            self.active = True
            logger.info("Scheduler service started")
        except Exception as e:
            logger.error(f"Failed to start scheduler: {str(e)}", exc_info=True)
            
    async def stop(self):
        """Stop the scheduler service."""
        if not self.active:
            logger.warning("Scheduler not running")
            return
            
        try:
            self.scheduler.shutdown()
            self.active = False
            logger.info("Scheduler service stopped")
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {str(e)}", exc_info=True)
            
    async def add_job(self, 
                     job_id: str, 
                     func: Union[str, Callable], 
                     trigger: Union[str, Any] = None,
                     **trigger_args):
        """
        Add a job to the scheduler.
        
        Args:
            job_id: Unique identifier for the job
            func: Function to execute
            trigger: Trigger type ('date', 'interval', 'cron') or an APScheduler trigger instance
            **trigger_args: Arguments for the trigger
            
        Returns:
            Job ID if successful, None if failed
        """
        if job_id in self.jobs:
            logger.warning(f"Job {job_id} already exists, removing existing job")
            await self.remove_job(job_id)
            
        try:
            # Build kwargs, avoiding duplicate replace_existing if provided by caller
            add_job_kwargs: Dict[str, Any] = {
                "id": job_id,
            }
            if "replace_existing" not in trigger_args:
                add_job_kwargs["replace_existing"] = True

            # For non-serializable callables (closures, bound methods), route the job to the
            # volatile in-memory job store when a persistent SQLAlchemy job store is present.
            # This avoids SQLAlchemyJobStore having to pickle nested callables like
            # SchedulerService.add_job.<locals>.wrapped_func, which is what caused the
            # agent_loop_runner serialization error.
            if not isinstance(func, str) and self.has_persistent_jobstore and self.has_volatile_jobstore:
                if "jobstore" not in trigger_args:
                    add_job_kwargs["jobstore"] = "volatile"

            # If func is provided as a string (module:function), pass through directly
            # so APScheduler/SQLAlchemyJobStore can serialize/import it.
            if isinstance(func, str):
                job = self.scheduler.add_job(
                    func,
                    trigger,
                    **add_job_kwargs,
                    **trigger_args,
                )
            else:
                # Wrap the function to capture execution metrics
                async def wrapped_func(*args, **kwargs):
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        execution_time = time.time() - start_time
                        success = True
                    except Exception as e:
                        execution_time = time.time() - start_time
                        logger.error(f"Error executing job {job_id}: {str(e)}", exc_info=True)
                        success = False
                        result = None

                    # Record metrics
                    if job_id not in self.job_metrics:
                        self.job_metrics[job_id] = {"executions": 0, "success": 0, "failures": 0, "avg_time": 0}

                    metrics = self.job_metrics[job_id]
                    metrics["executions"] += 1
                    if success:
                        metrics["success"] += 1
                    else:
                        metrics["failures"] += 1

                    # Update average execution time with exponential moving average
                    if metrics["executions"] == 1:
                        metrics["avg_time"] = execution_time
                    else:
                        metrics["avg_time"] = (metrics["avg_time"] * 0.8) + (execution_time * 0.2)

                    return result

                job = self.scheduler.add_job(
                    wrapped_func,
                    trigger,
                    **add_job_kwargs,
                    **trigger_args,
                )
            
            self.jobs[job_id] = job
            logger.info(f"Added job {job_id} with trigger {trigger}")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to add job {job_id}: {str(e)}", exc_info=True)
            return None
            
    async def remove_job(self, job_id: str) -> bool:
        """
        Remove a job from the scheduler.
        
        Args:
            job_id: ID of the job to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        try:
            self.scheduler.remove_job(job_id)
            if job_id in self.jobs:
                del self.jobs[job_id]
            logger.info(f"Removed job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove job {job_id}: {str(e)}", exc_info=True)
            return False
            
    async def pause_job(self, job_id: str) -> bool:
        """
        Pause a job temporarily.
        
        Args:
            job_id: ID of the job to pause
            
        Returns:
            True if paused successfully, False otherwise
        """
        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"Paused job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause job {job_id}: {str(e)}", exc_info=True)
            return False
            
    async def resume_job(self, job_id: str) -> bool:
        """
        Resume a paused job.
        
        Args:
            job_id: ID of the job to resume
            
        Returns:
            True if resumed successfully, False otherwise
        """
        try:
            self.scheduler.resume_job(job_id)
            logger.info(f"Resumed job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume job {job_id}: {str(e)}", exc_info=True)
            return False
            
    async def get_job(self, job_id: str) -> Dict[str, Any]:
        """
        Get information about a job.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Job information
        """
        try:
            job = self.scheduler.get_job(job_id)
            if not job:
                return None
                
            info = {
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            }
            
            # Add metrics if available
            if job_id in self.job_metrics:
                info["metrics"] = self.job_metrics[job_id]
                
            return info
            
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {str(e)}", exc_info=True)
            return None
            
    async def get_all_jobs(self) -> List[Dict[str, Any]]:
        """
        Get information about all jobs.
        
        Returns:
            List of job information dictionaries
        """
        try:
            jobs = []
            for job in self.scheduler.get_jobs():
                info = {
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                    "trigger": str(job.trigger)
                }
                
                # Add metrics if available
                if job.id in self.job_metrics:
                    info["metrics"] = self.job_metrics[job.id]
                    
                jobs.append(info)
                
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to get all jobs: {str(e)}", exc_info=True)
            return []
            
    async def reschedule_job(self, job_id: str, trigger: Union[str, Any] = None, **trigger_args) -> bool:
        """
        Reschedule an existing job with a new trigger.
        
        Args:
            job_id: ID of the job to reschedule
            trigger: New trigger
            **trigger_args: Arguments for the trigger
            
        Returns:
            True if rescheduled successfully, False otherwise
        """
        try:
            self.scheduler.reschedule_job(job_id, trigger=trigger, **trigger_args)
            logger.info(f"Rescheduled job {job_id} with trigger {trigger}")
            return True
        except Exception as e:
            logger.error(f"Failed to reschedule job {job_id}: {str(e)}", exc_info=True)
            return False
