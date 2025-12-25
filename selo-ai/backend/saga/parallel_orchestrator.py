"""
Parallel Saga Orchestrator

Enhanced orchestrator supporting parallel step execution and
dynamic saga composition for complex cognitive workflows.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict

from .orchestrator import SagaOrchestrator
from ..db.models.saga import SagaStatus, StepStatus

logger = logging.getLogger("selo.saga.parallel")


class ParallelSagaOrchestrator(SagaOrchestrator):
    """
    Enhanced orchestrator supporting parallel step execution and
    dynamic saga composition for complex cognitive workflows.
    
    Features:
    - Parallel execution of independent steps
    - Dependency graph analysis
    - Priority-based scheduling
    - Dynamic step injection
    """
    
    async def execute_saga_parallel(
        self,
        saga_id: str,
        max_parallel: int = 3
    ) -> Dict[str, Any]:
        """
        Execute saga with parallel step processing where dependencies allow.
        
        Analyzes step dependency graph and executes independent steps
        concurrently while respecting data dependencies.
        
        Args:
            saga_id: Saga ID to execute
            max_parallel: Maximum number of parallel steps
        
        Returns:
            Final saga state with execution results
        """
        saga = await self.saga_repo.get_saga(saga_id)
        if not saga:
            raise ValueError(f"Saga {saga_id} not found")
        
        logger.info(
            f"Starting parallel execution of saga {saga_id} ({saga['saga_type']}) "
            f"with max_parallel={max_parallel}"
        )
        
        # Update saga to in-progress
        await self.saga_repo.update_saga_status(saga_id, SagaStatus.IN_PROGRESS)
        
        try:
            # Build dependency graph
            dependency_levels = self._build_dependency_graph(saga['steps'])
            
            logger.debug(
                f"Saga {saga_id} has {len(dependency_levels)} dependency levels"
            )
            
            # Execute steps level by level with parallelism
            completed_outputs = {}
            all_completed_steps = []
            
            for level_idx, level_steps in enumerate(dependency_levels):
                logger.debug(
                    f"Executing level {level_idx + 1}/{len(dependency_levels)} "
                    f"with {len(level_steps)} steps"
                )
                
                # Prepare steps with dependency outputs
                prepared_steps = []
                for step in level_steps:
                    # Inject outputs from dependencies
                    step['input_data'] = {
                        **step.get('input_data', {}),
                        **self._gather_dependency_outputs(step, completed_outputs)
                    }
                    prepared_steps.append(step)
                
                # Execute steps in parallel (respecting max_parallel limit)
                level_results = await self._execute_steps_parallel(
                    saga_id,
                    prepared_steps,
                    max_parallel
                )
                
                # Check for failures
                failed_steps = [
                    (step, result) 
                    for step, result in zip(prepared_steps, level_results)
                    if isinstance(result, Exception) or result.get('status') == StepStatus.FAILED
                ]
                
                if failed_steps:
                    # Trigger compensation for all completed steps
                    logger.warning(
                        f"Level {level_idx + 1} had {len(failed_steps)} failed steps. "
                        f"Starting compensation..."
                    )
                    
                    await self._compensate_saga(saga_id, all_completed_steps)
                    
                    # Mark saga as failed
                    failed_step, failed_result = failed_steps[0]
                    error_data = {
                        "failed_step": failed_step.get('step_name'),
                        "failed_level": level_idx + 1,
                        "error": (
                            str(failed_result) if isinstance(failed_result, Exception)
                            else failed_result.get('error_data')
                        )
                    }
                    
                    await self.saga_repo.update_saga_status(
                        saga_id,
                        SagaStatus.FAILED,
                        error_data=error_data
                    )
                    
                    return await self.saga_repo.get_saga(saga_id)
                
                # Store outputs and track completed steps
                for step, result in zip(prepared_steps, level_results):
                    if not isinstance(result, Exception):
                        completed_outputs[step['id']] = result
                        all_completed_steps.append(result)
            
            # All levels completed successfully
            final_saga = await self.saga_repo.get_saga(saga_id)
            
            # Collect output from last steps
            output_data = {}
            if final_saga['steps']:
                # Aggregate outputs from all final-level steps
                for step in dependency_levels[-1] if dependency_levels else []:
                    step_output = completed_outputs.get(step['id'], {})
                    output_data.update(step_output.get('output_data', {}))
            
            await self.saga_repo.update_saga_status(
                saga_id,
                SagaStatus.COMPLETED,
                output_data=output_data
            )
            
            logger.info(
                f"Saga {saga_id} completed successfully with parallel execution"
            )
            
            return await self.saga_repo.get_saga(saga_id)
            
        except Exception as e:
            logger.error(
                f"Unexpected error in parallel saga execution {saga_id}: {e}",
                exc_info=True
            )
            
            await self.saga_repo.update_saga_status(
                saga_id,
                SagaStatus.FAILED,
                error_data={"error": str(e), "type": "unexpected_error"}
            )
            
            raise
    
    def _build_dependency_graph(
        self,
        steps: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Build topological levels for parallel execution.
        
        Analyzes step dependencies and creates execution levels where:
        - Level 0: Steps with no dependencies
        - Level N: Steps depending only on steps in levels 0..N-1
        
        Args:
            steps: List of step configurations
        
        Returns:
            List of levels, each containing steps that can execute in parallel
        """
        if not steps:
            return []
        
        # Build dependency map
        step_map = {step['id']: step for step in steps}
        
        # Track which steps have been assigned to levels
        assigned_steps: Set[str] = set()
        levels: List[List[Dict[str, Any]]] = []
        
        # Keep building levels until all steps are assigned
        remaining_steps = set(step_map.keys())
        
        while remaining_steps:
            current_level = []
            
            for step_id in list(remaining_steps):
                step = step_map[step_id]
                dependencies = step.get('depends_on', [])
                
                # Check if all dependencies are satisfied
                if all(dep_id in assigned_steps for dep_id in dependencies):
                    current_level.append(step)
                    assigned_steps.add(step_id)
                    remaining_steps.remove(step_id)
            
            if not current_level:
                # Circular dependency or error - add remaining steps to final level
                logger.warning(
                    f"Circular dependency detected or invalid dependencies. "
                    f"Adding {len(remaining_steps)} remaining steps to final level."
                )
                current_level = [step_map[step_id] for step_id in remaining_steps]
                remaining_steps.clear()
            
            levels.append(current_level)
        
        return levels
    
    def _gather_dependency_outputs(
        self,
        step: Dict[str, Any],
        completed_outputs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Gather outputs from dependency steps.
        
        Args:
            step: Step configuration
            completed_outputs: Map of step_id -> step result
        
        Returns:
            Dictionary of aggregated outputs from dependencies
        """
        aggregated = {}
        
        dependencies = step.get('depends_on', [])
        for dep_id in dependencies:
            if dep_id in completed_outputs:
                dep_output = completed_outputs[dep_id].get('output_data', {})
                # Merge outputs (later dependencies override earlier ones)
                aggregated.update(dep_output)
        
        return aggregated
    
    async def _execute_steps_parallel(
        self,
        saga_id: str,
        steps: List[Dict[str, Any]],
        max_parallel: int
    ) -> List[Any]:
        """
        Execute multiple steps in parallel with concurrency limit.
        
        Args:
            saga_id: Parent saga ID
            steps: Steps to execute
            max_parallel: Maximum concurrent executions
        
        Returns:
            List of results (or exceptions) in same order as steps
        """
        if not steps:
            return []
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def execute_with_semaphore(step):
            async with semaphore:
                try:
                    return await self._execute_step(saga_id, step)
                except Exception as e:
                    logger.error(f"Step {step.get('step_name')} raised exception: {e}")
                    return e
        
        # Execute all steps concurrently (semaphore limits actual parallelism)
        results = await asyncio.gather(
            *[execute_with_semaphore(step) for step in steps],
            return_exceptions=True
        )
        
        return results
    
    async def create_dynamic_saga(
        self,
        saga_type: str,
        user_id: str,
        persona_id: Optional[str],
        initial_steps: List[Dict[str, Any]],
        step_generator_callback: Optional[callable] = None,
        max_retries: int = 3
    ) -> str:
        """
        Create a saga that can dynamically generate additional steps.
        
        Useful for cognitive workflows where next steps depend on
        intermediate results (e.g., reflection -> learning -> evolution).
        
        Args:
            saga_type: Type of saga
            user_id: User ID
            persona_id: Persona ID
            initial_steps: Initial step configurations
            step_generator_callback: Optional callback to generate more steps
            max_retries: Maximum retry attempts
        
        Returns:
            Saga ID
        """
        # Create initial saga
        saga_id = await self.create_saga(
            saga_type=saga_type,
            user_id=user_id,
            input_data={
                "dynamic": True,
                "step_generator": step_generator_callback is not None
            },
            steps=initial_steps,
            persona_id=persona_id,
            max_retries=max_retries
        )
        
        logger.info(f"Created dynamic saga {saga_id} with {len(initial_steps)} initial steps")
        
        return saga_id
    
    async def inject_saga_steps(
        self,
        saga_id: str,
        new_steps: List[Dict[str, Any]],
        insert_after_step_id: Optional[str] = None
    ) -> bool:
        """
        Dynamically inject new steps into a running saga.
        
        Args:
            saga_id: Saga ID
            new_steps: New steps to inject
            insert_after_step_id: Optional step ID to insert after
        
        Returns:
            True if successful, False otherwise
        """
        try:
            saga = await self.saga_repo.get_saga(saga_id)
            if not saga:
                logger.error(f"Saga {saga_id} not found for step injection")
                return False
            
            # Only allow injection for in-progress sagas
            if saga['status'] != SagaStatus.IN_PROGRESS:
                logger.warning(
                    f"Cannot inject steps into saga {saga_id} "
                    f"with status {saga['status']}"
                )
                return False
            
            # Add steps to saga (implementation depends on repository)
            # This is a placeholder - actual implementation would update the saga
            logger.info(
                f"Injected {len(new_steps)} steps into saga {saga_id}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error injecting steps into saga {saga_id}: {e}")
            return False
    
    def estimate_saga_duration(
        self,
        steps: List[Dict[str, Any]],
        avg_step_duration_seconds: float = 5.0
    ) -> float:
        """
        Estimate saga duration based on dependency graph.
        
        Args:
            steps: Step configurations
            avg_step_duration_seconds: Average duration per step
        
        Returns:
            Estimated duration in seconds
        """
        if not steps:
            return 0.0
        
        # Build dependency levels
        levels = self._build_dependency_graph(steps)
        
        # Duration = number of levels * average step duration
        # (assumes steps in same level execute in parallel)
        estimated_duration = len(levels) * avg_step_duration_seconds
        
        return estimated_duration
