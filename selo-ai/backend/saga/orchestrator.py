"""
Saga orchestrator for coordinating distributed transactions.

Implements the saga pattern with forward execution and compensating
transactions for maintaining consistency across multi-step operations.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable, Awaitable

from ..db.repositories.saga import SagaRepository
from ..db.models.saga import SagaStatus, StepStatus

logger = logging.getLogger("selo.saga.orchestrator")


class SagaOrchestrator:
    """
    Orchestrates saga execution with automatic compensation on failure.
    
    The orchestrator executes saga steps sequentially. If any step fails,
    it automatically triggers compensation for all completed steps in
    reverse order to maintain consistency.
    """
    
    def __init__(self, saga_repo: Optional[SagaRepository] = None):
        self.saga_repo = saga_repo or SagaRepository()
        self._step_handlers: Dict[str, Callable] = {}
        self._compensation_handlers: Dict[str, Callable] = {}
    
    def register_step_handler(
        self,
        step_type: str,
        handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
    ):
        """
        Register a handler for a specific step type.
        
        Args:
            step_type: Type identifier for the step
            handler: Async function that executes the step
        """
        self._step_handlers[step_type] = handler
        logger.debug(f"Registered step handler for type: {step_type}")
    
    def register_compensation_handler(
        self,
        handler_name: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ):
        """
        Register a compensation handler.
        
        Args:
            handler_name: Name identifier for the handler
            handler: Async function that performs compensation
        """
        self._compensation_handlers[handler_name] = handler
        logger.debug(f"Registered compensation handler: {handler_name}")
    
    async def create_saga(
        self,
        saga_type: str,
        user_id: str,
        input_data: Dict[str, Any],
        steps: List[Dict[str, Any]],
        persona_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        max_retries: int = 3
    ) -> str:
        """
        Create a new saga.
        
        Args:
            saga_type: Type of saga (e.g., 'persona_evolution')
            user_id: User ID
            input_data: Input parameters for the saga
            steps: List of step configurations
            persona_id: Optional persona ID
            correlation_id: Optional correlation ID
            max_retries: Maximum retry attempts
            
        Returns:
            Saga ID
        """
        saga_data = {
            "saga_type": saga_type,
            "user_id": user_id,
            "persona_id": persona_id,
            "correlation_id": correlation_id,
            "input_data": input_data,
            "max_retries": max_retries,
            "status": SagaStatus.PENDING
        }
        
        saga = await self.saga_repo.create_saga(saga_data, steps)
        logger.info(f"Created saga {saga['id']} of type {saga_type}")
        return saga['id']
    
    async def execute_saga(self, saga_id: str) -> Dict[str, Any]:
        """
        Execute a saga from start to finish.
        
        Args:
            saga_id: ID of the saga to execute
            
        Returns:
            Final saga state with execution results
        """
        saga = await self.saga_repo.get_saga(saga_id)
        if not saga:
            raise ValueError(f"Saga {saga_id} not found")
        
        logger.info(f"Starting execution of saga {saga_id} ({saga['saga_type']})")
        
        # Update saga to in-progress
        await self.saga_repo.update_saga_status(saga_id, SagaStatus.IN_PROGRESS)
        
        try:
            # Execute steps sequentially
            completed_steps = []
            previous_output = {}
            
            for step in saga['steps']:
                # Merge previous step output into current step input
                if previous_output:
                    step['input_data'] = {**step['input_data'], **previous_output}
                
                step_result = await self._execute_step(saga_id, step)
                
                if step_result['status'] == StepStatus.COMPLETED:
                    completed_steps.append(step_result)
                    # Store output for next step
                    previous_output = step_result.get('output_data', {})
                    continue
                
                # Step failed - trigger compensation
                logger.warning(
                    f"Step {step['step_index']} ({step['step_name']}) failed in saga {saga_id}. "
                    f"Starting compensation..."
                )
                
                await self._compensate_saga(saga_id, completed_steps)
                
                # Mark saga as failed
                await self.saga_repo.update_saga_status(
                    saga_id,
                    SagaStatus.FAILED,
                    error_data={
                        "failed_step": step['step_name'],
                        "error": step_result.get('error_data')
                    }
                )
                
                return await self.saga_repo.get_saga(saga_id)
            
            # All steps completed successfully
            final_saga = await self.saga_repo.get_saga(saga_id)
            
            # Collect output from last step or aggregate
            output_data = {}
            if final_saga['steps']:
                last_step = final_saga['steps'][-1]
                output_data = last_step.get('output_data', {})
            
            await self.saga_repo.update_saga_status(
                saga_id,
                SagaStatus.COMPLETED,
                output_data=output_data
            )
            
            logger.info(f"Saga {saga_id} completed successfully")
            return await self.saga_repo.get_saga(saga_id)
            
        except Exception as e:
            logger.error(f"Unexpected error executing saga {saga_id}: {e}", exc_info=True)
            
            await self.saga_repo.update_saga_status(
                saga_id,
                SagaStatus.FAILED,
                error_data={"error": str(e), "type": "unexpected_error"}
            )
            
            raise
    
    async def _execute_step(
        self,
        saga_id: str,
        step: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single saga step with retry logic.
        
        Args:
            saga_id: Parent saga ID
            step: Step configuration
            
        Returns:
            Updated step state
        """
        step_id = step['id']
        step_type = step['step_type']
        
        # Get handler for this step type
        handler = self._step_handlers.get(step_type)
        if not handler:
            logger.error(f"No handler registered for step type: {step_type}")
            await self.saga_repo.update_step_status(
                step_id,
                StepStatus.FAILED,
                error_data={"error": f"No handler for step type: {step_type}"}
            )
            return await self.saga_repo.get_step(step_id)
        
        # Update step to in-progress
        await self.saga_repo.update_step_status(step_id, StepStatus.IN_PROGRESS)
        
        max_retries = step.get('max_retries', 3)
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                logger.debug(
                    f"Executing step {step['step_index']} ({step['step_name']}) "
                    f"in saga {saga_id}, attempt {retry_count + 1}/{max_retries + 1}"
                )
                
                # Execute the step handler
                result = await handler(step['input_data'])
                
                # Mark step as completed
                await self.saga_repo.update_step_status(
                    step_id,
                    StepStatus.COMPLETED,
                    output_data=result.get('output_data', {}),
                    compensation_data=result.get('compensation_data', {})
                )
                
                logger.info(f"Step {step['step_name']} completed successfully in saga {saga_id}")
                return await self.saga_repo.get_step(step_id)
                
            except Exception as e:
                retry_count += 1
                logger.warning(
                    f"Step {step['step_name']} failed (attempt {retry_count}/{max_retries + 1}): {e}"
                )
                
                if retry_count > max_retries:
                    # Max retries exceeded
                    await self.saga_repo.update_step_status(
                        step_id,
                        StepStatus.FAILED,
                        error_data={"error": str(e), "retry_count": retry_count},
                        retry_count=retry_count
                    )
                    return await self.saga_repo.get_step(step_id)
                
                # Wait before retry with exponential backoff
                await asyncio.sleep(min(2 ** retry_count, 30))
        
        # Should not reach here
        return await self.saga_repo.get_step(step_id)
    
    async def _compensate_saga(
        self,
        saga_id: str,
        completed_steps: List[Dict[str, Any]]
    ):
        """
        Execute compensation for completed steps in reverse order.
        
        Args:
            saga_id: Saga ID
            completed_steps: List of completed steps to compensate
        """
        if not completed_steps:
            logger.info(f"No steps to compensate for saga {saga_id}")
            return
        
        logger.info(f"Compensating {len(completed_steps)} steps for saga {saga_id}")
        
        # Update saga status to compensating
        await self.saga_repo.update_saga_status(saga_id, SagaStatus.COMPENSATING)
        
        # Compensate in reverse order
        for step in reversed(completed_steps):
            await self._compensate_step(step)
        
        # Mark saga as compensated
        await self.saga_repo.update_saga_status(saga_id, SagaStatus.COMPENSATED)
        logger.info(f"Saga {saga_id} compensation completed")
    
    async def _compensate_step(self, step: Dict[str, Any]):
        """
        Execute compensation for a single step.
        
        Args:
            step: Step to compensate
        """
        step_id = step['id']
        compensation_handler_name = step.get('compensation_handler')
        
        if not compensation_handler_name:
            logger.warning(f"No compensation handler defined for step {step['step_name']}")
            await self.saga_repo.update_step_status(step_id, StepStatus.COMPENSATED)
            return
        
        handler = self._compensation_handlers.get(compensation_handler_name)
        if not handler:
            logger.error(
                f"Compensation handler {compensation_handler_name} not registered "
                f"for step {step['step_name']}"
            )
            return
        
        # Update step to compensating
        await self.saga_repo.update_step_status(step_id, StepStatus.COMPENSATING)
        
        try:
            logger.debug(f"Compensating step {step['step_name']}")
            
            # Execute compensation handler
            compensation_data = step.get('compensation_data', {})
            await handler(compensation_data)
            
            # Mark as compensated
            await self.saga_repo.update_step_status(step_id, StepStatus.COMPENSATED)
            logger.info(f"Step {step['step_name']} compensated successfully")
            
        except Exception as e:
            logger.error(
                f"Failed to compensate step {step['step_name']}: {e}",
                exc_info=True
            )
            # Continue with other compensations even if one fails
    
    async def retry_failed_saga(self, saga_id: str) -> Dict[str, Any]:
        """
        Retry a failed saga.
        
        Args:
            saga_id: ID of the failed saga
            
        Returns:
            Updated saga state
        """
        saga = await self.saga_repo.get_saga(saga_id)
        if not saga:
            raise ValueError(f"Saga {saga_id} not found")
        
        if saga['status'] not in [SagaStatus.FAILED, SagaStatus.COMPENSATED]:
            raise ValueError(f"Saga {saga_id} is not in a retryable state: {saga['status']}")
        
        if saga['retry_count'] >= saga['max_retries']:
            raise ValueError(f"Saga {saga_id} has exceeded max retries")
        
        logger.info(f"Retrying saga {saga_id} (attempt {saga['retry_count'] + 1})")
        
        # Increment retry count and reset status
        await self.saga_repo.update_saga_status(
            saga_id,
            SagaStatus.PENDING,
            retry_count=saga['retry_count'] + 1
        )
        
        # Execute the saga again
        return await self.execute_saga(saga_id)
