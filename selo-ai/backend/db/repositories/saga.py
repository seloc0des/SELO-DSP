"""
Repository for saga pattern persistence operations.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.saga import Saga, SagaStep, SagaStatus, StepStatus
from ..session import get_session

logger = logging.getLogger("selo.db.saga")


class SagaRepository:
    """Repository for saga persistence operations."""
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        self.db_session = db_session
    
    async def create_saga(
        self,
        saga_data: Dict[str, Any],
        steps_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a new saga with its steps.
        
        Args:
            saga_data: Saga configuration
            steps_data: List of step configurations
            
        Returns:
            Created saga as dict with steps
        """
        async with get_session(self.db_session) as session:
            # Create saga
            saga = Saga(**{
                k: v for k, v in saga_data.items()
                if hasattr(Saga, k) and k not in ['steps']
            })
            
            saga.total_steps = len(steps_data)
            session.add(saga)
            await session.flush()
            
            # Create steps
            for idx, step_data in enumerate(steps_data):
                step = SagaStep(
                    saga_id=saga.id,
                    step_index=idx,
                    **{k: v for k, v in step_data.items() if hasattr(SagaStep, k)}
                )
                session.add(step)
            
            await session.flush()
            
            logger.info(f"Created saga {saga.id} ({saga.saga_type}) with {len(steps_data)} steps")
            # Note: get_session context manager handles commit; refresh after context would fail
            # Return saga dict directly to avoid detached instance issues
            saga_dict = saga.to_dict()
            saga_dict['steps'] = [step.to_dict() for step in sorted(saga.steps, key=lambda s: s.step_index)]
            return saga_dict
    
    async def get_saga(
        self,
        saga_id: str,
        session: Optional[AsyncSession] = None
    ) -> Optional[Dict[str, Any]]:
        """Get saga by ID with all steps."""
        async with get_session(session) as session:
            stmt = (
                select(Saga)
                .where(Saga.id == saga_id)
                .options(selectinload(Saga.steps))
            )
            result = await session.execute(stmt)
            saga = result.scalar_one_or_none()
            
            if not saga:
                return None
            
            saga_dict = saga.to_dict()
            saga_dict['steps'] = [step.to_dict() for step in sorted(saga.steps, key=lambda s: s.step_index)]
            return saga_dict
    
    async def update_saga_status(
        self,
        saga_id: str,
        status: SagaStatus,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Update saga status and optional fields."""
        async with get_session(self.db_session) as session:
            updates = {"status": status, "last_updated": datetime.now(timezone.utc)}
            
            if status == SagaStatus.IN_PROGRESS and "started_at" not in kwargs:
                updates["started_at"] = datetime.now(timezone.utc)
            elif status == SagaStatus.COMPLETED and "completed_at" not in kwargs:
                updates["completed_at"] = datetime.now(timezone.utc)
            elif status == SagaStatus.FAILED and "failed_at" not in kwargs:
                updates["failed_at"] = datetime.now(timezone.utc)
            
            updates.update(kwargs)
            
            await session.execute(
                update(Saga).where(Saga.id == saga_id).values(**updates)
            )
            # Note: get_session context manager handles commit automatically
            
            return await self.get_saga(saga_id, session=session)
    
    async def update_step_status(
        self,
        step_id: str,
        status: StepStatus,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Update step status and optional fields."""
        async with get_session(self.db_session) as session:
            updates = {"status": status}
            
            if status == StepStatus.IN_PROGRESS and "started_at" not in kwargs:
                updates["started_at"] = datetime.now(timezone.utc)
            elif status == StepStatus.COMPLETED and "completed_at" not in kwargs:
                updates["completed_at"] = datetime.now(timezone.utc)
            elif status == StepStatus.FAILED and "failed_at" not in kwargs:
                updates["failed_at"] = datetime.now(timezone.utc)
            elif status == StepStatus.COMPENSATED and "compensated_at" not in kwargs:
                updates["compensated_at"] = datetime.now(timezone.utc)
            
            updates.update(kwargs)
            
            await session.execute(
                update(SagaStep).where(SagaStep.id == step_id).values(**updates)
            )
            # Note: get_session context manager handles commit automatically
            
            # Get updated step
            result = await session.execute(
                select(SagaStep).where(SagaStep.id == step_id)
            )
            step = result.scalar_one_or_none()
            return step.to_dict() if step else None
    
    async def get_step(
        self,
        step_id: str,
        session: Optional[AsyncSession] = None
    ) -> Optional[Dict[str, Any]]:
        """Get step by ID."""
        async with get_session(session) as session:
            result = await session.execute(
                select(SagaStep).where(SagaStep.id == step_id)
            )
            step = result.scalar_one_or_none()
            return step.to_dict() if step else None
    
    async def list_active_sagas(
        self,
        saga_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List sagas that are in progress or pending."""
        async with get_session(self.db_session) as session:
            stmt = select(Saga).where(
                Saga.status.in_([SagaStatus.PENDING, SagaStatus.IN_PROGRESS, SagaStatus.COMPENSATING])
            )
            
            if saga_type:
                stmt = stmt.where(Saga.saga_type == saga_type)
            
            stmt = stmt.order_by(Saga.created_at.desc()).limit(limit)
            
            result = await session.execute(stmt)
            sagas = result.scalars().all()
            
            return [saga.to_dict() for saga in sagas]
    
    async def list_failed_sagas(
        self,
        saga_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List failed sagas for monitoring."""
        async with get_session(self.db_session) as session:
            stmt = select(Saga).where(Saga.status == SagaStatus.FAILED)
            
            if saga_type:
                stmt = stmt.where(Saga.saga_type == saga_type)
            
            stmt = stmt.order_by(Saga.failed_at.desc()).limit(limit)
            
            result = await session.execute(stmt)
            sagas = result.scalars().all()
            
            return [saga.to_dict() for saga in sagas]
