"""
Saga pattern models for distributed transaction coordination.

Implements the saga pattern for managing multi-step operations with
compensating transactions to ensure eventual consistency.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, String, DateTime, JSON, Integer, Text, Enum as SQLEnum, ForeignKey
from sqlalchemy.orm import relationship
import enum

from ..base import Base


class SagaStatus(str, enum.Enum):
    """Status of a saga execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"


class StepStatus(str, enum.Enum):
    """Status of an individual saga step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    SKIPPED = "skipped"


class Saga(Base):
    """
    Saga orchestration record for multi-step distributed transactions.
    
    A saga represents a long-running business transaction composed of multiple
    steps. If any step fails, compensating transactions are executed to undo
    completed steps and maintain consistency.
    """
    __tablename__ = "sagas"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    saga_type = Column(String, nullable=False, index=True)
    """Type of saga (e.g., 'persona_evolution', 'learning_consolidation')"""
    
    status = Column(SQLEnum(SagaStatus), nullable=False, default=SagaStatus.PENDING, index=True)
    """Current status of the saga"""
    
    # Context and metadata
    user_id = Column(String, nullable=False, index=True)
    persona_id = Column(String, nullable=True, index=True)
    correlation_id = Column(String, nullable=True, index=True)
    """External correlation ID for tracking across systems"""
    
    # Saga data
    input_data = Column(JSON, nullable=False, default=lambda: {})
    """Input parameters for the saga"""
    
    output_data = Column(JSON, nullable=True)
    """Final output data from successful completion"""
    
    error_data = Column(JSON, nullable=True)
    """Error information if saga failed"""
    
    # Execution tracking
    current_step_index = Column(Integer, nullable=False, default=0)
    """Index of the currently executing step"""
    
    total_steps = Column(Integer, nullable=False, default=0)
    """Total number of steps in this saga"""
    
    retry_count = Column(Integer, nullable=False, default=0)
    """Number of times this saga has been retried"""
    
    max_retries = Column(Integer, nullable=False, default=3)
    """Maximum retry attempts before marking as failed"""
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    failed_at = Column(DateTime(timezone=True), nullable=True)
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    steps = relationship("SagaStep", back_populates="saga", cascade="all, delete-orphan", order_by="SagaStep.step_index")
    
    def to_dict(self):
        """Convert saga to dictionary representation."""
        return {
            "id": self.id,
            "saga_type": self.saga_type,
            "status": self.status.value if isinstance(self.status, SagaStatus) else self.status,
            "user_id": self.user_id,
            "persona_id": self.persona_id,
            "correlation_id": self.correlation_id,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error_data": self.error_data,
            "current_step_index": self.current_step_index,
            "total_steps": self.total_steps,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "failed_at": self.failed_at.isoformat() if self.failed_at else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class SagaStep(Base):
    """
    Individual step within a saga execution.
    
    Each step represents an atomic operation with a forward action and
    a compensating action to undo the operation if needed.
    """
    __tablename__ = "saga_steps"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    saga_id = Column(String, ForeignKey("sagas.id"), nullable=False, index=True)
    """ID of the parent saga"""
    
    step_index = Column(Integer, nullable=False)
    """Order of this step in the saga (0-indexed)"""
    
    step_name = Column(String, nullable=False)
    """Name/identifier of this step"""
    
    step_type = Column(String, nullable=False)
    """Type of operation (e.g., 'create_learning', 'update_persona')"""
    
    status = Column(SQLEnum(StepStatus), nullable=False, default=StepStatus.PENDING)
    """Current status of this step"""
    
    # Step data
    input_data = Column(JSON, nullable=False, default=lambda: {})
    """Input parameters for this step"""
    
    output_data = Column(JSON, nullable=True)
    """Output data from successful execution"""
    
    error_data = Column(JSON, nullable=True)
    """Error information if step failed"""
    
    # Compensation tracking
    compensation_data = Column(JSON, nullable=True)
    """Data needed for compensation (e.g., IDs of created resources)"""
    
    compensation_handler = Column(String, nullable=True)
    """Name of the compensation handler function"""
    
    # Execution tracking
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    failed_at = Column(DateTime(timezone=True), nullable=True)
    compensated_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    saga = relationship("Saga", back_populates="steps")
    
    def to_dict(self):
        """Convert step to dictionary representation."""
        return {
            "id": self.id,
            "saga_id": self.saga_id,
            "step_index": self.step_index,
            "step_name": self.step_name,
            "step_type": self.step_type,
            "status": self.status.value if isinstance(self.status, StepStatus) else self.status,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error_data": self.error_data,
            "compensation_data": self.compensation_data,
            "compensation_handler": self.compensation_handler,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "failed_at": self.failed_at.isoformat() if self.failed_at else None,
            "compensated_at": self.compensated_at.isoformat() if self.compensated_at else None,
        }
