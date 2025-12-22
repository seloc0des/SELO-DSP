"""
Saga pattern implementation for distributed transaction coordination.

This module provides the saga pattern for managing multi-step operations
with automatic compensation on failure to maintain consistency.
"""

from .orchestrator import SagaOrchestrator
from .integration import SagaIntegration
from .monitor import SagaMonitor
from .handlers import (
    PersonaEvolutionHandlers,
    GoalManagementHandlers,
    ConversationProcessingHandlers,
    EpisodeGenerationHandlers
)

__all__ = [
    'SagaOrchestrator',
    'SagaIntegration',
    'SagaMonitor',
    'PersonaEvolutionHandlers',
    'GoalManagementHandlers',
    'ConversationProcessingHandlers',
    'EpisodeGenerationHandlers',
]
