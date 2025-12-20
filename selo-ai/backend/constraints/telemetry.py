"""
Constraint Telemetry Module

Tracks constraint violations, validation performance, and token usage
for monitoring and optimization.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from collections import defaultdict, deque
from threading import Lock

logger = logging.getLogger("selo.constraints.telemetry")


@dataclass
class ConstraintViolationEvent:
    """Records a constraint violation occurrence."""
    timestamp: datetime
    constraint_type: str  # "identity", "grounding", "sensory", etc.
    violation_term: str
    context: str
    severity: str  # "critical", "high", "medium", "low"
    auto_cleaned: bool
    task_type: str  # "reflection", "chat", "persona_prompt"
    persona_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return asdict(self)


@dataclass
class ValidationMetrics:
    """Metrics for a validation operation."""
    timestamp: datetime
    task_type: str
    validation_duration_ms: float
    passed: bool
    violation_count: int
    token_cost: int
    validator_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TokenUsageMetrics:
    """Tracks token usage for constraints."""
    timestamp: datetime
    constraint_set_name: str
    total_tokens: int
    enabled_constraint_count: int
    disabled_constraint_count: int
    task_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ConstraintTelemetry:
    """
    Centralized telemetry collection for constraint enforcement.
    """
    
    def __init__(self, max_events: int = 10000):
        """
        Initialize telemetry collector.
        
        Args:
            max_events: Maximum events to retain in memory (FIFO after limit)
        """
        self.max_events = max_events
        self._lock = Lock()
        
        # Event storage
        self.violations: deque = deque(maxlen=max_events)
        self.validations: deque = deque(maxlen=max_events)
        self.token_usage: deque = deque(maxlen=max_events)
        
        # Aggregated metrics
        self.violation_counts: Dict[str, int] = defaultdict(int)
        self.auto_clean_success_count: int = 0
        self.auto_clean_failure_count: int = 0
        self.validation_count: int = 0
        self.validation_failure_count: int = 0
        
        # Performance metrics
        self.total_validation_time_ms: float = 0.0
        self.avg_validation_time_ms: float = 0.0
    
    def record_violation(
        self,
        constraint_type: str,
        violation_term: str,
        context: str,
        severity: str,
        auto_cleaned: bool,
        task_type: str,
        persona_name: Optional[str] = None
    ) -> None:
        """Record a constraint violation event."""
        event = ConstraintViolationEvent(
            timestamp=datetime.now(timezone.utc),
            constraint_type=constraint_type,
            violation_term=violation_term,
            context=context[:200],  # Truncate context
            severity=severity,
            auto_cleaned=auto_cleaned,
            task_type=task_type,
            persona_name=persona_name
        )
        
        with self._lock:
            self.violations.append(event)
            self.violation_counts[constraint_type] += 1
            
            if auto_cleaned:
                self.auto_clean_success_count += 1
            else:
                self.auto_clean_failure_count += 1
    
    def record_validation(
        self,
        task_type: str,
        duration_ms: float,
        passed: bool,
        violation_count: int,
        token_cost: int,
        validator_name: str
    ) -> None:
        """Record a validation operation."""
        metrics = ValidationMetrics(
            timestamp=datetime.now(timezone.utc),
            task_type=task_type,
            validation_duration_ms=duration_ms,
            passed=passed,
            violation_count=violation_count,
            token_cost=token_cost,
            validator_name=validator_name
        )
        
        with self._lock:
            self.validations.append(metrics)
            self.validation_count += 1
            if not passed:
                self.validation_failure_count += 1
            
            # Update average validation time
            self.total_validation_time_ms += duration_ms
            self.avg_validation_time_ms = self.total_validation_time_ms / self.validation_count
    
    def record_token_usage(
        self,
        constraint_set_name: str,
        total_tokens: int,
        enabled_count: int,
        disabled_count: int,
        task_type: str
    ) -> None:
        """Record token usage for a constraint set."""
        metrics = TokenUsageMetrics(
            timestamp=datetime.now(timezone.utc),
            constraint_set_name=constraint_set_name,
            total_tokens=total_tokens,
            enabled_constraint_count=enabled_count,
            disabled_constraint_count=disabled_count,
            task_type=task_type
        )
        
        with self._lock:
            self.token_usage.append(metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self._lock:
            return {
                "total_violations": sum(self.violation_counts.values()),
                "violations_by_type": dict(self.violation_counts),
                "auto_clean_success_rate": (
                    self.auto_clean_success_count / 
                    (self.auto_clean_success_count + self.auto_clean_failure_count)
                    if (self.auto_clean_success_count + self.auto_clean_failure_count) > 0
                    else 0.0
                ),
                "total_validations": self.validation_count,
                "validation_failure_rate": (
                    self.validation_failure_count / self.validation_count
                    if self.validation_count > 0
                    else 0.0
                ),
                "avg_validation_time_ms": round(self.avg_validation_time_ms, 2),
                "total_validation_time_ms": round(self.total_validation_time_ms, 2)
            }
    
    def get_recent_violations(
        self,
        limit: int = 100,
        constraint_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent violation events.
        
        Args:
            limit: Maximum number of events to return
            constraint_type: Filter by constraint type
            
        Returns:
            List of violation event dictionaries
        """
        with self._lock:
            violations = list(self.violations)
            
            if constraint_type:
                violations = [v for v in violations if v.constraint_type == constraint_type]
            
            # Return most recent first
            violations.reverse()
            return [v.to_dict() for v in violations[:limit]]
    
    def get_recent_validations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent validation metrics."""
        with self._lock:
            validations = list(self.validations)
            validations.reverse()
            return [v.to_dict() for v in validations[:limit]]
    
    def get_token_usage_report(
        self,
        limit: int = 100,
        task_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get token usage report."""
        with self._lock:
            usage = list(self.token_usage)
            
            if task_type:
                usage = [u for u in usage if u.task_type == task_type]
            
            usage.reverse()
            return [u.to_dict() for u in usage[:limit]]
    
    def get_violation_trends(self, hours: int = 24) -> Dict[str, List[int]]:
        """
        Get violation trends over time.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary mapping constraint types to hourly counts
        """
        from datetime import timedelta
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        with self._lock:
            # Filter to recent violations
            recent = [v for v in self.violations if v.timestamp >= cutoff]
            
            # Group by constraint type and hour
            trends: Dict[str, List[int]] = defaultdict(lambda: [0] * hours)
            
            for violation in recent:
                hours_ago = int((datetime.now(timezone.utc) - violation.timestamp).total_seconds() / 3600)
                if hours_ago < hours:
                    trends[violation.constraint_type][hours - 1 - hours_ago] += 1
            
            return dict(trends)
    
    def clear(self) -> None:
        """Clear all telemetry data."""
        with self._lock:
            self.violations.clear()
            self.validations.clear()
            self.token_usage.clear()
            self.violation_counts.clear()
            self.auto_clean_success_count = 0
            self.auto_clean_failure_count = 0
            self.validation_count = 0
            self.validation_failure_count = 0
            self.total_validation_time_ms = 0.0
            self.avg_validation_time_ms = 0.0


# Global telemetry instance
_global_telemetry: Optional[ConstraintTelemetry] = None


def get_constraint_telemetry() -> ConstraintTelemetry:
    """Get or create the global constraint telemetry instance."""
    global _global_telemetry
    if _global_telemetry is None:
        _global_telemetry = ConstraintTelemetry()
    return _global_telemetry


def record_violation(
    constraint_type: str,
    violation_term: str,
    context: str,
    severity: str = "medium",
    auto_cleaned: bool = False,
    task_type: str = "unknown",
    persona_name: Optional[str] = None
) -> None:
    """Convenience function to record a violation."""
    telemetry = get_constraint_telemetry()
    telemetry.record_violation(
        constraint_type=constraint_type,
        violation_term=violation_term,
        context=context,
        severity=severity,
        auto_cleaned=auto_cleaned,
        task_type=task_type,
        persona_name=persona_name
    )


def record_validation(
    task_type: str,
    duration_ms: float,
    passed: bool,
    violation_count: int = 0,
    token_cost: int = 0,
    validator_name: str = "unknown"
) -> None:
    """Convenience function to record a validation."""
    telemetry = get_constraint_telemetry()
    telemetry.record_validation(
        task_type=task_type,
        duration_ms=duration_ms,
        passed=passed,
        violation_count=violation_count,
        token_cost=token_cost,
        validator_name=validator_name
    )


def with_telemetry(validator_name: str, task_type: str = "unknown"):
    """
    Decorator to automatically record validation telemetry.
    
    Usage:
        @with_telemetry("identity_validator", "reflection")
        def validate_something(...):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000
                
                # Infer pass/fail from result
                passed = True
                violation_count = 0
                if isinstance(result, tuple) and len(result) >= 2:
                    passed = result[0]
                    if len(result) >= 3 and isinstance(result[2], list):
                        violation_count = len(result[2])
                
                record_validation(
                    task_type=task_type,
                    duration_ms=duration_ms,
                    passed=passed,
                    violation_count=violation_count,
                    validator_name=validator_name
                )
                
                return result
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                record_validation(
                    task_type=task_type,
                    duration_ms=duration_ms,
                    passed=False,
                    validator_name=validator_name
                )
                raise
        return wrapper
    return decorator
