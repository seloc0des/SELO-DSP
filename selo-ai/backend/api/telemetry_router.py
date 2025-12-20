"""
Telemetry Dashboard API Router

Provides REST API endpoints for accessing constraint telemetry,
memory ranking metrics, and system performance data.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("selo.api.telemetry")

# Create router
router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


@router.get("/summary")
async def get_telemetry_summary() -> Dict[str, Any]:
    """
    Get summary of all telemetry data.
    
    Returns:
        Telemetry summary with violation counts, validation metrics, etc.
    """
    try:
        from backend.constraints import get_constraint_telemetry
        
        telemetry = get_constraint_telemetry()
        summary = telemetry.get_summary()
        
        return {
            "status": "success",
            "data": summary,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting telemetry summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/violations")
async def get_violations(
    limit: int = Query(100, ge=1, le=1000),
    constraint_type: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """
    Get recent constraint violations.
    
    Args:
        limit: Maximum number of violations to return
        constraint_type: Optional filter by constraint type
        
    Returns:
        List of violation events
    """
    try:
        from backend.constraints import get_constraint_telemetry
        
        telemetry = get_constraint_telemetry()
        violations = telemetry.get_recent_violations(
            limit=limit,
            constraint_type=constraint_type
        )
        
        return {
            "status": "success",
            "data": violations,
            "count": len(violations),
            "limit": limit,
            "filter": {"constraint_type": constraint_type} if constraint_type else None
        }
    except Exception as e:
        logger.error(f"Error getting violations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/violations/trends")
async def get_violation_trends(
    hours: int = Query(24, ge=1, le=168)
) -> Dict[str, Any]:
    """
    Get violation trends over time.
    
    Args:
        hours: Number of hours to analyze (1-168)
        
    Returns:
        Hourly violation counts by type
    """
    try:
        from backend.constraints import get_constraint_telemetry
        
        telemetry = get_constraint_telemetry()
        trends = telemetry.get_violation_trends(hours=hours)
        
        return {
            "status": "success",
            "data": trends,
            "hours": hours,
            "constraint_types": list(trends.keys())
        }
    except Exception as e:
        logger.error(f"Error getting violation trends: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations")
async def get_validations(
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """
    Get recent validation metrics.
    
    Args:
        limit: Maximum number of validation records to return
        
    Returns:
        List of validation metrics
    """
    try:
        from backend.constraints import get_constraint_telemetry
        
        telemetry = get_constraint_telemetry()
        validations = telemetry.get_recent_validations(limit=limit)
        
        # Calculate aggregate metrics
        if validations:
            passed_count = sum(1 for v in validations if v.get("passed", False))
            avg_duration = sum(v.get("validation_duration_ms", 0) for v in validations) / len(validations)
            
            aggregates = {
                "pass_rate": passed_count / len(validations),
                "avg_duration_ms": round(avg_duration, 2),
                "total_validations": len(validations)
            }
        else:
            aggregates = {
                "pass_rate": 0.0,
                "avg_duration_ms": 0.0,
                "total_validations": 0
            }
        
        return {
            "status": "success",
            "data": validations,
            "aggregates": aggregates,
            "count": len(validations)
        }
    except Exception as e:
        logger.error(f"Error getting validations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/token-usage")
async def get_token_usage(
    limit: int = Query(100, ge=1, le=1000),
    task_type: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """
    Get token usage metrics for constraints.
    
    Args:
        limit: Maximum number of records to return
        task_type: Optional filter by task type
        
    Returns:
        List of token usage records
    """
    try:
        from backend.constraints import get_constraint_telemetry
        
        telemetry = get_constraint_telemetry()
        usage = telemetry.get_token_usage_report(
            limit=limit,
            task_type=task_type
        )
        
        # Calculate aggregates
        if usage:
            avg_tokens = sum(u.get("total_tokens", 0) for u in usage) / len(usage)
            total_tokens = sum(u.get("total_tokens", 0) for u in usage)
            
            aggregates = {
                "avg_tokens_per_request": round(avg_tokens, 1),
                "total_tokens": total_tokens,
                "records_analyzed": len(usage)
            }
        else:
            aggregates = {
                "avg_tokens_per_request": 0.0,
                "total_tokens": 0,
                "records_analyzed": 0
            }
        
        return {
            "status": "success",
            "data": usage,
            "aggregates": aggregates,
            "count": len(usage)
        }
    except Exception as e:
        logger.error(f"Error getting token usage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Get system performance metrics.
    
    Returns:
        Performance metrics including validation times, token efficiency, etc.
    """
    try:
        from backend.constraints import get_constraint_telemetry
        
        telemetry = get_constraint_telemetry()
        summary = telemetry.get_summary()
        
        # Get recent data for trends
        recent_validations = telemetry.get_recent_validations(limit=100)
        recent_usage = telemetry.get_token_usage_report(limit=100)
        
        # Calculate performance metrics
        if recent_validations:
            durations = [v.get("validation_duration_ms", 0) for v in recent_validations]
            p50_duration = sorted(durations)[len(durations) // 2] if durations else 0
            p95_duration = sorted(durations)[int(len(durations) * 0.95)] if durations else 0
        else:
            p50_duration = 0
            p95_duration = 0
        
        if recent_usage:
            token_counts = [u.get("total_tokens", 0) for u in recent_usage]
            avg_constraint_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        else:
            avg_constraint_tokens = 0
        
        performance = {
            "validation_performance": {
                "avg_duration_ms": summary.get("avg_validation_time_ms", 0),
                "p50_duration_ms": round(p50_duration, 2),
                "p95_duration_ms": round(p95_duration, 2),
                "failure_rate": summary.get("validation_failure_rate", 0)
            },
            "constraint_efficiency": {
                "avg_constraint_tokens": round(avg_constraint_tokens, 1),
                "auto_clean_success_rate": summary.get("auto_clean_success_rate", 0)
            },
            "violation_stats": {
                "total_violations": summary.get("total_violations", 0),
                "violations_by_type": summary.get("violations_by_type", {})
            }
        }
        
        return {
            "status": "success",
            "data": performance,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_telemetry() -> Dict[str, Any]:
    """
    Clear all telemetry data.
    
    Returns:
        Confirmation message
    """
    try:
        from backend.constraints import get_constraint_telemetry
        
        telemetry = get_constraint_telemetry()
        telemetry.clear()
        
        return {
            "status": "success",
            "message": "Telemetry data cleared",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing telemetry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for telemetry system.
    
    Returns:
        Health status
    """
    try:
        from backend.constraints import get_constraint_telemetry
        
        telemetry = get_constraint_telemetry()
        summary = telemetry.get_summary()
        
        # Determine health status
        violation_rate = (
            summary.get("total_violations", 0) / 
            max(summary.get("total_validations", 1), 1)
        )
        
        if violation_rate > 0.5:
            health_status = "degraded"
            message = "High violation rate detected"
        elif summary.get("validation_failure_rate", 0) > 0.2:
            health_status = "degraded"
            message = "High validation failure rate"
        else:
            health_status = "healthy"
            message = "All systems nominal"
        
        return {
            "status": health_status,
            "message": message,
            "metrics": {
                "total_validations": summary.get("total_validations", 0),
                "total_violations": summary.get("total_violations", 0),
                "violation_rate": round(violation_rate, 3)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@router.get("/export")
async def export_telemetry(
    hours: int = Query(24, ge=1, le=168),
    format: str = Query("json", regex="^(json|csv)$")
) -> Dict[str, Any]:
    """
    Export telemetry data for external analysis.
    
    Args:
        hours: Number of hours of data to export
        format: Export format (json or csv)
        
    Returns:
        Exported data
    """
    try:
        from backend.constraints import get_constraint_telemetry
        
        telemetry = get_constraint_telemetry()
        
        # Get data
        violations = telemetry.get_recent_violations(limit=10000)
        validations = telemetry.get_recent_validations(limit=10000)
        usage = telemetry.get_token_usage_report(limit=10000)
        
        # Filter by time window
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        def filter_by_time(items: List[Dict], time_field: str = "timestamp"):
            return [
                item for item in items
                if datetime.fromisoformat(item.get(time_field, "").replace('Z', '+00:00')) >= cutoff
            ]
        
        violations_filtered = filter_by_time(violations)
        validations_filtered = filter_by_time(validations)
        usage_filtered = filter_by_time(usage)
        
        export_data = {
            "export_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "hours": hours,
                "format": format
            },
            "summary": telemetry.get_summary(),
            "violations": violations_filtered,
            "validations": validations_filtered,
            "token_usage": usage_filtered
        }
        
        # Note: CSV export format is reserved for future implementation
        # Currently only JSON format is supported
        if format == "csv":
            raise HTTPException(
                status_code=400,
                detail="CSV export format is not yet implemented. Use format='json' (default)."
            )
        
        return {
            "status": "success",
            "data": export_data,
            "record_counts": {
                "violations": len(violations_filtered),
                "validations": len(validations_filtered),
                "token_usage": len(usage_filtered)
            }
        }
    except Exception as e:
        logger.error(f"Error exporting telemetry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
