"""API endpoints for real-time performance, memory analysis, and accuracy validation metrics."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status

from ...models.metric_schemas import (
    FramePerformanceMetrics,
    SessionPerformanceMetrics,
)
from ...services.metrics_service import MetricsService

router = APIRouter(tags=["Performance & Validation Metrics"])


@router.get(
    "/latest",
    response_model=FramePerformanceMetrics,
    summary="Get Most Recent Frame Performance & Validation Metrics",
    description="Retrieve live measured timings, memory reduction vs. uniform baseline, and accuracy metrics for the latest processed frame.",
)
async def get_latest_metrics(
    session_id: Optional[str] = Query(None, description="Scope metrics to active session (Requirement 17)"),
) -> FramePerformanceMetrics:
    metric = MetricsService.get_latest_metrics(session_id=session_id)
    if metric is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No measured performance data available yet. Process or replay a frame to generate metrics.",
        )
    return metric


@router.get(
    "/frame/{frame_id}",
    response_model=FramePerformanceMetrics,
    summary="Get Performance & Validation Metrics for a Specific Frame",
    description="Retrieve recorded stage timings, uniform baseline comparison, and accuracy validation for a specific frame by ID.",
)
async def get_frame_metrics(
    frame_id: str,
    session_id: Optional[str] = Query(None, description="Scope frame lookup to active session (Requirement 18)"),
) -> FramePerformanceMetrics:
    metric = MetricsService.get_frame_metrics(frame_id=frame_id, session_id=session_id)
    if metric is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No performance metrics recorded for frame '{frame_id}'.",
        )
    return metric


@router.get(
    "/session/{session_id}",
    response_model=SessionPerformanceMetrics,
    summary="Get Aggregated Performance Metrics for Replay Session",
    description="Retrieve average stage latencies, average FPS, memory reduction, and sequence-level accuracy across all processed frames in a session.",
)
async def get_session_metrics(session_id: str) -> SessionPerformanceMetrics:
    metric = MetricsService.get_session_metrics(session_id)
    if metric is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No performance metrics recorded for replay session '{session_id}'.",
        )
    return metric
