"""LiDAR point cloud frame ingestion and retrieval routes."""

import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, status

from ...config import settings
from ...models.schemas import (
    FrameUploadResponse,
    FrameDetailsResponse,
    FrameMetadata,
    ProcessStatus,
    JsonPointCloudPayload,
)
from ...services.lidar_parser import LidarParser, LidarParserError
from ...services.storage import StorageService
from ...services.frame_registry import FrameRegistry

router = APIRouter()


@router.post(
    "/upload",
    response_model=FrameUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload SemanticKITTI .bin LiDAR frame",
    tags=["LiDAR Ingestion"],
)
async def upload_bin_frame(
    file: UploadFile = File(..., description="SemanticKITTI .bin float32 LiDAR point cloud file"),
    frame_id: Optional[str] = Query(None, description="Optional custom frame ID. If omitted, derives from filename stem."),
) -> FrameUploadResponse:
    """Accept and validate a multipart uploaded SemanticKITTI .bin point cloud file."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    # Read binary content
    try:
        raw_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    file_len = len(raw_bytes)
    if file_len > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file size ({file_len} bytes) exceeds maximum allowable limit of {settings.MAX_UPLOAD_SIZE_BYTES} bytes.",
        )

    # Parse and validate point cloud data (.bin, .pcd, .xyz, .ply)
    try:
        points = LidarParser.parse_point_cloud(raw_bytes, file.filename)
    except LidarParserError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    raw_id = frame_id.strip() if frame_id and frame_id.strip() else Path(file.filename).stem
    target_frame_id = FrameRegistry.canonicalize_frame_id(raw_id)

    bounds = LidarParser.compute_bounds(points)
    file_ext = Path(file.filename).suffix.lower()
    metadata = FrameMetadata(
        scan_source=f"Uploaded LiDAR file: {file.filename}",
        timestamp=now_iso,
        format=f"LiDAR float32 (x,y,z,i) [{file_ext}]",
        point_count=int(points.shape[0]),
        file_size_bytes=file_len,
        coordinate_frame="Sensor Origin (Ego Body Frame: +X forward, +Y left, +Z up)",
    )

    StorageService.save_frame(target_frame_id, points, metadata, overwrite=True)

    return FrameUploadResponse(
        frame_id=target_frame_id,
        point_count=int(points.shape[0]),
        file_size_bytes=file_len,
        status=ProcessStatus.COMPLETE,
        timestamp=now_iso,
        bounds=bounds,
        message=f"Successfully parsed and ingested {points.shape[0]} points from {file.filename}.",
    )


@router.post(
    "/upload-json",
    response_model=FrameUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload JSON Point Cloud for testing",
    tags=["LiDAR Ingestion"],
)
async def upload_json_frame(payload: JsonPointCloudPayload) -> FrameUploadResponse:
    """Accept a JSON point cloud array for development and testing workflows."""
    try:
        points = LidarParser.parse_json_points(payload)
    except LidarParserError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    target_frame_id = payload.frame_id if payload.frame_id else f"frame_{uuid.uuid4().hex[:8]}"

    bounds = LidarParser.compute_bounds(points)
    metadata = FrameMetadata(
        scan_source="JSON point cloud array payload",
        timestamp=now_iso,
        format="JSON float32 points",
        point_count=int(points.shape[0]),
        file_size_bytes=points.nbytes,
        coordinate_frame="Sensor Origin (Ego Body Frame)",
    )

    StorageService.save_frame(target_frame_id, points, metadata)

    return FrameUploadResponse(
        frame_id=target_frame_id,
        point_count=int(points.shape[0]),
        file_size_bytes=points.nbytes,
        status=ProcessStatus.COMPLETE,
        timestamp=now_iso,
        bounds=bounds,
        message=f"Successfully ingested {points.shape[0]} points from JSON payload.",
    )


@router.get(
    "/{frame_id}",
    response_model=FrameDetailsResponse,
    summary="Get LiDAR frame metadata and preview points",
    tags=["LiDAR Ingestion"],
)
async def get_frame_details(
    frame_id: str,
    sample_points: int = Query(4000, ge=100, le=20000, description="Max sampled points for UI preview"),
) -> FrameDetailsResponse:
    """Fetch stored frame metadata and sampled 3D points for fast frontend rendering."""
    details = StorageService.get_frame_details(frame_id, sample_limit=sample_points)
    if details is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Frame '{frame_id}' not found in storage.",
        )
    return details


@router.get(
    "/",
    response_model=List[str],
    summary="List stored frame IDs",
    tags=["LiDAR Ingestion"],
)
async def list_frames() -> List[str]:
    """Return all stored LiDAR frame identifiers."""
    return StorageService.list_frames()
