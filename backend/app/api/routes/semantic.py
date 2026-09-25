from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, status

from ...config import settings
from ...models.schemas import ProcessStatus
from ...models.object_schemas import (
    SemanticLabelUploadResponse,
    PredictionImportPayload,
    SemanticFrameResponse,
    ModelProviderStatus,
)
from ...services.semantic_mapping import SemanticMappingService, SemanticMappingError
from ...services.model_provider import ModelProviderService
from ...services.storage import StorageService
from ...services.frame_registry import FrameRegistry

router = APIRouter()


@router.post(
    "/upload-labels",
    response_model=SemanticLabelUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload SemanticKITTI-compatible .label binary file",
    tags=["Semantic AI"],
)
async def upload_label_file(
    frame_id: Optional[str] = Query(None, description="Target LiDAR frame identifier. If omitted, derives from filename stem."),
    file: UploadFile = File(..., description="SemanticKITTI uint32 .label file"),
) -> SemanticLabelUploadResponse:
    """Accept and validate a SemanticKITTI .label file for an existing LiDAR frame."""
    raw_id = frame_id.strip() if frame_id and frame_id.strip() else (Path(file.filename).stem if file.filename else None)
    if not raw_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Frame ID must be provided or derived from uploaded filename.",
        )
    target_frame_id = FrameRegistry.canonicalize_frame_id(raw_id)

    points = StorageService.get_frame_points(target_frame_id)
    if points is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target frame '{target_frame_id}' not found. Please upload the .bin frame before uploading its labels.",
        )

    expected_pts = points.shape[0]

    try:
        raw_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read .label file: {str(e)}",
        )

    try:
        sem_classes, inst_ids = SemanticMappingService.parse_kitti_label_bin(raw_bytes)
    except SemanticMappingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if len(sem_classes) != expected_pts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Label count mismatch: .label file contains {len(sem_classes)} labels, "
                f"but frame '{target_frame_id}' has {expected_pts} points. Point and label counts must match exactly."
            ),
        )

    # Persist parsed labels as NPZ and save raw .label binary
    StorageService.save_labels(target_frame_id, sem_classes, inst_ids)
    pred_path = FrameRegistry.get_prediction_label_path(target_frame_id)
    FrameRegistry.atomic_save_bytes(pred_path, raw_bytes)

    class_dist, cat_dist = SemanticMappingService.compute_distributions(sem_classes)

    return SemanticLabelUploadResponse(
        frame_id=target_frame_id,
        label_count=len(sem_classes),
        class_distribution=class_dist,
        category_distribution=cat_dist,
        status=ProcessStatus.COMPLETE,
        message=f"Successfully associated {len(sem_classes)} semantic labels with frame '{target_frame_id}'.",
    )


@router.post(
    "/import-predictions",
    response_model=SemanticLabelUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Import external predictions (e.g. SalsaNext)",
    tags=["Semantic AI"],
)
async def import_predictions(payload: PredictionImportPayload) -> SemanticLabelUploadResponse:
    """Import externally computed semantic predictions (e.g. from Kaggle/GPU SalsaNext inference)."""
    fid = FrameRegistry.canonicalize_frame_id(payload.frame_id)
    points = StorageService.get_frame_points(fid)
    if points is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target frame '{fid}' not found. Please upload the frame first.",
        )

    expected_pts = points.shape[0]

    try:
        sem_classes, inst_ids = ModelProviderService.process_imported_predictions(
            payload, expected_point_count=expected_pts
        )
    except SemanticMappingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    StorageService.save_labels(fid, sem_classes, inst_ids)
    class_dist, cat_dist = SemanticMappingService.compute_distributions(sem_classes)

    return SemanticLabelUploadResponse(
        frame_id=fid,
        label_count=len(sem_classes),
        class_distribution=class_dist,
        category_distribution=cat_dist,
        status=ProcessStatus.COMPLETE,
        message=f"Imported {payload.model_name} predictions for {len(sem_classes)} points.",
    )


@router.get(
    "/{frame_id}",
    response_model=SemanticFrameResponse,
    summary="Get semantic labels and labeled point preview for a frame",
    tags=["Semantic AI"],
)
async def get_semantic_frame(
    frame_id: str,
    sample_points: int = Query(4000, ge=100, le=20000),
) -> SemanticFrameResponse:
    """Retrieve stored semantic annotations and sample labeled points for UI preview."""
    fid = FrameRegistry.canonicalize_frame_id(frame_id)
    points = StorageService.get_frame_points(fid)
    if points is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Frame '{fid}' not found.",
        )

    labels_data = StorageService.get_labels(fid)
    now_iso = datetime.now(timezone.utc).isoformat()

    if labels_data is None:
        return SemanticFrameResponse(
            frame_id=fid,
            point_count=int(points.shape[0]),
            class_counts={},
            project_category_counts={},
            sample_labeled_points=[],
            model_provider_status=ModelProviderStatus.PREDICTION_UNAVAILABLE,
            created_at=now_iso,
        )

    sem_classes, inst_ids = labels_data
    class_dist, cat_dist = SemanticMappingService.compute_distributions(sem_classes)
    sample_labeled = SemanticMappingService.build_sample_labeled_points(
        points=points,
        semantic_classes=sem_classes,
        instance_ids=inst_ids,
        max_points=sample_points,
    )

    return SemanticFrameResponse(
        frame_id=fid,
        point_count=int(points.shape[0]),
        class_counts=class_dist,
        project_category_counts=cat_dist,
        sample_labeled_points=sample_labeled,
        model_provider_status=ModelProviderStatus.EXTERNAL_PREDICTIONS_LOADED,
        created_at=now_iso,
    )
