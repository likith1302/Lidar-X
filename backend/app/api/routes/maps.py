"""API Routes for 2.5D Adaptive Variable-Resolution Map Engine."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from ...models.schemas import ProcessStatus
from ...models.map_schemas import (
    MapMetadata,
    MapUpdateRequest,
    MapUpdateResponse,
    MapExportResponse,
    MapResetResponse,
    AdaptiveGridCell,
    MapMode,
)
from ...services.storage import StorageService
from ...services.map_fusion import MapFusionService
from ...services.map_serialization import MapSerializationService

router = APIRouter(tags=["2.5D Adaptive Grid & Maps"])


@router.post(
    "/{map_id}/update",
    response_model=MapUpdateResponse,
    summary="Update 2.5D Map with LiDAR Frame",
    description="Fuses LiDAR points, semantic classes, and tracked objects into local or persistent global 2.5D grid.",
)
async def update_map_with_frame(
    map_id: str,
    payload: MapUpdateRequest,
) -> MapUpdateResponse:
    # 1. Load frame points
    points = StorageService.get_frame_points(payload.frame_id)
    if points is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Frame '{payload.frame_id}' not found in storage. Upload frame first.",
        )

    # 2. Retrieve optional semantic labels
    labels_tuple = StorageService.get_labels(payload.frame_id)
    labels = labels_tuple[0] if labels_tuple is not None else None
    instance_ids = labels_tuple[1] if labels_tuple is not None else None

    # 3. Retrieve optional detected objects for dynamic positions
    objects_res = StorageService.get_objects(payload.frame_id)
    dynamic_positions: List[List[float]] = []
    if objects_res and objects_res.instances:
        for inst in objects_res.instances:
            if inst.is_dynamic:
                dynamic_positions.append(inst.centroid[:2])

    # 4. Perform map fusion
    try:
        metadata, all_cells = MapFusionService.update_map(
            map_id=map_id,
            frame_id=payload.frame_id,
            points=points,
            labels=labels,
            instance_ids=instance_ids,
            dynamic_track_positions=dynamic_positions,
            ego_pose=payload.ego_pose,
            override_policy=payload.override_policy,
        )

        return MapUpdateResponse(
            map_id=map_id,
            mode=metadata.mode,
            status=ProcessStatus.COMPLETE,
            updated_cell_count=len(all_cells),
            metadata=metadata,
            cells_sample=all_cells,  # Full spatial coverage across all resolution zones
            message=f"Map '{map_id}' successfully updated in {metadata.mode.value} mode with {len(all_cells)} cells.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Map update failed: {str(e)}",
        )


@router.get(
    "/{map_id}",
    response_model=MapMetadata,
    summary="Get 2.5D Map Metadata",
    description="Retrieve map summary, bounds, and cell counts per resolution level.",
)
async def get_map_metadata(map_id: str) -> MapMetadata:
    map_data = MapFusionService.get_map(map_id)
    if not map_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Map '{map_id}' not found. Update map with a frame first.",
        )
    meta, _, _ = map_data
    return meta


@router.get(
    "/{map_id}/cells",
    response_model=List[AdaptiveGridCell],
    summary="Query Map Cells",
    description="Query sparse multi-resolution cells with optional resolution level or bounding box filter.",
)
async def get_map_cells(
    map_id: str,
    level: Optional[str] = Query(None, description="Filter by resolution level: fine, medium, coarse, all"),
    min_x: Optional[float] = Query(None, description="Minimum X bounding box (m)"),
    max_x: Optional[float] = Query(None, description="Maximum X bounding box (m)"),
    min_y: Optional[float] = Query(None, description="Minimum Y bounding box (m)"),
    max_y: Optional[float] = Query(None, description="Maximum Y bounding box (m)"),
    limit: int = Query(5000, ge=1, le=50000, description="Max cells to return"),
) -> List[AdaptiveGridCell]:
    cells = MapFusionService.get_cells(
        map_id=map_id,
        level=level,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        limit=limit,
    )
    if cells is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Map '{map_id}' not found.",
        )
    return cells


@router.get(
    "/{map_id}/cells/{cell_key}",
    response_model=AdaptiveGridCell,
    summary="Get Detailed Cell Attributes",
    description="Inspect geometric, semantic, traversability, and hierarchical properties for a single cell.",
)
async def get_cell_details(map_id: str, cell_key: str) -> AdaptiveGridCell:
    cell = MapFusionService.get_cell_by_key(map_id=map_id, cell_key=cell_key)
    if not cell:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cell '{cell_key}' not found in map '{map_id}'.",
        )
    return cell


@router.post(
    "/{map_id}/reset",
    response_model=MapResetResponse,
    summary="Reset Map",
    description="Clear all cells and frames for a given map instance.",
)
async def reset_map(map_id: str) -> MapResetResponse:
    success = MapFusionService.reset_map(map_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Map '{map_id}' not found.",
        )
    return MapResetResponse(
        map_id=map_id,
        status="reset_complete",
        message=f"Map '{map_id}' has been cleared successfully.",
    )


@router.get(
    "/{map_id}/export",
    response_model=MapExportResponse,
    summary="Export Complete 2.5D Map",
    description="Export full map JSON including metadata, all sparse cells, and active policy snapshot.",
)
async def export_map(map_id: str) -> MapExportResponse:
    map_data = MapFusionService.get_map(map_id)
    if not map_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Map '{map_id}' not found.",
        )
    meta, cells, policy = map_data
    
    # Save to disk as well
    MapSerializationService.save_map(
        map_id=map_id,
        mode=meta.mode,
        metadata=meta,
        cells=cells,
        policy=policy,
    )

    return MapExportResponse(
        map_id=map_id,
        mode=meta.mode,
        metadata=meta,
        cells=cells,
        policy_snapshot=policy,
    )
