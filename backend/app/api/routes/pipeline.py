"""API Routes for strict pipeline gate status checks and automated sequential processing."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status

from ...models.schemas import ProcessStatus
from ...models.pipeline_schemas import (
    PipelineStatusResponse,
    PipelineProcessRequest,
    PipelineProcessResponse,
    StageExecutionResult,
)
from ...services.frame_registry import FrameRegistry
from ...services.storage import StorageService
from ...services.fast_frnet_inference import FastFRNetInferenceService
from ...services.salsanext_inference import SalsaNextInferenceService
from ...services.terrain_analysis import TerrainAnalysisEngine
from ...services.instance_clustering import InstanceClusteringService
from ...services.map_fusion import MapFusionService
from ...services.map_serialization import MapSerializationService
from ...config.model_settings import model_settings

router = APIRouter(prefix="/pipeline", tags=["Pipeline Orchestrator"])


@router.get(
    "/{frame_id}/status",
    response_model=PipelineStatusResponse,
    summary="Get Truthful Real-Artifact Pipeline Gate Status",
    description="Inspects real disk artifacts (points, labels, terrain, objects, maps) for the canonical frame ID.",
)
async def get_pipeline_status(frame_id: str) -> PipelineStatusResponse:
    fid = FrameRegistry.canonicalize_frame_id(frame_id)
    return FrameRegistry.get_pipeline_status(fid)


@router.post(
    "/{frame_id}/process",
    response_model=PipelineProcessResponse,
    summary="Execute Sequential Perception Pipeline Stages",
    description="Executes SalsaNext inference, terrain analysis, object detection, and 2.5D map fusion in strict dependency order.",
)
async def process_pipeline(
    frame_id: str,
    request: Optional[PipelineProcessRequest] = None,
) -> PipelineProcessResponse:
    fid = FrameRegistry.canonicalize_frame_id(frame_id)
    points = StorageService.get_frame_points(fid)
    if points is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Frame '{fid}' point cloud not found in storage. Upload frame first.",
        )

    stages = request.stages if request and request.stages else ["fast_frnet", "terrain", "objects", "adaptive_grid"]
    model_type = (request.model_type if request and request.model_type else "rellis").lower()
    executed: list[StageExecutionResult] = []
    overall_success = True

    # 1. Fast-FRNet Inference Stage (always runs live neural inference, never bypasses)
    stage_key = "fast_frnet" if "fast_frnet" in stages else ("salsanext" if "salsanext" in stages else None)
    if stage_key:
        model_st = FastFRNetInferenceService.get_status(model_type=model_type)
        if model_st.checkpoint_exists:
            try:
                sem_classes, inst_ids, meta = FastFRNetInferenceService.run_inference(
                    points=points, frame_id=fid, model_type=model_type, persist_artifact=True
                )
                labels_tuple = (sem_classes, inst_ids)
                executed.append(
                    StageExecutionResult(
                        stage=stage_key,
                        status=ProcessStatus.COMPLETE,
                        item_count=len(sem_classes),
                        message=f"Fast-FRNet ({model_type.upper()}) live inference completed on {len(points)} points.",
                        details=meta,
                    )
                )
            except Exception as e:
                executed.append(
                    StageExecutionResult(
                        stage=stage_key,
                        status=ProcessStatus.FAILED,
                        item_count=0,
                        message=f"LIVE INFERENCE FAILED: {str(e)}",
                    )
                )
                overall_success = False
        else:
            executed.append(
                StageExecutionResult(
                    stage=stage_key,
                    status=ProcessStatus.FAILED,
                    item_count=0,
                    message=f"LIVE INFERENCE FAILED: Fast-FRNet ({model_type}) checkpoint not found at {model_st.checkpoint_path}.",
                )
            )
            overall_success = False


    # 2. Terrain Analysis Stage
    if "terrain" in stages:
        try:
            terrain_res = TerrainAnalysisEngine.analyze(frame_id=fid, points=points)
            StorageService.save_terrain_result(terrain_res)
            executed.append(
                StageExecutionResult(
                    stage="terrain",
                    status=ProcessStatus.COMPLETE,
                    item_count=len(terrain_res.cells),
                    message=f"Terrain analysis completed: {len(terrain_res.cells)} cells analyzed.",
                )
            )
        except Exception as e:
            executed.append(
                StageExecutionResult(
                    stage="terrain",
                    status=ProcessStatus.FAILED,
                    item_count=0,
                    message=f"Terrain analysis failed: {str(e)}",
                )
            )
            overall_success = False

    # 3. Object Detection Stage
    if "objects" in stages:
        labels_tuple = StorageService.get_labels(fid)
        if labels_tuple is None:
            executed.append(
                StageExecutionResult(
                    stage="objects",
                    status=ProcessStatus.FAILED,
                    item_count=0,
                    message=f"Object detection requires point-wise semantic labels for frame '{fid}'. Run or import semantic segmentation first.",
                )
            )
            overall_success = False
        else:
            try:
                sem_classes, inst_ids = labels_tuple
                obj_res = InstanceClusteringService.detect_objects(
                    frame_id=fid,
                    points=points,
                    semantic_classes=sem_classes,
                    instance_ids=inst_ids,
                    dataset=model_type,
                )
                StorageService.save_objects(obj_res)
                executed.append(
                    StageExecutionResult(
                        stage="objects",
                        status=ProcessStatus.COMPLETE,
                        item_count=len(obj_res.instances),
                        message=f"Detected {len(obj_res.instances)} object instances.",
                    )
                )
            except Exception as e:
                executed.append(
                    StageExecutionResult(
                        stage="objects",
                        status=ProcessStatus.FAILED,
                        item_count=0,
                        message=f"Object detection failed: {str(e)}",
                    )
                )
                overall_success = False

    # 4. Adaptive Grid Stage
    if "adaptive_grid" in stages:
        labels_tuple = StorageService.get_labels(fid)
        sem_classes = labels_tuple[0] if labels_tuple is not None else None
        inst_ids = labels_tuple[1] if labels_tuple is not None else None

        obj_res = StorageService.get_objects(fid)
        dynamic_positions = []
        if obj_res and obj_res.instances:
            for inst in obj_res.instances:
                if inst.is_dynamic:
                    dynamic_positions.append(inst.centroid[:2])

        try:
            meta, all_cells = MapFusionService.update_map(
                map_id=fid,
                frame_id=fid,
                points=points,
                labels=sem_classes,
                instance_ids=inst_ids,
                dynamic_track_positions=dynamic_positions,
                dataset=model_type,
            )
            # Save map to disk
            map_data = MapFusionService.get_map(fid)
            if map_data:
                m_meta, m_cells, m_policy = map_data
                MapSerializationService.save_map(
                    map_id=fid,
                    mode=m_meta.mode,
                    metadata=m_meta,
                    cells=m_cells,
                    policy=m_policy,
                )
            # Alias or clone map for live_fovea_map_01 without redundant recalculation
            MapFusionService.clone_or_alias_map(fid, "live_fovea_map_01")
            executed.append(
                StageExecutionResult(
                    stage="adaptive_grid",
                    status=ProcessStatus.COMPLETE,
                    item_count=len(all_cells),
                    message=f"2.5D Adaptive grid generated with {len(all_cells)} cells.",
                )
            )
        except Exception as e:
            executed.append(
                StageExecutionResult(
                    stage="adaptive_grid",
                    status=ProcessStatus.FAILED,
                    item_count=0,
                    message=f"Adaptive grid generation failed: {str(e)}",
                )
            )
            overall_success = False

    pipe_status = FrameRegistry.get_pipeline_status(fid)
    return PipelineProcessResponse(
        frame_id=fid,
        success=overall_success,
        status=ProcessStatus.COMPLETE if overall_success else ProcessStatus.FAILED,
        executed_stages=executed,
        pipeline_status=pipe_status,
        message="All requested pipeline stages completed successfully." if overall_success else "Some pipeline stages failed or prerequisites were unmet.",
    )
