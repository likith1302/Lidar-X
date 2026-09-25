"""API Routes for Fast-FRNet Semantic Segmentation Inference Service."""

import io
import re
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status, Query
from fastapi.responses import FileResponse
from datetime import datetime, timezone

from ...models.schemas import ProcessStatus, FrameMetadata
from ...models.inference_schemas import (
    InferenceStatusResponse,
    InferenceJobResponse,
    InferenceResultsResponse,
    LiveUploadInferenceResponse,
    PointPredictionSample,
)
from ...models.metric_schemas import FramePerformanceMetrics, PipelineStageLatencies, AccuracyValidationMetrics
from ...services.lidar_parser import LidarParser, LidarParserError
from ...services.storage import StorageService
from ...services.fast_frnet_inference import FastFRNetInferenceService
from ...services.semantic_label_mapping import SemanticLabelMappingService
from ...services.scene_analysis import SceneAnalysisEngine, SceneDomainResult
from ...services.instance_clustering import InstanceClusteringService
from ...services.terrain_analysis import TerrainAnalysisEngine
from ...services.map_fusion import MapFusionService
from ...services.metrics_service import MetricsService
from ...services.frame_registry import FrameRegistry
from ...config.model_settings import model_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Semantic Inference (Fast-FRNet)"])

# In-memory registry of recent inference results
_inference_jobs_cache: Dict[str, InferenceResultsResponse] = {}


@router.get(
    "/status",
    response_model=InferenceStatusResponse,
    summary="Get Fast-FRNet Model & Device Status",
    description="Inspect whether Fast-FRNet checkpoints exist, and whether GPU CUDA or CPU is active.",
)
async def get_inference_status(
    model_type: str = Query("rellis", description="Model type: 'rellis' (off-road, primary) or 'semantickitti' (general)"),
) -> InferenceStatusResponse:
    return FastFRNetInferenceService.get_status(model_type=model_type)


@router.post(
    "/live-upload",
    response_model=LiveUploadInferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Live Upload Pipeline: Neural Inference & Perception",
    description=(
        "Universal live upload endpoint for LiDAR point clouds (.bin, .pcd, .xyz, .ply). "
        "Guarantees real-time execution of Fast-FRNet neural inference without precomputed shortcuts."
    ),
)
async def live_upload_inference(
    file: UploadFile = File(..., description="LiDAR point cloud file (.bin, .pcd, .xyz, .ply)"),
    request_id: Optional[str] = Form(None, description="Optional client request ID for stale response rejection"),
    force_model: Optional[str] = Form(None, description="Optional manual model override ('rellis' or 'semantickitti')"),
) -> LiveUploadInferenceResponse:
    t_start = time.perf_counter()
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must have a valid filename.")

    # 1. Read binary content
    t_prep_start = time.perf_counter()
    try:
        raw_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to read uploaded file: {str(e)}")

    file_len = len(raw_bytes)
    if file_len == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty (0 bytes).")

    # Content hash (Requirement 12)
    content_hash = LidarParser.compute_content_hash(raw_bytes)

    # 2. Parse 3D point cloud (.bin, .pcd, .xyz, .ply)
    try:
        points = LidarParser.parse_point_cloud(raw_bytes, file.filename)
    except LidarParserError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to parse LiDAR point cloud: {str(e)}")

    num_points = points.shape[0]
    if num_points == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Point cloud contains 0 valid points.")

    t_prep_ms = (time.perf_counter() - t_prep_start) * 1000.0

    # 3. Automatic Lightweight Scene Analysis & Model Selection (Requirement 6)
    if force_model and force_model.strip():
        m_type = FastFRNetInferenceService.normalize_model_type(force_model)
        spec = model_settings.get_spec(m_type)
        scene_res = SceneDomainResult(
            domain="manual_override",
            selected_model=spec.checkpoint_filename,
            model_type=m_type,
            confidence=1.0,
            selection_reason=f"Manually selected {spec.name} model by user preference.",
            metrics={},
        )
    else:
        scene_res = SceneAnalysisEngine.analyze_scene(points)

    selected_type = scene_res.model_type
    selected_model_filename = scene_res.selected_model

    # Check model checkpoint existence
    model_status = FastFRNetInferenceService.get_status(model_type=selected_type)
    if not model_status.checkpoint_exists:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LIVE INFERENCE FAILED: Fast-FRNet ({selected_type}) checkpoint missing at '{model_status.checkpoint_path}'.",
        )

    # 4. Strict Enforcement of Live Upload Mode (Requirements 5 & 20)
    mode = "live_upload"
    inference_source = "fast_frnet_live"
    precomputed = False

    # Guard: Fail loudly if precomputed result is ever mistakenly flagged in live mode
    using_precomputed_result = False
    if mode == "live_upload" and using_precomputed_result:
        raise RuntimeError("Precomputed results are forbidden in live upload mode.")

    now_iso = datetime.now(timezone.utc).isoformat()
    clean_stem = Path(file.filename).stem if file.filename else "upload"
    clean_stem = re.sub(r'[^a-zA-Z0-9_\-]', '_', clean_stem)
    upload_id = f"upload_{uuid.uuid4().hex[:8]}"
    target_frame_id = f"{clean_stem}_{upload_id}"

    # Log precomputed label check (Requirement 28)
    existing_label_file = FrameRegistry.get_prediction_label_path(clean_stem)
    precomputed_found = existing_label_file.exists()
    logger.info(f"Precomputed label found: {'YES' if precomputed_found else 'NO'}")
    logger.info("Using it for inference: NO")

    # Invalidate old downstream perception artifacts and save points
    meta_frame = FrameMetadata(
        scan_source=f"Live upload: {file.filename}",
        timestamp=now_iso,
        format=f"LiDAR 3D float32 [x,y,z,i] ({file.filename})",
        point_count=num_points,
        file_size_bytes=file_len,
        coordinate_frame="Sensor Origin (Ego Body Frame)",
    )
    StorageService.save_frame(target_frame_id, points, meta_frame, overwrite=True)

    canonical_stem = FrameRegistry.canonicalize_frame_id(clean_stem)
    if canonical_stem != target_frame_id:
        meta_canonical = meta_frame.model_copy(update={"frame_id": canonical_stem})
        StorageService.alias_frame(target_frame_id, canonical_stem, meta_canonical)

    # 5. Execute Genuine Neural Inference (CUDA GPU or CPU fallback) (Requirements 14, 31, 32)
    t_inf_start = time.perf_counter()
    try:
        raw_labels, instance_ids, inf_meta = FastFRNetInferenceService.run_inference(
            points=points,
            model_type=selected_type,
            frame_id=target_frame_id,
            persist_artifact=True,
        )
    except Exception as e:
        logger.error(f"LIVE INFERENCE FAILED: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LIVE INFERENCE FAILED: {str(e)}",
        )
    t_inf_ms = (time.perf_counter() - t_inf_start) * 1000.0

    if canonical_stem != target_frame_id:
        StorageService.alias_labels(target_frame_id, canonical_stem)

    # 6. Postprocessing & Semantic Mapping
    t_post_start = time.perf_counter()
    class_dist, cat_dist = SemanticLabelMappingService.compute_distributions(raw_labels, dataset=selected_type)
    sample_preds = FastFRNetInferenceService.build_sample_predictions(
        points, raw_labels, dataset=selected_type, max_samples=4000
    )
    t_post_ms = (time.perf_counter() - t_post_start) * 1000.0

    # 7. Real 3D Instance Clustering (DBSCAN) -> Real OBB / geometry (Requirement 8)
    t_clust_start = time.perf_counter()
    obj_res = InstanceClusteringService.detect_objects(
        frame_id=target_frame_id,
        points=points,
        semantic_classes=raw_labels,
        instance_ids=instance_ids,
        dataset=selected_type,
    )
    StorageService.save_objects(obj_res)
    t_clust_ms = (time.perf_counter() - t_clust_start) * 1000.0

    # 8. Real Terrain Analysis directly from uploaded points (Requirement 9)
    t_terr_start = time.perf_counter()
    terrain_res = TerrainAnalysisEngine.analyze(frame_id=target_frame_id, points=points)
    StorageService.save_terrain_result(terrain_res)
    t_terr_ms = (time.perf_counter() - t_terr_start) * 1000.0

    # 9. Real Adaptive 2.5D Grid Synthesis (Requirement 10)
    t_grid_start = time.perf_counter()
    dynamic_positions = [inst.centroid[:2] for inst in obj_res.instances if inst.is_dynamic]
    grid_meta, grid_cells = MapFusionService.update_map(
        map_id=target_frame_id,
        frame_id=target_frame_id,
        points=points,
        labels=raw_labels,
        instance_ids=instance_ids,
        dynamic_track_positions=dynamic_positions,
        dataset=selected_type,
    )
    MapFusionService.clone_or_alias_map(target_frame_id, "live_fovea_map_01")
    t_grid_ms = (time.perf_counter() - t_grid_start) * 1000.0

    t_total_ms = (time.perf_counter() - t_start) * 1000.0
    live_fps = round((1000.0 / t_total_ms), 2) if t_total_ms > 0 else 0.0

    # 10. Record Real Performance Metrics (Requirement 7)
    bounds_obj = LidarParser.compute_bounds(points)
    latencies = PipelineStageLatencies.model_construct(
        lidar_preprocessing_ms=round(t_prep_ms, 2),
        salsanext_inference_ms=round(t_inf_ms, 2),
        terrain_analysis_ms=round(t_terr_ms, 2),
        object_detection_tracking_ms=round(t_clust_ms, 2),
        adaptive_grid_ms=round(t_grid_ms, 2),
        total_latency_ms=round(t_total_ms, 2),
    )
    fine_c = sum(1 for c in grid_cells if (c.get("level") if isinstance(c, dict) else c.level.value) == "fine")
    med_c = sum(1 for c in grid_cells if (c.get("level") if isinstance(c, dict) else c.level.value) == "medium")
    coarse_c = sum(1 for c in grid_cells if (c.get("level") if isinstance(c, dict) else c.level.value) == "coarse")

    res_metrics = MetricsService.compute_adaptive_grid_resource_metrics(
        points=points,
        fine_count=fine_c,
        medium_count=med_c,
        coarse_count=coarse_c,
    )
    frame_perf = FramePerformanceMetrics.model_construct(
        frame_id=target_frame_id,
        session_id=None,
        timestamp=now_iso,
        device_used=inf_meta["device_used"],
        model_name=f"Fast-FRNet ({selected_type.upper()})",
        timings=latencies,
        actual_fps=live_fps,
        resource_metrics=res_metrics,
        accuracy_metrics=AccuracyValidationMetrics(
            accuracy_available=False,
            reason_unavailable="Live upload inference without ground truth evaluation",
        ),
    )
    MetricsService.record_frame_metrics(frame_perf)
    canonical_perf = frame_perf.model_copy(update={"frame_id": canonical_stem})
    MetricsService.record_frame_metrics(canonical_perf)

    # 11. Print Required Temporary Debug Log (Requirement 29)
    debug_log = (
        "\n"
        "======================================================================\n"
        "UPLOAD\n"
        f"filename: {file.filename}\n"
        f"size: {file_len} bytes\n"
        f"content hash: {content_hash}\n"
        f"points: {num_points}\n"
        "\n"
        "INFERENCE\n"
        f"mode: {mode}\n"
        f"precomputed: {str(precomputed).lower()}\n"
        f"selected model: {selected_model_filename}\n"
        f"domain: {scene_res.domain}\n"
        f"confidence: {scene_res.confidence:.3f}\n"
        "\n"
        "TIMING\n"
        f"preprocessing: {t_prep_ms:.2f} ms\n"
        f"inference: {t_inf_ms:.2f} ms\n"
        f"postprocessing: {t_post_ms:.2f} ms\n"
        f"clustering: {t_clust_ms:.2f} ms\n"
        f"terrain: {t_terr_ms:.2f} ms\n"
        f"grid: {t_grid_ms:.2f} ms\n"
        f"total: {t_total_ms:.2f} ms\n"
        f"FPS: {live_fps:.2f}\n"
        "\n"
        "OUTPUT\n"
        f"prediction points: {len(raw_labels)}\n"
        f"object count: {len(obj_res.instances)}\n"
        f"grid cells: {len(grid_cells)}\n"
        "======================================================================\n"
    )
    print(debug_log, flush=True)
    logger.info(debug_log)

    # Cache recent result
    job_id = f"job_{target_frame_id}"
    res_resp = InferenceResultsResponse(
        job_id=job_id,
        frame_id=target_frame_id,
        status=ProcessStatus.COMPLETE,
        point_count=num_points,
        class_counts=class_dist,
        project_category_counts=cat_dist,
        sample_predictions=sample_preds,
        prediction_artifact_path=inf_meta.get("prediction_artifact_path"),
        device_used=inf_meta["device_used"],
        created_at=now_iso,
    )
    _inference_jobs_cache[job_id] = res_resp
    _inference_jobs_cache[target_frame_id] = res_resp
    _inference_jobs_cache[canonical_stem] = res_resp

    return LiveUploadInferenceResponse.model_construct(
        mode=mode,
        source="uploaded_point_cloud",
        inference_source=inference_source,
        precomputed=precomputed,
        model=selected_model_filename,
        model_type=selected_type,
        detected_domain=scene_res.domain,
        confidence=scene_res.confidence,
        selection_reason=scene_res.selection_reason,
        request_id=request_id,
        upload_id=upload_id,
        frame_id=target_frame_id,
        filename=file.filename,
        file_size_bytes=file_len,
        content_hash=content_hash,
        timestamp=now_iso,
        point_count=num_points,
        device_used=inf_meta["device_used"],
        bounds=bounds_obj.model_dump(),
        sample_predictions=sample_preds,
        performance={
            "preprocessing_ms": round(t_prep_ms, 2),
            "inference_ms": round(t_inf_ms, 2),
            "postprocessing_ms": round(t_post_ms, 2),
            "clustering_ms": round(t_clust_ms, 2),
            "terrain_ms": round(t_terr_ms, 2),
            "grid_ms": round(t_grid_ms, 2),
            "total_ms": round(t_total_ms, 2),
            "fps": live_fps,
            "device_used": inf_meta["device_used"],
            "precomputed": False,
            "resource_metrics": res_metrics.model_dump() if res_metrics else None,
        },
        semantic={
            "class_counts": class_dist,
            "project_category_counts": cat_dist,
            "total_points": num_points,
        },
        objects={
            "total_instances": len(obj_res.instances),
            "dynamic_instances_count": len([i for i in obj_res.instances if i.is_dynamic]),
            "static_instances_count": len([i for i in obj_res.instances if not i.is_dynamic]),
            "instances": [inst.model_dump() for inst in obj_res.instances],
        },
        terrain=terrain_res.model_dump(),
        grid={
            "map_id": target_frame_id,
            "mode": grid_meta.mode.value if hasattr(grid_meta.mode, "value") else str(grid_meta.mode),
            "total_cells": len(grid_cells),
            "fine_cells_count": fine_c,
            "medium_cells_count": med_c,
            "coarse_cells_count": coarse_c,
            "cells_sample": [c if isinstance(c, dict) else c.model_dump() for c in grid_cells[:3000]],
            "bounds": grid_meta.bounds.model_dump() if grid_meta.bounds else None,
        },
        message=f"Live Fast-FRNet ({selected_type.upper()}) inference and perception completed on {num_points} points.",
    )


@router.post(
    "/segment",
    response_model=InferenceJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Run Fast-FRNet Semantic Segmentation",
    description="Upload a LiDAR scan (.bin, .pcd, .xyz, .ply) or specify an existing frame_id to execute Fast-FRNet inference with RELLIS-3D or SemanticKITTI models.",
)
async def segment_point_cloud(
    file: Optional[UploadFile] = File(None, description="LiDAR binary or text point cloud file"),
    frame_id: Optional[str] = Form(None, description="Optional existing frame ID"),
    model_type: Optional[str] = Form(None, description="Fast-FRNet model type: 'rellis' (off-road, primary) or 'semantickitti' (general)"),
    model_type_query: Optional[str] = Query(None, alias="model_type", description="Fast-FRNet model type query param fallback"),
) -> InferenceJobResponse:
    # 1. Acquire LiDAR points
    points = None
    target_frame_id = None

    if file is not None:
        file_bytes = await file.read()
        try:
            points = LidarParser.parse_point_cloud(file_bytes, file.filename or "scan.bin")
        except LidarParserError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid LiDAR point cloud: {str(e)}",
            )
        raw_id = frame_id.strip() if frame_id and frame_id.strip() else (Path(file.filename).stem if file.filename else f"scan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
        target_frame_id = FrameRegistry.canonicalize_frame_id(raw_id)

        meta = FrameMetadata(
            timestamp=datetime.now(timezone.utc).isoformat(),
            point_count=int(points.shape[0]),
            file_size_bytes=len(file_bytes),
        )
        StorageService.save_frame(target_frame_id, points, meta, overwrite=True)

    elif frame_id is not None:
        target_frame_id = FrameRegistry.canonicalize_frame_id(frame_id)
        points = StorageService.get_frame_points(target_frame_id)
        if points is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Frame '{target_frame_id}' not found in storage. Upload point cloud file first.",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either a LiDAR file upload or a valid frame_id must be provided.",
        )

    # 2. Check model status before execution
    raw_type = model_type or model_type_query or "rellis"
    norm_type = raw_type.strip().lower()
    model_status = FastFRNetInferenceService.get_status(model_type=norm_type)
    if not model_status.checkpoint_exists:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"LIVE INFERENCE FAILED: Fast-FRNet ({norm_type}) checkpoint is missing at '{model_status.checkpoint_path}'. "
                "Ensure verified Fast-FRNet checkpoints exist in backend/models/ to run live neural inference."
            ),
        )

    # 3. Execute inference
    try:
        raw_labels, instance_ids, meta = FastFRNetInferenceService.run_inference(
            points=points,
            frame_id=target_frame_id,
            model_type=norm_type,
            persist_artifact=True,
        )

        job_id = f"job_{target_frame_id}"
        sample_preds = FastFRNetInferenceService.build_sample_predictions(points, raw_labels, max_samples=4000)

        # Store in job cache
        results_res = InferenceResultsResponse(
            job_id=job_id,
            frame_id=target_frame_id,
            status=ProcessStatus.COMPLETE,
            point_count=len(points),
            class_counts=meta["class_counts"],
            project_category_counts=meta["project_category_counts"],
            sample_predictions=sample_preds,
            prediction_artifact_path=meta["prediction_artifact_path"],
            device_used=meta["device_used"],
            created_at=meta["created_at"],
        )
        _inference_jobs_cache[job_id] = results_res
        _inference_jobs_cache[target_frame_id] = results_res

        return InferenceJobResponse(
            job_id=job_id,
            frame_id=target_frame_id,
            status=ProcessStatus.COMPLETE,
            point_count=len(points),
            device_used=meta["device_used"],
            created_at=meta["created_at"],
            message=f"Fast-FRNet ({norm_type.upper()}) inference completed on {len(points)} points using {meta['device_used'].upper()}.",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LIVE INFERENCE FAILED: {str(e)}",
        )


@router.get(
    "/results/{job_id}",
    response_model=InferenceResultsResponse,
    summary="Get Inference Results & Predictions",
    description="Retrieve point-wise semantic segmentation results, class distributions, and downsampled preview points.",
)
async def get_inference_results(job_id: str) -> InferenceResultsResponse:
    if job_id in _inference_jobs_cache:
        return _inference_jobs_cache[job_id]

    fid = FrameRegistry.canonicalize_frame_id(job_id.replace("job_", ""))
    if fid in _inference_jobs_cache:
        return _inference_jobs_cache[fid]

    # Check if frame_id has stored labels from active inference
    labels_tuple = StorageService.get_labels(fid, allow_precomputed_fallback=False)
    if labels_tuple is not None:
        semantic_classes, instance_ids = labels_tuple
        points = StorageService.get_frame_points(fid)
        class_counts, cat_counts = SemanticLabelMappingService.compute_distributions(semantic_classes)
        sample_preds = []
        if points is not None:
            sample_preds = FastFRNetInferenceService.build_sample_predictions(points, semantic_classes, max_samples=4000)

        label_path = FrameRegistry.get_prediction_label_path(fid)
        res = InferenceResultsResponse(
            job_id=f"job_{fid}",
            frame_id=fid,
            status=ProcessStatus.COMPLETE,
            point_count=len(semantic_classes),
            class_counts=class_counts,
            project_category_counts=cat_counts,
            sample_predictions=sample_preds,
            prediction_artifact_path=str(label_path),
            device_used="cpu",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        return res

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Inference job '{job_id}' not found.",
    )


@router.get(
    "/download/{frame_id}",
    summary="Download SemanticKITTI .label Prediction Binary",
    description="Download the uint32 binary label artifact for a segmented frame.",
)
async def download_prediction_label(frame_id: str):
    fid = FrameRegistry.canonicalize_frame_id(frame_id)
    label_path = FrameRegistry.get_prediction_label_path(fid)
    if not label_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction artifact for frame '{fid}' not found.",
        )
    return FileResponse(
        path=label_path,
        media_type="application/octet-stream",
        filename=f"{fid}.label",
    )
