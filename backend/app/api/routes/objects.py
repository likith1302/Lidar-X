"""Object detection, instance clustering, and multi-object tracking routes."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from ...models.schemas import ProcessStatus
from ...models.object_schemas import (
    ObjectDetectionRequest,
    ObjectDetectionResponse,
    TrackingUpdateRequest,
    TrackingUpdateResponse,
    TrackedObject,
)
from ...services.storage import StorageService
from ...services.instance_clustering import InstanceClusteringService
from ...services.object_tracking import ObjectTrackingService
from ...services.frame_registry import FrameRegistry

router = APIRouter()


@router.post(
    "/detect",
    response_model=ObjectDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect object instances from labeled LiDAR point cloud",
    tags=["Object Detection"],
)
async def detect_object_instances(request: ObjectDetectionRequest) -> ObjectDetectionResponse:
    """Run geometric instance clustering (DBSCAN) on points with semantic annotations."""
    fid = FrameRegistry.canonicalize_frame_id(request.frame_id)
    points = StorageService.get_frame_points(fid)
    if points is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Frame '{fid}' not found. Upload LiDAR frame first.",
        )

    labels_data = StorageService.get_labels(fid)
    if labels_data is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No semantic labels found for frame '{fid}'. Upload labels or import predictions before running object detection.",
        )

    sem_classes, inst_ids = labels_data

    min_pts = request.min_points_per_cluster if request.min_points_per_cluster is not None else 5
    radius = request.clustering_radius_m if request.clustering_radius_m is not None else 0.8

    response = InstanceClusteringService.detect_objects(
        frame_id=fid,
        points=points,
        semantic_classes=sem_classes,
        instance_ids=inst_ids,
        min_points_per_cluster=min_pts,
        default_radius_m=radius,
    )

    StorageService.save_objects(response)
    return response


@router.get(
    "/{frame_id}",
    response_model=ObjectDetectionResponse,
    summary="Get detected object instances for a frame",
    tags=["Object Detection"],
)
async def get_detected_objects(frame_id: str) -> ObjectDetectionResponse:
    """Retrieve previously computed object instances for a given frame."""
    fid = FrameRegistry.canonicalize_frame_id(frame_id)
    result = StorageService.get_objects(fid)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No detected object instances found for frame '{fid}'. Run POST /objects/detect first.",
        )
    return result


# --- Object Tracking Endpoints ---

tracks_router = APIRouter()


@tracks_router.post(
    "/update",
    response_model=TrackingUpdateResponse,
    status_code=status.HTTP_200_OK,
    summary="Update temporal object tracks with frame detections",
    tags=["Object Tracking"],
)
async def update_temporal_tracks(request: TrackingUpdateRequest) -> TrackingUpdateResponse:
    """Associate frame detections with existing tracks and update Kalman state estimators."""
    fid = FrameRegistry.canonicalize_frame_id(request.frame_id)
    # Try loading detected objects for this frame
    detection_res = StorageService.get_objects(fid)
    if detection_res is None:
        # If not already detected, try to auto-detect if labels exist
        points = StorageService.get_frame_points(fid)
        labels_data = StorageService.get_labels(fid)
        if points is None or labels_data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No detected objects or semantic annotations found for frame '{fid}'. Run detection first.",
            )
        sem_classes, inst_ids = labels_data
        detection_res = InstanceClusteringService.detect_objects(
            frame_id=fid,
            points=points,
            semantic_classes=sem_classes,
            instance_ids=inst_ids,
        )
        StorageService.save_objects(detection_res)

    max_dist = request.max_association_distance_m if request.max_association_distance_m is not None else 2.5

    tracking_res = ObjectTrackingService.update_tracks(
        frame_id=fid,
        detected_instances=detection_res.instances,
        timestamp=request.timestamp,
        ego_pose=request.ego_pose,
        max_distance_m=max_dist,
    )

    return tracking_res


@tracks_router.get(
    "/",
    response_model=List[TrackedObject],
    summary="List all currently active tracked objects",
    tags=["Object Tracking"],
)
async def list_active_tracks() -> List[TrackedObject]:
    """Return all active object tracks."""
    return ObjectTrackingService.get_all_tracks()


@tracks_router.get(
    "/{track_id}",
    response_model=TrackedObject,
    summary="Get single tracked object details and trajectory history",
    tags=["Object Tracking"],
)
async def get_track_details(track_id: str) -> TrackedObject:
    """Retrieve trajectory and state history for a specific track ID."""
    track = ObjectTrackingService.get_track(track_id)
    if track is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Track '{track_id}' not found.",
        )
    return track
