"""Pydantic schemas for Semantic Segmentation, Object Detection, and Tracking."""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from .schemas import ProcessStatus, Point3D, BoundingBox3D


class ModelProviderStatus(str, Enum):
    EXTERNAL_PREDICTIONS_LOADED = "external_predictions_loaded"
    INFERENCE_SERVICE_NOT_CONNECTED = "inference_service_not_connected"
    PREDICTION_UNAVAILABLE = "prediction_unavailable"


class ProjectCategory(str, Enum):
    DRIVABLE = "drivable"
    NON_DRIVABLE = "non_drivable"
    STATIC_OBSTACLE = "static_obstacle"
    DYNAMIC_OBJECT = "dynamic_object"
    VEGETATION = "vegetation"
    INFRASTRUCTURE = "infrastructure"
    UNKNOWN = "unknown"


class ExtentCategory(str, Enum):
    COMPACT_ACTOR = "compact_actor"        # Pedestrians, cyclists
    SMALL_VEHICLE = "small_vehicle"        # Cars, SUVs
    LARGE_VEHICLE = "large_vehicle"        # Trucks, buses
    SLENDER_VERTICAL = "slender_vertical"  # Poles, traffic signs
    BROAD_STRUCTURE = "broad_structure"    # Buildings, walls, fences
    IRREGULAR_CLUSTER = "irregular_cluster"


class ObservationStatus(str, Enum):
    DIRECTLY_OBSERVED = "directly_observed"
    PARTIALLY_OCCLUDED = "partially_occluded"
    SPARSE_CLUSTER = "sparse_cluster"


class TrackLifecycleState(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    TEMPORARILY_LOST = "temporarily_lost"
    EXPIRED = "expired"


class TrackingCoordinateMode(str, Enum):
    LOCAL_FRAME = "local_frame"
    WORLD_FRAME = "world_frame"


# --- Semantic Schemas ---

class SemanticPoint3D(BaseModel):
    x: float
    y: float
    z: float
    intensity: float = 0.0
    raw_label_id: int
    semantic_class: str
    project_category: str
    instance_id: int = 0


class SemanticLabelUploadResponse(BaseModel):
    frame_id: str
    label_count: int
    class_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    status: ProcessStatus
    message: str = "Semantic labels parsed and associated successfully"


class PredictionImportPayload(BaseModel):
    frame_id: str
    model_name: str = Field(default="SalsaNext", description="Deep learning model architecture name")
    labels: List[int] = Field(..., description="Array of uint32 or uint16 semantic class IDs matching point count")
    metadata: Optional[Dict[str, Any]] = None


class SemanticFrameResponse(BaseModel):
    frame_id: str
    point_count: int
    class_counts: Dict[str, int]
    project_category_counts: Dict[str, int]
    sample_labeled_points: List[SemanticPoint3D] = []
    model_provider_status: ModelProviderStatus
    created_at: str


# --- Object Instance Schemas ---

class ObjectInstance(BaseModel):
    instance_id: str
    semantic_category: ProjectCategory
    object_type: str
    centroid: List[float] = Field(..., description="[x, y, z] centroid in meters")
    bounding_box: BoundingBox3D
    dimensions: List[float] = Field(..., description="[dx, dy, dz] bounding box dimensions in meters")
    extent_category: ExtentCategory
    point_count: int
    is_dynamic: bool
    source_frame_id: str
    observation_status: ObservationStatus
    sample_points: List[Point3D] = []
    yaw: float = 0.0
    oriented_bounding_box: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    class_id: int = 0


class ObjectDetectionRequest(BaseModel):
    frame_id: str
    min_points_per_cluster: Optional[int] = Field(default=5, ge=3, le=50)
    clustering_radius_m: Optional[float] = Field(default=0.8, ge=0.2, le=3.0)


class ObjectDetectionResponse(BaseModel):
    frame_id: str
    total_instances: int
    dynamic_instances_count: int
    static_instances_count: int
    instances: List[ObjectInstance]
    status: ProcessStatus
    created_at: str


# --- Tracking Schemas ---

class TrackedPosition(BaseModel):
    frame_id: str
    timestamp: str
    position: List[float] = Field(..., description="[x, y, z] in meters")


class TrackedObject(BaseModel):
    track_id: str
    object_type: str
    semantic_category: ProjectCategory
    current_position: List[float] = Field(..., description="[x, y, z] in meters")
    estimated_velocity: List[float] = Field(..., description="[vx, vy, vz] in meters/second")
    history: List[TrackedPosition] = []
    lifecycle_state: TrackLifecycleState
    age_frames: int
    frames_since_seen: int
    coordinate_mode: TrackingCoordinateMode
    bounding_box: BoundingBox3D
    associated_instance_id: Optional[str] = None
    yaw: float = 0.0
    oriented_bounding_box: Optional[Dict[str, Any]] = None


class TrackingUpdateRequest(BaseModel):
    frame_id: str
    timestamp: Optional[str] = None
    ego_pose: Optional[List[float]] = Field(default=None, description="Optional 4x4 matrix or [x, y, yaw]")
    max_association_distance_m: Optional[float] = Field(default=2.5, ge=0.5, le=10.0)


class TrackingUpdateResponse(BaseModel):
    frame_id: str
    coordinate_mode: TrackingCoordinateMode
    active_tracks_count: int
    new_tracks_count: int
    lost_tracks_count: int
    tracks: List[TrackedObject]
    timestamp: str
