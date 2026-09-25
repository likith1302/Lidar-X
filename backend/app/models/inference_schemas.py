"""Pydantic schemas for SalsaNext Semantic Segmentation Inference Service."""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from .schemas import ProcessStatus, Point3D


class InferenceModelStatus(str, Enum):
    MODEL_AVAILABLE = "model_available"
    MODEL_MISSING = "model_missing"
    LOADING = "loading"
    READY = "ready"
    GPU_AVAILABLE = "gpu_available"
    CPU_MODE = "cpu_mode"
    FAILED = "failed"
    NOT_CONNECTED = "not_connected"


class InferenceStatusResponse(BaseModel):
    status: InferenceModelStatus
    model_name: str = "Fast-FRNet"
    active_model: str = "rellis"
    device: str = "cpu"
    device_name: str = "CPU"
    checkpoint_path: str
    checkpoint_exists: bool
    arch_config_exists: bool = True
    data_config_exists: bool = True
    num_classes: int = 20
    cuda_available: bool = False
    message: str
    rellis_checkpoint_path: Optional[str] = None
    rellis_checkpoint_exists: Optional[bool] = None
    semantickitti_checkpoint_path: Optional[str] = None
    semantickitti_checkpoint_exists: Optional[bool] = None


class PointPredictionSample(BaseModel):
    x: float
    y: float
    z: float
    intensity: float = 0.0
    raw_label_id: int
    semantic_class: str
    project_category: str


class InferenceJobResponse(BaseModel):
    job_id: str
    frame_id: str
    status: ProcessStatus
    point_count: int
    device_used: str
    created_at: str
    message: str = "Inference completed successfully"


class InferenceResultsResponse(BaseModel):
    job_id: str
    frame_id: str
    status: ProcessStatus
    point_count: int
    class_counts: Dict[str, int]
    project_category_counts: Dict[str, int]
    sample_predictions: List[PointPredictionSample] = Field(
        default_factory=list,
        description="Downsampled predicted point preview for responsive web visualization",
    )
    prediction_artifact_path: Optional[str] = None
    device_used: str
    created_at: str


class LiveUploadInferenceResponse(BaseModel):
    mode: str = "live_upload"
    source: str = "uploaded_point_cloud"
    inference_source: str = "fast_frnet_live"
    precomputed: bool = False
    model: str
    model_type: str
    detected_domain: str
    confidence: float
    selection_reason: str
    request_id: Optional[str] = None
    upload_id: str
    frame_id: str
    filename: str
    file_size_bytes: int
    content_hash: str
    timestamp: str
    point_count: int
    device_used: str
    bounds: Optional[Dict[str, float]] = None
    sample_predictions: List[PointPredictionSample] = Field(
        default_factory=list,
        description="Sample points with XYZ, intensity, and predicted classes",
    )
    performance: Dict[str, Any]
    semantic: Dict[str, Any]
    objects: Dict[str, Any]
    terrain: Dict[str, Any]
    grid: Dict[str, Any]
    message: str = "Live Fast-FRNet inference completed successfully."

