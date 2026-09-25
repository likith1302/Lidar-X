"""Pydantic schemas for LiDAR sequence replay and streaming."""

from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field
from .schemas import BoundingBox3D, PreprocessingReport, TerrainAnalysisResponse, TerrainCell
from .object_schemas import ObjectDetectionResponse, TrackingUpdateResponse, ObjectInstance, TrackedObject
from .map_schemas import MapUpdateResponse, AdaptiveGridCell
from .metric_schemas import FramePerformanceMetrics


class ReplayDataMode(str):
    PRECOMPUTED_LABELS = "precomputed_labels"
    LIVE_INFERENCE = "live_inference"
    SCANS_ONLY = "scans_only"


class PlaybackMode(str):
    OFFLINE_PRECOMPUTED_REPLAY = "offline_precomputed_replay"
    LIVE_PROCESSING = "live_processing"


class ReplayPlaybackState(str):
    READY = "ready"
    PLAYING = "playing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    NO_PREDICTIONS = "no_predictions"


class PrecomputeProgressStatus(BaseModel):
    session_id: str
    sequence_name: str
    processed_frames: int
    total_frames: int
    percent_complete: float
    is_complete: bool = False
    is_running: bool = False
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    current_stage: str = "Fast-FRNet inference"
    failed_count: int = 0
    error_message: Optional[str] = None



class ReplayDiagnostics(BaseModel):
    session_id: str
    playback_mode: str
    requested_fps: float
    current_fps: float = 0.0
    device: str = "CPU"
    total_frames: int
    current_frame: int


class SetPlaybackModeRequest(BaseModel):
    playback_mode: str = Field(..., description="offline_precomputed_replay or live_processing")


class ReplayUploadResponse(BaseModel):
    session_id: str = Field(..., description="Unique replay session identifier")
    sequence_name: str = Field(..., description="Extracted sequence identifier or folder name")
    total_frames: int = Field(..., description="Number of valid consecutive LiDAR frames in sequence")
    has_predictions: bool = Field(..., description="True if matching SalsaNext prediction labels were found")
    data_mode: str = Field(..., description="Data mode: precomputed_labels, live_inference, or scans_only")
    has_poses: bool = False
    has_calibration: bool = False
    is_manual_upload: bool = False
    semantic_source: str = Field(default="LIVE GEOMETRIC", description="LIVE GEOMETRIC, LIVE SALSANEXT, GROUND TRUTH, or PRECOMPUTED")
    playback_mode: str = Field(default="offline_precomputed_replay", description="Playback mode")
    frame_filenames: List[str] = Field(default_factory=list, description="Sorted list of scan filenames")
    status: str = Field(default="ready", description="Initial session state")
    message: str = Field(..., description="Human-readable result summary")


class ReplaySessionStatus(BaseModel):
    session_id: str
    sequence_name: str
    state: str = Field(..., description="ready, playing, paused, completed, failed, or no_predictions")
    current_frame_index: int = Field(..., ge=0, description="Current 0-indexed frame position")
    total_frames: int = Field(..., ge=0, description="Total number of frames in sequence")
    fps: float = Field(default=2.0, description="Target replay playback frames per second")
    playback_mode: str = Field(default="offline_precomputed_replay")
    data_mode: str = Field(default="precomputed_labels")
    semantic_source: str = Field(default="LIVE GEOMETRIC", description="LIVE GEOMETRIC, LIVE SALSANEXT, GROUND TRUTH, or PRECOMPUTED")
    coordinate_mode: str = Field(default="local_frame", description="local_frame or world_frame")
    has_predictions: bool = True
    has_poses: bool = False
    has_calibration: bool = False
    has_timestamps: bool = False
    is_manual_upload: bool = False
    error_message: Optional[str] = None


class ReplaySeekRequest(BaseModel):
    frame_index: int = Field(..., ge=0, description="Target 0-indexed frame index to seek to")


class ReplayStartRequest(BaseModel):
    fps: Optional[float] = Field(default=2.0, ge=0.2, le=60.0, description="Playback speed in frames per second")
    playback_mode: Optional[str] = Field(default=None, description="Playback mode: offline_precomputed_replay or live_processing")


class ReplayPointSample(BaseModel):
    x: float
    y: float
    z: float
    intensity: float = 0.5
    semantic_class: str = "unknown"
    project_category: str = "unknown"


class ReplayFrameStreamPayload(BaseModel):
    session_id: str
    sequence_name: str
    frame_index: int
    total_frames: int
    frame_filename: str
    timestamp: Optional[str] = None
    point_count: int
    points_sample: List[Any] = Field(default_factory=list, description="Downsampled 3D points for responsive rendering")
    category_distribution: Dict[str, int] = Field(default_factory=dict)
    terrain_result: Optional[TerrainAnalysisResponse] = None
    objects_result: Optional[ObjectDetectionResponse] = None
    tracking_result: Optional[TrackingUpdateResponse] = None
    map_result: Optional[MapUpdateResponse] = None
    coordinate_mode: str = "local_frame"
    map_mode: str = "local_only"
    data_mode: str = "precomputed_labels"
    semantic_source: str = "LIVE GEOMETRIC"
    state: str = "playing"
    processing_time_ms: float = 0.0
    is_available: bool = True
    performance: Optional[FramePerformanceMetrics] = None

