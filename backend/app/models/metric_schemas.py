"""Pydantic schemas for real-time performance measurements, memory analysis, and accuracy validation."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class PipelineStageLatencies(BaseModel):
    """Measured latencies in milliseconds for each perception pipeline stage using time.perf_counter()."""
    lidar_preprocessing_ms: float = Field(..., ge=0.0, description="LiDAR ingestion, validation and range-gating latency (ms)")
    salsanext_inference_ms: float = Field(..., ge=0.0, description="SalsaNext/Fast-FRNet neural forward pass and unprojection latency (ms)")
    terrain_analysis_ms: float = Field(..., ge=0.0, description="Geometric slope, roughness, and curb elevation analysis latency (ms)")
    object_detection_tracking_ms: float = Field(..., ge=0.0, description="DBSCAN geometric clustering and Kalman MOT latency (ms)")
    adaptive_grid_ms: float = Field(..., ge=0.0, description="2.5D Adaptive variable-resolution grid synthesis latency (ms)")
    total_latency_ms: float = Field(..., ge=0.0, description="Total end-to-end processing latency (ms)")

    # Convenience aliases
    preprocessing_ms: Optional[float] = None
    inference_ms: Optional[float] = None
    terrain_ms: Optional[float] = None
    clustering_ms: Optional[float] = None
    grid_ms: Optional[float] = None
    total_ms: Optional[float] = None

    def model_post_init(self, __context: Any) -> None:
        if self.preprocessing_ms is None:
            self.preprocessing_ms = self.lidar_preprocessing_ms
        if self.inference_ms is None:
            self.inference_ms = self.salsanext_inference_ms
        if self.terrain_ms is None:
            self.terrain_ms = self.terrain_analysis_ms
        if self.clustering_ms is None:
            self.clustering_ms = self.object_detection_tracking_ms
        if self.grid_ms is None:
            self.grid_ms = self.adaptive_grid_ms
        if self.total_ms is None:
            self.total_ms = self.total_latency_ms


class SpatialBounds(BaseModel):
    """Spatial bounding box in sensor coordinates (meters)."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    span_x: float
    span_y: float
    span_z: float
    volume_m3: float


class AdaptiveGridResourceMetrics(BaseModel):
    """Real memory and storage footprint comparison against uniform 3D voxel analytical baseline."""
    input_point_count: int = Field(..., ge=0, description="Input raw LiDAR point count")
    fine_cells_count: int = Field(..., ge=0, description="Populated fine resolution cells count (0.5m)")
    medium_cells_count: int = Field(..., ge=0, description="Populated medium resolution cells count (1.0m)")
    coarse_cells_count: int = Field(..., ge=0, description="Populated coarse resolution cells count (2.0m)")
    total_cells_count: int = Field(..., ge=0, description="Total populated 2.5D grid cells")
    adaptive_grid_bytes: int = Field(..., ge=0, description="Estimated in-memory bytes occupied by stored adaptive cells")
    serialized_map_bytes: int = Field(..., ge=0, description="Serialized JSON export size in bytes")
    uniform_baseline_bytes: int = Field(..., ge=0, description="Uniform high-resolution 3D baseline analytical memory (bytes)")
    uniform_voxel_size_m: float = Field(0.5, gt=0.0, description="Voxel size used for uniform baseline comparison (meters)")
    uniform_bytes_per_voxel: int = Field(4, gt=0, description="Documented bytes per voxel assumption (4 bytes = occupancy float32)")
    uniform_total_voxels: int = Field(..., ge=0, description="Total 3D voxels in bounding volume at uniform resolution")
    memory_reduction_percentage: float = Field(..., ge=0.0, le=100.0, description="Calculated memory reduction percentage")
    spatial_bounds: SpatialBounds = Field(..., description="Actual spatial bounds used for voxel baseline")
    methodology_note: str = Field(
        "Uniform 3D baseline is calculated as (span_x/voxel_size) * (span_y/voxel_size) * (span_z/voxel_size) * bytes_per_voxel using real scan bounding box.",
        description="Transparent methodology statement",
    )


class ClassAccuracyMetric(BaseModel):
    """Point-wise accuracy, precision, recall, and IoU for a discrete semantic class or grouped category."""
    class_name: str
    class_id: Optional[int] = None
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    iou: float = Field(..., ge=0.0, le=1.0)
    support_points: int = Field(..., ge=0, description="Total ground truth points of this class")


class DistanceBinMetric(BaseModel):
    """Accuracy metrics grouped by point Euclidean distance from sensor origin."""
    bin_range: str = Field(..., description="Range identifier: 0-10m, 10-30m, 30-50m, 50-100m")
    min_dist_m: float = Field(..., ge=0.0)
    max_dist_m: float = Field(..., gt=0.0)
    point_count: int = Field(..., ge=0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    mean_iou: float = Field(..., ge=0.0, le=1.0)


class AccuracyValidationMetrics(BaseModel):
    """Validation metrics computed strictly when ground-truth labels exist."""
    accuracy_available: bool = Field(..., description="True only if matching ground truth .label was evaluated")
    reason_unavailable: Optional[str] = Field(None, description="Explanation when ground truth is absent")
    overall_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall point-wise classification accuracy")
    mean_iou: Optional[float] = Field(None, ge=0.0, le=1.0, description="Mean Intersection-over-Union across present classes")
    evaluated_points_count: Optional[int] = Field(None, ge=0, description="Points with non-zero ground truth evaluated")
    per_class_metrics: List[ClassAccuracyMetric] = Field(default_factory=list)
    object_class_summaries: Dict[str, ClassAccuracyMetric] = Field(default_factory=dict)
    distance_bins: List[DistanceBinMetric] = Field(default_factory=list)


class FramePerformanceMetrics(BaseModel):
    """Complete performance and validation record for a single processed LiDAR frame."""
    frame_id: str
    session_id: Optional[str] = None
    timestamp: str
    device_used: str = Field(..., description="Hardware device: cpu or cuda (e.g. NVIDIA RTX)")
    model_name: str = Field("SalsaNext", description="Semantic segmentation model architecture")
    timings: PipelineStageLatencies
    actual_fps: float = Field(..., ge=0.0, description="Actual processed frames per elapsed second")
    resource_metrics: Optional[AdaptiveGridResourceMetrics] = None
    accuracy_metrics: Optional[AccuracyValidationMetrics] = None


class SessionPerformanceMetrics(BaseModel):
    """Aggregated performance metrics across consecutive frames in a replay session."""
    session_id: str
    sequence_name: str
    total_frames_processed: int = Field(..., ge=0)
    average_timings: PipelineStageLatencies
    average_fps: float = Field(..., ge=0.0)
    average_memory_reduction_percentage: float = Field(..., ge=0.0, le=100.0)
    total_points_processed: int = Field(..., ge=0)
    total_cells_generated: int = Field(..., ge=0)
    session_accuracy: AccuracyValidationMetrics
    recent_frame_metrics: List[FramePerformanceMetrics] = Field(default_factory=list)
    timestamp: str
