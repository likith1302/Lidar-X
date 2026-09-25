"""Pydantic schemas and data contracts for LiDAR-X."""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# --- Enums ---

class ProcessStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    NOT_AVAILABLE = "not_available"


class ObservationState(str, Enum):
    DIRECTLY_OBSERVED = "directly_observed"
    PARTIALLY_OCCLUDED = "partially_occluded"
    SENSOR_FRINGE = "sensor_fringe"
    UNKNOWN = "unknown"


class SlopeCategory(str, Enum):
    FLAT = "flat"                    # < 5 deg
    GENTLE = "gentle"                # 5 - 15 deg
    MODERATE = "moderate"            # 15 - 25 deg
    STEEP = "steep"                  # 25 - 35 deg
    EXTREME = "extreme"              # > 35 deg


class RoughnessCategory(str, Enum):
    SMOOTH = "smooth"                # < 0.03m std dev
    LOW = "low_roughness"            # 0.03 - 0.08m
    MODERATE = "moderate_roughness"  # 0.08 - 0.15m
    ROUGH = "rough"                  # 0.15 - 0.30m
    HIGHLY_IRREGULAR = "highly_irregular"  # > 0.30m


class StepCategory(str, Enum):
    NONE = "none"                    # < 0.08m vertical discontinuity
    CURB = "curb"                    # 0.08m - 0.25m (typical road curb)
    STEP_BARRIER = "step_barrier"    # 0.25m - 0.50m (step / low barrier)
    HIGH_OBSTACLE = "high_obstacle"  # > 0.50m (wall / high barrier / car)


class ElevationSummary(str, Enum):
    GROUND_LEVEL = "ground_level"
    ELEVATED_SURFACE = "elevated_surface"
    DEPRESSION_SLOPE = "depression_slope"
    OVERHEAD_CLEARANCE = "overhead_clearance"
    VARIABLE_HEIGHT = "variable_height"


class TerrainInterpretation(str, Enum):
    PAVED_FLAT = "paved_flat"
    SLOPED_ROAD = "sloped_road"
    ROUGH_UNPAVED = "rough_unpaved"
    CURB_BOUNDARY = "curb_boundary"
    OBSTACLE_BARRIER = "obstacle_barrier"
    SPARSE_FOLIAGE_OR_OVERHANG = "sparse_foliage_or_overhang"
    UNKNOWN = "unknown"


class DrivabilityState(str, Enum):
    DRIVABLE_CANDIDATE = "drivable_candidate"
    NON_DRIVABLE_CANDIDATE = "non_drivable_candidate"
    CAUTION_IRREGULAR = "caution_irregular"
    OBSTACLE_HAZARD = "obstacle_hazard"
    UNKNOWN = "unknown"


class ResolutionZone(str, Enum):
    NEAR = "near"
    MID = "mid"
    FAR = "far"


# --- Point and Spatial Geometry ---

class Point3D(BaseModel):
    x: float
    y: float
    z: float
    intensity: Optional[float] = 0.0


class BoundingBox3D(BaseModel):
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float


# --- Preprocessing Schemas ---

class PreprocessingConfig(BaseModel):
    min_range: float = Field(default=0.5, description="Minimum Euclidean distance from sensor origin (m)")
    max_range: float = Field(default=80.0, description="Maximum Euclidean distance from sensor origin (m)")
    z_min: float = Field(default=-5.0, description="Minimum elevation cutoff (m)")
    z_max: float = Field(default=10.0, description="Maximum elevation cutoff (m)")
    remove_nan_inf: bool = Field(default=True, description="Filter out non-finite point records")


class PreprocessingReport(BaseModel):
    raw_points_count: int
    valid_points_count: int
    nan_inf_removed: int
    out_of_bounds_removed: int
    applied_config: PreprocessingConfig
    bounds: Optional[BoundingBox3D] = None


# --- Frame Ingestion Schemas ---

class FrameMetadata(BaseModel):
    scan_source: str = "SemanticKITTI .bin LiDAR record"
    timestamp: str
    format: str = "float32 (x,y,z,i)"
    point_count: int
    file_size_bytes: int
    coordinate_frame: str = "Sensor Origin (Ego Body Frame: +X forward, +Y left, +Z up)"


class FrameUploadResponse(BaseModel):
    frame_id: str
    point_count: int
    file_size_bytes: int
    status: ProcessStatus
    timestamp: str
    bounds: BoundingBox3D
    message: str = "LiDAR frame uploaded and parsed successfully"


class FrameDetailsResponse(BaseModel):
    frame_id: str
    point_count: int
    file_size_bytes: int
    bounds: BoundingBox3D
    metadata: FrameMetadata
    sample_points: List[Point3D] = Field(default=[], description="Downsampled point cloud for fast UI preview")
    created_at: str


class JsonPointCloudPayload(BaseModel):
    points: List[Point3D]
    frame_id: Optional[str] = None


# --- Terrain Analysis Schemas ---

class TerrainCell(BaseModel):
    id: str
    grid_x: int
    grid_y: int
    world_x: float
    world_y: float
    size_m: float
    point_count: int
    
    # Continuous geometric properties
    min_z: float
    max_z: float
    mean_z: float
    elevation_range: float
    slope_deg: float
    roughness_m: float
    step_height_m: float
    has_step: bool
    
    # Qualitative interpretations
    slope_category: SlopeCategory
    roughness_category: RoughnessCategory
    step_category: StepCategory
    observation_state: ObservationState
    elevation_summary: ElevationSummary
    terrain_interpretation: TerrainInterpretation
    drivability_state: DrivabilityState
    zone: ResolutionZone


class TerrainAnalysisSummary(BaseModel):
    total_cells: int
    drivable_cells: int
    non_drivable_cells: int
    caution_cells: int
    hazard_cells: int
    unknown_cells: int
    grid_resolution_m: float
    bounds: BoundingBox3D


class TerrainAnalysisRequest(BaseModel):
    frame_id: str
    grid_resolution_m: Optional[float] = 1.0
    preprocessing_config: Optional[PreprocessingConfig] = None


class TerrainAnalysisResponse(BaseModel):
    frame_id: str
    status: ProcessStatus
    preprocessing_report: Optional[PreprocessingReport] = None
    summary: Optional[TerrainAnalysisSummary] = None
    cells: List[TerrainCell] = []
    created_at: str
    error_message: Optional[str] = None


# --- Health & System Schemas ---

class HealthResponse(BaseModel):
    status: str = "healthy"
    project_name: str
    version: str
    timestamp: str
    storage_ready: bool
    features: List[str]
