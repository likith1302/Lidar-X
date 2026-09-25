"""Pydantic schemas and data contracts for Adaptive Variable-Resolution 2.5D Grid Engine."""

from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

from .schemas import ProcessStatus, BoundingBox3D, Point3D


# --- Grid & Map Enums ---

class ResolutionLevel(str, Enum):
    FINE = "fine"        # E.g., 0.5m
    MEDIUM = "medium"    # E.g., 1.0m
    COARSE = "coarse"    # E.g., 2.0m


class MapMode(str, Enum):
    LOCAL_ONLY = "local_only"        # Vehicle-centric frame, replaced on single-frame updates
    GLOBAL_FUSION = "global_fusion"  # World frame, integrated across sequential ego poses


class CellObservationState(str, Enum):
    OBSERVED = "observed"
    PARTIALLY_OCCLUDED = "partially_occluded"
    OUT_OF_RANGE = "out_of_range"
    UNKNOWN = "unknown"


class TraversabilityState(str, Enum):
    DRIVABLE = "drivable"
    NON_TRAVERSABLE = "non_traversable"
    CAUTION_IRREGULAR = "caution_irregular"
    COLLISION_HAZARD = "collision_hazard"
    UNCERTAIN = "uncertain"


class AmbiguityState(str, Enum):
    UNAMBIGUOUS = "unambiguous"
    MIXED_CLASSES = "mixed_classes"
    SPARSE_DATA = "sparse_data"
    HIGH_GRADIENT = "high_gradient"


# --- Multi-Layer Elevation Schema ---

class ElevationLayer(BaseModel):
    layer_index: int = 0
    layer_type: str = "ground"  # "ground", "intermediate", "overhang"
    elevation_min: float
    elevation_mean: float
    elevation_max: float
    elevation_range: float
    point_count: int
    dominant_category: str = "unknown"
    traversability_state: TraversabilityState = TraversabilityState.UNCERTAIN


# --- Policy Configuration ---

class GridPolicyConfig(BaseModel):
    near_zone_max_distance_m: float = Field(default=10.0, ge=1.0, le=50.0, description="Max radial distance for fine resolution (m)")
    mid_zone_max_distance_m: float = Field(default=30.0, ge=5.0, le=100.0, description="Max radial distance for medium resolution (m)")
    far_zone_max_distance_m: float = Field(default=100.0, ge=20.0, le=200.0, description="Max radial distance for coarse grid (m)")
    
    fine_resolution_m: float = Field(default=0.05, ge=0.01, le=2.0, description="Fine cell dimension (m)")
    medium_resolution_m: float = Field(default=0.20, ge=0.05, le=5.0, description="Medium cell dimension (m)")
    coarse_resolution_m: float = Field(default=0.50, ge=0.10, le=10.0, description="Coarse cell dimension (m)")
    
    safety_priority: bool = Field(default=True, description="Force fine detail for dynamic actors & boundaries")
    terrain_complexity_override: bool = Field(default=True, description="Refine cells with high slope or roughness")
    obstacle_override: bool = Field(default=True, description="Refine cells containing static obstacles or steps")
    
    slope_refinement_threshold_deg: float = Field(default=10.0, ge=1.0, le=45.0)
    roughness_refinement_threshold_m: float = Field(default=0.10, ge=0.01, le=1.0)
    step_refinement_threshold_m: float = Field(default=0.10, ge=0.01, le=1.0)
    obstacle_height_span_threshold_m: float = Field(default=0.30, ge=0.05, le=2.0)
    
    default_cells_query_limit: int = Field(default=5000, ge=100, le=50000)


# --- 2.5D Adaptive Grid Cell ---

class AdaptiveGridCell(BaseModel):
    cell_key: str = Field(..., description="Unique hierarchical key e.g. fine:12_4, medium:6_2")
    level: ResolutionLevel
    size_m: float
    grid_x: int
    grid_y: int
    world_x: float
    world_y: float
    bounds: List[float] = Field(..., description="[min_x, max_x, min_y, max_y] in meters")
    observation_state: CellObservationState
    
    # Semantic Content
    semantic_histogram: Dict[str, int] = Field(default_factory=dict)
    dominant_semantic_class: str = "unlabeled"
    dominant_category: str = "unknown"
    
    # Geometric & Terrain Properties
    terrain_state: str = "unknown"
    traversability_state: TraversabilityState
    is_static_obstacle: bool = False
    is_dynamic_obstacle: bool = False
    
    elevation_min: float
    elevation_mean: float
    elevation_max: float
    elevation_variation: float = Field(..., description="Elevation range (max_z - min_z) or std dev in meters")
    roughness_summary: float = 0.0
    slope_summary: float = 0.0
    point_count: int
    
    # Multi-Surface & Overhang Support
    has_overhang: bool = False
    overhead_clearance_m: Optional[float] = None
    layers: List[ElevationLayer] = Field(default_factory=list)
    
    # Metadata & Hierarchical Links
    last_frame_id: Optional[str] = None
    last_timestamp: Optional[str] = None
    ambiguity_state: AmbiguityState = AmbiguityState.UNAMBIGUOUS
    parent_key: Optional[str] = None
    child_keys: List[str] = Field(default_factory=list)

    @property
    def cell_size_m(self) -> float:
        return self.size_m

    @property
    def min_z(self) -> float:
        return self.elevation_min

    @property
    def max_z(self) -> float:
        return self.elevation_max

    @property
    def mean_z(self) -> float:
        return self.elevation_mean

    @property
    def elevation_range(self) -> float:
        return self.elevation_variation

    @property
    def slope_deg(self) -> float:
        return self.slope_summary

    @property
    def roughness_m(self) -> float:
        return self.roughness_summary

    @property
    def dominant_class(self) -> str:
        return self.dominant_semantic_class

    @property
    def traversability(self) -> TraversabilityState:
        return self.traversability_state

    @property
    def is_dynamic(self) -> bool:
        return self.is_dynamic_obstacle

    @property
    def parent_cell_key(self) -> Optional[str]:
        return self.parent_key

    @property
    def child_cell_keys(self) -> List[str]:
        return self.child_keys

    @property
    def last_observed_frame(self) -> Optional[str]:
        return self.last_frame_id

    @classmethod
    def create_fast(
        cls,
        cell_key: str,
        level: ResolutionLevel,
        size_m: float,
        grid_x: int,
        grid_y: int,
        world_x: float,
        world_y: float,
        bounds: List[float],
        observation_state: CellObservationState,
        semantic_histogram: Dict[str, int],
        dominant_semantic_class: str,
        dominant_category: str,
        terrain_state: str,
        traversability_state: TraversabilityState,
        is_static_obstacle: bool,
        is_dynamic_obstacle: bool,
        elevation_min: float,
        elevation_mean: float,
        elevation_max: float,
        elevation_variation: float,
        roughness_summary: float,
        slope_summary: float,
        point_count: int,
        has_overhang: bool = False,
        overhead_clearance_m: Optional[float] = None,
        layers: Optional[List[ElevationLayer]] = None,
        last_frame_id: Optional[str] = None,
        last_timestamp: Optional[str] = None,
        ambiguity_state: AmbiguityState = AmbiguityState.UNAMBIGUOUS,
        parent_key: Optional[str] = None,
        child_keys: Optional[List[str]] = None,
    ) -> "AdaptiveGridCell":
        """Ultra-fast instantiation skipping Pydantic dict.pop overhead for multi-thousand cell loops."""
        obj = cls.__new__(cls)
        object.__setattr__(obj, "__dict__", {
            "cell_key": cell_key,
            "level": level,
            "size_m": size_m,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "world_x": world_x,
            "world_y": world_y,
            "bounds": bounds,
            "observation_state": observation_state,
            "semantic_histogram": semantic_histogram,
            "dominant_semantic_class": dominant_semantic_class,
            "dominant_category": dominant_category,
            "terrain_state": terrain_state,
            "traversability_state": traversability_state,
            "is_static_obstacle": is_static_obstacle,
            "is_dynamic_obstacle": is_dynamic_obstacle,
            "elevation_min": elevation_min,
            "elevation_mean": elevation_mean,
            "elevation_max": elevation_max,
            "elevation_variation": elevation_variation,
            "roughness_summary": roughness_summary,
            "slope_summary": slope_summary,
            "point_count": point_count,
            "has_overhang": has_overhang,
            "overhead_clearance_m": overhead_clearance_m,
            "layers": layers if layers is not None else [],
            "last_frame_id": last_frame_id,
            "last_timestamp": last_timestamp,
            "ambiguity_state": ambiguity_state,
            "parent_key": parent_key,
            "child_keys": child_keys if child_keys is not None else [],
        })
        object.__setattr__(obj, "__pydantic_fields_set__", {
            "cell_key", "level", "size_m", "grid_x", "grid_y", "world_x", "world_y", "bounds",
            "observation_state", "semantic_histogram", "dominant_semantic_class", "dominant_category",
            "terrain_state", "traversability_state", "is_static_obstacle", "is_dynamic_obstacle",
            "elevation_min", "elevation_mean", "elevation_max", "elevation_variation", "roughness_summary",
            "slope_summary", "point_count", "has_overhang", "overhead_clearance_m", "layers",
            "last_frame_id", "last_timestamp", "ambiguity_state", "parent_key", "child_keys"
        })
        object.__setattr__(obj, "__pydantic_extra__", None)
        object.__setattr__(obj, "__pydantic_private__", None)
        return obj


# --- Map Metadata & Ingestion Contracts ---

class MapMetadata(BaseModel):
    map_id: str
    mode: MapMode
    frame_count: int
    total_cells: int
    fine_cells_count: int
    medium_cells_count: int
    coarse_cells_count: int
    bounds: BoundingBox3D
    created_at: str
    updated_at: str


class MapUpdateRequest(BaseModel):
    frame_id: str
    ego_pose: Optional[Union[List[List[float]], List[float]]] = Field(
        default=None,
        description="Optional 4x4 transform matrix or [x, y, yaw_rad]. If omitted, mode is local_only.",
    )
    override_policy: Optional[Dict[str, Any]] = None


class MapUpdateResponse(BaseModel):
    map_id: str
    mode: MapMode
    status: ProcessStatus
    updated_cell_count: int
    metadata: MapMetadata
    cells_sample: List[Any] = Field(default_factory=list)
    message: str = "Map updated successfully"


class MapExportResponse(BaseModel):
    map_id: str
    mode: MapMode
    metadata: MapMetadata
    cells: List[AdaptiveGridCell]
    policy_snapshot: GridPolicyConfig


class MapResetResponse(BaseModel):
    map_id: str
    status: str = "reset_complete"
    message: str = "Map cleared and reset successfully"
