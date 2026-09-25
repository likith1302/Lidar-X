"""Pydantic schemas for strict pipeline gate status and sequential execution."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from .schemas import ProcessStatus


class PipelineStatusResponse(BaseModel):
    """Truthful real-artifact pipeline status for a canonical frame ID."""
    frame_id: str = Field(..., description="Authoritative canonical frame ID")
    frame_exists: bool = Field(False, description="Whether the point cloud is stored for this frame")
    point_cloud_available: bool = Field(False, description="Point cloud availability")
    point_count: int = Field(0, description="Valid 3D point count")
    semantic_labels_available: bool = Field(False, description="Whether semantic labels exist for this frame")
    semantic_label_count: int = Field(0, description="Count of semantic labels stored")
    labels_match_point_count: bool = Field(False, description="Whether label count matches point count exactly")
    semantic_source: Optional[str] = Field(None, description="Source of semantic labels (e.g. salsanext, imported)")
    inference_state: Optional[str] = Field("not_run", description="Status of SalsaNext neural inference")
    terrain_available: bool = Field(False, description="Whether geometric terrain analysis exists")
    object_detection_available: bool = Field(False, description="Whether object detection instances exist")
    object_instance_count: int = Field(0, description="Count of detected object instances")
    adaptive_map_available: bool = Field(False, description="Whether 2.5D adaptive grid map exists")
    adaptive_grid_cell_count: int = Field(0, description="Count of 2.5D elevation cells generated")
    map_mode: Optional[str] = Field(None, description="Map mode: local_only or persistent_fused")
    last_error: Optional[str] = Field(None, description="Latest pipeline validation error if any")


class PipelineProcessRequest(BaseModel):
    """Request to process sequential perception pipeline stages for a frame."""
    stages: List[str] = Field(
        default=["fast_frnet", "terrain", "objects", "adaptive_grid"],
        description="List of requested stages in dependency order: fast_frnet (or salsanext), terrain, objects, adaptive_grid",
    )
    model_type: Optional[str] = Field(
        default="rellis",
        description="Fast-FRNet model type: 'rellis' (off-road, primary) or 'semantickitti' (general)",
    )
    allow_geometry_only_grid: bool = Field(
        default=False,
        description="If True, allows adaptive grid to synthesize from geometry only when semantic labels are unavailable",
    )


class StageExecutionResult(BaseModel):
    """Status of an individual pipeline stage execution."""
    stage: str
    status: ProcessStatus
    message: str
    item_count: int = 0
    details: Optional[Dict[str, Any]] = None


class PipelineProcessResponse(BaseModel):
    """Overall sequential pipeline execution response."""
    frame_id: str
    success: bool
    status: ProcessStatus
    executed_stages: List[StageExecutionResult]
    pipeline_status: PipelineStatusResponse
    message: str
