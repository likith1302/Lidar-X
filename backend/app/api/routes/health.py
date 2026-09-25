"""Health check and system capabilities endpoint."""

from datetime import datetime, timezone
from fastapi import APIRouter
from ...config import settings
from ...models.schemas import HealthResponse
from ...services.storage import StorageService

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def get_health() -> HealthResponse:
    """Return backend operational status, active storage directories, and feature flags."""
    now_iso = datetime.now(timezone.utc).isoformat()
    storage_ready = settings.FRAMES_DIR.exists() and settings.TERRAIN_DIR.exists()
    
    return HealthResponse(
        status="healthy",
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        timestamp=now_iso,
        storage_ready=storage_ready,
        features=[
            "kitti_bin_ingestion",
            "json_pointcloud_ingestion",
            "pointcloud_preprocessing",
            "geometric_terrain_analysis",
            "adaptive_cell_classification",
            "elevation_roughness_estimation",
            "step_curb_detection",
            "semantic_kitti_mapping",
            "dbscan_instance_clustering",
            "temporal_kalman_tracking",
            "adaptive_2.5d_grid_engine",
            "foveated_hierarchical_resolution",
            "local_and_global_map_fusion",
        ],
    )
