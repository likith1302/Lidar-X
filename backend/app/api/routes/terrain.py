"""Terrain Analysis API routes."""

from fastapi import APIRouter, HTTPException, status
from ...models.schemas import (
    TerrainAnalysisRequest,
    TerrainAnalysisResponse,
)
from ...services.storage import StorageService
from ...services.terrain_analysis import TerrainAnalysisEngine

router = APIRouter()


@router.post(
    "/analyze",
    response_model=TerrainAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Run geometric terrain analysis on a LiDAR frame",
    tags=["Terrain Analysis"],
)
async def analyze_terrain(request: TerrainAnalysisRequest) -> TerrainAnalysisResponse:
    """Execute geometric terrain analysis pipeline on a previously uploaded frame."""
    points = StorageService.get_frame_points(request.frame_id)
    if points is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Frame '{request.frame_id}' not found. Please upload the LiDAR frame first.",
        )

    grid_res = request.grid_resolution_m if request.grid_resolution_m is not None else 1.0
    if grid_res <= 0.05 or grid_res > 10.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="grid_resolution_m must be between 0.1m and 10.0m.",
        )

    response = TerrainAnalysisEngine.analyze(
        frame_id=request.frame_id,
        points=points,
        grid_resolution_m=grid_res,
        preprocessing_config=request.preprocessing_config,
    )

    # Persist analysis result for future queries
    StorageService.save_terrain_result(response)

    return response


@router.get(
    "/{frame_id}",
    response_model=TerrainAnalysisResponse,
    summary="Get saved terrain analysis results for a frame",
    tags=["Terrain Analysis"],
)
async def get_terrain_analysis(frame_id: str) -> TerrainAnalysisResponse:
    """Retrieve previously computed terrain analysis result for a given frame ID."""
    result = StorageService.get_terrain_result(frame_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No terrain analysis result found for frame '{frame_id}'. Trigger POST /terrain/analyze first.",
        )
    return result
