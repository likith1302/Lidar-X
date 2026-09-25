"""Unit tests for geometric terrain analysis engine."""

import numpy as np
import pytest
from app.services.terrain_analysis import TerrainAnalysisEngine
from app.models.schemas import (
    SlopeCategory,
    RoughnessCategory,
    StepCategory,
    DrivabilityState,
    TerrainInterpretation,
    ObservationState,
    ResolutionZone,
)


def test_flat_road_analysis(flat_road_points):
    """Test terrain analysis on a synthetic flat road surface."""
    result = TerrainAnalysisEngine.analyze(
        frame_id="test_flat_frame",
        points=flat_road_points,
        grid_resolution_m=1.0,
    )

    assert result.frame_id == "test_flat_frame"
    assert result.status == "complete"
    assert len(result.cells) > 0
    assert result.summary.total_cells == len(result.cells)
    assert result.summary.drivable_cells > 0

    # Check properties of central cells
    for cell in result.cells:
        if cell.observation_state == ObservationState.DIRECTLY_OBSERVED:
            # Slope should be flat (< 5 degrees)
            assert cell.slope_deg < 5.0
            assert cell.slope_category == SlopeCategory.FLAT
            # Roughness should be smooth / low
            assert cell.roughness_category in [RoughnessCategory.SMOOTH, RoughnessCategory.LOW]
            # Traversability should be drivable candidate
            assert cell.drivability_state == DrivabilityState.DRIVABLE_CANDIDATE
            assert cell.terrain_interpretation in [TerrainInterpretation.PAVED_FLAT, TerrainInterpretation.SLOPED_ROAD]


def test_sloped_ramp_analysis(sloped_ramp_points):
    """Test slope angle estimation on a 20-degree inclined ramp."""
    result = TerrainAnalysisEngine.analyze(
        frame_id="test_sloped_frame",
        points=sloped_ramp_points,
        grid_resolution_m=1.0,
    )

    assert len(result.cells) > 0
    # Find well-populated cells
    sloped_cells = [c for c in result.cells if c.point_count >= 10]
    assert len(sloped_cells) > 0

    for cell in sloped_cells:
        # Estimated slope should be close to 20 degrees (within 3 degrees)
        assert 15.0 <= cell.slope_deg <= 25.0
        assert cell.slope_category == SlopeCategory.MODERATE


def test_shallow_incline_slope_detection():
    """Verify shallow slope (3.5 degrees, range < 0.10m) is accurately estimated and not suppressed to 0."""
    x = np.linspace(2, 6, 25)
    y = np.linspace(-2, 2, 25)
    xx, yy = np.meshgrid(x, y)
    n_pts = xx.size

    pts = np.zeros((n_pts, 4), dtype=np.float32)
    pts[:, 0] = xx.ravel()
    pts[:, 1] = yy.ravel()
    # 3.5 degree slope: tan(3.5 deg) ~ 0.061. Elevation range in a 1m cell is ~0.06m (< 0.10m).
    pts[:, 2] = -1.73 + (xx.ravel() - 2.0) * np.tan(np.radians(3.5))
    pts[:, 3] = 0.5

    result = TerrainAnalysisEngine.analyze(
        frame_id="test_shallow_slope",
        points=pts,
        grid_resolution_m=1.0,
    )

    cells = [c for c in result.cells if c.point_count >= 10]
    assert len(cells) > 0
    for c in cells:
        # Slope must be close to 3.5 degrees (not 0.0)
        assert 2.0 <= c.slope_deg <= 5.0
        assert c.slope_category == SlopeCategory.FLAT or c.slope_category == SlopeCategory.GENTLE


def test_curb_detection(curb_points):
    """Test detection of 15cm curb height step."""
    result = TerrainAnalysisEngine.analyze(
        frame_id="test_curb_frame",
        points=curb_points,
        grid_resolution_m=1.0,
    )

    assert len(result.cells) > 0
    # Locate cells near the curb boundary y=2.0
    curb_cells = [c for c in result.cells if 1.0 <= c.world_y <= 3.0 and c.has_step]
    assert len(curb_cells) > 0

    for cell in curb_cells:
        assert cell.has_step is True
        assert cell.step_height_m >= 0.08
        assert cell.step_category in [StepCategory.CURB, StepCategory.STEP_BARRIER]
        assert cell.drivability_state in [DrivabilityState.CAUTION_IRREGULAR, DrivabilityState.NON_DRIVABLE_CANDIDATE]


def test_resolution_zones():
    """Test resolution zone assignment (near, mid, far) based on distance."""
    pts = np.array([
        [2.0, 2.0, -1.73, 0.5],    # dist ≈ 2.8m -> NEAR (<12m)
        [15.0, 5.0, -1.73, 0.5],   # dist ≈ 15.8m -> MID (12-28m)
        [30.0, 10.0, -1.73, 0.5],  # dist ≈ 31.6m -> FAR (>=28m)
    ], dtype=np.float32)

    result = TerrainAnalysisEngine.analyze(
        frame_id="test_zones_frame",
        points=pts,
        grid_resolution_m=1.0,
    )

    assert len(result.cells) == 3
    zones = {c.zone for c in result.cells}
    assert ResolutionZone.NEAR in zones
    assert ResolutionZone.MID in zones
    assert ResolutionZone.FAR in zones
