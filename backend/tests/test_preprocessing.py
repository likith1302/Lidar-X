"""Unit tests for point cloud preprocessing and validation."""

import numpy as np
import pytest
from app.services.preprocessing import PointCloudPreprocessor
from app.models.schemas import PreprocessingConfig


def test_nan_and_inf_removal():
    """Test filtering of non-finite point coordinates."""
    raw = np.array([
        [1.0, 2.0, -1.7, 0.5],
        [np.nan, 2.0, -1.7, 0.5],
        [3.0, np.inf, -1.7, 0.5],
        [4.0, 5.0, -np.inf, 0.5],
        [6.0, 7.0, -1.5, np.nan],
        [8.0, 9.0, -1.4, 0.8],
    ], dtype=np.float32)

    config = PreprocessingConfig(remove_nan_inf=True, min_range=0.1, max_range=50.0)
    sanitized, report = PointCloudPreprocessor.process(raw, config)

    assert sanitized.shape[0] == 2  # Only point 0 and point 5 are valid
    assert report.raw_points_count == 6
    assert report.nan_inf_removed == 4
    assert report.valid_points_count == 2
    assert np.all(np.isfinite(sanitized))


def test_range_gating():
    """Test min_range and max_range filtering."""
    pts = np.array([
        [0.1, 0.1, 0.0, 0.5],    # r ≈ 0.14m (below min_range 0.5m)
        [5.0, 0.0, 0.0, 0.5],    # r = 5.0m (within range)
        [20.0, 0.0, 0.0, 0.5],   # r = 20.0m (within range)
        [100.0, 0.0, 0.0, 0.5],  # r = 100.0m (above max_range 50.0m)
        [5.0, 0.0, -8.0, 0.5],   # z = -8.0m (below z_min -5.0m)
        [5.0, 0.0, 15.0, 0.5],   # z = 15.0m (above z_max 10.0m)
    ], dtype=np.float32)

    config = PreprocessingConfig(
        min_range=0.5,
        max_range=50.0,
        z_min=-5.0,
        z_max=10.0,
    )
    sanitized, report = PointCloudPreprocessor.process(pts, config)

    assert sanitized.shape[0] == 2  # points [5,0,0] and [20,0,0]
    assert report.raw_points_count == 6
    assert report.out_of_bounds_removed == 4
    assert report.valid_points_count == 2


def test_coordinate_preservation(flat_road_points):
    """Ensure coordinates are not rotated, scaled, or shifted during preprocessing."""
    config = PreprocessingConfig(min_range=0.0, max_range=100.0, z_min=-10.0, z_max=10.0)
    sanitized, report = PointCloudPreprocessor.process(flat_road_points, config)

    assert sanitized.shape == flat_road_points.shape
    assert np.array_equal(sanitized, flat_road_points)
