"""Unit tests for SemanticKITTI .bin parser and validator."""

import numpy as np
import pytest
from app.services.lidar_parser import LidarParser, LidarParserError
from app.models.schemas import JsonPointCloudPayload, Point3D


def test_parse_valid_kitti_bin(create_kitti_bin, flat_road_points):
    """Test parsing a valid float32 .bin point cloud."""
    bin_data = create_kitti_bin(flat_road_points)
    parsed = LidarParser.parse_kitti_bin(bin_data)

    assert parsed.shape == flat_road_points.shape
    assert parsed.dtype == np.float32
    assert np.allclose(parsed, flat_road_points, atol=1e-5)


def test_reject_empty_bin():
    """Test rejecting empty binary file."""
    with pytest.raises(LidarParserError) as exc_info:
        LidarParser.parse_kitti_bin(b"")
    assert "Empty LiDAR file" in str(exc_info.value)


def test_reject_malformed_byte_length():
    """Test rejecting binary data with non-multiple of 16 byte count."""
    # 25 bytes (not divisible by 16)
    corrupted_bytes = b"\x00" * 25
    with pytest.raises(LidarParserError) as exc_info:
        LidarParser.parse_kitti_bin(corrupted_bytes)
    assert "not a multiple of 16" in str(exc_info.value)


def test_parse_json_payload():
    """Test parsing JSON point list into NumPy array."""
    payload = JsonPointCloudPayload(
        points=[
            Point3D(x=1.0, y=2.0, z=-1.5, intensity=0.8),
            Point3D(x=3.0, y=4.0, z=-1.2, intensity=0.5),
            Point3D(x=5.0, y=6.0, z=-0.9, intensity=0.2),
        ]
    )
    arr = LidarParser.parse_json_points(payload)
    assert arr.shape == (3, 4)
    assert arr[0, 0] == 1.0
    assert arr[1, 1] == 4.0
    assert arr[2, 2] == -0.9


def test_compute_bounds():
    """Test 3D bounding box computation."""
    pts = np.array([
        [-5.0, -10.0, -2.0, 0.5],
        [15.0, 8.0, 3.0, 0.8],
        [0.0, 0.0, 0.0, 0.1],
    ], dtype=np.float32)

    bounds = LidarParser.compute_bounds(pts)
    assert bounds.min_x == -5.0
    assert bounds.max_x == 15.0
    assert bounds.min_y == -10.0
    assert bounds.max_y == 8.0
    assert bounds.min_z == -2.0
    assert bounds.max_z == 3.0


def test_downsample_for_preview(flat_road_points):
    """Test downsampling large point array for UI preview."""
    sampled = LidarParser.downsample_for_preview(flat_road_points, max_points=100)
    assert len(sampled) == 100
    assert isinstance(sampled[0], Point3D)
