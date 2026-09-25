"""API Route Integration Tests for FastAPI backend."""

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient


def test_health_route(client: TestClient):
    """Test GET /api/v1/health."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "features" in data
    assert "geometric_terrain_analysis" in data["features"]


def test_upload_valid_kitti_bin(client: TestClient, create_kitti_bin, flat_road_points):
    """Test POST /api/v1/frames/upload with valid binary LiDAR file."""
    bin_bytes = create_kitti_bin(flat_road_points)
    file_payload = {"file": ("test_frame.bin", io.BytesIO(bin_bytes), "application/octet-stream")}

    res = client.post("/api/v1/frames/upload?frame_id=frame_test_001", files=file_payload)
    assert res.status_code == 201
    data = res.json()
    assert data["frame_id"] == "frame_test_001"
    assert data["point_count"] == flat_road_points.shape[0]
    assert data["status"] == "complete"
    assert "bounds" in data


def test_upload_malformed_bin_fails(client: TestClient):
    """Test POST /api/v1/frames/upload with corrupted non-16-byte payload."""
    corrupted_bytes = b"random_corrupt_data_not_divisible_by_16"
    file_payload = {"file": ("corrupt.bin", io.BytesIO(corrupted_bytes), "application/octet-stream")}

    res = client.post("/api/v1/frames/upload", files=file_payload)
    assert res.status_code == 400
    assert "not a multiple of 16" in res.json()["detail"]


def test_upload_json_payload(client: TestClient):
    """Test POST /api/v1/frames/upload-json."""
    payload = {
        "frame_id": "frame_json_002",
        "points": [
            {"x": 1.0, "y": 2.0, "z": -1.5, "intensity": 0.5},
            {"x": 2.0, "y": 3.0, "z": -1.4, "intensity": 0.6},
        ],
    }
    res = client.post("/api/v1/frames/upload-json", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["frame_id"] == "frame_json_002"
    assert data["point_count"] == 2


def test_get_frame_details(client: TestClient):
    """Test GET /api/v1/frames/{frame_id}."""
    res = client.get("/api/v1/frames/frame_test_001")
    assert res.status_code == 200
    data = res.json()
    assert data["frame_id"] == "frame_test_001"
    assert len(data["sample_points"]) > 0
    assert "metadata" in data


def test_get_nonexistent_frame_returns_404(client: TestClient):
    """Test GET /api/v1/frames/non_existent returns 404."""
    res = client.get("/api/v1/frames/non_existent_9999")
    assert res.status_code == 404


def test_terrain_analysis_pipeline(client: TestClient):
    """Test POST /api/v1/terrain/analyze and GET /api/v1/terrain/{frame_id}."""
    analyze_payload = {
        "frame_id": "frame_test_001",
        "grid_resolution_m": 1.0,
        "preprocessing_config": {
            "min_range": 0.5,
            "max_range": 50.0,
            "z_min": -5.0,
            "z_max": 5.0,
        },
    }
    res = client.post("/api/v1/terrain/analyze", json=analyze_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["frame_id"] == "frame_test_001"
    assert data["status"] == "complete"
    assert len(data["cells"]) > 0
    assert data["summary"]["total_cells"] == len(data["cells"])

    # Query GET endpoint for saved result
    get_res = client.get("/api/v1/terrain/frame_test_001")
    assert get_res.status_code == 200
    assert get_res.json()["frame_id"] == "frame_test_001"


def test_terrain_analysis_nonexistent_frame_returns_404(client: TestClient):
    """Test POST /api/v1/terrain/analyze with missing frame returns 404."""
    res = client.post(
        "/api/v1/terrain/analyze",
        json={"frame_id": "missing_frame_id", "grid_resolution_m": 1.0},
    )
    assert res.status_code == 404
