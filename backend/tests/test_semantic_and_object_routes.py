"""API Route Integration Tests for Semantic Segmentation, Object Detection, and Tracking."""

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient


def test_semantic_label_upload_and_query(client: TestClient, create_kitti_bin, flat_road_points):
    """Test full workflow: upload frame -> upload labels -> query semantic frame."""
    # 1. Upload .bin frame
    bin_bytes = create_kitti_bin(flat_road_points)
    client.post(
        "/api/v1/frames/upload?frame_id=frame_sem_001",
        files={"file": ("scan.bin", io.BytesIO(bin_bytes), "application/octet-stream")},
    )

    n_pts = flat_road_points.shape[0]

    # 2. Upload .label file with matching count
    # 100 points car (10), remainder road (40)
    labels = np.full(n_pts, 40, dtype=np.uint32)
    labels[:100] = 10
    label_bytes = labels.tobytes()

    upload_res = client.post(
        "/api/v1/semantic/upload-labels?frame_id=frame_sem_001",
        files={"file": ("scan.label", io.BytesIO(label_bytes), "application/octet-stream")},
    )
    assert upload_res.status_code == 201
    u_data = upload_res.json()
    assert u_data["label_count"] == n_pts
    assert u_data["class_distribution"]["car"] == 100
    assert u_data["class_distribution"]["road"] == n_pts - 100

    # 3. Query GET /api/v1/semantic/{frame_id}
    get_res = client.get("/api/v1/semantic/frame_sem_001")
    assert get_res.status_code == 200
    g_data = get_res.json()
    assert g_data["point_count"] == n_pts
    assert len(g_data["sample_labeled_points"]) > 0
    assert g_data["model_provider_status"] == "external_predictions_loaded"


def test_label_count_mismatch_fails(client: TestClient):
    """Test upload fails when label count does not match point count."""
    # Upload label file with only 5 labels for frame_sem_001 (which has 1600 points)
    odd_labels = np.array([10, 20, 30, 40, 50], dtype=np.uint32)
    res = client.post(
        "/api/v1/semantic/upload-labels?frame_id=frame_sem_001",
        files={"file": ("scan.label", io.BytesIO(odd_labels.tobytes()), "application/octet-stream")},
    )
    assert res.status_code == 400
    assert "Label count mismatch" in res.json()["detail"]


def test_import_predictions_route(client: TestClient, flat_road_points):
    """Test POST /api/v1/semantic/import-predictions."""
    n_pts = flat_road_points.shape[0]
    payload = {
        "frame_id": "frame_sem_001",
        "model_name": "SalsaNext",
        "labels": [40] * n_pts,
    }
    res = client.post("/api/v1/semantic/import-predictions", json=payload)
    assert res.status_code == 201
    assert res.json()["label_count"] == n_pts


def test_object_detection_route(client: TestClient):
    """Test POST /api/v1/objects/detect and GET /api/v1/objects/{frame_id}."""
    res = client.post(
        "/api/v1/objects/detect",
        json={"frame_id": "frame_sem_001", "min_points_per_cluster": 5},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["frame_id"] == "frame_sem_001"
    assert "instances" in data

    get_res = client.get("/api/v1/objects/frame_sem_001")
    assert get_res.status_code == 200
    assert get_res.json()["frame_id"] == "frame_sem_001"


def test_tracking_routes(client: TestClient):
    """Test POST /api/v1/tracks/update and GET /api/v1/tracks."""
    update_res = client.post(
        "/api/v1/tracks/update",
        json={"frame_id": "frame_sem_001", "max_association_distance_m": 3.0},
    )
    assert update_res.status_code == 200
    data = update_res.json()
    assert "tracks" in data

    tracks_res = client.get("/api/v1/tracks")
    assert tracks_res.status_code == 200
    assert isinstance(tracks_res.json(), list)
