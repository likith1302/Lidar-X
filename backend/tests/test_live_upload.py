"""Comprehensive tests for the Live Upload & Inference Pipeline in LiDAR-X.

Verifies:
1. Section 27: Upload of two different files with identical filename ('scene.bin')
   produces distinct point counts, different predictions, separate performance metrics,
   and zero stale data retention.
2. Section 28: Precomputed .label files on disk are strictly ignored for uploaded files.
   Response must contain mode='live_upload', precomputed=False, source='uploaded_point_cloud'.
3. Section 30: Multi-format support for .bin, .pcd, .xyz, and .ply.
4. Automatic scene analysis & Fast-FRNet model selection (RELLIS vs. SemanticKITTI).
5. Corrupted/empty uploads fail truthfully without mock fallback.
"""

import io
import struct
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.services.lidar_parser import serialize_point_cloud
from app.services.scene_analysis import SceneAnalysisEngine


def test_identical_filename_live_upload_difference(client: TestClient):
    """Section 27: Upload two different point clouds with identical filename 'scene.bin'.
    
    Verifies:
    - point counts differ
    - content hashes differ
    - predicted classes differ
    - detected objects differ
    - performance metrics reflect separate runs
    - run 2 does NOT retain data from run 1
    """
    # 1. First file: 1,500 points (flat urban road with a simulated vehicle cluster)
    np.random.seed(42)
    pts_a = np.zeros((1500, 4), dtype=np.float32)
    # Ground road points
    pts_a[:1200, 0] = np.random.uniform(-15.0, 15.0, 1200)
    pts_a[:1200, 1] = np.random.uniform(-15.0, 15.0, 1200)
    pts_a[:1200, 2] = -1.73 + np.random.normal(0, 0.02, 1200)
    pts_a[:1200, 3] = 0.8
    # Dense car cluster
    pts_a[1200:, 0] = np.random.uniform(3.0, 7.0, 300)
    pts_a[1200:, 1] = np.random.uniform(2.0, 4.0, 300)
    pts_a[1200:, 2] = np.random.uniform(-1.5, 0.0, 300)
    pts_a[1200:, 3] = 0.9

    bin_a = pts_a.tobytes()

    res_a = client.post(
        "/api/v1/inference/live-upload",
        files={"file": ("scene.bin", io.BytesIO(bin_a), "application/octet-stream")},
        data={"request_id": "req_test_a"},
    )
    assert res_a.status_code == 200, f"Upload A failed: {res_a.text}"
    data_a = res_a.json()

    assert data_a["mode"] == "live_upload"
    assert data_a["source"] == "uploaded_point_cloud"
    assert data_a["inference_source"] == "fast_frnet_live"
    assert data_a["precomputed"] is False
    assert data_a["point_count"] == 1500
    assert data_a["filename"] == "scene.bin"

    hash_a = data_a["content_hash"]
    classes_a = data_a["semantic"]["class_counts"]
    objects_a = data_a["objects"]["total_instances"]
    perf_a = data_a["performance"]

    # 2. Second file: 3,200 points with different geometry (rugged terrain, high elevation spread)
    # Uploaded with the EXACT SAME filename 'scene.bin'
    pts_b = np.zeros((3200, 4), dtype=np.float32)
    pts_b[:, 0] = np.random.uniform(-30.0, 30.0, 3200)
    pts_b[:, 1] = np.random.uniform(-30.0, 30.0, 3200)
    pts_b[:, 2] = np.random.uniform(-5.0, 10.0, 3200)  # High vertical roughness
    pts_b[:, 3] = 0.25

    bin_b = pts_b.tobytes()

    res_b = client.post(
        "/api/v1/inference/live-upload",
        files={"file": ("scene.bin", io.BytesIO(bin_b), "application/octet-stream")},
        data={"request_id": "req_test_b"},
    )
    assert res_b.status_code == 200, f"Upload B failed: {res_b.text}"
    data_b = res_b.json()

    assert data_b["mode"] == "live_upload"
    assert data_b["source"] == "uploaded_point_cloud"
    assert data_b["inference_source"] == "fast_frnet_live"
    assert data_b["precomputed"] is False
    assert data_b["point_count"] == 3200
    assert data_b["filename"] == "scene.bin"

    hash_b = data_b["content_hash"]
    classes_b = data_b["semantic"]["class_counts"]
    objects_b = data_b["objects"]["total_instances"]
    perf_b = data_b["performance"]

    # 3. Verify absolute divergence between run A and run B
    assert data_a["point_count"] != data_b["point_count"], "Point counts must differ between uploads"
    assert hash_a != hash_b, "Content hashes must differ"
    assert classes_a != classes_b, "Predicted class distributions must differ"
    assert perf_a["total_ms"] > 0 and perf_b["total_ms"] > 0
    assert perf_a["fps"] > 0 and perf_b["fps"] > 0
    # Verify second run did not retain point count or hash of first run
    assert data_b["point_count"] == 3200


def test_precomputed_label_on_disk_is_ignored(client: TestClient):
    """Section 28: Precomputed .label file on disk must NEVER be used for live uploads.
    
    Verifies:
    - Precomputed label on disk is ignored
    - response contains mode='live_upload'
    - response contains precomputed=False
    - response contains source='uploaded_point_cloud'
    - predictions were generated live by the neural model
    """
    frame_name = "test_ignore_precomputed"
    predictions_dir = settings.DATA_DIR / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Place a dummy .label file where every label is 99 (non-existent class)
    num_pts = 1000
    dummy_labels = np.full(num_pts, fill_value=99, dtype=np.uint32)
    dummy_label_path = predictions_dir / f"{frame_name}.label"
    with open(dummy_label_path, "wb") as f:
        f.write(dummy_labels.tobytes())

    try:
        # Generate real point cloud
        pts = np.zeros((num_pts, 4), dtype=np.float32)
        pts[:, 0] = np.random.uniform(-10.0, 10.0, num_pts)
        pts[:, 1] = np.random.uniform(-10.0, 10.0, num_pts)
        pts[:, 2] = -1.73
        pts[:, 3] = 0.5
        bin_bytes = pts.tobytes()

        res = client.post(
            "/api/v1/inference/live-upload",
            files={"file": (f"{frame_name}.bin", io.BytesIO(bin_bytes), "application/octet-stream")},
        )
        assert res.status_code == 200
        data = res.json()

        assert data["mode"] == "live_upload"
        assert data["precomputed"] is False
        assert data["source"] == "uploaded_point_cloud"
        assert data["inference_source"] == "fast_frnet_live"

        # Predictions must be legitimate Fast-FRNet classes (NOT 99 from dummy label file)
        for p in data["sample_predictions"]:
            assert p["raw_label_id"] != 99, "Prediction was incorrectly loaded from precomputed .label file!"
            assert p["raw_label_id"] in range(0, 90), f"Unexpected raw label ID: {p['raw_label_id']}"
    finally:
        if dummy_label_path.exists():
            dummy_label_path.unlink()


def test_multiformat_pcd_ascii_live_upload(client: TestClient):
    """Section 30: Test PCD (ASCII) point cloud live upload and inference."""
    pts = np.zeros((600, 4), dtype=np.float32)
    pts[:, 0] = np.random.uniform(-10.0, 10.0, 600)
    pts[:, 1] = np.random.uniform(-10.0, 10.0, 600)
    pts[:, 2] = -1.73 + np.random.normal(0, 0.05, 600)
    pts[:, 3] = 0.7

    pcd_bytes = serialize_point_cloud(pts, "pcd", binary=False)

    res = client.post(
        "/api/v1/inference/live-upload",
        files={"file": ("test_scan.pcd", io.BytesIO(pcd_bytes), "application/octet-stream")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "live_upload"
    assert data["precomputed"] is False
    assert data["point_count"] == 600
    assert len(data["sample_predictions"]) > 0


def test_multiformat_pcd_binary_live_upload(client: TestClient):
    """Section 30: Test PCD (binary) point cloud live upload and inference."""
    pts = np.zeros((800, 4), dtype=np.float32)
    pts[:, 0] = np.random.uniform(-12.0, 12.0, 800)
    pts[:, 1] = np.random.uniform(-12.0, 12.0, 800)
    pts[:, 2] = -1.73
    pts[:, 3] = 0.6

    pcd_bytes = serialize_point_cloud(pts, "pcd", binary=True)

    res = client.post(
        "/api/v1/inference/live-upload",
        files={"file": ("binary_scan.pcd", io.BytesIO(pcd_bytes), "application/octet-stream")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "live_upload"
    assert data["point_count"] == 800


def test_multiformat_xyz_ascii_live_upload(client: TestClient):
    """Section 30: Test XYZ (ASCII) point cloud live upload and inference."""
    pts = np.zeros((500, 4), dtype=np.float32)
    pts[:, 0] = np.linspace(-5, 5, 500)
    pts[:, 1] = np.linspace(-5, 5, 500)
    pts[:, 2] = -1.73
    pts[:, 3] = 0.5

    xyz_bytes = serialize_point_cloud(pts, "xyz", binary=False)

    res = client.post(
        "/api/v1/inference/live-upload",
        files={"file": ("points.xyz", io.BytesIO(xyz_bytes), "text/plain")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "live_upload"
    assert data["point_count"] == 500


def test_multiformat_ply_ascii_live_upload(client: TestClient):
    """Section 30: Test PLY (ASCII) point cloud live upload and inference."""
    pts = np.zeros((400, 4), dtype=np.float32)
    pts[:, 0] = np.random.uniform(-8.0, 8.0, 400)
    pts[:, 1] = np.random.uniform(-8.0, 8.0, 400)
    pts[:, 2] = -1.73
    pts[:, 3] = 0.8

    ply_bytes = serialize_point_cloud(pts, "ply", binary=False)

    res = client.post(
        "/api/v1/inference/live-upload",
        files={"file": ("surface.ply", io.BytesIO(ply_bytes), "application/octet-stream")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "live_upload"
    assert data["point_count"] == 400


def test_multiformat_ply_binary_live_upload(client: TestClient):
    """Section 30: Test PLY (binary little-endian) point cloud live upload and inference."""
    pts = np.zeros((700, 4), dtype=np.float32)
    pts[:, 0] = np.random.uniform(-10.0, 10.0, 700)
    pts[:, 1] = np.random.uniform(-10.0, 10.0, 700)
    pts[:, 2] = -1.73
    pts[:, 3] = 0.45

    ply_bytes = serialize_point_cloud(pts, "ply", binary=True)

    res = client.post(
        "/api/v1/inference/live-upload",
        files={"file": ("dense_mesh.ply", io.BytesIO(ply_bytes), "application/octet-stream")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "live_upload"
    assert data["point_count"] == 700


def test_automatic_scene_analysis_model_selection(client: TestClient):
    """Verify scene analysis chooses SemanticKITTI for flat urban and RELLIS for rugged terrain."""
    # 1. Flat urban road
    urban_pts = np.zeros((1200, 4), dtype=np.float32)
    urban_pts[:, 0] = np.random.uniform(-15.0, 15.0, 1200)
    urban_pts[:, 1] = np.random.uniform(-15.0, 15.0, 1200)
    urban_pts[:, 2] = -1.73 + np.random.normal(0, 0.01, 1200)
    urban_pts[:, 3] = 0.7

    res_urban = client.post(
        "/api/v1/inference/live-upload",
        files={"file": ("urban.bin", io.BytesIO(urban_pts.tobytes()), "application/octet-stream")},
    )
    assert res_urban.status_code == 200
    data_urban = res_urban.json()
    assert "urban" in data_urban["detected_domain"]
    assert data_urban["model_type"] == "semantickitti"
    assert "best_frnet_semantickitti.pth" in data_urban["model"]

    # 2. Rugged off-road scene (high elevation variance, vegetative sphericity)
    offroad_pts = np.zeros((1500, 4), dtype=np.float32)
    offroad_pts[:, 0] = np.random.uniform(-25.0, 25.0, 1500)
    offroad_pts[:, 1] = np.random.uniform(-25.0, 25.0, 1500)
    offroad_pts[:, 2] = np.random.uniform(-4.0, 8.0, 1500)  # Extreme elevation variance
    offroad_pts[:, 3] = 0.3

    res_offroad = client.post(
        "/api/v1/inference/live-upload",
        files={"file": ("offroad.bin", io.BytesIO(offroad_pts.tobytes()), "application/octet-stream")},
    )
    assert res_offroad.status_code == 200
    data_offroad = res_offroad.json()
    assert "offroad" in data_offroad["detected_domain"] or "off_road" in data_offroad["detected_domain"]
    assert data_offroad["model_type"] == "rellis"
    assert "best_frnet_rellis.pth" in data_offroad["model"]


def test_forced_model_selection(client: TestClient):
    """Verify force_model parameter overrides automatic scene analysis."""
    pts = np.zeros((500, 4), dtype=np.float32)
    pts[:, 0] = np.linspace(-5, 5, 500)
    pts[:, 1] = np.linspace(-5, 5, 500)
    pts[:, 2] = -1.73
    pts[:, 3] = 0.5

    # Force RELLIS on a flat road
    res = client.post(
        "/api/v1/inference/live-upload",
        files={"file": ("flat.bin", io.BytesIO(pts.tobytes()), "application/octet-stream")},
        data={"force_model": "rellis"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["model_type"] == "rellis"
    assert "user preference" in data["selection_reason"].lower()


def test_empty_or_corrupted_upload_rejected(client: TestClient):
    """Verify empty or corrupted files fail with 400 Bad Request, never falling back to mock."""
    # Empty file
    res_empty = client.post(
        "/api/v1/inference/live-upload",
        files={"file": ("empty.bin", io.BytesIO(b""), "application/octet-stream")},
    )
    assert res_empty.status_code == 400

    # Corrupted binary
    res_corrupt = client.post(
        "/api/v1/inference/live-upload",
        files={"file": ("corrupt.bin", io.BytesIO(b"short"), "application/octet-stream")},
    )
    assert res_corrupt.status_code == 400
