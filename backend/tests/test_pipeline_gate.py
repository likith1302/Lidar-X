"""Integration tests for pipeline status and processing endpoints."""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.services.storage import StorageService
from app.models.schemas import FrameMetadata


@pytest.fixture
def client():
    return TestClient(app)


def test_pipeline_status_nonexistent_frame(client):
    res = client.get("/api/v1/pipeline/nonexistent_frame_9999/status")
    assert res.status_code == 200
    data = res.json()
    assert data["frame_exists"] is False
    assert data["point_cloud_available"] is False
    assert data["semantic_labels_available"] is False
    assert data["object_detection_available"] is False


def test_pipeline_process_end_to_end(client):
    frame_id = "test_pipeline_frame_01"
    
    # 1. Store synthetic points
    points = np.zeros((100, 4), dtype=np.float32)
    # Ground plane points
    for i in range(50):
        points[i] = [float(i % 10), float(i // 10), 0.0, 0.8]
    # Some object points (e.g. car)
    for i in range(50, 100):
        points[i] = [5.0 + (i - 50) * 0.05, 5.0 + (i - 50) * 0.05, 1.0, 0.9]

    meta = FrameMetadata(
        timestamp="2026-09-06T12:00:00Z",
        point_count=100,
        file_size_bytes=1600,
    )
    StorageService.save_frame(frame_id, points, meta)

    # Status before labels
    res = client.get(f"/api/v1/pipeline/{frame_id}/status")
    assert res.status_code == 200
    data = res.json()
    assert data["frame_exists"] is True
    assert data["point_count"] == 100
    assert data["semantic_labels_available"] is False

    # Store labels (first 50 road (40), next 50 car (10))
    sem_classes = np.zeros(100, dtype=np.uint16)
    sem_classes[:50] = 40  # road
    sem_classes[50:] = 10  # car
    inst_ids = np.zeros(100, dtype=np.uint16)
    StorageService.save_labels(frame_id, sem_classes, inst_ids)

    # Status after labels
    res = client.get(f"/api/v1/pipeline/{frame_id}/status")
    assert res.status_code == 200
    data = res.json()
    assert data["semantic_labels_available"] is True
    assert data["semantic_label_count"] == 100
    assert data["labels_match_point_count"] is True

    # Process all stages
    proc_res = client.post(f"/api/v1/pipeline/{frame_id}/process", json={"stages": ["terrain", "objects", "adaptive_grid"]})
    assert proc_res.status_code == 200
    pdata = proc_res.json()
    assert pdata["success"] is True
    assert len(pdata["executed_stages"]) == 3
    assert pdata["pipeline_status"]["terrain_available"] is True
    assert pdata["pipeline_status"]["object_detection_available"] is True
    assert pdata["pipeline_status"]["adaptive_map_available"] is True
