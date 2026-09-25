"""Test demo data endpoints and replay auto-loading."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_demo_sequence_replay_autoload():
    with TestClient(app) as client:
        # 1. Check replay status
        res = client.get("/api/v1/replay/semantic_kitti_sequence_00/status")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()
        assert data["session_id"] == "semantic_kitti_sequence_00"
        assert data["total_frames"] > 0
        assert "has_predictions" in data
        assert isinstance(data["has_predictions"], bool)

        # 2. Check advance / get next frame
        f_res = client.get("/api/v1/replay/semantic_kitti_sequence_00/next-frame")
        assert f_res.status_code == 200, f"Expected 200, got {f_res.status_code}: {f_res.text}"
        f_data = f_res.json()
        assert f_data["frame_index"] == 0
        assert f_data["point_count"] > 0
        assert "points_sample" in f_data

        # 3. Check session metrics
        s_res = client.get("/api/v1/metrics/session/semantic_kitti_sequence_00")
        assert s_res.status_code == 200, f"Expected 200, got {s_res.status_code}: {s_res.text}"


def test_demo_mock_frame_and_metrics_resolution():
    with TestClient(app) as client:
        # 1. Check frame metrics with mock name
        m_res = client.get("/api/v1/metrics/frame/FRAME_0001%20(Mock)")
        assert m_res.status_code == 200, f"Expected 200, got {m_res.status_code}: {m_res.text}"

        # 2. Check frame details with mock name
        fd_res = client.get("/api/v1/frames/FRAME_0001%20(Mock)")
        assert fd_res.status_code == 200, f"Expected 200, got {fd_res.status_code}: {fd_res.text}"
        fd_data = fd_res.json()
        assert fd_data["point_count"] > 0
        assert len(fd_data["sample_points"]) > 0
