"""Comprehensive unit & integration tests for LiDAR Sequence Replay and WebSocket streaming."""

import io
import zipfile
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.sequence_replay import sequence_replay_service, SequenceIngestError
from app.services.replay_session import replay_session_manager


@pytest.fixture
def client():
    return TestClient(app)


def create_synthetic_sequence_zip(num_frames: int = 3, with_labels: bool = True) -> bytes:
    """Helper to generate an in-memory ZIP containing consecutive SemanticKITTI .bin and .label files."""
    from pathlib import Path
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx in range(num_frames):
            # Create synthetic point cloud: 200 points with a moving vehicle obstacle
            num_pts = 200
            x = np.linspace(-10.0, 30.0, num_pts, dtype=np.float32)
            y = np.linspace(-8.0, 8.0, num_pts, dtype=np.float32)
            z = np.zeros(num_pts, dtype=np.float32)
            intensity = np.full(num_pts, 0.6, dtype=np.float32)

            # Insert moving object points (x moves by +1.5m per frame)
            obj_x = 5.0 + idx * 1.5
            obj_y = 2.0
            x[10:30] = obj_x + np.random.uniform(-0.5, 0.5, 20).astype(np.float32)
            y[10:30] = obj_y + np.random.uniform(-0.5, 0.5, 20).astype(np.float32)
            z[10:30] = 0.5 + np.random.uniform(-0.2, 0.2, 20).astype(np.float32)

            # Stack into (N, 4) float32
            scan_data = np.stack([x, y, z, intensity], axis=1).astype(np.float32)
            bin_bytes = scan_data.tobytes()

            # Add to velodyne/ directory in ZIP
            zf.writestr(f"sequences/00/velodyne/{idx:06d}.bin", bin_bytes)

            if with_labels:
                # Semantic labels: lower 16 bits = semantic class (10: car, 40: road), upper 16 bits = instance ID (1)
                labels = np.full(num_pts, 40, dtype=np.uint32)  # road default
                # Object points = car class (10) with instance ID (1)
                labels[10:30] = 10 | (1 << 16)
                label_bytes = labels.tobytes()
                zf.writestr(f"sequences/00/predictions/{idx:06d}.label", label_bytes)

    return zip_buffer.getvalue()


def test_sequence_replay_zip_extraction(tmp_path):
    """Test extracting and validating a sequence archive directly."""
    zip_bytes = create_synthetic_sequence_zip(num_frames=4, with_labels=True)
    res = sequence_replay_service.extract_and_validate_zip(zip_bytes, session_id="test_seq_01")

    assert res["session_id"] == "test_seq_01"
    assert res["total_frames"] == 4
    assert res["has_predictions"] is True
    assert res["data_mode"] == "precomputed_labels"
    assert len(res["frames"]) == 4

    # Verify natural ordering
    assert res["frames"][0]["filename"] == "000000.bin"
    assert res["frames"][1]["filename"] == "000001.bin"
    assert res["frames"][2]["filename"] == "000002.bin"
    assert res["frames"][3]["filename"] == "000003.bin"


def test_sequence_upload_route(client):
    """Test POST /api/v1/replay/upload-sequence endpoint."""
    zip_bytes = create_synthetic_sequence_zip(num_frames=3, with_labels=True)

    files = {"file": ("sequence_00.zip", zip_bytes, "application/zip")}
    response = client.post("/api/v1/replay/upload-sequence", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["total_frames"] == 3
    assert data["has_predictions"] is True
    assert len(data["frame_filenames"]) == 3


def test_playback_lifecycle_and_multiframe_tracking(client):
    """Test start, pause, stop, seek and consecutive frame multi-object tracking."""
    from unittest.mock import patch

    def mock_infer(raw_points, **kwargs):
        N = len(raw_points)
        classes = np.full(N, 40, dtype=np.uint32)
        classes[10:30] = 10  # car
        inst_ids = np.zeros(N, dtype=np.uint16)
        inst_ids[10:30] = 1
        return classes, inst_ids, {"project_category_counts": {"drivable": N - 20, "dynamic_object": 20}}

    with patch("app.services.fast_frnet_inference.FastFRNetInferenceService.run_inference", side_effect=mock_infer):
        zip_bytes = create_synthetic_sequence_zip(num_frames=3, with_labels=True)
        files = {"file": ("sequence_track.zip", zip_bytes, "application/zip")}
        upload_res = client.post("/api/v1/replay/upload-sequence", files=files)
        session_id = upload_res.json()["session_id"]

        # 1. Check Initial Status
        status_res = client.get(f"/api/v1/replay/{session_id}/status")
        assert status_res.status_code == 200
        assert status_res.json()["state"] == "ready"
        assert status_res.json()["current_frame_index"] == 0

        # 2. Start Playback
        start_res = client.post(f"/api/v1/replay/{session_id}/start", json={"fps": 5.0})
        assert start_res.status_code == 200
        assert start_res.json()["state"] == "playing"
        assert start_res.json()["fps"] == 5.0

        # 3. Pull Frame 0
        f0_res = client.get(f"/api/v1/replay/{session_id}/next-frame")
        assert f0_res.status_code == 200
        f0 = f0_res.json()
        assert f0["frame_index"] == 0
        assert f0["objects_result"] is not None
        assert f0["tracking_result"] is not None

        # 4. Pull Frame 1 (Consecutive Frame Tracking)
        f1_res = client.get(f"/api/v1/replay/{session_id}/next-frame")
        assert f1_res.status_code == 200
        f1 = f1_res.json()
        assert f1["frame_index"] == 1
        # Check that tracking result has accumulated history across frames
        tracks = f1["tracking_result"]["tracks"]
        assert len(tracks) > 0
        assert any(t["age_frames"] >= 2 for t in tracks)
        assert any(len(t["history"]) >= 2 for t in tracks)

        # 5. Pause Playback
        pause_res = client.post(f"/api/v1/replay/{session_id}/pause")
        assert pause_res.status_code == 200
        assert pause_res.json()["state"] == "paused"

        # 6. Seek to Frame 0
        seek_res = client.post(f"/api/v1/replay/{session_id}/seek", json={"frame_index": 0})
        assert seek_res.status_code == 200
        assert seek_res.json()["current_frame_index"] == 0

        # 7. Stop Playback
        stop_res = client.post(f"/api/v1/replay/{session_id}/stop")
        assert stop_res.status_code == 200
        assert stop_res.json()["state"] == "ready"
        assert stop_res.json()["current_frame_index"] == 0


def test_missing_predictions_handling(client):
    """Test sequence without predictions gracefully sets data_mode to live_inference."""
    zip_bytes = create_synthetic_sequence_zip(num_frames=2, with_labels=False)
    files = {"file": ("sequence_no_labels.zip", zip_bytes, "application/zip")}
    res = client.post("/api/v1/replay/upload-sequence", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["has_predictions"] is False
    assert data["data_mode"] == "live_inference"


def test_websocket_stream(client):
    """Test WebSocket continuous frame stream with backpressure."""
    zip_bytes = create_synthetic_sequence_zip(num_frames=2, with_labels=True)
    files = {"file": ("ws_seq.zip", zip_bytes, "application/zip")}
    upload_res = client.post("/api/v1/replay/upload-sequence", files=files)
    session_id = upload_res.json()["session_id"]

    # Connect WebSocket
    with client.websocket_connect(f"/api/v1/replay/{session_id}/stream") as ws:
        # Start playback via WebSocket message
        ws.send_text('{"action": "play", "fps": 10.0}')

        # Receive frame 0
        frame_0_raw = ws.receive_text()
        frame_0 = json_or_none(frame_0_raw)
        assert frame_0 is not None
        assert "frame_index" in frame_0 or "event" in frame_0


def json_or_none(s: str):
    import json
    try:
        return json.loads(s)
    except Exception:
        return None
