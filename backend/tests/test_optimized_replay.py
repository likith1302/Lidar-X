"""Tests for Optimized High-Rate Replay Pipeline & Vectorized Perception Engine."""

import pytest
import numpy as np
from fastapi.testclient import TestClient
try:
    from app.main import app
    from app.services.adaptive_grid import AdaptiveGridService
    from app.services.precompute_service import precompute_service
    from app.services.replay_session import replay_session_manager, ReplaySession
    from app.models.replay_schemas import PlaybackMode
except ImportError:
    from backend.app.main import app
    from backend.app.services.adaptive_grid import AdaptiveGridService
    from backend.app.services.precompute_service import precompute_service
    from backend.app.services.replay_session import replay_session_manager, ReplaySession
    from backend.app.models.replay_schemas import PlaybackMode

client = TestClient(app)


def test_vectorized_adaptive_grid_correctness():
    """Verify that vectorized adaptive grid generates valid cells with accurate statistics."""
    # 2000 points arranged in a plane with a synthetic obstacle
    np.random.seed(42)
    n_pts = 2000
    xs = np.random.uniform(-20, 20, n_pts)
    ys = np.random.uniform(-20, 20, n_pts)
    zs = np.random.normal(0.0, 0.05, n_pts)
    intensities = np.random.uniform(0.1, 0.9, n_pts)
    points = np.column_stack([xs, ys, zs, intensities])

    # Labels: 40 = road (drivable), 10 = car (dynamic)
    labels = np.full(n_pts, 40, dtype=np.uint16)
    # Add a cluster of dynamic car points
    car_mask = (xs > 2.0) & (xs < 5.0) & (ys > 2.0) & (ys < 4.0)
    labels[car_mask] = 10
    points[car_mask, 2] += 1.4  # height bump for vehicle

    cells = AdaptiveGridService.generate_grid_from_points(
        points=points,
        frame_id="test_vec_frame_01",
        labels=labels,
    )

    assert len(cells) > 0
    # Check that levels are assigned
    levels = {c.level.value for c in cells}
    assert "fine" in levels or "medium" in levels

    # Check that car cells are flagged as dynamic or collision hazard
    car_cells = [c for c in cells if c.is_dynamic_obstacle or c.dominant_semantic_class in ("car", "moving-car")]
    assert len(car_cells) > 0

    # Verify bounds and statistics are populated
    sample_cell = cells[0]
    assert len(sample_cell.bounds) == 4
    assert sample_cell.point_count > 0
    assert sample_cell.elevation_min <= sample_cell.elevation_max


def test_precompute_service_snapshot_persistence(tmp_path):
    """Verify that PrecomputeReplayService accurately caches and loads frame snapshots."""
    session_id = "test_precompute_sess_01"
    
    try:
        from app.models.replay_schemas import ReplayFrameStreamPayload, ReplayPointSample
    except ImportError:
        from backend.app.models.replay_schemas import ReplayFrameStreamPayload, ReplayPointSample
    payload = ReplayFrameStreamPayload(
        session_id=session_id,
        sequence_name="Test Sequence",
        frame_index=0,
        total_frames=5,
        frame_filename="000000.bin",
        point_count=100,
        points_sample=[
            ReplayPointSample(x=1.0, y=2.0, z=0.0, intensity=0.5, semantic_class="road", project_category="drivable")
        ],
        category_distribution={"drivable": 100},
        coordinate_mode="local_frame",
        map_mode="local_only",
        data_mode="precomputed_labels",
        state="playing",
        processing_time_ms=12.5,
    )

    # Save frame
    precompute_service.save_precomputed_frame(session_id, 0, payload)

    # Load frame
    loaded = precompute_service.load_precomputed_frame(session_id, 0)
    assert loaded is not None
    assert loaded.session_id == session_id
    assert loaded.frame_filename == "000000.bin"
    assert loaded.point_count == 100
    assert len(loaded.points_sample) == 1


def test_replay_session_modes_and_diagnostics():
    """Verify playback mode switching and high-precision telemetry."""
    session_data = {
        "session_id": "test_diag_sess_01",
        "session_dir": "dummy_dir",
        "sequence_name": "Diagnostic Seq",
        "total_frames": 10,
        "has_predictions": True,
        "data_mode": "precomputed_labels",
        "playback_mode": "offline_precomputed_replay",
        "frames": [{"filename": f"{i:06d}.bin", "bin_path": "fake.bin", "label_path": None, "point_count": 500} for i in range(10)],
    }

    session = replay_session_manager.create_session(session_data)
    assert session.playback_mode == "offline_precomputed_replay"

    # Start playback at 30 FPS
    status = session.start(fps=30.0)
    assert status.state == "playing"
    assert status.fps == 30.0

    # Switch mode to live_processing
    session.set_playback_mode("live_processing")
    assert session.playback_mode == "live_processing"

    # Check diagnostics
    diag = session.get_diagnostics()
    assert diag.playback_mode == "live_processing"
    assert diag.requested_fps == 30.0
    assert diag.device in ("CPU", "CUDA")


def test_precompute_api_routes():
    """Verify HTTP API endpoints for precomputation and mode setting."""
    session_id = "test_api_sess_01"
    session_data = {
        "session_id": session_id,
        "session_dir": "dummy_dir",
        "sequence_name": "API Seq",
        "total_frames": 5,
        "has_predictions": True,
        "data_mode": "precomputed_labels",
        "playback_mode": "offline_precomputed_replay",
        "frames": [{"filename": f"{i:06d}.bin", "bin_path": "fake.bin", "label_path": None, "point_count": 500} for i in range(5)],
    }
    replay_session_manager.create_session(session_data)

    # 1. Check precompute-status
    res_status = client.get(f"/api/v1/replay/{session_id}/precompute-status")
    assert res_status.status_code == 200
    data = res_status.json()
    assert data["session_id"] == session_id
    assert "percent_complete" in data

    # 2. Set mode
    res_mode = client.post(f"/api/v1/replay/{session_id}/set-mode", json={"playback_mode": "live_processing"})
    assert res_mode.status_code == 200
    assert res_mode.json()["playback_mode"] == "live_processing"

    # 3. Start replay with custom FPS
    res_start = client.post(f"/api/v1/replay/{session_id}/start", json={"fps": 60.0, "playback_mode": "offline_precomputed_replay"})
    assert res_start.status_code == 200
    assert res_start.json()["fps"] == 60.0
    assert res_start.json()["playback_mode"] == "offline_precomputed_replay"
