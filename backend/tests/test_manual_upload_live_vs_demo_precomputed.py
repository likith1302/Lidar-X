"""Verification tests for Manual ZIP Upload vs. Built-in Demo Precomputed Replay.

Rules verified:
1. Manual ZIP upload (e.g. uploading foveamap_sequence03_replay.zip or custom sequence)
   MUST NEVER load from precomputed disk/RAM cache.
2. Manual ZIP uploads MUST execute real computing (live Fast-FRNet neural segmentation,
   geometric terrain analysis, DBSCAN instance clustering & Kalman MOT, adaptive 2.5D grid).
3. Precomputed zero-overhead playback (60 FPS snapshot streaming) is strictly reserved
   for the built-in demo section.
"""

import io
import zipfile
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.sequence_replay import sequence_replay_service
from app.services.replay_session import replay_session_manager, ReplaySession
from app.services.precompute_service import precompute_service
from app.models.replay_schemas import PlaybackMode


@pytest.fixture
def client():
    return TestClient(app)


def test_manual_zip_upload_never_uses_precomputed(client):
    """Uploading a ZIP whose filename matches a precomputed demo sequence must NOT use precomputed cache."""
    # Create synthetic ZIP
    num_pts = 100
    pts = np.zeros((num_pts, 4), dtype=np.float32)
    pts[:, 0] = np.linspace(-10, 10, num_pts)
    pts[:, 1] = np.linspace(-5, 5, num_pts)
    pts[:, 2] = -1.73
    pts[:, 3] = 0.5

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sequences/03/velodyne/000000.bin", pts.tobytes())

    zip_bytes = zip_buffer.getvalue()

    # Upload with name identical to demo sequence
    files = {"file": ("foveamap_sequence03_replay.zip", zip_bytes, "application/zip")}
    res = client.post("/api/v1/replay/upload-sequence", files=files)
    assert res.status_code == 200
    data = res.json()

    # Verify manual upload metadata
    assert data["is_manual_upload"] is True
    assert data["playback_mode"] == "live_processing"
    assert data["semantic_source"] == "LIVE FAST-FRNET"
    # Ensure session ID was scoped to prevent collision with precomputed demo directory
    assert data["session_id"].startswith("upload_")

    session_id = data["session_id"]
    session = replay_session_manager.get_session(session_id)
    assert session is not None
    assert session.is_manual_upload is True
    assert session.is_demo is False
    assert session.playback_mode == PlaybackMode.LIVE_PROCESSING

    # Advance frame - must execute live pipeline, NOT load precomputed snapshot
    frame_0 = session.advance_and_get_frame()
    assert frame_0 is not None
    assert frame_0.semantic_source == "LIVE FAST-FRNET"
    assert frame_0.performance is not None
    assert "Fast-FRNet" in frame_0.performance.model_name
    assert frame_0.performance.model_name != "Precomputed Playback (Snapshot)"
    # Live compute latency must be tracked (not 0.0 ms)
    assert frame_0.performance.timings.total_latency_ms > 0

    # Verify status endpoint reflects live mode
    status_res = client.get(f"/api/v1/replay/{session_id}/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["is_manual_upload"] is True
    assert status_data["playback_mode"] == "live_processing"
    assert status_data["semantic_source"] == "LIVE FAST-FRNET"

    # Clean up
    replay_session_manager.delete_session(session_id)


def test_demo_section_retains_precomputed_playback():
    """Built-in demo sequence (foveamap_sequence01_replay) retains precomputed replay capabilities."""
    demo_session = replay_session_manager.get_session("foveamap_sequence01_replay")
    assert demo_session is not None
    assert demo_session.is_manual_upload is False
    assert demo_session.is_demo is True
    assert demo_session.playback_mode == PlaybackMode.OFFLINE_PRECOMPUTED_REPLAY
    assert demo_session.semantic_source == "PRECOMPUTED"
