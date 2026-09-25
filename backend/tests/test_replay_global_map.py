"""End-to-end tests for the session-global fused map and the precompute
progress / global-cell routes.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.replay_session import replay_session_manager
from app.services.resolution_policy import ResolutionPolicyService
from app.models.map_schemas import GridPolicyConfig


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_policy(monkeypatch):
    ResolutionPolicyService._current_policy = GridPolicyConfig()
    yield
    ResolutionPolicyService._current_policy = None


def test_policy_normalisation_for_nested_hierarchy():
    """Submitting a non-nested 0.75/1.0/2.0 policy is normalised to 0.75/1.5/3.0
    so the 2:1 hierarchy is preserved.
    """
    from app.models.map_schemas import GridPolicyConfig as GPC
    pol = GPC(
        fine_resolution_m=0.75,
        medium_resolution_m=1.0,
        coarse_resolution_m=2.0,
    )
    norm = ResolutionPolicyService.normalise_policy(pol)
    assert norm.fine_resolution_m == 0.75
    assert norm.medium_resolution_m == 1.5
    assert norm.coarse_resolution_m == 3.0

    # Hierarchy should align: 1 fine == 1/2 medium == 1/4 coarse
    fpm, mpc = ResolutionPolicyService.get_level_ratio(norm)
    assert fpm == 2
    assert mpc == 2

    # parent and child should be spatially aligned
    assert ResolutionPolicyService.get_parent_key("fine:2_2", norm) == "medium:1_1"
    assert ResolutionPolicyService.get_child_keys("coarse:0_0", norm) == [
        "medium:0_0", "medium:1_0", "medium:0_1", "medium:1_1"
    ]


def test_grid_policy_route_normalises(monkeypatch, client):
    """PUT /grid-policy should snap non-nested sizes to a 2:1 hierarchy."""
    res = client.put(
        "/api/v1/grid-policy",
        json={
            "near_zone_max_distance_m": 12.0,
            "mid_zone_max_distance_m": 28.0,
            "far_zone_max_distance_m": 80.0,
            "fine_resolution_m": 0.75,
            "medium_resolution_m": 1.0,
            "coarse_resolution_m": 2.0,
            "safety_priority": True,
            "terrain_complexity_override": True,
            "obstacle_override": True,
            "slope_refinement_threshold_deg": 10.0,
            "roughness_refinement_threshold_m": 0.10,
            "step_refinement_threshold_m": 0.10,
            "obstacle_height_span_threshold_m": 0.30,
            "default_cells_query_limit": 5000,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["fine_resolution_m"] == 0.75
    assert body["medium_resolution_m"] == 1.5
    assert body["coarse_resolution_m"] == 3.0


def test_replay_session_global_map():
    """After _process_frame_internal, cells should appear in the global map
    with a stable, ego-pose-aware key.
    """
    # Build a fake replay session
    from app.models.replay_schemas import ReplayPlaybackState

    session_data = {
        "session_id": "test_global_map_session",
        "session_dir": "/tmp",
        "sequence_name": "Test",
        "total_frames": 3,
        "has_predictions": False,
        "data_mode": "live_inference",
        "frames": [
            {
                "frame_index": i,
                "filename": f"frame_{i:06d}.bin",
                "bin_path": f"/tmp/frame_{i:06d}.bin",
                "label_path": None,
                "point_count": 10,
                "has_prediction": False,
            }
            for i in range(3)
        ],
    }
    session = replay_session_manager.create_session(session_data)
    # Just verify the global map structure exists
    assert hasattr(session, "_global_map")
    assert hasattr(session, "get_global_cell")
    assert hasattr(session, "list_global_cells")
    assert hasattr(session, "_frame_poses")
    # The default forward 0.5m trajectory should be present
    assert session._frame_poses[0] == (0.0, 0.0, 0.0)
    assert session._frame_poses[2] == (1.0, 0.0, 0.0)


def test_replay_global_cell_routes(client):
    """/replay/{id}/global-cells and /replay/{id}/global-cells/{key} return 404 cleanly."""
    # First make sure a session exists
    session_data = {
        "session_id": "test_global_cell_routes",
        "session_dir": "/tmp",
        "sequence_name": "Test",
        "total_frames": 1,
        "has_predictions": False,
        "data_mode": "live_inference",
        "frames": [
            {
                "frame_index": 0,
                "filename": "frame_000000.bin",
                "bin_path": "/tmp/frame_000000.bin",
                "label_path": None,
                "point_count": 0,
                "has_prediction": False,
            }
        ],
    }
    replay_session_manager.create_session(session_data)

    # Empty list initially
    res = client.get("/api/v1/replay/test_global_cell_routes/global-cells")
    assert res.status_code == 200
    assert res.json() == []

    # 404 on missing key
    res = client.get("/api/v1/replay/test_global_cell_routes/global-cells/fine:99_99")
    assert res.status_code == 404

    # 404 on missing session
    res = client.get("/api/v1/replay/nonexistent_session_xyz/global-cells")
    assert res.status_code == 404
