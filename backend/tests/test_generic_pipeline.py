"""Tests for fully generic, dataset-sequence independent LiDAR-X pipeline.

Validates:
1. Multiple different arbitrary sequence names (not Sequence 00/01/03).
2. Input with labels (Ground Truth mode) vs input without labels (Geometric fallback mode).
3. Monotonic frame advancement (0, 1, 2, ... N-1) with no stuck frames.
4. Cache namespace isolation between sequences.
5. Ingestion of directories / ZIPs.
6. Session listing and cleanup.
7. Preservation of the demo section.
"""

import io
import time
import zipfile
import numpy as np
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.services.sequence_replay import sequence_replay_service
from app.services.replay_session import replay_session_manager
from app.services.precompute_service import precompute_service


@pytest.fixture
def client():
    return TestClient(app)


def make_custom_sequence_zip(
    seq_folder_name: str = "custom_test_seq",
    scan_prefix: str = "scan",
    num_frames: int = 5,
    with_labels: bool = True,
    with_poses: bool = True,
) -> bytes:
    """Generate in-memory ZIP for an arbitrary sequence with custom folder/filenames."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        poses_lines = []
        for idx in range(num_frames):
            num_pts = 300
            xs = np.linspace(-15.0, 35.0, num_pts, dtype=np.float32)
            ys = np.linspace(-10.0, 10.0, num_pts, dtype=np.float32)
            zs = np.full(num_pts, -1.3, dtype=np.float32)  # Road plane height
            intensities = np.full(num_pts, 0.5, dtype=np.float32)

            # Insert moving vehicle points
            car_x = 8.0 + idx * 1.2
            xs[20:45] = car_x + np.random.uniform(-0.8, 0.8, 25).astype(np.float32)
            ys[20:45] = 1.5 + np.random.uniform(-0.5, 0.5, 25).astype(np.float32)
            zs[20:45] = -0.2 + np.random.uniform(-0.3, 0.3, 25).astype(np.float32)

            scan_array = np.stack([xs, ys, zs, intensities], axis=1).astype(np.float32)
            bin_data = scan_array.tobytes()

            fname = f"{scan_prefix}_{idx:04d}"
            zf.writestr(f"{seq_folder_name}/velodyne/{fname}.bin", bin_data)

            if with_labels:
                labels = np.full(num_pts, 40, dtype=np.uint32)  # road class 40
                labels[20:45] = 10 | (1 << 16)  # car class 10, instance 1
                zf.writestr(f"{seq_folder_name}/predictions/{fname}.label", labels.tobytes())

            # Optional 3x4 KITTI pose line
            # r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
            tx = idx * 0.8
            tz = idx * 0.05
            poses_lines.append(f"1.0 0.0 0.0 {tx:.4f} 0.0 1.0 0.0 0.0 0.0 0.0 1.0 {tz:.4f}\n")

        if with_poses:
            zf.writestr(f"{seq_folder_name}/poses.txt", "".join(poses_lines))
            zf.writestr(f"{seq_folder_name}/calib.txt", "P0: 1 0 0 0 0 1 0 0 0 0 1 0\n")
            zf.writestr(f"{seq_folder_name}/times.txt", "\n".join([f"{i * 0.1:.4f}" for i in range(num_frames)]))

    buf.seek(0)
    return buf.read()


def test_generic_sequence_with_labels(client):
    """Test full pipeline on arbitrary custom sequence WITH ground truth labels."""
    seq_id = "seq_test_custom_alpha"
    zip_bytes = make_custom_sequence_zip(
        seq_folder_name="alpha_dataset",
        scan_prefix="alpha_frame",
        num_frames=4,
        with_labels=True,
        with_poses=True,
    )

    # 1. Ingestion
    res = client.post(
        f"/api/v1/replay/upload-sequence?session_id={seq_id}",
        files={"file": (f"{seq_id}.zip", zip_bytes, "application/zip")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == seq_id
    assert data["total_frames"] == 4
    assert data["has_predictions"] is True
    assert data["has_poses"] is True
    assert data["data_mode"] == "precomputed_labels"
    assert data["semantic_source"] in ("GROUND TRUTH", "PRECOMPUTED")
    assert len(data["frame_filenames"]) == 4
    assert "alpha_frame_0000.bin" in data["frame_filenames"][0]

    # 2. Sequential frame advancement (must advance 0 -> 1 -> 2 -> 3 without sticking)
    seen_indices = []
    for expected_idx in range(4):
        f_res = client.get(f"/api/v1/replay/{seq_id}/next-frame")
        assert f_res.status_code == 200
        f_data = f_res.json()
        assert f_data["frame_index"] == expected_idx
        assert f_data["total_frames"] == 4
        assert f_data["semantic_source"] == "GROUND TRUTH"
        assert f_data["point_count"] > 0
        assert f_data["map_result"] is not None
        assert f_data["terrain_result"] is not None
        seen_indices.append(f_data["frame_index"])

    assert seen_indices == [0, 1, 2, 3]

    # 3. Seeking
    seek_res = client.post(f"/api/v1/replay/{seq_id}/seek", json={"frame_index": 1})
    assert seek_res.status_code == 200
    assert seek_res.json()["current_frame_index"] == 1

    next_after_seek = client.get(f"/api/v1/replay/{seq_id}/next-frame")
    assert next_after_seek.status_code == 200
    assert next_after_seek.json()["frame_index"] == 1


def test_generic_sequence_without_labels(client):
    """Test full pipeline on arbitrary sequence WITHOUT labels (graceful geometric fallback)."""
    seq_id = "seq_test_no_labels_beta"
    zip_bytes = make_custom_sequence_zip(
        seq_folder_name="beta_unlabeled",
        scan_prefix="beta_scan",
        num_frames=3,
        with_labels=False,
        with_poses=False,
    )

    # 1. Ingestion
    res = client.post(
        f"/api/v1/replay/upload-sequence?session_id={seq_id}",
        files={"file": (f"{seq_id}.zip", zip_bytes, "application/zip")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == seq_id
    assert data["total_frames"] == 3
    assert data["has_predictions"] is False
    assert data["has_poses"] is False
    assert data["data_mode"] == "live_inference"

    # 2. Advance frame - verifies fallback to LIVE GEOMETRIC
    f_res = client.get(f"/api/v1/replay/{seq_id}/next-frame")
    assert f_res.status_code == 200
    f_data = f_res.json()
    assert f_data["frame_index"] == 0
    assert f_data["semantic_source"] in ("LIVE GEOMETRIC", "LIVE FAST-FRNET", "LIVE SALSANEXT", "LIVE GEOMETRIC — SALSANEXT PROCESSING", "LIVE GEOMETRIC — FAST-FRNET PROCESSING")
    assert f_data["point_count"] > 0
    assert len(f_data["points_sample"]) > 0


def test_cache_namespace_isolation(client):
    """Verify that distinct sequences have isolated cache namespaces and don't leak frames."""
    seq1_id = "seq_isolated_1"
    seq2_id = "seq_isolated_2"

    z1 = make_custom_sequence_zip("seq1_dir", "s1_frame", num_frames=2, with_labels=True)
    z2 = make_custom_sequence_zip("seq2_dir", "s2_frame", num_frames=3, with_labels=False)

    client.post(f"/api/v1/replay/upload-sequence?session_id={seq1_id}", files={"file": (f"{seq1_id}.zip", z1, "application/zip")})
    client.post(f"/api/v1/replay/upload-sequence?session_id={seq2_id}", files={"file": (f"{seq2_id}.zip", z2, "application/zip")})

    # Read from seq1
    f1 = client.get(f"/api/v1/replay/{seq1_id}/next-frame").json()
    assert "s1_frame" in f1["frame_filename"]
    assert f1["semantic_source"] == "GROUND TRUTH"

    # Read from seq2
    f2 = client.get(f"/api/v1/replay/{seq2_id}/next-frame").json()
    assert "s2_frame" in f2["frame_filename"]
    assert f2["semantic_source"] in ("LIVE GEOMETRIC", "LIVE FAST-FRNET", "LIVE SALSANEXT", "LIVE GEOMETRIC — SALSANEXT PROCESSING", "LIVE GEOMETRIC — FAST-FRNET PROCESSING")

    # Verify session status respects individual frame totals
    st1 = client.get(f"/api/v1/replay/{seq1_id}/status").json()
    st2 = client.get(f"/api/v1/replay/{seq2_id}/status").json()
    assert st1["total_frames"] == 2
    assert st2["total_frames"] == 3


def test_session_listing_and_deletion(client):
    """Verify listing discovered sessions and deleting a session."""
    list_res = client.get("/api/v1/replay/sessions")
    assert list_res.status_code == 200
    sessions = list_res.json()
    assert isinstance(sessions, list)
    assert len(sessions) > 0

    # Delete one of the temporary test sessions
    del_res = client.delete("/api/v1/replay/seq_isolated_1")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True


def test_demo_section_preserved(client):
    """Verify that the demo section (foveamap_sequence01_replay and demo alias) is preserved."""
    res = client.get("/api/v1/replay/foveamap_sequence01_replay/status")
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == "foveamap_sequence01_replay"
    assert data["total_frames"] == 100

    # Test demo alias
    alias_res = client.get("/api/v1/replay/demo/status")
    assert alias_res.status_code == 200
    alias_data = alias_res.json()
    assert alias_data["total_frames"] == 100
