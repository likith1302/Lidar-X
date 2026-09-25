"""Unit tests for FrameRegistry canonicalization, atomic persistence, and status queries."""

import pytest
import numpy as np
from pathlib import Path

from app.services.frame_registry import FrameRegistry
from app.config import settings


def test_canonicalize_frame_id():
    assert FrameRegistry.canonicalize_frame_id("000000.bin") == "000000"
    assert FrameRegistry.canonicalize_frame_id("000001.label") == "000001"
    assert FrameRegistry.canonicalize_frame_id("scan_000002.bin") == "000002"
    assert FrameRegistry.canonicalize_frame_id("frame_000045") == "000045"
    assert FrameRegistry.canonicalize_frame_id("seq_frame_000100") == "000100"
    assert FrameRegistry.canonicalize_frame_id("FRAME_0001 (Mock)") == "FRAME_0001__Mock_"
    assert FrameRegistry.canonicalize_frame_id("custom_test_frame") == "custom_test_frame"
    assert FrameRegistry.canonicalize_frame_id("") == "000000"


def test_atomic_save_and_paths(tmp_path):
    # Test atomic JSON save
    test_json_path = tmp_path / "test.json"
    data = {"key": "value", "count": 42}
    FrameRegistry.atomic_save_json(test_json_path, data)
    assert test_json_path.exists()
    import json
    with open(test_json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["count"] == 42

    # Test atomic NPZ save
    test_npz_path = tmp_path / "test.npz"
    arr = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)
    FrameRegistry.atomic_save_npz(test_npz_path, points=arr)
    assert test_npz_path.exists()
    with np.load(test_npz_path) as npz_data:
        assert npz_data["points"].shape == (1, 4)

    # Test atomic bytes save
    test_bin_path = tmp_path / "test.bin"
    raw = b"\x01\x02\x03\x04"
    FrameRegistry.atomic_save_bytes(test_bin_path, raw)
    assert test_bin_path.exists()
    assert test_bin_path.read_bytes() == raw
