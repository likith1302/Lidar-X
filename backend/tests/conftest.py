"""Pytest fixtures and synthetic LiDAR generators for LiDAR-X Backend."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> Generator:
    """Create a temporary data directory for tests to isolate file storage."""
    test_dir = Path(tempfile.mkdtemp(prefix="foveamap_test_", dir=tempfile.gettempdir()))
    old_base_dir = settings.BASE_DIR
    settings.BASE_DIR = test_dir
    settings.DATA_DIR_NAME = "data"

    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    settings.TERRAIN_DIR.mkdir(parents=True, exist_ok=True)
    settings.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    settings.OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    settings.MAPS_DIR.mkdir(parents=True, exist_ok=True)

    zip_src = old_base_dir / "foveamap_sequence01_replay.zip"
    if zip_src.exists():
        shutil.copy2(zip_src, test_dir / "foveamap_sequence01_replay.zip")

    yield

    settings.BASE_DIR = old_base_dir
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def create_kitti_bin():
    """Helper factory to create SemanticKITTI binary bytes from (N, 4) numpy array."""
    def _create(points: np.ndarray) -> bytes:
        return points.astype(np.float32).tobytes()
    return _create


@pytest.fixture
def flat_road_points() -> np.ndarray:
    """Generate a clean synthetic flat road surface (z = -1.73m)."""
    x = np.linspace(-10, 10, 40)
    y = np.linspace(-10, 10, 40)
    xx, yy = np.meshgrid(x, y)
    n_pts = xx.size
    
    pts = np.zeros((n_pts, 4), dtype=np.float32)
    pts[:, 0] = xx.ravel()
    pts[:, 1] = yy.ravel()
    pts[:, 2] = -1.73 + np.random.normal(0, 0.005, n_pts)
    pts[:, 3] = 0.5
    return pts


@pytest.fixture
def sloped_ramp_points() -> np.ndarray:
    """Generate points on an inclined plane (20 degree slope along X)."""
    x = np.linspace(2, 10, 30)
    y = np.linspace(-5, 5, 30)
    xx, yy = np.meshgrid(x, y)
    n_pts = xx.size
    
    pts = np.zeros((n_pts, 4), dtype=np.float32)
    pts[:, 0] = xx.ravel()
    pts[:, 1] = yy.ravel()
    pts[:, 2] = -1.73 + (xx.ravel() - 2.0) * np.tan(np.radians(20.0))
    pts[:, 3] = 0.4
    return pts


@pytest.fixture
def curb_points() -> np.ndarray:
    """Generate points simulating a sidewalk curb (15cm vertical step at y = 2.0m)."""
    x = np.linspace(1, 8, 30)
    y_road = np.linspace(-2, 1.95, 20)
    y_sidewalk = np.linspace(2.05, 4, 20)
    
    xx_r, yy_r = np.meshgrid(x, y_road)
    road_pts = np.zeros((xx_r.size, 4), dtype=np.float32)
    road_pts[:, 0] = xx_r.ravel()
    road_pts[:, 1] = yy_r.ravel()
    road_pts[:, 2] = -1.73
    road_pts[:, 3] = 0.6

    xx_s, yy_s = np.meshgrid(x, y_sidewalk)
    side_pts = np.zeros((xx_s.size, 4), dtype=np.float32)
    side_pts[:, 0] = xx_s.ravel()
    side_pts[:, 1] = yy_s.ravel()
    side_pts[:, 2] = -1.73 + 0.15
    side_pts[:, 3] = 0.3

    return np.vstack([road_pts, side_pts])
