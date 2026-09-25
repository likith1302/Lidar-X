"""Unit tests for the Adaptive Variable-Resolution 2.5D Grid Engine."""

import numpy as np
import pytest

from app.models.map_schemas import (
    ResolutionLevel,
    MapMode,
    TraversabilityState,
    AmbiguityState,
    CellObservationState,
    GridPolicyConfig,
)
from app.services.resolution_policy import ResolutionPolicyService
from app.services.observability import ObservabilityService
from app.services.adaptive_grid import AdaptiveGridService
from app.services.map_fusion import MapFusionService


@pytest.fixture(autouse=True)
def _reset_default_policy(monkeypatch):
    """Force a deterministic 0.5/1.0/2.0m default policy for these tests."""
    default = GridPolicyConfig(
        fine_resolution_m=0.5,
        medium_resolution_m=1.0,
        coarse_resolution_m=2.0,
    )
    ResolutionPolicyService._current_policy = default
    yield
    ResolutionPolicyService._current_policy = None


def test_resolution_policy_zones():
    """Verify radial distance boundaries map to fine, medium, coarse levels."""
    policy = GridPolicyConfig(
        near_zone_max_distance_m=10.0,
        mid_zone_max_distance_m=25.0,
        far_zone_max_distance_m=70.0,
    )

    assert ResolutionPolicyService.determine_base_level(5.0, policy) == ResolutionLevel.FINE
    assert ResolutionPolicyService.determine_base_level(10.0, policy) == ResolutionLevel.FINE
    assert ResolutionPolicyService.determine_base_level(18.0, policy) == ResolutionLevel.MEDIUM
    assert ResolutionPolicyService.determine_base_level(25.0, policy) == ResolutionLevel.MEDIUM
    assert ResolutionPolicyService.determine_base_level(30.0, policy) == ResolutionLevel.COARSE
    assert ResolutionPolicyService.determine_base_level(65.0, policy) == ResolutionLevel.COARSE


def test_hierarchical_nesting_keys():
    """Verify power-of-two hierarchical parent and child key relations."""
    # Fine key (0.5m) -> Medium parent (1.0m)
    fine_key = "fine:10_6"
    parent = ResolutionPolicyService.get_parent_key(fine_key)
    assert parent == "medium:5_3"

    # Medium key (1.0m) -> Coarse parent (2.0m)
    med_key = "medium:5_3"
    parent_coarse = ResolutionPolicyService.get_parent_key(med_key)
    assert parent_coarse == "coarse:2_1"

    # Coarse key -> No parent
    assert ResolutionPolicyService.get_parent_key("coarse:2_1") is None

    # Coarse children -> 4 medium cells
    children = ResolutionPolicyService.get_child_keys("coarse:2_1")
    assert len(children) == 4
    assert "medium:4_2" in children
    assert "medium:5_2" in children
    assert "medium:4_3" in children
    assert "medium:5_3" in children

    # Medium children -> 4 fine cells
    med_children = ResolutionPolicyService.get_child_keys("medium:5_3")
    assert len(med_children) == 4
    assert "fine:10_6" in med_children
    assert "fine:11_6" in med_children
    assert "fine:10_7" in med_children
    assert "fine:11_7" in med_children


def test_feature_refinement_overrides():
    """Test safety, obstacle, and terrain complexity overrides."""
    policy = GridPolicyConfig()

    # Dynamic actor in far zone gets refined to FINE when safety_priority=True
    refined = ResolutionPolicyService.refine_level(
        base_level=ResolutionLevel.COARSE,
        is_dynamic=True,
        policy=policy,
    )
    assert refined == ResolutionLevel.FINE

    # Obstacle with vertical span gets refined to FINE
    refined_obs = ResolutionPolicyService.refine_level(
        base_level=ResolutionLevel.COARSE,
        is_obstacle=True,
        elevation_range_m=0.45,
        policy=policy,
    )
    assert refined_obs == ResolutionLevel.FINE

    # Steep slope promotes COARSE to MEDIUM
    refined_slope = ResolutionPolicyService.refine_level(
        base_level=ResolutionLevel.COARSE,
        slope_deg=22.0,
        policy=policy,
    )
    assert refined_slope == ResolutionLevel.MEDIUM

    # Steep slope promotes MEDIUM to FINE
    refined_slope_med = ResolutionPolicyService.refine_level(
        base_level=ResolutionLevel.MEDIUM,
        slope_deg=22.0,
        policy=policy,
    )
    assert refined_slope_med == ResolutionLevel.FINE


def test_grid_generation_and_aggregation():
    """Test point aggregation into 2.5D cells with geometric & semantic properties."""
    # Create synthetic points: near road + far obstacle + dynamic actor
    near_road = np.array([
        [2.0, 1.0, -1.73, 0.5],
        [2.1, 1.1, -1.72, 0.5],
        [2.2, 1.2, -1.74, 0.5],
        [2.3, 1.3, -1.73, 0.5],
    ], dtype=np.float32)

    far_road = np.array([
        [35.0, 5.0, -1.70, 0.5],
        [35.5, 5.2, -1.68, 0.5],
        [36.0, 5.5, -1.72, 0.5],
    ], dtype=np.float32)

    obstacle = np.array([
        [5.0, -3.0, -1.73, 0.8],
        [5.1, -3.1, -1.00, 0.8],
        [5.2, -3.2, 0.50, 0.8],
    ], dtype=np.float32)

    all_pts = np.vstack([near_road, far_road, obstacle])
    
    # Labels: 40=road for road, 50=building for obstacle
    labels = np.array([40, 40, 40, 40, 40, 40, 40, 50, 50, 50], dtype=np.uint32)

    cells = AdaptiveGridService.generate_grid_from_points(
        points=all_pts,
        frame_id="test_frame_01",
        labels=labels,
    )

    assert len(cells) >= 3

    # Check near road cell (level should be FINE)
    near_cells = [c for c in cells if c.world_x < 5.0 and c.world_y > 0]
    assert len(near_cells) > 0
    assert near_cells[0].level == ResolutionLevel.FINE
    assert near_cells[0].traversability_state == TraversabilityState.DRIVABLE
    assert near_cells[0].dominant_semantic_class == "road"

    # Check far road cell (level should be COARSE)
    far_cells = [c for c in cells if c.world_x > 30.0]
    assert len(far_cells) > 0
    assert far_cells[0].level == ResolutionLevel.COARSE

    # Check obstacle cell
    obs_cells = [c for c in cells if c.world_y < -2.0]
    assert len(obs_cells) > 0
    assert obs_cells[0].is_static_obstacle is True
    assert obs_cells[0].traversability_state in [
        TraversabilityState.COLLISION_HAZARD,
        TraversabilityState.NON_TRAVERSABLE,
    ]


def test_observability_and_occlusion():
    """Test geometric ray and line-of-sight occlusion logic."""
    # Obstacle at (10, 0)
    obs = np.array([[10.0, 0.0]])
    sectors = ObservabilityService.extract_occlusion_sectors(obs, sensor_x=0.0, sensor_y=0.0, obstacle_radius=1.0)
    assert len(sectors) == 1

    # Behind obstacle at (20, 0) with 0 points should be partially occluded
    obs_state = ObservabilityService.compute_cell_observability(
        cell_world_x=20.0,
        cell_world_y=0.0,
        point_count=0,
        sensor_x=0.0,
        sensor_y=0.0,
        max_range_m=80.0,
        occlusion_sectors=sectors,
    )
    assert obs_state == CellObservationState.PARTIALLY_OCCLUDED

    # Beyond max range (90m) should be out_of_range
    out_state = ObservabilityService.compute_cell_observability(
        cell_world_x=90.0,
        cell_world_y=0.0,
        point_count=0,
        max_range_m=80.0,
    )
    assert out_state == CellObservationState.OUT_OF_RANGE


def test_map_fusion_local_and_global():
    """Test local-only single frame update vs global coordinate transform fusion."""
    pts1 = np.array([
        [2.0, 0.0, -1.73, 0.5],
        [2.2, 0.1, -1.73, 0.5],
    ], dtype=np.float32)

    pts2 = np.array([
        [2.0, 0.0, -1.73, 0.5],
        [2.2, 0.1, -1.73, 0.5],
    ], dtype=np.float32)

    # 1. Local-only mode (ego_pose is None)
    meta_local, cells_local = MapFusionService.update_map(
        map_id="test_local_map",
        frame_id="frame_001",
        points=pts1,
        ego_pose=None,
    )
    assert meta_local.mode == MapMode.LOCAL_ONLY
    assert meta_local.frame_count == 1
    assert len(cells_local) > 0

    # 2. Global fusion mode with translation [10.0, 0.0, 0.0]
    meta_global, cells_global = MapFusionService.update_map(
        map_id="test_global_map",
        frame_id="frame_001",
        points=pts1,
        ego_pose=[10.0, 0.0, 0.0],  # translated 10m forward
    )
    assert meta_global.mode == MapMode.GLOBAL_FUSION
    # Center of cells should now be shifted to ~12m in world frame
    assert any(c.world_x > 11.0 for c in cells_global)
