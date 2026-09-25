import math
import numpy as np
import pytest

from app.services.semantic_registry import SemanticClassRegistry
from app.services.instance_clustering import (
    cluster_points_to_instances_3d,
    calculate_pca_yaw_and_obb,
)
from app.services.adaptive_grid import AdaptiveGridService


def test_semantic_registry_rellis_mappings():
    """Verify RELLIS-3D class roles and properties in SemanticClassRegistry."""
    dirt_info = SemanticClassRegistry.get_class_info(1, "rellis")
    assert dirt_info.name == "dirt"
    assert dirt_info.category.value == "drivable"
    assert dirt_info.surface_role == "ground"
    assert not dirt_info.is_dynamic
    assert not dirt_info.clusterable

    vehicle_info = SemanticClassRegistry.get_class_info(8, "rellis")
    assert vehicle_info.name == "vehicle"
    assert vehicle_info.category.value == "dynamic_object"
    assert vehicle_info.is_dynamic
    assert vehicle_info.clusterable

    person_info = SemanticClassRegistry.get_class_info(17, "rellis")
    assert person_info.name == "person"
    assert person_info.category.value == "dynamic_object"
    assert person_info.is_dynamic
    assert person_info.clusterable

    log_info = SemanticClassRegistry.get_class_info(15, "rellis")
    assert log_info.name == "log"
    assert log_info.category.value == "static_obstacle"
    assert not log_info.is_dynamic
    assert log_info.clusterable

    fence_info = SemanticClassRegistry.get_class_info(18, "rellis")
    assert fence_info.name == "fence"
    assert fence_info.category.value == "static_obstacle"


def test_semantic_registry_semantickitti_mappings():
    """Verify SemanticKITTI class roles and properties."""
    road_info = SemanticClassRegistry.get_class_info(40, "semantickitti")
    assert road_info.name == "road"
    assert road_info.category.value == "drivable"
    assert road_info.surface_role == "ground"

    car_info = SemanticClassRegistry.get_class_info(10, "semantickitti")
    assert car_info.name == "car"
    assert car_info.category.value == "dynamic_object"
    assert car_info.is_dynamic
    assert car_info.clusterable

    person_info = SemanticClassRegistry.get_class_info(30, "semantickitti")
    assert person_info.name == "person"
    assert person_info.category.value == "dynamic_object"
    assert person_info.is_dynamic
    assert person_info.clusterable


def test_pca_yaw_and_obb_tight_dimensions():
    """Verify PCA accurately extracts orientation and eliminates AABB bounding box inflation."""
    true_l = 4.0
    true_w = 1.8
    true_h = 1.4
    center = np.array([15.0, 5.0, -0.5])
    theta = math.radians(45.0)  # Rotated by 45 degrees

    rng = np.random.RandomState(42)
    local_pts = []
    for _ in range(300):
        lx = rng.uniform(-true_l / 2, true_l / 2)
        ly = rng.uniform(-true_w / 2, true_w / 2)
        lz = rng.uniform(-true_h / 2, true_h / 2)
        local_pts.append([lx, ly, lz])
    local_pts = np.array(local_pts)

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rot_matrix = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1]
    ])
    world_pts = (rot_matrix @ local_pts.T).T + center

    yaw, obb = calculate_pca_yaw_and_obb(world_pts, "vehicle")

    assert abs(obb["size"][0] - true_l) < 0.25, f"Expected length ~4.0, got {obb['size'][0]}"
    assert abs(obb["size"][1] - true_w) < 0.25, f"Expected width ~1.8, got {obb['size'][1]}"
    assert abs(obb["size"][2] - true_h) < 0.25, f"Expected height ~1.4, got {obb['size'][2]}"

    aabb_w = np.ptp(world_pts[:, 0])
    aabb_l = np.ptp(world_pts[:, 1])
    assert obb["size"][1] < aabb_w - 0.8, "OBB width should be substantially tighter than AABB width"


def test_3d_spatial_clustering_separates_adjacent_objects():
    """Verify 3D clustering separates two vehicles that are horizontally or spatially distinct."""
    rng = np.random.RandomState(10)
    pts1 = rng.uniform([-1.8, -0.8, -0.4], [1.8, 0.8, 0.4], size=(80, 3)) + np.array([10.0, 0.0, 0.0])
    pts2 = rng.uniform([-1.8, -0.8, -0.4], [1.8, 0.8, 0.4], size=(80, 3)) + np.array([10.0, 6.0, 0.0])

    all_pts = np.vstack([pts1, pts2])
    labels = np.full(len(all_pts), 8, dtype=np.int32)

    instances = cluster_points_to_instances_3d(all_pts, labels, dataset="rellis")
    assert len(instances) == 2, f"Expected 2 separate vehicle instances, got {len(instances)}"
    assert all(inst.object_type == "vehicle" for inst in instances)
    assert all(inst.is_dynamic for inst in instances)
    assert all(inst.oriented_bounding_box is not None for inst in instances)


def test_adaptive_grid_rellis_dirt_produces_drivable_cells():
    """Verify RELLIS dirt generates drivable terrain cells rather than unknown cells."""
    rng = np.random.RandomState(99)
    x = rng.uniform(0, 10, size=200)
    y = rng.uniform(0, 10, size=200)
    z = rng.uniform(-1.50, -1.45, size=200)
    pts_array = np.column_stack([x, y, z])
    labels = np.full(200, 1, dtype=np.int32)

    cells = AdaptiveGridService.generate_grid_from_points(
        points=pts_array,
        frame_id="rellis_00000",
        labels=labels,
        dataset="rellis",
    )
    assert len(cells) > 0

    for c in cells:
        assert c.dominant_category == "drivable"
        assert c.dominant_semantic_class == "dirt"
        assert c.traversability_state.value in ("drivable", "caution_irregular")

