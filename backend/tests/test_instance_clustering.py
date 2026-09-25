"""Unit tests for geometric instance clustering."""

import numpy as np
import pytest
from app.services.instance_clustering import InstanceClusteringService
from app.models.object_schemas import ExtentCategory, ObservationStatus, ProjectCategory


def test_detect_separated_objects():
    """Test clustering distinct point clusters into separate object instances."""
    # Reset mapping cache to ensure the fixed YAML is loaded (car is_dynamic=True)
    from app.services.semantic_mapping import SemanticMappingService
    SemanticMappingService._mapping_cache = None

    # Cluster 1: Car 1 at x=[5, 8], y=[2, 4], z=[-1.5, -0.5] (20 points, class 10)
    car1_x = np.linspace(5.0, 8.0, 20)
    car1_y = np.linspace(2.0, 4.0, 20)
    car1_pts = np.column_stack([car1_x, car1_y, np.full(20, -1.0), np.full(20, 0.5)])

    # Cluster 2: Car 2 at x=[15, 18], y=[-4, -2], z=[-1.5, -0.5] (20 points, class 10)
    car2_x = np.linspace(15.0, 18.0, 20)
    car2_y = np.linspace(-4.0, -2.0, 20)
    car2_pts = np.column_stack([car2_x, car2_y, np.full(20, -1.0), np.full(20, 0.5)])

    # Cluster 3: Person at x=[6, 6.4], y=[-2, -1.8], z=[-1.5, 0.2] (10 points, class 30)
    person_x = np.linspace(6.0, 6.4, 10)
    person_y = np.linspace(-2.0, -1.8, 10)
    person_z = np.linspace(-1.5, 0.2, 10)
    person_pts = np.column_stack([person_x, person_y, person_z, np.full(10, 0.3)])

    # Road background (50 points, class 40 - not clusterable)
    road_pts = np.zeros((50, 4), dtype=np.float32)
    road_pts[:, 0] = np.linspace(-5, 20, 50)
    road_pts[:, 2] = -1.73

    all_pts = np.vstack([car1_pts, car2_pts, person_pts, road_pts]).astype(np.float32)
    all_labels = np.array(
        [10] * 20 + [10] * 20 + [30] * 10 + [40] * 50,
        dtype=np.uint16,
    )

    res = InstanceClusteringService.detect_objects(
        frame_id="frame_cluster_test",
        points=all_pts,
        semantic_classes=all_labels,
        min_points_per_cluster=5,
    )

    assert res.frame_id == "frame_cluster_test"
    assert res.total_instances == 3  # 2 cars + 1 person
    # With the YAML fix, car and person are is_dynamic=True
    assert res.dynamic_instances_count == 3
    assert res.static_instances_count == 0

    car_instances = [inst for inst in res.instances if inst.object_type == "car"]
    assert len(car_instances) == 2

    person_instances = [inst for inst in res.instances if inst.object_type == "person"]
    assert len(person_instances) == 1

    p_inst = person_instances[0]
    assert p_inst.extent_category == ExtentCategory.COMPACT_ACTOR
    assert p_inst.semantic_category == ProjectCategory.DYNAMIC_OBJECT
    assert p_inst.is_dynamic is True  # Corrected: person is a dynamic class


def test_static_obstacle_clustering():
    """Test clustering a pole (slender vertical obstacle)."""
    # Vertical pole at x=4.0, y=3.0, z from -1.7 to 1.5 (15 points, class 80)
    z_vals = np.linspace(-1.7, 1.5, 15)
    pole_pts = np.column_stack([
        np.full(15, 4.0),
        np.full(15, 3.0),
        z_vals,
        np.full(15, 0.8),
    ]).astype(np.float32)

    labels = np.full(15, 80, dtype=np.uint16)

    res = InstanceClusteringService.detect_objects(
        frame_id="frame_pole_test",
        points=pole_pts,
        semantic_classes=labels,
        min_points_per_cluster=4,
    )

    assert res.total_instances == 1
    inst = res.instances[0]
    assert inst.object_type == "pole"
    assert inst.is_dynamic is False
    assert inst.extent_category == ExtentCategory.SLENDER_VERTICAL
    assert inst.semantic_category == ProjectCategory.STATIC_OBSTACLE
