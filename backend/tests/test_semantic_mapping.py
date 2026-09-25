"""Unit tests for SemanticKITTI label parsing and configurable class mapping."""

import numpy as np
import pytest
from app.services.semantic_mapping import SemanticMappingService, SemanticMappingError


def test_parse_kitti_label_bin():
    """Test parsing uint32 .label file with combined semantic and instance IDs."""
    # Create synthetic uint32 array:
    # Point 0: semantic class 10 (car), instance 1 -> (1 << 16) | 10 = 65546
    # Point 1: semantic class 10 (car), instance 1 -> (1 << 16) | 10 = 65546
    # Point 2: semantic class 30 (person), instance 2 -> (2 << 16) | 30 = 131102
    # Point 3: semantic class 40 (road), instance 0 -> 40
    raw_uint32 = np.array([65546, 65546, 131102, 40], dtype=np.uint32)
    raw_bytes = raw_uint32.tobytes()

    sem_classes, inst_ids = SemanticMappingService.parse_kitti_label_bin(raw_bytes)

    assert len(sem_classes) == 4
    assert len(inst_ids) == 4
    assert sem_classes[0] == 10
    assert inst_ids[0] == 1
    assert sem_classes[1] == 10
    assert inst_ids[1] == 1
    assert sem_classes[2] == 30
    assert inst_ids[2] == 2
    assert sem_classes[3] == 40
    assert inst_ids[3] == 0


def test_reject_empty_or_odd_label_bytes():
    """Test rejecting empty or non-multiple-of-4 byte arrays."""
    with pytest.raises(SemanticMappingError):
        SemanticMappingService.parse_kitti_label_bin(b"")

    with pytest.raises(SemanticMappingError):
        SemanticMappingService.parse_kitti_label_bin(b"\x00\x00\x00")  # 3 bytes


def test_class_mapping_categories():
    """Test mapping dataset class IDs to standardized project categories."""
    # Reset cache to ensure the fixed YAML is re-read
    SemanticMappingService._mapping_cache = None

    car_info = SemanticMappingService.get_class_info(10)
    assert car_info["name"] == "car"
    # Vehicle classes are now correctly marked is_dynamic=True per SemanticKITTI ontology
    assert car_info["is_dynamic"] is True
    moving_car_info = SemanticMappingService.get_class_info(252)
    assert moving_car_info["name"] == "moving-car"
    assert moving_car_info["is_dynamic"] is True

    road_info = SemanticMappingService.get_class_info(40)
    assert road_info["name"] == "road"
    assert road_info["category"] == "drivable"
    assert road_info["is_dynamic"] is False

    pole_info = SemanticMappingService.get_class_info(80)
    assert pole_info["name"] == "pole"
    assert pole_info["category"] == "static_obstacle"
    assert pole_info["is_dynamic"] is False

    sidewalk_info = SemanticMappingService.get_class_info(48)
    assert sidewalk_info["name"] == "sidewalk"
    assert sidewalk_info["category"] == "non_drivable"

    veg_info = SemanticMappingService.get_class_info(70)
    assert veg_info["name"] == "vegetation"
    assert veg_info["category"] == "vegetation"

    bldg_info = SemanticMappingService.get_class_info(50)
    assert bldg_info["name"] == "building"
    assert bldg_info["category"] == "infrastructure"


def test_compute_distributions():
    """Test class and category frequency counts."""
    sem_classes = np.array([10, 10, 10, 30, 40, 40, 80], dtype=np.uint16)
    class_dist, cat_dist = SemanticMappingService.compute_distributions(sem_classes)

    assert class_dist["car"] == 3
    assert class_dist["person"] == 1
    assert class_dist["road"] == 2
    assert class_dist["pole"] == 1

    assert cat_dist["dynamic_object"] == 4  # 3 cars + 1 person
    assert cat_dist["drivable"] == 2        # 2 roads
    assert cat_dist["static_obstacle"] == 1 # 1 pole
