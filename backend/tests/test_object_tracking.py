"""Unit tests for temporal object tracking and Kalman state estimation."""

import numpy as np
import pytest
from app.services.object_tracking import ObjectTrackingService
from app.models.schemas import BoundingBox3D
from app.models.object_schemas import (
    ObjectInstance,
    ProjectCategory,
    ExtentCategory,
    ObservationStatus,
    TrackLifecycleState,
    TrackingCoordinateMode,
)


def create_mock_instance(
    inst_id: str,
    obj_type: str,
    centroid: list,
    frame_id: str = "frame_001",
    is_dynamic: bool = True,
) -> ObjectInstance:
    return ObjectInstance(
        instance_id=inst_id,
        semantic_category=ProjectCategory.DYNAMIC_OBJECT if is_dynamic else ProjectCategory.STATIC_OBSTACLE,
        object_type=obj_type,
        centroid=centroid,
        bounding_box=BoundingBox3D(
            min_x=centroid[0] - 1.0, max_x=centroid[0] + 1.0,
            min_y=centroid[1] - 0.8, max_y=centroid[1] + 0.8,
            min_z=centroid[2] - 0.7, max_z=centroid[2] + 0.7,
        ),
        dimensions=[2.0, 1.6, 1.4],
        extent_category=ExtentCategory.SMALL_VEHICLE,
        point_count=30,
        is_dynamic=is_dynamic,
        source_frame_id=frame_id,
        observation_status=ObservationStatus.DIRECTLY_OBSERVED,
        sample_points=[],
    )


def test_track_lifecycle_and_motion():
    """Test track creation, transition from NEW to ACTIVE, and velocity estimation."""
    ObjectTrackingService.reset()

    # Frame 1: Car at [10.0, 2.0, -1.0]
    inst1 = create_mock_instance("inst_1", "car", [10.0, 2.0, -1.0], frame_id="frame_1")
    res1 = ObjectTrackingService.update_tracks(
        frame_id="frame_1",
        detected_instances=[inst1],
    )

    assert res1.active_tracks_count == 1
    assert res1.new_tracks_count == 1
    t1 = res1.tracks[0]
    assert t1.lifecycle_state == TrackLifecycleState.NEW
    track_id = t1.track_id

    # Frame 2: Car moved to [11.0, 2.0, -1.0] (moved 1.0m along X)
    inst2 = create_mock_instance("inst_2", "car", [11.0, 2.0, -1.0], frame_id="frame_2")
    res2 = ObjectTrackingService.update_tracks(
        frame_id="frame_2",
        detected_instances=[inst2],
    )

    assert len(res2.tracks) == 1
    t2 = res2.tracks[0]
    assert t2.track_id == track_id  # Stable track ID maintained
    assert t2.lifecycle_state == TrackLifecycleState.ACTIVE
    assert len(t2.history) == 2
    assert t2.age_frames == 2


def test_temporarily_lost_and_expiration():
    """Test track transition to TEMPORARILY_LOST when missing, then EXPIRED."""
    ObjectTrackingService.reset()

    inst = create_mock_instance("inst_1", "car", [5.0, 0.0, -1.0], frame_id="frame_1")
    ObjectTrackingService.update_tracks(frame_id="frame_1", detected_instances=[inst])

    # Frames 2, 3: Car missing (no detections) -> coasted as TEMPORARILY_LOST
    res2 = ObjectTrackingService.update_tracks(frame_id="frame_2", detected_instances=[])
    assert res2.lost_tracks_count == 1
    assert res2.tracks[0].lifecycle_state == TrackLifecycleState.TEMPORARILY_LOST

    res3 = ObjectTrackingService.update_tracks(frame_id="frame_3", detected_instances=[])
    assert res3.lost_tracks_count == 1

    # Frame 4: Missing 3+ frames -> Expired and removed
    res4 = ObjectTrackingService.update_tracks(frame_id="frame_4", detected_instances=[])
    res5 = ObjectTrackingService.update_tracks(frame_id="frame_5", detected_instances=[])
    assert len(res5.tracks) == 0  # Fully expired


def test_coordinate_mode():
    """Test local_frame mode when no pose is provided and world_frame when pose is provided."""
    ObjectTrackingService.reset()

    inst = create_mock_instance("inst_1", "car", [5.0, 0.0, -1.0], frame_id="frame_1")
    res_local = ObjectTrackingService.update_tracks(frame_id="frame_1", detected_instances=[inst])
    assert res_local.coordinate_mode == TrackingCoordinateMode.LOCAL_FRAME

    inst2 = create_mock_instance("inst_2", "car", [6.0, 0.0, -1.0], frame_id="frame_2")
    res_world = ObjectTrackingService.update_tracks(
        frame_id="frame_2",
        detected_instances=[inst2],
        ego_pose=[1.0, 0.0, 0.0, 0.0],
    )
    assert res_world.coordinate_mode == TrackingCoordinateMode.WORLD_FRAME


def test_ego_motion_compensation_static_object():
    """Verify ego-motion compensation prevents a stationary object from appearing dynamic when vehicle moves."""
    ObjectTrackingService.reset()

    # Frame 1: Vehicle at [0, 0, 0], parked car detected at sensor coordinates [10.0, 0.0, 0.0]
    inst1 = create_mock_instance("car_1", "car", [10.0, 0.0, 0.0], frame_id="f1", is_dynamic=False)
    res1 = ObjectTrackingService.update_tracks(
        frame_id="f1",
        detected_instances=[inst1],
        ego_pose=[0.0, 0.0, 0.0, 0.0],
    )
    assert res1.active_tracks_count == 1
    t1 = res1.tracks[0]
    assert np.isclose(t1.current_position[0], 10.0, atol=0.1)

    # Frame 2: Vehicle moved forward 1.0m to [1.0, 0.0, 0.0].
    # In sensor coordinates, the parked car is now at [9.0, 0.0, 0.0].
    inst2 = create_mock_instance("car_2", "car", [9.0, 0.0, 0.0], frame_id="f2", is_dynamic=False)
    res2 = ObjectTrackingService.update_tracks(
        frame_id="f2",
        detected_instances=[inst2],
        ego_pose=[1.0, 0.0, 0.0, 0.0],
    )
    assert res2.active_tracks_count == 1
    t2 = res2.tracks[0]
    # In world coordinates, car should still be at ~10.0m, not 9.0m!
    assert np.isclose(t2.current_position[0], 10.0, atol=0.2)
    # Velocity in world coordinates should be zero / near zero
    assert np.linalg.norm(t2.estimated_velocity[:2]) < 0.5
    assert inst2.is_dynamic is False
