"""Multi-Object Tracking (MOT) pipeline with Kalman state estimation and lifecycle management."""

from datetime import datetime, timezone
import math
from typing import List, Dict, Optional, Tuple, Any, Set
import numpy as np

from ..models.schemas import BoundingBox3D
from ..models.object_schemas import (
    ObjectInstance,
    TrackedObject,
    TrackedPosition,
    TrackLifecycleState,
    TrackingCoordinateMode,
    TrackingUpdateResponse,
    ProjectCategory,
)

TRACKABLE_TYPES: Set[str] = {
    "car",
    "truck",
    "bus",
    "person",
    "pedestrian",
    "bicyclist",
    "motorcyclist",
    "motorcycle",
    "bicycle",
    "other-vehicle",
    "moving-car",
    "moving-truck",
    "moving-bus",
    "moving-person",
    "moving-bicyclist",
    "moving-motorcyclist",
    "moving-on-rails",
    "moving-other-vehicle",
}


def is_trackable_instance(inst: ObjectInstance) -> bool:
    """Check whether an instance is an actor or vehicle eligible for temporal tracking."""
    if inst.is_dynamic or inst.semantic_category == ProjectCategory.DYNAMIC_OBJECT:
        return True
    c_lower = inst.object_type.lower()
    return c_lower in TRACKABLE_TYPES or c_lower.startswith("moving-")


class KalmanBoxTracker:
    """Constant-velocity 3D Kalman Filter for a single tracked spatial object."""

    def __init__(self, initial_pos: np.ndarray, dt: float = 0.1):
        """State: [x, y, z, vx, vy, vz]. Measurement: [x, y, z]."""
        self.dt = dt
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.x[:3, 0] = initial_pos[:3]

        # State transition matrix F
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement matrix H
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Covariance matrices
        self.P = np.eye(6, dtype=np.float64) * 10.0
        self.P[3:, 3:] *= 50.0  # initial velocity uncertainty

        self.Q = np.eye(6, dtype=np.float64) * 0.1
        self.Q[3:, 3:] *= 0.5

        self.R = np.eye(3, dtype=np.float64) * 0.25

    def predict(self) -> np.ndarray:
        """Predict state forward in time."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:3, 0]

    def update(self, measurement: np.ndarray):
        """Update state with observed 3D position."""
        z = measurement[:3].reshape((3, 1))
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        I = np.eye(6, dtype=np.float64)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    @property
    def position(self) -> np.ndarray:
        return self.x[:3, 0]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:, 0]


def transform_centroid_to_world(centroid: List[float], ego_pose: Any) -> List[float]:
    """Transform 3D centroid from vehicle sensor coordinates to world coordinates using ego pose."""
    if ego_pose is None:
        return centroid
    xyz = np.array(centroid[:3], dtype=np.float64)
    # Check if 4x4 or 3x4 matrix
    if isinstance(ego_pose, (list, tuple)) and len(ego_pose) in (3, 4) and isinstance(ego_pose[0], (list, tuple)):
        mat = np.array(ego_pose, dtype=np.float64)
        R = mat[:3, :3]
        T = mat[:3, 3] if mat.shape[1] >= 4 else np.zeros(3)
        res = R @ xyz + T
        return [float(res[0]), float(res[1]), float(res[2])]
    # Vector: [x, y, yaw] or [x, y, z, yaw]
    pose_arr = np.array(ego_pose, dtype=np.float64).flatten()
    if len(pose_arr) >= 3:
        tx = float(pose_arr[0])
        ty = float(pose_arr[1])
        tz = float(pose_arr[2]) if len(pose_arr) >= 4 else 0.0
        yaw = float(pose_arr[-1])
        c = math.cos(yaw)
        s = math.sin(yaw)
        x_w = xyz[0] * c - xyz[1] * s + tx
        y_w = xyz[0] * s + xyz[1] * c + ty
        z_w = xyz[2] + tz
        return [float(x_w), float(y_w), float(z_w)]
    return centroid


class TrackState:
    """Internal state representation for an active track."""

    def __init__(
        self,
        track_id: str,
        instance: ObjectInstance,
        timestamp: str,
        coord_mode: TrackingCoordinateMode,
        world_centroid: Optional[List[float]] = None,
    ):
        self.track_id = track_id
        self.object_type = instance.object_type
        self.semantic_category = instance.semantic_category
        self.bounding_box = instance.bounding_box
        self.associated_instance_id = instance.instance_id

        pos = np.array(world_centroid if world_centroid is not None else instance.centroid, dtype=np.float64)
        self.kf = KalmanBoxTracker(pos)
        self.history: List[TrackedPosition] = [
            TrackedPosition(
                frame_id=instance.source_frame_id,
                timestamp=timestamp,
                position=[round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)],
            )
        ]
        self.lifecycle_state = TrackLifecycleState.NEW
        self.age_frames = 1
        self.frames_since_seen = 0
        self.coordinate_mode = coord_mode
        self.is_explicit_moving = instance.object_type.lower().startswith("moving-") or instance.is_dynamic

    def predict(self):
        self.kf.predict()
        self.age_frames += 1
        self.frames_since_seen += 1

    def update_matched(self, instance: ObjectInstance, timestamp: str, world_centroid: Optional[List[float]] = None):
        pos = np.array(world_centroid if world_centroid is not None else instance.centroid, dtype=np.float64)
        self.kf.update(pos)
        self.bounding_box = instance.bounding_box
        self.associated_instance_id = instance.instance_id
        self.frames_since_seen = 0
        self.lifecycle_state = TrackLifecycleState.ACTIVE

        self.history.append(
            TrackedPosition(
                frame_id=instance.source_frame_id,
                timestamp=timestamp,
                position=[round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)],
            )
        )
        if len(self.history) > 30:
            self.history.pop(0)

    @property
    def is_dynamic_motion(self) -> bool:
        """Determine if track is actively moving or stationary based on estimated velocity."""
        if self.is_explicit_moving:
            return True
        if self.age_frames >= 2 and self.frames_since_seen == 0:
            vel = self.kf.velocity
            speed = float(np.linalg.norm(vel[:2]))  # horizontal ground speed
            return speed > 0.4
        return False

    def to_schema(self) -> TrackedObject:
        pos = self.kf.position
        vel = self.kf.velocity
        speed = float(np.linalg.norm(vel[:2]))
        # Suppress jitter velocity for stationary objects
        effective_vel = vel if (speed > 0.3 or self.is_explicit_moving) else np.zeros(3, dtype=np.float64)

        return TrackedObject(
            track_id=self.track_id,
            object_type=self.object_type,
            semantic_category=self.semantic_category,
            current_position=[round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)],
            estimated_velocity=[round(float(effective_vel[0]), 3), round(float(effective_vel[1]), 3), round(float(effective_vel[2]), 3)],
            history=self.history,
            lifecycle_state=self.lifecycle_state,
            age_frames=self.age_frames,
            frames_since_seen=self.frames_since_seen,
            coordinate_mode=self.coordinate_mode,
            bounding_box=self.bounding_box,
            associated_instance_id=self.associated_instance_id,
        )


class ObjectTrackingService:
    """Temporal tracker maintaining dynamic object tracks across sequential LiDAR frames."""

    _active_tracks: Dict[str, TrackState] = {}
    _track_id_counter: int = 0
    _max_missed_frames: int = 3

    @classmethod
    def reset(cls):
        """Reset all active tracks (e.g. at sequence start)."""
        cls._active_tracks.clear()
        cls._track_id_counter = 0

    @classmethod
    def update_tracks(
        cls,
        frame_id: str,
        detected_instances: List[ObjectInstance],
        timestamp: Optional[str] = None,
        ego_pose: Optional[List[float]] = None,
        max_distance_m: float = 2.5,
    ) -> TrackingUpdateResponse:
        """Associate detections with existing tracks and update temporal state."""
        now_iso = timestamp or datetime.now(timezone.utc).isoformat()
        coord_mode = TrackingCoordinateMode.WORLD_FRAME if ego_pose is not None else TrackingCoordinateMode.LOCAL_FRAME

        # Select trackable actor instances
        trackable_dets = [inst for inst in detected_instances if is_trackable_instance(inst)]
        det_positions = [
            transform_centroid_to_world(inst.centroid, ego_pose) for inst in trackable_dets
        ]

        # 1. Predict all existing tracks
        for track in cls._active_tracks.values():
            track.predict()
            if track.frames_since_seen > 0:
                track.lifecycle_state = TrackLifecycleState.TEMPORARILY_LOST

        # 2. Match detections with existing tracks using Greedy association
        unmatched_dets = set(range(len(trackable_dets)))
        unmatched_tracks = set(cls._active_tracks.keys())
        matched_pairs: List[Tuple[str, int]] = []

        if len(cls._active_tracks) > 0 and len(trackable_dets) > 0:
            track_ids = list(cls._active_tracks.keys())
            cost_matrix = np.full((len(track_ids), len(trackable_dets)), np.inf)

            for i, tid in enumerate(track_ids):
                t_state = cls._active_tracks[tid]
                pred_pos = t_state.kf.position

                for j, det in enumerate(trackable_dets):
                    # Check semantic compatibility
                    if det.object_type != t_state.object_type and not (
                        "moving-" in det.object_type or "moving-" in t_state.object_type
                    ):
                        continue

                    det_pos = np.array(det_positions[j], dtype=np.float64)
                    dist = float(np.linalg.norm(pred_pos - det_pos))

                    if dist <= max_distance_m:
                        cost_matrix[i, j] = dist

            while True:
                min_val = np.min(cost_matrix)
                if min_val == np.inf or np.isnan(min_val):
                    break
                min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                t_idx, d_idx = int(min_idx[0]), int(min_idx[1])
                tid = track_ids[t_idx]

                matched_pairs.append((tid, d_idx))
                unmatched_tracks.discard(tid)
                unmatched_dets.discard(d_idx)

                cost_matrix[t_idx, :] = np.inf
                cost_matrix[:, d_idx] = np.inf

        # 3. Update matched tracks and update dynamic motion flag on detected instances
        for tid, d_idx in matched_pairs:
            det = trackable_dets[d_idx]
            track = cls._active_tracks[tid]
            track.update_matched(det, now_iso, world_centroid=det_positions[d_idx])
            det.is_dynamic = track.is_dynamic_motion

        # 4. Create new tracks for unmatched detections
        new_track_count = 0
        for d_idx in unmatched_dets:
            det = trackable_dets[d_idx]
            cls._track_id_counter += 1
            new_tid = f"TRK_{cls._track_id_counter:03d}"
            new_track = TrackState(
                track_id=new_tid,
                instance=det,
                timestamp=now_iso,
                coord_mode=coord_mode,
                world_centroid=det_positions[d_idx],
            )
            cls._active_tracks[new_tid] = new_track
            new_track_count += 1
            det.is_dynamic = new_track.is_dynamic_motion

        # 5. Remove expired tracks
        expired_track_ids = [
            tid for tid, t in cls._active_tracks.items()
            if t.frames_since_seen > cls._max_missed_frames
        ]
        for tid in expired_track_ids:
            del cls._active_tracks[tid]

        # 6. Build response
        active_list: List[TrackedObject] = [t.to_schema() for t in cls._active_tracks.values()]
        active_cnt = sum(1 for t in active_list if t.lifecycle_state in [TrackLifecycleState.ACTIVE, TrackLifecycleState.NEW])
        lost_cnt = sum(1 for t in active_list if t.lifecycle_state == TrackLifecycleState.TEMPORARILY_LOST)

        return TrackingUpdateResponse(
            frame_id=frame_id,
            coordinate_mode=coord_mode,
            active_tracks_count=active_cnt,
            new_tracks_count=new_track_count,
            lost_tracks_count=lost_cnt,
            tracks=active_list,
            timestamp=now_iso,
        )

    @classmethod
    def get_all_tracks(cls) -> List[TrackedObject]:
        """Return all currently active tracks."""
        return [t.to_schema() for t in cls._active_tracks.values()]

    @classmethod
    def get_track(cls, track_id: str) -> Optional[TrackedObject]:
        """Return a specific track by its ID."""
        track = cls._active_tracks.get(track_id)
        return track.to_schema() if track else None


class ObjectTracker:
    """Instance-based Kalman Multi-Object Tracker for session isolation."""

    def __init__(self, max_association_distance_m: float = 2.5, max_missed_frames: int = 3):
        self.max_distance_m = max_association_distance_m
        self.max_missed_frames = max_missed_frames
        self.active_tracks: Dict[str, TrackState] = {}
        self.track_id_counter: int = 0

    def reset(self):
        """Reset all active tracks."""
        self.active_tracks.clear()
        self.track_id_counter = 0

    def get_active_tracks(self) -> List[TrackedObject]:
        """Return list of active tracked objects as schemas."""
        return [t.to_schema() for t in self.active_tracks.values()]

    def update(
        self,
        detected_instances: List[ObjectInstance],
        timestamp: Optional[str] = None,
        ego_pose: Optional[List[float]] = None,
    ) -> List[TrackedObject]:
        """Associate detections and update tracks, returning updated track list."""
        now_iso = timestamp or datetime.now(timezone.utc).isoformat()
        coord_mode = TrackingCoordinateMode.WORLD_FRAME if ego_pose is not None else TrackingCoordinateMode.LOCAL_FRAME

        trackable_dets = [inst for inst in detected_instances if is_trackable_instance(inst)]
        det_positions = [
            transform_centroid_to_world(inst.centroid, ego_pose) for inst in trackable_dets
        ]

        # 1. Predict all existing tracks
        for track in self.active_tracks.values():
            track.predict()
            if track.frames_since_seen > 0:
                track.lifecycle_state = TrackLifecycleState.TEMPORARILY_LOST

        # 2. Match detections with existing tracks
        unmatched_dets = set(range(len(trackable_dets)))
        unmatched_tracks = set(self.active_tracks.keys())
        matched_pairs: List[Tuple[str, int]] = []

        if len(self.active_tracks) > 0 and len(trackable_dets) > 0:
            track_ids = list(self.active_tracks.keys())
            cost_matrix = np.full((len(track_ids), len(trackable_dets)), np.inf)

            for i, tid in enumerate(track_ids):
                t_state = self.active_tracks[tid]
                pred_pos = t_state.kf.position

                for j, det in enumerate(trackable_dets):
                    if det.object_type != t_state.object_type and not (
                        "moving-" in det.object_type or "moving-" in t_state.object_type
                    ):
                        continue

                    det_pos = np.array(det_positions[j], dtype=np.float64)
                    dist = float(np.linalg.norm(pred_pos - det_pos))

                    if dist <= self.max_distance_m:
                        cost_matrix[i, j] = dist

            while True:
                min_val = np.min(cost_matrix)
                if min_val == np.inf or np.isnan(min_val):
                    break
                min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                t_idx, d_idx = int(min_idx[0]), int(min_idx[1])
                tid = track_ids[t_idx]

                matched_pairs.append((tid, d_idx))
                unmatched_tracks.discard(tid)
                unmatched_dets.discard(d_idx)

                cost_matrix[t_idx, :] = np.inf
                cost_matrix[:, d_idx] = np.inf

        # 3. Update matched tracks and update dynamic motion flag on detected instances
        for tid, d_idx in matched_pairs:
            det = trackable_dets[d_idx]
            track = self.active_tracks[tid]
            track.update_matched(det, now_iso, world_centroid=det_positions[d_idx])
            det.is_dynamic = track.is_dynamic_motion

        # 4. Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            det = trackable_dets[d_idx]
            self.track_id_counter += 1
            new_tid = f"TRK_{self.track_id_counter:03d}"
            new_track = TrackState(
                track_id=new_tid,
                instance=det,
                timestamp=now_iso,
                coord_mode=coord_mode,
                world_centroid=det_positions[d_idx],
            )
            self.active_tracks[new_tid] = new_track
            det.is_dynamic = new_track.is_dynamic_motion

        # 5. Remove expired tracks
        expired_track_ids = [
            tid for tid, t in self.active_tracks.items()
            if t.frames_since_seen > self.max_missed_frames
        ]
        for tid in expired_track_ids:
            del self.active_tracks[tid]

        return [t.to_schema() for t in self.active_tracks.values()]

