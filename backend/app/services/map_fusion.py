"""Map Fusion Service: Manages local vehicle-centric and persistent global multi-frame map updates."""

import threading
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from datetime import datetime, timezone

from ..models.map_schemas import (
    AdaptiveGridCell,
    MapMetadata,
    MapMode,
    ResolutionLevel,
    TraversabilityState,
    GridPolicyConfig,
    MapExportResponse,
)
from .adaptive_grid import AdaptiveGridService
from .resolution_policy import ResolutionPolicyService


class StoredMapInstance:
    """In-memory representation of an active 2.5D map."""

    def __init__(self, map_id: str, mode: MapMode = MapMode.LOCAL_ONLY):
        self.map_id = map_id
        self.mode = mode
        self.cells: Dict[str, AdaptiveGridCell] = {}
        self.frame_count: int = 0
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.updated_at: str = self.created_at
        self.policy_snapshot: GridPolicyConfig = ResolutionPolicyService.get_policy()


class MapFusionService:
    """Coordinates frame updates, pose transformations, and spatial fusion."""

    _lock = threading.Lock()
    _maps: Dict[str, StoredMapInstance] = {}

    @classmethod
    def transform_points_to_world(
        cls,
        points: np.ndarray,
        ego_pose: Union[List[List[float]], List[float]],
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Apply rigid body transformation to points.
        
        Args:
            points: (N, 3) or (N, 4) local coordinates
            ego_pose: 4x4 matrix or 3x4 matrix or [x, y, yaw_rad] or [x, y, z, yaw_rad]
            
        Returns:
            (world_points, (sensor_x, sensor_y, sensor_z))
        """
        xyz = points[:, :3].copy()
        
        # 1. If 4x4 or 3x4 transformation matrix
        if isinstance(ego_pose, list) and len(ego_pose) in (3, 4) and isinstance(ego_pose[0], list):
            mat = np.array(ego_pose, dtype=np.float64)
            R = mat[:3, :3]
            T = mat[:3, 3] if mat.shape[1] >= 4 else np.zeros(3)
            world_xyz = (R @ xyz.T).T + T
            sensor_pos = (float(T[0]), float(T[1]), float(T[2]))
            
            # Reattach intensity if present
            if points.shape[1] > 3:
                res = np.column_stack([world_xyz, points[:, 3:]])
                return res, sensor_pos
            return world_xyz, sensor_pos

        # 2. If vector [x, y, yaw] or [x, y, z, yaw]
        pose_arr = np.array(ego_pose, dtype=np.float64).flatten()
        if len(pose_arr) >= 3:
            tx = float(pose_arr[0])
            ty = float(pose_arr[1])
            tz = float(pose_arr[2]) if len(pose_arr) >= 4 else 0.0
            yaw = float(pose_arr[-1])  # Yaw angle in radians
            
            c = np.cos(yaw)
            s = np.sin(yaw)
            
            x_w = xyz[:, 0] * c - xyz[:, 1] * s + tx
            y_w = xyz[:, 0] * s + xyz[:, 1] * c + ty
            z_w = xyz[:, 2] + tz
            
            world_xyz = np.column_stack([x_w, y_w, z_w])
            sensor_pos = (tx, ty, tz)
            
            if points.shape[1] > 3:
                res = np.column_stack([world_xyz, points[:, 3:]])
                return res, sensor_pos
            return world_xyz, sensor_pos

        # Fallback: identity
        return points, (0.0, 0.0, 0.0)

    @classmethod
    def update_map(
        cls,
        map_id: str,
        frame_id: str,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        instance_ids: Optional[np.ndarray] = None,
        dynamic_track_positions: Optional[List[List[float]]] = None,
        ego_pose: Optional[Union[List[List[float]], List[float]]] = None,
        override_policy: Optional[Dict[str, Any]] = None,
        dataset: Optional[str] = None,
    ) -> Tuple[MapMetadata, List[AdaptiveGridCell]]:
        """
        Process a frame update and fuse into local or global map.
        """
        with cls._lock:
            mode = MapMode.GLOBAL_FUSION if ego_pose is not None else MapMode.LOCAL_ONLY

            # Initialize or retrieve map
            if map_id not in cls._maps:
                cls._maps[map_id] = StoredMapInstance(map_id=map_id, mode=mode)
            
            map_inst = cls._maps[map_id]
            map_inst.mode = mode
            map_inst.frame_count += 1
            map_inst.updated_at = datetime.now(timezone.utc).isoformat()

            # Merge override policy if provided
            active_policy = ResolutionPolicyService.get_policy()
            if override_policy:
                pol_dict = active_policy.model_dump()
                pol_dict.update(override_policy)
                active_policy = GridPolicyConfig(**pol_dict)
            # Normalise to ensure 2:1 hierarchy for parent/child key alignment
            active_policy = ResolutionPolicyService.normalise_policy(active_policy)
            map_inst.policy_snapshot = active_policy

            # Transform points if in global fusion mode
            if mode == MapMode.GLOBAL_FUSION and ego_pose is not None:
                trans_pts, sensor_origin = cls.transform_points_to_world(points, ego_pose)
            else:
                trans_pts = points
                sensor_origin = (0.0, 0.0, 0.0)

            # Generate new frame cells
            new_cells = AdaptiveGridService.generate_grid_from_points(
                points=trans_pts,
                frame_id=frame_id,
                labels=labels,
                instance_ids=instance_ids,
                dynamic_track_positions=dynamic_track_positions,
                policy=active_policy,
                sensor_origin=sensor_origin,
                dataset=dataset,
            )

            if mode == MapMode.LOCAL_ONLY:
                # Replace cells completely for single-frame vehicle-centric map
                map_inst.cells = {c.cell_key: c for c in new_cells}
            else:
                # Multi-frame global fusion
                for cell in new_cells:
                    key = cell.cell_key
                    if key not in map_inst.cells:
                        map_inst.cells[key] = cell
                    else:
                        existing = map_inst.cells[key]
                        total_pts = existing.point_count + cell.point_count
                        
                        # Recursive Kalman / Exponential update for elevation to prevent ghost trails
                        if not cell.is_dynamic_obstacle:
                            # Dynamic learning rate: higher if newly observed with high point density
                            alpha = min(0.40, max(0.10, cell.point_count / max(1, existing.point_count + cell.point_count)))
                            existing.elevation_mean = round((1.0 - alpha) * existing.elevation_mean + alpha * cell.elevation_mean, 3)
                            existing.elevation_min = round((1.0 - alpha) * existing.elevation_min + alpha * cell.elevation_min, 3)
                            existing.elevation_max = round((1.0 - alpha) * existing.elevation_max + alpha * cell.elevation_max, 3)
                            existing.elevation_variation = round(max(0.0, existing.elevation_max - existing.elevation_min), 3)

                        # Merge semantic histogram
                        for cname, cnt in cell.semantic_histogram.items():
                            existing.semantic_histogram[cname] = existing.semantic_histogram.get(cname, 0) + cnt
                        
                        # Recompute dominant class
                        dom_class = max(existing.semantic_histogram.keys(), key=lambda k: existing.semantic_histogram[k])
                        existing.dominant_semantic_class = dom_class
                        
                        # Update flags & timestamps
                        existing.point_count = total_pts
                        existing.last_frame_id = frame_id
                        existing.last_timestamp = cell.last_timestamp
                        existing.traversability_state = cell.traversability_state
                        existing.observation_state = cell.observation_state
                        existing.is_dynamic_obstacle = cell.is_dynamic_obstacle
                        existing.is_static_obstacle = cell.is_static_obstacle

                # Bounded memory: cap global map size to prevent unbounded memory growth
                MAX_GLOBAL_MAP_CELLS = 25000
                if len(map_inst.cells) > MAX_GLOBAL_MAP_CELLS:
                    excess = len(map_inst.cells) - MAX_GLOBAL_MAP_CELLS
                    sorted_keys = sorted(map_inst.cells.keys(), key=lambda k: map_inst.cells[k].last_timestamp or "")
                    for k in sorted_keys[:excess]:
                        del map_inst.cells[k]

            all_cells = list(map_inst.cells.values())
            metadata = AdaptiveGridService.compute_map_metadata(
                map_id=map_id,
                mode=mode,
                cells=all_cells,
                frame_count=map_inst.frame_count,
                created_at=map_inst.created_at,
            )

            return metadata, all_cells

    @classmethod
    def get_map(cls, map_id: str) -> Optional[Tuple[MapMetadata, List[AdaptiveGridCell], GridPolicyConfig]]:
        """Retrieve full map instance."""
        with cls._lock:
            if map_id not in cls._maps:
                return None
            inst = cls._maps[map_id]
            cells = list(inst.cells.values())
            meta = AdaptiveGridService.compute_map_metadata(
                map_id=map_id,
                mode=inst.mode,
                cells=cells,
                frame_count=inst.frame_count,
                created_at=inst.created_at,
            )
            return meta, cells, inst.policy_snapshot

    @classmethod
    def get_cells(
        cls,
        map_id: str,
        level: Optional[str] = None,
        min_x: Optional[float] = None,
        max_x: Optional[float] = None,
        min_y: Optional[float] = None,
        max_y: Optional[float] = None,
        limit: int = 5000,
    ) -> Optional[List[AdaptiveGridCell]]:
        """Filter sparse cells by level and spatial bounding box."""
        with cls._lock:
            if map_id not in cls._maps:
                return None
            cells = list(cls._maps[map_id].cells.values())

        # Filter by level
        if level and level != "all":
            cells = [c for c in cells if c.level.value == level or c.level == level]

        # Spatial bounding box filter
        if min_x is not None:
            cells = [c for c in cells if c.bounds[1] >= min_x]
        if max_x is not None:
            cells = [c for c in cells if c.bounds[0] <= max_x]
        if min_y is not None:
            cells = [c for c in cells if c.bounds[3] >= min_y]
        if max_y is not None:
            cells = [c for c in cells if c.bounds[2] <= max_y]

        return cells[:limit]

    @classmethod
    def get_cell_by_key(cls, map_id: str, cell_key: str) -> Optional[AdaptiveGridCell]:
        """Query detailed properties of a single hierarchical cell."""
        with cls._lock:
            if map_id not in cls._maps:
                return None
            return cls._maps[map_id].cells.get(cell_key)

    @classmethod
    def reset_map(cls, map_id: str) -> bool:
        """Clear and reset a map."""
        with cls._lock:
            if map_id in cls._maps:
                cls._maps[map_id].cells.clear()
                cls._maps[map_id].frame_count = 0
                return True
            return False

    @classmethod
    def clone_or_alias_map(cls, source_map_id: str, target_map_id: str, mode: MapMode = MapMode.LOCAL_ONLY) -> Optional[StoredMapInstance]:
        """Alias or clone an existing map's cells without re-running grid synthesis."""
        with cls._lock:
            src = cls._maps.get(source_map_id)
            if src is None:
                return None
            target_inst = StoredMapInstance(target_map_id, mode=mode)
            target_inst.cells = dict(src.cells)
            target_inst.frame_count = src.frame_count
            target_inst.policy_snapshot = src.policy_snapshot
            target_inst.updated_at = src.updated_at
            cls._maps[target_map_id] = target_inst
            return target_inst

