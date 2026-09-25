"""Observability and Occlusion Service: Geometric ray and sector line-of-sight analysis."""

import numpy as np
from typing import List, Tuple, Dict, Set
from ..models.map_schemas import CellObservationState


class ObservabilityService:
    """Estimates visibility, direct observation, and occlusion sectors."""

    @classmethod
    def compute_cell_observability(
        cls,
        cell_world_x: float,
        cell_world_y: float,
        point_count: int,
        sensor_x: float = 0.0,
        sensor_y: float = 0.0,
        max_range_m: float = 80.0,
        min_range_m: float = 0.5,
        occlusion_sectors: List[Tuple[float, float, float]] = None,
    ) -> CellObservationState:
        """
        Determine observation state for a 2.5D cell.
        
        Args:
            cell_world_x: Cell center X coordinate in meters
            cell_world_y: Cell center Y coordinate in meters
            point_count: Number of LiDAR points recorded inside the cell
            sensor_x, sensor_y: Position of sensor origin in coordinate frame
            max_range_m: Sensor maximum reliable range
            min_range_m: Sensor blind-spot radius
            occlusion_sectors: List of (min_azimuth_rad, max_azimuth_rad, min_obstacle_dist)
        """
        dx = cell_world_x - sensor_x
        dy = cell_world_y - sensor_y
        dist = np.hypot(dx, dy)

        # 1. Directly observed if LiDAR points are present within range
        if point_count >= 1 and dist <= max_range_m:
            return CellObservationState.OBSERVED

        # 2. Out of sensor range
        if dist > max_range_m or dist < min_range_m:
            return CellObservationState.OUT_OF_RANGE

        # 3. Check for angular occlusion behind detected obstacles
        if occlusion_sectors:
            azimuth = np.arctan2(dy, dx)
            for az_min, az_max, obs_dist in occlusion_sectors:
                # Handle angle wrapping
                if az_min <= azimuth <= az_max and dist > obs_dist + 0.5:
                    return CellObservationState.PARTIALLY_OCCLUDED

        return CellObservationState.UNKNOWN

    @classmethod
    def extract_occlusion_sectors(
        cls,
        obstacle_positions: np.ndarray,
        sensor_x: float = 0.0,
        sensor_y: float = 0.0,
        obstacle_radius: float = 1.0,
    ) -> List[Tuple[float, float, float]]:
        """
        Build angular shadow wedges cast by detected obstacles.
        
        Args:
            obstacle_positions: (N, 2) array of [x, y] obstacle coordinates
            sensor_x, sensor_y: Position of sensor origin
            obstacle_radius: Radius around obstacle center for angular occlusion spread
            
        Returns:
            List of (azimuth_min_rad, azimuth_max_rad, obstacle_distance_m)
        """
        sectors: List[Tuple[float, float, float]] = []
        if obstacle_positions is None or len(obstacle_positions) == 0:
            return sectors

        for pos in obstacle_positions:
            dx = pos[0] - sensor_x
            dy = pos[1] - sensor_y
            dist = float(np.hypot(dx, dy))
            if dist < 0.5:
                continue

            center_angle = float(np.arctan2(dy, dx))
            # Angular spread proportional to obstacle size
            half_angle = float(np.arcsin(min(1.0, obstacle_radius / dist)))
            sectors.append((center_angle - half_angle, center_angle + half_angle, dist))

        return sectors
