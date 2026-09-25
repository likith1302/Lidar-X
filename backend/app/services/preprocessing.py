"""Point cloud preprocessing, validation, and transparent sanitization reporting."""

import numpy as np
from typing import Tuple
from ..models.schemas import PreprocessingConfig, PreprocessingReport, BoundingBox3D
from .lidar_parser import LidarParser


class PointCloudPreprocessor:
    """Preprocesses raw LiDAR point cloud arrays with range gating and NaN rejection."""

    @classmethod
    def process(
        cls,
        points: np.ndarray,
        config: PreprocessingConfig,
    ) -> Tuple[np.ndarray, PreprocessingReport]:
        """Preprocess input point cloud according to configuration parameters.
        
        Args:
            points: Raw numpy array (N, 4) with [x, y, z, intensity].
            config: PreprocessingConfig parameters.
            
        Returns:
            Tuple of (sanitized_points: np.ndarray, report: PreprocessingReport).
        """
        raw_count = int(points.shape[0]) if points is not None else 0
        if raw_count == 0:
            report = PreprocessingReport.model_construct(
                raw_points_count=0,
                valid_points_count=0,
                nan_inf_removed=0,
                out_of_bounds_removed=0,
                applied_config=config,
                bounds=BoundingBox3D.model_construct(
                    min_x=0.0, max_x=0.0,
                    min_y=0.0, max_y=0.0,
                    min_z=0.0, max_z=0.0,
                ),
            )
            return np.empty((0, 4), dtype=np.float32), report

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        min_r2 = float(config.min_range ** 2)
        max_r2 = float(config.max_range ** 2)
        r_sq = x * x + y * y

        if config.remove_nan_inf:
            finite_mask = np.isfinite(r_sq) & np.isfinite(z)
            if points.shape[1] > 3:
                finite_mask = finite_mask & np.isfinite(points[:, 3])
            nan_inf_count = int(raw_count - np.count_nonzero(finite_mask))
            range_mask = (
                finite_mask &
                (r_sq >= min_r2) &
                (r_sq <= max_r2) &
                (z >= config.z_min) &
                (z <= config.z_max)
            )
        else:
            nan_inf_count = 0
            range_mask = (
                (r_sq >= min_r2) &
                (r_sq <= max_r2) &
                (z >= config.z_min) &
                (z <= config.z_max)
            )

        valid_count = int(np.count_nonzero(range_mask))
        out_of_bounds_count = int(raw_count - nan_inf_count - valid_count)
        sanitized_pts = points[range_mask]

        if valid_count == 0:
            bounds = BoundingBox3D.model_construct(
                min_x=0.0, max_x=0.0,
                min_y=0.0, max_y=0.0,
                min_z=0.0, max_z=0.0,
            )
        else:
            min_v = sanitized_pts[:, :3].min(axis=0)
            max_v = sanitized_pts[:, :3].max(axis=0)
            bounds = BoundingBox3D.model_construct(
                min_x=float(min_v[0]), max_x=float(max_v[0]),
                min_y=float(min_v[1]), max_y=float(max_v[1]),
                min_z=float(min_v[2]), max_z=float(max_v[2]),
            )

        report = PreprocessingReport.model_construct(
            raw_points_count=raw_count,
            valid_points_count=valid_count,
            nan_inf_removed=nan_inf_count,
            out_of_bounds_removed=out_of_bounds_count,
            applied_config=config,
            bounds=bounds,
        )

        return sanitized_pts, report

