"""Geometric Terrain Analysis Service.

Estimates local 2.5D elevation properties, ground planes, slope gradients, surface roughness,
step/curb height discontinuities, and traversability candidate classifications without static mock data.
"""

from datetime import datetime, timezone
import math
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..models.schemas import (
    TerrainCell,
    TerrainAnalysisSummary,
    TerrainAnalysisResponse,
    ProcessStatus,
    ObservationState,
    SlopeCategory,
    RoughnessCategory,
    StepCategory,
    ElevationSummary,
    TerrainInterpretation,
    DrivabilityState,
    ResolutionZone,
    BoundingBox3D,
    PreprocessingConfig,
)
from .preprocessing import PointCloudPreprocessor
from .lidar_parser import LidarParser


class TerrainAnalysisEngine:
    """Rigorous geometric terrain analysis engine operating on 3D point clouds."""

    @classmethod
    def analyze(
        cls,
        frame_id: str,
        points: np.ndarray,
        grid_resolution_m: float = 1.0,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        include_cells: bool = True,
    ) -> TerrainAnalysisResponse:
        """Run end-to-end preprocessing and geometric terrain analysis on a point cloud.
        
        Args:
            frame_id: Identifier for the LiDAR frame.
            points: Raw point array (N, 4).
            grid_resolution_m: Physical cell side length in meters (default 1.0m).
            preprocessing_config: Optional custom preprocessing parameters.
            include_cells: Whether to instantiate full TerrainCell objects (set False for stream summaries).
            
        Returns:
            TerrainAnalysisResponse containing structured grid cells and spatial summary.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # 1. Preprocess points (skip if already sanitized, indicated by preprocessing_config is None)
        if preprocessing_config is not None:
            sanitized_pts, prep_report = PointCloudPreprocessor.process(points, preprocessing_config)
        else:
            sanitized_pts = points
            prep_report = None

        if sanitized_pts.shape[0] == 0:
            return TerrainAnalysisResponse(
                frame_id=frame_id,
                status=ProcessStatus.COMPLETE,
                preprocessing_report=prep_report,
                summary=TerrainAnalysisSummary(
                    total_cells=0,
                    drivable_cells=0,
                    non_drivable_cells=0,
                    caution_cells=0,
                    hazard_cells=0,
                    unknown_cells=0,
                    grid_resolution_m=grid_resolution_m,
                    bounds=BoundingBox3D(min_x=0, max_x=0, min_y=0, max_y=0, min_z=0, max_z=0),
                ),
                cells=[],
                created_at=now_iso,
            )

        # 2. Estimate global reference ground level from lowest height quantile (CZM fallback)
        xyz = sanitized_pts[:, :3]
        num_pts = xyz.shape[0]
        s_z_raw = xyz[:, 2]
        global_ground_z = float(np.percentile(s_z_raw, 5.0))

        # 3. Fast Vectorized Grid Binning
        res = max(0.02, float(grid_resolution_m))
        gx = np.floor(xyz[:, 0] / res).astype(np.int32)
        gy = np.floor(xyz[:, 1] / res).astype(np.int32)
        packed = (gx.astype(np.int64) << 32) | (gy.astype(np.int64) & 0xFFFFFFFF)

        sort_ord = np.argsort(packed)
        s_packed = packed[sort_ord]
        u_packed, u_starts, counts = np.unique(s_packed, return_index=True, return_counts=True)
        num_cells = len(u_packed)

        u_gx = (u_packed >> 32).astype(np.int64).astype(np.int32)
        u_gy = (u_packed & 0xFFFFFFFF).astype(np.uint32).view(np.int32)

        world_xs = (u_gx + 0.5) * res
        world_ys = (u_gy + 0.5) * res
        dist_origins = np.hypot(world_xs, world_ys)

        s_x = xyz[sort_ord, 0]
        s_y = xyz[sort_ord, 1]
        s_z = xyz[sort_ord, 2]

        # 4. Vectorized Reductions via reduceat
        z_max = np.maximum.reduceat(s_z, u_starts)
        z_min = np.minimum.reduceat(s_z, u_starts)
        z_sum = np.add.reduceat(s_z, u_starts)
        z_mean = z_sum / counts
        elev_ranges = z_max - z_min

        z_sq_sum = np.add.reduceat(s_z * s_z, u_starts)
        z_var = np.maximum(0.0, (z_sq_sum / counts) - (z_mean * z_mean))
        roughness_arr = np.sqrt(z_var)

        # 5. Fast Analytical Slope Vectorization via moments
        sum_x = np.add.reduceat(s_x, u_starts)
        sum_y = np.add.reduceat(s_y, u_starts)
        sum_xx = np.add.reduceat(s_x * s_x, u_starts)
        sum_yy = np.add.reduceat(s_y * s_y, u_starts)
        sum_xy = np.add.reduceat(s_x * s_y, u_starts)
        sum_xz = np.add.reduceat(s_x * s_z, u_starts)
        sum_yz = np.add.reduceat(s_y * s_z, u_starts)

        N = counts.astype(np.float32)
        Sxx = sum_xx - (sum_x * sum_x) / N
        Syy = sum_yy - (sum_y * sum_y) / N
        Sxy = sum_xy - (sum_x * sum_y) / N
        Sxz = sum_xz - (sum_x * z_sum) / N
        Syz = sum_yz - (sum_y * z_sum) / N

        det = Sxx * Syy - Sxy * Sxy
        # Slope is valid for populated cells with non-degenerate det and non-trivial elevation range
        valid_det = (np.abs(det) > 1e-5) & (N >= 3) & (elev_ranges >= 0.01)

        a = np.where(valid_det, (Syy * Sxz - Sxy * Syz) / np.maximum(det, 1e-6), 0.0)
        b = np.where(valid_det, (Sxx * Syz - Sxy * Sxz) / np.maximum(det, 1e-6), 0.0)
        slopes_deg = np.where(valid_det, np.degrees(np.arctan(np.hypot(a, b))), 0.0).astype(np.float32)

        # 6. Fast Vectorized Neighbor Step Check & Terrain Classification
        min_gx, max_gx = int(np.min(u_gx)), int(np.max(u_gx))
        min_gy, max_gy = int(np.min(u_gy)), int(np.max(u_gy))
        span_x = max_gx - min_gx + 3
        span_y = max_gy - min_gy + 3

        dense_h = np.full((span_x, span_y), np.nan, dtype=np.float32)
        ix = u_gx - min_gx + 1
        iy = u_gy - min_gy + 1
        dense_h[ix, iy] = z_min

        h_curr = dense_h[ix, iy]
        diff_l = np.where(np.isnan(dense_h[ix - 1, iy]), 0.0, np.abs(h_curr - dense_h[ix - 1, iy]))
        diff_r = np.where(np.isnan(dense_h[ix + 1, iy]), 0.0, np.abs(h_curr - dense_h[ix + 1, iy]))
        diff_d = np.where(np.isnan(dense_h[ix, iy - 1]), 0.0, np.abs(h_curr - dense_h[ix, iy - 1]))
        diff_u = np.where(np.isnan(dense_h[ix, iy + 1]), 0.0, np.abs(h_curr - dense_h[ix, iy + 1]))
        max_neighbor_step = np.maximum.reduce([diff_l, diff_r, diff_d, diff_u]).astype(np.float32)

        adj_ranges = np.where(
            (elev_ranges < 0.08) & (max_neighbor_step >= 0.12),
            np.maximum(elev_ranges, max_neighbor_step),
            elev_ranges
        )
        has_step_arr = adj_ranges >= 0.08

        step_codes = np.where(
            adj_ranges < 0.08, 0,
            np.where(adj_ranges < 0.25, 1,
            np.where(adj_ranges < 0.50, 2, 3))
        ).astype(np.int8)

        slope_codes = np.where(
            slopes_deg < 5.0, 0,
            np.where(slopes_deg < 15.0, 1,
            np.where(slopes_deg < 25.0, 2,
            np.where(slopes_deg < 35.0, 3, 4)))
        ).astype(np.int8)

        rough_codes = np.where(
            roughness_arr < 0.03, 0,
            np.where(roughness_arr < 0.08, 1,
            np.where(roughness_arr < 0.15, 2,
            np.where(roughness_arr < 0.30, 3, 4)))
        ).astype(np.int8)

        # Patchwork++ Concentric Zone Model (CZM) ground reference (4 rings x 16 sectors = 64 bins)
        ring_idx = np.where(dist_origins < 12.0, 0,
                   np.where(dist_origins < 28.0, 1,
                   np.where(dist_origins < 50.0, 2, 3))).astype(np.int32)
        angles = np.arctan2(world_ys, world_xs)
        sector_idx = np.clip(np.floor((angles + np.pi) / (2.0 * np.pi / 16.0)).astype(np.int32), 0, 15)
        czm_bin = ring_idx * 16 + sector_idx

        czm_ground = np.full(64, np.nan, dtype=np.float32)
        u_bins = np.unique(czm_bin)
        for b in u_bins:
            b_mask = (czm_bin == b)
            czm_ground[b] = np.percentile(z_min[b_mask], 10.0)

        for r in range(4):
            r_vals = czm_ground[r * 16 : (r + 1) * 16]
            valid_r = r_vals[~np.isnan(r_vals)]
            r_ref = float(np.median(valid_r)) if len(valid_r) > 0 else global_ground_z
            nan_sub = np.isnan(r_vals)
            if np.any(nan_sub):
                czm_ground[r * 16 + np.where(nan_sub)[0]] = r_ref

        cell_ground_ref = czm_ground[czm_bin]
        delta_ground = z_mean - cell_ground_ref

        elev_codes = np.where(
            adj_ranges > 0.45, 4,
            np.where(delta_ground > 2.0, 3,
            np.where(delta_ground > 0.25, 1,
            np.where(delta_ground < -0.25, 2, 0)))
        ).astype(np.int8)

        _SLOPE_CATS = [SlopeCategory.FLAT, SlopeCategory.GENTLE, SlopeCategory.MODERATE, SlopeCategory.STEEP, SlopeCategory.EXTREME]
        _ROUGH_CATS = [RoughnessCategory.SMOOTH, RoughnessCategory.LOW, RoughnessCategory.MODERATE, RoughnessCategory.ROUGH, RoughnessCategory.HIGHLY_IRREGULAR]
        _STEP_CATS = [StepCategory.NONE, StepCategory.CURB, StepCategory.STEP_BARRIER, StepCategory.HIGH_OBSTACLE]
        _ELEV_SUMMS = [ElevationSummary.GROUND_LEVEL, ElevationSummary.ELEVATED_SURFACE, ElevationSummary.DEPRESSION_SLOPE, ElevationSummary.OVERHEAD_CLEARANCE, ElevationSummary.VARIABLE_HEIGHT]

        is_hazard = (step_codes == 3) | (adj_ranges > 0.70)
        is_non_drivable = (~is_hazard) & ((step_codes == 2) | (slope_codes >= 3))
        is_caution = (~is_hazard) & (~is_non_drivable) & (
            (elev_codes == 3) | (step_codes == 1) | (slope_codes == 2) | (rough_codes >= 3)
        )
        is_drivable = (~is_hazard) & (~is_non_drivable) & (~is_caution)

        hazard_c = int(np.sum(is_hazard))
        non_drivable_c = int(np.sum(is_non_drivable))
        caution_c = int(np.sum(is_caution))
        drivable_c = int(np.sum(is_drivable))
        unknown_c = 0

        analyzed_cells: List[TerrainCell] = []
        if include_cells:
            r_world_x = np.round(world_xs, 3).tolist()
            r_world_y = np.round(world_ys, 3).tolist()
            r_z_min = np.round(z_min, 3).tolist()
            r_z_max = np.round(z_max, 3).tolist()
            r_z_mean = np.round(z_mean, 3).tolist()
            r_adj_ranges = np.round(adj_ranges, 3).tolist()
            r_slopes = np.round(slopes_deg, 2).tolist()
            r_rough = np.round(roughness_arr, 4).tolist()

            for i in range(num_cells):
                gx_i = int(u_gx[i])
                gy_i = int(u_gy[i])
                cnt = int(counts[i])
                dist = float(dist_origins[i])

                zone = ResolutionZone.NEAR if dist < 12.0 else (ResolutionZone.MID if dist < 28.0 else ResolutionZone.FAR)
                obs_state = ObservationState.DIRECTLY_OBSERVED if cnt >= 15 else (ObservationState.PARTIALLY_OCCLUDED if cnt >= 4 else (ObservationState.SENSOR_FRINGE if cnt >= 1 else ObservationState.UNKNOWN))

                drivability = DrivabilityState.OBSTACLE_HAZARD if is_hazard[i] else (
                    DrivabilityState.NON_DRIVABLE_CANDIDATE if is_non_drivable[i] else (
                        DrivabilityState.CAUTION_IRREGULAR if is_caution[i] else DrivabilityState.DRIVABLE_CANDIDATE
                    )
                )

                s_code = int(slope_codes[i])
                r_code = int(rough_codes[i])
                st_code = int(step_codes[i])
                e_code = int(elev_codes[i])

                slope_cat = _SLOPE_CATS[s_code]
                rough_cat = _ROUGH_CATS[r_code]
                step_cat = _STEP_CATS[st_code]
                elev_summ = _ELEV_SUMMS[e_code]

                if is_hazard[i] or (is_non_drivable[i] and st_code == 2):
                    terrain_interp = TerrainInterpretation.OBSTACLE_BARRIER
                elif is_non_drivable[i] or s_code in (2, 3, 4):
                    terrain_interp = TerrainInterpretation.SLOPED_ROAD
                elif e_code == 3:
                    terrain_interp = TerrainInterpretation.SPARSE_FOLIAGE_OR_OVERHANG
                elif st_code == 1:
                    terrain_interp = TerrainInterpretation.CURB_BOUNDARY
                elif r_code in (3, 4):
                    terrain_interp = TerrainInterpretation.ROUGH_UNPAVED
                else:
                    terrain_interp = TerrainInterpretation.PAVED_FLAT

                rng_val = r_adj_ranges[i]
                cell = TerrainCell.model_construct(
                    id=f"cell_{gx_i}_{gy_i}",
                    grid_x=gx_i,
                    grid_y=gy_i,
                    world_x=r_world_x[i],
                    world_y=r_world_y[i],
                    size_m=res,
                    point_count=cnt,
                    min_z=r_z_min[i],
                    max_z=r_z_max[i],
                    mean_z=r_z_mean[i],
                    elevation_range=rng_val,
                    slope_deg=r_slopes[i],
                    roughness_m=r_rough[i],
                    step_height_m=rng_val,
                    has_step=bool(has_step_arr[i]),
                    slope_category=slope_cat,
                    roughness_category=rough_cat,
                    step_category=step_cat,
                    observation_state=obs_state,
                    elevation_summary=elev_summ,
                    terrain_interpretation=terrain_interp,
                    drivability_state=drivability,
                    zone=zone,
                )
                analyzed_cells.append(cell)

        if prep_report is not None and prep_report.bounds is not None:
            t_bounds = prep_report.bounds
        elif num_pts > 0:
            min_v = xyz.min(axis=0)
            max_v = xyz.max(axis=0)
            t_bounds = BoundingBox3D.model_construct(
                min_x=float(min_v[0]), max_x=float(max_v[0]),
                min_y=float(min_v[1]), max_y=float(max_v[1]),
                min_z=float(min_v[2]), max_z=float(max_v[2]),
            )
        else:
            t_bounds = BoundingBox3D.model_construct(min_x=0.0, max_x=0.0, min_y=0.0, max_y=0.0, min_z=0.0, max_z=0.0)

        summary = TerrainAnalysisSummary.model_construct(
            total_cells=num_cells,
            drivable_cells=drivable_c,
            non_drivable_cells=non_drivable_c,
            caution_cells=caution_c,
            hazard_cells=hazard_c,
            unknown_cells=unknown_c,
            grid_resolution_m=res,
            bounds=t_bounds,
        )

        return TerrainAnalysisResponse.model_construct(
            frame_id=frame_id,
            status=ProcessStatus.COMPLETE,
            preprocessing_report=prep_report,
            summary=summary,
            cells=analyzed_cells,
            created_at=now_iso,
        )
