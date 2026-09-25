"""Adaptive Variable-Resolution 2.5D Grid Engine.

Aggregates classified 3D LiDAR points, geometric terrain measurements, and tracked objects
into an elevation-aware, foveated 2.5D multi-resolution grid.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from datetime import datetime, timezone

from ..models.schemas import BoundingBox3D
from ..models.map_schemas import (
    ResolutionLevel,
    CellObservationState,
    TraversabilityState,
    AmbiguityState,
    AdaptiveGridCell,
    ElevationLayer,
    GridPolicyConfig,
    MapMetadata,
    MapMode,
)
from .resolution_policy import ResolutionPolicyService
from .observability import ObservabilityService
from .semantic_mapping import SemanticMappingService


from .semantic_registry import SemanticClassRegistry


def _level_rank(level: ResolutionLevel) -> int:
    """Return 0=FINE, 1=MEDIUM, 2=COARSE for ordinal comparisons."""
    return {"fine": 0, "medium": 1, "coarse": 2}[level.value]


# Dataset-aware lookup caches
def _get_label_mappings(dataset: Optional[str] = None):
    d_norm = SemanticClassRegistry.normalize_dataset_name(dataset)
    store = SemanticClassRegistry.get_dataset_mapping_dict(d_norm)

    name_map: Dict[int, str] = {}
    cat_map: Dict[int, str] = {}
    dyn_cids: List[int] = []

    for cid, info in store.items():
        name_map[cid] = info["name"]
        cat_map[cid] = info["category"]
        if info["is_dynamic"]:
            dyn_cids.append(cid)

    return name_map, cat_map, dyn_cids


_LABEL_TO_NAME, _LABEL_TO_CAT, _DYNAMIC_CIDS = _get_label_mappings("semantickitti")
_RL_LABEL_TO_NAME, _RL_LABEL_TO_CAT, _RL_DYNAMIC_CIDS = _get_label_mappings("rellis")


class AdaptiveGridService:
    """Core 2.5D Foveated Grid generation engine."""

    @classmethod
    def generate_grid_from_points(
        cls,
        points: np.ndarray,
        frame_id: str,
        labels: Optional[np.ndarray] = None,
        instance_ids: Optional[np.ndarray] = None,
        dynamic_track_positions: Optional[List[List[float]]] = None,
        policy: Optional[GridPolicyConfig] = None,
        sensor_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        dataset: Optional[str] = None,
    ) -> List[AdaptiveGridCell]:
        """Build sparse 2.5D variable-resolution grid cells from point cloud and annotations."""
        if points is None or len(points) == 0:
            return []

        active_policy = policy or ResolutionPolicyService.get_policy()
        xyz = points[:, :3]
        num_points = xyz.shape[0]

        d_name, d_cat_map, dyn_cids = _get_label_mappings(
            dataset or ("rellis" if "rellis" in frame_id.lower() else "semantickitti")
        )

        # 1. Dynamic mask: points near tracked dynamic actors or classified as dynamic_object
        dynamic_mask = np.zeros(num_points, dtype=bool)
        if dynamic_track_positions:
            for dpos in dynamic_track_positions:
                dx = xyz[:, 0] - dpos[0]
                dy = xyz[:, 1] - dpos[1]
                dynamic_mask |= (dx * dx + dy * dy) <= 4.0  # 2m radius
        if labels is not None and len(labels) == num_points:
            dynamic_mask |= np.isin(labels, dyn_cids)

        # 3. Base foveation by radial distance from sensor origin
        distances_xy = np.hypot(
            xyz[:, 0] - sensor_origin[0],
            xyz[:, 1] - sensor_origin[1],
        )
        d_fine = active_policy.near_zone_max_distance_m
        d_med = active_policy.mid_zone_max_distance_m

        is_fine = distances_xy <= d_fine
        if active_policy.safety_priority:
            is_fine = is_fine | dynamic_mask
        is_med = (~is_fine) & (distances_xy <= d_med)
        is_coarse = ~(is_fine | is_med)

        # Per-point level codes (0=FINE, 1=MEDIUM, 2=COARSE) for safe numpy ops
        level_codes = np.where(
            is_fine, 0,
            np.where(is_med, 1, 2),
        ).astype(np.int8)
        code_to_level = {
            0: ResolutionLevel.FINE,
            1: ResolutionLevel.MEDIUM,
            2: ResolutionLevel.COARSE,
        }

        r_fine = ResolutionPolicyService.get_cell_size(ResolutionLevel.FINE, active_policy)
        r_med = ResolutionPolicyService.get_cell_size(ResolutionLevel.MEDIUM, active_policy)
        r_coarse = ResolutionPolicyService.get_cell_size(ResolutionLevel.COARSE, active_policy)

        # 4. Vectorized multi-resolution bucketing & refinement
        grid_cells: List[AdaptiveGridCell] = []
        
        pts_x = xyz[:, 0]
        pts_y = xyz[:, 1]
        pts_z = xyz[:, 2]

        now_iso = datetime.now(timezone.utc).isoformat()
        obstacle_centers: List[Tuple[float, float]] = []
        fpm, mpc = ResolutionPolicyService.get_level_ratio(active_policy)

        # Vectorized pass across each resolution level
        for code in (0, 1, 2):
            mask = (level_codes == code)
            if not np.any(mask):
                continue
            p_idxs = np.where(mask)[0]
            r = ResolutionPolicyService.get_cell_size(code_to_level[code], active_policy)
            lvl_enum = code_to_level[code]
            lvl_str = lvl_enum.value

            sub_x = pts_x[p_idxs]
            sub_y = pts_y[p_idxs]
            sub_z = pts_z[p_idxs]
            sub_lbl = labels[p_idxs] if labels is not None else None
            sub_dyn = dynamic_mask[p_idxs]

            gx = np.floor(sub_x / r).astype(np.int32)
            gy = np.floor(sub_y / r).astype(np.int32)
            packed = (gx.astype(np.int64) << 32) | (gy.astype(np.int64) & 0xFFFFFFFF)

            sort_ord = np.argsort(packed)
            s_packed = packed[sort_ord]
            u_packed, u_starts, counts = np.unique(s_packed, return_index=True, return_counts=True)
            num_c = len(u_packed)

            u_gx = (u_packed >> 32).astype(np.int64).astype(np.int32)
            u_gy = (u_packed & 0xFFFFFFFF).astype(np.uint32).view(np.int32)

            s_z = sub_z[sort_ord]
            s_x = sub_x[sort_ord]
            s_y = sub_y[sort_ord]
            s_dyn = sub_dyn[sort_ord]

            # Vectorized Reductions via reduceat
            z_max = np.maximum.reduceat(s_z, u_starts)
            z_min = np.minimum.reduceat(s_z, u_starts)
            z_sum = np.add.reduceat(s_z, u_starts)
            z_mean = z_sum / counts
            z_range = z_max - z_min

            z_sq_sum = np.add.reduceat(s_z * s_z, u_starts)
            z_var = np.maximum(0.0, (z_sq_sum / counts) - (z_mean * z_mean))
            roughness_arr = np.sqrt(z_var)

            dyn_sum = np.add.reduceat(s_dyn.astype(np.int32), u_starts)
            is_dyn_arr = dyn_sum > 0

            # Vectorized analytical slopes via moments
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
            valid_det = (np.abs(det) > 1e-5) & (N >= 3) & (z_range >= 0.01)

            a = np.where(valid_det, (Syy * Sxz - Sxy * Syz) / np.maximum(det, 1e-6), 0.0)
            b = np.where(valid_det, (Sxx * Syz - Sxy * Sxz) / np.maximum(det, 1e-6), 0.0)
            slopes_deg = np.where(valid_det, np.degrees(np.arctan(np.hypot(a, b))), 0.0).astype(np.float32)
            # Physical bound: numerical least-squares slope cannot exceed physical geometry across cell width
            max_geom_slope = np.degrees(np.arctan(z_range / np.maximum(0.25 * r, 0.10))).astype(np.float32)
            slopes_deg = np.minimum(slopes_deg, max_geom_slope)

            # Fast Dominant Class Extraction
            dom_cids = np.zeros(num_c, dtype=np.uint16)
            if sub_lbl is not None:
                s_lbl = sub_lbl[sort_ord]
                for c_idx in range(num_c):
                    cnt = counts[c_idx]
                    st = u_starts[c_idx]
                    if cnt == 1 or s_lbl[st] == s_lbl[st + cnt - 1]:
                        dom_cids[c_idx] = s_lbl[st]
                    else:
                        dom_cids[c_idx] = np.bincount(s_lbl[st : st + cnt]).argmax()

            # Vectorized pre-rounding for fast Python cell model instantiations
            w_x = np.round((u_gx + 0.5) * r, 3).tolist()
            w_y = np.round((u_gy + 0.5) * r, 3).tolist()
            b_min_x = np.round(u_gx * r, 3).tolist()
            b_max_x = np.round((u_gx + 1) * r, 3).tolist()
            b_min_y = np.round(u_gy * r, 3).tolist()
            b_max_y = np.round((u_gy + 1) * r, 3).tolist()
            r_z_min = np.round(z_min, 3).tolist()
            r_z_max = np.round(z_max, 3).tolist()
            r_z_mean = np.round(z_mean, 3).tolist()
            r_z_range = np.round(z_range, 3).tolist()
            r_slope = np.round(slopes_deg, 2).tolist()
            r_rough = np.round(roughness_arr, 3).tolist()

            # Build AdaptiveGridCell models
            for c_idx in range(num_c):
                g_x = int(u_gx[c_idx])
                g_y = int(u_gy[c_idx])
                cnt = int(counts[c_idx])
                mn_z = r_z_min[c_idx]
                mx_z = r_z_max[c_idx]
                mu_z = r_z_mean[c_idx]
                rng_z = r_z_range[c_idx]
                slp = r_slope[c_idx]
                rgh = r_rough[c_idx]
                is_d = bool(is_dyn_arr[c_idx])

                cid = int(dom_cids[c_idx])
                d_cls = d_name.get(cid, "unlabeled")
                d_cat = d_cat_map.get(cid, "unknown")

                min_x = b_min_x[c_idx]
                max_x = b_max_x[c_idx]
                min_y = b_min_y[c_idx]
                max_y = b_max_y[c_idx]
                world_x = w_x[c_idx]
                world_y = w_y[c_idx]

                is_s = (d_cat in ["static_obstacle", "infrastructure"]) or (rng_z >= active_policy.obstacle_height_span_threshold_m and not is_d)

                if is_s:
                    obstacle_centers.append((world_x, world_y))

                # Overhang check on candidate cells
                has_overhang = False
                overhead_clearance_m: Optional[float] = None
                layers: List[ElevationLayer] = []

                if cnt >= 6 and rng_z > 1.8:
                    st = u_starts[c_idx]
                    cell_z_sorted = np.sort(s_z[st : st + cnt])
                    z_diffs = np.diff(cell_z_sorted)
                    max_gap_idx = int(np.argmax(z_diffs))
                    max_gap = float(z_diffs[max_gap_idx])
                    if max_gap >= 1.5:
                        ground_pts = cell_z_sorted[:max_gap_idx + 1]
                        overhang_pts = cell_z_sorted[max_gap_idx + 1:]
                        g_min, g_max = float(np.min(ground_pts)), float(np.max(ground_pts))
                        o_min, o_max = float(np.min(overhang_pts)), float(np.max(overhang_pts))
                        has_overhang = True
                        overhead_clearance_m = round(o_min - g_max, 3)
                        layers = [
                            ElevationLayer.model_construct(
                                layer_index=0,
                                layer_type="ground",
                                elevation_min=round(g_min, 3),
                                elevation_mean=round(float(np.mean(ground_pts)), 3),
                                elevation_max=round(g_max, 3),
                                elevation_range=round(g_max - g_min, 3),
                                point_count=len(ground_pts),
                                dominant_category=d_cat,
                                traversability_state=TraversabilityState.DRIVABLE if (d_cat == "drivable" or (g_max - g_min <= 0.25)) else TraversabilityState.CAUTION_IRREGULAR,
                            ),
                            ElevationLayer.model_construct(
                                layer_index=1,
                                layer_type="overhang",
                                elevation_min=round(o_min, 3),
                                elevation_mean=round(float(np.mean(overhang_pts)), 3),
                                elevation_max=round(o_max, 3),
                                elevation_range=round(o_max - o_min, 3),
                                point_count=len(overhang_pts),
                                dominant_category="infrastructure" if d_cat in ["infrastructure", "building"] else "vegetation",
                                traversability_state=TraversabilityState.NON_TRAVERSABLE,
                            ),
                        ]

                # Traversability
                if is_d:
                    traversability = TraversabilityState.COLLISION_HAZARD
                elif is_s and rng_z >= 0.5 and not (has_overhang and overhead_clearance_m is not None and overhead_clearance_m >= 2.0 and d_cat == "drivable"):
                    traversability = TraversabilityState.COLLISION_HAZARD
                elif has_overhang and overhead_clearance_m is not None and overhead_clearance_m >= 2.0 and d_cat == "drivable":
                    traversability = TraversabilityState.DRIVABLE
                elif d_cat in ["non_drivable", "vegetation", "infrastructure"] or slp > 30.0:
                    traversability = TraversabilityState.NON_TRAVERSABLE
                elif slp > 15.0 or rgh > 0.15 or rng_z > 0.20:
                    traversability = TraversabilityState.CAUTION_IRREGULAR
                elif d_cat == "drivable" or (slp <= 15.0 and rgh <= 0.15 and rng_z <= 0.25):
                    traversability = TraversabilityState.DRIVABLE
                else:
                    traversability = TraversabilityState.UNCERTAIN

                # Ambiguity
                ambiguity = AmbiguityState.SPARSE_DATA if cnt < 3 else (AmbiguityState.HIGH_GRADIENT if (rng_z > 0.60 or slp > 20.0) else AmbiguityState.UNAMBIGUOUS)

                # Terrain state string
                if is_s:
                    terrain_state = "obstacle_barrier"
                elif has_overhang:
                    terrain_state = "overhead_clearance"
                elif slp > 15.0:
                    terrain_state = "sloped_road"
                elif rgh > 0.15:
                    terrain_state = "rough_unpaved"
                elif d_cat == "drivable":
                    terrain_state = "paved_flat"
                elif d_cat == "vegetation":
                    terrain_state = "foliage"
                else:
                    terrain_state = "unknown"

                key = f"{lvl_str}:{g_x}_{g_y}"
                if code == 0:
                    parent_key = f"medium:{g_x // fpm}_{g_y // fpm}"
                    child_keys = []
                elif code == 1:
                    parent_key = f"coarse:{g_x // mpc}_{g_y // mpc}"
                    child_keys = [f"fine:{g_x * fpm + dx}_{g_y * fpm + dy}" for dy in range(fpm) for dx in range(fpm)]
                else:
                    parent_key = None
                    child_keys = [f"medium:{g_x * mpc + dx}_{g_y * mpc + dy}" for dy in range(mpc) for dx in range(mpc)]

                obs_state = CellObservationState.OBSERVED if cnt >= 4 else (CellObservationState.PARTIALLY_OCCLUDED if cnt >= 1 else CellObservationState.UNKNOWN)

                cell = AdaptiveGridCell.create_fast(
                    cell_key=key,
                    level=lvl_enum,
                    size_m=r,
                    grid_x=g_x,
                    grid_y=g_y,
                    world_x=world_x,
                    world_y=world_y,
                    bounds=[min_x, max_x, min_y, max_y],
                    observation_state=obs_state,
                    semantic_histogram={d_cls: cnt} if d_cls != "unlabeled" else {},
                    dominant_semantic_class=d_cls,
                    dominant_category=d_cat,
                    terrain_state=terrain_state,
                    traversability_state=traversability,
                    is_static_obstacle=is_s,
                    is_dynamic_obstacle=is_d,
                    elevation_min=mn_z,
                    elevation_mean=mu_z,
                    elevation_max=mx_z,
                    elevation_variation=rng_z,
                    roughness_summary=rgh,
                    slope_summary=slp,
                    point_count=cnt,
                    has_overhang=has_overhang,
                    overhead_clearance_m=overhead_clearance_m,
                    layers=layers,
                    last_frame_id=frame_id,
                    last_timestamp=now_iso,
                    ambiguity_state=ambiguity,
                    parent_key=parent_key,
                    child_keys=child_keys,
                )
                grid_cells.append(cell)

        return grid_cells

    @classmethod
    def generate_grid_from_points_fast(
        cls,
        points: np.ndarray,
        frame_id: str,
        labels: Optional[np.ndarray] = None,
        instance_ids: Optional[np.ndarray] = None,
        dynamic_track_positions: Optional[List[List[float]]] = None,
        policy: Optional[GridPolicyConfig] = None,
        sensor_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        dataset: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Ultra-fast zero-overhead 2.5D Adaptive Grid generation emitting lightweight dicts."""
        if points is None or len(points) == 0:
            return []

        active_policy = policy or ResolutionPolicyService.get_policy()
        xyz = points[:, :3]
        num_points = xyz.shape[0]

        d_name, d_cat_map, dyn_cids = _get_label_mappings(
            dataset or ("rellis" if "rellis" in frame_id.lower() else "semantickitti")
        )

        # 1. Dynamic mask
        dynamic_mask = np.zeros(num_points, dtype=bool)
        if dynamic_track_positions:
            for dpos in dynamic_track_positions:
                dx = xyz[:, 0] - dpos[0]
                dy = xyz[:, 1] - dpos[1]
                dynamic_mask |= (dx * dx + dy * dy) <= 4.0
        if labels is not None and len(labels) == num_points:
            dynamic_mask |= np.isin(labels, dyn_cids)

        # 2. Base foveation
        distances_xy = np.hypot(xyz[:, 0] - sensor_origin[0], xyz[:, 1] - sensor_origin[1])
        d_fine = active_policy.near_zone_max_distance_m
        d_med = active_policy.mid_zone_max_distance_m

        is_fine = distances_xy <= d_fine
        if active_policy.safety_priority:
            is_fine = is_fine | dynamic_mask
        is_med = (~is_fine) & (distances_xy <= d_med)

        level_codes = np.where(is_fine, 0, np.where(is_med, 1, 2)).astype(np.int8)
        code_to_level = {0: "fine", 1: "medium", 2: "coarse"}

        pts_x = xyz[:, 0]
        pts_y = xyz[:, 1]
        pts_z = xyz[:, 2]

        now_iso = datetime.now(timezone.utc).isoformat()
        fpm, mpc = ResolutionPolicyService.get_level_ratio(active_policy)
        grid_cells: List[Dict[str, Any]] = []

        for code in (0, 1, 2):
            mask = (level_codes == code)
            if not np.any(mask):
                continue
            p_idxs = np.where(mask)[0]
            lvl_str = code_to_level[code]
            r = ResolutionPolicyService.get_cell_size(ResolutionLevel(lvl_str), active_policy)

            sub_x = pts_x[p_idxs]
            sub_y = pts_y[p_idxs]
            sub_z = pts_z[p_idxs]
            sub_lbl = labels[p_idxs] if labels is not None else None
            sub_dyn = dynamic_mask[p_idxs]

            gx = np.floor(sub_x / r).astype(np.int32)
            gy = np.floor(sub_y / r).astype(np.int32)
            packed = (gx.astype(np.int64) << 32) | (gy.astype(np.int64) & 0xFFFFFFFF)

            sort_ord = np.argsort(packed)
            s_packed = packed[sort_ord]
            u_packed, u_starts, counts = np.unique(s_packed, return_index=True, return_counts=True)
            num_c = len(u_packed)

            u_gx = (u_packed >> 32).astype(np.int64).astype(np.int32)
            u_gy = (u_packed & 0xFFFFFFFF).astype(np.uint32).view(np.int32)

            s_z = sub_z[sort_ord]
            s_x = sub_x[sort_ord]
            s_y = sub_y[sort_ord]
            s_dyn = sub_dyn[sort_ord]

            # Vectorized Reductions via reduceat
            z_max = np.maximum.reduceat(s_z, u_starts)
            z_min = np.minimum.reduceat(s_z, u_starts)
            z_sum = np.add.reduceat(s_z, u_starts)
            z_mean = z_sum / counts
            z_range = z_max - z_min

            z_sq_sum = np.add.reduceat(s_z * s_z, u_starts)
            z_var = np.maximum(0.0, (z_sq_sum / counts) - (z_mean * z_mean))
            roughness_arr = np.sqrt(z_var)

            dyn_sum = np.add.reduceat(s_dyn.astype(np.int32), u_starts)
            is_dyn_arr = dyn_sum > 0

            # Vectorized analytical slopes
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
            valid_det = (np.abs(det) > 1e-5) & (N >= 3) & (z_range >= 0.12)

            a = np.where(valid_det, (Syy * Sxz - Sxy * Syz) / np.maximum(det, 1e-6), 0.0)
            b = np.where(valid_det, (Sxx * Syz - Sxy * Sxz) / np.maximum(det, 1e-6), 0.0)
            slopes_deg = np.where(valid_det, np.degrees(np.arctan(np.hypot(a, b))), 0.0).astype(np.float32)

            # Dominant labels — correct statistical mode via bincount.
            # For each cell we count occurrences of each label ID and pick the most frequent.
            # Using np.bincount is O(max_label) per cell but avoids any Python-level looping
            # over individual points, keeping this faster than a per-point dict approach.
            dom_cids = np.zeros(num_c, dtype=np.uint16)
            if sub_lbl is not None:
                s_lbl = sub_lbl[sort_ord]
                max_lbl = int(s_lbl.max()) + 1 if len(s_lbl) > 0 else 1
                for c_idx in range(num_c):
                    cnt = counts[c_idx]
                    st = u_starts[c_idx]
                    seg = s_lbl[st : st + cnt]
                    if cnt == 1 or seg[0] == seg[-1]:
                        # All same label — fast path, no bincount needed.
                        dom_cids[c_idx] = seg[0]
                    else:
                        dom_cids[c_idx] = np.argmax(np.bincount(seg, minlength=max_lbl))

            w_x = np.round((u_gx + 0.5) * r, 3)
            w_y = np.round((u_gy + 0.5) * r, 3)
            b_min_x = np.round(u_gx * r, 3)
            b_max_x = np.round((u_gx + 1) * r, 3)
            b_min_y = np.round(u_gy * r, 3)
            b_max_y = np.round((u_gy + 1) * r, 3)
            r_z_min = np.round(z_min, 3)
            r_z_max = np.round(z_max, 3)
            r_z_mean = np.round(z_mean, 3)
            r_z_range = np.round(z_range, 3)
            r_slope = np.round(slopes_deg, 2)
            r_rough = np.round(roughness_arr, 3)

            # Build lightweight dicts directly
            obs_thresh = active_policy.obstacle_height_span_threshold_m
            is_static_arr = (r_z_range >= obs_thresh) & (~is_dyn_arr)

            for c_idx in range(num_c):
                g_x = int(u_gx[c_idx])
                g_y = int(u_gy[c_idx])
                cnt = int(counts[c_idx])
                mn_z = float(r_z_min[c_idx])
                mx_z = float(r_z_max[c_idx])
                mu_z = float(r_z_mean[c_idx])
                rng_z = float(r_z_range[c_idx])
                slp = float(r_slope[c_idx])
                rgh = float(r_rough[c_idx])
                is_d = bool(is_dyn_arr[c_idx])
                is_s = bool(is_static_arr[c_idx])

                cid = int(dom_cids[c_idx])
                d_cls = d_name.get(cid, "unlabeled")
                d_cat = d_cat_map.get(cid, "unknown")
                if d_cat in ["static_obstacle", "infrastructure"]:
                    is_s = True

                # Overhang check on candidate cells
                has_overhang = False
                overhead_clearance_m = None
                layers = []

                if cnt >= 6 and rng_z > 1.8:
                    st = u_starts[c_idx]
                    cell_z_sorted = np.sort(s_z[st : st + cnt])
                    z_diffs = np.diff(cell_z_sorted)
                    max_gap_idx = int(np.argmax(z_diffs))
                    max_gap = float(z_diffs[max_gap_idx])
                    if max_gap >= 1.5:
                        ground_pts = cell_z_sorted[:max_gap_idx + 1]
                        overhang_pts = cell_z_sorted[max_gap_idx + 1:]
                        g_min, g_max = float(np.min(ground_pts)), float(np.max(ground_pts))
                        o_min, o_max = float(np.min(overhang_pts)), float(np.max(overhang_pts))
                        has_overhang = True
                        overhead_clearance_m = round(o_min - g_max, 3)
                        layers = [
                            {
                                "layer_index": 0,
                                "layer_type": "ground",
                                "elevation_min": round(g_min, 3),
                                "elevation_mean": round(float(np.mean(ground_pts)), 3),
                                "elevation_max": round(g_max, 3),
                                "elevation_range": round(g_max - g_min, 3),
                                "point_count": len(ground_pts),
                                "dominant_category": d_cat,
                                "traversability_state": "drivable" if (d_cat == "drivable" or (g_max - g_min <= 0.25)) else "caution_irregular",
                            },
                            {
                                "layer_index": 1,
                                "layer_type": "overhang",
                                "elevation_min": round(o_min, 3),
                                "elevation_mean": round(float(np.mean(overhang_pts)), 3),
                                "elevation_max": round(o_max, 3),
                                "elevation_range": round(o_max - o_min, 3),
                                "point_count": len(overhang_pts),
                                "dominant_category": "infrastructure" if d_cat in ["infrastructure", "building"] else "vegetation",
                                "traversability_state": "non_traversable",
                            },
                        ]

                # Traversability
                if is_d:
                    trav = "collision_hazard"
                elif is_s and rng_z >= 0.5 and not (has_overhang and overhead_clearance_m is not None and overhead_clearance_m >= 2.0 and d_cat == "drivable"):
                    trav = "collision_hazard"
                elif has_overhang and overhead_clearance_m is not None and overhead_clearance_m >= 2.0 and d_cat == "drivable":
                    trav = "drivable"
                elif d_cat in ["non_drivable", "vegetation", "infrastructure"] or slp > 30.0:
                    trav = "non_traversable"
                elif slp > 15.0 or rgh > 0.15 or rng_z > 0.20:
                    trav = "caution_irregular"
                elif d_cat == "drivable" or (slp <= 15.0 and rgh <= 0.15 and rng_z <= 0.25):
                    trav = "drivable"
                else:
                    trav = "uncertain"

                # Terrain state string
                if is_s:
                    terrain_state = "obstacle_barrier"
                elif has_overhang:
                    terrain_state = "overhead_clearance"
                elif slp > 15.0:
                    terrain_state = "sloped_road"
                elif rgh > 0.15:
                    terrain_state = "rough_unpaved"
                elif d_cat == "drivable":
                    terrain_state = "paved_flat"
                elif d_cat == "vegetation":
                    terrain_state = "foliage"
                else:
                    terrain_state = "unknown"

                obs_state = "observed" if cnt >= 4 else ("partially_occluded" if cnt >= 1 else "unknown")
                key = f"{lvl_str}:{g_x}_{g_y}"

                cell_dict = {
                    "cell_key": key,
                    "level": lvl_str,
                    "size_m": r,
                    "grid_x": g_x,
                    "grid_y": g_y,
                    "world_x": float(w_x[c_idx]),
                    "world_y": float(w_y[c_idx]),
                    "bounds": [float(b_min_x[c_idx]), float(b_max_x[c_idx]), float(b_min_y[c_idx]), float(b_max_y[c_idx])],
                    "observation_state": obs_state,
                    "dominant_semantic_class": d_cls,
                    "dominant_category": d_cat,
                    "terrain_state": terrain_state,
                    "traversability_state": trav,
                    "is_static_obstacle": is_s,
                    "is_dynamic_obstacle": is_d,
                    "elevation_min": mn_z,
                    "elevation_mean": mu_z,
                    "elevation_max": mx_z,
                    "elevation_variation": rng_z,
                    "roughness_summary": rgh,
                    "slope_summary": slp,
                    "point_count": cnt,
                    "has_overhang": has_overhang,
                    "overhead_clearance_m": overhead_clearance_m,
                    "layers": layers,
                    "last_frame_id": frame_id,
                    "last_timestamp": now_iso,
                }
                grid_cells.append(cell_dict)

        return grid_cells

    @classmethod
    def compute_map_metadata(
        cls,
        map_id: str,
        mode: MapMode,
        cells: List[Any],
        frame_count: int = 1,
        created_at: Optional[str] = None,
    ) -> MapMetadata:
        """Compute overall metadata and bounding box for a collection of cells (objects or dicts)."""
        now_iso = datetime.now(timezone.utc).isoformat()
        c_at = created_at or now_iso

        fine_cnt = 0
        med_cnt = 0
        coarse_cnt = 0

        min_x = 1e9
        max_x = -1e9
        min_y = 1e9
        max_y = -1e9
        min_z = 1e9
        max_z = -1e9

        for c in cells:
            if isinstance(c, dict):
                lvl = c.get("level")
                if lvl in ("fine", ResolutionLevel.FINE):
                    fine_cnt += 1
                elif lvl in ("medium", ResolutionLevel.MEDIUM):
                    med_cnt += 1
                elif lvl in ("coarse", ResolutionLevel.COARSE):
                    coarse_cnt += 1
                b = c.get("bounds", [0, 0, 0, 0])
                if b[0] < min_x: min_x = b[0]
                if b[1] > max_x: max_x = b[1]
                if b[2] < min_y: min_y = b[2]
                if b[3] > max_y: max_y = b[3]
                em = c.get("elevation_min", 0.0)
                ex = c.get("elevation_max", 0.0)
                if em < min_z: min_z = em
                if ex > max_z: max_z = ex
            else:
                lvl = c.level
                if lvl == ResolutionLevel.FINE:
                    fine_cnt += 1
                elif lvl == ResolutionLevel.MEDIUM:
                    med_cnt += 1
                elif lvl == ResolutionLevel.COARSE:
                    coarse_cnt += 1
                if c.bounds[0] < min_x: min_x = c.bounds[0]
                if c.bounds[1] > max_x: max_x = c.bounds[1]
                if c.bounds[2] < min_y: min_y = c.bounds[2]
                if c.bounds[3] > max_y: max_y = c.bounds[3]
                if c.elevation_min < min_z: min_z = c.elevation_min
                if c.elevation_max > max_z: max_z = c.elevation_max

        if not cells:
            min_x = max_x = min_y = max_y = min_z = max_z = 0.0

        return MapMetadata(
            map_id=map_id,
            mode=mode,
            frame_count=frame_count,
            total_cells=len(cells),
            fine_cells_count=fine_cnt,
            medium_cells_count=med_cnt,
            coarse_cells_count=coarse_cnt,
            bounds=BoundingBox3D(
                min_x=round(float(min_x), 3),
                max_x=round(float(max_x), 3),
                min_y=round(float(min_y), 3),
                max_y=round(float(max_y), 3),
                min_z=round(float(min_z), 3),
                max_z=round(float(max_z), 3),
            ),
            created_at=c_at,
            updated_at=now_iso,
        )
