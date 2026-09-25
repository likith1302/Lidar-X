"""Geometric Object Instance Clustering for point clouds with semantic annotations.

Performs 3D-aware spatial clustering, PCA yaw / heading angle estimation,
and oriented bounding box (OBB) extraction for discrete semantic classes.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from scipy.ndimage import label as nd_label
from sklearn.cluster import DBSCAN

from ..models.schemas import BoundingBox3D, Point3D
from ..models.object_schemas import (
    ObjectInstance,
    ObjectDetectionResponse,
    ProjectCategory,
    ExtentCategory,
    ObservationStatus,
    ProcessStatus,
)
from .semantic_registry import SemanticClassRegistry
from .lidar_parser import LidarParser


class InstanceClusteringService:
    """Performs 3D spatial clustering and oriented bounding box fitting on segmented point clouds."""

    @classmethod
    def detect_objects(
        cls,
        frame_id: str,
        points: np.ndarray,
        semantic_classes: np.ndarray,
        instance_ids: Optional[np.ndarray] = None,
        min_points_per_cluster: int = 5,
        default_radius_m: float = 0.8,
        include_sample_points: bool = False,
        dataset: Optional[str] = None,
    ) -> ObjectDetectionResponse:
        """Group semantic points into distinct 3D geometric object instances with OBB.

        Args:
            frame_id: Target LiDAR frame ID.
            points: Float32 point array (N, >=3).
            semantic_classes: Uint16 class IDs array (N,).
            instance_ids: Optional pre-existing instance IDs from dataset.
            min_points_per_cluster: Minimum points to form an object instance.
            default_radius_m: Spatial search radius epsilon in meters.
            include_sample_points: Whether to downsample and include preview points.
            dataset: "rellis" or "semantickitti".

        Returns:
            ObjectDetectionResponse with structured ObjectInstance list.
        """
        if points.shape[0] == 0 or len(semantic_classes) == 0:
            return ObjectDetectionResponse(
                frame_id=frame_id,
                total_instances=0,
                dynamic_instances_count=0,
                static_instances_count=0,
                instances=[],
                status=ProcessStatus.COMPLETE,
                created_at="",
            )

        d_norm = SemanticClassRegistry.normalize_dataset_name(dataset or ("rellis" if "rellis" in frame_id.lower() else "semantickitti"))
        unique_class_ids = np.unique(semantic_classes)

        instances: List[ObjectInstance] = []
        cluster_counter = 0

        for cid_raw in unique_class_ids:
            cid = int(cid_raw)
            c_info = SemanticClassRegistry.get_class_info(cid, dataset=d_norm)

            # Only cluster discrete object classes (vehicles, pedestrians, poles, obstacles, buildings, vegetation)
            if not c_info.clusterable:
                continue

            class_mask = (semantic_classes == cid)
            class_pts = points[class_mask]

            if class_pts.shape[0] < min_points_per_cluster:
                continue

            c_name = c_info.name.lower()

            # For unlabeled/unknown points, only consider elevated points
            if cid == 0 or c_name in ("unknown", "unlabeled", "void", "outlier"):
                elev_mask = class_pts[:, 2] > -1.1
                class_pts = class_pts[elev_mask]
                if class_pts.shape[0] < max(8, min_points_per_cluster):
                    continue

            is_actor = any(k in c_name for k in (
                "person", "pedestrian", "bicyclist", "motorcyclist",
                "bicycle", "motorcycle", "car", "truck", "bus", "vehicle",
                "pole", "sign", "trunk"
            ))

            # Class-specific 3D spatial clustering tolerances
            if any(k in c_name for k in ("bicyclist", "motorcyclist")):
                eps = 0.45
                min_samples = max(4, min_points_per_cluster)
                min_pts_thresh = 10
                voxel_xy = 0.35
                voxel_z = 0.45
                max_extent = 2.8
            elif any(k in c_name for k in ("person", "pedestrian")):
                eps = 0.40
                min_samples = max(3, min_points_per_cluster - 1)
                min_pts_thresh = 5
                voxel_xy = 0.35
                voxel_z = 0.45
                max_extent = 2.0
            elif any(k in c_name for k in ("bicycle", "motorcycle")):
                eps = 0.45
                min_samples = max(4, min_points_per_cluster)
                min_pts_thresh = 10
                voxel_xy = 0.35
                voxel_z = 0.45
                max_extent = 3.0
            elif any(k in c_name for k in ("car", "truck", "bus", "vehicle")):
                eps = 0.70
                min_samples = max(6, min_points_per_cluster)
                min_pts_thresh = 8
                voxel_xy = 0.60
                voxel_z = 0.70
                max_extent = 14.0
            elif any(k in c_name for k in ("pole", "sign", "trunk")):
                eps = 0.35
                min_samples = 3
                min_pts_thresh = 3
                voxel_xy = 0.30
                voxel_z = 0.90
                max_extent = 1.8
            elif any(k in c_name for k in ("building", "structure")):
                eps = 0.90
                min_samples = max(10, min_points_per_cluster + 4)
                min_pts_thresh = 20
                voxel_xy = 0.80
                voxel_z = 1.00
                max_extent = 60.0
            elif any(k in c_name for k in ("fence", "wall", "barrier")):
                eps = 0.75
                min_samples = max(8, min_points_per_cluster + 2)
                min_pts_thresh = 10
                voxel_xy = 0.65
                voxel_z = 0.70
                max_extent = 40.0
            elif any(k in c_name for k in ("log", "rubble")):
                eps = 0.60
                min_samples = 5
                min_pts_thresh = 5
                voxel_xy = 0.50
                voxel_z = 0.50
                max_extent = 12.0
            elif "vegetation" in c_name or "bush" in c_name or "tree" in c_name:
                eps = 0.85
                min_samples = max(12, min_points_per_cluster + 6)
                min_pts_thresh = 18
                voxel_xy = 0.75
                voxel_z = 0.90
                max_extent = 30.0
            else:
                eps = default_radius_m
                min_samples = max(10, min_points_per_cluster + 4)
                min_pts_thresh = 15
                voxel_xy = 0.75
                voxel_z = 0.80
                max_extent = 25.0

            # Downsample massive non-actor background point sets
            if not is_actor and class_pts.shape[0] > 1200:
                step = max(1, class_pts.shape[0] // 600)
                cluster_input_pts = class_pts[::step]
            else:
                cluster_input_pts = class_pts

            # 3D-Aware Spatial Voxel Connected Components with DBSCAN Fallback
            min_x = float(np.min(cluster_input_pts[:, 0]))
            min_y = float(np.min(cluster_input_pts[:, 1]))
            min_z = float(np.min(cluster_input_pts[:, 2]))

            vx = np.floor((cluster_input_pts[:, 0] - min_x) / voxel_xy).astype(np.int32)
            vy = np.floor((cluster_input_pts[:, 1] - min_y) / voxel_xy).astype(np.int32)
            vz = np.floor((cluster_input_pts[:, 2] - min_z) / voxel_z).astype(np.int32)

            nx = int(np.max(vx)) + 1
            ny = int(np.max(vy)) + 1
            nz = int(np.max(vz)) + 1

            total_voxels = nx * ny * nz
            if total_voxels <= 1_500_000:
                grid = np.zeros((nx, ny, nz), dtype=bool)
                grid[vx, vy, vz] = True
                structure = np.ones((3, 3, 3), dtype=bool)
                labeled_grid, _ = nd_label(grid, structure=structure)
                labels = labeled_grid[vx, vy, vz]
            else:
                db = DBSCAN(eps=eps, min_samples=min_samples).fit(cluster_input_pts[:, :3])
                labels = db.labels_ + 1

            valid_mask = (labels > 0)
            if not np.any(valid_mask):
                continue

            valid_pts = cluster_input_pts[valid_mask]
            valid_lbls = labels[valid_mask]
            sort_idx = np.argsort(valid_lbls)
            s_pts = valid_pts[sort_idx]
            s_lbls = valid_lbls[sort_idx]
            u_cids, u_starts, u_counts = np.unique(s_lbls, return_index=True, return_counts=True)

            c_min = np.minimum.reduceat(s_pts[:, :3], u_starts)
            c_max = np.maximum.reduceat(s_pts[:, :3], u_starts)
            c_sum = np.add.reduceat(s_pts[:, :3], u_starts)
            c_mean = c_sum / u_counts[:, None]
            c_dims = c_max - c_min

            for i in range(len(u_cids)):
                pt_count = int(u_counts[i])
                if pt_count < min_pts_thresh:
                    continue

                dx = float(c_dims[i, 0])
                dy = float(c_dims[i, 1])
                dz = float(c_dims[i, 2])
                horiz_extent = max(dx, dy)

                # Reject oversized merged clusters
                if horiz_extent > max_extent:
                    continue

                # Reject flat ground specks or undersized actor noise with strict class validation
                if any(k in c_name for k in ("bicycle", "motorcycle")):
                    if dz < 0.40 or pt_count < 10 or horiz_extent < 0.45 or horiz_extent > 3.0:
                        continue
                elif any(k in c_name for k in ("bicyclist", "motorcyclist")):
                    if dz < 0.50 or pt_count < 10 or horiz_extent < 0.35:
                        continue
                elif any(k in c_name for k in ("person", "pedestrian")):
                    if dz < 0.45 or pt_count < 5 or horiz_extent < 0.20 or horiz_extent > 1.8:
                        continue
                elif dz < 0.12 and pt_count < 15:
                    continue

                cluster_counter += 1
                cluster_pts = s_pts[u_starts[i] : u_starts[i] + pt_count]

                yaw, obb_dict = cls.calculate_pca_yaw_and_obb(cluster_pts, c_name)
                length = obb_dict["length"]
                width = obb_dict["width"]

                # Determine Extent Category
                if any(k in c_name for k in ("person", "pedestrian", "bicyclist", "motorcyclist")):
                    extent_cat = ExtentCategory.COMPACT_ACTOR
                elif any(k in c_name for k in ("pole", "traffic-sign", "trunk")) or (dz > 1.2 and horiz_extent < 1.0):
                    extent_cat = ExtentCategory.SLENDER_VERTICAL
                elif horiz_extent > 6.0 or (dx * dy * dz) > 25.0:
                    extent_cat = ExtentCategory.LARGE_VEHICLE if c_info.is_dynamic else ExtentCategory.BROAD_STRUCTURE
                elif 1.1 <= horiz_extent <= 6.0:
                    extent_cat = ExtentCategory.SMALL_VEHICLE
                else:
                    extent_cat = ExtentCategory.COMPACT_ACTOR if horiz_extent < 1.0 else ExtentCategory.IRREGULAR_CLUSTER

                # Observation Status
                if pt_count >= 30:
                    obs_status = ObservationStatus.DIRECTLY_OBSERVED
                elif pt_count >= 10:
                    obs_status = ObservationStatus.PARTIALLY_OCCLUDED
                else:
                    obs_status = ObservationStatus.SPARSE_CLUSTER

                # Preview points (if requested)
                sample_pts = LidarParser.downsample_for_preview(cluster_pts, max_points=120) if include_sample_points else []

                instance_id = f"inst_{frame_id}_{c_info.name}_{cluster_counter:02d}"

                instances.append(
                    ObjectInstance.model_construct(
                        instance_id=instance_id,
                        semantic_category=c_info.category,
                        object_type=c_info.name,
                        centroid=[round(float(c_mean[i, 0]), 3), round(float(c_mean[i, 1]), 3), round(float(c_mean[i, 2]), 3)],
                        bounding_box=BoundingBox3D.model_construct(
                            min_x=round(float(c_min[i, 0]), 3),
                            max_x=round(float(c_max[i, 0]), 3),
                            min_y=round(float(c_min[i, 1]), 3),
                            max_y=round(float(c_max[i, 1]), 3),
                            min_z=round(float(c_min[i, 2]), 3),
                            max_z=round(float(c_max[i, 2]), 3),
                        ),
                        dimensions=[round(float(max(0.15, length)), 3), round(float(max(0.15, width)), 3), round(float(max(0.15, dz)), 3)],
                        extent_category=extent_cat,
                        point_count=pt_count,
                        is_dynamic=c_info.is_dynamic,
                        source_frame_id=frame_id,
                        observation_status=obs_status,
                        sample_points=sample_pts,
                        yaw=round(float(yaw), 4),
                        oriented_bounding_box=obb_dict,
                        confidence=1.0,
                        class_id=cid,
                    )
                )

        # Merge fragmented rider-vehicle clusters and suppress duplicate detections
        instances = cls._consolidate_and_suppress_duplicates(instances, frame_id)

        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        dynamic_cnt = sum(1 for inst in instances if inst.is_dynamic)
        static_cnt = len(instances) - dynamic_cnt

        return ObjectDetectionResponse.model_construct(
            frame_id=frame_id,
            total_instances=len(instances),
            dynamic_instances_count=dynamic_cnt,
            static_instances_count=static_cnt,
            instances=instances,
            status=ProcessStatus.COMPLETE,
            created_at=now_iso,
        )

    @classmethod
    def _consolidate_and_suppress_duplicates(
        cls, instances: List[ObjectInstance], frame_id: str
    ) -> List[ObjectInstance]:
        """Merge fragmented rider-vehicle clusters and suppress duplicate bounding boxes."""
        if len(instances) <= 1:
            return instances

        merged_ids = set()
        rider_types = {"motorcyclist", "bicyclist"}
        vehicle_types = {"motorcycle", "bicycle"}

        consolidated: List[ObjectInstance] = []

        # 1. Fragmented Rider-Vehicle Merging (e.g. motorcyclist + motorcycle, bicyclist + bicycle)
        for i, inst_a in enumerate(instances):
            if inst_a.instance_id in merged_ids:
                continue

            a_type = inst_a.object_type.lower()
            if a_type in rider_types or a_type in vehicle_types:
                target_types = vehicle_types if a_type in rider_types else rider_types
                best_partner_idx = None
                best_dist = 1.6

                for j, inst_b in enumerate(instances):
                    if i == j or inst_b.instance_id in merged_ids:
                        continue
                    b_type = inst_b.object_type.lower()
                    if b_type in target_types:
                        dx = inst_a.centroid[0] - inst_b.centroid[0]
                        dy = inst_a.centroid[1] - inst_b.centroid[1]
                        dz = inst_a.centroid[2] - inst_b.centroid[2]
                        d = float(np.hypot(np.hypot(dx, dy), dz))
                        if d < best_dist:
                            best_dist = d
                            best_partner_idx = j

                if best_partner_idx is not None:
                    inst_b = instances[best_partner_idx]
                    merged_ids.add(inst_a.instance_id)
                    merged_ids.add(inst_b.instance_id)

                    rider_inst = inst_a if a_type in rider_types else inst_b
                    veh_inst = inst_b if a_type in rider_types else inst_a

                    u_min_x = min(inst_a.bounding_box.min_x, inst_b.bounding_box.min_x)
                    u_max_x = max(inst_a.bounding_box.max_x, inst_b.bounding_box.max_x)
                    u_min_y = min(inst_a.bounding_box.min_y, inst_b.bounding_box.min_y)
                    u_max_y = max(inst_a.bounding_box.max_y, inst_b.bounding_box.max_y)
                    u_min_z = min(inst_a.bounding_box.min_z, inst_b.bounding_box.min_z)
                    u_max_z = max(inst_a.bounding_box.max_z, inst_b.bounding_box.max_z)

                    total_pts = inst_a.point_count + inst_b.point_count
                    w_a = inst_a.point_count / total_pts if total_pts > 0 else 0.5
                    w_b = inst_b.point_count / total_pts if total_pts > 0 else 0.5

                    u_cx = round(float(inst_a.centroid[0] * w_a + inst_b.centroid[0] * w_b), 3)
                    u_cy = round(float(inst_a.centroid[1] * w_a + inst_b.centroid[1] * w_b), 3)
                    u_cz = round(float(inst_a.centroid[2] * w_a + inst_b.centroid[2] * w_b), 3)

                    u_len = round(float(max(0.2, u_max_x - u_min_x)), 3)
                    u_wid = round(float(max(0.2, u_max_y - u_min_y)), 3)
                    u_hgt = round(float(max(0.2, u_max_z - u_min_z)), 3)

                    merged_obb = dict(rider_inst.oriented_bounding_box or {})
                    merged_obb["center"] = [u_cx, u_cy, u_cz]
                    merged_obb["size"] = [u_len, u_wid, u_hgt]
                    merged_obb["length"] = u_len
                    merged_obb["width"] = u_wid
                    merged_obb["height"] = u_hgt

                    combined_sample = (inst_a.sample_points or []) + (inst_b.sample_points or [])

                    merged_instance = ObjectInstance.model_construct(
                        instance_id=rider_inst.instance_id,
                        semantic_category=rider_inst.semantic_category,
                        object_type=rider_inst.object_type,
                        centroid=[u_cx, u_cy, u_cz],
                        bounding_box=BoundingBox3D.model_construct(
                            min_x=u_min_x, max_x=u_max_x,
                            min_y=u_min_y, max_y=u_max_y,
                            min_z=u_min_z, max_z=u_max_z,
                        ),
                        dimensions=[u_len, u_wid, u_hgt],
                        extent_category=rider_inst.extent_category,
                        point_count=total_pts,
                        is_dynamic=True,
                        source_frame_id=frame_id,
                        observation_status=ObservationStatus.DIRECTLY_OBSERVED,
                        sample_points=combined_sample[:120],
                        yaw=rider_inst.yaw,
                        oriented_bounding_box=merged_obb,
                        confidence=1.0,
                        class_id=rider_inst.class_id,
                    )
                    consolidated.append(merged_instance)
                    continue

            if inst_a.instance_id not in merged_ids:
                consolidated.append(inst_a)

        # 2. Duplicate Suppression: Remove redundant overlapping same-category clusters
        final_instances: List[ObjectInstance] = []
        suppressed_ids = set()

        for i, inst_a in enumerate(consolidated):
            if inst_a.instance_id in suppressed_ids:
                continue

            for j in range(i + 1, len(consolidated)):
                inst_b = consolidated[j]
                if inst_b.instance_id in suppressed_ids:
                    continue

                if inst_a.semantic_category == inst_b.semantic_category:
                    dx = inst_a.centroid[0] - inst_b.centroid[0]
                    dy = inst_a.centroid[1] - inst_b.centroid[1]
                    dz = inst_a.centroid[2] - inst_b.centroid[2]
                    dist_3d = float(np.hypot(np.hypot(dx, dy), dz))

                    if dist_3d < 0.85:
                        if inst_b.point_count <= inst_a.point_count:
                            suppressed_ids.add(inst_b.instance_id)
                        else:
                            suppressed_ids.add(inst_a.instance_id)
                            break

            if inst_a.instance_id not in suppressed_ids:
                final_instances.append(inst_a)

        return final_instances

    @staticmethod
    def calculate_pca_yaw_and_obb(cluster_pts: np.ndarray, class_name: str = "vehicle") -> Tuple[float, Dict[str, Any]]:
        """Compute search-based Minimum Bounding Rectangle (MBR / L-Shape) heading and tight 3D OBB.
        
        Solves PCA diagonal misalignment by searching for the bounding box orientation that
        minimizes area and maximizes edge point alignment on the cluster's 2D horizontal footprint.
        """
        pt_count = len(cluster_pts)
        c_min = np.min(cluster_pts, axis=0)
        c_max = np.max(cluster_pts, axis=0)
        dx = float(c_max[0] - c_min[0])
        dy = float(c_max[1] - c_min[1])
        dz = float(c_max[2] - c_min[2])
        c_mean = np.mean(cluster_pts, axis=0)

        is_oriented_class = any(k in class_name.lower() for k in (
            "car", "truck", "bus", "vehicle", "log", "barrier",
            "bicycle", "motorcycle", "person", "pedestrian"
        ))

        if pt_count >= 6 and is_oriented_class:
            xy_pts = cluster_pts[:, :2]
            mean_xy = np.mean(xy_pts, axis=0)
            centered_xy = xy_pts - mean_xy

            # Downsample large clusters for sub-millisecond MBR search
            if centered_xy.shape[0] > 120:
                step = max(1, centered_xy.shape[0] // 80)
                search_pts = centered_xy[::step]
            else:
                search_pts = centered_xy

            # Search 45 candidate angles over [0, pi/2) at 2-degree resolution
            num_angles = 45
            thetas = np.linspace(0.0, np.pi / 2.0, num_angles, endpoint=False)
            cos_t = np.cos(thetas)
            sin_t = np.sin(thetas)

            # (N, 1) * (45,) - (N, 1) * (45,) -> (N, 45)
            rot_x = search_pts[:, 0:1] * cos_t - search_pts[:, 1:2] * sin_t
            rot_y = search_pts[:, 0:1] * sin_t + search_pts[:, 1:2] * cos_t

            min_rx = np.min(rot_x, axis=0)
            max_rx = np.max(rot_x, axis=0)
            min_ry = np.min(rot_y, axis=0)
            max_ry = np.max(rot_y, axis=0)

            span_x = max_rx - min_rx
            span_y = max_ry - min_ry
            areas = span_x * span_y

            best_idx = int(np.argmin(areas))
            best_theta = float(thetas[best_idx])
            best_len = float(span_x[best_idx])
            best_wid = float(span_y[best_idx])
            best_min_rx = float(min_rx[best_idx])
            best_max_rx = float(max_rx[best_idx])
            best_min_ry = float(min_ry[best_idx])
            best_max_ry = float(max_ry[best_idx])

            # For vehicles, align heading with the principal axis of travel (length >= width)
            if best_len < best_wid and any(k in class_name.lower() for k in ("car", "truck", "bus", "vehicle", "log")):
                best_len, best_wid = best_wid, best_len
                best_theta = (best_theta + np.pi / 2.0) % (2.0 * np.pi)
                if best_theta > np.pi:
                    best_theta -= 2.0 * np.pi
                center_rx = (best_min_ry + best_max_ry) / 2.0
                center_ry = -(best_min_rx + best_max_rx) / 2.0
            else:
                center_rx = (best_min_rx + best_max_rx) / 2.0
                center_ry = (best_min_ry + best_max_ry) / 2.0

            # Re-project center to world coordinates
            cos_b = np.cos(best_theta)
            sin_b = np.sin(best_theta)
            obb_center_xy = mean_xy + np.array([center_rx * cos_b - center_ry * sin_b, center_rx * sin_b + center_ry * cos_b])

            yaw = best_theta
            length = best_len
            width = best_wid
        elif pt_count >= 3 and is_oriented_class:
            xy_pts = cluster_pts[:, :2]
            mean_xy = np.mean(xy_pts, axis=0)
            centered_xy = xy_pts - mean_xy
            cov = np.cov(centered_xy, rowvar=False)
            if cov.shape == (2, 2) and not np.any(np.isnan(cov)):
                eig_vals, eig_vecs = np.linalg.eigh(cov)
                major_vec = eig_vecs[:, 1]
                yaw = float(np.arctan2(major_vec[1], major_vec[0]))
                length = dx
                width = dy
                obb_center_xy = mean_xy
            else:
                yaw = 0.0
                length = dx
                width = dy
                obb_center_xy = np.array([c_mean[0], c_mean[1]])
        else:
            yaw = 0.0
            length = dx
            width = dy
            obb_center_xy = np.array([c_mean[0], c_mean[1]])

        obb_center_z = float(c_mean[2])
        obb_dict = {
            "center": [round(float(obb_center_xy[0]), 3), round(float(obb_center_xy[1]), 3), round(float(obb_center_z), 3)],
            "size": [round(float(max(0.15, length)), 3), round(float(max(0.15, width)), 3), round(float(max(0.15, dz)), 3)],
            "yaw": round(float(yaw), 4),
            "center_x": round(float(obb_center_xy[0]), 3),
            "center_y": round(float(obb_center_xy[1]), 3),
            "center_z": round(float(obb_center_z), 3),
            "length": round(float(max(0.15, length)), 3),
            "width": round(float(max(0.15, width)), 3),
            "height": round(float(max(0.15, dz)), 3),
        }
        return yaw, obb_dict


def calculate_pca_yaw_and_obb(cluster_pts: np.ndarray, class_name: str = "vehicle") -> Tuple[float, Dict[str, Any]]:
    """Convenience helper for calculating PCA yaw and tight OBB."""
    return InstanceClusteringService.calculate_pca_yaw_and_obb(cluster_pts, class_name)


def cluster_points_to_instances_3d(points: np.ndarray, semantic_classes: np.ndarray, dataset: Optional[str] = None) -> List[ObjectInstance]:
    """Convenience helper for clustering points directly to ObjectInstance list."""
    resp = InstanceClusteringService.detect_objects(
        frame_id="000000",
        points=points,
        semantic_classes=semantic_classes,
        dataset=dataset,
    )
    return resp.instances

