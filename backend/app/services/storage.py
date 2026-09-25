"""Local file-based storage service for LiDAR frames, terrain, semantic labels, and objects."""

import json
import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict
import numpy as np

from ..config import settings
from ..models.schemas import (
    FrameMetadata,
    FrameDetailsResponse,
    TerrainAnalysisResponse,
)
from ..models.object_schemas import (
    ObjectDetectionResponse,
    TrackingUpdateResponse,
    TrackedObject,
)
from .lidar_parser import LidarParser
from .frame_registry import FrameRegistry


class StorageService:
    """Manages persistence of frames, point cloud arrays, labels, objects, and tracks with bounded LRU memory cache."""

    _cache_lock = threading.Lock()
    _points_cache: OrderedDict[str, np.ndarray] = OrderedDict()
    _labels_cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray]] = OrderedDict()
    _MAX_CACHE_SIZE = 16

    @classmethod
    def _cache_put_points(cls, frame_id: str, points: np.ndarray) -> None:
        with cls._cache_lock:
            cls._points_cache[frame_id] = points.copy()
            cls._points_cache.move_to_end(frame_id)
            if len(cls._points_cache) > cls._MAX_CACHE_SIZE:
                cls._points_cache.popitem(last=False)

    @classmethod
    def _cache_get_points(cls, frame_id: str) -> Optional[np.ndarray]:
        with cls._cache_lock:
            if frame_id in cls._points_cache:
                cls._points_cache.move_to_end(frame_id)
                return cls._points_cache[frame_id].copy()
            return None

    @classmethod
    def _cache_put_labels(cls, frame_id: str, labels: np.ndarray, instances: np.ndarray) -> None:
        with cls._cache_lock:
            cls._labels_cache[frame_id] = (labels.copy(), instances.copy())
            cls._labels_cache.move_to_end(frame_id)
            if len(cls._labels_cache) > cls._MAX_CACHE_SIZE:
                cls._labels_cache.popitem(last=False)

    @classmethod
    def _cache_get_labels(cls, frame_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        with cls._cache_lock:
            if frame_id in cls._labels_cache:
                cls._labels_cache.move_to_end(frame_id)
                lbl, inst = cls._labels_cache[frame_id]
                return lbl.copy(), inst.copy()
            return None

    @classmethod
    def save_frame(
        cls,
        frame_id: str,
        points: np.ndarray,
        metadata: FrameMetadata,
        overwrite: bool = True,
    ) -> Path:
        """Save a frame's point cloud array (NPZ) and metadata (JSON) atomically with canonical ID."""
        fid = FrameRegistry.canonicalize_frame_id(frame_id)
        frame_dir = FrameRegistry.get_frame_dir(fid)
        npz_path = FrameRegistry.get_points_path(fid)
        meta_path = FrameRegistry.get_frame_meta_path(fid)

        if not overwrite and npz_path.exists() and meta_path.exists():
            return npz_path

        # Invalidate old downstream perception artifacts so fresh pipelines execute on new data
        if overwrite:
            try:
                FrameRegistry.get_terrain_path(fid).unlink(missing_ok=True)
                FrameRegistry.get_objects_path(fid).unlink(missing_ok=True)
                FrameRegistry.get_map_path(fid).unlink(missing_ok=True)
                FrameRegistry.get_labels_path(fid).unlink(missing_ok=True)
                FrameRegistry.get_prediction_label_path(fid).unlink(missing_ok=True)
            except Exception:
                pass

        FrameRegistry.atomic_save_npz(npz_path, points=points)
        FrameRegistry.atomic_save_json(meta_path, metadata.model_dump())
        cls._cache_put_points(fid, points)

        return npz_path

    @classmethod
    def alias_frame(
        cls,
        source_frame_id: str,
        target_frame_id: str,
        metadata: Optional[FrameMetadata] = None,
    ) -> Optional[Path]:
        """Alias or copy saved frame artifacts to another frame ID without recompression."""
        src_fid = FrameRegistry.canonicalize_frame_id(source_frame_id)
        tgt_fid = FrameRegistry.canonicalize_frame_id(target_frame_id)
        if src_fid == tgt_fid:
            return FrameRegistry.get_points_path(src_fid)

        src_npz = FrameRegistry.get_points_path(src_fid)
        tgt_npz = FrameRegistry.get_points_path(tgt_fid)
        if src_npz.exists():
            tgt_npz.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_npz, tgt_npz)

        if metadata is not None:
            tgt_meta = FrameRegistry.get_frame_meta_path(tgt_fid)
            FrameRegistry.atomic_save_json(tgt_meta, metadata.model_dump())

        # Propagate cache entry
        cached_pts = cls._cache_get_points(src_fid)
        if cached_pts is not None:
            cls._cache_put_points(tgt_fid, cached_pts)

        return tgt_npz

    @classmethod
    def _resolve_frame_id(cls, frame_id: str) -> str:
        """Resolve frame ID to stored canonical format with fallback for mock/demo requests."""
        fid = FrameRegistry.canonicalize_frame_id(frame_id)
        if (FrameRegistry.get_points_path(fid)).exists():
            return fid

        # Try matching 6-digit padded number
        if fid.isdigit():
            padded = f"{int(fid):06d}"
            if (FrameRegistry.get_points_path(padded)).exists():
                return padded

        # Try checking .bin directory variations in FRAMES_DIR
        if (settings.FRAMES_DIR / f"{fid}.bin" / "points.npz").exists():
            return f"{fid}.bin"

        # Only fall back to first stored frame for explicit demo/sample aliases
        if frame_id in ("demo", "sample", "default"):
            all_frames = cls.list_frames()
            if all_frames:
                if "000000" in all_frames:
                    return "000000"
                return all_frames[0]

        return fid

    @classmethod
    def get_frame_points(cls, frame_id: str) -> Optional[np.ndarray]:
        """Load points array from storage using canonical frame ID with memory cache."""
        fid = cls._resolve_frame_id(frame_id)
        cached = cls._cache_get_points(fid)
        if cached is not None:
            return cached

        npz_path = FrameRegistry.get_points_path(fid)
        if not npz_path.exists():
            return None

        try:
            with np.load(npz_path) as data:
                pts = data["points"].copy()
                cls._cache_put_points(fid, pts)
                return pts
        except Exception:
            return None


    @classmethod
    def get_frame_details(cls, frame_id: str, sample_limit: int = 4000) -> Optional[FrameDetailsResponse]:
        """Load frame metadata and sampled points for UI preview."""
        fid = cls._resolve_frame_id(frame_id)
        npz_path = FrameRegistry.get_points_path(fid)
        meta_path = FrameRegistry.get_frame_meta_path(fid)

        if not npz_path.exists() or not meta_path.exists():
            if "Mock" in frame_id or frame_id.startswith("FRAME_"):
                from datetime import datetime, timezone
                x = np.linspace(-10, 10, 20)
                y = np.linspace(-10, 10, 20)
                xx, yy = np.meshgrid(x, y)
                pts = np.zeros((xx.size, 4), dtype=np.float32)
                pts[:, 0] = xx.ravel()
                pts[:, 1] = yy.ravel()
                pts[:, 2] = -1.73
                pts[:, 3] = 0.5
                bounds = LidarParser.compute_bounds(pts)
                sample_pts = LidarParser.downsample_for_preview(pts, max_points=sample_limit)
                meta = FrameMetadata(
                    scan_source="Synthetic Mock LiDAR Scan",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    format="SemanticKITTI float32 (x,y,z,i)",
                    point_count=int(pts.shape[0]),
                    file_size_bytes=pts.nbytes,
                    coordinate_frame="Sensor Origin (Ego Body Frame)",
                )
                return FrameDetailsResponse(
                    frame_id=fid,
                    point_count=int(pts.shape[0]),
                    file_size_bytes=pts.nbytes,
                    bounds=bounds,
                    metadata=meta,
                    sample_points=sample_pts,
                    created_at=meta.timestamp,
                )
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)
        except Exception:
            return None

        points = cls.get_frame_points(fid)
        if points is None:
            return None

        bounds = LidarParser.compute_bounds(points)
        sample_pts = LidarParser.downsample_for_preview(points, max_points=sample_limit)

        metadata = FrameMetadata(**meta_dict)
        return FrameDetailsResponse(
            frame_id=fid,
            point_count=int(points.shape[0]),
            file_size_bytes=meta_dict.get("file_size_bytes", 0),
            bounds=bounds,
            metadata=metadata,
            sample_points=sample_pts,
            created_at=meta_dict.get("timestamp", ""),
        )

    @classmethod
    def list_frames(cls) -> List[str]:
        """List all stored frame IDs."""
        if not settings.FRAMES_DIR.exists():
            return []
        return [
            d.name for d in sorted(settings.FRAMES_DIR.iterdir())
            if d.is_dir() and (d / "points.npz").exists()
        ]


    # --- Terrain Analysis Storage ---

    @classmethod
    def save_terrain_result(cls, response: TerrainAnalysisResponse) -> Path:
        """Save a terrain analysis response JSON atomically."""
        fid = FrameRegistry.canonicalize_frame_id(response.frame_id)
        result_path = FrameRegistry.get_terrain_path(fid)
        return FrameRegistry.atomic_save_json(result_path, response.model_dump())

    @classmethod
    def get_terrain_result(cls, frame_id: str) -> Optional[TerrainAnalysisResponse]:
        """Load a terrain analysis response JSON."""
        fid = cls._resolve_frame_id(frame_id)
        result_path = FrameRegistry.get_terrain_path(fid)
        if not result_path.exists():
            return None

        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TerrainAnalysisResponse(**data)
        except Exception:
            return None

    # --- Semantic Labels Storage ---

    @classmethod
    def save_labels(
        cls,
        frame_id: str,
        semantic_classes: np.ndarray,
        instance_ids: np.ndarray,
        overwrite: bool = True,
    ) -> Path:
        """Save semantic labels and instance IDs array (NPZ) atomically."""
        fid = FrameRegistry.canonicalize_frame_id(frame_id)
        npz_path = FrameRegistry.get_labels_path(fid)
        if not overwrite and npz_path.exists():
            return npz_path
        res = FrameRegistry.atomic_save_npz(
            npz_path,
            semantic_classes=semantic_classes,
            instance_ids=instance_ids,
        )
        cls._cache_put_labels(fid, semantic_classes, instance_ids)
        return res

    @classmethod
    def alias_labels(
        cls,
        source_frame_id: str,
        target_frame_id: str,
    ) -> Optional[Path]:
        """Alias or copy saved labels artifact to another frame ID without recompression."""
        src_fid = FrameRegistry.canonicalize_frame_id(source_frame_id)
        tgt_fid = FrameRegistry.canonicalize_frame_id(target_frame_id)
        if src_fid == tgt_fid:
            return FrameRegistry.get_labels_path(src_fid)

        src_npz = FrameRegistry.get_labels_path(src_fid)
        tgt_npz = FrameRegistry.get_labels_path(tgt_fid)
        if src_npz.exists():
            tgt_npz.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_npz, tgt_npz)

        cached_lbl = cls._cache_get_labels(src_fid)
        if cached_lbl is not None:
            cls._cache_put_labels(tgt_fid, cached_lbl[0], cached_lbl[1])

        return tgt_npz

    @classmethod
    def get_labels(cls, frame_id: str, allow_precomputed_fallback: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load semantic labels and instance IDs for a frame with memory cache."""
        fid = cls._resolve_frame_id(frame_id)
        cached = cls._cache_get_labels(fid)
        if cached is not None:
            return cached

        npz_path = FrameRegistry.get_labels_path(fid)
        if npz_path.exists():
            try:
                with np.load(npz_path) as data:
                    sem = data["semantic_classes"].copy()
                    inst = data["instance_ids"].copy()
                    cls._cache_put_labels(fid, sem, inst)
                    return sem, inst
            except Exception:
                pass


        if allow_precomputed_fallback:
            # Fallback to reading uint32 prediction binary only when explicitly permitted
            pred_path = FrameRegistry.get_prediction_label_path(fid)
            if pred_path.exists():
                try:
                    raw_bytes = pred_path.read_bytes()
                    labels_uint32 = np.frombuffer(raw_bytes, dtype=np.uint32)
                    sem_classes = (labels_uint32 & 0xFFFF).astype(np.uint16)
                    inst_ids = ((labels_uint32 >> 16) & 0xFFFF).astype(np.uint16)
                    return sem_classes, inst_ids
                except Exception:
                    pass

        return None


    # --- Object Instances Storage ---

    @classmethod
    def save_objects(cls, response: ObjectDetectionResponse) -> Path:
        """Save detected object instances JSON atomically."""
        fid = FrameRegistry.canonicalize_frame_id(response.frame_id)
        obj_path = FrameRegistry.get_objects_path(fid)
        return FrameRegistry.atomic_save_json(obj_path, response.model_dump())

    @classmethod
    def get_objects(cls, frame_id: str) -> Optional[ObjectDetectionResponse]:
        """Load detected object instances JSON."""
        fid = FrameRegistry.canonicalize_frame_id(frame_id)
        obj_path = FrameRegistry.get_objects_path(fid)
        if not obj_path.exists():
            return None

        try:
            with open(obj_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ObjectDetectionResponse(**data)
        except Exception:
            return None

    # --- 2.5D Adaptive Map Storage ---

    @classmethod
    def save_map(cls, map_export: Any) -> Path:
        """Save exported 2.5D map JSON atomically."""
        map_path = FrameRegistry.get_map_path(map_export.map_id)
        return FrameRegistry.atomic_save_json(map_path, map_export.model_dump())

    @classmethod
    def get_map(cls, map_id: str) -> Optional[Dict[str, Any]]:
        """Load 2.5D map dictionary."""
        map_path = FrameRegistry.get_map_path(map_id)
        if not map_path.exists():
            return None
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

