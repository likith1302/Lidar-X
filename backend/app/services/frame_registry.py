"""Central authoritative registry and artifact manager for canonical LiDAR frames."""

import os
import json
import re
import uuid
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from ..config import settings
from ..models.pipeline_schemas import PipelineStatusResponse


class FrameRegistry:
    """Central authority for canonical frame ID resolution, artifact paths, atomic persistence, and pipeline status."""

    @classmethod
    def canonicalize_frame_id(cls, filename_or_id: str) -> str:
        """
        Derive canonical frame ID from filename or raw identifier.
        e.g.:
          - "000000.bin" -> "000000"
          - "000001.label" -> "000001"
          - "scan_000002.bin" -> "000002"
          - "FRAME_0001 (Mock)" -> "FRAME_0001"
          - "frame_123456" -> "123456" (or clean identifier)
        """
        if not filename_or_id:
            return "000000"

        clean = str(filename_or_id).strip()
        # Remove known extensions
        clean = re.sub(r'\.(bin|label|npz|json|pcd|las|laz)$', '', clean, flags=re.IGNORECASE)

        # Match pure numeric stems like '000000' or '000123'
        if re.match(r'^\d+$', clean):
            return clean

        # If has prefix like 'scan_000000' or 'frame_000000', extract number if present
        m = re.match(r'^(?:scan|frame|seq_frame)_(\d+)$', clean, flags=re.IGNORECASE)
        if m:
            return m.group(1)

        # Clean generic identifier to safe characters
        sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', clean)
        return sanitized if sanitized else "000000"

    # --- Path Resolvers ---

    @classmethod
    def get_frame_dir(cls, frame_id: str) -> Path:
        fid = cls.canonicalize_frame_id(frame_id)
        settings.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        return settings.FRAMES_DIR / fid

    @classmethod
    def get_points_path(cls, frame_id: str) -> Path:
        return cls.get_frame_dir(frame_id) / "points.npz"

    @classmethod
    def get_frame_meta_path(cls, frame_id: str) -> Path:
        return cls.get_frame_dir(frame_id) / "metadata.json"

    @classmethod
    def get_labels_path(cls, frame_id: str) -> Path:
        fid = cls.canonicalize_frame_id(frame_id)
        settings.LABELS_DIR.mkdir(parents=True, exist_ok=True)
        return settings.LABELS_DIR / f"{fid}_labels.npz"

    @classmethod
    def get_prediction_label_path(cls, frame_id: str) -> Path:
        fid = cls.canonicalize_frame_id(frame_id)
        settings.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        return settings.PREDICTIONS_DIR / f"{fid}.label"

    @classmethod
    def get_terrain_path(cls, frame_id: str) -> Path:
        fid = cls.canonicalize_frame_id(frame_id)
        settings.TERRAIN_DIR.mkdir(parents=True, exist_ok=True)
        return settings.TERRAIN_DIR / f"{fid}.json"

    @classmethod
    def get_objects_path(cls, frame_id: str) -> Path:
        fid = cls.canonicalize_frame_id(frame_id)
        settings.OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
        return settings.OBJECTS_DIR / f"{fid}.json"

    @classmethod
    def get_map_path(cls, map_id_or_frame: str) -> Path:
        settings.MAPS_DIR.mkdir(parents=True, exist_ok=True)
        return settings.MAPS_DIR / f"{map_id_or_frame}.json"

    # --- Atomic File Persistence ---

    @classmethod
    def _safe_atomic_replace(cls, temp_file: Path, target_path: Path) -> Path:
        """Replace temp_file to target_path atomically with Windows retry and fallback."""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                temp_file.replace(target_path)
                return target_path
            except (PermissionError, OSError) as e:
                if attempt == max_attempts - 1:
                    try:
                        shutil.copyfile(temp_file, target_path)
                        temp_file.unlink(missing_ok=True)
                        return target_path
                    except Exception:
                        raise e
                time.sleep(0.02 * (attempt + 1))
        return target_path

    @classmethod
    def atomic_save_json(cls, target_path: Path, data: Any) -> Path:
        """Write JSON atomically via temporary file and atomic replace."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target_path.with_name(f"{target_path.name}.tmp.{uuid.uuid4().hex}")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            cls._safe_atomic_replace(temp_file, target_path)
            return target_path
        finally:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)

    @classmethod
    def atomic_save_npz(cls, target_path: Path, **arrays: np.ndarray) -> Path:
        """Write compressed NPZ atomically via temporary file and atomic replace."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target_path.with_name(f"{target_path.name}.tmp.{uuid.uuid4().hex}")
        try:
            with open(temp_file, "wb") as f:
                np.savez_compressed(f, **arrays)
            cls._safe_atomic_replace(temp_file, target_path)
            return target_path
        finally:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)

    @classmethod
    def atomic_save_bytes(cls, target_path: Path, raw_bytes: bytes) -> Path:
        """Write binary bytes atomically via temporary file and atomic replace."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target_path.with_name(f"{target_path.name}.tmp.{uuid.uuid4().hex}")
        try:
            with open(temp_file, "wb") as f:
                f.write(raw_bytes)
            cls._safe_atomic_replace(temp_file, target_path)
            return target_path
        finally:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)

    # --- Truthful Pipeline Gate Status ---

    @classmethod
    def get_pipeline_status(cls, frame_id: str) -> PipelineStatusResponse:
        """Query real persisted artifacts for frame_id and return truthful pipeline status."""
        fid = cls.canonicalize_frame_id(frame_id)
        pts_path = cls.get_points_path(fid)
        labels_path = cls.get_labels_path(fid)
        pred_path = cls.get_prediction_label_path(fid)
        terrain_path = cls.get_terrain_path(fid)
        objects_path = cls.get_objects_path(fid)
        map_path = cls.get_map_path(fid)
        live_map_path = cls.get_map_path("live_fovea_map_01")

        frame_exists = pts_path.exists()
        point_count = 0
        if frame_exists:
            try:
                with np.load(pts_path) as data:
                    point_count = int(data["points"].shape[0])
            except Exception:
                frame_exists = False

        # Semantic Labels
        labels_available = labels_path.exists() or pred_path.exists()
        semantic_label_count = 0
        semantic_source = None
        if labels_path.exists():
            try:
                with np.load(labels_path) as ldata:
                    semantic_label_count = int(ldata["semantic_classes"].shape[0])
                    semantic_source = "backend_labels_npz"
            except Exception:
                labels_available = False

        if not labels_path.exists() and pred_path.exists():
            try:
                raw_b = pred_path.read_bytes()
                arr = np.frombuffer(raw_b, dtype=np.uint32)
                semantic_label_count = len(arr)
                semantic_source = "salsanext_prediction_binary"
                labels_available = True
            except Exception:
                labels_available = False

        labels_match_points = frame_exists and labels_available and (semantic_label_count == point_count) and point_count > 0

        # Terrain
        terrain_available = terrain_path.exists()

        # Objects
        object_detection_available = objects_path.exists()
        object_instance_count = 0
        if object_detection_available:
            try:
                with open(objects_path, "r", encoding="utf-8") as f:
                    obj_dict = json.load(f)
                    object_instance_count = int(obj_dict.get("total_instances", len(obj_dict.get("instances", []))))
            except Exception:
                object_detection_available = False

        # Adaptive Map
        adaptive_map_available = False
        adaptive_grid_cell_count = 0
        map_mode = None

        target_map = map_path if map_path.exists() else live_map_path if live_map_path.exists() else None
        if target_map and target_map.exists():
            try:
                with open(target_map, "r", encoding="utf-8") as f:
                    mdict = json.load(f)
                    adaptive_map_available = True
                    metadata = mdict.get("metadata", {})
                    adaptive_grid_cell_count = metadata.get("total_cells", len(mdict.get("cells", [])))
                    map_mode = metadata.get("mode", mdict.get("mode", "local_only"))
            except Exception:
                adaptive_map_available = False
        else:
            try:
                from .map_fusion import MapFusionService
                in_mem = MapFusionService.get_map(fid) or MapFusionService.get_map("live_fovea_map_01")
                if in_mem:
                    meta, cells, _ = in_mem
                    adaptive_map_available = True
                    adaptive_grid_cell_count = len(cells)
                    map_mode = meta.mode.value if hasattr(meta.mode, "value") else str(meta.mode)
            except Exception:
                pass

        return PipelineStatusResponse(
            frame_id=fid,
            frame_exists=frame_exists,
            point_cloud_available=frame_exists,
            point_count=point_count,
            semantic_labels_available=labels_available,
            semantic_label_count=semantic_label_count,
            labels_match_point_count=labels_match_points,
            semantic_source=semantic_source,
            inference_state="ready" if labels_available else "not_run",
            terrain_available=terrain_available,
            object_detection_available=object_detection_available,
            object_instance_count=object_instance_count,
            adaptive_map_available=adaptive_map_available,
            adaptive_grid_cell_count=adaptive_grid_cell_count,
            map_mode=map_mode,
            last_error=None if (not frame_exists or labels_match_points) else (
                f"Label count ({semantic_label_count}) != point count ({point_count})" if labels_available else None
            ),
        )
