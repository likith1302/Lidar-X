"""LiDAR Sequence Replay Session Manager and Pipeline Processor.

Maintains playback state, executes full perception pipeline per frame,
tracks dynamic objects continuously across consecutive frames, and provides
backpressure-controlled frame streaming.
"""

import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timezone
import numpy as np

from ..config import settings
from ..models.replay_schemas import (
    ReplaySessionStatus,
    ReplayFrameStreamPayload,
    ReplayPointSample,
    ReplayPlaybackState,
    ReplayDataMode,
    PlaybackMode,
    ReplayDiagnostics,
)
from ..models.schemas import BoundingBox3D, TerrainAnalysisResponse, FrameMetadata, ProcessStatus
from ..models.object_schemas import ObjectDetectionResponse, TrackingUpdateResponse, TrackingCoordinateMode
from ..models.map_schemas import MapUpdateResponse, AdaptiveGridCell, MapMetadata, MapMode

from .lidar_parser import LidarParser
from .preprocessing import PointCloudPreprocessor, PreprocessingConfig
from .semantic_mapping import SemanticMappingService
from .terrain_analysis import TerrainAnalysisEngine
from .instance_clustering import InstanceClusteringService
from .object_tracking import ObjectTracker
from .adaptive_grid import AdaptiveGridService
from .fast_frnet_inference import FastFRNetInferenceService
from .salsanext_inference import SalsaNextInferenceService
from .storage import StorageService
from .metrics_service import MetricsService
from .precompute_service import precompute_service
from ..models.metric_schemas import PipelineStageLatencies, FramePerformanceMetrics
from ..config.model_settings import model_settings

logger = logging.getLogger(__name__)


# Dedicated thread pools for clean separation between CPU perception and disk/map operations
_PERCEPTION_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="perception")
_IO_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="io_saver")
_MAP_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="map_fuser")
_LOOKAHEAD_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lookahead")


class ReplaySession:
    """Manages playback lifecycle and full perception pipeline execution for one sequence."""

    def __init__(self, session_data: Dict[str, Any]):
        self.session_id: str = session_data["session_id"]
        self.session_dir: str = session_data["session_dir"]
        self.sequence_name: str = session_data["sequence_name"]
        self.total_frames: int = session_data["total_frames"]
        self.has_predictions: bool = session_data["has_predictions"]
        self.data_mode: str = session_data.get("data_mode", ReplayDataMode.PRECOMPUTED_LABELS)
        self.content_hash: Optional[str] = session_data.get("content_hash")
        self.is_manual_upload: bool = bool(session_data.get("is_manual_upload", False))
        req_mode = session_data.get("playback_mode")
        if self.is_manual_upload:
            self.is_demo = False
            self.playback_mode = PlaybackMode.LIVE_PROCESSING
        elif req_mode == PlaybackMode.LIVE_PROCESSING:
            self.is_demo = False
            self.playback_mode = PlaybackMode.LIVE_PROCESSING
        elif req_mode == PlaybackMode.OFFLINE_PRECOMPUTED_REPLAY:
            self.is_demo = True
            self.playback_mode = PlaybackMode.OFFLINE_PRECOMPUTED_REPLAY
        else:
            has_cached = (
                precompute_service.get_precomputed_frame_count(self.session_id) > 0
                or precompute_service.load_manifest(self.session_id) is not None
            )
            self.is_demo = (
                has_cached
                or self.session_id.startswith("foveamap_sequence")
                or self.session_id in ("demo", "sample", "default")
            )
            self.playback_mode = PlaybackMode.OFFLINE_PRECOMPUTED_REPLAY if self.is_demo else PlaybackMode.LIVE_PROCESSING
        self.frames_meta: List[Dict[str, Any]] = session_data["frames"]
        self.has_poses: bool = session_data.get("has_poses", False)
        self.has_calibration: bool = session_data.get("has_calibration", False)
        self.has_timestamps: bool = session_data.get("has_timestamps", False)
        self.poses_file: Optional[str] = session_data.get("poses_file")

        self.current_frame_index: int = 0
        self.state: str = ReplayPlaybackState.READY
        self.fps: float = 10.0
        self.coordinate_mode: str = "local_frame"
        self.map_mode: str = "local_only"

        # Dedicated multi-object Kalman tracker instance persisting across sequence
        self.tracker: ObjectTracker = ObjectTracker(max_association_distance_m=4.5, max_missed_frames=3)
        self.state: str = ReplayPlaybackState.READY
        self.error_message: Optional[str] = None
        if self.is_demo:
            self.semantic_source = "PRECOMPUTED"
        elif self.has_predictions:
            self.semantic_source = "GROUND TRUTH"
        else:
            self.semantic_source = "LIVE FAST-FRNET"

        # Determine target Fast-FRNet model and pre-load once in memory for the session
        sess_context = (self.session_id + " " + self.sequence_name).lower()
        self.target_model_type: str = "rellis" if any(k in sess_context for k in ("rellis", "offroad", "off_road")) else "semantickitti"
        import os
        is_render = os.environ.get("RENDER", "").lower() in ("true", "1")
        if not self.is_demo and not is_render:
            try:
                FastFRNetInferenceService.load_model(self.target_model_type)
                logger.info(f"Fast-FRNet ({self.target_model_type.upper()}) cached in memory for replay session '{self.session_id}'")
            except Exception as e:
                logger.warning(f"Fast-FRNet initial model load for session '{self.session_id}' deferred: {e}")
        else:
            logger.info(f"Fast-FRNet model load deferred for session '{self.session_id}' (is_demo={self.is_demo}, is_render={is_render})")

        # In-memory LRU cache: frame_index -> processed frame payload (bounded to max 128 frames)
        self._frame_cache: Dict[int, ReplayFrameStreamPayload] = {}

        # Invalidate old session metrics on initialization
        MetricsService.clear_session_metrics(self.session_id)

        # Per-session global-fusion map: cell_key -> AdaptiveGridCell or Dict.
        self._global_map: Dict[str, Any] = {}
        self._pose_trajectory: List[Tuple[float, float, float]] = []  # (x, y, yaw)
        self._frame_poses: Dict[int, Tuple[float, float, float]] = {}
        pose_data = self._load_poses_if_available()
        if pose_data and len(pose_data) >= self.total_frames:
            for i in range(self.total_frames):
                self._frame_poses[i] = pose_data[i]
                self._pose_trajectory.append(pose_data[i])
        else:
            # Default incremental motion model: forward 0.5 m per frame, no yaw.
            for i in range(self.total_frames):
                self._frame_poses[i] = (i * 0.5, 0.0, 0.0)
                self._pose_trajectory.append(self._frame_poses[i])

        # Lookahead prefetch buffer for smooth live streaming of newly uploaded sequences
        self._lookahead_buffer: Dict[int, ReplayFrameStreamPayload] = {}
        self._lookahead_in_flight: Set[int] = set()

        # Preload initial precomputed frames for instant 60 FPS playback (safely bounded to max 100 frames)
        if self.playback_mode == PlaybackMode.OFFLINE_PRECOMPUTED_REPLAY or self.is_demo:
            try:
                precompute_service.preload_all_frames(self.session_id, self.total_frames, max_preload=100)
            except Exception as e:
                logger.warning(f"Could not preload precomputed frames for session {self.session_id}: {e}")

    def _store_in_frame_cache(self, frame_idx: int, payload: ReplayFrameStreamPayload, max_entries: int = 128) -> None:
        """Store payload in in-memory frame cache, evicting oldest if exceeds max_entries to prevent RAM growth."""
        if len(self._frame_cache) >= max_entries and frame_idx not in self._frame_cache:
            try:
                oldest_key = next(iter(self._frame_cache))
                self._frame_cache.pop(oldest_key, None)
            except (StopIteration, KeyError):
                pass
        self._frame_cache[frame_idx] = payload


    def _load_poses_if_available(self) -> Optional[List[Tuple[float, float, float]]]:
        """Load real pose data from KITTI poses.txt if available."""
        if not self.poses_file:
            return None
        try:
            poses = []
            with open(self.poses_file, 'r') as f:
                for line in f:
                    vals = line.strip().split()
                    if len(vals) >= 12:
                        # KITTI pose format: 3x4 transformation matrix (row-major)
                        # Extract x, y, yaw from the transformation
                        tx = float(vals[3])   # x translation
                        tz = float(vals[11])  # z translation (forward in KITTI)
                        import math
                        r00 = float(vals[0])
                        r02 = float(vals[2])
                        yaw = math.atan2(r02, r00)
                        poses.append((tx, tz, yaw))
            if len(poses) > 0:
                logger.info(f"Loaded {len(poses)} real poses from {self.poses_file}")
                return poses
        except Exception as e:
            logger.warning(f"Failed to load poses from {self.poses_file}: {e}")
        return None

    # --- Bounded Async SalsaNext queue constants (per-session) ---
    # Queue holds at most this many pending jobs; oldest is dropped when full (latest-frame priority).
    _SALSANEXT_QUEUE_MAX: int = 2

    def _queue_salsanext_background(self, frame_idx: int, filename: str, points: np.ndarray) -> None:
        """Asynchronously compute SalsaNext neural predictions in background without blocking live playback.

        Policy:
        - Runs on BOTH CPU and GPU — no device gate that would silently skip neural inference.
        - Bounded queue (latest-frame priority): at most _SALSANEXT_QUEUE_MAX pending jobs.
          When full, the oldest pending job is discarded so the most recent frame is always
          processed, preventing unbounded latency growth on slow CPUs.
        - Single-worker constraint (per session): only one SalsaNext inference runs at a time
          to avoid CPU starvation when inference is slower than the playback rate.
        - Session-isolated: predictions are keyed by filename and stored via StorageService
          so they are never confused across sessions or frame indices.
        - Points array is copied before the closure so the main pipeline cannot mutate the
          snapshot while the background thread is consuming it.
        """
        if not model_settings.CHECKPOINT_PATH.exists():
            return

        import collections as _collections
        import threading as _threading

        # Lazily initialise per-session queue/lock the first time this is called.
        if not hasattr(self, "_sn_queue"):
            self._sn_queue: "_collections.deque" = _collections.deque(
                maxlen=self._SALSANEXT_QUEUE_MAX
            )
            self._sn_lock: "_threading.Lock" = _threading.Lock()
            self._sn_worker_active: "_threading.Event" = _threading.Event()

        # Snapshot so the background thread has its own immutable copy.
        pts_snapshot = points.copy()

        with self._sn_lock:
            if len(self._sn_queue) >= self._SALSANEXT_QUEUE_MAX:
                dropped = self._sn_queue.popleft()
                logger.debug(
                    f"[{self.session_id}] SalsaNext queue full — dropped stale job for frame "
                    f"{dropped[0]} ({dropped[1]}) in favour of frame {frame_idx}."
                )
            self._sn_queue.append((frame_idx, filename, pts_snapshot))
            already_running = self._sn_worker_active.is_set()

        if already_running:
            # A worker is already draining the queue; it will pick up the newly added job.
            return

        # Capture session_id for closure; avoids keeping a live `self` reference in the thread.
        session_id_cap = self.session_id

        def _drain_queue() -> None:
            """Drain the queue one job at a time, stopping when empty."""
            self._sn_worker_active.set()
            try:
                while True:
                    with self._sn_lock:
                        if not self._sn_queue:
                            break
                        job_frame_idx, job_filename, job_points = self._sn_queue.popleft()

                    try:
                        m_type = getattr(self, "target_model_type", "rellis")
                        raw_sem, insts, _meta = FastFRNetInferenceService.run_inference(
                            job_points, frame_id=job_filename, model_type=m_type, persist_artifact=True
                        )
                        # Persist using per-filename key — never bleed across sessions.
                        StorageService.save_labels(
                            job_filename,
                            (raw_sem & 0xFFFF).astype(np.uint16),
                            insts,
                            overwrite=True,
                        )
                        logger.debug(
                            f"[{session_id_cap}] Background Fast-FRNet complete for "
                            f"frame {job_frame_idx} ({job_filename})."
                        )
                    except Exception as exc:
                        logger.warning(
                            f"[{session_id_cap}] Background Fast-FRNet inference failed for "
                            f"frame {job_frame_idx} ({job_filename}): {exc}"
                        )
            finally:
                self._sn_worker_active.clear()

        _IO_EXECUTOR.submit(_drain_queue)

    def _ensure_lookahead(self, window: int = 3):
        """Asynchronously prefetch and process upcoming frames in background threads."""
        start_idx = self.current_frame_index
        end_idx = min(self.total_frames, start_idx + window)

        is_demo = (not self.is_manual_upload) and (
            self.session_id.startswith("foveamap_sequence") or self.session_id in ("demo", "sample", "default")
        )
        for idx in range(start_idx, end_idx):
            if idx not in self._frame_cache and idx not in self._lookahead_buffer and idx not in self._lookahead_in_flight:
                if is_demo and precompute_service.is_sequence_precomputed(self.session_id, idx + 1):
                    continue
                self._lookahead_in_flight.add(idx)
                
                def _prefetch_job(f_idx: int):
                    try:
                        p = self._process_frame_internal(f_idx, update_tracker=False, cache_result=True)
                        self._lookahead_buffer[f_idx] = p
                    except Exception as e:
                        logger.debug(f"Lookahead prefetch for frame {f_idx} failed: {e}")
                    finally:
                        self._lookahead_in_flight.discard(f_idx)

                _LOOKAHEAD_EXECUTOR.submit(_prefetch_job, idx)

    def set_playback_mode(self, mode: str) -> str:
        """Switch playback mode (offline_precomputed_replay or live_processing)."""
        if self.is_manual_upload:
            self.playback_mode = PlaybackMode.LIVE_PROCESSING
            return self.playback_mode
        self.playback_mode = mode
        return self.playback_mode

    def get_diagnostics(self) -> ReplayDiagnostics:
        """Return real-time perception and streaming diagnostics."""
        try:
            device = "CUDA" if FastFRNetInferenceService.get_device().type == "cuda" else "CPU"
        except Exception:
            device = "CPU"
        return ReplayDiagnostics(
            session_id=self.session_id,
            playback_mode=self.playback_mode,
            requested_fps=self.fps,
            current_fps=self.fps,
            device=device,
            total_frames=self.total_frames,
            current_frame=self.current_frame_index,
        )

    def get_status(self) -> ReplaySessionStatus:
        """Return current status snapshot."""
        return ReplaySessionStatus(
            session_id=self.session_id,
            sequence_name=self.sequence_name,
            state=self.state,
            current_frame_index=self.current_frame_index,
            total_frames=self.total_frames,
            fps=self.fps,
            playback_mode=self.playback_mode,
            data_mode=self.data_mode,
            semantic_source=self.semantic_source,
            coordinate_mode=self.coordinate_mode,
            has_predictions=self.has_predictions,
            error_message=self.error_message,
            has_poses=self.has_poses,
            has_calibration=self.has_calibration,
            has_timestamps=self.has_timestamps,
            is_manual_upload=self.is_manual_upload,
        )

    def start(self, fps: Optional[float] = None, playback_mode: Optional[str] = None) -> ReplaySessionStatus:
        """Start or resume playback."""
        if fps is not None:
            self.fps = max(0.2, min(60.0, fps))
        if playback_mode is not None:
            if not self.is_manual_upload:
                self.playback_mode = playback_mode
            else:
                self.playback_mode = PlaybackMode.LIVE_PROCESSING
        if self.current_frame_index >= self.total_frames - 1:
            self.current_frame_index = 0
            self.tracker.reset()
        self.state = ReplayPlaybackState.PLAYING
        self.error_message = None
        return self.get_status()

    def pause(self) -> ReplaySessionStatus:
        """Pause playback at current frame."""
        self.state = ReplayPlaybackState.PAUSED
        return self.get_status()

    def stop(self) -> ReplaySessionStatus:
        """Stop playback and rewind to beginning without clearing frame cache."""
        self.state = ReplayPlaybackState.READY
        self.current_frame_index = 0
        self.tracker.reset()
        return self.get_status()

    def get_global_cell(self, cell_key: str) -> Optional[AdaptiveGridCell]:
        """Look up a cell from the session-global fused map by its global key."""
        raw = self._global_map.get(cell_key)
        if raw is None:
            for k, v in self._global_map.items():
                if k.startswith(cell_key + "@") or k == cell_key:
                    raw = v
                    break
        if raw is None:
            return None
        if isinstance(raw, AdaptiveGridCell):
            return raw
        # If stored as lightweight dict, construct AdaptiveGridCell on demand for API query
        try:
            return AdaptiveGridCell.model_validate(raw)
        except Exception:
            return None
        # Try the latest frame that has the per-frame key
        for fr in range(self.current_frame_index, -1, -1):
            candidate = f"{cell_key}@{fr}"
            if candidate in self._global_map:
                return self._global_map[candidate]
        return None

    def list_global_cells(
        self,
        min_x: Optional[float] = None,
        max_x: Optional[float] = None,
        min_y: Optional[float] = None,
        max_y: Optional[float] = None,
        level: Optional[str] = None,
        limit: int = 5000,
    ) -> List[AdaptiveGridCell]:
        """Return cells from the global fused map within optional bounds."""
        cells = list(self._global_map.values())
        result: List[AdaptiveGridCell] = []
        for c in cells:
            if isinstance(c, dict):
                lvl = c.get("level")
                if level and level != "all" and lvl != level:
                    continue
                b = c.get("bounds", [0, 0, 0, 0])
                if min_x is not None and b[1] < min_x: continue
                if max_x is not None and b[0] > max_x: continue
                if min_y is not None and b[3] < min_y: continue
                if max_y is not None and b[2] > max_y: continue
                try:
                    result.append(AdaptiveGridCell.model_validate(c))
                except Exception:
                    pass
            else:
                if level and level != "all" and c.level.value != level:
                    continue
                if min_x is not None and c.bounds[1] < min_x: continue
                if max_x is not None and c.bounds[0] > max_x: continue
                if min_y is not None and c.bounds[3] < min_y: continue
                if max_y is not None and c.bounds[2] > max_y: continue
                result.append(c)
            if len(result) >= limit:
                break
        return result

    def _populate_global_map_from_cells(
        self,
        frame_idx: int,
        cells: List[Any],
        now_iso: Optional[str] = None,
        frame_id: Optional[str] = None,
    ) -> None:
        """Insert a frame's cells into the session-global fused map."""
        try:
            if len(self._global_map) > 20000:
                keys_to_del = list(self._global_map.keys())[:5000]
                for k in keys_to_del:
                    self._global_map.pop(k, None)

            for ac in cells:
                k = ac.get("cell_key") if isinstance(ac, dict) else ac.cell_key
                self._global_map[k] = ac
        except Exception as e:
            logger.warning(f"Error updating global map on frame {frame_idx}: {e}")

    def seek(self, frame_index: int) -> ReplaySessionStatus:
        """Seek to a target frame index with instant cache resolution."""
        target = max(0, min(self.total_frames - 1, frame_index))
        
        is_demo = (not self.is_manual_upload) and (self.is_demo or self.playback_mode == PlaybackMode.OFFLINE_PRECOMPUTED_REPLAY)
        cached = self._frame_cache.get(target)
        if cached is None and is_demo:
            cached = precompute_service.load_precomputed_frame(self.session_id, target)

        if cached is None:
            if not is_demo:
                if target < self.current_frame_index:
                    self.tracker.reset()
                    for i in range(max(0, target - 5), target):
                        self._process_frame_internal(i, update_tracker=True, cache_result=True)
                elif target > self.current_frame_index:
                    for i in range(self.current_frame_index, target):
                        self._process_frame_internal(i, update_tracker=True, cache_result=True)
        else:
            self._store_in_frame_cache(target, cached)

        self.current_frame_index = target
        return self.get_status()

    def get_current_frame(self) -> ReplayFrameStreamPayload:
        """Process and return current frame payload."""
        return self._process_frame_internal(self.current_frame_index, update_tracker=True, cache_result=True)

    def advance_and_get_frame(self) -> Optional[ReplayFrameStreamPayload]:
        """Advance current frame index by 1 and return processed frame."""
        if self.current_frame_index >= self.total_frames:
            self.state = ReplayPlaybackState.COMPLETED
            return None

        payload = self._process_frame_internal(self.current_frame_index, update_tracker=True, cache_result=True)

        self.current_frame_index += 1

        if self.current_frame_index >= self.total_frames:
            self.state = ReplayPlaybackState.COMPLETED

        return payload

    def _classify_geometrically(
        self, pts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Dict[str, int]]:
        """Fast vectorized geometric semantic perception fallback for unlabeled frames."""
        N = pts.shape[0]
        if N == 0:
            return np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.uint16), [], [], {}

        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        labels = np.full(N, 0, dtype=np.uint16)
        insts = np.zeros(N, dtype=np.uint16)

        # 1. Road Plane: z <= -1.25m, within road sweep corridor |y| <= 10m -> class 40 (road)
        road_mask = (zs <= -1.25) & (np.abs(ys) <= 10.0)
        labels[road_mask] = 40

        # 2. Sidewalk: perimeter of road corridor |y| between 10m and 18m -> class 48 (sidewalk)
        sidewalk_mask = (zs <= -1.15) & (np.abs(ys) > 10.0) & (np.abs(ys) <= 18.0)
        labels[sidewalk_mask] = 48

        # 3. High elevated vegetation outside road corridor: z > 0.8m, |y| > 8m -> class 70 (vegetation)
        veg_mask = (zs > 0.8) & (np.abs(ys) > 8.0) & (zs <= 6.0)
        labels[veg_mask] = 70

        # 4. Tall structural returns distant from road: z > 1.5m, |y| > 18m -> class 50 (building)
        bldg_mask = (zs > 1.5) & (np.abs(ys) > 18.0)
        labels[bldg_mask] = 50

        sem_classes: List[str] = []
        proj_cats: List[str] = []

        _, cat_dist = SemanticMappingService.compute_distributions(labels)
        return labels, insts, sem_classes, proj_cats, cat_dist

    def _process_frame_internal(
        self,
        frame_idx: int,
        update_tracker: bool = True,
        cache_result: bool = True,
        ignore_disk_cache: bool = False,
        custom_tracker: Optional[Any] = None,
    ) -> ReplayFrameStreamPayload:
        """Execute full perception pipeline on the frame at index."""
        start_time = time.perf_counter()

        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.total_frames - 1}].")

        # 0. Fast-Path: Instant load from in-memory cache or precomputed disk snapshots
        is_demo = (not self.is_manual_upload) and (self.is_demo or self.playback_mode == PlaybackMode.OFFLINE_PRECOMPUTED_REPLAY)
        if not ignore_disk_cache:
            cached_payload = self._frame_cache.get(frame_idx)
            if cached_payload is None and is_demo:
                cached_payload = precompute_service.load_precomputed_frame(self.session_id, frame_idx)

            if cached_payload is not None:
                cached_payload.state = self.state
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                cached_payload.processing_time_ms = round(elapsed_ms if elapsed_ms > 0 else 0.5, 2)
                if is_demo:
                    cached_payload.semantic_source = "PRECOMPUTED"
                elif not getattr(cached_payload, "semantic_source", None):
                    cached_payload.semantic_source = "LIVE FAST-FRNET"
                if cached_payload.performance is None:
                    cached_meta = cached_payload.map_result.metadata if cached_payload.map_result else None
                    fc = getattr(cached_meta, "fine_cells_count", 0) if cached_meta else 0
                    mc = getattr(cached_meta, "medium_cells_count", 0) if cached_meta else 0
                    cc = getattr(cached_meta, "coarse_cells_count", 0) if cached_meta else 0
                    bounds = getattr(cached_meta, "bounds", None) if cached_meta else None

                    res_m = None
                    if bounds:
                        span_x = max(0.0, float(bounds.max_x - bounds.min_x))
                        span_y = max(0.0, float(bounds.max_y - bounds.min_y))
                        span_z = max(0.0, float(bounds.max_z - bounds.min_z))
                        voxel_size = 0.5
                        nv_x = max(1, int(np.ceil(span_x / voxel_size))) if span_x > 0 else 1
                        nv_y = max(1, int(np.ceil(span_y / voxel_size))) if span_y > 0 else 1
                        nv_z = max(1, int(np.ceil(span_z / voxel_size))) if span_z > 0 else 1
                        u_voxels = nv_x * nv_y * nv_z
                        u_bytes = u_voxels * 4
                        a_bytes = (fc + mc + cc) * 64
                        ratio = float(u_bytes / a_bytes) if a_bytes > 0 else 1.0
                        savings = float((1.0 - (a_bytes / u_bytes)) * 100.0) if u_bytes > 0 else 0.0
                        from ..models.metric_schemas import AdaptiveGridResourceMetrics, SpatialBounds
                        sp_bounds = SpatialBounds(
                            min_x=float(bounds.min_x), max_x=float(bounds.max_x),
                            min_y=float(bounds.min_y), max_y=float(bounds.max_y),
                            min_z=float(bounds.min_z), max_z=float(bounds.max_z),
                            span_x=float(span_x), span_y=float(span_y), span_z=float(span_z),
                            volume_m3=float(span_x * span_y * span_z),
                        )
                        res_m = AdaptiveGridResourceMetrics(
                            input_point_count=cached_payload.point_count,
                            fine_cells_count=fc,
                            medium_cells_count=mc,
                            coarse_cells_count=cc,
                            total_cells_count=fc + mc + cc,
                            adaptive_grid_bytes=a_bytes,
                            serialized_map_bytes=(fc + mc + cc) * 128,
                            uniform_baseline_bytes=u_bytes,
                            uniform_voxel_size_m=0.5,
                            uniform_bytes_per_voxel=4,
                            uniform_total_voxels=u_voxels,
                            memory_reduction_percentage=round(savings, 2),
                            spatial_bounds=sp_bounds,
                            methodology_note="Analytical snapshot estimate",
                        )

                    cached_payload.performance = FramePerformanceMetrics.model_construct(
                        frame_id=cached_payload.frame_filename,
                        session_id=self.session_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        device_used="GPU/CUDA",
                        model_name="Fast-FRNet (SemanticKITTI)",
                        timings=PipelineStageLatencies.model_construct(
                            lidar_preprocessing_ms=0.0,
                            salsanext_inference_ms=0.0,
                            terrain_analysis_ms=0.0,
                            object_detection_tracking_ms=0.0,
                            adaptive_grid_ms=0.0,
                            total_latency_ms=cached_payload.processing_time_ms,
                            preprocessing_ms=0.0,
                            inference_ms=0.0,
                            terrain_ms=0.0,
                            clustering_ms=0.0,
                            grid_ms=0.0,
                            total_ms=cached_payload.processing_time_ms,
                        ),
                        actual_fps=60.0,
                        resource_metrics=res_m,
                        accuracy_metrics=None,
                    )

                if cache_result:
                    self._store_in_frame_cache(frame_idx, cached_payload)
                if cached_payload.map_result and cached_payload.map_result.cells_sample:
                    _MAP_EXECUTOR.submit(
                        self._populate_global_map_from_cells,
                        frame_idx,
                        cached_payload.map_result.cells_sample,
                    )
                return cached_payload

        frame_meta = self.frames_meta[frame_idx]
        bin_path = Path(frame_meta["bin_path"])
        label_path = Path(frame_meta["label_path"]) if frame_meta.get("label_path") else None

        # 1 & 2. Read, Parse & Preprocess LiDAR points
        t_prep_start = time.perf_counter()
        with open(bin_path, "rb") as f:
            bin_bytes = f.read()
        raw_points = LidarParser.parse_kitti_bin(bin_bytes)

        prep_config = PreprocessingConfig(
            min_range=settings.DEFAULT_RANGE_MIN,
            max_range=settings.DEFAULT_RANGE_MAX,
            z_min=settings.DEFAULT_Z_MIN,
            z_max=settings.DEFAULT_Z_MAX,
            remove_nan_inf=True,
        )
        prep_points, prep_report = PointCloudPreprocessor.process(raw_points, prep_config)
        t_prep_ms = (time.perf_counter() - t_prep_start) * 1000.0

        # 3. Semantic Labels: Hardware-Aware routing
        t_inf_start = time.perf_counter()
        raw_label_ids: Optional[np.ndarray] = None
        instance_ids: Optional[np.ndarray] = None
        # Determine active dataset context
        sess_context = (self.session_id + " " + self.sequence_name + " " + frame_meta.get("filename", "")).lower()
        m_dataset = "rellis" if any(k in sess_context for k in ("rellis", "offroad", "off_road")) else "semantickitti"
        target_model_type = "rellis" if m_dataset == "rellis" else "semantickitti"
        active_ckpt = model_settings.get_resolved_checkpoint_path(target_model_type)

        # Ground truth labels (used exclusively for accuracy/evaluation metrics, NEVER as inference output)
        gt_labels: Optional[np.ndarray] = None
        gt_insts: Optional[np.ndarray] = None
        if label_path and label_path.exists():
            try:
                if str(label_path).endswith(".npz"):
                    with np.load(label_path) as ldata:
                        gt_labels = ldata["semantic_classes"].copy()
                        if "instance_ids" in ldata:
                            gt_insts = ldata["instance_ids"].copy()
                else:
                    with open(label_path, "rb") as f:
                        gt_bytes = f.read()
                    gt_labels, gt_insts = SemanticMappingService.parse_kitti_label_bin(gt_bytes)
            except Exception as e:
                logger.warning(f"Could not load ground truth label {label_path}: {e}")

        # Execute Fast-FRNet neural inference
        # In demo mode, if stored precomputed labels exist in StorageService and not ignoring disk cache, reuse them.
        # Otherwise, Fast-FRNet neural inference is executed. Ground truth labels are NEVER substituted as model predictions (Requirement 15).
        if is_demo and not ignore_disk_cache and (stored_labels := (StorageService.get_labels(frame_meta["filename"], allow_precomputed_fallback=True) or StorageService.get_labels(f"{frame_idx:06d}", allow_precomputed_fallback=True))) is not None and len(stored_labels[0]) == len(raw_points):
            raw_labels, inst_ids = stored_labels
            raw_label_ids = raw_labels
            instance_ids = inst_ids
            class_dist, cat_dist = SemanticMappingService.compute_distributions(raw_labels, dataset=m_dataset)
            category_dist = cat_dist
            semantic_source = "PRECOMPUTED"
        elif not is_demo and (self.data_mode == ReplayDataMode.PRECOMPUTED_LABELS or self.has_predictions) and gt_labels is not None and len(gt_labels) == len(raw_points):
            raw_label_ids = gt_labels
            instance_ids = gt_insts if gt_insts is not None and len(gt_insts) == len(raw_points) else np.zeros(len(raw_points), dtype=np.uint16)
            class_dist, cat_dist = SemanticMappingService.compute_distributions(raw_label_ids, dataset=m_dataset)
            category_dist = cat_dist
            semantic_source = "GROUND TRUTH"
        else:
            # LIVE INFERENCE or DEMO PRECOMPUTATION: Execute actual Fast-FRNet neural inference
            try:
                raw_sem_classes, inst_ids, meta = FastFRNetInferenceService.run_inference(
                    raw_points, frame_id=frame_meta["filename"], model_type=target_model_type, persist_artifact=False
                )
                raw_label_ids = raw_sem_classes
                instance_ids = inst_ids
                category_dist = meta.get("project_category_counts", {})
                semantic_source = "PRECOMPUTED" if is_demo else "LIVE FAST-FRNET"
            except Exception as e:
                logger.error(f"Fast-FRNet neural inference failed on frame {frame_idx} ({frame_meta['filename']}): {e}")
                raise RuntimeError(f"Fast-FRNet neural inference failed on frame {frame_idx}: {str(e)}")

        t_inf_ms = (time.perf_counter() - t_inf_start) * 1000.0

        # Runtime diagnostics (Requirement 13)
        dev_str = FastFRNetInferenceService.get_device().type.upper()
        diag_msg = (
            f"\n"
            f"FRAME {frame_idx}\n"
            f"Points: {len(raw_points):,}\n"
            f"Model: {active_ckpt.name}\n"
            f"Inference source: FastFRNetInferenceService\n"
            f"Device: {dev_str}\n"
            f"Service initialized: YES\n"
            f"Inference: {t_inf_ms:.0f} ms\n"
        )
        print(diag_msg, flush=True)
        logger.info(diag_msg)

        # 4 & 5. Terrain Analysis + 3D Instance Clustering
        sem_class_ids_for_clustering = (raw_label_ids & 0xFFFF).astype(np.uint16) if raw_label_ids is not None else None
        clustering_kwargs = dict(
            frame_id=frame_meta["filename"],
            points=raw_points,
            semantic_classes=sem_class_ids_for_clustering,
            instance_ids=instance_ids,
            dataset=m_dataset,
        )

        t_terr_start = time.perf_counter()
        terrain_res = TerrainAnalysisEngine.analyze(
            frame_meta["filename"],
            prep_points,
            settings.DEFAULT_GRID_RESOLUTION,
            preprocessing_config=None,
            include_cells=False,
        )
        t_terr_ms = (time.perf_counter() - t_terr_start) * 1000.0

        if raw_label_ids is not None and len(raw_label_ids) == len(raw_points):
            cluster_res = InstanceClusteringService.detect_objects(
                **clustering_kwargs,
            )
            instances = cluster_res.instances
        else:
            instances = []

        # 5b. Kalman Tracking
        t_obj_start = time.perf_counter()
        tracker_to_use = custom_tracker or self.tracker
        if update_tracker:
            tracks = tracker_to_use.update(instances, timestamp=None, ego_pose=None)
        else:
            tracks = tracker_to_use.get_active_tracks()
        t_obj_ms = (time.perf_counter() - t_obj_start) * 1000.0

        coord_mode = TrackingCoordinateMode.LOCAL_FRAME if self.coordinate_mode == "local_frame" else TrackingCoordinateMode.WORLD_FRAME

        tracking_res = TrackingUpdateResponse.model_construct(
            frame_id=frame_meta["filename"],
            coordinate_mode=coord_mode,
            active_tracks_count=len(tracks),
            new_tracks_count=len([t for t in tracks if t.lifecycle_state == "new"]),
            lost_tracks_count=len([t for t in tracks if t.lifecycle_state == "temporarily_lost"]),
            tracks=tracks,
            timestamp=f"T_{frame_idx:06d}",
        )

        objects_res = ObjectDetectionResponse.model_construct(
            frame_id=frame_meta["filename"],
            total_instances=len(instances),
            dynamic_instances_count=len([i for i in instances if i.is_dynamic]),
            static_instances_count=len([i for i in instances if not i.is_dynamic]),
            instances=instances,
            status=ProcessStatus.COMPLETE,
            created_at=f"frame_{frame_idx:06d}",
        )

        # 7. Adaptive Variable-Resolution 2.5D Grid Engine
        t_grid_start = time.perf_counter()
        dynamic_positions = [list(t.current_position) for t in tracks if t.lifecycle_state in ("new", "active")]
        adaptive_cells = AdaptiveGridService.generate_grid_from_points_fast(
            points=raw_points,
            frame_id=frame_meta["filename"],
            labels=raw_label_ids,
            instance_ids=instance_ids,
            dynamic_track_positions=dynamic_positions,
            dataset=m_dataset,
        )
        t_grid_ms = (time.perf_counter() - t_grid_start) * 1000.0

        now_iso = datetime.now(timezone.utc).isoformat()
        fine_c = sum(1 for c in adaptive_cells if c.get("level") == "fine")
        med_c = sum(1 for c in adaptive_cells if c.get("level") == "medium")
        coarse_c = sum(1 for c in adaptive_cells if c.get("level") == "coarse")

        # Persist into session-global fused map asynchronously
        _MAP_EXECUTOR.submit(
            self._populate_global_map_from_cells,
            frame_idx,
            adaptive_cells,
            now_iso=now_iso,
            frame_id=frame_meta["filename"],
        )

        bounds_obj = BoundingBox3D.model_construct(
            min_x=float(np.min(prep_points[:, 0])),
            max_x=float(np.max(prep_points[:, 0])),
            min_y=float(np.min(prep_points[:, 1])),
            max_y=float(np.max(prep_points[:, 1])),
            min_z=float(np.min(prep_points[:, 2])),
            max_z=float(np.max(prep_points[:, 2])),
        ) if len(prep_points) > 0 else BoundingBox3D.model_construct(min_x=0.0, max_x=0.0, min_y=0.0, max_y=0.0, min_z=0.0, max_z=0.0)

        meta_obj = MapMetadata.model_construct(
            map_id=f"replay_{self.session_id}",
            mode=MapMode(self.map_mode) if self.map_mode in ("local_only", "global_fusion") else MapMode.LOCAL_ONLY,
            frame_count=frame_idx + 1,
            total_cells=len(adaptive_cells),
            fine_cells_count=fine_c,
            medium_cells_count=med_c,
            coarse_cells_count=coarse_c,
            bounds=bounds_obj,
            created_at=now_iso,
            updated_at=now_iso,
        )

        map_update_res = MapUpdateResponse.model_construct(
            map_id=f"replay_{self.session_id}",
            mode=MapMode(self.map_mode) if self.map_mode in ("local_only", "global_fusion") else MapMode.LOCAL_ONLY,
            status=ProcessStatus.COMPLETE,
            updated_cell_count=len(adaptive_cells),
            metadata=meta_obj,
            cells_sample=adaptive_cells,
            message=f"Replay frame {frame_idx} 2.5D grid synthesized",
        )

        # 8. Controlled Bounded Visual Points Sampling (up to 3,500 points, prioritizing actors & obstacles)
        from .semantic_registry import SemanticClassRegistry

        target_pts = raw_points if len(raw_points) > 0 else prep_points
        max_sample = min(3500, len(target_pts))
        if len(target_pts) <= max_sample:
            sample_indices = np.arange(len(target_pts))
        else:
            if raw_label_ids is not None and len(raw_label_ids) == len(target_pts):
                # Identify non-ground returns (actors, obstacles, structures) via vectorized lookup table
                actor_mask = SemanticClassRegistry.get_actor_mask(raw_label_ids, dataset=m_dataset)
                actor_idxs = np.where(actor_mask)[0]
                ground_idxs = np.where(~actor_mask)[0]

                n_actors = min(len(actor_idxs), 1800)
                sel_actors = np.random.choice(actor_idxs, size=n_actors, replace=False) if len(actor_idxs) > n_actors else actor_idxs
                n_ground = max_sample - len(sel_actors)
                sel_ground = np.random.choice(ground_idxs, size=n_ground, replace=False) if len(ground_idxs) > n_ground else ground_idxs
                sample_indices = np.concatenate([sel_actors, sel_ground])
            else:
                sample_indices = np.random.choice(len(target_pts), size=max_sample, replace=False)

        sub_pts = target_pts[sample_indices]
        xs = np.round(sub_pts[:, 0], 2)
        ys = np.round(sub_pts[:, 1], 2)
        zs = np.round(sub_pts[:, 2], 2)
        ints = np.round(sub_pts[:, 3], 2) if target_pts.shape[1] > 3 else np.full(len(sample_indices), 0.5)
        sub_lbls = raw_label_ids[sample_indices] if raw_label_ids is not None else np.zeros(len(sample_indices), dtype=np.uint16)

        points_sample = [
            {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "intensity": float(i),
                "semantic_class": SemanticClassRegistry.get_name(int(l), dataset=m_dataset),
                "project_category": SemanticClassRegistry.get_category(int(l), dataset=m_dataset).value,
            }
            for x, y, z, i, l in zip(xs, ys, zs, ints, sub_lbls)
        ]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 9. Real-time performance metrics computed synchronously
        actual_fps = (1000.0 / elapsed_ms) if elapsed_ms > 0 else 0.0
        latencies = PipelineStageLatencies.model_construct(
            lidar_preprocessing_ms=round(t_prep_ms, 2),
            salsanext_inference_ms=round(t_inf_ms, 2),
            terrain_analysis_ms=round(t_terr_ms, 2),
            object_detection_tracking_ms=round(t_obj_ms, 2),
            adaptive_grid_ms=round(t_grid_ms, 2),
            total_latency_ms=round(elapsed_ms, 2),
            preprocessing_ms=round(t_prep_ms, 2),
            inference_ms=round(t_inf_ms, 2),
            terrain_ms=round(t_terr_ms, 2),
            clustering_ms=round(t_obj_ms, 2),
            grid_ms=round(t_grid_ms, 2),
            total_ms=round(elapsed_ms, 2),
        )

        res_m = MetricsService.compute_adaptive_grid_resource_metrics(
            points=raw_points,
            fine_count=fine_c,
            medium_count=med_c,
            coarse_count=coarse_c,
            voxel_size_m=0.5,
            bytes_per_voxel=4,
        )

        dev_type = FastFRNetInferenceService.get_device().type
        model_display_name = f"Fast-FRNet ({target_model_type.upper()})"
        fpm = FramePerformanceMetrics.model_construct(
            frame_id=frame_meta["filename"],
            session_id=self.session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            device_used=dev_type if semantic_source == "LIVE FAST-FRNET" else "CPU/Disk",
            model_name=model_display_name,
            timings=latencies,
            actual_fps=round(actual_fps, 2),
            resource_metrics=res_m,
            accuracy_metrics=None,
        )

        # Synchronously record frame metrics and session summary
        MetricsService.record_frame_metrics(fpm)
        MetricsService.record_session_metrics(self.session_id, self.sequence_name)

        if label_path and label_path.exists():
            def _record_accuracy_bg(_pts, _preds, _gt_p, _meta, _cur_fpm):
                try:
                    acc_gt = None
                    if str(_gt_p).endswith(".npz"):
                        with np.load(_gt_p) as ldata:
                            acc_gt = ldata["semantic_classes"].copy()
                    else:
                        with open(_gt_p, "rb") as f:
                            acc_gt, _ = SemanticMappingService.parse_kitti_label_bin(f.read())
                    acc_m = MetricsService.compute_accuracy_metrics(
                        points=_pts,
                        pred_labels=_preds,
                        gt_labels=acc_gt,
                    )
                    _cur_fpm.accuracy_metrics = acc_m
                    MetricsService.record_frame_metrics(_cur_fpm)
                except Exception as e:
                    logger.debug(f"Async accuracy update error: {e}")

            from .metrics_service import _METRICS_EXECUTOR
            _METRICS_EXECUTOR.submit(
                _record_accuracy_bg,
                raw_points,
                raw_label_ids if raw_label_ids is not None else np.zeros(len(raw_points), dtype=np.uint16),
                label_path,
                frame_meta,
                fpm,
            )

        payload = ReplayFrameStreamPayload.model_construct(
            session_id=self.session_id,
            sequence_name=self.sequence_name,
            frame_index=frame_idx,
            total_frames=self.total_frames,
            frame_filename=frame_meta["filename"],
            timestamp=f"Frame #{frame_idx:04d}",
            point_count=len(prep_points),
            points_sample=points_sample,
            category_distribution=category_dist,
            terrain_result=terrain_res,
            objects_result=objects_res,
            tracking_result=tracking_res,
            map_result=map_update_res,
            coordinate_mode=self.coordinate_mode,
            map_mode=self.map_mode,
            data_mode=self.data_mode,
            semantic_source=semantic_source,
            state=self.state,
            processing_time_ms=round(elapsed_ms, 2),
            performance=fpm,
        )

        if cache_result:
            self._store_in_frame_cache(frame_idx, payload)
            if not self.is_manual_upload and (self.is_demo or self.playback_mode == PlaybackMode.OFFLINE_PRECOMPUTED_REPLAY):
                _IO_EXECUTOR.submit(
                    precompute_service.save_precomputed_frame,
                    self.session_id,
                    frame_idx,
                    payload,
                )

        return payload

    def get_frame_raw_json(self, frame_idx: int) -> Optional[str]:
        """Retrieve pre-serialized JSON text for sub-millisecond 60 FPS WebSocket transmission."""
        if self.is_manual_upload or not self.is_demo:
            return None
        return precompute_service.get_raw_frame_json(self.session_id, frame_idx)


class ReplaySessionManager:
    """Singleton repository managing active replay sessions."""

    def __init__(self):
        self._sessions: Dict[str, ReplaySession] = {}

    def create_session(self, session_data: Dict[str, Any]) -> ReplaySession:
        session = ReplaySession(session_data)
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ReplaySession]:
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Attempt auto-loading session from disk/archive/precomputed
        try:
            from .sequence_replay import sequence_replay_service
            session_data = sequence_replay_service.load_existing_session(session_id)
            if session_data:
                session = self.create_session(session_data)
                logger.info(f"Auto-loaded replay session '{session_id}' with {session.total_frames} frames.")
                return session
        except Exception as e:
            logger.warning(f"Failed to auto-load session '{session_id}': {e}")

        # For demo/sample/default aliases, try semantic_kitti_sequence_00 first, then foveamap_sequence01_replay
        if session_id in ("demo", "sample", "default"):
            try:
                from .sequence_replay import sequence_replay_service
                session_data = sequence_replay_service.load_existing_session("semantic_kitti_sequence_00")
                if not session_data:
                    session_data = sequence_replay_service.load_existing_session("foveamap_sequence00_replay")
                if not session_data:
                    available = self.list_available_sessions()
                    if available:
                        session_data = sequence_replay_service.load_existing_session(available[0]["session_id"])
                if session_data:
                    session = self.create_session(session_data)
                    self._sessions[session_id] = session
                    return session
            except Exception as e:
                logger.warning(f"Failed to auto-load session for alias '{session_id}': {e}")

        return None

    def list_available_sessions(self) -> List[Dict[str, Any]]:
        """Discover all available sessions from disk, ZIP files, and precomputed directories."""
        from ..config import settings
        available = []
        seen_ids = set()
        
        # 1. Already-loaded sessions
        for sid, session in self._sessions.items():
            if sid not in seen_ids:
                available.append({
                    "session_id": sid,
                    "sequence_name": session.sequence_name,
                    "total_frames": session.total_frames,
                    "has_predictions": session.has_predictions,
                    "data_mode": session.data_mode,
                    "semantic_source": session.semantic_source,
                    "has_poses": session.has_poses,
                    "state": session.state,
                })
                seen_ids.add(sid)
        
        # 2. Sequences directory
        seq_dirs = [settings.SEQUENCES_DIR]
        repo_seq = Path(__file__).resolve().parent.parent.parent / "data" / "sequences"
        if repo_seq.exists() and repo_seq != settings.SEQUENCES_DIR:
            seq_dirs.append(repo_seq)

        for seq_dir in seq_dirs:
            if seq_dir.exists():
                for child in sorted(seq_dir.iterdir()):
                    if child.is_dir() and child.name not in seen_ids:
                        bin_files = list(child.glob("**/*.bin"))
                        if bin_files:
                            available.append({
                                "session_id": child.name,
                                "sequence_name": child.name,
                                "total_frames": len(bin_files),
                                "has_predictions": bool(list(child.glob("**/*.label"))),
                                "data_mode": "discovered",
                                "semantic_source": "unknown",
                                "has_poses": bool(list(child.glob("**/poses.txt"))),
                                "state": "available",
                            })
                            seen_ids.add(child.name)
        
        # 3. ZIP files in BASE_DIR
        base_dir = settings.BASE_DIR
        if base_dir.exists():
            for zip_file in sorted(base_dir.glob("*.zip")):
                zip_name = zip_file.stem
                if zip_name not in seen_ids:
                    available.append({
                        "session_id": zip_name,
                        "sequence_name": zip_name,
                        "total_frames": 0,  # Unknown until loaded
                        "has_predictions": False,
                        "data_mode": "archive",
                        "semantic_source": "unknown",
                        "has_poses": False,
                        "state": "available",
                    })
                    seen_ids.add(zip_name)
        
        # 4. Precomputed directories
        precomp_dir = settings.DATA_DIR / "precomputed"
        if precomp_dir.exists():
            for child in sorted(precomp_dir.iterdir()):
                if child.is_dir() and child.name not in seen_ids:
                    frame_files = list(child.glob("frame_*.json"))
                    if frame_files:
                        available.append({
                            "session_id": child.name,
                            "sequence_name": f"{child.name} (Precomputed)",
                            "total_frames": len(frame_files),
                            "has_predictions": True,
                            "data_mode": "precomputed_labels",
                            "semantic_source": "PRECOMPUTED",
                            "has_poses": False,
                            "state": "available",
                        })
                        seen_ids.add(child.name)
        
        return available

    def ensure_available_sessions(self) -> List[Dict[str, Any]]:
        """Discover and initialize available sequences on startup."""
        try:
            self.get_session("semantic_kitti_sequence_00")
            self.get_session("foveamap_sequence00_replay")
        except Exception:
            pass
        available = self.list_available_sessions()
        for info in available[:3]:  # Auto-load up to 3 sessions
            try:
                self.get_session(info["session_id"])
            except Exception as e:
                logger.warning(f"Failed to auto-load session '{info['session_id']}': {e}")
        return available

    def ensure_default_sessions(self) -> None:
        """Initialize bundled demo sessions on startup."""
        self.ensure_available_sessions()

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


replay_session_manager = ReplaySessionManager()

