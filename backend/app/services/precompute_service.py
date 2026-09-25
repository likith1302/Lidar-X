"""Offline Precomputed LiDAR Perception & Demo Replay Engine.

Executes the full perception pipeline (Fast-FRNet, Terrain Analysis, DBSCAN,
Kalman Multi-Object Tracking, and 2.5D Adaptive Grid) ahead of time for demo sequences.
Persists compact JSON frame snapshots and a complete manifest to disk (backend/data/demo_cache/<session_id>/)
and maintains an LRU memory cache for high-rate, zero-inference dashboard playback up to 60 FPS.
Supports arbitrary sequence lengths (e.g. 4,000+ frames) with resumable execution and streaming memory safety.
"""

import time
import json
import asyncio
import logging
import gc
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Any, Callable, List, Tuple, Set
from datetime import datetime, timezone

from ..config import settings
from ..models.replay_schemas import (
    ReplayFrameStreamPayload,
    PrecomputeProgressStatus,
    ReplayPlaybackState,
    ReplayPointSample,
)

logger = logging.getLogger(__name__)


class PrecomputeReplayService:
    """Manages offline sequence precomputation jobs, manifest tracking, and demo snapshot persistence."""

    def __init__(self):
        self.demo_cache_dir: Path = settings.DEMO_CACHE_DIR
        self.demo_cache_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_precompute_dir: Path = settings.DATA_DIR / "precomputed"
        self.legacy_precompute_dir.mkdir(parents=True, exist_ok=True)

        self._progress: Dict[str, PrecomputeProgressStatus] = {}
        self._active_tasks: Dict[str, asyncio.Task] = {}

        # Bounded LRU in-memory caches (max 128 frames per session to avoid RAM exhaustion on 4,000+ frames)
        self._max_cache_frames: int = 128
        self._snapshot_cache: Dict[str, OrderedDict[int, ReplayFrameStreamPayload]] = {}
        self._raw_json_cache: Dict[str, OrderedDict[int, str]] = {}

    def _get_lru_raw(self, session_id: str, frame_idx: int) -> Optional[str]:
        """Get raw JSON from bounded LRU cache."""
        cache = self._raw_json_cache.get(session_id)
        if cache is not None and frame_idx in cache:
            cache.move_to_end(frame_idx)
            return cache[frame_idx]
        return None

    def _put_lru_raw(self, session_id: str, frame_idx: int, raw_json: str) -> None:
        """Store raw JSON into bounded LRU cache."""
        if session_id not in self._raw_json_cache:
            self._raw_json_cache[session_id] = OrderedDict()
        cache = self._raw_json_cache[session_id]
        cache[frame_idx] = raw_json
        cache.move_to_end(frame_idx)
        while len(cache) > self._max_cache_frames:
            cache.popitem(last=False)

    def _get_lru_payload(self, session_id: str, frame_idx: int) -> Optional[ReplayFrameStreamPayload]:
        """Get payload from bounded LRU cache."""
        cache = self._snapshot_cache.get(session_id)
        if cache is not None and frame_idx in cache:
            cache.move_to_end(frame_idx)
            return cache[frame_idx]
        return None

    def _put_lru_payload(self, session_id: str, frame_idx: int, payload: ReplayFrameStreamPayload) -> None:
        """Store payload into bounded LRU cache."""
        if session_id not in self._snapshot_cache:
            self._snapshot_cache[session_id] = OrderedDict()
        cache = self._snapshot_cache[session_id]
        cache[frame_idx] = payload
        cache.move_to_end(frame_idx)
        while len(cache) > self._max_cache_frames:
            cache.popitem(last=False)

    def clear_session(self, session_id: str, delete_disk: bool = False) -> None:
        """Purge in-memory caches, and optionally delete on-disk demo cache."""
        if session_id in self._snapshot_cache:
            del self._snapshot_cache[session_id]
        if session_id in self._raw_json_cache:
            del self._raw_json_cache[session_id]
        if session_id in self._progress:
            del self._progress[session_id]

        if delete_disk and session_id not in ("semantic_kitti_sequence_00", "foveamap_sequence00_replay"):
            import shutil
            demo_dir = self.demo_cache_dir / session_id
            if demo_dir.exists():
                shutil.rmtree(demo_dir, ignore_errors=True)
            legacy_dir = self.legacy_precompute_dir / session_id
            if legacy_dir.exists():
                shutil.rmtree(legacy_dir, ignore_errors=True)

    def get_session_dir(self, session_id: str) -> Path:
        """Return directory path for demo cache frames of a session."""
        # 1. Primary demo_cache directory
        primary = self.demo_cache_dir / session_id
        if primary.exists():
            return primary

        # 2. Check legacy precomputed directory
        legacy = self.legacy_precompute_dir / session_id
        if legacy.exists():
            return legacy

        # 3. Check data/precomputed relative to file
        source_dir = Path(__file__).resolve().parent.parent.parent / "data" / "precomputed" / session_id
        if source_dir.exists():
            return source_dir

        # Default to creating in demo_cache
        primary.mkdir(parents=True, exist_ok=True)
        (primary / "frames").mkdir(parents=True, exist_ok=True)
        return primary


    def get_manifest_path(self, session_id: str) -> Path:
        """Return path to manifest.json for session."""
        return self.get_session_dir(session_id) / "manifest.json"

    def load_manifest(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load manifest.json if present on disk."""
        m_path = self.get_manifest_path(session_id)
        if m_path.exists():
            try:
                with open(m_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read manifest for session {session_id}: {e}")
        return None

    def save_manifest(self, session_id: str, manifest_data: Dict[str, Any]) -> None:
        """Write manifest.json to disk atomically."""
        session_dir = self.get_session_dir(session_id)
        m_path = session_dir / "manifest.json"
        tmp_path = session_dir / "manifest.json.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
            tmp_path.replace(m_path)
        except Exception as e:
            logger.error(f"Failed to write manifest for session {session_id}: {e}")

    def get_progress(self, session_id: str) -> Optional[PrecomputeProgressStatus]:
        """Get current precomputation progress for a session."""
        if session_id in self._progress:
            return self._progress[session_id]

        # Derive progress from manifest or disk
        manifest = self.load_manifest(session_id)
        if manifest:
            total = manifest.get("total_frames", 0)
            completed = manifest.get("completed_count", 0)
            failed = len(manifest.get("failed_frames", []))
            pct = round((completed / total * 100.0), 2) if total > 0 else 0.0
            is_comp = manifest.get("status") == "completed" or (total > 0 and completed >= total)
            return PrecomputeProgressStatus(
                session_id=session_id,
                sequence_name=manifest.get("sequence_name", session_id),
                processed_frames=completed,
                total_frames=total,
                percent_complete=pct,
                is_complete=is_comp,
                is_running=False,
                elapsed_seconds=0.0,
                eta_seconds=0.0,
                current_stage="Completed" if is_comp else "Ready",
                failed_count=failed,
                error_message=None,
            )
        return None

    def cancel_precompute(self, session_id: str) -> bool:
        """Cancel an in-progress precomputation job."""
        task = self._active_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
            if session_id in self._progress:
                self._progress[session_id].is_running = False
                self._progress[session_id].error_message = "Cancelled by user"
            return True
        return False

    def is_sequence_precomputed(self, session_id: str, total_frames: int) -> bool:
        """Check if all frames of a sequence have been precomputed and saved."""
        if total_frames <= 0:
            return False

        manifest = self.load_manifest(session_id)
        if manifest and manifest.get("status") == "completed" and manifest.get("total_frames") == total_frames:
            return True

        completed_indices = self.get_completed_frame_indices(session_id)
        return len(completed_indices) >= total_frames

    def get_completed_frame_indices(self, session_id: str) -> Set[int]:
        """Return set of 0-indexed frame integers that are already saved and non-empty on disk."""
        session_dir = self.get_session_dir(session_id)
        completed: Set[int] = set()

        if not session_dir.exists():
            return completed

        # Check frames/ directory
        frames_dir = session_dir / "frames"
        if frames_dir.exists():
            for f in frames_dir.glob("*.json"):
                if f.stat().st_size > 50:
                    stem = f.stem
                    if stem.isdigit():
                        completed.add(int(stem))

        # Check root of session_dir for frame_XXXXXX.json
        for f in session_dir.glob("frame_*.json"):
            if f.stat().st_size > 50:
                idx_str = f.stem.replace("frame_", "")
                if idx_str.isdigit():
                    completed.add(int(idx_str))

        return completed

    def get_precomputed_frame_count(self, session_id: str) -> int:
        """Count how many precomputed frames are available on disk."""
        return len(self.get_completed_frame_indices(session_id))

    def get_raw_frame_json(self, session_id: str, frame_idx: int) -> Optional[str]:
        """Retrieve pre-serialized JSON string from LRU cache or disk for zero-copy 60 FPS streaming."""
        # 1. LRU in-memory cache
        cached = self._get_lru_raw(session_id, frame_idx)
        if cached is not None:
            return cached

        # 2. Disk lookup
        session_dir = self.get_session_dir(session_id)
        candidate_paths = [
            session_dir / "frames" / f"{frame_idx:06d}.json",
            session_dir / f"frame_{frame_idx:06d}.json",
        ]

        file_path = None
        for p in candidate_paths:
            if p.exists() and p.stat().st_size > 0:
                file_path = p
                break

        if file_path is None:
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_json = f.read()

            self._put_lru_raw(session_id, frame_idx, raw_json)
            return raw_json
        except Exception as e:
            logger.error(f"Failed to read raw frame {frame_idx} for session {session_id}: {e}")
            return None

    def load_precomputed_frame(
        self, session_id: str, frame_idx: int
    ) -> Optional[ReplayFrameStreamPayload]:
        """Load a precomputed frame snapshot from memory or disk on demand."""
        # Check LRU payload cache
        cached = self._get_lru_payload(session_id, frame_idx)
        if cached is not None:
            return cached

        # Check raw JSON string
        raw_text = self.get_raw_frame_json(session_id, frame_idx)
        if raw_text is None:
            return None

        try:
            payload = ReplayFrameStreamPayload.model_validate_json(raw_text)
            self._put_lru_payload(session_id, frame_idx, payload)
            return payload
        except Exception as e:
            logger.error(f"Failed to parse precomputed frame {frame_idx} for session {session_id}: {e}")
            return None

    def save_precomputed_frame(
        self, session_id: str, frame_idx: int, payload: ReplayFrameStreamPayload
    ) -> None:
        """Save a precomputed frame snapshot to disk in compact JSON format."""
        session_dir = self.get_session_dir(session_id)
        frames_dir = session_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        path_in_frames = frames_dir / f"{frame_idx:06d}.json"
        path_in_root = session_dir / f"frame_{frame_idx:06d}.json"

        try:
            raw_json = payload.model_dump_json(exclude_none=True)
            with open(path_in_frames, "w", encoding="utf-8") as f:
                f.write(raw_json)

            # Also maintain frame_XXXXXX.json in root for legacy lookups
            with open(path_in_root, "w", encoding="utf-8") as f:
                f.write(raw_json)

            # Store in bounded LRU caches
            self._put_lru_raw(session_id, frame_idx, raw_json)
            self._put_lru_payload(session_id, frame_idx, payload)
        except Exception as e:
            logger.error(f"Failed to save precomputed frame {frame_idx} for session {session_id}: {e}")

    def record_failed_frame(
        self, session_id: str, frame_idx: int, frame_id: str, error_msg: str
    ) -> None:
        """Record failed frame into manifest for graceful fault-tolerant replay."""
        manifest = self.load_manifest(session_id) or {
            "session_id": session_id,
            "total_frames": 0,
            "failed_frames": [],
            "frames": [],
        }
        failed_list = manifest.setdefault("failed_frames", [])
        failed_list.append({
            "index": frame_idx,
            "frame_id": frame_id,
            "error": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.save_manifest(session_id, manifest)

    def _create_unavailable_stub(
        self, session: Any, frame_idx: int, error_msg: str
    ) -> ReplayFrameStreamPayload:
        """Create fallback unavailable frame payload for failed frames."""
        frame_meta = session.frames_meta[frame_idx] if frame_idx < len(session.frames_meta) else {}
        fname = frame_meta.get("filename", f"{frame_idx:06d}.bin")
        return ReplayFrameStreamPayload(
            session_id=session.session_id,
            sequence_name=session.sequence_name,
            frame_index=frame_idx,
            total_frames=session.total_frames,
            frame_filename=fname,
            timestamp=f"Frame #{frame_idx:04d} (Unavailable)",
            point_count=0,
            points_sample=[],
            category_distribution={},
            coordinate_mode="local_frame",
            map_mode="local_only",
            data_mode="precomputed_labels",
            semantic_source="FRAME UNAVAILABLE",
            state="ready",
            processing_time_ms=0.0,
            is_available=False,
        )

    def preload_all_frames(
        self, session_id: str, total_frames: Optional[int] = None, preload_models: bool = False, max_preload: int = 100
    ) -> int:
        """Preload initial precomputed frames into RAM LRU cache for zero-latency startup.
        
        Safely bounds preloading up to max_preload (default 100) to prevent RAM exhaustion on 4,000+ frame datasets.
        """
        session_dir = self.get_session_dir(session_id)
        if not session_dir.exists():
            return 0

        completed_indices = sorted(list(self.get_completed_frame_indices(session_id)))
        if not completed_indices:
            return 0

        # Preload up to max_preload frames (or total_frames if smaller)
        limit = max_preload if total_frames is None or total_frames > max_preload else total_frames
        target_indices = completed_indices[:limit]
        loaded_count = 0

        for idx in target_indices:
            raw_text = self.get_raw_frame_json(session_id, idx)
            if raw_text is not None:
                loaded_count += 1
                if preload_models:
                    try:
                        self.load_precomputed_frame(session_id, idx)
                    except Exception:
                        pass

        logger.info(f"Preloaded {loaded_count} initial demo frames into LRU RAM cache for session {session_id}")
        return loaded_count

    def start_precompute_background(self, session: Any) -> PrecomputeProgressStatus:
        """Launch background asynchronous precomputation task if not already in progress."""
        session_id = session.session_id
        if session_id in self._active_tasks and not self._active_tasks[session_id].done():
            prog = self.get_progress(session_id)
            if prog:
                return prog

        task = asyncio.create_task(self.precompute_sequence_async(session))
        self._active_tasks[session_id] = task

        completed_indices = self.get_completed_frame_indices(session_id)
        processed = len(completed_indices)
        total = session.total_frames
        pct = round((processed / total * 100.0), 2) if total > 0 else 0.0

        return PrecomputeProgressStatus(
            session_id=session_id,
            sequence_name=session.sequence_name,
            processed_frames=processed,
            total_frames=total,
            percent_complete=pct,
            is_complete=processed >= total and total > 0,
            is_running=True,
            elapsed_seconds=0.0,
            eta_seconds=0.0,
            current_stage="Fast-FRNet inference",
            failed_count=0,
        )

    def cancel_precompute(self, session_id: str) -> bool:
        """Cancel an active sequence precomputation task."""
        task = self._active_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
            if session_id in self._progress:
                self._progress[session_id].is_running = False
                self._progress[session_id].current_stage = "cancelled"
                self._progress[session_id].error_message = "Cancelled by user"
            logger.info(f"Precomputation task cancelled for session {session_id}")
            return True
        return False

    async def precompute_sequence_async(
        self,
        session: Any,  # ReplaySession instance
        progress_callback: Optional[Callable[[PrecomputeProgressStatus], None]] = None,
    ) -> PrecomputeProgressStatus:
        """Execute resumable precomputation across all sequence frames asynchronously.
        
        Processes frame-by-frame with memory safety, discarding raw point arrays,
        periodically releasing garbage, and maintaining manifest.json state.
        """
        session_id = session.session_id
        total_frames = session.total_frames
        seq_name = session.sequence_name
        content_hash = getattr(session, "content_hash", None)

        # 1. Initialize or load manifest
        manifest = self.load_manifest(session_id) or {
            "session_id": session_id,
            "sequence_name": seq_name,
            "total_frames": total_frames,
            "content_hash": content_hash,
            "status": "in_progress",
            "completed_count": 0,
            "failed_frames": [],
            "frames": [
                {
                    "index": i,
                    "frame_id": session.frames_meta[i]["filename"] if i < len(session.frames_meta) else f"{i:06d}.bin",
                    "result_file": f"frames/{i:06d}.json",
                    "status": "pending",
                }
                for i in range(total_frames)
            ],
        }

        # 2. Check already-completed frames for resumable continuation
        completed_indices = self.get_completed_frame_indices(session_id)
        for idx in completed_indices:
            if idx < len(manifest.get("frames", [])):
                manifest["frames"][idx]["status"] = "completed"

        already_completed = len(completed_indices)
        logger.info(
            f"Precompute session {session_id}: {already_completed}/{total_frames} frames already completed on disk. Resuming."
        )

        status = PrecomputeProgressStatus(
            session_id=session_id,
            sequence_name=seq_name,
            processed_frames=already_completed,
            total_frames=total_frames,
            percent_complete=round((already_completed / total_frames * 100.0), 2) if total_frames > 0 else 0.0,
            is_complete=already_completed >= total_frames and total_frames > 0,
            is_running=True,
            elapsed_seconds=0.0,
            eta_seconds=0.0,
            current_stage="Fast-FRNet inference",
            failed_count=len(manifest.get("failed_frames", [])),
        )
        self._progress[session_id] = status

        if status.is_complete:
            status.is_running = False
            manifest["status"] = "completed"
            manifest["completed_count"] = total_frames
            self.save_manifest(session_id, manifest)
            return status

        start_time = time.perf_counter()
        # Dedicated multi-object tracker instance isolated from live session
        from .object_tracking import ObjectTracker
        isolated_tracker = ObjectTracker(max_association_distance_m=4.5, max_missed_frames=3)

        # Pre-load required Fast-FRNet model once
        try:
            target_model = getattr(session, "target_model_type", "semantickitti")
            from .fast_frnet_inference import FastFRNetInferenceService
            FastFRNetInferenceService.load_model(target_model)
            logger.info(f"Fast-FRNet ({target_model.upper()}) cached once in memory for precompute session {session_id}")
        except Exception as e:
            logger.warning(f"Fast-FRNet pre-cache warning: {e}")

        frames_processed_this_run = 0

        try:
            for i in range(total_frames):
                # Skip already completed frames for instant resumption
                if i in completed_indices:
                    continue

                status.current_stage = f"Fast-FRNet neural inference (Frame {i + 1}/{total_frames})"
                
                try:
                    # Process perception pipeline in thread pool
                    payload = await asyncio.wait_for(
                        asyncio.to_thread(
                            session._process_frame_internal,
                            i,
                            update_tracker=True,
                            cache_result=False,
                            ignore_disk_cache=True,
                            custom_tracker=isolated_tracker,
                        ),
                        timeout=45.0,
                    )
                    
                    status.current_stage = f"Serializing 2.5D map & results (Frame {i + 1}/{total_frames})"
                    self.save_precomputed_frame(session_id, i, payload)
                    completed_indices.add(i)
                    if i < len(manifest["frames"]):
                        manifest["frames"][i]["status"] = "completed"

                except Exception as e:
                    logger.error(f"Frame {i} processing failed for session {session_id}: {e}", exc_info=True)
                    self.record_failed_frame(
                        session_id,
                        i,
                        session.frames_meta[i]["filename"] if i < len(session.frames_meta) else f"{i:06d}.bin",
                        str(e),
                    )
                    stub = self._create_unavailable_stub(session, i, str(e))
                    self.save_precomputed_frame(session_id, i, stub)
                    completed_indices.add(i)
                    if i < len(manifest["frames"]):
                        manifest["frames"][i]["status"] = "failed"
                    manifest.setdefault("failed_frames", []).append({
                        "index": i,
                        "frame_id": session.frames_meta[i]["filename"] if i < len(session.frames_meta) else f"{i:06d}.bin",
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                frames_processed_this_run += 1
                elapsed = time.perf_counter() - start_time
                total_done = len(completed_indices)
                percent = round((total_done / total_frames) * 100.0, 2)
                fps = frames_processed_this_run / elapsed if elapsed > 0 else 1.0
                eta = (total_frames - total_done) / fps if fps > 0 else 0.0

                status.processed_frames = total_done
                status.percent_complete = percent
                status.elapsed_seconds = round(elapsed, 1)
                status.eta_seconds = round(eta, 1)
                status.failed_count = len(manifest.get("failed_frames", []))

                # Periodic manifest flush and memory release every 25 frames
                if frames_processed_this_run % 25 == 0 or total_done >= total_frames:
                    manifest["completed_count"] = total_done
                    self.save_manifest(session_id, manifest)
                    gc.collect()

                if progress_callback:
                    progress_callback(status)

                # Yield control briefly every frame
                await asyncio.sleep(0.001)

            status.is_complete = True
            status.is_running = False
            status.percent_complete = 100.0
            status.eta_seconds = 0.0
            status.current_stage = "Precomputation complete"
            
            manifest["status"] = "completed"
            manifest["completed_count"] = total_frames
            self.save_manifest(session_id, manifest)
            logger.info(f"Precomputation completed for session {session_id} in {status.elapsed_seconds}s")
            return status

        except asyncio.CancelledError:
            logger.info(f"Precomputation cancelled for session {session_id}")
            status.is_running = False
            status.error_message = "Cancelled by user"
            manifest["status"] = "cancelled"
            manifest["completed_count"] = len(completed_indices)
            self.save_manifest(session_id, manifest)
            return status

        except Exception as e:
            logger.error(f"Error during precomputation for session {session_id}: {e}", exc_info=True)
            status.is_running = False
            status.error_message = str(e)
            manifest["status"] = "failed"
            manifest["completed_count"] = len(completed_indices)
            self.save_manifest(session_id, manifest)
            return status


precompute_service = PrecomputeReplayService()

