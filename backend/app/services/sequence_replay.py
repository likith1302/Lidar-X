"""LiDAR Sequence Replay Ingestion and Package Validator.

Handles ZIP package extraction, SemanticKITTI float32 validation,
natural numerical frame sorting, and SalsaNext prediction pairing.
"""

import os
import re
import zipfile
import uuid
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from ..config import settings

logger = logging.getLogger(__name__)


def natural_sort_key(s: str) -> List[Any]:
    """Helper for natural alphanumeric sorting (e.g. 000001 < 000002 < 000010)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


class SequenceIngestError(Exception):
    """Raised when sequence validation or extraction fails."""
    pass


import hashlib
import json
import shutil
from typing import Union

class SequenceReplayService:
    """Service to parse, validate, and index SemanticKITTI sequence archives."""

    @staticmethod
    def _compute_source_hash(source: Union[bytes, Path, str]) -> str:
        """Compute stable 16-char hex digest of zip bytes or file on disk without excessive RAM."""
        hasher = hashlib.sha256()
        if isinstance(source, bytes):
            # Compute hash in 1MB chunks
            chunk_size = 1024 * 1024
            for i in range(0, len(source), chunk_size):
                hasher.update(source[i : i + chunk_size])
        else:
            path = Path(source)
            with open(path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    hasher.update(chunk)
        return hasher.hexdigest()[:16]

    @staticmethod
    def _compute_zip_hash(zip_bytes: bytes) -> str:
        """Backward-compatible helper."""
        return SequenceReplayService._compute_source_hash(zip_bytes)[:12]

    @classmethod
    def extract_and_validate_zip(
        cls, zip_source: Union[bytes, Path, str], session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract sequence ZIP into target session directory and validate scans/predictions.
        
        Supports bytes or Path/str directly to safely ingest multi-GB archives without RAM exhaustion.
        """
        if isinstance(zip_source, bytes):
            if not zip_source or len(zip_source) == 0:
                raise SequenceIngestError("Empty ZIP file payload received.")
            source_size = len(zip_source)
            zip_hash = cls._compute_source_hash(zip_source)
        else:
            source_path = Path(zip_source)
            if not source_path.exists() or not source_path.is_file():
                raise SequenceIngestError(f"ZIP archive not found at path: {source_path}")
            source_size = source_path.stat().st_size
            if source_size == 0:
                raise SequenceIngestError(f"Empty ZIP file at path: {source_path}")
            zip_hash = cls._compute_source_hash(source_path)

        if not session_id:
            session_id = f"seq_{zip_hash}"

        session_dir = settings.SEQUENCES_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        zip_target_path = session_dir / "archive.zip"
        
        # Determine if extraction is needed
        existing_bins = [p for p in session_dir.rglob("*.bin") if p.is_file()]
        should_extract = True
        
        if existing_bins and zip_target_path.exists():
            if zip_target_path.stat().st_size == source_size:
                should_extract = False
        elif existing_bins and not isinstance(zip_source, bytes):
            # If already extracted from this path with valid scans
            should_extract = False

        if should_extract:
            if isinstance(zip_source, bytes):
                with open(zip_target_path, "wb") as f:
                    f.write(zip_source)
                unpack_from = zip_target_path
            else:
                unpack_from = Path(zip_source)
                # If source is external, copy to archive.zip only if not already same file
                if unpack_from.resolve() != zip_target_path.resolve():
                    try:
                        shutil.copyfile(unpack_from, zip_target_path)
                    except Exception as e:
                        logger.warning(f"Could not copy source zip into session dir: {e}")
                        unpack_from = Path(zip_source)

            # Unpack ZIP archive with streaming decompression
            try:
                with zipfile.ZipFile(unpack_from, "r") as zf:
                    zf.extractall(session_dir)
            except zipfile.BadZipFile as e:
                raise SequenceIngestError(f"Malformed or corrupt ZIP archive: {str(e)}")

        # Index the directory contents using recursive discovery
        result = cls._index_directory(session_dir, session_id=session_id)
        result["content_hash"] = zip_hash
        return result

    @classmethod
    def load_existing_session(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """Load and index an existing session from disk, zip archive, or demo cache."""
        # 0. Check if session_id is a direct path to a directory or zip file on disk
        try:
            direct_path = Path(session_id)
            if direct_path.exists():
                if direct_path.is_dir():
                    return cls._index_directory(direct_path, session_id=direct_path.name)
                elif direct_path.is_file() and direct_path.suffix.lower() == ".zip":
                    return cls.extract_and_validate_zip(direct_path, session_id=direct_path.stem)
        except Exception:
            pass

        # 1. Check demo_cache first for precomputed sequence manifest
        demo_cache_dir = settings.DEMO_CACHE_DIR / session_id
        if not demo_cache_dir.exists():
            repo_demo = Path(__file__).resolve().parent.parent.parent / "data" / "demo_cache" / session_id
            if repo_demo.exists():
                demo_cache_dir = repo_demo

        if demo_cache_dir.exists():
            manifest_path = demo_cache_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    total_f = manifest.get("total_frames", 0)
                    if total_f > 0:
                        validated_frames = [
                            {
                                "frame_index": i,
                                "filename": frame_item.get("frame_id", f"{i:06d}.bin"),
                                "bin_path": str(demo_cache_dir / frame_item.get("result_file", f"frames/{i:06d}.json")),
                                "label_path": None,
                                "point_count": frame_item.get("point_count", 100000),
                                "has_prediction": True,
                            }
                            for i, frame_item in enumerate(manifest.get("frames", []))
                        ]
                        return {
                            "session_id": session_id,
                            "session_dir": str(demo_cache_dir),
                            "sequence_name": manifest.get("sequence_name", session_id),
                            "total_frames": total_f,
                            "has_predictions": True,
                            "data_mode": "precomputed_labels",
                            "playback_mode": "offline_precomputed_replay",
                            "frames": validated_frames,
                            "content_hash": manifest.get("content_hash"),
                        }
                except Exception as e:
                    logger.warning(f"Error loading demo cache manifest for {session_id}: {e}")

        session_dir = settings.SEQUENCES_DIR / session_id
        if not session_dir.exists():
            repo_seq = Path(__file__).resolve().parent.parent.parent / "data" / "sequences" / session_id
            if repo_seq.exists():
                session_dir = repo_seq

        # 2. Check if session directory exists with scans or archive.zip
        if session_dir.exists():
            bin_files = list(session_dir.rglob("*.bin"))
            if bin_files:
                try:
                    return cls._index_directory(session_dir, session_id=session_id)
                except Exception as e:
                    logger.warning(f"Failed to index session directory {session_dir}: {e}")

            zip_path = session_dir / "archive.zip"
            if zip_path.exists():
                try:
                    return cls.extract_and_validate_zip(zip_path, session_id=session_id)
                except Exception as e:
                    logger.warning(f"Failed to load session from {zip_path}: {e}")

        # 3. Check candidate ZIP locations
        candidate_zips = [
            settings.BASE_DIR / f"{session_id}.zip",
            settings.BASE_DIR.parent / f"{session_id}.zip",
            Path(f"{session_id}.zip"),
        ]
        for candidate_zip in candidate_zips:
            if candidate_zip.exists():
                try:
                    return cls.extract_and_validate_zip(candidate_zip, session_id=session_id)
                except Exception as e:
                    logger.warning(f"Failed to extract candidate zip {candidate_zip}: {e}")

        # 4. Check legacy precomputed directory as fallback
        precomputed_dir = settings.DATA_DIR / "precomputed" / session_id
        if precomputed_dir.exists():
            precomp_frames = sorted(list(precomputed_dir.glob("frame_*.json")), key=lambda p: natural_sort_key(p.name))
            if precomp_frames:
                validated_frames = [
                    {
                        "frame_index": i,
                        "filename": f"{i:06d}.bin",
                        "bin_path": str(f_path),
                        "label_path": None,
                        "point_count": 100000,
                        "has_prediction": True,
                    }
                    for i, f_path in enumerate(precomp_frames)
                ]
                return {
                    "session_id": session_id,
                    "session_dir": str(precomputed_dir),
                    "sequence_name": f"{session_id} (Precomputed)",
                    "total_frames": len(validated_frames),
                    "has_predictions": True,
                    "data_mode": "precomputed_labels",
                    "playback_mode": "offline_precomputed_replay",
                    "frames": validated_frames,
                }

        return None

    @classmethod
    def _index_directory(cls, session_dir: Path, session_id: str) -> Dict[str, Any]:
        """Index an already-extracted session directory with recursive scan discovery and natural sorting."""
        # Recursive discovery of all .bin files
        all_bin_files = [p for p in session_dir.rglob("*.bin") if p.is_file()]
        if not all_bin_files:
            raise SequenceIngestError(
                f"No SemanticKITTI float32 .bin scans found in {session_dir.name}."
            )

        # Sort all discovered scans strictly in natural numerical order
        raw_bin_files = sorted(all_bin_files, key=lambda p: natural_sort_key(p.name))

        # Build lookup table for matching predictions / ground truth labels
        label_lookup: Dict[str, Path] = {}
        for p in session_dir.rglob("*.label"):
            if p.is_file():
                label_lookup[p.stem] = p
        for p in session_dir.rglob("*.npz"):
            if p.is_file() and p.stem not in label_lookup:
                label_lookup[p.stem] = p

        # Locate pose, calibration, and timing files
        poses_file = next(session_dir.rglob("poses.txt"), None)
        calib_file = next(session_dir.rglob("calib.txt"), None)
        times_file = next(session_dir.rglob("times.txt"), None)

        validated_frames: List[Dict[str, Any]] = []
        has_predictions = False

        for idx, bin_file in enumerate(raw_bin_files):
            file_size = bin_file.stat().st_size
            if file_size == 0 or file_size % 16 != 0:
                logger.warning(f"Skipping invalid scan {bin_file.name}: size {file_size} is not multiple of 16 bytes")
                continue

            stem = bin_file.stem
            matching_label_file: Optional[Path] = label_lookup.get(stem)
            
            if matching_label_file:
                has_predictions = True

            validated_frames.append({
                "frame_index": len(validated_frames),
                "filename": bin_file.name,
                "bin_path": str(bin_file),
                "label_path": str(matching_label_file) if matching_label_file else None,
                "point_count": file_size // 16,
                "has_prediction": matching_label_file is not None,
            })

        if not validated_frames:
            raise SequenceIngestError("No valid SemanticKITTI float32 .bin scans found in archive.")

        # Determine sequence name from directory structure or session_id
        first_bin = raw_bin_files[0]
        parent_name = first_bin.parent.name
        grandparent_name = first_bin.parent.parent.name if first_bin.parent.parent != session_dir else None
        
        if grandparent_name and grandparent_name not in ("sequences", "archive", session_dir.name):
            seq_name = f"{grandparent_name}_{parent_name}"
        elif parent_name not in ("velodyne", "scans", "bins", session_dir.name):
            seq_name = parent_name
        else:
            seq_name = session_id

        data_mode = "precomputed_labels" if has_predictions else "live_inference"

        return {
            "session_id": session_id,
            "session_dir": str(session_dir),
            "sequence_name": seq_name,
            "total_frames": len(validated_frames),
            "has_predictions": has_predictions,
            "has_poses": poses_file is not None,
            "poses_file": str(poses_file) if poses_file else None,
            "has_calibration": calib_file is not None,
            "calib_file": str(calib_file) if calib_file else None,
            "has_timestamps": times_file is not None,
            "times_file": str(times_file) if times_file else None,
            "data_mode": data_mode,
            "frames": validated_frames,
        }


sequence_replay_service = SequenceReplayService()


