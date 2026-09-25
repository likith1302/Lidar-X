"""Offline Recomputation Script for SemanticKITTI Sequence 00.

Processes every frame from raw .bin LiDAR points through the complete perception pipeline:
Raw .bin -> Points -> Scene/Preprocessing -> Fast-FRNet -> Semantic Mapping ->
Instance Clustering (with merging & duplicate suppression) -> Kalman MOT ->
Terrain Analysis -> Adaptive Foveated 2.5D Grid -> Performance Metrics -> JSON Snapshot.
"""

import sys
import os
import time
import json
import gc
import argparse
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR.parent))

from backend.app.config import settings
from backend.app.services.sequence_replay import sequence_replay_service
from backend.app.services.replay_session import ReplaySession
from backend.app.services.precompute_service import precompute_service
from backend.app.services.fast_frnet_inference import FastFRNetInferenceService
from backend.app.services.object_tracking import ObjectTracker


def main():
    parser = argparse.ArgumentParser(description="Recompute SemanticKITTI sequence demo cache from scratch")
    parser.add_argument("--session-id", default="semantic_kitti_sequence_00", help="Session ID")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--count", type=int, default=-1, help="Number of frames to process (-1 for all)")
    parser.add_argument("--threads", type=int, default=12, help="PyTorch CPU threads")
    parser.add_argument("--batch-flush", type=int, default=10, help="Manifest flush frequency")
    parser.add_argument("--limit-manifest", action="store_true", help="Limit manifest total_frames to count")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    seq_dir = settings.SEQUENCES_DIR / args.session_id
    if not seq_dir.exists():
        print(f"Error: Sequence directory not found: {seq_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Indexing raw LiDAR scans from: {seq_dir}")
    session_data = sequence_replay_service._index_directory(seq_dir, session_id=args.session_id)
    total_frames = session_data["total_frames"]
    effective_total = min(total_frames, args.count) if (args.limit_manifest and args.count > 0) else total_frames
    print(f"[*] Total frames found in sequence: {total_frames} (target demo frames: {effective_total})")

    session = ReplaySession(session_data)
    session.is_demo = True
    session.playback_mode = "offline_precomputed_replay"

    session_cache_dir = settings.DEMO_CACHE_DIR / args.session_id
    frames_cache_dir = session_cache_dir / "frames"
    frames_cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = session_cache_dir / "manifest.json"
    manifest = precompute_service.load_manifest(args.session_id) or {
        "session_id": args.session_id,
        "sequence_name": session_data.get("sequence_name", args.session_id),
        "total_frames": effective_total,
        "status": "in_progress",
        "completed_count": 0,
        "failed_frames": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frames": [
            {
                "index": i,
                "frame_id": session_data["frames"][i]["filename"],
                "result_file": f"frames/{i:06d}.json",
                "status": "pending",
            }
            for i in range(effective_total)
        ],
    }

    if args.limit_manifest and args.count > 0:
        manifest["total_frames"] = effective_total
        manifest["frames"] = manifest["frames"][:effective_total]

    completed_indices = precompute_service.get_completed_frame_indices(args.session_id)
    print(f"[*] Existing precomputed frames on disk: {len(completed_indices)}/{effective_total}")

    print("[*] Pre-loading Fast-FRNet (SemanticKITTI) model...")
    FastFRNetInferenceService.load_model("semantickitti")
    print("[*] Model loaded successfully.")

    tracker = ObjectTracker(max_association_distance_m=4.5, max_missed_frames=3)

    start_idx = max(0, args.start)
    end_idx = effective_total if args.count <= 0 else min(effective_total, start_idx + args.count)
    target_count = end_idx - start_idx
    print(f"[*] Processing frames {start_idx} through {end_idx - 1} ({target_count} frames)...")

    start_time = time.perf_counter()
    processed_count = 0

    for i in range(start_idx, end_idx):
        if i in completed_indices:
            continue

        f_start = time.perf_counter()
        try:
            payload = session._process_frame_internal(
                i,
                update_tracker=True,
                cache_result=False,
                ignore_disk_cache=True,
                custom_tracker=tracker,
            )

            precompute_service.save_precomputed_frame(args.session_id, i, payload)
            completed_indices.add(i)
            if i < len(manifest["frames"]):
                manifest["frames"][i]["status"] = "completed"

            f_duration_ms = (time.perf_counter() - f_start) * 1000.0
            processed_count += 1

            insts = payload.objects_result.instances if payload.objects_result else []
            motorcyclists = [inst for inst in insts if inst.object_type in ("motorcycle", "motorcyclist")]

            elapsed = time.perf_counter() - start_time
            fps = processed_count / elapsed if elapsed > 0 else 1.0
            remaining = end_idx - i - 1
            eta_sec = remaining / fps if fps > 0 else 0.0

            print(
                f"[PROGRESS] Frame {i+1:04d}/{total_frames} ({((i+1)/total_frames)*100:.1f}%) | "
                f"{f_duration_ms:.0f}ms | Objects: {len(insts)} | Moto: {len(motorcyclists)} | "
                f"Speed: {fps:.2f} FPS | ETA: {eta_sec/60:.1f}m",
                flush=True,
            )

        except Exception as e:
            print(f"[ERROR] Frame {i} failed: {e}", file=sys.stderr, flush=True)
            precompute_service.record_failed_frame(
                args.session_id,
                i,
                session.frames_meta[i]["filename"] if i < len(session.frames_meta) else f"{i:06d}.bin",
                str(e),
            )
            stub = precompute_service._create_unavailable_stub(session, i, str(e))
            precompute_service.save_precomputed_frame(args.session_id, i, stub)
            completed_indices.add(i)
            if i < len(manifest["frames"]):
                manifest["frames"][i]["status"] = "failed"

        if processed_count % args.batch_flush == 0 or i == end_idx - 1:
            manifest["completed_count"] = len(completed_indices)
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            if len(completed_indices) >= total_frames:
                manifest["status"] = "completed"
            precompute_service.save_manifest(args.session_id, manifest)
            gc.collect()

    manifest["completed_count"] = len(completed_indices)
    if len(completed_indices) >= total_frames:
        manifest["status"] = "completed"
    precompute_service.save_manifest(args.session_id, manifest)

    total_elapsed = time.perf_counter() - start_time
    print(f"\n[DONE] Finished processing {processed_count} frames in {total_elapsed:.1f}s.")
    print(f"[*] Total completed frames on disk: {len(completed_indices)}/{total_frames}")


if __name__ == "__main__":
    main()
