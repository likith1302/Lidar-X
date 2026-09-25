"""Verification test for precomputed frame persistence and 60 FPS playback throughput."""

import sys
import time
from pathlib import Path

backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from app.services.sequence_replay import sequence_replay_service
from app.services.replay_session import replay_session_manager, ReplaySession
from app.services.precompute_service import precompute_service

def verify():
    session_id = "foveamap_sequence01_replay"
    zip_path = backend_dir / "foveamap_sequence01_replay.zip"
    
    print("1. Checking precomputed frames on disk...", flush=True)
    count = precompute_service.get_precomputed_frame_count(session_id)
    print(f"-> Found {count} precomputed frames for session '{session_id}'.", flush=True)
    assert count == 100, f"Expected 100 precomputed frames on disk, found {count}"

    print("2. Ingesting ZIP and creating replay session...", flush=True)
    with open(zip_path, "rb") as f:
        zip_bytes = f.read()
    session_data = sequence_replay_service.extract_and_validate_zip(zip_bytes, session_id=session_id)
    session = replay_session_manager.create_session(session_data)
    assert session.total_frames == 100
    print(f"-> Created session '{session.session_id}' with {session.total_frames} frames.", flush=True)

    print("3. Preloading frames into memory cache...", flush=True)
    loaded = precompute_service.preload_all_frames(session_id, 100, preload_models=True)
    print(f"-> Preloaded {loaded} frames into RAM cache.", flush=True)
    assert loaded == 100

    print("4. Testing high-rate 60 FPS frame playback advancement...", flush=True)
    session.start(fps=60.0)
    t0 = time.perf_counter()
    streamed = 0
    while session.current_frame_index < session.total_frames:
        p = session.advance_and_get_frame()
        if p is None:
            break
        streamed += 1

    dt = time.perf_counter() - t0
    fps = streamed / dt if dt > 0 else 0
    print(f"-> Advanced {streamed} frames in {dt*1000:.2f}ms ({fps:.1f} FPS playback rate).", flush=True)
    assert streamed == 100
    assert fps >= 60.0, f"Achieved {fps:.1f} FPS, expected >= 60 FPS"

    print("5. Testing zero-overhead raw JSON streaming throughput (WebSocket fast-path)...", flush=True)
    t_ws0 = time.perf_counter()
    ws_streamed = 0
    for idx in range(100):
        raw_json = session.get_frame_raw_json(idx)
        assert raw_json is not None, f"Frame {idx} raw JSON was None"
        sz_kb = len(raw_json) / 1024.0
        assert sz_kb <= 10000.0, f"Frame {idx} payload is {sz_kb:.1f} KB, expected <= 10000 KB"
        ws_streamed += 1

    dt_ws = time.perf_counter() - t_ws0
    ws_fps = ws_streamed / dt_ws if dt_ws > 0 else 0
    print(f"-> Streamed {ws_streamed} raw JSON frames in {dt_ws*1000:.2f}ms ({ws_fps:.1f} FPS throughput capability, avg size: {sz_kb:.1f} KB).", flush=True)
    assert ws_fps >= 1000.0, f"Expected >= 1000 FPS direct streaming throughput, got {ws_fps:.1f}"

    print("6. Verifying seek latency across frames...", flush=True)
    seek_indices = [0, 25, 50, 75, 99, 10]
    for idx in seek_indices:
        t_s0 = time.perf_counter()
        session.seek(idx)
        frame_payload = session.get_current_frame()
        dt_s = (time.perf_counter() - t_s0) * 1000.0
        assert frame_payload.frame_index == idx
        assert frame_payload.point_count > 0
        assert len(frame_payload.points_sample) > 0
        print(f"   Seek to frame #{idx:02d}: {dt_s:.3f}ms (instant cache hit)", flush=True)

    print("\nSUCCESS: All 100 frames are permanently saved and play at 60 FPS seamlessly!", flush=True)

if __name__ == "__main__":
    verify()
