"""Compact existing precomputed sequence frame snapshots on disk for 60 FPS replay."""

import os
import json
import time
from pathlib import Path

backend_dir = Path(__file__).resolve().parent
precomputed_root = backend_dir / "data" / "precomputed"

def compact_frame_dict(d: dict) -> dict:
    # 1. Prune terrain cells (keep summary & reports)
    if "terrain_result" in d and d["terrain_result"]:
        d["terrain_result"]["cells"] = []

    # 2. Prune sample_points on object instances
    if "objects_result" in d and d["objects_result"] and "instances" in d["objects_result"]:
        for inst in d["objects_result"]["instances"]:
            inst["sample_points"] = []

    # 3. Downsample points to 1500 points with rounded floats
    if "points_sample" in d:
        pts = d["points_sample"][:1500]
        for p in pts:
            p["x"] = round(float(p["x"]), 2)
            p["y"] = round(float(p["y"]), 2)
            p["z"] = round(float(p["z"]), 2)
            p["intensity"] = round(float(p.get("intensity", 0.5)), 2)
        d["points_sample"] = pts

    # 4. Foveated 2.5D grid cells sampling: top ~800 cells
    if "map_result" in d and d["map_result"] and "cells_sample" in d["map_result"]:
        raw_cells = d["map_result"]["cells_sample"]
        fine = [c for c in raw_cells if c.get("level") == "fine"]
        med = [c for c in raw_cells if c.get("level") == "medium"]
        coarse = [c for c in raw_cells if c.get("level") == "coarse"]

        selected = fine[:450] + med[:200] + coarse[:150]
        trimmed_cells = []
        for c in selected:
            trimmed = {
                "cell_key": c["cell_key"],
                "level": c["level"],
                "size_m": round(float(c["size_m"]), 2),
                "grid_x": int(c["grid_x"]),
                "grid_y": int(c["grid_y"]),
                "world_x": round(float(c["world_x"]), 2),
                "world_y": round(float(c["world_y"]), 2),
                "bounds": [round(float(b), 2) for b in c.get("bounds", [])],
                "observation_state": c.get("observation_state", "directly_observed"),
                "dominant_semantic_class": c.get("dominant_semantic_class", "unlabeled"),
                "dominant_category": c.get("dominant_category", "unknown"),
                "traversability_state": c.get("traversability_state", "drivable"),
                "is_static_obstacle": c.get("is_static_obstacle", False),
                "is_dynamic_obstacle": c.get("is_dynamic_obstacle", False),
                "elevation_min": round(float(c.get("elevation_min", 0.0)), 2),
                "elevation_mean": round(float(c.get("elevation_mean", 0.0)), 2),
                "elevation_max": round(float(c.get("elevation_max", 0.0)), 2),
                "elevation_variation": round(float(c.get("elevation_variation", 0.0)), 2),
                "roughness_summary": round(float(c.get("roughness_summary", 0.0)), 2),
                "slope_summary": round(float(c.get("slope_summary", 0.0)), 2),
                "point_count": int(c.get("point_count", 0)),
            }
            trimmed_cells.append(trimmed)
        d["map_result"]["cells_sample"] = trimmed_cells

    return d

def main():
    if not precomputed_root.exists():
        print("No precomputed directory found.")
        return

    sessions = [p for p in precomputed_root.iterdir() if p.is_dir()]
    print(f"Found {len(sessions)} session directories in {precomputed_root}")

    total_converted = 0
    total_orig_bytes = 0
    total_new_bytes = 0

    t0 = time.perf_counter()

    for s in sessions:
        frame_files = list(s.glob("frame_*.json"))
        if not frame_files:
            continue
        print(f"Compacting {len(frame_files)} frames in '{s.name}'...")
        for f in frame_files:
            try:
                orig_sz = f.stat().st_size
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                compacted = compact_frame_dict(data)
                compact_text = json.dumps(compacted, separators=(",", ":"))
                with open(f, "w", encoding="utf-8") as fp:
                    fp.write(compact_text)
                new_sz = len(compact_text)
                total_orig_bytes += orig_sz
                total_new_bytes += new_sz
                total_converted += 1
            except Exception as e:
                print(f"Error compacting {f}: {e}")

    dt = time.perf_counter() - t0
    orig_mb = total_orig_bytes / (1024 * 1024)
    new_mb = total_new_bytes / (1024 * 1024)
    reduction = ((total_orig_bytes - total_new_bytes) / total_orig_bytes * 100.0) if total_orig_bytes > 0 else 0

    print(f"\nCOMPACTION COMPLETE in {dt:.2f}s:")
    print(f"- Total frames compacted: {total_converted}")
    print(f"- Original size: {orig_mb:.2f} MB")
    print(f"- Compacted size: {new_mb:.2f} MB ({new_mb/max(1, total_converted)*1024:.1f} KB/frame)")
    print(f"- Storage & Bandwidth Reduction: {reduction:.1f}%")

if __name__ == "__main__":
    main()
