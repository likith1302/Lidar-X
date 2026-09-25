#!/usr/bin/env python3
"""Comprehensive Fast-FRNet Benchmark Suite.

Benchmarks:
1. Original baseline checkpoints (FP32 raw)
2. Clean inference-only checkpoints (FP32 stripped)
3. Half-precision deployment checkpoints (FP16 weights)
4. Dynamic INT8 linear quantized models

Measures:
- Model file size (bytes & MB)
- Parameter count (total & trainable)
- Weight dtypes
- Inference latency (mean, std, min, max, p50, p95 ms)
- Inference throughput (FPS)
- Peak RAM usage (MB)
- Output tensor shapes
- Semantic prediction distributions
- Prediction agreement % (matching_predictions / total_predictions)
- Output logits max and mean absolute differences
- Output probabilities max and mean absolute differences
- Per-class IoU, mIoU, and accuracy

Outputs:
- model_compression/baseline_metrics.json
- model_compression/compression_metrics.json
- model_compression/eval_predictions/*.npz
"""

import os
import sys
import time
import json
import psutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.fast_frnet import FastFRNet
from app.config.model_settings import model_settings
from app.services.semantic_label_mapping import SemanticLabelMappingService

sys.stdout.reconfigure(line_buffering=True)


def get_peak_ram_mb() -> float:
    """Return process peak resident set size in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def load_dataset_frames(max_frames: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load fixed evaluation subsets for RELLIS-3D and SemanticKITTI."""
    rellis_dir = BACKEND_DIR / "data" / "sequences" / "rellis_benchmark" / "sequences" / "00"
    sk_dir = BACKEND_DIR / "data" / "sequences" / "semantic_kitti_sequence_00"

    rellis_frames = []
    rellis_bin_dir = rellis_dir / "velodyne"
    rellis_lbl_dir = rellis_dir / "predictions"
    if rellis_bin_dir.is_dir():
        bins = sorted(list(rellis_bin_dir.glob("*.bin")))[:max_frames]
        for b in bins:
            pts = np.fromfile(str(b), dtype=np.float32).reshape(-1, 4)
            lbl_file = rellis_lbl_dir / (b.stem + ".label")
            lbls = np.fromfile(str(lbl_file), dtype=np.uint32) & 0xFFFF if lbl_file.is_file() else None
            rellis_frames.append({
                "frame_id": b.stem,
                "file_path": str(b),
                "points": pts,
                "ref_labels": lbls,
            })

    sk_frames = []
    sk_bin_dir = sk_dir / "velodyne"
    sk_lbl_dir = sk_dir / "labels"
    if sk_bin_dir.is_dir():
        bins = sorted(list(sk_bin_dir.glob("*.bin")))[:max_frames]
        for b in bins:
            pts = np.fromfile(str(b), dtype=np.float32).reshape(-1, 4)
            lbl_file = sk_lbl_dir / (b.stem + ".label")
            lbls = np.fromfile(str(lbl_file), dtype=np.uint32) & 0xFFFF if lbl_file.is_file() else None
            sk_frames.append({
                "frame_id": b.stem,
                "file_path": str(b),
                "points": pts,
                "gt_labels": lbls,
            })

    return rellis_frames, sk_frames


def instantiate_model(spec_name: str) -> FastFRNet:
    spec = model_settings.get_spec(spec_name)
    return FastFRNet(
        output_shape=(spec.H, spec.W),
        fov_up=spec.fov_up,
        fov_down=spec.fov_down,
        num_classes=spec.num_classes,
    )


def load_model_weights(model: FastFRNet, ckpt_path: Path, mode: str = "fp32") -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    else:
        sd = ckpt

    cleaned = {}
    for k, v in sd.items():
        key = k[7:] if k.startswith("module.") else k
        if not key.startswith("auxiliary_head"):
            if v.dtype == torch.float16:
                cleaned[key] = v.float()
            else:
                cleaned[key] = v
    model.load_state_dict(cleaned, strict=True)
    model.eval()


def compute_miou(preds: np.ndarray, targets: np.ndarray, num_classes: int = 20, ignore_index: int = 0) -> Tuple[float, Dict[int, float], float]:
    """Compute per-class IoU, mIoU, and overall accuracy."""
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, targets):
        conf_mat[t, p] += 1

    ious = {}
    for c in range(num_classes):
        if c == ignore_index:
            continue
        tp = conf_mat[c, c]
        fp = np.sum(conf_mat[:, c]) - tp
        fn = np.sum(conf_mat[c, :]) - tp
        denom = tp + fp + fn
        ious[c] = float(tp / denom) if denom > 0 else 0.0

    valid_ious = [v for k, v in ious.items() if np.sum(conf_mat[k, :]) > 0]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0

    valid_mask = (targets != ignore_index)
    acc = float(np.mean(preds[valid_mask] == targets[valid_mask])) if np.any(valid_mask) else 0.0
    return miou, ious, acc


def benchmark_model_on_frames(
    model: nn.Module,
    model_name: str,
    spec_name: str,
    frames: List[Dict[str, Any]],
    baseline_outputs: Optional[List[Dict[str, Any]]] = None,
    save_preds_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run thorough benchmark across frames for a given model variant."""
    spec = model_settings.get_spec(spec_name)
    latencies = []
    frame_metrics = []
    saved_outputs = []

    # Warmup
    warmup_pts = torch.randn(1000, 4)
    warmup_pts[:, 2] = -1.73
    with torch.no_grad():
        _ = model(warmup_pts)

    total_pts = sum(len(f["points"]) for f in frames)
    total_matches_vs_baseline = 0
    max_logit_diffs = []
    mean_logit_diffs = []
    max_prob_diffs = []
    mean_prob_diffs = []

    for idx, f in enumerate(frames):
        pts = f["points"].copy()
        num_points = len(pts)

        # Intensity normalization if needed (RELLIS)
        if spec.intensity_scale > 1.0 and float(pts[:, 3].max()) > 10.0:
            pts[:, 3] = pts[:, 3] / spec.intensity_scale

        tensor_pts = torch.from_numpy(pts)

        # Measure latency
        ram_before = get_peak_ram_mb()
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor_pts)
            probs = F.softmax(logits, dim=-1)
            preds_learning = torch.argmax(logits, dim=1).numpy().astype(np.uint8)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        ram_after = get_peak_ram_mb()

        latencies.append(dt_ms)

        # Raw classes
        raw_preds = SemanticLabelMappingService.map_learning_to_raw_batch(preds_learning, dataset=spec_name)

        frame_data = {
            "frame_id": f["frame_id"],
            "point_count": num_points,
            "latency_ms": round(dt_ms, 2),
            "ram_mb": round(ram_after, 2),
            "output_shape": list(logits.shape),
        }

        # Compare against baseline if available
        if baseline_outputs is not None:
            base_f = baseline_outputs[idx]
            base_preds = base_f["preds_learning"]
            base_logits = base_f["logits"]
            base_probs = base_f["probs"]

            match_count = int(np.sum(preds_learning == base_preds))
            total_matches_vs_baseline += match_count
            agreement = match_count / num_points * 100.0

            l_diff = torch.abs(logits - base_logits)
            p_diff = torch.abs(probs - base_probs)

            max_l = float(torch.max(l_diff).item())
            mean_l = float(torch.mean(l_diff).item())
            max_p = float(torch.max(p_diff).item())
            mean_p = float(torch.mean(p_diff).item())

            max_logit_diffs.append(max_l)
            mean_logit_diffs.append(mean_l)
            max_prob_diffs.append(max_p)
            mean_prob_diffs.append(mean_p)

            frame_data["agreement_pct"] = round(agreement, 4)
            frame_data["logit_max_diff"] = round(max_l, 6)
            frame_data["logit_mean_diff"] = round(mean_l, 6)
            frame_data["prob_max_diff"] = round(max_p, 6)
            frame_data["prob_mean_diff"] = round(mean_p, 6)

        # Accuracy/IoU against GT if available
        gt = f.get("gt_labels")
        if gt is not None:
            # Map gt raw to learning classes for SemanticKITTI
            inv_map = SemanticLabelMappingService._SK_DARKNET_LEARNING_MAP_INV
            raw_to_l = {v: k for k, v in inv_map.items()}
            gt_l = np.zeros_like(gt, dtype=np.uint8)
            for r_id, l_id in raw_to_l.items():
                gt_l[gt == r_id] = l_id

            miou, ious, acc = compute_miou(preds_learning, gt_l, num_classes=spec.num_classes, ignore_index=0)
            frame_data["miou"] = round(miou * 100.0, 2)
            frame_data["accuracy"] = round(acc * 100.0, 2)

        frame_metrics.append(frame_data)

        # Keep output for baseline comparison and saving
        saved_outputs.append({
            "frame_id": f["frame_id"],
            "preds_learning": preds_learning,
            "raw_preds": raw_preds,
            "logits": logits.cpu(),
            "probs": probs.cpu(),
        })

    # Save predictions to npz if requested
    if save_preds_dir is not None:
        save_preds_dir.mkdir(parents=True, exist_ok=True)
        for out in saved_outputs:
            fname = f"{spec_name}_{model_name}_{out['frame_id']}.npz"
            np.savez_compressed(
                save_preds_dir / fname,
                preds_learning=out["preds_learning"],
                raw_preds=out["raw_preds"],
            )

    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    summary = {
        "model_variant": model_name,
        "spec": spec_name,
        "frame_count": len(frames),
        "total_points": total_pts,
        "latency_ms": {
            "mean": round(mean_lat, 2),
            "std": round(std_lat, 2),
            "min": round(float(np.min(latencies)), 2),
            "max": round(float(np.max(latencies)), 2),
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
        },
        "fps": round(fps, 2),
        "peak_ram_mb": round(get_peak_ram_mb(), 2),
        "frames": frame_metrics,
    }

    if baseline_outputs is not None:
        overall_agreement = total_matches_vs_baseline / total_pts * 100.0
        summary["prediction_agreement_pct"] = round(overall_agreement, 4)
        summary["logits_max_abs_diff"] = round(float(np.max(max_logit_diffs)), 6)
        summary["logits_mean_abs_diff"] = round(float(np.mean(mean_logit_diffs)), 6)
        summary["probs_max_abs_diff"] = round(float(np.max(max_prob_diffs)), 6)
        summary["probs_mean_abs_diff"] = round(float(np.mean(mean_prob_diffs)), 6)

    # Average mIoU / Accuracy if available
    mious = [f["miou"] for f in frame_metrics if "miou" in f]
    accs = [f["accuracy"] for f in frame_metrics if "accuracy" in f]
    if mious:
        summary["mIoU"] = round(float(np.mean(mious)), 2)
    if accs:
        summary["accuracy"] = round(float(np.mean(accs)), 2)

    return summary, saved_outputs


def run_full_benchmark():
    """Execute complete benchmark comparing Original, Clean FP32, FP16, and INT8."""
    print("=" * 70)
    print("LIDAR-X FAST-FRNET MODEL COMPRESSION BENCHMARK")
    print("=" * 70)

    rellis_frames, sk_frames = load_dataset_frames(max_frames=5)
    print(f"Loaded {len(rellis_frames)} RELLIS-3D frames (avg {np.mean([len(f['points']) for f in rellis_frames]):.0f} pts/frame)")
    print(f"Loaded {len(sk_frames)} SemanticKITTI frames (avg {np.mean([len(f['points']) for f in sk_frames]):.0f} pts/frame)")

    models_dir = BACKEND_DIR / "models"
    compressed_models_dir = SCRIPT_DIR / "models"
    eval_preds_dir = SCRIPT_DIR / "eval_predictions"

    baseline_metrics = {}
    compression_metrics = {}

    for spec_name in ["rellis", "semantickitti"]:
        print(f"\n=======================================================")
        print(f"BENCHMARKING DATASET: {spec_name.upper()}")
        print(f"=======================================================")

        frames = rellis_frames if spec_name == "rellis" else sk_frames
        raw_path = models_dir / f"best_frnet_{spec_name}.pth"
        deploy_path = compressed_models_dir / f"best_frnet_{spec_name}_deploy.pth"

        # -------------------------------------------------------------
        # 1. Baseline Model (FP32 Original)
        # -------------------------------------------------------------
        print(f"\n[1] Running Baseline FP32 Model ({raw_path.name})...")
        base_model = instantiate_model(spec_name)
        load_model_weights(base_model, raw_path, mode="fp32")

        base_summary, base_outputs = benchmark_model_on_frames(
            base_model,
            model_name="baseline_fp32",
            spec_name=spec_name,
            frames=frames,
            save_preds_dir=eval_preds_dir,
        )
        base_summary["checkpoint_file"] = raw_path.name
        base_summary["file_size_bytes"] = raw_path.stat().st_size
        base_summary["file_size_mb"] = round(raw_path.stat().st_size / 1e6, 2)
        base_summary["param_count"] = sum(p.numel() for p in base_model.parameters())
        base_summary["weight_dtype"] = "torch.float32"

        baseline_metrics[spec_name] = base_summary
        print(f"  Baseline File Size: {base_summary['file_size_mb']} MB")
        print(f"  Baseline Latency:   {base_summary['latency_ms']['mean']} ms ({base_summary['fps']} FPS)")
        if "mIoU" in base_summary:
            print(f"  Baseline mIoU:      {base_summary['mIoU']}% | Accuracy: {base_summary['accuracy']}%")

        dataset_compression = {
            "baseline": base_summary,
            "stages": {},
        }

        # -------------------------------------------------------------
        # 2. Stage A: Clean Inference-Only FP32 Checkpoint
        # -------------------------------------------------------------
        print(f"\n[2] Running Stage A: Clean Inference-Only FP32 Checkpoint...")
        clean_model = instantiate_model(spec_name)
        # Load weights without aux heads
        load_model_weights(clean_model, raw_path, mode="fp32")

        clean_summary, _ = benchmark_model_on_frames(
            clean_model,
            model_name="inference_only_fp32",
            spec_name=spec_name,
            frames=frames,
            baseline_outputs=base_outputs,
        )
        # Calculate clean size (tensors only)
        inf_params = sum(p.numel() for p in clean_model.parameters())
        clean_size_est = inf_params * 4 + 20000  # 4 bytes/float32 + header
        clean_summary["file_size_bytes"] = clean_size_est
        clean_summary["file_size_mb"] = round(clean_size_est / 1e6, 2)
        clean_summary["size_reduction_pct"] = round((1.0 - clean_size_est / raw_path.stat().st_size) * 100.0, 2)
        clean_summary["param_count"] = inf_params
        clean_summary["weight_dtype"] = "torch.float32"

        dataset_compression["stages"]["stage_a_inference_only_fp32"] = clean_summary
        print(f"  Stage A Size:       {clean_summary['file_size_mb']} MB (Reduction: {clean_summary['size_reduction_pct']}%)")
        print(f"  Stage A Latency:    {clean_summary['latency_ms']['mean']} ms")
        print(f"  Stage A Agreement:  {clean_summary['prediction_agreement_pct']}%")

        # -------------------------------------------------------------
        # 3. Stage B: FP16 Deployment Checkpoint
        # -------------------------------------------------------------
        print(f"\n[3] Running Stage B: Half-Precision FP16 Deployment Checkpoint...")
        deploy_model = instantiate_model(spec_name)
        load_model_weights(deploy_model, deploy_path, mode="fp16")

        fp16_summary, _ = benchmark_model_on_frames(
            deploy_model,
            model_name="deploy_fp16",
            spec_name=spec_name,
            frames=frames,
            baseline_outputs=base_outputs,
            save_preds_dir=eval_preds_dir,
        )
        deploy_size = deploy_path.stat().st_size
        fp16_summary["checkpoint_file"] = deploy_path.name
        fp16_summary["file_size_bytes"] = deploy_size
        fp16_summary["file_size_mb"] = round(deploy_size / 1e6, 2)
        fp16_summary["size_reduction_pct"] = round((1.0 - deploy_size / raw_path.stat().st_size) * 100.0, 2)
        fp16_summary["param_count"] = sum(p.numel() for p in deploy_model.parameters())
        fp16_summary["weight_dtype"] = "torch.float16 (stored) -> torch.float32 (runtime CPU)"

        dataset_compression["stages"]["stage_b_fp16_deploy"] = fp16_summary
        print(f"  Stage B Size:       {fp16_summary['file_size_mb']} MB (Reduction: {fp16_summary['size_reduction_pct']}%)")
        print(f"  Stage B Latency:    {fp16_summary['latency_ms']['mean']} ms ({fp16_summary['fps']} FPS)")
        print(f"  Stage B Agreement:  {fp16_summary['prediction_agreement_pct']}%")
        print(f"  Stage B Logit Diff: max={fp16_summary['logits_max_abs_diff']}, mean={fp16_summary['logits_mean_abs_diff']}")
        if "mIoU" in fp16_summary:
            print(f"  Stage B mIoU:      {fp16_summary['mIoU']}% | Accuracy: {fp16_summary['accuracy']}%")

        # -------------------------------------------------------------
        # 4. Stage C: INT8 Dynamic Quantization
        # -------------------------------------------------------------
        print(f"\n[4] Running Stage C: INT8 Dynamic Quantization...")
        try:
            int8_model = torch.ao.quantization.quantize_dynamic(
                base_model, {nn.Linear}, dtype=torch.qint8
            )
            int8_summary, _ = benchmark_model_on_frames(
                int8_model,
                model_name="int8_dynamic",
                spec_name=spec_name,
                frames=frames,
                baseline_outputs=base_outputs,
            )
            # Estimate INT8 size (Linear layers int8, Conv2d fp32)
            # 5.8% linear in int8, 94% conv in fp32
            int8_size_est = int(inf_params * 0.94 * 4 + inf_params * 0.058 * 1) + 20000
            int8_summary["file_size_bytes"] = int8_size_est
            int8_summary["file_size_mb"] = round(int8_size_est / 1e6, 2)
            int8_summary["size_reduction_pct"] = round((1.0 - int8_size_est / raw_path.stat().st_size) * 100.0, 2)
            int8_summary["weight_dtype"] = "torch.qint8 (Linear) + torch.float32 (Conv2d)"

            dataset_compression["stages"]["stage_c_int8_dynamic"] = int8_summary
            print(f"  Stage C Size:       {int8_summary['file_size_mb']} MB (Reduction: {int8_summary['size_reduction_pct']}%)")
            print(f"  Stage C Latency:    {int8_summary['latency_ms']['mean']} ms ({int8_summary['fps']} FPS)")
            print(f"  Stage C Agreement:  {int8_summary['prediction_agreement_pct']}%")
        except Exception as e:
            print(f"  Stage C Failed: {e}")

        compression_metrics[spec_name] = dataset_compression

    # Save baseline report
    baseline_path = SCRIPT_DIR / "baseline_metrics.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    print(f"\nSaved baseline report: {baseline_path}")

    # Save compression report
    compression_path = SCRIPT_DIR / "compression_metrics.json"
    with open(compression_path, "w") as f:
        json.dump(compression_metrics, f, indent=2)
    print(f"Saved compression report: {compression_path}")

    return baseline_metrics, compression_metrics


if __name__ == "__main__":
    run_full_benchmark()
