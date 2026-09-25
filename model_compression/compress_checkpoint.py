#!/usr/bin/env python3
"""Fast-FRNet Model Compression Utility.

Creates production-ready, deployment-optimized Fast-FRNet checkpoints by:
1. Stripping non-inference data (optimizer states, schedulers, training metadata, auxiliary heads)
2. Converting FP32 weights to FP16 half-precision
3. Embedding architecture metadata for deployment verification
4. Verifying checkpoint integrity with a forward-pass test

Usage:
    python compress_checkpoint.py --all
    python compress_checkpoint.py --model_type rellis
    python compress_checkpoint.py --model_type semantickitti
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

# Add backend directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.fast_frnet import FastFRNet
from app.config.model_settings import model_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compress_checkpoint")


def inspect_raw_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Inspect contents of raw checkpoint and report findings."""
    raw_size = checkpoint_path.stat().st_size
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    top_keys = list(ckpt.keys()) if isinstance(ckpt, dict) else ["<non-dict>"]
    has_optimizer = "optimizer" in ckpt if isinstance(ckpt, dict) else False
    has_scheduler = "param_schedulers" in ckpt if isinstance(ckpt, dict) else False
    has_meta = "meta" in ckpt if isinstance(ckpt, dict) else False
    
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    aux_keys = [k for k in sd.keys() if "auxiliary_head" in k]
    
    return {
        "file_name": checkpoint_path.name,
        "raw_size_bytes": raw_size,
        "raw_size_mb": round(raw_size / 1e6, 2),
        "top_level_keys": top_keys,
        "has_optimizer": has_optimizer,
        "has_scheduler": has_scheduler,
        "has_meta": has_meta,
        "auxiliary_head_keys_count": len(aux_keys),
        "total_tensors": len(sd),
    }


def extract_inference_state_dict(checkpoint_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Extract clean inference-only state dict stripping training artifacts."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt.get("model_state_dict", ckpt)))
    else:
        sd = ckpt

    filtered_sd = {}
    stripped_keys = []
    for k, v in sd.items():
        key = k[7:] if k.startswith("module.") else k
        if key.startswith("auxiliary_head"):
            stripped_keys.append(k)
            continue
        filtered_sd[key] = v

    stats = {
        "original_tensor_count": len(sd),
        "inference_tensor_count": len(filtered_sd),
        "stripped_tensor_count": len(stripped_keys),
        "total_inference_params": sum(p.numel() for p in filtered_sd.values()),
    }
    return filtered_sd, stats


def convert_state_dict_precision(
    state_dict: Dict[str, torch.Tensor],
    precision: str = "fp16"
) -> Dict[str, torch.Tensor]:
    """Convert floating-point tensors to specified precision while preserving integer buffers."""
    converted = {}
    for k, v in state_dict.items():
        if v.dtype == torch.float32 and precision == "fp16":
            converted[k] = v.half()
        elif v.dtype == torch.float32 and precision == "bfloat16":
            converted[k] = v.bfloat16()
        else:
            converted[k] = v
    return converted


def verify_checkpoint(
    checkpoint_path: Path,
    spec_name: str,
    device: str = "cpu"
) -> bool:
    """Verify that the compressed checkpoint loads and runs a forward pass successfully."""
    spec = model_settings.get_spec(spec_name)
    model = FastFRNet(
        output_shape=(spec.H, spec.W),
        fov_up=spec.fov_up,
        fov_down=spec.fov_down,
        num_classes=spec.num_classes,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Load with automatic CPU fp16->fp32 conversion if needed
    cleaned = {}
    for k, v in sd.items():
        key = k[7:] if k.startswith("module.") else k
        if not key.startswith("auxiliary_head"):
            if device == "cpu" and v.dtype == torch.float16:
                cleaned[key] = v.float()
            else:
                cleaned[key] = v

    model.load_state_dict(cleaned, strict=True)
    model.eval()

    # Run dummy forward pass with 50 points
    test_pts = torch.randn(50, 4, dtype=torch.float32)
    test_pts[:, 2] = -1.73
    with torch.no_grad():
        logits = model(test_pts)

    assert logits.shape == (50, spec.num_classes), (
        f"Shape mismatch: expected (50, {spec.num_classes}), got {logits.shape}"
    )
    assert not torch.isnan(logits).any(), "NaN values detected in model output logits"
    logger.info(f"Verification PASSED for '{checkpoint_path.name}' (output shape: {logits.shape})")
    return True


def compress_model(
    model_type: str,
    input_path: Path,
    output_path: Path,
    precision: str = "fp16",
    verify: bool = True
) -> Dict[str, Any]:
    """Execute complete compression pipeline on a single model variant."""
    logger.info(f"--- Compressing {model_type.upper()} Model ---")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")

    raw_stats = inspect_raw_checkpoint(input_path)
    logger.info(
        f"Raw checkpoint: {raw_stats['raw_size_mb']} MB | "
        f"Optimizer present: {raw_stats['has_optimizer']} | "
        f"Scheduler present: {raw_stats['has_scheduler']}"
    )

    # 1. Strip unnecessary data
    inf_sd, strip_stats = extract_inference_state_dict(input_path)
    logger.info(
        f"Extracted inference state dict: {strip_stats['inference_tensor_count']} tensors, "
        f"{strip_stats['total_inference_params']:,} parameters (stripped {strip_stats['stripped_tensor_count']} training tensors)"
    )

    # 2. Precision conversion
    converted_sd = convert_state_dict_precision(inf_sd, precision=precision)

    # 3. Create deployment payload with metadata
    spec = model_settings.get_spec(model_type)
    deployment_payload = {
        "format": "fast_frnet_deploy",
        "model_type": model_type,
        "spec": {
            "name": spec.name,
            "H": spec.H,
            "W": spec.W,
            "fov_up": spec.fov_up,
            "fov_down": spec.fov_down,
            "num_classes": spec.num_classes,
        },
        "precision": precision,
        "param_count": strip_stats["total_inference_params"],
        "compressed_at": datetime.now(timezone.utc).isoformat(),
        "state_dict": converted_sd,
    }

    # 4. Save deployment checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(deployment_payload, output_path)
    compressed_size = output_path.stat().st_size
    reduction_pct = (1.0 - compressed_size / raw_stats["raw_size_bytes"]) * 100.0

    logger.info(
        f"Saved compressed checkpoint: {compressed_size / 1e6:.2f} MB "
        f"(Reduced by {reduction_pct:.1f}% from {raw_stats['raw_size_mb']} MB)"
    )

    # 5. Verify checkpoint
    if verify:
        verify_checkpoint(output_path, model_type)

    return {
        "model_type": model_type,
        "precision": precision,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "raw_size_bytes": raw_stats["raw_size_bytes"],
        "compressed_size_bytes": compressed_size,
        "reduction_pct": round(reduction_pct, 2),
        "param_count": strip_stats["total_inference_params"],
    }


def main():
    parser = argparse.ArgumentParser(description="Compress Fast-FRNet LiDAR models for deployment.")
    parser.add_argument("--model_type", choices=["rellis", "semantickitti"], help="Target model variant")
    parser.add_argument("--all", action="store_true", help="Compress all model variants")
    parser.add_argument("--precision", choices=["fp16", "fp32", "bfloat16"], default="fp16", help="Target weight precision")
    parser.add_argument("--output_dir", type=Path, default=SCRIPT_DIR / "models", help="Destination directory")
    parser.add_argument("--no_verify", action="store_true", help="Skip forward pass verification")
    args = parser.parse_args()

    models_to_compress = []
    if args.all or (not args.model_type):
        models_to_compress = ["rellis", "semantickitti"]
    else:
        models_to_compress = [args.model_type]

    raw_dir = BACKEND_DIR / "models"
    out_dir = args.output_dir
    backend_models_dir = BACKEND_DIR / "models"

    results = []
    for m_type in models_to_compress:
        raw_name = f"best_frnet_{m_type}.pth"
        deploy_name = f"best_frnet_{m_type}_deploy.pth"
        in_path = raw_dir / raw_name
        out_path = out_dir / deploy_name

        if not in_path.is_file():
            logger.error(f"Source model not found: {in_path}")
            continue

        res = compress_model(
            model_type=m_type,
            input_path=in_path,
            output_path=out_path,
            precision=args.precision,
            verify=not args.no_verify,
        )
        results.append(res)

        # Also copy to backend/models for immediate live inference integration
        backend_deploy_path = backend_models_dir / deploy_name
        if out_path.resolve() != backend_deploy_path.resolve():
            import shutil
            shutil.copy2(out_path, backend_deploy_path)
            logger.info(f"Synchronized deploy model to: {backend_deploy_path}")

    print("\n" + "=" * 60)
    print("COMPRESSION SUMMARY")
    print("=" * 60)
    for r in results:
        print(
            f"Model: {r['model_type'].upper():<14} | "
            f"Orig: {r['raw_size_bytes'] / 1e6:>6.2f} MB -> "
            f"Deploy: {r['compressed_size_bytes'] / 1e6:>6.2f} MB | "
            f"Reduction: {r['reduction_pct']:>5.1f}%"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
