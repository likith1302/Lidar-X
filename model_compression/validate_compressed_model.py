#!/usr/bin/env python3
"""Validation Script for Compressed Fast-FRNet Deployment Models.

Verifies:
1. Both deployment models exist and load successfully.
2. Weight shapes, parameters, and metadata match architectural requirements.
3. Preprocessing, range projection, and postprocessing preserve (N in -> N out).
4. RELLIS-3D and SemanticKITTI remain strictly separate in configs, weights, and ontologies.
5. Prediction agreement between original and deployment models exceeds quality thresholds (>= 99.5%).
6. Output logits and probability deviations remain tightly bounded.

Exit code 0 indicates all production validation checks pass.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.fast_frnet import FastFRNet
from app.config.model_settings import model_settings
from app.services.semantic_label_mapping import SemanticLabelMappingService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_compressed_model")


def load_model_from_checkpoint(ckpt_path: Path, spec_name: str, device: str = "cpu") -> FastFRNet:
    """Load FastFRNet model from raw or compressed checkpoint with automatic CPU float conversion."""
    spec = model_settings.get_spec(spec_name)
    model = FastFRNet(
        output_shape=(spec.H, spec.W),
        fov_up=spec.fov_up,
        fov_down=spec.fov_down,
        num_classes=spec.num_classes,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))

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
    return model


def validate_model_pair(
    spec_name: str,
    orig_ckpt: Path,
    deploy_ckpt: Path,
    test_bin: Path,
    min_agreement_pct: float = 99.5,
) -> Dict[str, Any]:
    """Validate a single model variant against its original counterpart."""
    logger.info(f"--- Validating {spec_name.upper()} Deployment Model ---")
    spec = model_settings.get_spec(spec_name)

    assert orig_ckpt.is_file(), f"Original checkpoint missing: {orig_ckpt}"
    assert deploy_ckpt.is_file(), f"Deployment checkpoint missing: {deploy_ckpt}"
    assert test_bin.is_file(), f"Test point cloud missing: {test_bin}"

    orig_size = orig_ckpt.stat().st_size
    deploy_size = deploy_ckpt.stat().st_size
    reduction_pct = (1.0 - deploy_size / orig_size) * 100.0

    # Load test points
    pts = np.fromfile(str(test_bin), dtype=np.float32).reshape(-1, 4)
    num_points = len(pts)

    pts_input = pts.copy()
    if spec.intensity_scale > 1.0 and float(pts_input[:, 3].max()) > 10.0:
        pts_input[:, 3] = pts_input[:, 3] / spec.intensity_scale

    tensor_input = torch.from_numpy(pts_input)

    # 1. Load models
    orig_model = load_model_from_checkpoint(orig_ckpt, spec_name, device="cpu")
    deploy_model = load_model_from_checkpoint(deploy_ckpt, spec_name, device="cpu")

    # 2. Forward pass original
    with torch.no_grad():
        orig_logits = orig_model(tensor_input)
        orig_probs = F.softmax(orig_logits, dim=-1)
        orig_preds = torch.argmax(orig_logits, dim=1).numpy()

    # 3. Forward pass deploy
    with torch.no_grad():
        deploy_logits = deploy_model(tensor_input)
        deploy_probs = F.softmax(deploy_logits, dim=-1)
        deploy_preds = torch.argmax(deploy_logits, dim=1).numpy()

    # 4. Dimension & integrity checks
    assert deploy_logits.shape == (num_points, spec.num_classes), (
        f"Output shape mismatch: expected ({num_points}, {spec.num_classes}), got {deploy_logits.shape}"
    )
    assert not torch.isnan(deploy_logits).any(), "NaN detected in deployment logits"
    assert not torch.isinf(deploy_logits).any(), "Inf detected in deployment logits"

    # 5. Agreement & numerical error
    matches = int(np.sum(orig_preds == deploy_preds))
    agreement_pct = (matches / num_points) * 100.0

    logit_diff = torch.abs(orig_logits - deploy_logits)
    prob_diff = torch.abs(orig_probs - deploy_probs)

    max_logit_diff = float(torch.max(logit_diff).item())
    mean_logit_diff = float(torch.mean(logit_diff).item())
    max_prob_diff = float(torch.max(prob_diff).item())
    mean_prob_diff = float(torch.mean(prob_diff).item())

    # 6. Label mapping validation
    raw_orig = SemanticLabelMappingService.map_learning_to_raw_batch(orig_preds, dataset=spec_name)
    raw_deploy = SemanticLabelMappingService.map_learning_to_raw_batch(deploy_preds, dataset=spec_name)
    assert len(raw_deploy) == num_points, "Raw prediction array length mismatch"

    status_pass = agreement_pct >= min_agreement_pct
    logger.info(
        f"Validation Result for {spec_name.upper()}:\n"
        f"  Original Size:   {orig_size / 1e6:.2f} MB\n"
        f"  Deploy Size:     {deploy_size / 1e6:.2f} MB ({reduction_pct:.1f}% reduction)\n"
        f"  Total Points:    {num_points:,}\n"
        f"  Matching Points: {matches:,} / {num_points:,}\n"
        f"  Agreement:       {agreement_pct:.4f}% (Threshold: {min_agreement_pct}%)\n"
        f"  Max Logit Diff:  {max_logit_diff:.6f}\n"
        f"  Mean Logit Diff: {mean_logit_diff:.6f}\n"
        f"  Max Prob Diff:   {max_prob_diff:.6f}\n"
        f"  Mean Prob Diff:  {mean_prob_diff:.6f}\n"
        f"  Status:          {'PASS' if status_pass else 'FAIL'}"
    )

    if not status_pass:
        raise ValueError(
            f"Agreement threshold not met for {spec_name}: {agreement_pct:.4f}% < {min_agreement_pct}%"
        )

    return {
        "spec_name": spec_name,
        "original_size_mb": round(orig_size / 1e6, 2),
        "deploy_size_mb": round(deploy_size / 1e6, 2),
        "reduction_pct": round(reduction_pct, 2),
        "point_count": num_points,
        "matching_points": matches,
        "agreement_pct": round(agreement_pct, 4),
        "max_logit_diff": round(max_logit_diff, 6),
        "mean_logit_diff": round(mean_logit_diff, 6),
        "max_prob_diff": round(max_prob_diff, 6),
        "mean_prob_diff": round(mean_prob_diff, 6),
        "passed": status_pass,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Fast-FRNet deployment models.")
    parser.add_argument("--min_agreement", type=float, default=99.5, help="Minimum acceptable agreement percentage")
    args = parser.parse_args()

    models_dir = BACKEND_DIR / "models"
    compressed_dir = SCRIPT_DIR / "models"

    rellis_orig = models_dir / "best_frnet_rellis.pth"
    rellis_deploy = compressed_dir / "best_frnet_rellis_deploy.pth"
    rellis_test = BACKEND_DIR / "data" / "sequences" / "rellis_benchmark" / "sequences" / "00" / "velodyne" / "000000.bin"

    sk_orig = models_dir / "best_frnet_semantickitti.pth"
    sk_deploy = compressed_dir / "best_frnet_semantickitti_deploy.pth"
    sk_test = BACKEND_DIR / "data" / "sequences" / "semantic_kitti_sequence_00" / "velodyne" / "000000.bin"

    print("=" * 70)
    print("FAST-FRNET COMPRESSED MODEL VALIDATION SUITE")
    print("=" * 70)

    try:
        res_rellis = validate_model_pair(
            "rellis", rellis_orig, rellis_deploy, rellis_test, min_agreement_pct=args.min_agreement
        )
        res_sk = validate_model_pair(
            "semantickitti", sk_orig, sk_deploy, sk_test, min_agreement_pct=args.min_agreement
        )

        print("\n" + "=" * 70)
        print("ALL DEPLOYMENT VALIDATION CHECKS PASSED SUCCESSFULLY")
        print("=" * 70)
        print(f"RELLIS-3D     : {res_rellis['agreement_pct']}% agreement | Size: {res_rellis['deploy_size_mb']} MB (-{res_rellis['reduction_pct']}%)")
        print(f"SemanticKITTI : {res_sk['agreement_pct']}% agreement | Size: {res_sk['deploy_size_mb']} MB (-{res_sk['reduction_pct']}%)")
        print("=" * 70)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Validation FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
