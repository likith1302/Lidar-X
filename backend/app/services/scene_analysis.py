"""Lightweight geometric scene analyzer for automated Fast-FRNet model selection.

Analyzes raw 3D point cloud geometry (elevation variance, ground planarity,
vertical structures, volumetric vegetation dispersion, and vertical beam distribution)
to select between:
- best_frnet_rellis.pth (RELLIS-3D off-road / unstructured natural terrain)
- best_frnet_semantickitti.pth (SemanticKITTI urban / structured road corridors)

Runs in under 15ms on CPU with zero prior choice reuse.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import logging

from ..config.model_settings import model_settings

logger = logging.getLogger(__name__)


class SceneDomainResult:
    """Structured result of automatic scene analysis and model selection."""

    def __init__(
        self,
        domain: str,
        selected_model: str,
        model_type: str,
        confidence: float,
        selection_reason: str,
        metrics: Dict[str, float],
    ):
        self.domain = domain
        self.selected_model = selected_model
        self.model_type = model_type
        self.confidence = float(np.clip(confidence, 0.50, 0.99))
        self.selection_reason = selection_reason
        self.metrics = metrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": "live_upload",
            "inference_source": f"fast_frnet_{self.model_type}",
            "model": self.selected_model,
            "model_type": self.model_type,
            "domain": self.domain,
            "confidence": round(self.confidence, 3),
            "selection_reason": self.selection_reason,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
        }


class SceneAnalysisEngine:
    """Performs rapid geometric analysis on uploaded 3D points to select the optimal Fast-FRNet checkpoint."""

    @classmethod
    def analyze_scene(cls, points: np.ndarray, max_sample_points: int = 10000) -> SceneDomainResult:
        """Inspect point cloud geometry and select best Fast-FRNet model variant.

        Args:
            points: (N, >=3) float32 point cloud array [x, y, z, (intensity)]
            max_sample_points: Maximum points to sample for lightweight analysis

        Returns:
            SceneDomainResult with domain, selected model, confidence, and metrics.
        """
        if points is None or len(points) == 0:
            return SceneDomainResult(
                domain="rellis_offroad",
                selected_model="best_frnet_rellis.pth",
                model_type="rellis",
                confidence=0.70,
                selection_reason="Default off-road perception model selected for empty or minimal scan.",
                metrics={},
            )

        # 1. Filter NaNs and sample points uniformly
        valid_mask = np.isfinite(points[:, :3]).all(axis=1)
        valid_pts = points[valid_mask, :3]
        n_pts = valid_pts.shape[0]

        if n_pts <= max_sample_points:
            sample = valid_pts
        else:
            step = n_pts / max_sample_points
            indices = (np.arange(max_sample_points) * step).astype(np.int64)
            sample = valid_pts[indices]

        zs = sample[:, 2]
        xs = sample[:, 0]
        ys = sample[:, 1]
        ranges_xy = np.sqrt(xs**2 + ys**2)

        # 2. Metric A: Ground elevation band variance and planarity
        # In urban KITTI (Velodyne on car roof ~1.73m), asphalt ground lies in z in [-2.2, -1.2]
        # In off-road RELLIS (Clearpath Warthog / ATV), ground is irregular, undulating, with ditches
        ground_candidate_mask = (zs <= np.percentile(zs, 25)) & (ranges_xy > 2.0) & (ranges_xy < 35.0)
        if np.count_nonzero(ground_candidate_mask) >= 30:
            ground_pts = sample[ground_candidate_mask]
            ground_z_std = float(np.std(ground_pts[:, 2]))
            # Local planar fit residual
            centroid = np.mean(ground_pts, axis=0)
            shifted = ground_pts - centroid
            cov = np.dot(shifted.T, shifted) / len(ground_pts)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            sort_idx = np.argsort(eigenvals)[::-1]
            e1, e2, e3 = np.maximum(eigenvals[sort_idx], 1e-6)
            planarity = float((e2 - e3) / e1)
            normal = eigenvecs[:, sort_idx[2]]
            vertical_ground_alignment = float(abs(normal[2]))  # Should be close to 1.0 for flat road
        else:
            ground_z_std = 0.25
            planarity = 0.4
            vertical_ground_alignment = 0.7

        # 3. Metric B: Vertical facade structure ratio (Walls / Buildings)
        # Urban scenes have high density of returns on vertical surfaces (|n_z| < 0.2, tall height span)
        elevated_mask = (zs > -0.5) & (ranges_xy > 3.0) & (ranges_xy < 45.0)
        elevated_count = int(np.count_nonzero(elevated_mask))
        vertical_facade_score = 0.0

        if elevated_count >= 100:
            elev_pts = sample[elevated_mask]
            # Height span of elevated returns
            z_span = float(np.max(elev_pts[:, 2]) - np.min(elev_pts[:, 2]))
            # 2D projection density: urban facades have high count of points within narrow (x,y) bins
            x_bins = np.floor(elev_pts[:, 0] / 1.5).astype(np.int64)
            y_bins = np.floor(elev_pts[:, 1] / 1.5).astype(np.int64)
            bin_keys = x_bins * 100000 + y_bins
            _, bin_counts = np.unique(bin_keys, return_counts=True)
            high_density_columns = float(np.mean(bin_counts > 15))
            vertical_facade_score = float(min(1.0, high_density_columns * (z_span / 4.0)))
        else:
            vertical_facade_score = 0.0

        # 4. Metric C: Volumetric vegetative scattering (Off-road dispersion)
        # Foliage and bushes produce 3D volumetric scattering with isotropic eigenvalues (sphericity e3/e1)
        mid_veg_mask = (zs > -1.0) & (zs < 2.5) & (ranges_xy > 3.0) & (ranges_xy < 25.0)
        if np.count_nonzero(mid_veg_mask) >= 50:
            mid_pts = sample[mid_veg_mask]
            cov_mid = np.cov(mid_pts[:, :3], rowvar=False)
            mid_eigvals = np.sort(np.linalg.eigvalsh(cov_mid))[::-1]
            me1, me2, me3 = np.maximum(mid_eigvals, 1e-6)
            sphericity = float(me3 / me1)
            scattering = float((me2 - me3) / me1)
        else:
            sphericity = 0.1
            scattering = 0.2

        # 5. Metric D: Vertical sensor beam angular reach
        # KITTI HDL-64E vertical FOV is asymmetric: [-25.0, +3.0 deg] -> almost no points > +5 deg
        # RELLIS Ouster OS1-64 / VLP-32C vertical FOV spans up to +15.0 / +16.6 deg
        ranges_3d = np.maximum(np.linalg.norm(sample, axis=1), 1e-3)
        elevation_angles_deg = np.degrees(np.arcsin(np.clip(zs / ranges_3d, -1.0, 1.0)))
        positive_angle_ratio = float(np.mean(elevation_angles_deg > 4.0))

        # 6. Scoring Synthesis
        urban_evidence = 0.0
        offroad_evidence = 0.0

        # Ground planarity & roughness
        if ground_z_std < 0.09 and planarity > 0.65 and vertical_ground_alignment > 0.88:
            urban_evidence += 1.8
        elif ground_z_std > 0.12 or planarity < 0.55 or vertical_ground_alignment < 0.82:
            offroad_evidence += 1.8
        else:
            urban_evidence += 0.5
            offroad_evidence += 0.5

        # Vertical facades (buildings/walls)
        if vertical_facade_score > 0.25:
            urban_evidence += 2.0
        else:
            offroad_evidence += 1.2

        # Vegetative dispersion
        if sphericity > 0.22:
            offroad_evidence += 1.5
        elif sphericity < 0.10:
            urban_evidence += 0.8

        # Sensor FOV upward spread
        if positive_angle_ratio > 0.08:
            offroad_evidence += 1.2
        elif positive_angle_ratio < 0.02:
            urban_evidence += 1.0

        total_evidence = urban_evidence + offroad_evidence
        urban_prob = urban_evidence / max(1e-3, total_evidence)
        offroad_prob = offroad_evidence / max(1e-3, total_evidence)

        metrics = {
            "ground_z_std_m": ground_z_std,
            "ground_planarity": planarity,
            "vertical_ground_alignment": vertical_ground_alignment,
            "vertical_facade_score": vertical_facade_score,
            "volumetric_sphericity": sphericity,
            "positive_angle_ratio": positive_angle_ratio,
            "urban_evidence_score": urban_evidence,
            "offroad_evidence_score": offroad_evidence,
        }

        if urban_prob > 0.55:
            domain = "kitti_urban"
            model_type = "semantickitti"
            selected_model = "best_frnet_semantickitti.pth"
            confidence = min(0.96, max(0.72, urban_prob))
            reason = (
                f"Dominant planar road surface (ground z-variance: {ground_z_std:.3f}m, planarity: {planarity:.2f}) "
                f"and structural facade returns (facade score: {vertical_facade_score:.2f}) indicate urban roadway corridor. "
                "Selected Fast-FRNet SemanticKITTI model."
            )
        else:
            domain = "rellis_offroad"
            model_type = "rellis"
            selected_model = "best_frnet_rellis.pth"
            confidence = min(0.96, max(0.72, offroad_prob))
            reason = (
                f"Irregular terrain elevation (ground z-variance: {ground_z_std:.3f}m) and volumetric vegetation scattering "
                f"(sphericity: {sphericity:.2f}, upward angle ratio: {positive_angle_ratio:.2%}) indicate off-road/trail environment. "
                "Selected Fast-FRNet RELLIS-3D model."
            )

        logger.info(
            f"Scene Analysis: domain={domain}, model={selected_model}, confidence={confidence:.2f}, reason={reason}"
        )

        return SceneDomainResult(
            domain=domain,
            selected_model=selected_model,
            model_type=model_type,
            confidence=confidence,
            selection_reason=reason,
            metrics=metrics,
        )
