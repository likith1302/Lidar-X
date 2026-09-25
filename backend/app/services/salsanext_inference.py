"""SalsaNext Inference Compatibility Wrapper.

This module delegates directly to FastFRNetInferenceService to preserve
backward compatibility with any existing callers or tests while ensuring
all inference runs through the verified Fast-FRNet checkpoints.
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch

from ..models.inference_schemas import (
    InferenceStatusResponse,
    PointPredictionSample,
    ProcessStatus,
)
from .fast_frnet_inference import FastFRNetInferenceService


class SalsaNextInferenceService:
    """Backward-compatible adapter forwarding all calls to FastFRNetInferenceService."""

    @classmethod
    def get_device(cls) -> torch.device:
        """Determine hardware device with graceful CPU fallback."""
        return FastFRNetInferenceService.get_device()

    @classmethod
    def get_status(cls, model_type: str = "rellis") -> InferenceStatusResponse:
        """Inspect and report truthful model file presence and device state."""
        return FastFRNetInferenceService.get_status(model_type=model_type)

    @classmethod
    def load_model(cls, model_type: str = "rellis") -> torch.nn.Module:
        """Load neural network from checkpoint once and cache in memory."""
        return FastFRNetInferenceService.load_model(model_type=model_type)

    @classmethod
    def run_inference(
        cls,
        points: np.ndarray,
        frame_id: Optional[str] = None,
        model_type: str = "rellis",
        persist_artifact: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Execute full point-wise semantic segmentation pipeline using Fast-FRNet."""
        return FastFRNetInferenceService.run_inference(
            points=points,
            frame_id=frame_id,
            model_type=model_type,
            persist_artifact=persist_artifact,
        )

    @classmethod
    def build_sample_predictions(
        cls,
        points: np.ndarray,
        raw_labels: np.ndarray,
        max_samples: int = 4000,
    ) -> List[PointPredictionSample]:
        """Build downsampled list of predictions for responsive frontend rendering."""
        return FastFRNetInferenceService.build_sample_predictions(
            points=points,
            raw_labels=raw_labels,
            max_samples=max_samples,
        )
