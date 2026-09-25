"""Model provider interface and SalsaNext prediction adapter."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import httpx

from ..models.object_schemas import ModelProviderStatus, PredictionImportPayload
from .semantic_mapping import SemanticMappingService, SemanticMappingError


class ModelProviderService:
    """Service managing deep learning segmentation models and precomputed prediction imports."""

    @classmethod
    def process_imported_predictions(
        cls,
        payload: PredictionImportPayload,
        expected_point_count: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and parse an imported SalsaNext / external model prediction payload.
        
        Args:
            payload: PredictionImportPayload containing integer labels array.
            expected_point_count: Point count of the corresponding LiDAR frame.
            
        Returns:
            Tuple of (semantic_classes: np.ndarray[uint16], instance_ids: np.ndarray[uint16]).
            
        Raises:
            SemanticMappingError: If label count doesn't match point count or array is invalid.
        """
        raw_labels = payload.labels
        if len(raw_labels) == 0:
            raise SemanticMappingError("Empty prediction labels array in payload.")

        if len(raw_labels) != expected_point_count:
            raise SemanticMappingError(
                f"Label count mismatch: received {len(raw_labels)} prediction labels, but frame "
                f"has {expected_point_count} points. Each point must have exactly one label."
            )

        labels_arr = np.array(raw_labels, dtype=np.uint32)
        semantic_classes = (labels_arr & 0xFFFF).astype(np.uint16)
        instance_ids = (labels_arr >> 16).astype(np.uint16)

        return semantic_classes, instance_ids

    @classmethod
    async def query_live_inference_service(
        cls,
        endpoint_url: str,
        points: np.ndarray,
        timeout_seconds: float = 3.0,
    ) -> Tuple[ModelProviderStatus, Optional[np.ndarray], Optional[str]]:
        """Attempt to query external SalsaNext / deep learning inference server.
        
        Returns honest status without mocking local model availability.
        """
        if not endpoint_url or not endpoint_url.strip():
            return ModelProviderStatus.INFERENCE_SERVICE_NOT_CONNECTED, None, "No inference endpoint configured."

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                res = await client.post(
                    endpoint_url,
                    json={"points": points[:, :4].tolist()},
                )
                if res.status_code == 200:
                    data = res.json()
                    labels = np.array(data.get("predictions", []), dtype=np.uint32)
                    return ModelProviderStatus.EXTERNAL_PREDICTIONS_LOADED, labels, None
                else:
                    return (
                        ModelProviderStatus.PREDICTION_UNAVAILABLE,
                        None,
                        f"External inference service returned HTTP {res.status_code}: {res.text}",
                    )
        except Exception as e:
            return (
                ModelProviderStatus.INFERENCE_SERVICE_NOT_CONNECTED,
                None,
                f"Could not connect to external model inference service: {str(e)}",
            )
