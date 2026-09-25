"""Map Serialization Service: Persists and loads 2.5D Adaptive Maps to/from disk."""

import json
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timezone

from ..config import settings
from ..models.map_schemas import (
    AdaptiveGridCell,
    MapMetadata,
    MapExportResponse,
    GridPolicyConfig,
    MapMode,
)


from .frame_registry import FrameRegistry


class MapSerializationService:
    """Manages disk persistence of full maps in backend/data/maps/."""

    @classmethod
    def save_map(
        cls,
        map_id: str,
        mode: MapMode,
        metadata: MapMetadata,
        cells: List[AdaptiveGridCell],
        policy: GridPolicyConfig,
    ) -> Path:
        """Save entire map to JSON on disk atomically."""
        export_data = MapExportResponse(
            map_id=map_id,
            mode=mode,
            metadata=metadata,
            cells=cells,
            policy_snapshot=policy,
        )
        file_path = FrameRegistry.get_map_path(map_id)
        return FrameRegistry.atomic_save_json(file_path, export_data.model_dump())

    @classmethod
    def load_map(cls, map_id: str) -> Optional[MapExportResponse]:
        """Load exported map from disk."""
        file_path = FrameRegistry.get_map_path(map_id)
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return MapExportResponse(**data)
        except Exception:
            return None

    @classmethod
    def list_saved_maps(cls) -> List[str]:
        """List all saved map IDs."""
        if not settings.MAPS_DIR.exists():
            return []
        return [
            f.stem for f in settings.MAPS_DIR.glob("*.json")
        ]
