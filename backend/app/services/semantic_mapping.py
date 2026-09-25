"""SemanticKITTI label parser, bitmask separation, and configurable class mapping."""

from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import yaml

from ..config import settings
from ..models.object_schemas import SemanticPoint3D, ProjectCategory


class SemanticMappingError(Exception):
    """Exception raised for semantic mapping or label parsing failures."""
    pass


class SemanticMappingService:
    """Service for loading semantic YAML mapping, parsing .label files, and classifying points."""

    _mapping_cache: Optional[Dict[int, Dict[str, Any]]] = None

    @classmethod
    def load_mapping(cls) -> Dict[int, Dict[str, Any]]:
        """Load semantic mapping from YAML configuration file."""
        if cls._mapping_cache is not None:
            return cls._mapping_cache

        yaml_path = settings.CONFIG_DIR / "semantic_mapping.yaml"
        if not yaml_path.exists():
            # Fallback default minimal mapping
            cls._mapping_cache = {
                0: {"name": "unlabeled", "category": "unknown", "is_dynamic": False, "clusterable": False},
                1: {"name": "outlier", "category": "unknown", "is_dynamic": False, "clusterable": False},
                10: {"name": "car", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                11: {"name": "bicycle", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                13: {"name": "bus", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                15: {"name": "motorcycle", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                16: {"name": "on-rails", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                18: {"name": "truck", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                20: {"name": "other-vehicle", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                30: {"name": "person", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                31: {"name": "bicyclist", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                32: {"name": "motorcyclist", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                40: {"name": "road", "category": "drivable", "is_dynamic": False, "clusterable": False},
                44: {"name": "parking", "category": "drivable", "is_dynamic": False, "clusterable": False},
                48: {"name": "sidewalk", "category": "non_drivable", "is_dynamic": False, "clusterable": False},
                49: {"name": "other-ground", "category": "non_drivable", "is_dynamic": False, "clusterable": False},
                50: {"name": "building", "category": "infrastructure", "is_dynamic": False, "clusterable": False},
                51: {"name": "fence", "category": "static_obstacle", "is_dynamic": False, "clusterable": True},
                70: {"name": "vegetation", "category": "vegetation", "is_dynamic": False, "clusterable": False},
                71: {"name": "trunk", "category": "vegetation", "is_dynamic": False, "clusterable": True},
                72: {"name": "terrain", "category": "non_drivable", "is_dynamic": False, "clusterable": False},
                80: {"name": "pole", "category": "static_obstacle", "is_dynamic": False, "clusterable": True},
                81: {"name": "traffic-sign", "category": "static_obstacle", "is_dynamic": False, "clusterable": True},
                99: {"name": "other-object", "category": "static_obstacle", "is_dynamic": False, "clusterable": True},
                252: {"name": "moving-car", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                253: {"name": "moving-bicyclist", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                254: {"name": "moving-person", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                255: {"name": "moving-motorcyclist", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                256: {"name": "moving-on-rails", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                257: {"name": "moving-bus", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                258: {"name": "moving-truck", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
                259: {"name": "moving-other-vehicle", "category": "dynamic_object", "is_dynamic": True, "clusterable": True},
            }
            return cls._mapping_cache

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        classes_raw = data.get("classes", {})
        mapping: Dict[int, Dict[str, Any]] = {}
        for k, v in classes_raw.items():
            mapping[int(k)] = {
                "name": v.get("name", "unknown"),
                "category": v.get("category", "unknown"),
                "is_dynamic": bool(v.get("is_dynamic", False)),
                "clusterable": bool(v.get("clusterable", False)),
            }

        cls._mapping_cache = mapping
        return cls._mapping_cache

    @classmethod
    def get_class_info(cls, class_id: int, dataset: Optional[str] = None) -> Dict[str, Any]:
        """Get mapped properties for a numeric semantic class ID with dataset awareness."""
        from .semantic_registry import SemanticClassRegistry
        info = SemanticClassRegistry.get_class_info(class_id, dataset)
        return {
            "name": info.name,
            "category": info.category.value,
            "is_dynamic": info.is_dynamic,
            "clusterable": info.clusterable,
            "color_hex": info.color_hex,
            "surface_role": info.surface_role,
        }

    @classmethod
    def parse_kitti_label_bin(cls, raw_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """Parse SemanticKITTI .label binary file.
        
        Extracts lower 16 bits as semantic class ID and upper 16 bits as instance ID.
        
        Args:
            raw_bytes: Raw binary bytes of uint32 records.
            
        Returns:
            Tuple of (semantic_classes: np.ndarray[uint16], instance_ids: np.ndarray[uint16]).
            
        Raises:
            SemanticMappingError: If byte length is not a multiple of 4 or empty.
        """
        byte_len = len(raw_bytes)
        if byte_len == 0:
            raise SemanticMappingError("Empty .label file received (0 bytes).")

        if byte_len % 4 != 0:
            raise SemanticMappingError(
                f"Malformed SemanticKITTI .label file: size {byte_len} bytes is not a "
                f"multiple of 4 (uint32 per point label)."
            )

        try:
            raw_labels = np.frombuffer(raw_bytes, dtype=np.uint32)
            # Lower 16 bits = semantic class
            semantic_classes = (raw_labels & 0xFFFF).astype(np.uint16)
            # Upper 16 bits = instance ID
            instance_ids = (raw_labels >> 16).astype(np.uint16)
            return semantic_classes, instance_ids
        except Exception as e:
            raise SemanticMappingError(f"Failed to decode uint32 binary labels: {str(e)}")

    @classmethod
    def compute_distributions(
        cls,
        semantic_classes: np.ndarray,
        dataset: Optional[str] = None,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Compute class-wise and project category counts with dataset awareness."""
        class_counts: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}

        unique_ids, counts = np.unique(semantic_classes, return_counts=True)
        for class_id, count in zip(unique_ids, counts):
            info = cls.get_class_info(int(class_id), dataset=dataset)
            c_name = info["name"]
            cat_name = info["category"]

            class_counts[c_name] = class_counts.get(c_name, 0) + int(count)
            category_counts[cat_name] = category_counts.get(cat_name, 0) + int(count)

        return class_counts, category_counts

    @classmethod
    def build_sample_labeled_points(
        cls,
        points: np.ndarray,
        semantic_classes: np.ndarray,
        instance_ids: Optional[np.ndarray] = None,
        max_points: int = 4000,
        dataset: Optional[str] = None,
    ) -> List[SemanticPoint3D]:
        """Create a downsampled list of SemanticPoint3D for frontend visualization."""
        n_pts = points.shape[0]
        if n_pts == 0:
            return []

        if n_pts <= max_points:
            indices = np.arange(n_pts)
        else:
            step = n_pts / max_points
            indices = (np.arange(max_points) * step).astype(np.int64)

        result: List[SemanticPoint3D] = []
        for idx in indices:
            cid = int(semantic_classes[idx]) if idx < len(semantic_classes) else 0
            inst_id = int(instance_ids[idx]) if instance_ids is not None and idx < len(instance_ids) else 0
            info = cls.get_class_info(cid, dataset=dataset)

            result.append(
                SemanticPoint3D(
                    x=float(points[idx, 0]),
                    y=float(points[idx, 1]),
                    z=float(points[idx, 2]),
                    intensity=float(points[idx, 3]) if points.shape[1] > 3 else 0.0,
                    raw_label_id=cid,
                    semantic_class=info["name"],
                    project_category=info["category"],
                    instance_id=inst_id,
                )
            )

        return result

    @classmethod
    def get_project_category_mapping(cls) -> Dict[str, str]:
        """Return dict mapping semantic class_name -> project_category."""
        mapping = cls.load_mapping()
        res = {v["name"]: v["category"] for v in mapping.values()}
        res["unlabeled"] = "unknown"
        return res

    @classmethod
    def map_labels_batch(cls, raw_labels: np.ndarray) -> Tuple[List[str], List[str]]:
        """Map an array of raw uint32/uint16 labels to (class_names, project_categories)."""
        mapping = cls.load_mapping()
        class_names: List[str] = []
        project_categories: List[str] = []
        label_ids = (raw_labels & 0xFFFF).astype(int)
        for lid in label_ids:
            info = mapping.get(lid, {"name": "unlabeled", "category": "unknown"})
            class_names.append(info["name"])
            project_categories.append(info["category"])
        return class_names, project_categories

