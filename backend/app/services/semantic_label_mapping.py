"""Semantic Label Mapping Service: Bridges Fast-FRNet learning classes, RELLIS-3D & SemanticKITTI raw labels, and LiDAR-X project categories."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class SemanticLabelMappingService:
    """Manages translations between 20 Fast-FRNet learning IDs (0..19), raw labels (RELLIS-3D & SemanticKITTI), and project categories."""

    # Pre-built lookup tables for zero-overhead vectorization
    _sk_learning_inv_lookup: Optional[np.ndarray] = None
    _sk_darknet_learning_inv_lookup: Optional[np.ndarray] = None
    _rl_learning_inv_lookup: Optional[np.ndarray] = None
    _cached_data_cfg: Optional[Dict[str, Any]] = None
    _learning_map_inv_lookup: Optional[np.ndarray] = None

    @classmethod
    def load_data_cfg(cls) -> Dict[str, Any]:
        """Backward-compatible helper returning dictionary of learning maps and labels."""
        cls._ensure_lookups()
        cls._cached_data_cfg = {
            "learning_map_inv": cls._SK_DARKNET_LEARNING_MAP_INV,
            "labels": cls._SK_RAW_NAMES,
        }
        cls._learning_map_inv_lookup = cls._sk_darknet_learning_inv_lookup
        return cls._cached_data_cfg

    # SemanticKITTI taxonomies
    _SK_RAW_NAMES: Dict[int, str] = {
        0: "unlabeled", 1: "outlier", 10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle",
        16: "on-rails", 18: "truck", 20: "other-vehicle", 30: "person", 31: "bicyclist", 32: "motorcyclist",
        40: "road", 44: "parking", 48: "sidewalk", 49: "other-ground", 50: "building",
        51: "fence", 52: "other-structure", 60: "lane-marking", 70: "vegetation", 71: "trunk",
        72: "terrain", 80: "pole", 81: "traffic-sign", 99: "other-object",
        252: "moving-car", 253: "moving-bicyclist", 254: "moving-person", 255: "moving-motorcyclist",
        256: "moving-on-rails", 257: "moving-bus", 258: "moving-truck", 259: "moving-other-vehicle",
    }

    # Standard SemanticKITTI taxonomy (0: unlabeled, 1: car, ..., 9: road, ..., 19: traffic-sign)
    _SK_DARKNET_LEARNING_MAP_INV: Dict[int, int] = {
        0: 0, 1: 10, 2: 11, 3: 15, 4: 18, 5: 20, 6: 30, 7: 31, 8: 32, 9: 40,
        10: 44, 11: 48, 12: 49, 13: 50, 14: 51, 15: 70, 16: 71, 17: 72, 18: 80, 19: 81,
    }

    # Fast-FRNet & SalsaNext SemanticKITTI taxonomy aligned with standard benchmark evaluations
    _SK_LEARNING_MAP_INV: Dict[int, int] = _SK_DARKNET_LEARNING_MAP_INV

    # RELLIS-3D taxonomies
    _RL_RAW_NAMES: Dict[int, str] = {
        0: "void", 1: "dirt", 3: "grass", 4: "tree", 5: "pole", 6: "water",
        7: "sky", 8: "vehicle", 9: "object", 10: "asphalt", 12: "building",
        15: "log", 17: "person", 18: "fence", 19: "bush", 23: "concrete",
        27: "barrier", 31: "puddle", 33: "mud", 34: "rubble",
    }

    _RL_LEARNING_MAP_INV: Dict[int, int] = {
        0: 1, 1: 3, 2: 4, 3: 5, 4: 6,
        5: 8, 6: 9, 7: 10, 8: 12, 9: 15,
        10: 17, 11: 18, 12: 19, 13: 23, 14: 27,
        15: 31, 16: 33, 17: 34, 18: 0, 19: 0,
    }

    # Isolated SemanticKITTI Project Categories
    _SK_RAW_TO_PROJECT: Dict[int, str] = {
        0: "unknown",           # unlabeled
        1: "unknown",           # outlier
        10: "dynamic_object",   # car
        11: "dynamic_object",   # bicycle
        13: "dynamic_object",   # bus
        15: "dynamic_object",   # motorcycle
        16: "dynamic_object",   # on-rails
        18: "dynamic_object",   # truck
        20: "dynamic_object",   # other-vehicle
        30: "dynamic_object",   # person
        31: "dynamic_object",   # bicyclist
        32: "dynamic_object",   # motorcyclist
        40: "drivable",         # road
        44: "drivable",         # parking
        48: "non_drivable",     # sidewalk
        49: "non_drivable",     # other-ground
        50: "infrastructure",   # building
        51: "static_obstacle",  # fence
        52: "infrastructure",   # other-structure
        60: "drivable",         # lane-marking
        70: "vegetation",       # vegetation
        71: "vegetation",       # trunk
        72: "non_drivable",     # terrain
        80: "static_obstacle",  # pole
        81: "static_obstacle",  # traffic-sign
        99: "unknown",
        252: "dynamic_object",  # moving-car
        253: "dynamic_object",  # moving-bicyclist
        254: "dynamic_object",  # moving-person
        255: "dynamic_object",  # moving-motorcyclist
        256: "dynamic_object",  # moving-on-rails
        257: "dynamic_object",  # moving-bus
        258: "dynamic_object",  # moving-truck
        259: "dynamic_object",  # moving-other-vehicle
    }

    # Isolated RELLIS-3D Project Categories
    _RL_RAW_TO_PROJECT: Dict[int, str] = {
        0: "unknown",           # void
        1: "drivable",          # dirt (drivable off-road trail)
        3: "vegetation",        # grass
        4: "vegetation",        # tree
        5: "static_obstacle",   # pole
        6: "non_drivable",      # water
        7: "unknown",           # sky
        8: "dynamic_object",    # vehicle
        9: "static_obstacle",   # object
        10: "drivable",         # asphalt (paved road)
        12: "infrastructure",   # building
        15: "static_obstacle",  # log
        17: "dynamic_object",   # person
        18: "static_obstacle",  # fence
        19: "vegetation",       # bush
        23: "drivable",         # concrete
        27: "static_obstacle",  # barrier
        31: "drivable",         # puddle
        33: "non_drivable",     # mud
        34: "static_obstacle",  # rubble
    }

    # Backward-compatibility alias defaulting to SemanticKITTI
    _RAW_TO_CAT: Dict[int, str] = _SK_RAW_TO_PROJECT

    @classmethod
    def _ensure_lookups(cls):
        """Construct fast vectorized NumPy lookup tables if not already initialized."""
        if cls._sk_learning_inv_lookup is None:
            sk_arr = np.zeros(256, dtype=np.uint16)
            for k, v in cls._SK_LEARNING_MAP_INV.items():
                sk_arr[int(k)] = int(v)
            cls._sk_learning_inv_lookup = sk_arr

        if cls._sk_darknet_learning_inv_lookup is None:
            sk_dn_arr = np.zeros(256, dtype=np.uint16)
            for k, v in cls._SK_DARKNET_LEARNING_MAP_INV.items():
                sk_dn_arr[int(k)] = int(v)
            cls._sk_darknet_learning_inv_lookup = sk_dn_arr

        if cls._rl_learning_inv_lookup is None:
            rl_arr = np.zeros(256, dtype=np.uint16)
            for k, v in cls._RL_LEARNING_MAP_INV.items():
                rl_arr[int(k)] = int(v)
            cls._rl_learning_inv_lookup = rl_arr

    @classmethod
    def map_learning_to_raw_batch(
        cls, learning_ids: np.ndarray, dataset: Optional[str] = None
    ) -> np.ndarray:
        """Vectorized conversion of learning class predictions (0..19) into raw class IDs.

        Args:
            learning_ids: (N,) uint8 or int array of 0..19 model predictions
            dataset: "rellis", "semantickitti", or "salsanext". If None, uses active session configuration.
        """
        cls._ensure_lookups()
        safe_ids = np.clip(learning_ids, 0, 255).astype(np.uint8)

        d = (dataset or "").lower().strip()
        if "rellis" in d:
            return cls._rl_learning_inv_lookup[safe_ids]
        elif "salsa" in d or "darknet" in d:
            return cls._sk_darknet_learning_inv_lookup[safe_ids]
        elif cls._learning_map_inv_lookup is not None and dataset is None:
            return cls._learning_map_inv_lookup[safe_ids]
        return cls._sk_learning_inv_lookup[safe_ids]

    @classmethod
    def get_raw_class_name(cls, raw_label_id: int, dataset: Optional[str] = None) -> str:
        """Get descriptive name for raw label ID across RELLIS and SemanticKITTI ontologies."""
        from .semantic_registry import SemanticClassRegistry
        return SemanticClassRegistry.get_name(raw_label_id, dataset=dataset)

    @classmethod
    def get_project_category(cls, raw_label_id: int, dataset: Optional[str] = None) -> str:
        """Get project category (e.g. drivable, static_obstacle, dynamic_object) with strict dataset isolation."""
        d = (dataset or "").lower().strip()
        if "rellis" in d:
            return cls._RL_RAW_TO_PROJECT.get(raw_label_id, "unknown")
        return cls._SK_RAW_TO_PROJECT.get(raw_label_id, "unknown")

    @classmethod
    def compute_distributions(
        cls, raw_labels: np.ndarray, dataset: Optional[str] = None
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Compute class-wise and category-wise count distributions for a point cloud."""
        from .semantic_registry import SemanticClassRegistry
        class_counts: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}

        unique_ids, counts = np.unique(raw_labels, return_counts=True)
        for lid, cnt in zip(unique_ids, counts):
            raw_id = int(lid)
            info = SemanticClassRegistry.get_class_info(raw_id, dataset=dataset)
            c_name = info.name
            cat_name = cls.get_project_category(raw_id, dataset=dataset)

            class_counts[c_name] = class_counts.get(c_name, 0) + int(cnt)
            category_counts[cat_name] = category_counts.get(cat_name, 0) + int(cnt)

        return class_counts, category_counts
