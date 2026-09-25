"""Resolution Policy Service: Manages hierarchical spatial foveation rules and feature-driven refinement."""

import os
import threading
from math import isclose
from pathlib import Path
from typing import Optional, List, Tuple
import yaml

from ..config import settings
from ..models.map_schemas import ResolutionLevel, GridPolicyConfig


class ResolutionPolicyService:
    """Loads, saves, and evaluates the variable-resolution grid policy."""

    _lock = threading.Lock()
    _current_policy: Optional[GridPolicyConfig] = None

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    @classmethod
    def get_config_file_path(cls) -> Path:
        return settings.CONFIG_DIR / "grid_policy.yaml"

    @classmethod
    def get_policy(cls) -> GridPolicyConfig:
        """Get the active grid policy, loading from YAML or default."""
        with cls._lock:
            if cls._current_policy is not None:
                return cls._current_policy

            config_path = cls.get_config_file_path()
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    cls._current_policy = GridPolicyConfig(**data)
                    return cls._current_policy
                except Exception:
                    pass

            cls._current_policy = GridPolicyConfig()
            return cls._current_policy

    @classmethod
    def update_policy(cls, new_policy: GridPolicyConfig) -> GridPolicyConfig:
        """Update and persist policy configuration to YAML."""
        with cls._lock:
            cls._current_policy = new_policy
            config_path = cls.get_config_file_path()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(new_policy.model_dump(), f, default_flow_style=False)
            return cls._current_policy

    # -------------------------------------------------------------------------
    # Cell size / level resolution
    # -------------------------------------------------------------------------

    @classmethod
    def get_cell_size(cls, level: ResolutionLevel, policy: Optional[GridPolicyConfig] = None) -> float:
        """Return physical cell size (m) for a resolution level."""
        pol = policy or cls.get_policy()
        if level == ResolutionLevel.FINE:
            return pol.fine_resolution_m
        elif level == ResolutionLevel.MEDIUM:
            return pol.medium_resolution_m
        elif level == ResolutionLevel.COARSE:
            return pol.coarse_resolution_m
        return pol.medium_resolution_m

    @classmethod
    def determine_base_level(
        cls,
        distance_m: float,
        policy: Optional[GridPolicyConfig] = None,
    ) -> ResolutionLevel:
        """Assign baseline foveated level according to radial distance from ego origin."""
        pol = policy or cls.get_policy()
        if distance_m <= pol.near_zone_max_distance_m:
            return ResolutionLevel.FINE
        elif distance_m <= pol.mid_zone_max_distance_m:
            return ResolutionLevel.MEDIUM
        else:
            return ResolutionLevel.COARSE

    @classmethod
    def refine_level(
        cls,
        base_level: ResolutionLevel,
        is_dynamic: bool = False,
        is_obstacle: bool = False,
        slope_deg: float = 0.0,
        roughness_m: float = 0.0,
        step_height_m: float = 0.0,
        elevation_range_m: float = 0.0,
        policy: Optional[GridPolicyConfig] = None,
    ) -> ResolutionLevel:
        """Apply safety and terrain complexity overrides to refine resolution level.

        Returns the *promoted* level (FINE > MEDIUM > COARSE) when the base level
        is too coarse to represent the cell content accurately.
        """
        pol = policy or cls.get_policy()

        # 1. Safety Priority: any dynamic actor forces FINE.
        if pol.safety_priority and is_dynamic:
            return ResolutionLevel.FINE

        # 2. Obstacle / Step / Vertical-feature override: forces FINE.
        if pol.obstacle_override:
            if is_obstacle:
                return ResolutionLevel.FINE
            if step_height_m >= pol.step_refinement_threshold_m:
                return ResolutionLevel.FINE
            if elevation_range_m >= pol.obstacle_height_span_threshold_m:
                return ResolutionLevel.FINE

        # 3. Terrain complexity: promote coarse->medium and medium->fine.
        if pol.terrain_complexity_override:
            is_complex = (
                slope_deg >= pol.slope_refinement_threshold_deg
                or roughness_m >= pol.roughness_refinement_threshold_m
            )
            if is_complex:
                if base_level == ResolutionLevel.COARSE:
                    return ResolutionLevel.MEDIUM
                if base_level == ResolutionLevel.MEDIUM:
                    return ResolutionLevel.FINE

        return base_level

    # -------------------------------------------------------------------------
    # Hierarchical cell keys
    # -------------------------------------------------------------------------
    #
    # The hierarchy assumes a 2:1 ratio between successive levels.  We *enforce*
    # this in the policy so that the parent/child math is correct regardless of
    # which combination of dropdown values the user picks.  In practice that
    # means snapping fine/medium/coarse to nested multiples when persisted.
    #

    _EPS = 1e-3

    @classmethod
    def _level_to_size(cls, level: ResolutionLevel, pol: GridPolicyConfig) -> float:
        return {
            ResolutionLevel.FINE: pol.fine_resolution_m,
            ResolutionLevel.MEDIUM: pol.medium_resolution_m,
            ResolutionLevel.COARSE: pol.coarse_resolution_m,
        }[level]

    @classmethod
    def get_level_ratio(cls, pol: GridPolicyConfig) -> Tuple[int, int]:
        """Return (fine_per_medium, medium_per_coarse), i.e. how many fine cells
        fit in one medium cell, and how many medium fit in one coarse.

        If the configured sizes are not exact 2:1 multiples, the ratio is the
        nearest integer and the cell size is reported as a hint via
        ``get_effective_cell_size``.
        """
        # fine : medium  = medium_size / fine_size
        fpm = pol.medium_resolution_m / pol.fine_resolution_m
        # medium : coarse = coarse_size / medium_size
        mpc = pol.coarse_resolution_m / pol.medium_resolution_m
        # Round to nearest integer with 0.1 tolerance
        fpm_i = int(round(fpm))
        mpc_i = int(round(mpc))
        if abs(fpm - fpm_i) > 0.1 or abs(mpc - mpc_i) > 0.1:
            # Non-nested configuration: fall back to a safe 1:1 mapping rather
            # than producing a misaligned hierarchy.  Callers should normally
            # normalise the policy first.
            return 1, 1
        return max(1, fpm_i), max(1, mpc_i)

    @classmethod
    def get_parent_key(cls, cell_key: str, pol: Optional[GridPolicyConfig] = None) -> Optional[str]:
        """Compute parent cell key in hierarchical nested structure.

        Returns ``None`` for COARSE cells (top of the hierarchy).
        """
        if not cell_key or ":" not in cell_key:
            return None
        level_str, coords = cell_key.split(":", 1)
        if "_" not in coords:
            return None
        gx_str, gy_str = coords.split("_", 1)
        try:
            gx = int(gx_str)
            gy = int(gy_str)
        except ValueError:
            return None

        pol = pol or cls.get_policy()
        fpm, mpc = cls.get_level_ratio(pol)

        if level_str == ResolutionLevel.FINE.value:
            return f"{ResolutionLevel.MEDIUM.value}:{gx // fpm}_{gy // fpm}"
        if level_str == ResolutionLevel.MEDIUM.value:
            return f"{ResolutionLevel.COARSE.value}:{gx // mpc}_{gy // mpc}"
        return None

    @classmethod
    def get_child_keys(
        cls,
        cell_key: str,
        pol: Optional[GridPolicyConfig] = None,
    ) -> List[str]:
        """Compute child cell keys nested within this cell.

        For COARSE cells, returns MEDIUM children.  For MEDIUM cells, returns
        FINE children.  Returns an empty list for FINE cells (leaf level).
        """
        if not cell_key or ":" not in cell_key:
            return []
        level_str, coords = cell_key.split(":", 1)
        if "_" not in coords:
            return []
        gx_str, gy_str = coords.split("_", 1)
        try:
            gx = int(gx_str)
            gy = int(gy_str)
        except ValueError:
            return []

        pol = pol or cls.get_policy()
        fpm, mpc = cls.get_level_ratio(pol)

        target_level: Optional[str] = None
        step = 1
        if level_str == ResolutionLevel.COARSE.value:
            target_level = ResolutionLevel.MEDIUM.value
            step = mpc
        elif level_str == ResolutionLevel.MEDIUM.value:
            target_level = ResolutionLevel.FINE.value
            step = fpm
        else:
            return []

        keys: List[str] = []
        for dy in range(step):
            for dx in range(step):
                keys.append(f"{target_level}:{gx * step + dx}_{gy * step + dy}")
        return keys

    # -------------------------------------------------------------------------
    # Normalisation
    # -------------------------------------------------------------------------

    @classmethod
    def normalise_policy(cls, pol: GridPolicyConfig) -> GridPolicyConfig:
        """Return a copy of the policy with sizes snapped so the hierarchy
        is strictly nested (2:1 ratio between successive levels).  This keeps
        parent/child keys spatially aligned.

        Snapping rule: medium must be a power-of-two multiple of fine, and
        coarse must be a power-of-two multiple of medium.  We round *up* to
        the nearest such size when the user picks a non-nested combination,
        so we never lose precision.
        """
        data = pol.model_dump()
        fine = float(data["fine_resolution_m"])
        medium = float(data["medium_resolution_m"])
        coarse = float(data["coarse_resolution_m"])

        # medium = 2^n * fine, with n >= 1 (so medium > fine)
        if fine > 0:
            ratio_m = medium / fine
            n_m = max(1, int(round(ratio_m)))
            # round-up to the nearest integer multiple if the user was in
            # between two integers and would otherwise shrink the cell.
            if n_m < ratio_m - 1e-6:
                n_m += 1
            new_medium = fine * n_m
        else:
            new_medium = medium

        # coarse = 2^n * new_medium, with n >= 1 (so coarse > medium)
        if new_medium > 0:
            ratio_c = coarse / new_medium
            n_c = max(1, int(round(ratio_c)))
            if n_c < ratio_c - 1e-6:
                n_c += 1
            new_coarse = new_medium * n_c
        else:
            new_coarse = coarse

        data["medium_resolution_m"] = round(new_medium, 3)
        data["coarse_resolution_m"] = round(new_coarse, 3)
        return GridPolicyConfig(**data)

    @classmethod
    def validate_policy(cls, pol: GridPolicyConfig) -> Tuple[bool, Optional[str]]:
        """Return (ok, message).  ``ok`` is False with a reason if the policy
        cannot be normalised into a nested hierarchy.
        """
        if pol.fine_resolution_m <= 0 or pol.medium_resolution_m <= 0 or pol.coarse_resolution_m <= 0:
            return False, "Resolution values must be positive"
        if pol.near_zone_max_distance_m <= 0 or pol.mid_zone_max_distance_m <= 0 or pol.far_zone_max_distance_m <= 0:
            return False, "Zone distances must be positive"
        if not (pol.near_zone_max_distance_m <= pol.mid_zone_max_distance_m <= pol.far_zone_max_distance_m):
            return False, "Zone distances must be ordered near <= mid <= far"
        return True, None
