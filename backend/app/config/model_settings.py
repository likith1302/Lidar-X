"""Configuration settings for Fast-FRNet deep learning inference models."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from ..config import settings

# Base backend directory
_DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent.parent


class FastFRNetSpec(BaseModel):
    """Configuration specification for a Fast-FRNet checkpoint variant."""
    name: str
    checkpoint_filename: str
    H: int
    W: int
    fov_up: float
    fov_down: float
    num_classes: int = 20
    ignore_index: int = 19
    intensity_scale: float = 1.0
    description: str


class ModelSettings(BaseModel):
    """Fast-FRNet model paths, hardware device preference, and inference settings."""

    # Base directory for Fast-FRNet models
    MODELS_DIR: Path = Field(
        default_factory=lambda: Path(
            os.environ.get("FAST_FRNET_MODELS_DIR", str(settings.BASE_DIR / "models"))
        )
    )

    # Checkpoint paths (resolved relative to backend directory)
    RELLIS_CHECKPOINT_PATH: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "FRNET_RELLIS_CHECKPOINT",
                str(settings.BASE_DIR / "models" / "best_frnet_rellis.pth"),
            )
        )
    )
    SEMANTICKITTI_CHECKPOINT_PATH: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "FRNET_SEMANTICKITTI_CHECKPOINT",
                str(settings.BASE_DIR / "models" / "best_frnet_semantickitti.pth"),
            )
        )
    )

    # Default model type: RELLIS is the primary model for LiDAR-X off-road pipeline
    DEFAULT_MODEL_TYPE: str = "rellis"

    # Backward compatibility aliases
    @property
    def MODEL_DIR(self) -> Path:
        return self.MODELS_DIR

    @property
    def CHECKPOINT_PATH(self) -> Path:
        return self.RELLIS_CHECKPOINT_PATH

    # Output predictions directory
    PREDICTIONS_DIR: Path = settings.DATA_DIR / "predictions"

    # Device configuration
    DEVICE_PREFERENCE: str = "cuda"  # Will fallback to "cpu" if CUDA unavailable
    BATCH_SIZE: int = 1

    # Model specifications
    SPECS: Dict[str, FastFRNetSpec] = {
        "rellis": FastFRNetSpec(
            name="Fast-FRNet (RELLIS-3D Off-Road)",
            checkpoint_filename="best_frnet_rellis.pth",
            H=32,
            W=512,
            fov_up=15.0,
            fov_down=-25.0,
            num_classes=20,
            ignore_index=19,
            intensity_scale=255.0,
            description="Fine-tuned for off-road Velodyne VLP-32C 32-beam perception (primary LiDAR-X model)",
        ),
        "semantickitti": FastFRNetSpec(
            name="Fast-FRNet (SemanticKITTI Urban)",
            checkpoint_filename="best_frnet_semantickitti.pth",
            H=64,
            W=512,
            fov_up=3.0,
            fov_down=-25.0,
            num_classes=20,
            ignore_index=19,
            intensity_scale=1.0,
            description="Pretrained for urban/general HDL-64E 64-beam perception",
        ),
    }

    # Sensor grid dimensions for active default
    PROJ_H: int = 32
    PROJ_W: int = 512
    NCLASSES: int = 20

    def get_checkpoint_path(self, model_type: str = "rellis") -> Path:
        """Resolve checkpoint path for specified model variant ('rellis' or 'semantickitti').

        Prioritizes deployment-optimized models, falling back to original checkpoints if needed.
        """
        normalized = model_type.lower().strip()
        if "kitti" in normalized or "urban" in normalized:
            # Check deployment compressed model first
            deploy = self.MODELS_DIR / "best_frnet_semantickitti_deploy.pth"
            if deploy.is_file():
                return deploy
            p = Path(self.SEMANTICKITTI_CHECKPOINT_PATH)
            if not p.is_file():
                fallback = self.MODELS_DIR / "best_frnet_semantickitti.pth"
                if fallback.is_file():
                    return fallback
            return p

        # Default to RELLIS: check deployment compressed model first
        deploy = self.MODELS_DIR / "best_frnet_rellis_deploy.pth"
        if deploy.is_file():
            return deploy
        p = Path(self.RELLIS_CHECKPOINT_PATH)
        if not p.is_file():
            fallback = self.MODELS_DIR / "best_frnet_rellis.pth"
            if fallback.is_file():
                return fallback
        return p

    def get_resolved_checkpoint_path(self, model_type: str = "rellis") -> Path:
        """Alias for get_checkpoint_path ensuring absolute resolved Path."""
        return self.get_checkpoint_path(model_type).resolve()

    def get_spec(self, model_type: str = "rellis") -> FastFRNetSpec:
        """Return FastFRNetSpec configuration for requested model variant."""
        normalized = model_type.lower().strip()
        if "kitti" in normalized or "urban" in normalized:
            return self.SPECS["semantickitti"]
        return self.SPECS["rellis"]


model_settings = ModelSettings()

# Ensure directories exist
model_settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_settings.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


