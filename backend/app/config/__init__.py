"""Configuration management for the LiDAR-X backend."""

import os
from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and runtime parameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "LiDAR-X"
    VERSION: str = "0.3.0"
    API_V1_STR: str = "/api/v1"

    # Data Directory Setup
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR_NAME: str = "data"

    @property
    def DATA_DIR(self) -> Path:
        return self.BASE_DIR / self.DATA_DIR_NAME

    @property
    def FRAMES_DIR(self) -> Path:
        return self.DATA_DIR / "frames"

    @property
    def TERRAIN_DIR(self) -> Path:
        return self.DATA_DIR / "terrain"

    @property
    def LABELS_DIR(self) -> Path:
        return self.DATA_DIR / "labels"

    @property
    def OBJECTS_DIR(self) -> Path:
        return self.DATA_DIR / "objects"

    @property
    def TRACKS_DIR(self) -> Path:
        return self.DATA_DIR / "tracks"

    @property
    def MAPS_DIR(self) -> Path:
        return self.DATA_DIR / "maps"

    @property
    def PREDICTIONS_DIR(self) -> Path:
        return self.DATA_DIR / "predictions"

    @property
    def SEQUENCES_DIR(self) -> Path:
        return self.DATA_DIR / "sequences"

    @property
    def METRICS_DIR(self) -> Path:
        return self.DATA_DIR / "metrics"

    @property
    def DEMO_CACHE_DIR(self) -> Path:
        return self.DATA_DIR / "demo_cache"

    @property
    def CONFIG_DIR(self) -> Path:
        return Path(__file__).resolve().parent

    # CORS settings
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, str) and v.startswith("["):
            import json
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return v if isinstance(v, list) else []

    # Upload Limits & Sensor Defaults (Supports arbitrary large sequences up to 50GB)
    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024 * 1024  # 50 GB for large multi-thousand frame archives
    DEFAULT_GRID_RESOLUTION: float = 1.0  # meters per grid cell
    DEFAULT_RANGE_MIN: float = 0.5  # meters (exclude points too close to ego sensor)
    DEFAULT_RANGE_MAX: float = 80.0  # meters
    DEFAULT_Z_MIN: float = -5.0  # meters relative to sensor origin
    DEFAULT_Z_MAX: float = 10.0  # meters relative to sensor origin


settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
settings.TERRAIN_DIR.mkdir(parents=True, exist_ok=True)
settings.LABELS_DIR.mkdir(parents=True, exist_ok=True)
settings.OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
settings.TRACKS_DIR.mkdir(parents=True, exist_ok=True)
settings.MAPS_DIR.mkdir(parents=True, exist_ok=True)
settings.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
settings.SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)
settings.METRICS_DIR.mkdir(parents=True, exist_ok=True)
settings.DEMO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

