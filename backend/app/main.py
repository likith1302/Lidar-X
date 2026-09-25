"""FastAPI Application Entrypoint for LiDAR-X Backend."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .api.routes import health, frames, terrain, semantic, objects, grid_policy, maps, inference, replay, metrics, pipeline
from .services.replay_session import replay_session_manager
from .services.fast_frnet_inference import FastFRNetInferenceService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown routines."""
    import os
    import gc
    try:
        is_render = os.environ.get("RENDER", "").lower() in ("true", "1")
        startup_info = FastFRNetInferenceService.validate_startup(skip_smoke_test=is_render)
        logger.info(f"Fast-FRNet models initialized: {startup_info['status']} on {startup_info['device']}")
    except Exception as e:
        logger.warning(f"Fast-FRNet startup check encountered notice: {e}")

    try:
        replay_session_manager.ensure_available_sessions()
        logger.info("Available replay sessions initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not auto-initialize default replay session: {e}")
    gc.collect()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-oriented backend for LiDAR-X: Geometric terrain analysis, "
        "SemanticKITTI & RELLIS-3D LiDAR ingestion, Fast-FRNet semantic segmentation, DBSCAN object clustering, "
        "temporal tracking, and Adaptive Variable-Resolution 2.5D Grid Engine."
    ),
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)


# Configure CORS for local development and Render deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"^https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes with both /api/v1 and root prefixes for seamless frontend compatibility
for pfx in (settings.API_V1_STR, ""):
    app.include_router(health.router, prefix=pfx)
    app.include_router(frames.router, prefix=f"{pfx}/frames")
    app.include_router(terrain.router, prefix=f"{pfx}/terrain")
    app.include_router(semantic.router, prefix=f"{pfx}/semantic")
    app.include_router(objects.router, prefix=f"{pfx}/objects")
    app.include_router(objects.tracks_router, prefix=f"{pfx}/tracks")
    app.include_router(grid_policy.router, prefix=pfx)
    app.include_router(maps.router, prefix=f"{pfx}/maps")
    app.include_router(maps.router, prefix=f"{pfx}/map")
    app.include_router(inference.router, prefix=f"{pfx}/inference")
    app.include_router(replay.router, prefix=f"{pfx}/replay")
    app.include_router(metrics.router, prefix=f"{pfx}/metrics")
    app.include_router(pipeline.router, prefix=pfx)


@app.get("/", tags=["Root"])
async def root():
    """Root landing endpoint with system status and documentation links."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": f"{settings.API_V1_STR}/docs",
        "health_url": f"{settings.API_V1_STR}/health",
        "status": "online",
    }
