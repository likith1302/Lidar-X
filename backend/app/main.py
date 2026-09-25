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
    try:
        startup_info = FastFRNetInferenceService.validate_startup()
        logger.info(f"Fast-FRNet models initialized: {startup_info['status']} on {startup_info['device']}")
    except Exception as e:
        logger.warning(f"Fast-FRNet startup check encountered notice: {e}")

    try:
        replay_session_manager.ensure_available_sessions()
        logger.info("Available replay sessions initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not auto-initialize default replay session: {e}")
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


# Configure CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 Routes
app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(frames.router, prefix=f"{settings.API_V1_STR}/frames")
app.include_router(terrain.router, prefix=f"{settings.API_V1_STR}/terrain")
app.include_router(semantic.router, prefix=f"{settings.API_V1_STR}/semantic")
app.include_router(objects.router, prefix=f"{settings.API_V1_STR}/objects")
app.include_router(objects.tracks_router, prefix=f"{settings.API_V1_STR}/tracks")
app.include_router(grid_policy.router, prefix=settings.API_V1_STR)
app.include_router(maps.router, prefix=f"{settings.API_V1_STR}/maps")
app.include_router(inference.router, prefix=f"{settings.API_V1_STR}/inference")
app.include_router(replay.router, prefix=f"{settings.API_V1_STR}/replay")
app.include_router(metrics.router, prefix=f"{settings.API_V1_STR}/metrics")
app.include_router(pipeline.router, prefix=settings.API_V1_STR)


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
