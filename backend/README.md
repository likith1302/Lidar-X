# LiDAR-X Backend - Terrain Analysis, Semantic Tracking, 2.5D Adaptive Grid Engine, Fast-FRNet Inference & Sequence Replay

Modular, high-performance Python FastAPI backend powering geometric terrain analysis, SemanticKITTI LiDAR ingestion, point-wise Fast-FRNet deep semantic segmentation, DBSCAN object instance clustering, temporal multi-object tracking, the **Adaptive Variable-Resolution 2.5D Grid Engine**, and **Continuous LiDAR Sequence Video Replay with WebSocket Streaming** for LiDAR-X.

## Architecture Overview

```
backend/
├── app/
│   ├── main.py                  # FastAPI application entrypoint, CORS middleware, routing
│   ├── config/
│   │   ├── __init__.py          # Settings, storage paths, limits, sensor defaults
│   │   ├── model_settings.py    # SalsaNext checkpoint paths, inference device configuration
│   │   ├── semantic_mapping.yaml # SemanticKITTI to project categories mapping
│   │   └── grid_policy.yaml      # Variable-resolution spatial foveation rules & overrides
│   ├── api/
│   │   ├── routes/
│   │   │   ├── health.py        # GET /api/v1/health system status & feature flags
│   │   │   ├── frames.py        # LiDAR frame ingestion (.bin upload, JSON, details)
│   │   │   ├── terrain.py       # Geometric terrain analysis pipeline & cell queries
│   │   │   ├── semantic.py      # Semantic .label upload, prediction queries
│   │   │   ├── inference.py     # SalsaNext deep learning inference (status, segment, results)
│   │   │   ├── objects.py       # Geometric instance clustering (DBSCAN) & Kalman tracking
│   │   │   ├── grid_policy.py   # GET & PUT /api/v1/grid-policy
│   │   │   ├── maps.py          # 2.5D Adaptive Grid creation, cell inspection, export, reset
│   │   │   └── replay.py        # Sequence ZIP upload, playback lifecycle, WebSocket stream
│   ├── models/
│   │   ├── schemas.py           # Core LiDAR, preprocessing, and terrain schemas
│   │   ├── object_schemas.py    # Semantic, instance clustering, and tracking schemas
│   │   ├── map_schemas.py       # 2.5D Grid cell, metadata, policy, and export schemas
│   │   ├── inference_schemas.py  # SalsaNext status, job, prediction, and distribution schemas
│   │   └── replay_schemas.py     # Sequence replay session, status, frame payload schemas
│   └── services/
│       ├── lidar_parser.py      # SemanticKITTI float32 .bin parser & validator
│       ├── preprocessing.py     # NaN/Inf sanitization, range gating, transparent reporting
│       ├── terrain_analysis.py  # 2.5D elevation grid, slope, roughness, curb/step engine
│       ├── semantic_mapping.py  # .label binary parsing, bitmask extraction, category mapping
│       ├── salsanext_inference.py # SalsaNext spherical projection, PyTorch forward pass, unprojection
│       ├── semantic_label_mapping.py # Vectorized 20-class learning map to SemanticKITTI/LiDAR-X translation
│       ├── instance_clustering.py # Geometric DBSCAN instance segmentation
│       ├── object_tracking.py   # Kalman filter temporal tracking & lifecycle management
│       ├── sequence_replay.py   # ZIP sequence ingestion, frame validation, and pairing
│       ├── replay_session.py    # Thread-safe playback state machine, frame advancement, and live pipeline
│       ├── resolution_policy.py # Distance zone & feature override resolution evaluator
│       ├── observability.py     # Geometric ray visibility and occlusion sector analysis
│       ├── adaptive_grid.py     # Hierarchical foveated 2.5D elevation grid engine
│       ├── map_fusion.py        # Single-frame local vs multi-frame global pose fusion
│       ├── map_serialization.py # Disk persistence of maps in backend/data/maps/
│       └── storage.py           # File-based NPZ/JSON persistence
├── models/
│   └── salsanext/
│       ├── SalsaNext            # Official PyTorch checkpoint (exact filename, no extension)
│       ├── arch_cfg.yaml        # Range projection geometry (64x2048) & model architecture config
│       └── data_cfg.yaml        # SemanticKITTI 20-class learning map & normalization statistics
├── salsanext_source/            # Official SalsaNext neural network modules & LaserScan projection
├── tests/                       # Comprehensive Pytest test suite (56 unit & integration tests)
├── Dockerfile                   # Python 3.10 + PyTorch CUDA container for standalone inference
├── DEPLOYMENT.md                # GPU deployment and containerization guide
├── requirements.txt             # Python dependencies
├── .env.example                 # Example configuration
└── README.md
```

## Features

- **SalsaNext Deep Semantic Segmentation**:
  - Full spherical range-image projection ($[5, 64, 2048]$: $x, y, z, \text{depth}, \text{intensity}$).
  - PyTorch inference on CPU or NVIDIA GPU (`cuda`).
  - Strict 20-class SemanticKITTI learning-map decoding back to native uint32 `.label` files and project category mappings.
  - Transparent hardware reporting (`ready`, `model_missing`, `cpu_mode`, `gpu_available`).
- **Adaptive Variable-Resolution 2.5D Grid Engine**:
  - Projects classified 3D LiDAR points into hierarchical nested elevation cells.
  - Near-field fine spatial detail ($0.5\text{m}$), mid-field intermediate detail ($1.0\text{m}$), far-field coarse detail ($2.0\text{m}$).
  - Hierarchical nested indexing: 4 fine cells nest precisely inside 1 medium cell; 4 medium cells inside 1 coarse cell.
  - Feature-driven refinement: dynamic actors (`safety_priority`), steep/rough slopes (`terrain_complexity_override`), and static obstacles (`obstacle_override`) automatically refine to fine resolution.
  - Computes elevation metrics ($\min, \text{mean}, \max, \text{range}$), surface roughness, slope angle, dominant class, semantic histogram, and traversability states.
- **SemanticKITTI `.bin` & `.label` Ingestion**:
  - Ingests 4-channel float32 LiDAR scans `[x, y, z, intensity]`.
  - Ingests uint32 `.label` files with bitmask splitting: lower 16 bits = semantic class ID, upper 16 bits = instance ID.
- **Geometric Object Instance Clustering (DBSCAN)**:
  - Class-adaptive Euclidean clustering across discrete actor and obstacle classes.
  - Derived 3D bounding boxes, centroids, and extent categories.
- **Temporal Multi-Object Tracking (MOT)**:
  - Constant-velocity 3D Kalman filter for motion smoothing and velocity estimation.
  - Track lifecycle management: `new` -> `active` -> `temporarily_lost` -> `expired`.

---

## Getting Started

### 1. Prerequisites

- Python 3.10+ (tested on Python 3.10 and 3.13)
- PyTorch (`torch`, `torchvision`, `torchaudio`)
- `pip` package manager

### 2. Dependency Installation

```bash
cd backend
pip install -r requirements.txt
```

### 3. Model Checkpoint Setup

Download or place the pre-trained SalsaNext checkpoint at:
```
backend/models/salsanext/SalsaNext
```
*(Note: Do not add `.pth` extension. Keep exact filename `SalsaNext`)*

### 4. Running the Development Server

```bash
# From within the backend directory:
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation:
- **Swagger UI**: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
- **ReDoc**: [http://localhost:8000/api/v1/redoc](http://localhost:8000/api/v1/redoc)

---

## API Routes

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | System health check, active capabilities, and storage readiness |
| `GET` | `/api/v1/inference/status` | SalsaNext model status, device (`cuda`/`cpu`), GPU availability |
| `POST` | `/api/v1/inference/segment` | Run SalsaNext segmentation on `.bin` upload or `frame_id` |
| `GET` | `/api/v1/inference/results/{job_id}` | Retrieve segmentation results, distribution, and point predictions |
| `GET` | `/api/v1/inference/download/{frame_id}` | Download generated uint32 `.label` binary file |
| `POST` | `/api/v1/frames/upload` | Multipart upload for SemanticKITTI `.bin` files (`file` form field) |
| `POST` | `/api/v1/frames/upload-json` | JSON point cloud upload for testing |
| `GET` | `/api/v1/frames/{frame_id}` | Fetch frame metadata, bounds, and downsampled preview points |
| `GET` | `/api/v1/frames/` | List all stored frame IDs |
| `POST` | `/api/v1/terrain/analyze` | Execute geometric terrain analysis on a frame |
| `GET` | `/api/v1/terrain/{frame_id}` | Retrieve previously computed terrain analysis result |
| `POST` | `/api/v1/semantic/upload-labels` | Upload SemanticKITTI uint32 `.label` file |
| `POST` | `/api/v1/semantic/import-predictions` | Import SalsaNext / external model predictions JSON |
| `GET` | `/api/v1/semantic/{frame_id}` | Retrieve semantic annotations and labeled preview points |
| `POST` | `/api/v1/objects/detect` | Run DBSCAN instance clustering on labeled points |
| `GET` | `/api/v1/objects/{frame_id}` | Retrieve detected 3D object instances |
| `POST` | `/api/v1/tracks/update` | Update Kalman tracks with current frame detections |
| `GET` | `/api/v1/tracks/` | List all active temporal tracks |
| `GET` | `/api/v1/tracks/{track_id}` | Retrieve track trajectory and historical state |
| `GET` | `/api/v1/grid-policy` | Retrieve active variable-resolution distance thresholds & overrides |
| `PUT` | `/api/v1/grid-policy` | Update & persist variable-resolution grid policy parameters |
| `POST` | `/api/v1/maps/{map_id}/update` | Update 2.5D Adaptive Grid with frame LiDAR points & semantic classes |
| `GET` | `/api/v1/maps/{map_id}` | Retrieve 2.5D map metadata and cell count by resolution level |
| `GET` | `/api/v1/maps/{map_id}/cells` | Query sparse map cells with optional level & spatial bounding box filters |
| `GET` | `/api/v1/maps/{map_id}/cells/{cell_key}` | Inspect detailed geometric, semantic, and hierarchical attributes for a cell |
| `POST` | `/api/v1/maps/{map_id}/reset` | Clear and reset a map instance |
| `GET` | `/api/v1/maps/{map_id}/export` | Export complete 2.5D map JSON (metadata, all cells, policy snapshot) |
| `GET` | `/api/v1/pipeline/{frame_id}/status` | Truthful real-artifact pipeline gate status for canonical frame ID |
| `POST` | `/api/v1/pipeline/{frame_id}/process` | Execute sequential perception pipeline stages in strict dependency order |
| `POST` | `/api/v1/replay/upload-sequence` | Ingest SemanticKITTI sequence ZIP (`sequences/00/velodyne/` & `predictions/`) |
| `GET` | `/api/v1/replay/{session_id}/status` | Check playback state (`ready`, `playing`, `paused`, `completed`) |
| `POST` | `/api/v1/replay/{session_id}/start` | Start or resume sequence replay at specified FPS |
| `POST` | `/api/v1/replay/{session_id}/pause` | Pause sequence replay |
| `POST` | `/api/v1/replay/{session_id}/stop` | Stop replay and rewind to frame 0 |
| `POST` | `/api/v1/replay/{session_id}/seek` | Seek directly to specific frame index |
| `GET` | `/api/v1/replay/{session_id}/next-frame` | Step forward and process next frame synchronously |
| `WS` | `/api/v1/replay/{session_id}/stream` | Continuous WebSocket frame feed with backpressure & client control |

---

## Canonical Frame Identity & Artifact Schema

Artifacts across the entire perception stack are routed via `FrameRegistry`:

- **Point Cloud**: `backend/data/frames/{frame_id}/points.npz`
- **Metadata**: `backend/data/frames/{frame_id}/metadata.json`
- **Semantic Labels (NPZ)**: `backend/data/labels/{frame_id}_labels.npz`
- **Prediction Binary (.label)**: `backend/data/predictions/{frame_id}.label`
- **Geometric Terrain**: `backend/data/terrain/{frame_id}.json`
- **Detected Objects**: `backend/data/objects/{frame_id}.json`
- **2.5D Adaptive Maps**: `backend/data/maps/{map_id}.json`

---

## Running Backend Tests

Execute the complete Pytest test suite (66 tests):

```bash
python -m pytest backend/tests -v
```
