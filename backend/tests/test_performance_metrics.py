"""Unit and integration tests for Performance & Validation Metrics Service and API."""

import pytest
import numpy as np
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from app.main import app
from app.services.metrics_service import MetricsService, OBJECT_CLASS_GROUPS
from app.models.metric_schemas import (
    PipelineStageLatencies,
    SpatialBounds,
    AdaptiveGridResourceMetrics,
    ClassAccuracyMetric,
    DistanceBinMetric,
    AccuracyValidationMetrics,
    FramePerformanceMetrics,
    SessionPerformanceMetrics,
)

client = TestClient(app)


def test_spatial_bounds_and_uniform_baseline_calculation():
    """Verify analytical 3D uniform voxel baseline calculation."""
    # Create synthetic point cloud spanning 10m x 20m x 4m
    np.random.seed(42)
    x = np.linspace(-5.0, 5.0, 100)
    y = np.linspace(-10.0, 10.0, 100)
    z = np.linspace(-2.0, 2.0, 100)
    points = np.column_stack([x, y, z])

    bounds, total_voxels, baseline_bytes = MetricsService.compute_spatial_bounds_and_baseline(
        points, voxel_size_m=0.5, bytes_per_voxel=4
    )

    assert bounds.min_x == -5.0
    assert bounds.max_x == 5.0
    assert bounds.span_x == 10.0
    assert bounds.span_y == 20.0
    assert bounds.span_z == 4.0

    # nx = 10 / 0.5 = 20, ny = 20 / 0.5 = 40, nz = 4 / 0.5 = 8
    # total_voxels = 20 * 40 * 8 = 6400
    assert total_voxels == 6400
    assert baseline_bytes == 6400 * 4  # 25600 bytes


def test_adaptive_grid_resource_metrics_reduction():
    """Verify memory reduction calculation vs uniform baseline."""
    np.random.seed(42)
    points = np.random.uniform(-10.0, 10.0, size=(1000, 3))

    res_metrics = MetricsService.compute_adaptive_grid_resource_metrics(
        points=points,
        fine_count=100,
        medium_count=50,
        coarse_count=25,
        voxel_size_m=0.5,
        bytes_per_voxel=4,
    )

    assert res_metrics.input_point_count == 1000
    assert res_metrics.total_cells_count == 175
    assert res_metrics.fine_cells_count == 100
    assert res_metrics.medium_cells_count == 50
    assert res_metrics.coarse_cells_count == 25
    assert res_metrics.adaptive_grid_bytes == 175 * 64
    assert res_metrics.memory_reduction_percentage > 0.0
    assert res_metrics.memory_reduction_percentage <= 100.0


def test_accuracy_metrics_with_ground_truth():
    """Verify point-wise accuracy, IoU, per-class metrics, and distance binning."""
    np.random.seed(42)
    n_pts = 1000
    # Points placed at various distances from origin
    r = np.linspace(1.0, 80.0, n_pts)
    theta = np.linspace(0, 2 * np.pi, n_pts)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros(n_pts)
    points = np.column_stack([x, y, z])

    # Ground truth: classes 10 (car), 40 (road), 70 (vegetation)
    gt_labels = np.zeros(n_pts, dtype=np.uint16)
    gt_labels[:400] = 10  # car
    gt_labels[400:800] = 40  # road
    gt_labels[800:] = 70  # vegetation

    # Predictions: 90% correct
    pred_labels = gt_labels.copy()
    corrupt_idx = np.random.choice(n_pts, size=100, replace=False)
    pred_labels[corrupt_idx] = 80  # pole

    acc_res = MetricsService.compute_accuracy_metrics(points, pred_labels, gt_labels)

    assert acc_res.accuracy_available is True
    assert acc_res.reason_unavailable is None
    assert acc_res.evaluated_points_count == 1000
    assert acc_res.overall_accuracy is not None
    assert 0.85 <= acc_res.overall_accuracy <= 0.95
    assert acc_res.mean_iou is not None
    assert acc_res.mean_iou > 0.0

    # Check per-class metrics
    class_names = [m.class_name for m in acc_res.per_class_metrics]
    assert "car" in class_names
    assert "road" in class_names
    assert "vegetation" in class_names

    # Check distance bins
    assert len(acc_res.distance_bins) == 4
    bin_ranges = [b.bin_range for b in acc_res.distance_bins]
    assert bin_ranges == ["0-10m", "10-30m", "30-50m", "50-100m"]
    assert all(b.point_count > 0 for b in acc_res.distance_bins)

    # Check object class summaries
    assert "vehicle" in acc_res.object_class_summaries
    vehicle_metric = acc_res.object_class_summaries["vehicle"]
    assert vehicle_metric.support_points == 400


def test_accuracy_metrics_missing_ground_truth():
    """Verify that when ground truth is missing, accuracy_available is False without fabricated numbers."""
    points = np.random.uniform(-5, 5, size=(100, 3))
    preds = np.ones(100, dtype=np.uint16) * 10

    acc_res = MetricsService.compute_accuracy_metrics(points, preds, gt_labels=None)

    assert acc_res.accuracy_available is False
    assert acc_res.overall_accuracy is None
    assert acc_res.mean_iou is None
    assert acc_res.evaluated_points_count is None
    assert len(acc_res.per_class_metrics) == 0
    assert "No ground-truth" in (acc_res.reason_unavailable or "")


def test_record_and_retrieve_frame_and_session_metrics():
    """Verify recording frame metrics, retrieving them, and session aggregation."""
    # 1. Record frame 1
    latencies_1 = PipelineStageLatencies(
        lidar_preprocessing_ms=1.5,
        salsanext_inference_ms=12.0,
        terrain_analysis_ms=3.2,
        object_detection_tracking_ms=4.1,
        adaptive_grid_ms=2.8,
        total_latency_ms=23.6,
    )
    bounds = SpatialBounds(
        min_x=-10.0, max_x=10.0, min_y=-10.0, max_y=10.0, min_z=-2.0, max_z=2.0,
        span_x=20.0, span_y=20.0, span_z=4.0, volume_m3=1600.0,
    )
    res_1 = AdaptiveGridResourceMetrics(
        input_point_count=5000,
        fine_cells_count=200,
        medium_cells_count=100,
        coarse_cells_count=50,
        total_cells_count=350,
        adaptive_grid_bytes=350 * 64,
        serialized_map_bytes=350 * 128,
        uniform_baseline_bytes=100000,
        uniform_voxel_size_m=0.5,
        uniform_bytes_per_voxel=4,
        uniform_total_voxels=25000,
        memory_reduction_percentage=77.6,
        spatial_bounds=bounds,
    )
    acc_1 = AccuracyValidationMetrics(
        accuracy_available=True,
        overall_accuracy=0.92,
        mean_iou=0.85,
        evaluated_points_count=5000,
        per_class_metrics=[],
        object_class_summaries={},
        distance_bins=[],
    )
    frame_1 = FramePerformanceMetrics(
        frame_id="test_frame_0001",
        session_id="test_session_A",
        timestamp=datetime.now(timezone.utc).isoformat(),
        device_used="cpu",
        model_name="SalsaNext",
        timings=latencies_1,
        actual_fps=42.37,
        resource_metrics=res_1,
        accuracy_metrics=acc_1,
    )

    MetricsService.record_frame_metrics(frame_1)
    retrieved_1 = MetricsService.get_frame_metrics("test_frame_0001")
    assert retrieved_1 is not None
    assert retrieved_1.frame_id == "test_frame_0001"
    assert retrieved_1.timings.total_latency_ms == 23.6

    # 2. Record frame 2
    latencies_2 = PipelineStageLatencies(
        lidar_preprocessing_ms=1.8,
        salsanext_inference_ms=13.0,
        terrain_analysis_ms=3.0,
        object_detection_tracking_ms=4.0,
        adaptive_grid_ms=3.0,
        total_latency_ms=24.8,
    )
    frame_2 = FramePerformanceMetrics(
        frame_id="test_frame_0002",
        session_id="test_session_A",
        timestamp=datetime.now(timezone.utc).isoformat(),
        device_used="cpu",
        model_name="SalsaNext",
        timings=latencies_2,
        actual_fps=40.32,
        resource_metrics=res_1,
        accuracy_metrics=acc_1,
    )
    MetricsService.record_frame_metrics(frame_2)

    # 3. Aggregate session
    session_res = MetricsService.record_session_metrics("test_session_A", "Sequence 00")
    assert session_res.session_id == "test_session_A"
    assert session_res.total_frames_processed == 2
    assert session_res.average_timings.total_latency_ms == 24.2
    assert session_res.average_fps == 41.34


def test_api_metrics_endpoints():
    """Verify HTTP API endpoints for /api/v1/metrics/."""
    # Test GET /api/v1/metrics/latest
    res = client.get("/api/v1/metrics/latest")
    assert res.status_code == 200
    data = res.json()
    assert "timings" in data
    assert "resource_metrics" in data
    assert "accuracy_metrics" in data
    assert "actual_fps" in data

    # Test GET /api/v1/metrics/frame/test_frame_0001
    res_f = client.get("/api/v1/metrics/frame/test_frame_0001")
    assert res_f.status_code == 200
    data_f = res_f.json()
    assert data_f["frame_id"] == "test_frame_0001"
    assert data_f["timings"]["salsanext_inference_ms"] == 12.0

    # Test GET /api/v1/metrics/session/test_session_A
    res_s = client.get("/api/v1/metrics/session/test_session_A")
    assert res_s.status_code == 200
    data_s = res_s.json()
    assert data_s["session_id"] == "test_session_A"
    assert data_s["total_frames_processed"] == 2

    # Test 404 for non-existent frame
    res_404 = client.get("/api/v1/metrics/frame/non_existent_frame_99999")
    assert res_404.status_code == 404
