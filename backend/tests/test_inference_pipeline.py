"""Integration and unit tests for Fast-FRNet semantic segmentation inference pipeline."""

import io
from pathlib import Path
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.config.model_settings import model_settings
from app.models.inference_schemas import InferenceModelStatus
from app.models.fast_frnet import FastFRNet
from app.services.fast_frnet_inference import FastFRNetInferenceService
from app.services.salsanext_inference import SalsaNextInferenceService
from app.services.semantic_label_mapping import SemanticLabelMappingService


def test_semantic_label_mapping():
    """Verify bidirectional mappings between 20 learning classes, SemanticKITTI raw labels, and categories."""
    # Test learning map inverse
    learning_ids = np.array([0, 1, 6, 9, 13, 15, 18, 19], dtype=np.uint8)
    raw_labels = SemanticLabelMappingService.map_learning_to_raw_batch(learning_ids)

    expected_raw = np.array([0, 10, 30, 40, 50, 70, 80, 81], dtype=np.uint16)
    np.testing.assert_array_equal(raw_labels, expected_raw)

    # Test class names
    assert SemanticLabelMappingService.get_raw_class_name(10) == "car"
    assert SemanticLabelMappingService.get_raw_class_name(30) == "person"
    assert SemanticLabelMappingService.get_raw_class_name(40) == "road"
    assert SemanticLabelMappingService.get_raw_class_name(50) == "building"
    assert SemanticLabelMappingService.get_raw_class_name(80) == "pole"

    # Test project categories
    assert SemanticLabelMappingService.get_project_category(10) == "dynamic_object"
    assert SemanticLabelMappingService.get_project_category(30) == "dynamic_object"
    assert SemanticLabelMappingService.get_project_category(40) == "drivable"
    assert SemanticLabelMappingService.get_project_category(50) == "infrastructure"
    assert SemanticLabelMappingService.get_project_category(70) == "vegetation"
    assert SemanticLabelMappingService.get_project_category(80) == "static_obstacle"

    # Test distribution computation
    sample_raw = np.array([40, 40, 40, 10, 80], dtype=np.uint16)
    c_counts, cat_counts = SemanticLabelMappingService.compute_distributions(sample_raw)
    assert c_counts["road"] == 3
    assert c_counts["car"] == 1
    assert c_counts["pole"] == 1
    assert cat_counts["drivable"] == 3
    assert cat_counts["dynamic_object"] == 1
    assert cat_counts["static_obstacle"] == 1


def test_fast_frnet_model_forward():
    """Test FastFRNet pure-PyTorch forward pass produces [N, 20] logits for (N, 4) points."""
    model = FastFRNet(
        output_shape=(32, 512),
        fov_up=15.0,
        fov_down=-25.0,
        num_classes=20,
    )
    model.eval()

    # Generate 500 synthetic LiDAR points (N, 4)
    pts = np.random.uniform(-20.0, 20.0, (500, 4)).astype(np.float32)
    pts[:, 2] = -1.73
    pts[:, 3] = np.random.uniform(0.0, 1.0, 500).astype(np.float32)

    tensor_pts = torch.from_numpy(pts)
    with torch.no_grad():
        logits = model(tensor_pts)

    assert logits.shape == (500, 20)
    assert logits.dtype == torch.float32


def test_fast_frnet_inference_service_startup():
    """Verify FastFRNetInferenceService startup validator discovers checkpoints and passes smoke test."""
    info = FastFRNetInferenceService.validate_startup()
    assert info["status"] == "READY"
    assert Path(info["rellis_path"]).is_file()
    assert info["rellis_size_bytes"] > 0
    assert Path(info["semantickitti_path"]).is_file()
    assert info["semantickitti_size_bytes"] > 0
    assert info["smoke_test_points"] > 0


def test_inference_status_reporting(client: TestClient):
    """Test GET /api/v1/inference/status truthfully reports Fast-FRNet checkpoint and device status."""
    # Test default (RELLIS-3D)
    res_rellis = client.get("/api/v1/inference/status?model_type=rellis")
    assert res_rellis.status_code == 200
    data_rellis = res_rellis.json()
    assert "Fast-FRNet" in data_rellis["model_name"]
    assert data_rellis["active_model"] == "rellis"
    assert data_rellis["checkpoint_exists"] is True
    assert "best_frnet_rellis" in data_rellis["checkpoint_path"]
    assert data_rellis["num_classes"] == 20

    # Test SemanticKITTI
    res_kitti = client.get("/api/v1/inference/status?model_type=semantickitti")
    assert res_kitti.status_code == 200
    data_kitti = res_kitti.json()
    assert "Fast-FRNet" in data_kitti["model_name"]
    assert data_kitti["active_model"] == "semantickitti"
    assert data_kitti["checkpoint_exists"] is True
    assert "best_frnet_semantickitti" in data_kitti["checkpoint_path"]


def test_inference_execution_with_real_checkpoints(client: TestClient, flat_road_points, create_kitti_bin):
    """Test end-to-end Fast-FRNet inference, results query, and label download with RELLIS model."""
    bin_data = create_kitti_bin(flat_road_points)

    # 1. Run inference on uploaded .bin
    seg_res = client.post(
        "/api/v1/inference/segment",
        files={"file": ("segment_test.bin", io.BytesIO(bin_data), "application/octet-stream")},
        data={"model_type": "rellis"},
    )
    assert seg_res.status_code == 200
    job_data = seg_res.json()
    assert job_data["status"] == "complete"
    assert job_data["point_count"] == len(flat_road_points)
    job_id = job_data["job_id"]

    # 2. Query inference results
    results_res = client.get(f"/api/v1/inference/results/{job_id}")
    assert results_res.status_code == 200
    res_data = results_res.json()
    assert res_data["point_count"] == len(flat_road_points)
    assert len(res_data["sample_predictions"]) > 0
    assert "class_counts" in res_data
    assert "project_category_counts" in res_data

    # 3. Download label artifact
    frame_id = job_data["frame_id"]
    dl_res = client.get(f"/api/v1/inference/download/{frame_id}")
    assert dl_res.status_code == 200
    assert len(dl_res.content) == len(flat_road_points) * 4  # uint32 per point


def test_semantickitti_inference_execution(client: TestClient, flat_road_points, create_kitti_bin):
    """Test Fast-FRNet inference with SemanticKITTI pretrained checkpoint."""
    bin_data = create_kitti_bin(flat_road_points)

    seg_res = client.post(
        "/api/v1/inference/segment",
        files={"file": ("kitti_test.bin", io.BytesIO(bin_data), "application/octet-stream")},
        data={"model_type": "semantickitti"},
    )
    assert seg_res.status_code == 200
    job_data = seg_res.json()
    assert job_data["status"] == "complete"
    assert job_data["point_count"] == len(flat_road_points)


def test_backward_compatible_salsanext_adapter(flat_road_points):
    """Verify that SalsaNextInferenceService shim delegates seamlessly to Fast-FRNet."""
    status = SalsaNextInferenceService.get_status()
    assert status.checkpoint_exists is True
    assert "Fast-FRNet" in status.model_name

    raw_labels, inst_ids, meta = SalsaNextInferenceService.run_inference(flat_road_points)
    assert len(raw_labels) == len(flat_road_points)
    assert len(inst_ids) == len(flat_road_points)
    assert "class_counts" in meta


def test_inference_invalid_file(client: TestClient):
    """Test POST /api/v1/inference/segment returns 400 for malformed binary data."""
    bad_bytes = b"not_a_multiple_of_16"
    res = client.post(
        "/api/v1/inference/segment",
        files={"file": ("corrupt.bin", io.BytesIO(bad_bytes), "application/octet-stream")},
    )
    assert res.status_code == 400
