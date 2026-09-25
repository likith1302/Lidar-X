"""Integration tests for 2.5D Adaptive Grid & Map API Routes."""

import io
import pytest
from fastapi.testclient import TestClient


def test_grid_policy_endpoints(client: TestClient):
    """Test GET and PUT /grid-policy endpoints."""
    # 1. GET active policy
    res = client.get("/api/v1/grid-policy")
    assert res.status_code == 200
    data = res.json()
    assert "near_zone_max_distance_m" in data
    assert "fine_resolution_m" in data

    # 2. PUT update policy
    updated_payload = data.copy()
    updated_payload["near_zone_max_distance_m"] = 15.0
    updated_payload["fine_resolution_m"] = 0.4

    put_res = client.put("/api/v1/grid-policy", json=updated_payload)
    assert put_res.status_code == 200
    put_data = put_res.json()
    assert put_data["near_zone_max_distance_m"] == 15.0
    assert put_data["fine_resolution_m"] == 0.4


def test_map_workflow_routes(client: TestClient, flat_road_points, create_kitti_bin):
    """Test full map update, cell query, inspector, export, and reset lifecycle."""
    # 1. Upload a frame
    bin_data = create_kitti_bin(flat_road_points)
    upload_res = client.post(
        "/api/v1/frames/upload",
        files={"file": ("map_test_frame.bin", io.BytesIO(bin_data), "application/octet-stream")},
    )
    assert upload_res.status_code == 201
    frame_id = upload_res.json()["frame_id"]

    map_id = "test_run_map_01"

    # 2. Update map in local_only mode
    update_res = client.post(
        f"/api/v1/maps/{map_id}/update",
        json={"frame_id": frame_id},
    )
    assert update_res.status_code == 200
    update_data = update_res.json()
    assert update_data["map_id"] == map_id
    assert update_data["mode"] == "local_only"
    assert update_data["updated_cell_count"] > 0
    assert len(update_data["cells_sample"]) > 0

    # 3. GET map metadata
    meta_res = client.get(f"/api/v1/maps/{map_id}")
    assert meta_res.status_code == 200
    meta_data = meta_res.json()
    assert meta_data["total_cells"] == update_data["updated_cell_count"]
    assert meta_data["fine_cells_count"] > 0

    # 4. GET map cells with level filter
    cells_res = client.get(f"/api/v1/maps/{map_id}/cells?level=fine&limit=50")
    assert cells_res.status_code == 200
    cells = cells_res.json()
    assert len(cells) > 0
    first_cell_key = cells[0]["cell_key"]

    # 5. GET single cell details
    cell_detail_res = client.get(f"/api/v1/maps/{map_id}/cells/{first_cell_key}")
    assert cell_detail_res.status_code == 200
    cell_detail = cell_detail_res.json()
    assert cell_detail["cell_key"] == first_cell_key
    assert "elevation_mean" in cell_detail
    assert "traversability_state" in cell_detail
    assert "dominant_semantic_class" in cell_detail

    # 6. GET export map
    export_res = client.get(f"/api/v1/maps/{map_id}/export")
    assert export_res.status_code == 200
    export_data = export_res.json()
    assert export_data["map_id"] == map_id
    assert len(export_data["cells"]) == meta_data["total_cells"]
    assert "policy_snapshot" in export_data

    # 7. POST reset map
    reset_res = client.post(f"/api/v1/maps/{map_id}/reset")
    assert reset_res.status_code == 200
    assert reset_res.json()["status"] == "reset_complete"

    # 8. Check metadata after reset
    meta_after = client.get(f"/api/v1/maps/{map_id}")
    assert meta_after.status_code == 200
    assert meta_after.json()["total_cells"] == 0


def test_map_errors(client: TestClient):
    """Test 404 responses for missing frames, maps, and cell keys."""
    # Non-existent frame update
    res = client.post(
        "/api/v1/maps/non_existent_map/update",
        json={"frame_id": "missing_frame_99999"},
    )
    assert res.status_code == 404

    # Non-existent map metadata
    res = client.get("/api/v1/maps/never_created_map")
    assert res.status_code == 404

    # Non-existent cell in non-existent map
    res = client.get("/api/v1/maps/never_created_map/cells/fine:0_0")
    assert res.status_code == 404
