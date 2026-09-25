/**
 * Grid Policy Service
 * Connects to /api/v1/grid-policy for configurable variable-resolution rules.
 */

import { apiClient, ApiResponse } from './api';
import { GridPolicyConfig } from '../types/map';

export const defaultGridPolicyConfig: GridPolicyConfig = {
  near_zone_max_distance_m: 10.0,
  mid_zone_max_distance_m: 30.0,
  far_zone_max_distance_m: 100.0,
  fine_resolution_m: 0.05,
  medium_resolution_m: 0.20,
  coarse_resolution_m: 0.50,
  safety_priority: true,
  terrain_complexity_override: true,
  obstacle_override: true,
  slope_refinement_threshold_deg: 10.0,
  roughness_refinement_threshold_m: 0.10,
  step_refinement_threshold_m: 0.10,
  obstacle_height_span_threshold_m: 0.30,
  default_cells_query_limit: 5000,
};

export class GridPolicyService {
  public async getPolicy(): Promise<ApiResponse<GridPolicyConfig>> {
    return apiClient.request<GridPolicyConfig>(
      '/grid-policy',
      { method: 'GET' },
      () => defaultGridPolicyConfig
    );
  }

  public async updatePolicy(newPolicy: GridPolicyConfig): Promise<ApiResponse<GridPolicyConfig>> {
    return apiClient.request<GridPolicyConfig>(
      '/grid-policy',
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newPolicy),
      },
      () => newPolicy
    );
  }
}

export const gridPolicyService = new GridPolicyService();
