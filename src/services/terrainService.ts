/**
 * Terrain Analysis Service Layer
 * Coordinates with the Python FastAPI backend for geometric terrain analysis,
 * slope estimation, surface roughness calculation, and curb detection.
 */

import { apiClient, ApiResponse } from './api';
import {
  TerrainAnalysisRequest,
  TerrainAnalysisResponse,
  HealthResponse,
} from '../types/terrain';

export class TerrainService {
  /**
   * Request geometric terrain analysis on a specified LiDAR frame
   */
  public async analyzeTerrain(
    request: TerrainAnalysisRequest
  ): Promise<ApiResponse<TerrainAnalysisResponse>> {
    return apiClient.post<TerrainAnalysisResponse>(
      '/terrain/analyze',
      request
    );
  }

  /**
   * Fetch previously computed terrain analysis result for a given frame ID
   */
  public async getTerrainResult(
    frameId: string
  ): Promise<ApiResponse<TerrainAnalysisResponse>> {
    return apiClient.get<TerrainAnalysisResponse>(
      `/terrain/${encodeURIComponent(frameId)}`
    );
  }

  /**
   * Ping backend health status
   */
  public async checkHealth(): Promise<ApiResponse<HealthResponse>> {
    return apiClient.get<HealthResponse>('/health');
  }
}

export const terrainService = new TerrainService();
