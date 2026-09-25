/**
 * Performance & Validation Metrics Service.
 * Strictly communicates with backend metrics endpoints without mock or simulated values.
 */

import { apiClient, ApiResponse } from './api';
import { FramePerformanceMetrics, SessionPerformanceMetrics } from '../types/metrics';

class MetricsApiService {
  /**
   * Fetch the most recent frame's performance and validation metrics.
   * Returns null if no frame has been measured yet.
   */
  public async getLatestMetrics(): Promise<ApiResponse<FramePerformanceMetrics | null>> {
    try {
      const response = await apiClient.get<FramePerformanceMetrics>('/metrics/latest');
      return {
        success: true,
        data: response.data,
        isMock: false,
        timestamp: new Date().toISOString(),
      };
    } catch {
      return {
        success: false,
        data: null,
        isMock: false,
        message: 'No measured data yet',
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Fetch performance metrics for a specific frame ID.
   */
  public async getFrameMetrics(frameId: string): Promise<ApiResponse<FramePerformanceMetrics | null>> {
    try {
      const response = await apiClient.get<FramePerformanceMetrics>(`/metrics/frame/${encodeURIComponent(frameId)}`);
      return {
        success: true,
        data: response.data,
        isMock: false,
        timestamp: new Date().toISOString(),
      };
    } catch {
      return {
        success: false,
        data: null,
        isMock: false,
        message: `No metrics available for frame ${frameId}`,
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Fetch aggregated session metrics for a replay sequence.
   */
  public async getSessionMetrics(sessionId: string): Promise<ApiResponse<SessionPerformanceMetrics | null>> {
    try {
      const response = await apiClient.get<SessionPerformanceMetrics>(`/metrics/session/${encodeURIComponent(sessionId)}`);
      return {
        success: true,
        data: response.data,
        isMock: false,
        timestamp: new Date().toISOString(),
      };
    } catch {
      return {
        success: false,
        data: null,
        isMock: false,
        message: `No metrics available for session ${sessionId}`,
        timestamp: new Date().toISOString(),
      };
    }
  }
}

export const metricsService = new MetricsApiService();
