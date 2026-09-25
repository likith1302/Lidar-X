/**
 * Multi-Object Tracking Service Layer
 * Coordinates with the Python FastAPI backend for temporal Kalman object tracking,
 * state estimation, and track lifecycle management across consecutive LiDAR frames.
 */

import { apiClient, ApiResponse } from './api';
import {
  TrackingUpdateRequest,
  TrackingUpdateResponse,
  TrackedObject,
} from '../types/objects';

export class TrackingService {
  /**
   * Update object tracks with instances detected in the current frame
   */
  public async updateTracks(
    request: TrackingUpdateRequest
  ): Promise<ApiResponse<TrackingUpdateResponse>> {
    return apiClient.post<TrackingUpdateResponse>(
      '/tracks/update',
      request
    );
  }

  /**
   * List all currently active tracked objects
   */
  public async listTracks(): Promise<ApiResponse<TrackedObject[]>> {
    return apiClient.get<TrackedObject[]>(
      '/tracks/',
      () => []
    );
  }

  /**
   * Fetch single track trajectory and historical state
   */
  public async getTrack(trackId: string): Promise<ApiResponse<TrackedObject>> {
    return apiClient.get<TrackedObject>(
      `/tracks/${encodeURIComponent(trackId)}`
    );
  }
}

export const trackingService = new TrackingService();
