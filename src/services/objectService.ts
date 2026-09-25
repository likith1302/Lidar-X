/**
 * Object Detection Service Layer
 * Coordinates with the Python FastAPI backend for geometric DBSCAN instance clustering
 * on semantically segmented LiDAR point clouds.
 */

import { apiClient, ApiResponse } from './api';
import {
  ObjectDetectionRequest,
  ObjectDetectionResponse,
} from '../types/objects';

export class ObjectService {
  /**
   * Run geometric object instance clustering on a labeled LiDAR frame
   */
  public async detectObjects(
    request: ObjectDetectionRequest
  ): Promise<ApiResponse<ObjectDetectionResponse>> {
    return apiClient.post<ObjectDetectionResponse>(
      '/objects/detect',
      request
    );
  }

  /**
   * Fetch previously computed object instances for a frame
   */
  public async getDetectedObjects(
    frameId: string
  ): Promise<ApiResponse<ObjectDetectionResponse>> {
    return apiClient.get<ObjectDetectionResponse>(
      `/objects/${encodeURIComponent(frameId)}`
    );
  }
}

export const objectService = new ObjectService();
