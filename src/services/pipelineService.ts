/**
 * Pipeline Service
 * Interacts with backend pipeline gate status and sequential execution APIs.
 */

import { apiClient, ApiResponse } from './api';
import { PipelineStatusResponse, PipelineProcessResponse } from '../types/pipeline';

export class PipelineService {
  /**
   * Query truthful disk-persisted perception pipeline gate status for a canonical frame ID.
   */
  public static async getPipelineStatus(frameId: string): Promise<ApiResponse<PipelineStatusResponse>> {
    return apiClient.get<PipelineStatusResponse>(
      `/pipeline/${encodeURIComponent(frameId)}/status`,
      () => ({
        frame_id: frameId,
        frame_exists: false,
        point_cloud_available: false,
        point_count: 0,
        semantic_labels_available: false,
        semantic_label_count: 0,
        labels_match_point_count: false,
        terrain_available: false,
        object_detection_available: false,
        object_instance_count: 0,
        adaptive_map_available: false,
        adaptive_grid_cell_count: 0,
        map_mode: 'local_only',
        last_error: 'Operating in mock preview mode without live backend connection.',
      })
    );
  }

  /**
   * Execute sequential perception pipeline stages on the backend.
   */
  public static async processPipeline(
    frameId: string,
    stages: string[] = ['fast_frnet', 'terrain', 'objects', 'adaptive_grid'],
    modelType: string = 'rellis'
  ): Promise<ApiResponse<PipelineProcessResponse>> {
    return apiClient.post<PipelineProcessResponse>(
      `/pipeline/${encodeURIComponent(frameId)}/process`,
      { stages, model_type: modelType },
      () => ({
        frame_id: frameId,
        success: false,
        status: 'mock_preview',
        executed_stages: stages.map((st) => ({
          stage: st,
          status: 'skipped',
          message: 'Mock preview mode: Live backend required to run real perception pipeline.',
          item_count: 0,
        })),
        pipeline_status: {
          frame_id: frameId,
          frame_exists: false,
          point_cloud_available: false,
          point_count: 0,
          semantic_labels_available: false,
          semantic_label_count: 0,
          labels_match_point_count: false,
          terrain_available: false,
          object_detection_available: false,
          object_instance_count: 0,
          adaptive_map_available: false,
          adaptive_grid_cell_count: 0,
        },
        message: 'Mock preview mode: Live backend required.',
      })
    );
  }
}
