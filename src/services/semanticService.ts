/**
 * Semantic Deep Learning & Annotation Service
 * Manages SemanticKITTI .label uploads, SalsaNext predictions import, and live neural inference.
 */

import { apiClient, ApiResponse } from './api';
import { SemanticModelSpec } from '../types/semantic';
import {
  SemanticLabelUploadResponse,
  PredictionImportPayload,
  SemanticFrameResponse,
  InferenceStatusResponse,
  InferenceJobResponse,
  InferenceResultsResponse,
  LiveUploadInferenceResponse,
} from '../types/objects';
import { SUPPORTED_MODELS } from '../mocks/mockSettings';

export class SemanticService {
  /**
   * Universal Live Upload & Perception Pipeline:
   * Always executes Fast-FRNet live on uploaded point cloud (.bin, .pcd, .xyz, .ply)
   * with automatic scene analysis model selection and zero precomputed reuse.
   */
  public async liveUpload(
    file: File,
    requestId?: string,
    forceModel?: string
  ): Promise<ApiResponse<LiveUploadInferenceResponse>> {
    const formData = new FormData();
    formData.append('file', file);
    if (requestId) {
      formData.append('request_id', requestId);
    }
    if (forceModel) {
      formData.append('force_model', forceModel);
    }

    // Never pass mock fallback — live uploads must strictly report truthful live execution
    return apiClient.request<LiveUploadInferenceResponse>(
      '/inference/live-upload',
      {
        method: 'POST',
        body: formData,
      }
    );
  }

  /**
   * Check Fast-FRNet inference model and hardware device status
   */
  public async getInferenceStatus(modelType: string = 'rellis'): Promise<ApiResponse<InferenceStatusResponse>> {
    return apiClient.request<InferenceStatusResponse>(
      `/inference/status?model_type=${encodeURIComponent(modelType)}`,
      { method: 'GET' },
      () => ({
        status: 'not_connected',
        model_name: 'Fast-FRNet',
        active_model: modelType,
        device: 'cpu',
        device_name: 'CPU Mode (Mock Fallback)',
        checkpoint_path: `backend/models/best_frnet_${modelType}.pth`,
        checkpoint_exists: false,
        arch_config_exists: true,
        data_config_exists: true,
        num_classes: 20,
        cuda_available: false,
        message: 'Mock Mode: Backend inference service not connected.',
      })
    );
  }

  /**
   * Run Fast-FRNet semantic segmentation on a .bin point cloud file or existing frame_id
   */
  public async runInference(
    file?: File,
    frameId?: string,
    modelType: string = 'rellis'
  ): Promise<ApiResponse<InferenceJobResponse>> {
    const formData = new FormData();
    if (file) {
      formData.append('file', file);
    }
    if (frameId) {
      formData.append('frame_id', frameId);
    }
    formData.append('model_type', modelType);

    return apiClient.request<InferenceJobResponse>(
      '/inference/segment',
      {
        method: 'POST',
        body: formData,
      },
      () => ({
        job_id: `mock_job_${frameId || '001'}`,
        frame_id: frameId || 'FRAME_0001 (Mock)',
        status: 'complete',
        point_count: 5000,
        device_used: 'cpu',
        created_at: new Date().toISOString(),
        message: `Mock inference simulated (${modelType.toUpperCase()}).`,
      })
    );
  }

  /**
   * Fetch point-wise inference results for a given job or frame
   */
  public async getInferenceResults(
    jobId: string
  ): Promise<ApiResponse<InferenceResultsResponse>> {
    return apiClient.request<InferenceResultsResponse>(
      `/inference/results/${encodeURIComponent(jobId)}`,
      { method: 'GET' },
      () => ({
        job_id: jobId,
        frame_id: jobId,
        status: 'complete',
        point_count: 5000,
        class_counts: { road: 2800, car: 600, vegetation: 800, building: 800 },
        project_category_counts: { drivable: 2800, dynamic_object: 600, vegetation: 800, infrastructure: 800 },
        sample_predictions: [],
        device_used: 'cpu',
        created_at: new Date().toISOString(),
      })
    );
  }

  /**
   * Upload SemanticKITTI-compatible .label file for a LiDAR frame
   */
  public async uploadLabelsBin(
    file: File,
    frameId: string
  ): Promise<ApiResponse<SemanticLabelUploadResponse>> {
    return apiClient.uploadFile<SemanticLabelUploadResponse>(
      '/semantic/upload-labels',
      file,
      'file',
      { frame_id: frameId }
    );
  }

  /**
   * Import precomputed predictions (e.g. SalsaNext inference from Kaggle/GPU)
   */
  public async importPredictions(
    payload: PredictionImportPayload
  ): Promise<ApiResponse<SemanticLabelUploadResponse>> {
    return apiClient.post<SemanticLabelUploadResponse>(
      '/semantic/import-predictions',
      payload
    );
  }

  /**
   * Fetch semantic annotations and labeled point previews for a frame
   */
  public async getSemanticFrame(
    frameId: string
  ): Promise<ApiResponse<SemanticFrameResponse>> {
    return apiClient.get<SemanticFrameResponse>(
      `/semantic/${encodeURIComponent(frameId)}?sample_points=6000`,
      () => ({
        frame_id: frameId,
        point_count: 5000,
        class_counts: {
          road: 2400,
          car: 650,
          vegetation: 800,
          building: 750,
          pole: 120,
          person: 45,
          sidewalk: 235,
        },
        project_category_counts: {
          drivable: 2400,
          dynamic_object: 695,
          vegetation: 800,
          infrastructure: 750,
          static_obstacle: 120,
          non_drivable: 235,
        },
        sample_labeled_points: [],
        model_provider_status: 'external_predictions_loaded',
        created_at: new Date().toISOString(),
      })
    );
  }

  /**
   * Get active model specification and metadata
   */
  public async getModelSpec(modelName?: string): Promise<ApiResponse<SemanticModelSpec>> {
    const activeName = modelName || SUPPORTED_MODELS[0].modelName;
    const spec = SUPPORTED_MODELS.find(m => m.modelName === activeName) || SUPPORTED_MODELS[0];

    return apiClient.request<SemanticModelSpec>(
      `/models/status`,
      {},
      () => spec
    );
  }

  /**
   * List all available model architectures
   */
  public getSupportedModels(): SemanticModelSpec[] {
    return SUPPORTED_MODELS;
  }
}

export const semanticService = new SemanticService();
