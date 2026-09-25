/**
 * Pipeline gate and sequential processing types.
 */

export interface PipelineStatusResponse {
  frame_id: string;
  frame_exists: boolean;
  point_cloud_available: boolean;
  point_count: number;
  semantic_labels_available: boolean;
  semantic_label_count: number;
  labels_match_point_count: boolean;
  semantic_source?: string | null;
  inference_state?: string | null;
  terrain_available: boolean;
  object_detection_available: boolean;
  object_instance_count: number;
  adaptive_map_available: boolean;
  adaptive_grid_cell_count: number;
  map_mode?: string | null;
  last_error?: string | null;
}

export interface StageExecutionResult {
  stage: string;
  status: string;
  message: string;
  item_count: number;
  details?: Record<string, any> | null;
}

export interface PipelineProcessResponse {
  frame_id: string;
  success: boolean;
  status: string;
  executed_stages: StageExecutionResult[];
  pipeline_status: PipelineStatusResponse;
  message: string;
}

export type PipelineStageStatus = 'waiting' | 'processing' | 'complete' | 'failed';

export interface PipelineStages {
  upload: PipelineStageStatus;
  rawRender: PipelineStageStatus;
  salsanext: PipelineStageStatus;
  terrain: PipelineStageStatus;
  objects: PipelineStageStatus;
  adaptiveGrid: PipelineStageStatus;
}
