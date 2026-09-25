/**
 * Type definitions for Real-Time Performance & Validation Metrics.
 */

export interface PipelineStageLatencies {
  lidar_preprocessing_ms: number;
  salsanext_inference_ms: number;
  terrain_analysis_ms: number;
  object_detection_tracking_ms: number;
  adaptive_grid_ms: number;
  total_latency_ms: number;
  preprocessing_ms?: number;
  inference_ms?: number;
  terrain_ms?: number;
  clustering_ms?: number;
  grid_ms?: number;
  total_ms?: number;
}

export interface SpatialBounds {
  min_x: number;
  max_x: number;
  min_y: number;
  max_y: number;
  min_z: number;
  max_z: number;
  span_x: number;
  span_y: number;
  span_z: number;
  volume_m3: number;
}

export interface AdaptiveGridResourceMetrics {
  input_point_count: number;
  fine_cells_count: number;
  medium_cells_count: number;
  coarse_cells_count: number;
  total_cells_count: number;
  adaptive_grid_bytes: number;
  serialized_map_bytes: number;
  uniform_baseline_bytes: number;
  uniform_voxel_size_m: number;
  uniform_bytes_per_voxel: number;
  uniform_total_voxels: number;
  memory_reduction_percentage: number;
  spatial_bounds: SpatialBounds;
  methodology_note: string;
}

export interface ClassAccuracyMetric {
  class_name: string;
  class_id?: number | null;
  precision: number;
  recall: number;
  f1_score: number;
  iou: number;
  support_points: number;
}

export interface DistanceBinMetric {
  bin_range: string;
  min_dist_m: number;
  max_dist_m: number;
  point_count: number;
  accuracy: number;
  mean_iou: number;
}

export interface AccuracyValidationMetrics {
  accuracy_available: boolean;
  reason_unavailable?: string | null;
  overall_accuracy?: number | null;
  mean_iou?: number | null;
  evaluated_points_count?: number | null;
  per_class_metrics: ClassAccuracyMetric[];
  object_class_summaries: Record<string, ClassAccuracyMetric>;
  distance_bins: DistanceBinMetric[];
}

export interface FramePerformanceMetrics {
  frame_id: string;
  session_id?: string | null;
  timestamp: string;
  device_used: string;
  model_name: string;
  timings: PipelineStageLatencies;
  actual_fps: number;
  resource_metrics?: AdaptiveGridResourceMetrics | null;
  accuracy_metrics?: AccuracyValidationMetrics | null;
}

export interface SessionPerformanceMetrics {
  session_id: string;
  sequence_name: string;
  total_frames_processed: number;
  average_timings: PipelineStageLatencies;
  average_fps: number;
  average_memory_reduction_percentage: number;
  total_points_processed: number;
  total_cells_generated: number;
  session_accuracy: AccuracyValidationMetrics;
  recent_frame_metrics: FramePerformanceMetrics[];
  timestamp: string;
}
