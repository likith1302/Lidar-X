/**
 * TypeScript definitions for Semantic Objects, Instance Clustering, Tracking, and SalsaNext Inference
 */

import { BoundingBox3D, ProcessStatus, TerrainAnalysisResponse } from './terrain';
import { Point3D } from './lidar';
import { MapUpdateResponse } from './map';

export type ModelProviderStatus =
  | 'external_predictions_loaded'
  | 'inference_service_not_connected'
  | 'prediction_unavailable';

export type InferenceModelStatus =
  | 'model_available'
  | 'model_missing'
  | 'loading'
  | 'ready'
  | 'gpu_available'
  | 'cpu_mode'
  | 'failed'
  | 'not_connected';

export interface InferenceStatusResponse {
  status: InferenceModelStatus;
  model_name: string;
  active_model?: string;
  device: string;
  device_name: string;
  checkpoint_path: string;
  checkpoint_exists: boolean;
  arch_config_exists?: boolean;
  data_config_exists?: boolean;
  num_classes: number;
  cuda_available: boolean;
  message: string;
  rellis_checkpoint_path?: string;
  rellis_checkpoint_exists?: boolean;
  semantickitti_checkpoint_path?: string;
  semantickitti_checkpoint_exists?: boolean;
}

export interface PointPredictionSample {
  x: number;
  y: number;
  z: number;
  intensity: number;
  raw_label_id: number;
  semantic_class: string;
  project_category: string;
}

export interface InferenceJobResponse {
  job_id: string;
  frame_id: string;
  status: ProcessStatus;
  point_count: number;
  device_used: string;
  created_at: string;
  message: string;
}

export interface InferenceResultsResponse {
  job_id: string;
  frame_id: string;
  status: ProcessStatus;
  point_count: number;
  class_counts: Record<string, number>;
  project_category_counts: Record<string, number>;
  sample_predictions: PointPredictionSample[];
  prediction_artifact_path?: string;
  device_used: string;
  created_at: string;
}

export interface LiveUploadInferenceResponse {
  mode: string;
  source: string;
  inference_source: string;
  precomputed: boolean;
  model: string;
  model_type: string;
  detected_domain: string;
  confidence: number;
  selection_reason: string;
  request_id?: string;
  upload_id: string;
  frame_id: string;
  filename: string;
  file_size_bytes: number;
  content_hash: string;
  timestamp: string;
  point_count: number;
  device_used: string;
  bounds?: {
    min_x: number;
    max_x: number;
    min_y: number;
    max_y: number;
    min_z: number;
    max_z: number;
  } | Record<string, number>;
  sample_predictions: PointPredictionSample[];
  performance: {
    preprocessing_ms?: number;
    inference_ms?: number;
    postprocessing_ms?: number;
    clustering_ms?: number;
    terrain_ms?: number;
    grid_ms?: number;
    total_ms?: number;
    total_pipeline_ms?: number;
    fps?: number;
    device?: string;
    device_used?: string;
    precomputed?: boolean;
    points_per_second?: number;
    stages?: Record<string, number>;
    resource_metrics?: any;
  };
  semantic: {
    class_counts: Record<string, number>;
    project_category_counts: Record<string, number>;
    total_points?: number;
  };
  objects: ObjectDetectionResponse;
  terrain: TerrainAnalysisResponse;
  grid: MapUpdateResponse;
  message: string;
}

export type ProjectCategory =
  | 'drivable'
  | 'non_drivable'
  | 'static_obstacle'
  | 'dynamic_object'
  | 'vegetation'
  | 'infrastructure'
  | 'unknown';

export type ExtentCategory =
  | 'compact_actor'
  | 'small_vehicle'
  | 'large_vehicle'
  | 'slender_vertical'
  | 'broad_structure'
  | 'irregular_cluster';

export type ObservationStatus =
  | 'directly_observed'
  | 'partially_occluded'
  | 'sparse_cluster';

export type TrackLifecycleState =
  | 'new'
  | 'active'
  | 'temporarily_lost'
  | 'expired';

export type TrackingCoordinateMode =
  | 'local_frame'
  | 'world_frame';

export interface SemanticPoint3D {
  x: number;
  y: number;
  z: number;
  intensity: number;
  raw_label_id: number;
  semantic_class: string;
  project_category: string;
  instance_id: number;
}

export interface SemanticLabelUploadResponse {
  frame_id: string;
  label_count: number;
  class_distribution: Record<string, number>;
  category_distribution: Record<string, number>;
  status: ProcessStatus;
  message: string;
}

export interface PredictionImportPayload {
  frame_id: string;
  model_name?: string;
  labels: number[];
  metadata?: Record<string, any>;
}

export interface SemanticFrameResponse {
  frame_id: string;
  point_count: number;
  class_counts: Record<string, number>;
  project_category_counts: Record<string, number>;
  sample_labeled_points: SemanticPoint3D[];
  model_provider_status: ModelProviderStatus;
  created_at: string;
}

export interface OrientedBoundingBox3D {
  center_x?: number;
  center_y?: number;
  center_z?: number;
  length?: number;
  width?: number;
  height?: number;
  center?: [number, number, number];
  size?: [number, number, number];
  yaw: number;
}

export interface ObjectInstance {
  instance_id: string;
  semantic_category: ProjectCategory;
  object_type: string;
  centroid: [number, number, number];
  bounding_box: BoundingBox3D;
  dimensions: [number, number, number];
  extent_category: ExtentCategory;
  point_count: number;
  is_dynamic: boolean;
  source_frame_id: string;
  observation_status: ObservationStatus;
  sample_points: Point3D[];
  yaw?: number;
  oriented_bounding_box?: OrientedBoundingBox3D;
  confidence?: number;
  class_id?: number;
}

export interface ObjectDetectionRequest {
  frame_id: string;
  min_points_per_cluster?: number;
  clustering_radius_m?: number;
}

export interface ObjectDetectionResponse {
  frame_id: string;
  total_instances: number;
  dynamic_instances_count: number;
  static_instances_count: number;
  instances: ObjectInstance[];
  status: ProcessStatus;
  created_at: string;
}

export interface TrackedPosition {
  frame_id: string;
  timestamp: string;
  position: [number, number, number];
}

export interface TrackedObject {
  track_id: string;
  object_type: string;
  semantic_category: ProjectCategory;
  current_position: [number, number, number];
  estimated_velocity: [number, number, number];
  history: TrackedPosition[];
  lifecycle_state: TrackLifecycleState;
  age_frames: number;
  frames_since_seen: number;
  coordinate_mode: TrackingCoordinateMode;
  bounding_box: BoundingBox3D;
  associated_instance_id?: string;
  yaw?: number;
  oriented_bounding_box?: OrientedBoundingBox3D;
  confidence?: number;
}

export interface TrackingUpdateRequest {
  frame_id: string;
  timestamp?: string;
  ego_pose?: number[];
  max_association_distance_m?: number;
}

export interface TrackingUpdateResponse {
  frame_id: string;
  coordinate_mode: TrackingCoordinateMode;
  active_tracks_count: number;
  new_tracks_count: number;
  lost_tracks_count: number;
  tracks: TrackedObject[];
  timestamp: string;
}

