/**
 * TypeScript definitions for LiDAR Sequence Replay and WebSocket Streaming
 */

import { TerrainAnalysisResponse } from './terrain';
import { ObjectDetectionResponse, TrackingUpdateResponse } from './objects';
import { MapUpdateResponse } from './map';
import { FramePerformanceMetrics } from './metrics';

export type ReplayState = 'ready' | 'playing' | 'paused' | 'completed' | 'failed' | 'no_predictions';

export type ReplayDataMode = 'precomputed_labels' | 'live_inference' | 'scans_only';

export type SemanticSource =
  | 'LIVE GEOMETRIC'
  | 'LIVE GEOMETRIC — FAST-FRNET PROCESSING'
  | 'LIVE FAST-FRNET'
  | 'LIVE GEOMETRIC \u2014 SALSANEXT PROCESSING'
  | 'LIVE SALSANEXT'
  | 'GROUND TRUTH'
  | 'PRECOMPUTED';

export interface ReplayUploadResponse {
  session_id: string;
  sequence_name: string;
  total_frames: number;
  has_predictions: boolean;
  data_mode: ReplayDataMode;
  semantic_source?: SemanticSource;
  playback_mode?: string;
  frame_filenames: string[];
  status: ReplayState;
  message: string;
}

export interface ReplaySessionStatus {
  session_id: string;
  sequence_name: string;
  state: ReplayState;
  current_frame_index: number;
  total_frames: number;
  fps: number;
  playback_mode?: string;
  data_mode: ReplayDataMode;
  semantic_source?: SemanticSource;
  coordinate_mode: string;
  has_predictions: boolean;
  error_message?: string;
}

export interface ReplayPointSample {
  x: number;
  y: number;
  z: number;
  intensity: number;
  semantic_class: string;
  project_category: string;
}

export interface ReplayFrameStreamPayload {
  session_id: string;
  sequence_name: string;
  frame_index: number;
  total_frames: number;
  frame_filename: string;
  timestamp?: string;
  point_count: number;
  points_sample: ReplayPointSample[];
  category_distribution: Record<string, number>;
  terrain_result?: TerrainAnalysisResponse;
  objects_result?: ObjectDetectionResponse;
  tracking_result?: TrackingUpdateResponse;
  map_result?: MapUpdateResponse;
  coordinate_mode: string;
  map_mode: string;
  data_mode: ReplayDataMode;
  semantic_source?: SemanticSource;
  state: ReplayState;
  processing_time_ms: number;
  performance?: FramePerformanceMetrics | null;
  is_available?: boolean;
}

export interface ReplayStartRequest {
  fps?: number;
  playback_mode?: string;
}

export interface ReplaySeekRequest {
  frame_index: number;
}

export interface PrecomputeProgressStatus {
  session_id: string;
  sequence_name: string;
  processed_frames: number;
  total_frames: number;
  percent_complete: number;
  is_complete: boolean;
  is_running: boolean;
  elapsed_seconds?: number;
  eta_seconds?: number;
  current_stage?: string;
  failed_count?: number;
  error_message?: string;
}

