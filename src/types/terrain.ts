/**
 * TypeScript definitions for Backend Terrain Analysis & LiDAR Ingestion
 */

export type ProcessStatus = 'queued' | 'processing' | 'complete' | 'failed' | 'not_available';

export type ObservationState = 'directly_observed' | 'partially_occluded' | 'sensor_fringe' | 'unknown';

export type SlopeCategory = 'flat' | 'gentle' | 'moderate' | 'steep' | 'extreme';

export type RoughnessCategory = 'smooth' | 'low_roughness' | 'moderate_roughness' | 'rough' | 'highly_irregular';

export type StepCategory = 'none' | 'curb' | 'step_barrier' | 'high_obstacle';

export type ElevationSummary =
  | 'ground_level'
  | 'elevated_surface'
  | 'depression_slope'
  | 'overhead_clearance'
  | 'variable_height';

export type TerrainInterpretation =
  | 'paved_flat'
  | 'sloped_road'
  | 'rough_unpaved'
  | 'curb_boundary'
  | 'obstacle_barrier'
  | 'sparse_foliage_or_overhang'
  | 'unknown';

export type DrivabilityState =
  | 'drivable_candidate'
  | 'non_drivable_candidate'
  | 'caution_irregular'
  | 'obstacle_hazard'
  | 'unknown';

export type ResolutionZone = 'near' | 'mid' | 'far';

export interface BoundingBox3D {
  min_x: number;
  max_x: number;
  min_y: number;
  max_y: number;
  min_z: number;
  max_z: number;
}

export interface PreprocessingConfig {
  min_range?: number;
  max_range?: number;
  z_min?: number;
  z_max?: number;
  remove_nan_inf?: boolean;
}

export interface PreprocessingReport {
  raw_points_count: number;
  valid_points_count: number;
  nan_inf_removed: number;
  out_of_bounds_removed: number;
  applied_config: PreprocessingConfig;
  bounds?: BoundingBox3D;
}

export interface TerrainCell {
  id: string;
  grid_x: number;
  grid_y: number;
  world_x: number;
  world_y: number;
  size_m: number;
  point_count: number;

  // Continuous geometric properties
  min_z: number;
  max_z: number;
  mean_z: number;
  elevation_range: number;
  slope_deg: number;
  roughness_m: number;
  step_height_m: number;
  has_step: boolean;

  // Qualitative interpretations
  slope_category: SlopeCategory;
  roughness_category: RoughnessCategory;
  step_category: StepCategory;
  observation_state: ObservationState;
  elevation_summary: ElevationSummary;
  terrain_interpretation: TerrainInterpretation;
  drivability_state: DrivabilityState;
  zone: ResolutionZone;
}

export interface TerrainAnalysisSummary {
  total_cells: number;
  drivable_cells: number;
  non_drivable_cells: number;
  caution_cells: number;
  hazard_cells: number;
  unknown_cells: number;
  grid_resolution_m: number;
  bounds: BoundingBox3D;
}

export interface TerrainAnalysisRequest {
  frame_id: string;
  grid_resolution_m?: number;
  preprocessing_config?: PreprocessingConfig;
}

export interface TerrainAnalysisResponse {
  frame_id: string;
  status: ProcessStatus;
  preprocessing_report?: PreprocessingReport;
  summary?: TerrainAnalysisSummary;
  cells: TerrainCell[];
  created_at: string;
  error_message?: string;
}

export interface FrameUploadResponse {
  frame_id: string;
  point_count: number;
  file_size_bytes: number;
  status: ProcessStatus;
  timestamp: string;
  bounds: BoundingBox3D;
  message: string;
}

export interface HealthResponse {
  status: string;
  project_name: string;
  version: string;
  timestamp: string;
  storage_ready: boolean;
  features: string[];
}
