/**
 * 2.5D Adaptive Elevation Map & Cell Inspector Types
 */

import { SemanticClassId } from './semantic';
import { TerrainCell } from './terrain';
import { BoundingBox3D } from './lidar';

export type ResolutionZoneType = 'near' | 'mid' | 'far';
export type ResolutionLevel = 'fine' | 'medium' | 'coarse';
export type MapMode = 'local_only' | 'global_fusion';
export type CellObservationState = 'observed' | 'partially_occluded' | 'out_of_range' | 'unknown';
export type TraversabilityState = 'drivable' | 'non_traversable' | 'caution_irregular' | 'collision_hazard' | 'uncertain';
export type AmbiguityState = 'unambiguous' | 'mixed_classes' | 'sparse_data' | 'high_gradient';

export interface ElevationLayer {
  layer_index: number;
  layer_type: string;
  elevation_min: number;
  elevation_mean: number;
  elevation_max: number;
  elevation_range: number;
  point_count: number;
  dominant_category: string;
  traversability_state: TraversabilityState;
}

// Backend Contract for 2.5D Adaptive Grid Cell
export interface AdaptiveGridCellBackend {
  cell_key: string;
  level: ResolutionLevel;
  size_m: number;
  grid_x: number;
  grid_y: number;
  world_x: number;
  world_y: number;
  bounds: [number, number, number, number]; // [min_x, max_x, min_y, max_y]
  observation_state: CellObservationState;
  
  // Semantic Content
  semantic_histogram: Record<string, number>;
  dominant_semantic_class: string;
  dominant_category: string;
  
  // Geometric & Terrain Properties
  terrain_state: string;
  traversability_state: TraversabilityState;
  is_static_obstacle: boolean;
  is_dynamic_obstacle: boolean;
  
  elevation_min: number;
  elevation_mean: number;
  elevation_max: number;
  elevation_variation: number;
  roughness_summary: number;
  slope_summary: number;
  point_count: number;
  
  // Multi-Surface & Overhang Support
  has_overhang?: boolean;
  overhead_clearance_m?: number | null;
  layers?: ElevationLayer[];
  
  // Hierarchical Links & Metadata
  last_frame_id: string;
  last_timestamp?: string;
  ambiguity_state: AmbiguityState;
  parent_key?: string | null;
  child_keys: string[];
}

export interface MapMetadata {
  map_id: string;
  mode: MapMode;
  frame_count: number;
  total_cells: number;
  fine_cells_count: number;
  medium_cells_count: number;
  coarse_cells_count: number;
  bounds: BoundingBox3D;
  created_at: string;
  updated_at: string;
}

export interface GridPolicyConfig {
  near_zone_max_distance_m: number;
  mid_zone_max_distance_m: number;
  far_zone_max_distance_m: number;
  fine_resolution_m: number;
  medium_resolution_m: number;
  coarse_resolution_m: number;
  safety_priority: boolean;
  terrain_complexity_override: boolean;
  obstacle_override: boolean;
  slope_refinement_threshold_deg: number;
  roughness_refinement_threshold_m: number;
  step_refinement_threshold_m: number;
  obstacle_height_span_threshold_m: number;
  default_cells_query_limit: number;
}

export interface MapUpdateRequest {
  frame_id: string;
  ego_pose?: number[][] | number[];
  override_policy?: Partial<GridPolicyConfig>;
}

export interface MapUpdateResponse {
  map_id: string;
  mode: MapMode;
  status: string;
  updated_cell_count: number;
  metadata: MapMetadata;
  cells_sample: AdaptiveGridCellBackend[];
  message: string;
}

export interface MapExportResponse {
  map_id: string;
  mode: MapMode;
  metadata: MapMetadata;
  cells: AdaptiveGridCellBackend[];
  policy_snapshot: GridPolicyConfig;
}

export interface GridCell {
  id: string;
  gridX: number; // grid index
  gridY: number;
  worldX: number; // approximate relative coordinate for rendering
  worldY: number;
  size: number;   // cell physical dimension (e.g., fine, medium, coarse)
  zone: ResolutionZoneType;
  semanticType: SemanticClassId;
  semanticLabel: string;
  traversability: 'Drivable / Safe' | 'Non-Traversable' | 'Caution / Irregular' | 'Collision Hazard' | 'Uncertain';
  elevation: 'Ground Level' | 'Elevated Surface' | 'Depression / Slope' | 'Overhead Clearance' | 'Variable Height';
  surfaceCondition: 'Paved / Smooth' | 'Rough / Unpaved' | 'Curved / Sloped' | 'Obstructed' | 'Unclassified';
  resolutionZone: 'Fine Detail Zone (Near-Field)' | 'Medium Resolution Zone (Mid-Field)' | 'Distant Context Zone (Far-Field)';
  observationState: 'Directly Observed' | 'Partially Occluded' | 'Prior Inferred' | 'Sensor Fringe';
  hasElevationGradient?: boolean;
  isObstacle?: boolean;
  isDynamic?: boolean;

  // Real Backend Geometric Terrain Data
  backendTerrain?: TerrainCell;
  // Real Backend Adaptive 2.5D Cell
  backendAdaptiveCell?: AdaptiveGridCellBackend;
}

export interface MapLayersState {
  semanticClasses: boolean;
  elevation: boolean;
  traversability: boolean;
  staticObstacles: boolean;
  dynamicObstacles: boolean;
  unknownAreas: boolean;
}

export interface ResolutionPolicyConfig {
  nearFieldDetail: string;
  middleFieldDetail: string;
  farFieldDetail: string;
  safetyPriority: string;
  terrainComplexity: string;
  obstacleOverride: string;
}

export type ResolutionPolicyConfigFrontend = ResolutionPolicyConfig;

