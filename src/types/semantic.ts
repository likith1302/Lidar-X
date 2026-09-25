/**
 * Semantic Perception and Deep Learning Types
 */

export type SemanticClassId =
  | 'drivable'
  | 'nondrivable'
  | 'staticObstacle'
  | 'dynamicObstacle'
  | 'vegetation'
  | 'infrastructure'
  | 'unknown';

export interface SemanticClassDefinition {
  id: SemanticClassId;
  name: string;
  color: string;
  description: string;
  navRole: string;
}

export const SEMANTIC_CLASSES: Record<SemanticClassId, SemanticClassDefinition> = {
  drivable: {
    id: 'drivable',
    name: 'Drivable Terrain',
    color: '#00E5FF', // Cyan / Electric blue
    description: 'Paved roadway, designated lane corridors, and flat traversable surfaces',
    navRole: 'Primary navigation path corridor',
  },
  nondrivable: {
    id: 'nondrivable',
    name: 'Non-Drivable Terrain',
    color: '#64748B', // Slate
    description: 'Shoulders, steep slopes, curbs, sidewalk surfaces, and terrain barriers',
    navRole: 'Boundary and exclusion zone',
  },
  staticObstacle: {
    id: 'staticObstacle',
    name: 'Static Obstacles',
    color: '#F59E0B', // Amber / Orange
    description: 'Stationary vehicles, barriers, guardrails, poles, and construction markers',
    navRole: 'Static spatial collision hazard',
  },
  dynamicObstacle: {
    id: 'dynamicObstacle',
    name: 'Dynamic Objects',
    color: '#EF4444', // Red / Crimson
    description: 'Moving vehicles, pedestrians, cyclists, and transient road occupants',
    navRole: 'Active safety trajectory critical',
  },
  vegetation: {
    id: 'vegetation',
    name: 'Vegetation',
    color: '#10B981', // Emerald
    description: 'Trees, shrubs, grass verges, and roadside foliage with variable density',
    navRole: 'Overhanging and roadside boundary',
  },
  infrastructure: {
    id: 'infrastructure',
    name: 'Infrastructure',
    color: '#8B5CF6', // Purple / Violet
    description: 'Building facades, overpasses, gantry structures, and street architecture',
    navRole: 'Permanent geometric environment',
  },
  unknown: {
    id: 'unknown',
    name: 'Unknown / Occluded',
    color: '#475569', // Muted slate
    description: 'Unclassified points, LiDAR sensor dropouts, and occluded shadow sectors',
    navRole: 'Uncertainty exploration requirement',
  },
};

export interface SemanticModelSpec {
  modelName: string;
  framework: string;
  inputType: string;
  outputType: string;
  inferenceEndpoint: string;
  modelStatus: string;
  architectureNotes: string;
}
