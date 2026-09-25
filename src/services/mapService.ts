/**
 * 2.5D Adaptive Elevation Map Service
 * Manages variable-resolution foveated map cells, layer filtering, and real backend cell transformations.
 */

import { apiClient, ApiResponse } from './api';
import {
  GridCell,
  AdaptiveGridCellBackend,
  MapMetadata,
  MapUpdateRequest,
  MapUpdateResponse,
  MapExportResponse,
  ResolutionPolicyConfigFrontend,
} from '../types/map';
import { TerrainCell } from '../types/terrain';
import { SemanticClassId } from '../types/semantic';
import { generateMockGridCells, defaultResolutionPolicy } from '../mocks/mockMapData';

export class MapService {
  private cachedCells: GridCell[] | null = null;

  /**
   * Convert backend AdaptiveGridCellBackend array into frontend GridCell format for 2.5D canvas rendering
   */
  public convertAdaptiveCellsToGridCells(adaptiveCells: (AdaptiveGridCellBackend | any)[]): GridCell[] {
    return adaptiveCells.map((ac: any) => {
      const cellKey = ac.cell_key || ac.id || 'unknown';
      const domCat = ac.dominant_category || ac.project_category || 'unknown';
      const domCls = ac.dominant_semantic_class || ac.semantic_class || 'unlabeled';
      const isDynamic = Boolean(ac.is_dynamic_obstacle || domCat === 'dynamic_object');
      const isStatic = Boolean(ac.is_static_obstacle || domCat === 'static_obstacle');
      const travState = ac.traversability_state || 'drivable';
      const elevMax = ac.elevation_max ?? 0;
      const elevMin = ac.elevation_min ?? 0;
      const elevVar = ac.elevation_variation ?? 0;
      const roughness = ac.roughness_summary ?? 0;
      const slope = ac.slope_summary ?? 0;
      const obsState = ac.observation_state || 'observed';
      const level = ac.level || 'fine';

      // Map category to semantic visual class
      let semType: SemanticClassId = 'drivable';
      let semLabel = domCls || 'Drivable Road';

      if (isDynamic) {
        semType = 'dynamicObstacle';
        semLabel = `Dynamic: ${domCls}`;
      } else if (isStatic) {
        semType = 'staticObstacle';
        semLabel = `Hazard: ${domCls}`;
      } else if (domCat === 'vegetation') {
        semType = 'vegetation';
        semLabel = `Vegetation: ${domCls}`;
      } else if (domCat === 'infrastructure' || domCat === 'building') {
        semType = 'infrastructure';
        semLabel = `Structure: ${domCls}`;
      } else if (domCat === 'non_drivable') {
        semType = 'nondrivable';
        semLabel = `Non-Drivable: ${domCls}`;
      } else if (domCat === 'drivable') {
        semType = 'drivable';
        semLabel = `Drivable: ${domCls}`;
      } else {
        semType = 'unknown';
        semLabel = 'Unknown Region';
      }

      // Map traversability string
      let traversability: GridCell['traversability'] = 'Drivable / Safe';
      if (travState === 'collision_hazard') traversability = 'Collision Hazard';
      else if (travState === 'non_traversable') traversability = 'Non-Traversable';
      else if (travState === 'caution_irregular') traversability = 'Caution / Irregular';
      else if (travState === 'uncertain') traversability = 'Uncertain';

      // Map elevation summary
      let elevation: GridCell['elevation'] = 'Ground Level';
      if (elevMax > 0.5) elevation = 'Elevated Surface';
      else if (elevMin < -2.2) elevation = 'Depression / Slope';
      else if (elevVar > 0.4) elevation = 'Variable Height';

      // Map surface condition
      let surfaceCondition: GridCell['surfaceCondition'] = 'Paved / Smooth';
      if (roughness > 0.15) surfaceCondition = 'Rough / Unpaved';
      else if (slope > 10.0) surfaceCondition = 'Curved / Sloped';
      else if (isStatic) surfaceCondition = 'Obstructed';

      // Map observation state
      let observationState: GridCell['observationState'] = 'Directly Observed';
      if (obsState === 'partially_occluded') observationState = 'Partially Occluded';
      else if (obsState === 'out_of_range' || obsState === 'sensor_fringe') observationState = 'Sensor Fringe';
      else if (obsState === 'unknown') observationState = 'Prior Inferred';

      // Resolution Zone label
      let resolutionZone: GridCell['resolutionZone'] = 'Fine Detail Zone (Near-Field)';
      let zone: 'near' | 'mid' | 'far' = 'near';
      if (level === 'medium') {
        resolutionZone = 'Medium Resolution Zone (Mid-Field)';
        zone = 'mid';
      } else if (level === 'coarse') {
        resolutionZone = 'Distant Context Zone (Far-Field)';
        zone = 'far';
      }

      return {
        id: cellKey,
        gridX: ac.grid_x ?? 0,
        gridY: ac.grid_y ?? 0,
        worldX: ac.world_x ?? 0,
        worldY: ac.world_y ?? 0,
        size: ac.size_m ?? 0.05,
        zone,
        semanticType: semType,
        semanticLabel: semLabel,
        traversability,
        elevation,
        surfaceCondition,
        resolutionZone,
        observationState,
        hasElevationGradient: slope > 5.0,
        isObstacle: isStatic || isDynamic,
        isDynamic: isDynamic,
        backendAdaptiveCell: ac,
      };
    });
  }

  /**
   * Convert backend TerrainCell array into frontend GridCell format for 2.5D canvas rendering
   */
  public convertTerrainCellsToGridCells(terrainCells: TerrainCell[]): GridCell[] {
    return terrainCells.map((tc) => {
      let semType: SemanticClassId = 'drivable';
      let semLabel = 'Drivable Surface (Geometric)';

      if (tc.drivability_state === 'obstacle_hazard' || tc.terrain_interpretation === 'obstacle_barrier') {
        semType = 'staticObstacle';
        semLabel = 'Obstacle Barrier (Geometric)';
      } else if (tc.step_category === 'curb' || tc.terrain_interpretation === 'curb_boundary') {
        semType = 'nondrivable';
        semLabel = 'Curb Boundary (Geometric)';
      } else if (tc.terrain_interpretation === 'sparse_foliage_or_overhang') {
        semType = 'vegetation';
        semLabel = 'Overhanging Structure (Geometric)';
      } else if (tc.drivability_state === 'non_drivable_candidate') {
        semType = 'nondrivable';
        semLabel = 'Non-Drivable Surface (Geometric)';
      } else if (tc.observation_state === 'unknown') {
        semType = 'unknown';
        semLabel = 'Unobserved Region';
      }

      let traversability: GridCell['traversability'] = 'Drivable / Safe';
      if (tc.drivability_state === 'obstacle_hazard') traversability = 'Collision Hazard';
      else if (tc.drivability_state === 'non_drivable_candidate') traversability = 'Non-Traversable';
      else if (tc.drivability_state === 'caution_irregular') traversability = 'Caution / Irregular';
      else if (tc.drivability_state === 'unknown') traversability = 'Uncertain';

      let elevation: GridCell['elevation'] = 'Ground Level';
      if (tc.elevation_summary === 'elevated_surface') elevation = 'Elevated Surface';
      else if (tc.elevation_summary === 'depression_slope') elevation = 'Depression / Slope';
      else if (tc.elevation_summary === 'overhead_clearance') elevation = 'Overhead Clearance';
      else if (tc.elevation_summary === 'variable_height') elevation = 'Variable Height';

      let surfaceCondition: GridCell['surfaceCondition'] = 'Paved / Smooth';
      if (tc.roughness_category === 'rough' || tc.roughness_category === 'highly_irregular') {
        surfaceCondition = 'Rough / Unpaved';
      } else if (tc.slope_category === 'moderate' || tc.slope_category === 'steep' || tc.slope_category === 'extreme') {
        surfaceCondition = 'Curved / Sloped';
      } else if (tc.has_step) {
        surfaceCondition = 'Obstructed';
      }

      let observationState: GridCell['observationState'] = 'Directly Observed';
      if (tc.observation_state === 'partially_occluded') observationState = 'Partially Occluded';
      else if (tc.observation_state === 'sensor_fringe') observationState = 'Sensor Fringe';
      else if (tc.observation_state === 'unknown') observationState = 'Sensor Fringe';

      let resolutionZone: GridCell['resolutionZone'] = 'Fine Detail Zone (Near-Field)';
      if (tc.zone === 'mid') resolutionZone = 'Medium Resolution Zone (Mid-Field)';
      else if (tc.zone === 'far') resolutionZone = 'Distant Context Zone (Far-Field)';

      return {
        id: tc.id,
        gridX: tc.grid_x,
        gridY: tc.grid_y,
        worldX: tc.world_x,
        worldY: tc.world_y,
        size: tc.size_m,
        zone: tc.zone,
        semanticType: semType,
        semanticLabel: semLabel,
        traversability,
        elevation,
        surfaceCondition,
        resolutionZone,
        observationState,
        hasElevationGradient: tc.slope_deg > 3.0,
        isObstacle: tc.has_step || tc.drivability_state === 'obstacle_hazard',
        isDynamic: false,
        backendTerrain: tc,
      };
    });
  }

  /**
   * Update or create a 2.5D map using frame LiDAR & semantic data
   */
  public async updateMap(mapId: string, req: MapUpdateRequest): Promise<ApiResponse<MapUpdateResponse>> {
    return apiClient.request<MapUpdateResponse>(
      `/maps/${mapId}/update`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
      },
      () => ({
        map_id: mapId,
        mode: req.ego_pose ? 'global_fusion' : 'local_only',
        status: 'complete',
        updated_cell_count: 120,
        metadata: {
          map_id: mapId,
          mode: req.ego_pose ? 'global_fusion' : 'local_only',
          frame_count: 1,
          total_cells: 120,
          fine_cells_count: 60,
          medium_cells_count: 40,
          coarse_cells_count: 20,
          bounds: { min_x: -20, max_x: 40, min_y: -20, max_y: 20, min_z: -3, max_z: 5 },
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
        cells_sample: [],
        message: 'Mock map updated',
      })
    );
  }

  /**
   * Get metadata for a map instance
   */
  public async getMapMetadata(mapId: string): Promise<ApiResponse<MapMetadata>> {
    return apiClient.request<MapMetadata>(
      `/maps/${mapId}`,
      { method: 'GET' },
      () => ({
        map_id: mapId,
        mode: 'local_only',
        frame_count: 1,
        total_cells: 120,
        fine_cells_count: 60,
        medium_cells_count: 40,
        coarse_cells_count: 20,
        bounds: { min_x: -20, max_x: 40, min_y: -20, max_y: 20, min_z: -3, max_z: 5 },
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
    );
  }

  /**
   * Query sparse cells from backend map
   */
  public async getMapCells(
    mapId: string,
    params?: { level?: string; min_x?: number; max_x?: number; min_y?: number; max_y?: number; limit?: number }
  ): Promise<ApiResponse<AdaptiveGridCellBackend[]>> {
    const query = new URLSearchParams();
    if (params?.level) query.set('level', params.level);
    if (params?.min_x !== undefined) query.set('min_x', params.min_x.toString());
    if (params?.max_x !== undefined) query.set('max_x', params.max_x.toString());
    if (params?.min_y !== undefined) query.set('min_y', params.min_y.toString());
    if (params?.max_y !== undefined) query.set('max_y', params.max_y.toString());
    if (params?.limit !== undefined) query.set('limit', params.limit.toString());

    const qs = query.toString();
    const endpoint = `/maps/${mapId}/cells${qs ? `?${qs}` : ''}`;

    return apiClient.request<AdaptiveGridCellBackend[]>(
      endpoint,
      { method: 'GET' },
      () => []
    );
  }

  /**
   * Get single cell details by hierarchical key
   */
  public async getCellDetails(mapId: string, cellKey: string): Promise<ApiResponse<AdaptiveGridCellBackend | null>> {
    return apiClient.request<AdaptiveGridCellBackend | null>(
      `/maps/${mapId}/cells/${cellKey}`,
      { method: 'GET' },
      () => null
    );
  }

  /**
   * Reset a map instance
   */
  public async resetMap(mapId: string): Promise<ApiResponse<{ map_id: string; status: string; message: string }>> {
    return apiClient.request<{ map_id: string; status: string; message: string }>(
      `/maps/${mapId}/reset`,
      { method: 'POST' },
      () => ({ map_id: mapId, status: 'reset_complete', message: 'Mock map reset' })
    );
  }

  /**
   * Export complete map JSON
   */
  public async exportMap(mapId: string): Promise<ApiResponse<MapExportResponse>> {
    return apiClient.request<MapExportResponse>(
      `/maps/${mapId}/export`,
      { method: 'GET' },
      () => ({
        map_id: mapId,
        mode: 'local_only',
        metadata: {
          map_id: mapId,
          mode: 'local_only',
          frame_count: 1,
          total_cells: 0,
          fine_cells_count: 0,
          medium_cells_count: 0,
          coarse_cells_count: 0,
          bounds: { min_x: 0, max_x: 0, min_y: 0, max_y: 0, min_z: 0, max_z: 0 },
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
        cells: [],
        policy_snapshot: {
          near_zone_max_distance_m: 12.0,
          mid_zone_max_distance_m: 28.0,
          far_zone_max_distance_m: 80.0,
          fine_resolution_m: 0.5,
          medium_resolution_m: 1.0,
          coarse_resolution_m: 2.0,
          safety_priority: true,
          terrain_complexity_override: true,
          obstacle_override: true,
          slope_refinement_threshold_deg: 10.0,
          roughness_refinement_threshold_m: 0.10,
          step_refinement_threshold_m: 0.10,
          obstacle_height_span_threshold_m: 0.30,
          default_cells_query_limit: 5000,
        },
      })
    );
  }

  /**
   * Get fallback mock 2.5D local map grid cells
   */
  public async getLocalMapCells(): Promise<ApiResponse<GridCell[]>> {
    if (!this.cachedCells) {
      this.cachedCells = generateMockGridCells();
    }

    return apiClient.request<GridCell[]>(
      `/map/local-cells`,
      {},
      () => this.cachedCells || generateMockGridCells()
    );
  }

  /**
   * Get variable-resolution policy configuration
   */
  public async getResolutionPolicy(): Promise<ApiResponse<ResolutionPolicyConfigFrontend>> {
    return apiClient.request<ResolutionPolicyConfigFrontend>(
      `/map/resolution-policy`,
      {},
      () => defaultResolutionPolicy
    );
  }
}

export const mapService = new MapService();
