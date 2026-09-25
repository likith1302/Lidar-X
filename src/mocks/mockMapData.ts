/**
 * Mock 2.5D Adaptive Grid Data
 * Foveated local map representation with non-uniform resolution zones and descriptive attributes.
 */

import { GridCell, ResolutionPolicyConfig } from '../types/map';

export function generateMockGridCells(): GridCell[] {
  const cells: GridCell[] = [];

  // Define concentric zones:
  // Near-field: |x| <= 10, |y| <= 10 -> Fine cells (step 1.25)
  // Mid-field:  10 < |x| <= 22, 10 < |y| <= 22 -> Medium cells (step 2.5)
  // Far-field:  22 < |x| <= 36, 22 < |y| <= 36 -> Coarse cells (step 4.5)

  // We generate a cohesive spatial grid:
  // Center is vehicle at (0, 0)
  // Road runs longitudinally along X axis from X: -30 to +35, width Y: -5 to +5

  // 1. NEAR-FIELD ZONE (Fine Cells, high detail near ego vehicle)
  for (let x = -10; x <= 12; x += 1.6) {
    for (let y = -10; y <= 10; y += 1.6) {
      const cellId = `cell_near_${x.toFixed(1)}_${y.toFixed(1)}`;
      const dist = Math.sqrt(x * x + y * y);

      let semanticType: any = 'nondrivable';
      let traversability: any = 'Caution / Irregular';
      let elevation: any = 'Ground Level';
      let surfaceCondition: any = 'Rough / Unpaved';
      let isObstacle = false;
      let isDynamic = false;
      let obsState: any = 'Directly Observed';

      // Roadway corridor
      if (Math.abs(y) <= 4.2) {
        semanticType = 'drivable';
        traversability = 'Drivable / Safe';
        elevation = 'Ground Level';
        surfaceCondition = 'Paved / Smooth';
      }

      // Dynamic Lead Vehicle at X ≈ 7..9, Y ≈ 0
      if (x >= 6.5 && x <= 9.5 && Math.abs(y) <= 1.5) {
        semanticType = 'dynamicObstacle';
        traversability = 'Collision Hazard';
        elevation = 'Elevated Surface';
        surfaceCondition = 'Obstructed';
        isDynamic = true;
        isObstacle = true;
      }

      // Dynamic Pedestrian near crosswalk at X ≈ 4, Y ≈ 3.5
      if (Math.abs(x - 4) < 1.4 && Math.abs(y - 3.5) < 1.2) {
        semanticType = 'dynamicObstacle';
        traversability = 'Collision Hazard';
        elevation = 'Elevated Surface';
        surfaceCondition = 'Obstructed';
        isDynamic = true;
        isObstacle = true;
      }

      // Static Obstacle (Curb barrier / parked car at X ≈ -3..-1, Y ≈ -5.5)
      if (x >= -4 && x <= 0 && y <= -4.5 && y >= -7.0) {
        semanticType = 'staticObstacle';
        traversability = 'Non-Traversable';
        elevation = 'Elevated Surface';
        surfaceCondition = 'Obstructed';
        isObstacle = true;
      }

      // Sidewalk & foliage
      if (Math.abs(y) > 4.2 && Math.abs(y) <= 7.0 && !isObstacle) {
        if (Math.abs(x) > 5) {
          semanticType = 'vegetation';
          traversability = 'Non-Traversable';
          elevation = 'Variable Height';
          surfaceCondition = 'Rough / Unpaved';
        } else {
          semanticType = 'nondrivable';
          traversability = 'Caution / Irregular';
          elevation = 'Ground Level';
          surfaceCondition = 'Paved / Smooth';
        }
      }

      // Infrastructure on outer boundary
      if (Math.abs(y) > 7.0) {
        semanticType = 'infrastructure';
        traversability = 'Non-Traversable';
        elevation = 'Elevated Surface';
        surfaceCondition = 'Obstructed';
      }

      // Occlusion shadow behind lead vehicle (X > 10, Y ≈ 0)
      if (x > 9.5 && Math.abs(y) < 1.2) {
        semanticType = 'unknown';
        traversability = 'Uncertain';
        elevation = 'Variable Height';
        surfaceCondition = 'Unclassified';
        obsState = 'Partially Occluded';
      }

      cells.push({
        id: cellId,
        gridX: Math.round(x),
        gridY: Math.round(y),
        worldX: x,
        worldY: y,
        size: 1.5,
        zone: 'near',
        semanticType,
        semanticLabel: getSemanticLabel(semanticType),
        traversability,
        elevation,
        surfaceCondition,
        resolutionZone: 'Fine Detail Zone (Near-Field)',
        observationState: obsState,
        isObstacle,
        isDynamic,
      });
    }
  }

  // 2. MID-FIELD ZONE (Medium Cells, surrounding near-field)
  const midRanges = [
    { xMin: -22, xMax: -11, yMin: -20, yMax: 20 },
    { xMin: 13, xMax: 26, yMin: -20, yMax: 20 },
    { xMin: -10, xMax: 12, yMin: 11, yMax: 20 },
    { xMin: -10, xMax: 12, yMin: -20, yMax: -11 },
  ];

  midRanges.forEach(range => {
    for (let x = range.xMin; x <= range.xMax; x += 3.2) {
      for (let y = range.yMin; y <= range.yMax; y += 3.2) {
        const cellId = `cell_mid_${x.toFixed(1)}_${y.toFixed(1)}`;
        let semanticType: any = 'nondrivable';
        let traversability: any = 'Caution / Irregular';
        let elevation: any = 'Ground Level';
        let isObstacle = false;
        let isDynamic = false;

        if (Math.abs(y) <= 4.5) {
          semanticType = 'drivable';
          traversability = 'Drivable / Safe';
          elevation = 'Ground Level';
        } else if (Math.abs(y) <= 8.5) {
          semanticType = 'vegetation';
          traversability = 'Non-Traversable';
          elevation = 'Variable Height';
        } else {
          semanticType = 'infrastructure';
          traversability = 'Non-Traversable';
          elevation = 'Elevated Surface';
        }

        // Oncoming vehicle in mid-field (X ≈ 20, Y ≈ -2)
        if (Math.abs(x - 20) < 2.5 && Math.abs(y - (-2)) < 2.0) {
          semanticType = 'dynamicObstacle';
          traversability = 'Collision Hazard';
          isDynamic = true;
          isObstacle = true;
        }

        // Static road sign / pole
        if (Math.abs(x - (-16)) < 2.0 && Math.abs(y - 5.5) < 2.0) {
          semanticType = 'staticObstacle';
          traversability = 'Non-Traversable';
          isObstacle = true;
        }

        cells.push({
          id: cellId,
          gridX: Math.round(x),
          gridY: Math.round(y),
          worldX: x,
          worldY: y,
          size: 3.0,
          zone: 'mid',
          semanticType,
          semanticLabel: getSemanticLabel(semanticType),
          traversability,
          elevation,
          surfaceCondition: isObstacle ? 'Obstructed' : 'Paved / Smooth',
          resolutionZone: 'Medium Resolution Zone (Mid-Field)',
          observationState: 'Directly Observed',
          isObstacle,
          isDynamic,
        });
      }
    }
  });

  // 3. FAR-FIELD ZONE (Coarse Cells, outer boundary context)
  const farRanges = [
    { xMin: -36, xMax: -23, yMin: -30, yMax: 30 },
    { xMin: 27, xMax: 40, yMin: -30, yMax: 30 },
    { xMin: -22, xMax: 26, yMin: 21, yMax: 30 },
    { xMin: -22, xMax: 26, yMin: -30, yMax: -21 },
  ];

  farRanges.forEach(range => {
    for (let x = range.xMin; x <= range.xMax; x += 5.5) {
      for (let y = range.yMin; y <= range.yMax; y += 5.5) {
        const cellId = `cell_far_${x.toFixed(1)}_${y.toFixed(1)}`;
        let semanticType: any = 'unknown';
        let traversability: any = 'Uncertain';

        if (Math.abs(y) <= 5.0) {
          semanticType = 'drivable';
          traversability = 'Drivable / Safe';
        } else if (Math.abs(y) > 15) {
          semanticType = 'infrastructure';
          traversability = 'Non-Traversable';
        } else {
          semanticType = 'unknown';
          traversability = 'Uncertain';
        }

        cells.push({
          id: cellId,
          gridX: Math.round(x),
          gridY: Math.round(y),
          worldX: x,
          worldY: y,
          size: 5.2,
          zone: 'far',
          semanticType,
          semanticLabel: getSemanticLabel(semanticType),
          traversability,
          elevation: 'Variable Height',
          surfaceCondition: 'Unclassified',
          resolutionZone: 'Distant Context Zone (Far-Field)',
          observationState: 'Prior Inferred',
          isObstacle: false,
          isDynamic: false,
        });
      }
    }
  });

  return cells;
}

function getSemanticLabel(type: string): string {
  switch (type) {
    case 'drivable':
      return 'Drivable Surface';
    case 'nondrivable':
      return 'Non-Drivable Ground';
    case 'staticObstacle':
      return 'Static Structure / Obstacle';
    case 'dynamicObstacle':
      return 'Dynamic Obstacle';
    case 'vegetation':
      return 'Vegetation / Foliage';
    case 'infrastructure':
      return 'Infrastructure / Facade';
    case 'unknown':
    default:
      return 'Unknown / Occluded Sector';
  }
}

export const defaultResolutionPolicy: ResolutionPolicyConfig = {
  nearFieldDetail: 'High Precision (Fine Grid)',
  middleFieldDetail: 'Balanced Representation (Medium Grid)',
  farFieldDetail: 'Efficient Context (Coarse Grid)',
  safetyPriority: 'Maximum Collision Buffer',
  terrainComplexity: 'Adaptive Elevation Gradient',
  obstacleOverride: 'Forced Fine Cell Refinement',
};
