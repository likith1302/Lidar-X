/**
 * Mock 3D LiDAR Point Cloud Generation
 * Generates an authentic autonomous driving scene with vehicle, road, buildings, vegetation, and obstacles.
 */

import { LidarFrame, Point3D, FrameMetadata } from '../types/lidar';

export function generateMockPointCloud(): Point3D[] {
  const points: Point3D[] = [];

  // 1. Drivable Road Surface (X: -15 to +35, Y: -4 to +4, Z ≈ -1.73 sensor height)
  for (let x = -15; x <= 35; x += 0.7) {
    for (let y = -4; y <= 4; y += 0.6) {
      // Add slight road crown and subtle LiDAR ring texture
      const dist = Math.sqrt(x * x + y * y);
      const ring = Math.floor(dist * 1.5) % 64;
      const jitterX = (Math.random() - 0.5) * 0.15;
      const jitterY = (Math.random() - 0.5) * 0.15;
      const z = -1.73 + Math.sin(x * 0.05) * 0.04 - (y * y) * 0.003 + (Math.random() - 0.5) * 0.02;
      
      // Road lane markings have higher intensity
      const isMarking = Math.abs(y) < 0.1 || Math.abs(y - 3.5) < 0.1 || Math.abs(y + 3.5) < 0.1;
      const intensity = isMarking ? 0.95 : 0.25 + Math.random() * 0.15;

      points.push({
        x: x + jitterX,
        y: y + jitterY,
        z,
        intensity,
        semanticClass: 'drivable',
        ring,
      });
    }
  }

  // 2. Sidewalk & Non-Drivable Ground (Y: 4 to 8 and -8 to -4)
  for (let x = -15; x <= 35; x += 0.9) {
    for (let y of [4.5, 5.3, 6.2, 7.1, -4.5, -5.3, -6.2, -7.1]) {
      const dist = Math.sqrt(x * x + y * y);
      const ring = Math.floor(dist * 1.5) % 64;
      const z = -1.58 + (Math.random() - 0.5) * 0.04; // Curb elevation step
      points.push({
        x: x + (Math.random() - 0.5) * 0.2,
        y: y + (Math.random() - 0.5) * 0.2,
        z,
        intensity: 0.35 + Math.random() * 0.2,
        semanticClass: 'nondrivable',
        ring,
      });
    }
  }

  // 3. Buildings / Infrastructure Facades (Y > 8 and Y < -8)
  const buildingBlocks = [
    { xMin: -10, xMax: 5, y: 8.5, height: 6 },
    { xMin: 8, xMax: 28, y: 9.0, height: 7.5 },
    { xMin: -12, xMax: 2, y: -9.0, height: 5 },
    { xMin: 6, xMax: 25, y: -8.8, height: 6.5 },
  ];

  buildingBlocks.forEach(b => {
    for (let bx = b.xMin; bx <= b.xMax; bx += 0.8) {
      for (let bz = -1.5; bz <= b.height - 1.5; bz += 0.7) {
        const dist = Math.sqrt(bx * bx + b.y * b.y + bz * bz);
        const ring = Math.floor(dist * 1.2) % 64;
        points.push({
          x: bx + (Math.random() - 0.5) * 0.1,
          y: b.y + (Math.random() - 0.5) * 0.15,
          z: bz + (Math.random() - 0.5) * 0.1,
          intensity: 0.4 + Math.random() * 0.3,
          semanticClass: 'infrastructure',
          ring,
        });
      }
    }
  });

  // 4. Vegetation / Street Trees
  const treePositions = [
    { x: -5, y: 5.8, height: 4.5, radius: 1.8 },
    { x: 12, y: 6.2, height: 5.0, radius: 2.1 },
    { x: 26, y: 6.0, height: 4.8, radius: 1.9 },
    { x: -2, y: -6.2, height: 4.2, radius: 1.7 },
    { x: 18, y: -6.5, height: 5.2, radius: 2.2 },
  ];

  treePositions.forEach(t => {
    // Trunk
    for (let tz = -1.5; tz <= 1.0; tz += 0.4) {
      points.push({
        x: t.x + (Math.random() - 0.5) * 0.25,
        y: t.y + (Math.random() - 0.5) * 0.25,
        z: tz,
        intensity: 0.3,
        semanticClass: 'vegetation',
      });
    }
    // Foliage sphere
    for (let i = 0; i < 65; i++) {
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);
      const r = Math.cbrt(Math.random()) * t.radius;
      const sinPhi = Math.sin(phi);
      points.push({
        x: t.x + r * sinPhi * Math.cos(theta),
        y: t.y + r * sinPhi * Math.sin(theta),
        z: 1.5 + r * Math.cos(phi),
        intensity: 0.2 + Math.random() * 0.4,
        semanticClass: 'vegetation',
      });
    }
  });

  // 5. Dynamic Lead Vehicle Ahead (X = 14, Y = 0.3)
  const leadCar = { x: 14, y: 0.3, length: 4.2, width: 1.8, height: 1.4 };
  for (let cx = -leadCar.length / 2; cx <= leadCar.length / 2; cx += 0.35) {
    for (let cy = -leadCar.width / 2; cy <= leadCar.width / 2; cy += 0.35) {
      for (let cz = 0; cz <= leadCar.height; cz += 0.35) {
        // Only surface points
        const isBorder =
          Math.abs(cx) > leadCar.length / 2 - 0.4 ||
          Math.abs(cy) > leadCar.width / 2 - 0.4 ||
          cz > leadCar.height - 0.4;
        if (isBorder) {
          points.push({
            x: leadCar.x + cx + (Math.random() - 0.5) * 0.05,
            y: leadCar.y + cy + (Math.random() - 0.5) * 0.05,
            z: -1.6 + cz + (Math.random() - 0.5) * 0.05,
            intensity: 0.75 + Math.random() * 0.2,
            semanticClass: 'dynamicObstacle',
          });
        }
      }
    }
  }

  // 6. Dynamic Pedestrian / Cyclist (X = 6.5, Y = 3.8)
  for (let pz = -1.5; pz <= 0.2; pz += 0.15) {
    for (let i = 0; i < 6; i++) {
      points.push({
        x: 6.5 + (Math.random() - 0.5) * 0.4,
        y: 3.8 + (Math.random() - 0.5) * 0.4,
        z: pz,
        intensity: 0.6 + Math.random() * 0.3,
        semanticClass: 'dynamicObstacle',
      });
    }
  }

  // 7. Static Obstacles (Parked car, barrier, bollards)
  // Static Parked Car (X = 2, Y = -4.8)
  for (let px = -1.8; px <= 1.8; px += 0.4) {
    for (let py = -0.8; py <= 0.8; py += 0.4) {
      for (let pz = 0; pz <= 1.3; pz += 0.4) {
        if (Math.abs(px) > 1.4 || Math.abs(py) > 0.6 || pz > 1.0) {
          points.push({
            x: 2 + px + (Math.random() - 0.5) * 0.05,
            y: -4.8 + py + (Math.random() - 0.5) * 0.05,
            z: -1.6 + pz,
            intensity: 0.55 + Math.random() * 0.25,
            semanticClass: 'staticObstacle',
          });
        }
      }
    }
  }

  // Traffic Bollards / Guardrail markers
  for (let bx = -10; bx <= 20; bx += 4) {
    for (let bz = -1.5; bz <= -0.5; bz += 0.2) {
      points.push({
        x: bx,
        y: 4.1 + (Math.random() - 0.5) * 0.05,
        z: bz,
        intensity: 0.85,
        semanticClass: 'staticObstacle',
      });
    }
  }

  return points;
}

export const mockFrameMetadata: FrameMetadata = {
  scanSource: 'Awaiting Data Connection',
  timestamp: 'Available When Loaded',
  calibration: 'Optional / Pending Sensor Extrinsics',
  labels: 'Awaiting Inference Pipeline',
  vehiclePose: 'Not Supplied (Awaiting Odometry Stream)',
  sensorConfiguration: 'Mechanical 360° LiDAR / Solid-State Hybrid (Simulated)',
  beamCount: 'Multi-Channel Array Pattern',
  horizontalFov: 'Full 360° Azimuth Envelope',
  rangeCapability: 'Variable Envelope (Foveated Scale)',
};

export const mockLidarFrame: LidarFrame = {
  frameId: 'FRAME_MOCK_0001',
  sequenceId: 'SEQ_RESEARCH_CAMPUS_01',
  points: generateMockPointCloud(),
  metadata: mockFrameMetadata,
};
