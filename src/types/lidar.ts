/**
 * 3D LiDAR Point Cloud Types and Frame Metadata
 */

export interface Point3D {
  x: number;
  y: number;
  z: number;
  intensity?: number;
  semanticClass?: string;
  ring?: number;
}

export interface BoundingBox3D {
  min_x: number;
  max_x: number;
  min_y: number;
  max_y: number;
  min_z: number;
  max_z: number;
}

export type ViewMode = 'perspective' | 'top' | 'intensity' | 'semantic' | 'raw';

export interface FrameMetadata {
  scanSource: string;
  timestamp: string;
  calibration: string;
  labels: string;
  vehiclePose: string;
  sensorConfiguration: string;
  beamCount: string;
  horizontalFov: string;
  rangeCapability: string;
}

export interface LidarFrame {
  frameId: string;
  sequenceId: string;
  points: Point3D[];
  metadata: FrameMetadata;
}

export type ConnectionStatus = 'mock' | 'not-connected' | 'integration-ready' | 'connected';
