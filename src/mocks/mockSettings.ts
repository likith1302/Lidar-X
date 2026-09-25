/**
 * Mock Settings & Configurable Models
 */

import { SemanticModelSpec } from '../types/semantic';

export interface AppSettings {
  selectedModelName: string;
  backendApiEndpoint: string;
  lidarStreamEndpoint: string;
  semanticServiceEndpoint: string;
  mappingEngineEndpoint: string;
  simulationPlaybackSpeed: string;
  renderingTheme: string;
  enableElevationShading: boolean;
  enableGridOverlay: boolean;
}

export const SUPPORTED_MODELS: SemanticModelSpec[] = [
  {
    modelName: 'Fast-FRNet (RELLIS-3D Off-Road LiDAR)',
    framework: 'PyTorch 2.x / Fast-FRNet',
    inputType: 'Point Cloud [N, 4] -> Frustum Projection [32x512]',
    outputType: '20-Class Point Logits [N, 20]',
    inferenceEndpoint: 'http://localhost:8000/api/v1/inference/segment',
    modelStatus: 'Active / Verified Checkpoint',
    architectureNotes: 'ResNet-34 Frustum-Point fusion network fine-tuned on RELLIS-3D off-road dataset. Primary model for LiDAR-X.',
  },
  {
    modelName: 'Fast-FRNet (SemanticKITTI Urban LiDAR)',
    framework: 'PyTorch 2.x / Fast-FRNet',
    inputType: 'Point Cloud [N, 4] -> Frustum Projection [64x512]',
    outputType: '20-Class Point Logits [N, 20]',
    inferenceEndpoint: 'http://localhost:8000/api/v1/inference/segment',
    modelStatus: 'Active / Verified Checkpoint',
    architectureNotes: 'ResNet-34 Frustum-Point fusion network pretrained on SemanticKITTI for general and urban LiDAR scenes.',
  },
  {
    modelName: 'PointNet++ (Hierarchical Point Set)',
    framework: 'PyTorch / CUDA C++',
    inputType: 'Raw Point Coordinates [N, 4]',
    outputType: 'Point Semantic Classes [N, 7]',
    inferenceEndpoint: 'http://localhost:8001/v1/models/pointnet2:predict',
    modelStatus: 'Integration Pending',
    architectureNotes: 'Multi-scale grouping point set architecture for fine spatial structure capture.',
  },
  {
    modelName: 'Cylinder3D (Cylindrical Sparse Voxel)',
    framework: 'PyTorch / SpConv v2.3',
    inputType: 'Cylindrical Voxel Grid [480x360x32]',
    outputType: 'Voxel Class Predictions [480x360x32, 7]',
    inferenceEndpoint: 'http://localhost:8001/v1/models/cylinder3d:predict',
    modelStatus: 'Integration Pending',
    architectureNotes: 'Asymmetrical 3D convolution on cylindrical coordinate voxels for LiDAR point distribution.',
  },
  {
    modelName: 'RangeNet++ (Spherical Projection)',
    framework: 'TensorRT / C++ Runtime',
    inputType: 'Spherical Range Matrix [5, 64, 2048]',
    outputType: 'Semantic Range Mask [64, 2048]',
    inferenceEndpoint: 'http://localhost:8001/v1/models/rangenet:predict',
    modelStatus: 'Integration Pending',
    architectureNotes: 'Lightweight DarkNet backbone projection for real-time edge embedded platforms.',
  },
  {
    modelName: 'MinkowskiEngine (Sparse Tensor ConvNet)',
    framework: 'PyTorch / MinkowskiEngine 0.5',
    inputType: 'Sparse 3D Coordinate Hash Tensor',
    outputType: 'Point Semantic Labels',
    inferenceEndpoint: 'http://localhost:8001/v1/models/minkowski:predict',
    modelStatus: 'Integration Pending',
    architectureNotes: 'High-capacity generalized sparse convolution network for full 3D point cloud labeling.',
  },
];

export const defaultAppSettings: AppSettings = {
  selectedModelName: 'Fast-FRNet (RELLIS-3D Off-Road LiDAR)',
  backendApiEndpoint: 'http://localhost:8000/api/v1',
  lidarStreamEndpoint: 'ws://localhost:8765/lidar/stream',
  semanticServiceEndpoint: 'http://localhost:8001/v1/models/foveamap:predict',
  mappingEngineEndpoint: 'ws://localhost:8766/map/adaptive-grid',
  simulationPlaybackSpeed: '1.0x (Standard)',
  renderingTheme: 'Deep Charcoal / Electric Cyan (Technical Dark)',
  enableElevationShading: true,
  enableGridOverlay: true,
};
