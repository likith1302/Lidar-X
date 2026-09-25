/**
 * Mock Architecture & Pipeline Stages
 */

import { PipelineStage, SystemConnectionItem } from '../types/navigation';

export interface ArchitectureModule {
  id: string;
  name: string;
  category: 'Ingestion' | 'Deep Learning' | 'Mapping' | 'Integration';
  status: 'Backend Active' | 'Integration Ready' | 'Planned' | 'Frontend Live (Mock)';
  description: string;
  technology: string;
  inputFormat: string;
  outputFormat: string;
  integrationNote: string;
}

export const ARCHITECTURE_MODULES: ArchitectureModule[] = [
  {
    id: 'lidar-adapter',
    name: 'LiDAR Input Adapter & Ingestion',
    category: 'Ingestion',
    status: 'Backend Active',
    description: 'Ingests float32 SemanticKITTI .bin scans, sequence ZIP archives, and point cloud streams via FastAPI and WebSocket.',
    technology: 'Python FastAPI / WebSocket Streaming / NumPy',
    inputFormat: 'Binary SemanticKITTI .bin / Sequence ZIP Archives',
    outputFormat: 'Normalized [N, 4] (X, Y, Z, Intensity) float32 Arrays',
    integrationNote: 'Active backend service supporting single-frame and continuous 60 FPS stream replay.',
  },
  {
    id: 'preprocessing',
    name: 'Point Cloud Preprocessing',
    category: 'Ingestion',
    status: 'Backend Active',
    description: 'Applies NaN/Inf sanitization, radial range gating (0.5m - 80m), and elevation bounding filters.',
    technology: 'Vectorized NumPy / PointCloudPreprocessor',
    inputFormat: 'Raw Point Cloud Arrays',
    outputFormat: 'Sanitized Point Cloud & Preprocessing Report',
    integrationNote: 'Active in backend pipeline with zero dropped point data loss.',
  },
  {
    id: 'semantic-segmentation',
    name: 'Fast-FRNet Deep Semantic Segmentation',
    category: 'Deep Learning',
    status: 'Backend Active',
    description: 'PyTorch deep neural network executing point-wise semantic labeling using verified Fast-FRNet checkpoints (RELLIS-3D off-road primary and SemanticKITTI urban).',
    technology: 'PyTorch / Fast-FRNet Frustum-Point Fusion [32x512 / 64x512]',
    inputFormat: 'LiDAR Point Coordinates [N, 4] -> Frustum Feature Projection',
    outputFormat: 'Point-wise Semantic Class Predictions & Distributions',
    integrationNote: 'Active backend inference service with genuine neural inference, CUDA GPU acceleration, and CPU fallback.',
  },
  {
    id: 'class-mapping',
    name: 'Terrain & Affordance Class Mapping',
    category: 'Deep Learning',
    status: 'Backend Active',
    description: 'Translates 20 learning classes to standardized project categories (drivable, non-drivable, static obstacle, dynamic actor).',
    technology: 'SemanticLabelMappingService / Vectorized Lookups',
    inputFormat: '20-Class Learning Label Arrays',
    outputFormat: 'Semantic Affordances & Project Categories',
    integrationNote: 'Active in backend perception pipeline.',
  },
  {
    id: 'geometric-analysis',
    name: 'Geometric Terrain Analysis Engine',
    category: 'Mapping',
    status: 'Backend Active',
    description: 'Extracts 3D PCA surface normals, slope angles, RMS roughness, and curb/step height discontinuities.',
    technology: 'NumPy SVD / Covariance Eigen-Decomposition',
    inputFormat: '3D Spatial Point Cloud',
    outputFormat: 'Slope, Roughness, Step Height, & Traversability State',
    integrationNote: 'Active backend service supporting ultra-fine cell resolutions.',
  },
  {
    id: 'adaptive-policy',
    name: 'Adaptive Variable-Resolution Policy',
    category: 'Mapping',
    status: 'Backend Active',
    description: 'Dynamically allocates fine resolution (5cm within 10m) and coarse resolution (up to 50cm at 100m) with feature overrides.',
    technology: 'ResolutionPolicyService / Nested Spatial Index',
    inputFormat: 'Point Cloud Proximity + Terrain Complexity + Actor Triggers',
    outputFormat: 'Hierarchical Resolution Mask & 2:1 Nested Keys',
    integrationNote: 'Active backend engine with persistent YAML configuration.',
  },
  {
    id: 'grid-engine',
    name: '2.5D Adaptive Grid Engine',
    category: 'Mapping',
    status: 'Backend Active',
    description: 'Projects classified 3D points into a foveated elevation-aware grid with multi-surface overhang clearance support.',
    technology: 'AdaptiveGridService / Multi-Layer Elevation Hash',
    inputFormat: 'Classified 3D Points + Resolution Policy',
    outputFormat: '2.5D Multi-Resolution Grid State (Fine/Med/Coarse)',
    integrationNote: 'Active backend service with real-time cell querying and multi-layer interval detection.',
  },
  {
    id: 'local-map-state',
    name: 'Multi-Frame Map Fusion & SLAM Pose Integration',
    category: 'Mapping',
    status: 'Backend Active',
    description: 'Transforms local points using vehicle ego-motion odometry and fuses static properties across sequential frames.',
    technology: 'MapFusionService / Rigid Body SE(2)/SE(3) Transforms',
    inputFormat: 'Local Frame Cells + Ego Pose Trajectory',
    outputFormat: 'Fused Global World Map State',
    integrationNote: 'Active backend service supporting local-only and global fusion modes.',
  },
  {
    id: 'pose-odometry',
    name: 'Future Pose & Odometry Service',
    category: 'Integration',
    status: 'Planned',
    description: 'Ingests high-rate IMU/GNSS/LiDAR SLAM state to register map layers across time and vehicle motion.',
    technology: 'LiDAR Odometry / EKF / ROS2 nav_msgs/Odometry',
    inputFormat: 'Sensor IMU + Wheel Ticks + LiDAR Scans',
    outputFormat: '6-DoF Ego Vehicle Pose Matrix',
    integrationNote: 'Will synchronize visualization frame rates with vehicle pose.',
  },
  {
    id: 'web-api',
    name: 'Web Visualization API',
    category: 'Integration',
    status: 'Frontend Live (Mock)',
    description: 'High-performance interactive web application UI rendering 3D LiDAR point clouds and 2.5D adaptive semantic maps.',
    technology: 'React / TypeScript / Three.js / Canvas2D / Tailwind CSS',
    inputFormat: 'Mock State Objects (Future: gRPC-Web / WebSocket Stream)',
    outputFormat: 'Interactive Visual Presentation & Inspection Shell',
    integrationNote: 'Active frontend prototype foundation.',
  },
];

export const SYSTEM_CONNECTIONS: SystemConnectionItem[] = [
  {
    id: 'lidar-source',
    name: 'LiDAR Data Source',
    statusText: 'Active Stream',
    badgeVariant: 'emerald',
    description: 'SemanticKITTI binary scans and sequence archive ingestion active with 60 FPS streaming via WebSocket.',
    endpointPlaceholder: 'ws://localhost:8000/api/replay/ws',
  },
  {
    id: 'semantic-service',
    name: 'Semantic Model Service',
    statusText: 'Fast-FRNet Online',
    badgeVariant: 'emerald',
    description: 'Deep learning inference pipeline active with PyTorch Fast-FRNet checkpoints (RELLIS-3D primary) and GPU/CPU execution.',
    endpointPlaceholder: 'http://localhost:8000/api/v1/inference/segment',
  },
  {
    id: 'mapping-engine',
    name: 'Mapping Engine',
    statusText: 'Adaptive 2.5D Active',
    badgeVariant: 'emerald',
    description: 'Adaptive 2.5D variable-resolution grid engine active with multi-surface overhang clearance detection.',
    endpointPlaceholder: 'http://localhost:8000/api/map/grid',
  },
];

export const OVERVIEW_PIPELINE_STAGES: PipelineStage[] = [
  {
    id: 'p1',
    stepNumber: 1,
    title: 'Raw LiDAR Input',
    shortDescription: 'Captures high-frequency 3D point cloud streams from autonomous vehicle sensors.',
    fullDescription: 'Ingests unorganized 3D coordinate rays (X, Y, Z, Intensity) directly from LiDAR laser sweeps, representing the raw spatial environment.',
    iconName: 'Radio',
    status: 'Backend Active',
    inputs: 'LiDAR Laser Returns [X, Y, Z, Intensity]',
    outputs: 'Unclassified 3D Point Stream',
  },
  {
    id: 'p2',
    stepNumber: 2,
    title: 'Semantic AI Model',
    shortDescription: 'Deep learning neural network performs point-wise semantic scene segmentation.',
    fullDescription: 'Passes spatial points through deep learning architectures (Fast-FRNet) to classify every point into terrain, vehicle, vegetation, or structural classes.',
    iconName: 'Cpu',
    isDeepLearning: true,
    status: 'Backend Active',
    inputs: 'Point Cloud Coordinates & Frustum Projections [32x512 / 64x512]',
    outputs: 'Per-Point Semantic Class Probabilities',
  },
  {
    id: 'p3',
    stepNumber: 3,
    title: 'Terrain & Object Understanding',
    shortDescription: 'Synthesizes classified points into spatial boundaries and traversability layers.',
    fullDescription: 'Extracts geometric slope, curb boundaries, obstacle clusters, and dynamic motion tracks from semantic labels to evaluate navigable corridors.',
    iconName: 'Layers',
    status: 'Backend Active',
    inputs: 'Labeled 3D Points + Normal Vectors',
    outputs: 'Drivable, Hazard & Static Object Layers',
  },
  {
    id: 'p4',
    stepNumber: 4,
    title: 'Adaptive 2.5D Map',
    shortDescription: 'Projects points into variable-resolution foveated elevation grid cells.',
    fullDescription: 'Constructs an elevation-aware 2.5D map using fine grid cells (5cm) in safety-critical near-field zones and larger cells (up to 50cm) in distant context zones.',
    iconName: 'Grid',
    status: 'Backend Active',
    inputs: 'Segmented Point Features + Ego Pose',
    outputs: 'Multi-Resolution 2.5D Elevation Grid',
  },
  {
    id: 'p5',
    stepNumber: 5,
    title: 'Navigation Insight',
    shortDescription: 'Provides real-time spatial intelligence for safe autonomous vehicle trajectory planning.',
    fullDescription: 'Feeds structured drivability, elevation hazards, and dynamic obstacle boundaries directly into downstream path planning and motion control modules.',
    iconName: 'Navigation',
    status: 'Backend Active',
    inputs: 'Local 2.5D Grid State',
    outputs: 'Motion Planning Safe Corridor Constraints',
  },
];
