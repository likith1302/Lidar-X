/**
 * Application Navigation & System Connection Types
 */

export type NavigationTab =
  | 'overview'
  | 'mapping-console'
  | 'architecture'
  | 'performance-lab'
  | 'scenes-replay'
  | 'diagnostics';

export interface SystemConnectionItem {
  id: string;
  name: string;
  statusText: 'Mock Mode' | 'Not Connected' | 'Integration Ready' | 'Active Stream' | 'Fast-FRNet Online' | 'SalsaNext Online' | 'Adaptive 2.5D Active';
  badgeVariant: 'neutral' | 'warning' | 'purple' | 'cyan' | 'emerald';
  description: string;
  endpointPlaceholder: string;
}

export interface PipelineStage {
  id: string;
  stepNumber: number;
  title: string;
  shortDescription: string;
  fullDescription: string;
  iconName: string;
  isDeepLearning?: boolean;
  status: 'Mock Implemented' | 'Planned' | 'Integration Ready' | 'Backend Active';
  inputs: string;
  outputs: string;
}
