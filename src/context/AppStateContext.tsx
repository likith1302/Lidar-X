/**
 * Global Application State Context
 * Coordinates UI navigation, layer toggles, selected cell inspection,
 * model selection, live backend frame/terrain state, semantic predictions, objects, tracks, 2.5D maps, and SalsaNext inference.
 */

import React, { createContext, useContext, useState, useEffect } from 'react';
import { NavigationTab } from '../types/navigation';
import {
  GridCell,
  MapLayersState,
  ResolutionPolicyConfigFrontend,
  GridPolicyConfig,
  MapUpdateResponse,
} from '../types/map';
import { ViewMode, LidarFrame } from '../types/lidar';
import { TerrainAnalysisResponse } from '../types/terrain';
import {
  SemanticFrameResponse,
  ObjectDetectionResponse,
  TrackedObject,
  InferenceStatusResponse,
  InferenceResultsResponse,
} from '../types/objects';
import { FramePerformanceMetrics } from '../types/metrics';
import { defaultResolutionPolicy } from '../mocks/mockMapData';
import { SUPPORTED_MODELS } from '../mocks/mockSettings';
import { terrainService } from '../services/terrainService';
import { gridPolicyService, defaultGridPolicyConfig } from '../services/gridPolicyService';
import { semanticService } from '../services/semanticService';

interface AppStateContextType {
  currentTab: NavigationTab;
  setCurrentTab: (tab: NavigationTab) => void;
  isSidebarCollapsed: boolean;
  setIsSidebarCollapsed: (collapsed: boolean | ((prev: boolean) => boolean)) => void;
  isMobileDrawerOpen: boolean;
  setIsMobileDrawerOpen: (open: boolean) => void;
  mapLayers: MapLayersState;
  toggleLayer: (layerKey: keyof MapLayersState) => void;
  selectedCell: GridCell | null;
  setSelectedCell: (cell: GridCell | null) => void;
  selectedModelName: string;
  setSelectedModelName: (name: string) => void;
  selectedSequence: string;
  setSelectedSequence: (seq: string) => void;
  selectedFrameId: string;
  setSelectedFrameId: (frameId: string) => void;
  pointCloudViewMode: ViewMode;
  setPointCloudViewMode: (mode: ViewMode) => void;
  resolutionPolicy: ResolutionPolicyConfigFrontend;
  updateResolutionPolicy: (key: keyof ResolutionPolicyConfigFrontend, value: string) => void;
  notifications: { id: string; title: string; desc: string; type: 'info' | 'ready' | 'warn' }[];
  isNotificationsOpen: boolean;
  setIsNotificationsOpen: (open: boolean) => void;

  // Real Backend State
  isBackendConnected: boolean;
  setIsBackendConnected: (connected: boolean) => void;
  activeTerrainResponse: TerrainAnalysisResponse | null;
  setActiveTerrainResponse: (res: TerrainAnalysisResponse | null) => void;
  activeLidarFrame: LidarFrame | null;
  setActiveLidarFrame: (frame: LidarFrame | null) => void;

  // Semantic & Tracking State
  activeSemanticFrame: SemanticFrameResponse | null;
  setActiveSemanticFrame: (frame: SemanticFrameResponse | null) => void;
  activeObjectDetection: ObjectDetectionResponse | null;
  setActiveObjectDetection: (res: ObjectDetectionResponse | null) => void;
  activeTracks: TrackedObject[];
  setActiveTracks: (tracks: TrackedObject[]) => void;

  // 2.5D Adaptive Grid State
  activeGridPolicy: GridPolicyConfig;
  setActiveGridPolicy: (policy: GridPolicyConfig) => void;
  activeMapResponse: MapUpdateResponse | null;
  setActiveMapResponse: (res: MapUpdateResponse | null) => void;

  // SalsaNext Inference State
  activeInferenceStatus: InferenceStatusResponse | null;
  setActiveInferenceStatus: (status: InferenceStatusResponse | null) => void;
  activeInferenceResults: InferenceResultsResponse | null;
  setActiveInferenceResults: (results: InferenceResultsResponse | null) => void;

  // Frame Performance Telemetry
  activePerformance: FramePerformanceMetrics | null;
  setActivePerformance: (perf: FramePerformanceMetrics | null) => void;
}

const AppStateContext = createContext<AppStateContextType | undefined>(undefined);

export const AppStateProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentTab, setCurrentTab] = useState<NavigationTab>('overview');
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isMobileDrawerOpen, setIsMobileDrawerOpen] = useState(false);

  const [mapLayers, setMapLayers] = useState<MapLayersState>({
    semanticClasses: true,
    elevation: true,
    traversability: true,
    staticObstacles: true,
    dynamicObstacles: true,
    unknownAreas: true,
  });

  const [selectedCell, setSelectedCell] = useState<GridCell | null>(null);
  const [selectedModelName, setSelectedModelName] = useState<string>(SUPPORTED_MODELS[0].modelName);
  const [selectedSequence, setSelectedSequence] = useState<string>('SEQ_CAMPUS_AUTONOMOUS_01 (Mock)');
  const [selectedFrameId, setSelectedFrameId] = useState<string>('FRAME_0001 (Mock)');
  const [pointCloudViewMode, setPointCloudViewMode] = useState<ViewMode>('perspective');
  const [resolutionPolicy, setResolutionPolicy] = useState<ResolutionPolicyConfigFrontend>(defaultResolutionPolicy);
  const [isNotificationsOpen, setIsNotificationsOpen] = useState(false);

  // Backend connection & result state
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const [activeTerrainResponse, setActiveTerrainResponse] = useState<TerrainAnalysisResponse | null>(null);
  const [activeLidarFrame, setActiveLidarFrame] = useState<LidarFrame | null>(null);
  const [activeSemanticFrame, setActiveSemanticFrame] = useState<SemanticFrameResponse | null>(null);
  const [activeObjectDetection, setActiveObjectDetection] = useState<ObjectDetectionResponse | null>(null);
  const [activeTracks, setActiveTracks] = useState<TrackedObject[]>([]);
  const [activeGridPolicy, setActiveGridPolicy] = useState<GridPolicyConfig>(defaultGridPolicyConfig);
  const [activeMapResponse, setActiveMapResponse] = useState<MapUpdateResponse | null>(null);

  // SalsaNext Inference State
  const [activeInferenceStatus, setActiveInferenceStatus] = useState<InferenceStatusResponse | null>(null);
  const [activeInferenceResults, setActiveInferenceResults] = useState<InferenceResultsResponse | null>(null);

  // Real-Time Frame Performance State
  const [activePerformance, setActivePerformance] = useState<FramePerformanceMetrics | null>(null);

  const [notifications] = useState([
    {
      id: 'notif-1',
      title: 'Backend Pipeline Ready',
      desc: 'Python FastAPI server support initialized for terrain analysis, semantic segmentation, and tracking.',
      type: 'ready' as const,
    },
    {
      id: 'notif-2',
      title: 'Fast-FRNet Inference Service Active',
      desc: 'PyTorch deep learning semantic segmentation pipeline connected with frustum-point projection.',
      type: 'info' as const,
    },
    {
      id: 'notif-3',
      title: '2.5D Foveated Grid Ready',
      desc: 'Variable-resolution foveated elevation grid with static/dynamic hazard separation available.',
      type: 'ready' as const,
    },
  ]);

  // Check backend health periodically and sync grid policy + model status
  useEffect(() => {
    let isMounted = true;
    const checkBackend = async () => {
      try {
        const res = await terrainService.checkHealth();
        if (isMounted && res.success && !res.isMock) {
          setIsBackendConnected(true);

          // Sync grid policy
          const polRes = await gridPolicyService.getPolicy();
          if (polRes.success && polRes.data) {
            setActiveGridPolicy(polRes.data);
          }

          // Sync inference model status
          const infStatusRes = await semanticService.getInferenceStatus();
          if (infStatusRes.success && infStatusRes.data) {
            setActiveInferenceStatus(infStatusRes.data);
          }
        }
      } catch {
        if (isMounted) setIsBackendConnected(false);
      }
    };
    checkBackend();
    const interval = setInterval(checkBackend, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const toggleLayer = (layerKey: keyof MapLayersState) => {
    setMapLayers(prev => ({
      ...prev,
      [layerKey]: !prev[layerKey],
    }));
  };

  const updateResolutionPolicy = (key: keyof ResolutionPolicyConfigFrontend, value: string) => {
    setResolutionPolicy(prev => ({
      ...prev,
      [key]: value,
    }));
  };

  return (
    <AppStateContext.Provider
      value={{
        currentTab,
        setCurrentTab,
        isSidebarCollapsed,
        setIsSidebarCollapsed,
        isMobileDrawerOpen,
        setIsMobileDrawerOpen,
        mapLayers,
        toggleLayer,
        selectedCell,
        setSelectedCell,
        selectedModelName,
        setSelectedModelName,
        selectedSequence,
        setSelectedSequence,
        selectedFrameId,
        setSelectedFrameId,
        pointCloudViewMode,
        setPointCloudViewMode,
        resolutionPolicy,
        updateResolutionPolicy,
        notifications,
        isNotificationsOpen,
        setIsNotificationsOpen,
        isBackendConnected,
        setIsBackendConnected,
        activeTerrainResponse,
        setActiveTerrainResponse,
        activeLidarFrame,
        setActiveLidarFrame,
        activeSemanticFrame,
        setActiveSemanticFrame,
        activeObjectDetection,
        setActiveObjectDetection,
        activeTracks,
        setActiveTracks,
        activeGridPolicy,
        setActiveGridPolicy,
        activeMapResponse,
        setActiveMapResponse,
        activeInferenceStatus,
        setActiveInferenceStatus,
        activeInferenceResults,
        setActiveInferenceResults,
        activePerformance,
        setActivePerformance,
      }}
    >
      {children}
    </AppStateContext.Provider>
  );
};

export function useAppState(): AppStateContextType {
  const context = useContext(AppStateContext);
  if (!context) {
    throw new Error('useAppState must be used within an AppStateProvider');
  }
  return context;
}
