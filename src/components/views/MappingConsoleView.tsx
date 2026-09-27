import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  Layers,
  Eye,
  EyeOff,
  Crosshair,
  Info,
  Shield,
  Activity,
  AlertCircle,
  HelpCircle,
  Maximize2,
  TrendingUp,
  Sliders,
  Sparkles,
  Compass,
  Navigation,
  Radio,
  Car,
  User,
  Box,
  Grid,
  Download,
  Upload,
  Cpu,
  Mountain,
  Film,
  Play,
  Loader2,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { useAppState } from '../../context/AppStateContext';
import { MapCanvas2D } from '../visualizers/MapCanvas2D';
import { LidarCanvas3D } from '../visualizers/LidarCanvas3D';
import { ElevationMap2D } from '../visualizers/ElevationMap2D';
import { TraversabilityMap2D } from '../visualizers/TraversabilityMap2D';
import type { PipelineStages } from '../../types/pipeline';
import { SequenceReplayBar, ObjectDetectionActionState } from '../console/SequenceReplayBar';
import { PerformanceValidationPanel } from '../console/PerformanceValidationPanel';
import { PipelineService } from '../../services/pipelineService';
import { PipelineStatusResponse } from '../../types/pipeline';
import { lidarService } from '../../services/lidarService';
import { semanticService } from '../../services/semanticService';
import { terrainService } from '../../services/terrainService';
import { objectService } from '../../services/objectService';
import { trackingService } from '../../services/trackingService';
import { mapService } from '../../services/mapService';
import { replayService } from '../../services/replayService';
import { GridCell, MapLayersState } from '../../types/map';
import { Point3D, ViewMode } from '../../types/lidar';
import { ObjectInstance, TrackedObject } from '../../types/objects';
import {
  ReplayState,
  ReplayDataMode,
  ReplayFrameStreamPayload,
  ReplaySessionStatus,
  SemanticSource,
} from '../../types/replay';
import { Badge } from '../common/Badge';

export interface MappingConsoleViewProps {
  initialMode?: 'single-frame' | 'sequence-replay';
}

export const MappingConsoleView: React.FC<MappingConsoleViewProps> = ({ initialMode }) => {
  const {
    mapLayers,
    toggleLayer,
    selectedCell,
    setSelectedCell,
    selectedFrameId,
    setSelectedFrameId,
    activeLidarFrame,
    setActiveLidarFrame,
    activeTerrainResponse,
    setActiveTerrainResponse,
    activeSemanticFrame,
    setActiveSemanticFrame,
    activeObjectDetection,
    setActiveObjectDetection,
    activeTracks,
    setActiveTracks,
    activeMapResponse,
    setActiveMapResponse,
    activeInferenceStatus,
    activeInferenceResults,
    setActiveInferenceResults,
    activePerformance,
    setActivePerformance,
    isBackendConnected,
    setCurrentTab,
  } = useAppState();

  // Active Map Cells
  const [cells, setCells] = useState<GridCell[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isPipelineRunning, setIsPipelineRunning] = useState(false);

  // Mode: 'single-frame' vs 'sequence-replay'
  const [consoleMode, setConsoleMode] = useState<'single-frame' | 'sequence-replay'>(initialMode || 'sequence-replay');

  useEffect(() => {
    if (initialMode) {
      setConsoleMode(initialMode);
    }
  }, [initialMode]);
  // Controls whether the user has uploaded/selected data or is viewing the ingestion portal
  const [hasUploaded, setHasUploaded] = useState<boolean>(false);
  const sequenceInputRef = useRef<HTMLInputElement | null>(null);

  // Sequence Replay State
  const [replaySessionId, setReplaySessionId] = useState<string | null>(null);
  const [replaySequenceName, setReplaySequenceName] = useState<string>('SemanticKITTI Sequence');
  const [replayPlaybackState, setReplayPlaybackState] = useState<ReplayState>('ready');
  const [replayFrameIndex, setReplayFrameIndex] = useState<number>(0);
  const [replayTotalFrames, setReplayTotalFrames] = useState<number>(0);
  const [replayFps, setReplayFps] = useState<number>(10.0);
  const [realtimeFps, setRealtimeFps] = useState<number>(0.0);
  const frameArrivalsRef = useRef<number[]>([]);
  const [replayDataMode, setReplayDataMode] = useState<ReplayDataMode>('precomputed_labels');
  const [replaySemanticSource, setReplaySemanticSource] = useState<SemanticSource>('LIVE GEOMETRIC');
  const [replayProcessingTimeMs, setReplayProcessingTimeMs] = useState<number>(0);
  const [isUploadingSequence, setIsUploadingSequence] = useState<boolean>(false);
  const wsStreamRef = useRef<{ send: (msg: any) => void; close: () => void; isConnected?: () => boolean } | null>(null);
  const [isWebSocketConnected, setIsWebSocketConnected] = useState<boolean>(false);

  // Precompute progress (per replay session)
  const [precomputeStatus, setPrecomputeStatus] = useState<{
    isRunning: boolean;
    isComplete: boolean;
    percentComplete: number;
    processedFrames: number;
    totalFrames: number;
    etaSeconds: number;
    currentStage?: string;
    failedCount?: number;
    errorMessage?: string | null;
  } | null>(null);

  const [uploadPlaybackMode, setUploadPlaybackMode] = useState<'offline_precomputed_replay' | 'live_processing'>('live_processing');
  const [localArchivePath, setLocalArchivePath] = useState<string>('');
  const [isIngestingArchive, setIsIngestingArchive] = useState<boolean>(false);

  const precomputePollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const activeUploadIdRef = useRef<string | null>(null);

  // Refs for the selected cell resolver (avoids stale-closure)
  const replaySessionIdRef = useRef<string | null>(null);
  useEffect(() => {
    replaySessionIdRef.current = replaySessionId;
  }, [replaySessionId]);

  // Raw & Semantic 3D Point Cloud View Controls
  const [rawViewMode, setRawViewMode] = useState<ViewMode>('perspective');
  const [pointSize, setPointSize] = useState<number>(0.18);
  const [isGrayscale, setIsGrayscale] = useState<boolean>(false);
  const [selectedInstanceId, setSelectedInstanceId] = useState<string | null>(null);

  // Responsive Side-Panel Layout & Dynamic Map Expansion State
  const [isLeftCollapsed, setIsLeftCollapsed] = useState<boolean>(false);
  const [isRightCollapsed, setIsRightCollapsed] = useState<boolean>(false);
  const [isDesktop, setIsDesktop] = useState<boolean>(() =>
    typeof window !== 'undefined' ? window.innerWidth >= 1024 : true
  );

  useEffect(() => {
    const media = window.matchMedia('(min-width: 1024px)');
    const updateMatch = (e: MediaQueryListEvent | MediaQueryList) => setIsDesktop(e.matches);
    updateMatch(media);
    media.addEventListener('change', updateMatch);
    return () => media.removeEventListener('change', updateMatch);
  }, []);

  // Fire window resize events after side-panel layout transitions finish
  useEffect(() => {
    // 320ms matches the duration-300 CSS transition
    const timer = setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
    }, 320);
    return () => clearTimeout(timer);
  }, [isLeftCollapsed, isRightCollapsed]);

  // Semantic Category Filter Toggles
  const [categoryFilters, setCategoryFilters] = useState<Record<string, boolean>>({
    drivable: true,
    nondrivable: true,
    staticObstacle: true,
    dynamicObstacle: true,
    vegetation: true,
    infrastructure: true,
    unknown: true,
  });

  // Pipeline Sequential Stages (for single-frame mode)
  const [stages, setStages] = useState<PipelineStages>({
    upload: 'waiting',
    rawRender: 'waiting',
    salsanext: 'waiting',
    terrain: 'waiting',
    objects: 'waiting',
    adaptiveGrid: 'waiting',
  });

  // Action state for "Run Object Detection"
  const [detectionActionState, setDetectionActionState] = useState<ObjectDetectionActionState>('idle');

  // Pipeline Gate Status State
  const [pipelineGateStatus, setPipelineGateStatus] = useState<PipelineStatusResponse | null>(null);
  const [isGateLoading, setIsGateLoading] = useState<boolean>(false);
  const [isProcessingGatePipeline, setIsProcessingGatePipeline] = useState<boolean>(false);

  // Pipeline Event Logger (no-op helper)
  const addEvent = useCallback((_title: string, _detail?: string, _status: string = 'complete') => {}, []);

  // Fetch Pipeline Gate Status
  const fetchPipelineGateStatus = useCallback(async (frameId?: string) => {
    const fid = frameId || selectedFrameId || '000000';
    setIsGateLoading(true);
    try {
      const res = await PipelineService.getPipelineStatus(fid);
      if (res.success && res.data) {
        setPipelineGateStatus(res.data);
      }
    } catch (err) {
      console.warn('Failed to fetch pipeline gate status:', err);
    } finally {
      setIsGateLoading(false);
    }
  }, [selectedFrameId]);

  // Run End-to-End Perception Pipeline Gate
  const handleRunPipelineGate = async () => {
    const fid = selectedFrameId || '000000';
    setIsProcessingGatePipeline(true);
    try {
      const res = await PipelineService.processPipeline(fid);
      if (res.success && res.data) {
        setPipelineGateStatus(res.data.pipeline_status);
        // Refresh frame, terrain, objects, and map
        const [frameRes, semRes, terRes, detRes, trackRes, mapRes] = await Promise.allSettled([
          lidarService.getFrame(fid),
          semanticService.getSemanticFrame(fid),
          terrainService.analyzeTerrain({ frame_id: fid }),
          objectService.detectObjects({ frame_id: fid }),
          trackingService.updateTracks({ frame_id: fid }),
          mapService.getMapCells(fid, { limit: 5000 }),
        ]);

        if (frameRes.status === 'fulfilled' && frameRes.value.success && frameRes.value.data) {
          setActiveLidarFrame(frameRes.value.data);
        }
        if (semRes.status === 'fulfilled' && semRes.value.success && semRes.value.data) {
          setActiveSemanticFrame(semRes.value.data);
        }
        if (terRes.status === 'fulfilled' && terRes.value.success && terRes.value.data) {
          setActiveTerrainResponse(terRes.value.data);
        }
        if (detRes.status === 'fulfilled' && detRes.value.success && detRes.value.data) {
          setActiveObjectDetection(detRes.value.data);
        }
        if (trackRes.status === 'fulfilled' && trackRes.value.success && trackRes.value.data) {
          setActiveTracks(trackRes.value.data.tracks);
        }
        if (mapRes.status === 'fulfilled' && mapRes.value.success && mapRes.value.data && mapRes.value.data.length > 0) {
          const gridCells = mapService.convertAdaptiveCellsToGridCells(mapRes.value.data);
          setCells(gridCells);
        }
      }
    } catch (err) {
      console.error('Pipeline process execution error:', err);
    } finally {
      setIsProcessingGatePipeline(false);
    }
  };

  useEffect(() => {
    if (selectedFrameId) {
      fetchPipelineGateStatus(selectedFrameId);
    }
  }, [selectedFrameId, fetchPipelineGateStatus]);

  // Synchronize rendered cells from active 2.5D map or terrain analysis or mock
  useEffect(() => {
    let isMounted = true;

    // 1. Real 2.5D Adaptive Grid Map response
    if (activeMapResponse && activeMapResponse.cells_sample && activeMapResponse.cells_sample.length > 0) {
      const adaptiveGridCells = mapService.convertAdaptiveCellsToGridCells(activeMapResponse.cells_sample);
      setCells(adaptiveGridCells);
      setIsLoading(false);
      return;
    }

    // 2. Real backend terrain result
    if (activeTerrainResponse && activeTerrainResponse.cells && activeTerrainResponse.cells.length > 0) {
      const realGridCells = mapService.convertTerrainCellsToGridCells(activeTerrainResponse.cells);
      setCells(realGridCells);
      setIsLoading(false);
      return;
    }

    // 3. Fallback: Load mock map cells only on initial empty state
    if (!activeMapResponse && !activeTerrainResponse) {
      const loadCells = async () => {
        try {
          const response = await mapService.getLocalMapCells();
          if (isMounted && response.data) {
            setCells(response.data);
          }
        } catch (err) {
          console.warn('Error loading map cells:', err);
        }
      };
      loadCells();
    }

    return () => {
      isMounted = false;
    };
  }, [activeMapResponse, activeTerrainResponse]);

  // Handle stream frame arrival from WebSocket or Step API
  const handleFramePayload = useCallback(
    (payload: ReplayFrameStreamPayload) => {
      // 1. Update active frame ID and raw points
      setSelectedFrameId(payload.frame_filename);
      const adaptedRawPoints: Point3D[] = (payload.points_sample || []).map((p) => ({
        x: p.x,
        y: p.y,
        z: p.z,
        intensity: p.intensity,
        semanticClass: p.project_category === 'drivable' ? 'drivable' : p.project_category === 'vegetation' ? 'vegetation' : p.project_category === 'dynamic_object' ? 'dynamicObstacle' : p.project_category === 'static_obstacle' ? 'staticObstacle' : p.project_category === 'infrastructure' ? 'infrastructure' : 'unknown',
      }));

      setActiveLidarFrame({
        frameId: payload.frame_filename,
        sequenceId: payload.sequence_name,
        points: adaptedRawPoints,
        metadata: {
          scanSource: `Replay Sequence "${payload.sequence_name}"`,
          timestamp: payload.timestamp || `Frame #${payload.frame_index}`,
          calibration: 'Sensor Coordinate Frame',
          labels: payload.data_mode === 'precomputed_labels' ? 'Fast-FRNet Neural Labels' : 'Live Inference',
          vehiclePose: 'Sensor Origin Body Frame (Local Only)',
          sensorConfiguration: 'HDL-64E Float32 (x,y,z,i)',
          beamCount: `${payload.point_count} pts total`,
          horizontalFov: '360 deg Continuous Sweep',
          rangeCapability: '80m Radial Radius',
        },
      });

      // 2. Update semantic frame annotations
      setActiveSemanticFrame({
        frame_id: payload.frame_filename,
        point_count: payload.point_count,
        class_counts: {},
        project_category_counts: payload.category_distribution || {},
        sample_labeled_points: (payload.points_sample || []).map((p, idx) => ({
          x: p.x,
          y: p.y,
          z: p.z,
          intensity: p.intensity,
          raw_label_id: 0,
          semantic_class: p.semantic_class,
          project_category: p.project_category,
          instance_id: 0,
        })),
        model_provider_status: 'external_predictions_loaded',
        created_at: payload.timestamp || new Date().toISOString(),
      });

      // 3. Update terrain analysis response
      if (payload.terrain_result) {
        setActiveTerrainResponse(payload.terrain_result);
      }

      // 4. Update detected objects
      if (payload.objects_result) {
        setActiveObjectDetection(payload.objects_result);
      }

      // 5. Update multi-frame Kalman tracks
      if (payload.tracking_result && payload.tracking_result.tracks) {
        setActiveTracks(payload.tracking_result.tracks);
      }

      // 6. Update 2.5D Adaptive Grid
      if (payload.map_result) {
        setActiveMapResponse(payload.map_result);
        if (payload.map_result.cells_sample && payload.map_result.cells_sample.length > 0) {
          const gridCells = mapService.convertAdaptiveCellsToGridCells(payload.map_result.cells_sample);
          setCells(gridCells);
        }
      }

      // 7. Dynamic Real-Time Streaming FPS Calculation
      const now = performance.now();
      const arrivals = frameArrivalsRef.current;
      arrivals.push(now);

      // Keep recent sliding buffer (up to 12 frames or arrivals within last 5 seconds)
      while (arrivals.length > 12 || (arrivals.length > 2 && arrivals[0] < now - 5000)) {
        arrivals.shift();
      }

      const computeFps =
        payload.performance?.actual_fps && payload.performance.actual_fps > 0
          ? payload.performance.actual_fps
          : payload.processing_time_ms > 0
          ? 1000.0 / payload.processing_time_ms
          : null;

      if (arrivals.length > 1) {
        const elapsedSec = (now - arrivals[0]) / 1000;
        if (elapsedSec > 0) {
          const measuredDeliveryFps = (arrivals.length - 1) / elapsedSec;
          setRealtimeFps(Number(measuredDeliveryFps.toFixed(1)));
        } else if (computeFps !== null && computeFps > 0) {
          setRealtimeFps(Number(computeFps.toFixed(1)));
        }
      } else if (computeFps !== null && computeFps > 0) {
        setRealtimeFps(Number(computeFps.toFixed(1)));
      }

      // 8. Update Replay state counters
      setReplayFrameIndex(payload.frame_index);
      setReplayTotalFrames(payload.total_frames);
      if (payload.semantic_source) {
        setReplaySemanticSource(payload.semantic_source);
      }
      if (payload.sequence_name) {
        setReplaySequenceName(payload.sequence_name);
      }
      
      // CRITICAL FIX: Do NOT overwrite active playing state with payload.state (saved as 'ready' in precomputed JSON files).
      // Only transition state if the frame sequence is completed or failed.
      if (payload.state === 'failed') {
        setReplayPlaybackState('failed');
      } else if (payload.frame_index >= payload.total_frames - 1) {
        setReplayPlaybackState('completed');
      }
      setReplayProcessingTimeMs(payload.processing_time_ms);

      // 9. Update live frame performance telemetry
      if (payload.performance) {
        setActivePerformance(payload.performance);
      }

      // Throttled event logging for high-speed 60 FPS playback (logs once every 60 frames or at key milestones)
      if (
        payload.frame_index === 0 ||
        payload.frame_index === payload.total_frames - 1 ||
        (payload.frame_index + 1) % 60 === 0 ||
        payload.state !== 'playing'
      ) {
        addEvent(
          `Frame ${payload.frame_index + 1}/${payload.total_frames} Streamed`,
          `${payload.point_count} pts • ${payload.tracking_result?.tracks?.length || 0} active tracks • Stream Live`,
          'complete'
        );
      }
    },
    [addEvent, setActiveLidarFrame, setActiveSemanticFrame, setActiveTerrainResponse, setActiveObjectDetection, setActiveTracks, setActiveMapResponse, setSelectedFrameId, setActivePerformance]
  );

  // Handle Sequence ZIP Upload - Always executes real computing (live Fast-FRNet + terrain + clustering + adaptive grid)
  const handleUploadSequenceZip = async (file: File) => {
    setActivePerformance(null);
    setIsUploadingSequence(true);
    setConsoleMode('sequence-replay');
    setHasUploaded(true);
    addEvent(`Ingesting Sequence ZIP "${file.name}"`, `${(file.size / (1024 * 1024)).toFixed(2)} MB package (Real Fast-FRNet Computing Pipeline)`, 'processing');

    try {
      const cleanSessionId = file.name.replace(/\.[^/.]+$/, '').toLowerCase().replace(/[^a-z0-9_-]/g, '_');
      // Manual uploads NEVER use precomputed values; always enforce real computing
      const uploadRes = await replayService.uploadSequence(file, cleanSessionId, 'live_processing');
      if (uploadRes.success && uploadRes.data) {
        const data = uploadRes.data;
        setReplaySessionId(data.session_id);
        setReplaySequenceName(data.sequence_name);
        setReplayTotalFrames(data.total_frames);
        setReplayFrameIndex(0);
        setReplayPlaybackState('ready');
        setReplayDataMode('live_inference');
        setReplaySemanticSource('LIVE FAST-FRNET');

        addEvent(
          `Sequence Archive Ingested`,
          `Extracted and indexed ${data.total_frames.toLocaleString()} consecutive scans for Real Live Computing.`,
          'complete'
        );

        // Connect WebSocket Stream
        if (wsStreamRef.current) {
          wsStreamRef.current.close();
        }
        wsStreamRef.current = replayService.connectWebSocketStream(
          data.session_id,
          (frame) => handleFramePayload(frame),
          (status, info) => {
            setReplayPlaybackState(status.state);
            setReplayFrameIndex(status.current_frame_index);
            setReplayTotalFrames(status.total_frames);
            if (info.resync) {
              addEvent('Replay WebSocket Resynced', `Recovered to frame ${status.current_frame_index + 1} of ${status.total_frames}`, 'info');
            }
          },
          (err) => {
            console.warn('Replay WebSocket error:', err);
          },
          (connected) => setIsWebSocketConnected(connected)
        );

        // Precompute cache is strictly for demo mode; disable any polling for manual uploads
        if (precomputePollTimerRef.current) {
          clearInterval(precomputePollTimerRef.current);
          precomputePollTimerRef.current = null;
        }
        setPrecomputeStatus(null);

        // Fetch frame 0 immediately
        const f0 = await replayService.getNextFrame(data.session_id);
        if (f0.success && f0.data) {
          handleFramePayload(f0.data);
        } else {
          setReplayPlaybackState('failed');
          const errDetail = f0.message || 'Fast-FRNet live inference failed on frame 0';
          addEvent('Live Inference Failed', errDetail, 'failed');
          console.error('Sequence frame 0 live inference failed:', errDetail);
        }
      } else {
        addEvent(`Sequence Ingestion Failed`, uploadRes.message || 'Invalid ZIP structure', 'failed');
      }
    } catch (err) {
      addEvent(`Sequence Upload Error`, String(err), 'failed');
    } finally {
      setIsUploadingSequence(false);
    }
  };

  // Handle Ingest Local Sequence Archive path directly
  const handleIngestLocalArchive = async (customPath?: string) => {
    const targetPath = customPath || localArchivePath;
    if (!targetPath.trim()) return;
    setIsIngestingArchive(true);
    setConsoleMode('sequence-replay');
    setHasUploaded(true);
    addEvent(`Ingesting Local Sequence Archive`, targetPath, 'processing');

    try {
      // Local archive ingestion also strictly runs live computing
      const res = await replayService.ingestArchive(targetPath.trim(), undefined, 'live_processing');
      if (res.success && res.data) {
        const data = res.data;
        setReplaySessionId(data.session_id);
        setReplaySequenceName(data.sequence_name);
        setReplayTotalFrames(data.total_frames);
        setReplayFrameIndex(0);
        setReplayPlaybackState('ready');
        setReplayDataMode('live_inference');
        setReplaySemanticSource('LIVE FAST-FRNET');

        addEvent(
          `Sequence Archive Ingested`,
          `Indexed ${data.total_frames.toLocaleString()} consecutive scans for Real Live Computing.`,
          'complete'
        );

        if (wsStreamRef.current) {
          wsStreamRef.current.close();
        }
        wsStreamRef.current = replayService.connectWebSocketStream(
          data.session_id,
          (frame) => handleFramePayload(frame),
          (status, info) => {
            setReplayPlaybackState(status.state);
            setReplayFrameIndex(status.current_frame_index);
            setReplayTotalFrames(status.total_frames);
            if (info.resync) {
              addEvent('Replay WebSocket Resynced', `Recovered to frame ${status.current_frame_index + 1} of ${status.total_frames}`, 'info');
            }
          },
          (err) => console.warn('Replay WebSocket error:', err),
          (connected) => setIsWebSocketConnected(connected)
        );

        if (precomputePollTimerRef.current) {
          clearInterval(precomputePollTimerRef.current);
          precomputePollTimerRef.current = null;
        }
        setPrecomputeStatus(null);

        const f0 = await replayService.getNextFrame(data.session_id);
        if (f0.success && f0.data) {
          handleFramePayload(f0.data);
        }
      } else {
        addEvent(`Archive Ingestion Failed`, res.message || 'Invalid archive path', 'failed');
      }
    } catch (err: any) {
      addEvent(`Archive Ingestion Error`, err?.message || String(err), 'failed');
    } finally {
      setIsIngestingArchive(false);
    }
  };

  // Playback Control Handlers
  const handlePlayReplay = () => {
    if (!replaySessionId) return;
    setReplayPlaybackState('playing');
    frameArrivalsRef.current = [];
    if (wsStreamRef.current) {
      wsStreamRef.current.send({ action: 'play', fps: replayFps });
    } else {
      replayService.startReplay(replaySessionId, replayFps);
    }
    addEvent('Replay Playback Started', `Streaming live LiDAR feed at ${replayFps} FPS`, 'complete');
  };

  const handlePauseReplay = () => {
    if (!replaySessionId) return;
    setReplayPlaybackState('paused');
    frameArrivalsRef.current = [];
    if (wsStreamRef.current) {
      wsStreamRef.current.send({ action: 'pause' });
    } else {
      replayService.pauseReplay(replaySessionId);
    }
    addEvent('Replay Playback Paused', `Feed paused at frame ${replayFrameIndex + 1}`, 'complete');
  };

  const handleStopReplay = () => {
    if (!replaySessionId) return;
    setReplayPlaybackState('ready');
    setReplayFrameIndex(0);
    frameArrivalsRef.current = [];
    setRealtimeFps(0.0);
    if (wsStreamRef.current) {
      wsStreamRef.current.send({ action: 'stop' });
    } else {
      replayService.stopReplay(replaySessionId);
    }
    addEvent('Replay Rewound to Frame 0', 'Tracker reset to initial state', 'complete');
  };

  const handleSeekReplay = (targetIndex: number) => {
    if (!replaySessionId) return;
    setReplayFrameIndex(targetIndex);
    if (wsStreamRef.current) {
      wsStreamRef.current.send({ action: 'seek', frame_index: targetIndex });
    } else {
      replayService.seekReplay(replaySessionId, targetIndex);
    }
  };

  const handleTriggerPrecompute = async () => {
    if (!replaySessionId) return;
    try {
      const trig = await replayService.triggerPrecompute(replaySessionId);
      if (trig.success && trig.data) {
        setPrecomputeStatus({
          isRunning: true,
          isComplete: false,
          percentComplete: trig.data.percent_complete || 0,
          processedFrames: trig.data.processed_frames || 0,
          totalFrames: trig.data.total_frames || replayTotalFrames,
          etaSeconds: trig.data.eta_seconds || 0,
          currentStage: trig.data.current_stage || 'precomputing',
          failedCount: trig.data.failed_count || 0,
          errorMessage: null,
        });
        if (precomputePollTimerRef.current) clearInterval(precomputePollTimerRef.current);
        precomputePollTimerRef.current = setInterval(async () => {
          try {
            const st = await replayService.getPrecomputeStatus(replaySessionId!);
            if (st.success && st.data) {
              setPrecomputeStatus({
                isRunning: !!st.data.is_running,
                isComplete: !!st.data.is_complete,
                percentComplete: st.data.percent_complete || 0,
                processedFrames: st.data.processed_frames || 0,
                totalFrames: st.data.total_frames || replayTotalFrames,
                etaSeconds: st.data.eta_seconds || 0,
                currentStage: st.data.current_stage,
                failedCount: st.data.failed_count,
                errorMessage: st.data.error_message || null,
              });
              if (st.data.is_complete && precomputePollTimerRef.current) {
                clearInterval(precomputePollTimerRef.current);
                precomputePollTimerRef.current = null;
              }
            }
          } catch {
            if (precomputePollTimerRef.current) {
              clearInterval(precomputePollTimerRef.current);
              precomputePollTimerRef.current = null;
            }
          }
        }, 1000);
      }
    } catch (e) {
      console.warn('Precompute trigger failed:', e);
    }
  };

  const handleCancelPrecompute = async () => {
    if (!replaySessionId) return;
    try {
      if (precomputePollTimerRef.current) {
        clearInterval(precomputePollTimerRef.current);
        precomputePollTimerRef.current = null;
      }
      const res = await replayService.cancelPrecompute(replaySessionId);
      if (res.success && res.data?.cancelled) {
        setPrecomputeStatus((prev) =>
          prev
            ? {
                ...prev,
                isRunning: false,
                currentStage: 'cancelled',
                errorMessage: 'Cancelled by user',
              }
            : null
        );
        addEvent('Processing Cancelled', 'Sequence processing stopped by user', 'info');
      }
    } catch (e) {
      console.warn('Precompute cancel failed:', e);
    }
  };

  // Resolve the selected cell against the latest frame.  The selected cell
  // keeps its `id` even if the downsampled cells_sample of the current frame
  // does not include it.  In that case we re-fetch the full cell from the
  // session-global fused map (the backend keeps every cell in a long-lived
  // map keyed by frame index), so the Cell Inspector stays accurate.
  useEffect(() => {
    if (!selectedCell || !replaySessionIdRef.current) return;
    if (cells.find((c) => c.id === selectedCell.id)) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await replayService.getGlobalCell(replaySessionIdRef.current!, selectedCell.id);
        if (cancelled || !res.success || !res.data) return;
        const fresh: GridCell = {
          ...selectedCell,
          worldX: res.data.world_x,
          worldY: res.data.world_y,
          size: res.data.size_m,
          zone: res.data.level as 'near' | 'mid' | 'far',
          traversability: res.data.traversability_state,
          backendAdaptiveCell: res.data,
        };
        setSelectedCell(fresh);
      } catch {
        // backend offline or cell evicted; keep the last-known data
      }
    })();
    return () => { cancelled = true; };
  }, [cells, selectedCell]);

  const handleStepBackward = () => {
    if (replayFrameIndex > 0) {
      handleSeekReplay(replayFrameIndex - 1);
    }
  };

  const handleStepForward = async () => {
    if (replaySessionId && replayFrameIndex < replayTotalFrames - 1) {
      if (wsStreamRef.current) {
        wsStreamRef.current.send({ action: 'seek', frame_index: replayFrameIndex + 1 });
      } else {
        const nextRes = await replayService.getNextFrame(replaySessionId);
        if (nextRes.success && nextRes.data) {
          handleFramePayload(nextRes.data);
        }
      }
    }
  };

  const handleSetReplayFps = (fps: number) => {
    setReplayFps(fps);
    if (wsStreamRef.current) {
      wsStreamRef.current.send({ action: 'set_fps', fps });
    }
  };

  // Clean up WebSocket stream on unmount
  useEffect(() => {
    return () => {
      if (wsStreamRef.current) {
        wsStreamRef.current.close();
      }
      if (precomputePollTimerRef.current) {
        clearInterval(precomputePollTimerRef.current);
        precomputePollTimerRef.current = null;
      }
    };
  }, []);

  // Reset precompute status when the active session changes
  useEffect(() => {
    if (!replaySessionId) {
      setPrecomputeStatus(null);
      setIsWebSocketConnected(false);
    }
  }, [replaySessionId]);

  // Handle uploaded LiDAR scan (.bin, .pcd, .xyz, .ply) - strictly executes live Fast-FRNet neural inference
  const handleFileUpload = async (file: File) => {
    const uploadId = `upload_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
    activeUploadIdRef.current = uploadId;

    setConsoleMode('single-frame');
    setHasUploaded(true);

    // CRITICAL: Immediately purge old scene data on new upload
    setActiveLidarFrame(null);
    setActiveSemanticFrame(null);
    setActiveObjectDetection(null);
    setActiveTerrainResponse(null);
    setActiveMapResponse(null);
    setActiveInferenceResults(null);
    setActiveTracks([]);
    setCells([]);
    setActivePerformance(null);

    setStages({
      upload: 'processing',
      rawRender: 'waiting',
      salsanext: 'waiting',
      terrain: 'waiting',
      objects: 'waiting',
      adaptiveGrid: 'waiting',
    });

    addEvent(
      `1. Ingesting Point Cloud "${file.name}"`,
      `${(file.size / 1024).toFixed(1)} KB • Initializing live Fast-FRNet neural pipeline`,
      'processing'
    );

    try {
      setStages((s) => ({ ...s, upload: 'complete', rawRender: 'processing', salsanext: 'processing' }));
      addEvent('2. Live Neural Inference', 'Executing Fast-FRNet forward pass (live inference)...', 'processing');

      // Execute unified live upload inference route
      const uploadRes = await semanticService.liveUpload(file, uploadId);

      // Stale response check: discard if a newer upload was started
      if (activeUploadIdRef.current !== uploadId) {
        console.log('Discarding stale upload response for', uploadId);
        return;
      }

      if (!uploadRes.success || !uploadRes.data) {
        throw new Error(uploadRes.message || 'Live neural inference failed on backend');
      }

      const data = uploadRes.data;
      if (data.precomputed) {
        throw new Error('Live computing required: Backend returned cached data for an uploaded file.');
      }

      setSelectedFrameId(data.frame_id);

      // 1. Raw & Semantic 3D points
      const adaptedPoints: Point3D[] = (data.sample_predictions || []).map((p) => ({
        x: p.x,
        y: p.y,
        z: p.z,
        intensity: p.intensity,
        semanticClass:
          p.project_category === 'drivable'
            ? 'drivable'
            : p.project_category === 'vegetation'
            ? 'vegetation'
            : p.project_category === 'dynamic_object'
            ? 'dynamicObstacle'
            : p.project_category === 'static_obstacle'
            ? 'staticObstacle'
            : p.project_category === 'infrastructure'
            ? 'infrastructure'
            : 'unknown',
      }));

      setActiveLidarFrame({
        frameId: data.frame_id,
        sequenceId: 'live_upload',
        points: adaptedPoints,
        metadata: {
          scanSource: `Live Upload: ${data.filename}`,
          timestamp: data.timestamp,
          calibration: 'Sensor Coordinate Frame',
          labels: `LIVE FAST-FRNET (${data.model_type.toUpperCase()} / ${data.detected_domain})`,
          vehiclePose: 'Sensor Origin Body Frame',
          sensorConfiguration: `${data.point_count} Points (${data.device_used.toUpperCase()})`,
          beamCount: `${data.point_count} pts live`,
          horizontalFov: '360 deg Continuous Sweep',
          rangeCapability: 'Adaptive Range',
        },
      });

      // 2. Semantic Frame
      setActiveSemanticFrame({
        frame_id: data.frame_id,
        point_count: data.point_count,
        class_counts: data.semantic?.class_counts || {},
        project_category_counts: data.semantic?.project_category_counts || {},
        sample_labeled_points: (data.sample_predictions || []).map((p) => ({
          x: p.x,
          y: p.y,
          z: p.z,
          intensity: p.intensity,
          raw_label_id: p.raw_label_id,
          semantic_class: p.semantic_class,
          project_category: p.project_category as any,
          instance_id: 0,
        })),
        model_provider_status: 'external_predictions_loaded',
        created_at: data.timestamp,
      });

      // 3. Inference Results
      setActiveInferenceResults({
        job_id: data.upload_id,
        frame_id: data.frame_id,
        status: 'complete',
        point_count: data.point_count,
        class_counts: data.semantic?.class_counts || {},
        project_category_counts: data.semantic?.project_category_counts || {},
        sample_predictions: (data.sample_predictions || []).map((p) => ({
          x: p.x,
          y: p.y,
          z: p.z,
          intensity: p.intensity,
          raw_label_id: p.raw_label_id,
          semantic_class: p.semantic_class,
          project_category: p.project_category as any,
        })),
        device_used: data.device_used,
        created_at: data.timestamp,
      });

      // 4. Object Detection
      if (data.objects) {
        setActiveObjectDetection(data.objects);
      }

      // 5. Terrain Analysis
      if (data.terrain) {
        setActiveTerrainResponse(data.terrain);
      }

      // 6. Adaptive Grid Map
      if (data.grid) {
        setActiveMapResponse(data.grid);
        if (data.grid.cells_sample && data.grid.cells_sample.length > 0) {
          setCells(mapService.convertAdaptiveCellsToGridCells(data.grid.cells_sample));
        }
      }

      // 7. Live FPS and latency
      const perfTotalMs = data.performance?.total_pipeline_ms ?? data.performance?.total_ms ?? 0;
      const measuredFps =
        data.performance?.fps ||
        (perfTotalMs > 0 ? Number((1000.0 / perfTotalMs).toFixed(1)) : 0.0);
      setRealtimeFps(measuredFps);

      if (data.performance) {
        const perf = data.performance;
        setActivePerformance({
          frame_id: data.frame_id,
          timestamp: data.timestamp,
          device_used: data.device_used,
          model_name: `Fast-FRNet (${(data.model_type || 'semantickitti').toUpperCase()})`,
          timings: {
            lidar_preprocessing_ms: perf.preprocessing_ms ?? 0,
            salsanext_inference_ms: perf.inference_ms ?? 0,
            terrain_analysis_ms: perf.terrain_ms ?? 0,
            object_detection_tracking_ms: perf.clustering_ms ?? 0,
            adaptive_grid_ms: perf.grid_ms ?? 0,
            total_latency_ms: perf.total_ms ?? perfTotalMs,
            preprocessing_ms: perf.preprocessing_ms ?? 0,
            inference_ms: perf.inference_ms ?? 0,
            terrain_ms: perf.terrain_ms ?? 0,
            clustering_ms: perf.clustering_ms ?? 0,
            grid_ms: perf.grid_ms ?? 0,
            total_ms: perf.total_ms ?? perfTotalMs,
          },
          actual_fps: measuredFps,
          resource_metrics: perf.resource_metrics || null,
          accuracy_metrics: null,
        });
      }

      // 8. Stages completion
      setStages({
        upload: 'complete',
        rawRender: 'complete',
        salsanext: 'complete',
        terrain: 'complete',
        objects: 'complete',
        adaptiveGrid: 'complete',
      });

      addEvent(
        'Live Perception Complete',
        `Model: ${data.model} (${data.detected_domain}) • ${data.point_count.toLocaleString()} pts • Latency: ${perfTotalMs.toFixed(1)}ms (${measuredFps} FPS)`,
        'complete'
      );
    } catch (err: any) {
      setStages({
        upload: 'failed',
        rawRender: 'waiting',
        salsanext: 'failed',
        terrain: 'waiting',
        objects: 'waiting',
        adaptiveGrid: 'waiting',
      });
      const errMsg = err?.message || String(err);
      addEvent('LIVE INFERENCE FAILED', errMsg, 'failed');
      console.error('LIVE INFERENCE FAILED:', err);
    }
  };

  // Handler for standalone "Run Object Detection" action button
  const handleRunObjectDetectionAction = async () => {
    const frameId = selectedFrameId || activeLidarFrame?.frameId;
    if (!frameId) {
      addEvent('Object Detection Error', 'No active LiDAR frame selected.', 'failed');
      setDetectionActionState('error');
      return;
    }

    // 1. Check labels availability
    setDetectionActionState('checking_labels');
    addEvent('Object Detection Triggered', `Checking semantic labels for frame "${frameId}"...`, 'processing');

    const hasLabels = (activeSemanticFrame && activeSemanticFrame.sample_labeled_points.length > 0) || (replayDataMode === 'precomputed_labels');
    if (!hasLabels) {
      setDetectionActionState('error');
      addEvent('Object Detection Halted', 'Matching Fast-FRNet prediction file is required for object detection.', 'failed');
      return;
    }

    // 2. Class mapping complete -> Clustering
    setDetectionActionState('mapping_classes');
    await new Promise((r) => setTimeout(r, 120));
    setDetectionActionState('clustering');
    addEvent('DBSCAN Clustering in Progress', 'Grouping semantic points into 3D geometric object instances...', 'processing');

    try {
      const detRes = await objectService.detectObjects({ frame_id: frameId });
      if (detRes.success && detRes.data) {
        setActiveObjectDetection(detRes.data);

        const dynamicCount = detRes.data.dynamic_instances_count;
        const totalInstances = detRes.data.total_instances;

        if (totalInstances > 0) {
          setDetectionActionState('instances_found');
          addEvent(
            'Object Detection Complete',
            `${totalInstances} object instances detected (${dynamicCount} dynamic, ${detRes.data.static_instances_count} static)`,
            'complete'
          );
        } else {
          // Check if there are dynamic points in semantic frame
          const dynPoints = activeSemanticFrame?.project_category_counts?.['dynamic_object'] || 0;
          if (dynPoints === 0) {
            setDetectionActionState('no_dynamic_points');
            addEvent('Clustering Result', 'No dynamic-object points found in this frame.', 'complete');
          } else {
            setDetectionActionState('no_valid_clusters');
            addEvent('Clustering Result', 'No valid clusters formed from points.', 'complete');
          }
        }

        // Call tracking update if consecutive frames
        const trackRes = await trackingService.updateTracks({ frame_id: frameId });
        if (trackRes.success && trackRes.data) {
          setActiveTracks(trackRes.data.tracks);
        }
      } else {
        setDetectionActionState('error');
        addEvent('Object Detection Failed', detRes.message || 'Backend clustering error', 'failed');
      }
    } catch (err) {
      setDetectionActionState('error');
      addEvent('Object Detection Error', String(err), 'failed');
    }
  };

  // Run Individual Stage (Single frame workflow)
  const handleRunStage = async (stageName: keyof PipelineStages) => {
    const frameId = selectedFrameId || 'FRAME_0001 (Mock)';

    if (stageName === 'rawRender') {
      setStages((s) => ({ ...s, rawRender: 'processing' }));
      const frameRes = await lidarService.getFrame(frameId);
      if (frameRes.success && frameRes.data) {
        setActiveLidarFrame(frameRes.data);
        setStages((s) => ({ ...s, rawRender: 'complete' }));
        addEvent('Raw LiDAR Loaded', `${frameRes.data.points.length} points active`, 'complete');
      } else {
        setStages((s) => ({ ...s, rawRender: 'failed' }));
      }
    } else if (stageName === 'salsanext') {
      setStages((s) => ({ ...s, salsanext: 'processing' }));
      addEvent('Fast-FRNet Inference Triggered', `Processing frustum features on ${activeInferenceStatus?.device || 'CPU'}`, 'processing');
      try {
        const infRes = await semanticService.runInference(undefined, frameId);
        if (infRes.success && infRes.data) {
          const resultsRes = await semanticService.getInferenceResults(infRes.data.job_id);
          if (resultsRes.success && resultsRes.data) {
            setActiveInferenceResults(resultsRes.data);
            const semRes = await semanticService.getSemanticFrame(frameId);
            if (semRes.success && semRes.data) {
              setActiveSemanticFrame(semRes.data);
            }
            setStages((s) => ({ ...s, salsanext: 'complete' }));
            addEvent('Fast-FRNet Segmentation Complete', `Point-wise 20-class classification ready`, 'complete');
          }
        }
      } catch {
        setStages((s) => ({ ...s, salsanext: 'failed' }));
        addEvent('Fast-FRNet Inference Failed', 'Check model checkpoint availability', 'failed');
      }
    } else if (stageName === 'terrain') {
      setStages((s) => ({ ...s, terrain: 'processing' }));
      addEvent('Geometric Terrain Analysis Triggered', `Analyzing elevation, roughness, slope, and curbs`, 'processing');
      try {
        const terRes = await terrainService.analyzeTerrain({ frame_id: frameId });
        if (terRes.success && terRes.data) {
          setActiveTerrainResponse(terRes.data);
          setStages((s) => ({ ...s, terrain: 'complete' }));
          addEvent('Terrain Analysis Complete', `${terRes.data.cells.length} elevation cells computed`, 'complete');
        }
      } catch {
        setStages((s) => ({ ...s, terrain: 'failed' }));
      }
    } else if (stageName === 'objects') {
      setStages((s) => ({ ...s, objects: 'processing' }));
      addEvent('DBSCAN Clustering & MOT Triggered', 'Clustering instances & updating Kalman tracks', 'processing');
      try {
        const detRes = await objectService.detectObjects({ frame_id: frameId });
        if (detRes.success && detRes.data) {
          setActiveObjectDetection(detRes.data);
          const trackRes = await trackingService.updateTracks({ frame_id: frameId });
          if (trackRes.success && trackRes.data) {
            setActiveTracks(trackRes.data.tracks);
          }
          setStages((s) => ({ ...s, objects: 'complete' }));
          addEvent('Object Detection & Tracking Updated', `${detRes.data.total_instances} instances clustered`, 'complete');
        }
      } catch {
        setStages((s) => ({ ...s, objects: 'failed' }));
      }
    } else if (stageName === 'adaptiveGrid') {
      setStages((s) => ({ ...s, adaptiveGrid: 'processing' }));
      addEvent('2.5D Adaptive Grid Synthesis Triggered', 'Fusing point cloud into hierarchical foveated cells', 'processing');
      try {
        const mapRes = await mapService.updateMap('live_fovea_map_01', { frame_id: frameId });
        if (mapRes.success && mapRes.data) {
          const cellsRes = await mapService.getMapCells('live_fovea_map_01', { limit: 5000 });
          if (cellsRes.success && cellsRes.data && cellsRes.data.length > 0) {
            mapRes.data.cells_sample = cellsRes.data;
          }
          setActiveMapResponse(mapRes.data);
          setStages((s) => ({ ...s, adaptiveGrid: 'complete' }));
          addEvent('2.5D Adaptive Grid Fused', `${mapRes.data.updated_cell_count} elevation cells generated`, 'complete');
        }
      } catch {
        setStages((s) => ({ ...s, adaptiveGrid: 'failed' }));
      }
    }
  };

  // Run Complete Sequential Perception Pipeline
  const handleRunFullPipeline = async () => {
    setIsPipelineRunning(true);
    const frameId = selectedFrameId || 'FRAME_0001 (Mock)';

    try {
      // 1. Raw Render
      setStages((s) => ({ ...s, rawRender: 'processing' }));
      const frameRes = await lidarService.getFrame(frameId);
      if (frameRes.success && frameRes.data) {
        setActiveLidarFrame(frameRes.data);
        setStages((s) => ({ ...s, rawRender: 'complete' }));
        addEvent('Raw LiDAR Loaded', `${frameRes.data.points.length} points active`, 'complete');
      }

      // 2. Fast-FRNet Semantic Segmentation
      setStages((s) => ({ ...s, salsanext: 'processing' }));
      try {
        const infRes = await semanticService.runInference(undefined, frameId);
        if (infRes.success && infRes.data) {
          const resultsRes = await semanticService.getInferenceResults(infRes.data.job_id);
          if (resultsRes.success && resultsRes.data) {
            setActiveInferenceResults(resultsRes.data);
          }
        }
        const semRes = await semanticService.getSemanticFrame(frameId);
        if (semRes.success && semRes.data) {
          setActiveSemanticFrame(semRes.data);
        }
        setStages((s) => ({ ...s, salsanext: 'complete' }));
        addEvent('Fast-FRNet Segmentation Complete', '20-class point classification ready', 'complete');
      } catch {
        setStages((s) => ({ ...s, salsanext: 'failed' }));
      }

      // 3. Terrain Analysis
      setStages((s) => ({ ...s, terrain: 'processing' }));
      const terRes = await terrainService.analyzeTerrain({ frame_id: frameId });
      if (terRes.success && terRes.data) {
        setActiveTerrainResponse(terRes.data);
        setStages((s) => ({ ...s, terrain: 'complete' }));
        addEvent('Terrain Analysis Complete', `${terRes.data.cells.length} elevation cells computed`, 'complete');
      }

      // 4. Object Detection & Tracking
      setStages((s) => ({ ...s, objects: 'processing' }));
      const detRes = await objectService.detectObjects({ frame_id: frameId });
      if (detRes.success && detRes.data) {
        setActiveObjectDetection(detRes.data);
        const trackRes = await trackingService.updateTracks({ frame_id: frameId });
        if (trackRes.success && trackRes.data) {
          setActiveTracks(trackRes.data.tracks);
        }
        setStages((s) => ({ ...s, objects: 'complete' }));
        addEvent('Object Detection & Tracking Complete', `${detRes.data.total_instances} instances detected`, 'complete');
      }

      // 5. 2.5D Adaptive Grid
      setStages((s) => ({ ...s, adaptiveGrid: 'processing' }));
      const mapRes = await mapService.updateMap('live_fovea_map_01', { frame_id: frameId });
      if (mapRes.success && mapRes.data) {
        const cellsRes = await mapService.getMapCells('live_fovea_map_01', { limit: 5000 });
        if (cellsRes.success && cellsRes.data && cellsRes.data.length > 0) {
          mapRes.data.cells_sample = cellsRes.data;
        }
        setActiveMapResponse(mapRes.data);
        setStages((s) => ({ ...s, adaptiveGrid: 'complete' }));
        addEvent('2.5D Adaptive Grid Synthesized', `${mapRes.data.updated_cell_count} elevation cells generated`, 'complete');
      }
    } catch (err) {
      addEvent('Pipeline Encountered Error', String(err), 'failed');
    } finally {
      setIsPipelineRunning(false);
    }
  };

  // Export Map JSON
  const handleExportMap = async () => {
    try {
      const res = await mapService.exportMap('live_fovea_map_01');
      if (res.success && res.data) {
        const blob = new Blob([JSON.stringify(res.data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `lidar_x_2.5d_${res.data.map_id}.json`;
        a.click();
        URL.revokeObjectURL(url);
        addEvent('Map Exported', `Downloaded 2.5D map JSON (${res.data.cells.length} cells)`, 'complete');
      }
    } catch (err) {
      addEvent('Export Failed', String(err), 'failed');
    }
  };

  const toggleCategoryFilter = (catKey: string) => {
    setCategoryFilters((prev) => ({
      ...prev,
      [catKey]: !prev[catKey],
    }));
  };

  // Launch pre-bundled demo sequence replay
  const handleLoadSampleSequence = async () => {
    setConsoleMode('sequence-replay');
    setHasUploaded(true);
    const demoSessionId = 'semantic_kitti_sequence_00';
    setReplaySessionId(demoSessionId);
    setReplaySequenceName('SemanticKITTI Sequence 00');
    setReplayFrameIndex(0);
    setReplayPlaybackState('ready');
    setReplayDataMode('precomputed_labels');

    if (wsStreamRef.current) {
      wsStreamRef.current.close();
    }
    wsStreamRef.current = replayService.connectWebSocketStream(
      demoSessionId,
      (frame) => handleFramePayload(frame),
      (status, info) => {
        setReplayPlaybackState(status.state);
        setReplayFrameIndex(status.current_frame_index);
        setReplayTotalFrames(status.total_frames);
        if (info.resync) {
          addEvent('Replay WebSocket Resynced', `Recovered to frame ${status.current_frame_index + 1} of ${status.total_frames}`, 'info');
        }
      },
      (err) => {
        console.warn('Replay WebSocket error:', err);
      },
      (connected) => setIsWebSocketConnected(connected)
    );

    // Try to fetch precompute progress for the demo session (it should
    // already be precomputed and 100% complete).
    try {
      const st = await replayService.getPrecomputeStatus(demoSessionId);
      if (st.success && st.data) {
        if (st.data.total_frames) {
          setReplayTotalFrames(st.data.total_frames);
        }
        setPrecomputeStatus({
          isRunning: !!st.data.is_running,
          isComplete: !!st.data.is_complete,
          percentComplete: st.data.percent_complete || 0,
          processedFrames: st.data.processed_frames || 0,
          totalFrames: st.data.total_frames || 0,
          etaSeconds: st.data.eta_seconds || 0,
          currentStage: st.data.current_stage,
          failedCount: st.data.failed_count,
          errorMessage: st.data.error_message || null,
        });
      }
    } catch { /* ignore */ }

    // Fetch frame 0 immediately
    try {
      const f0 = await replayService.getNextFrame(demoSessionId);
      if (f0.success && f0.data) {
        handleFramePayload(f0.data);
      }
    } catch (e) {
      console.warn('Error loading initial replay frame:', e);
    }
  };

  const selectedObject = useMemo(() => {
    if (!selectedInstanceId) return null;
    const inst = activeObjectDetection?.instances.find((i) => i.instance_id === selectedInstanceId);
    if (inst) return inst;
    const tr = activeTracks.find((t) => t.track_id === selectedInstanceId);
    if (tr) {
      return {
        instance_id: tr.track_id,
        semantic_category: tr.semantic_category,
        object_type: tr.object_type,
        centroid: tr.current_position,
        bounding_box: tr.bounding_box,
        dimensions: [2.5, 1.8, 1.5] as [number, number, number],
        extent_category: 'compact_actor' as const,
        point_count: 0,
        is_dynamic: true,
        source_frame_id: selectedFrameId,
        observation_status: 'directly_observed' as const,
        sample_points: [],
      };
    }
    return null;
  }, [selectedInstanceId, activeObjectDetection, activeTracks, selectedFrameId]);

  const rawPoints: Point3D[] = useMemo(() => {
    if (activeLidarFrame && activeLidarFrame.points.length > 0) {
      return activeLidarFrame.points;
    }
    return [];
  }, [activeLidarFrame]);

  const semanticPoints: Point3D[] = useMemo(() => {
    if (activeSemanticFrame && activeSemanticFrame.sample_labeled_points.length > 0) {
      return activeSemanticFrame.sample_labeled_points.map((p) => ({
        x: p.x,
        y: p.y,
        z: p.z,
        intensity: p.intensity,
        semanticClass: p.semantic_class || (p.project_category === 'drivable' ? 'drivable' : p.project_category === 'vegetation' ? 'vegetation' : p.project_category === 'dynamic_object' ? 'dynamicObstacle' : p.project_category === 'static_obstacle' ? 'staticObstacle' : p.project_category === 'infrastructure' ? 'infrastructure' : 'unknown'),
      }));
    }
    if (activeLidarFrame && activeLidarFrame.points.length > 0) {
      return activeLidarFrame.points;
    }
    return [];
  }, [activeSemanticFrame, activeLidarFrame]);

  const activeDataset: 'rellis' | 'semantickitti' = useMemo(() => {
    const name = (replaySequenceName || '').toLowerCase();
    if (name.includes('rellis') || name.includes('00000') || name.includes('offroad') || name.includes('desert') || name.includes('forest')) {
      return 'rellis';
    }
    return 'semantickitti';
  }, [replaySequenceName]);

  const ac = selectedCell?.backendAdaptiveCell;
  const tc = selectedCell?.backendTerrain;

  // If no data has been uploaded or selected, present the unified Upload & Source Selection Portal
  if (!hasUploaded) {
    return (
      <div className="space-y-6 max-w-5xl mx-auto py-8 px-4">
        {/* Header */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-mono font-medium">
            <Radio className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
            <span>Autonomous LiDAR Perception & 2.5D Mapping</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
            LiDAR Data Ingestion & Replay Portal
          </h1>
          <p className="text-gray-400 text-sm sm:text-base max-w-2xl mx-auto font-sans">
            Choose a data format to upload. Once ingested, the full autonomous perception dashboard and 2.5D foveated map will open.
          </p>
        </div>

        {/* LiDAR Sequence Video Feed */}
        <div className="max-w-xl mx-auto pt-2">
          <div className="tech-panel rounded-3xl p-6 md:p-8 border border-white/10 hover:border-cyan-500/50 transition-all duration-300 flex flex-col justify-between space-y-6 group hover:shadow-glow-cyan bg-dark-900/90">
            <div className="space-y-4">
              <div className="w-14 h-14 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 group-hover:scale-105 transition-transform">
                <Film className="w-7 h-7" />
              </div>
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <h2 className="text-xl font-bold text-white">LiDAR Sequence Video Feed</h2>
                  <Badge variant="cyan" className="text-[10px]">VIDEO REPLAY</Badge>
                </div>
                <p className="text-xs text-cyan-300 font-mono">SemanticKITTI Multi-Frame Replay Stream</p>
              </div>
              <p className="text-sm text-gray-300 leading-relaxed">
                Ingest a sequence archive containing continuous sensor sweeps (<code className="text-cyan-300 text-xs font-mono">velodyne/*.bin</code> + optional <code className="text-purple-300 text-xs font-mono">predictions/*.label</code>).
              </p>
              <ul className="text-xs text-gray-400 space-y-2 font-mono">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  <span>Real-time WebSocket streaming playback</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  <span>Interactive timeline scrubber (1x, 2x, 5x, 10x FPS)</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  <span>Continuous 2.5D foveated grid map accumulation</span>
                </li>
              </ul>
            </div>

            <div className="space-y-3 pt-2">
              <input
                ref={sequenceInputRef}
                type="file"
                accept=".zip"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleUploadSequenceZip(file);
                }}
                className="hidden"
              />
              <button
                onClick={() => sequenceInputRef.current?.click()}
                disabled={isUploadingSequence}
                className="w-full py-3.5 px-4 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-black font-bold text-sm transition-all shadow-glow-cyan flex items-center justify-center gap-2 disabled:opacity-50 cursor-pointer"
              >
                {isUploadingSequence ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Ingesting Sequence ZIP...</span>
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4 stroke-[2.5]" />
                    <span>Upload Sequence ZIP (.zip)</span>
                  </>
                )}
              </button>

              <button
                onClick={handleLoadSampleSequence}
                className="w-full py-2.5 px-4 rounded-xl bg-dark-850 hover:bg-dark-800 text-gray-300 hover:text-white border border-white/10 text-xs font-mono transition-colors flex items-center justify-center gap-2 cursor-pointer"
              >
                <Play className="w-3.5 h-3.5 text-cyan-400" />
                <span>Launch Demo Sequence Replay</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4 max-w-[1700px] mx-auto pb-10">
      {/* Top Replay Control Bar */}
      <SequenceReplayBar
        sessionId={replaySessionId}
        sequenceName={replaySequenceName}
        playbackState={replayPlaybackState}
        currentFrameIndex={replayFrameIndex}
        totalFrames={replayTotalFrames}
        fps={replayFps}
        dataMode={replayDataMode}
        processingTimeMs={replayProcessingTimeMs}
        isUploading={isUploadingSequence}
        hasPredictions={Boolean((activeSemanticFrame && activeSemanticFrame.sample_labeled_points.length > 0) || replayDataMode === 'precomputed_labels')}
        detectionState={detectionActionState}
        detectedCount={activeObjectDetection?.total_instances || 0}
        onRunObjectDetection={handleRunObjectDetectionAction}
        onUploadZip={handleUploadSequenceZip}
        onPlay={handlePlayReplay}
        onPause={handlePauseReplay}
        onStop={handleStopReplay}
        onStepBackward={handleStepBackward}
        onStepForward={handleStepForward}
        onSeek={handleSeekReplay}
        onSetFps={handleSetReplayFps}
        precompute={precomputeStatus}
        onTriggerPrecompute={handleTriggerPrecompute}
        onCancelPrecompute={handleCancelPrecompute}
        isWebSocketConnected={isWebSocketConnected}
        liveFps={realtimeFps}
        semanticSource={replaySemanticSource}
      />

      {/* 3. Main Multi-Column Autonomous Perception Console Layout */}
      <div
        className="grid grid-cols-1 gap-4 items-start w-full min-w-0 transition-all duration-300"
        style={{
          gridTemplateColumns: isDesktop
            ? `${isLeftCollapsed ? '52px' : 'clamp(280px, 22vw, 360px)'} minmax(0, 1fr) ${isRightCollapsed ? '52px' : 'clamp(280px, 22vw, 360px)'}`
            : '1fr',
        }}
      >
        {/* ================= LEFT COLUMN: Perception & Sensors ================= */}
        {isLeftCollapsed ? (
          <aside className="tech-panel rounded-2xl p-2 border border-white/10 flex flex-col items-center justify-between py-4 bg-dark-900/80 hover:bg-dark-900/95 transition-all w-full min-w-0 shadow-lg select-none group min-h-[520px]">
            <div className="flex flex-col items-center gap-3 w-full">
              <button
                type="button"
                onClick={() => setIsLeftCollapsed(false)}
                className="w-9 h-9 flex items-center justify-center rounded-xl bg-cyan-500/10 hover:bg-cyan-500/25 text-cyan-400 border border-cyan-500/30 transition-all hover:scale-105 active:scale-95 shadow-md cursor-pointer"
                title="Expand Perception Panels (LiDAR & Fast-FRNet)"
                aria-label="Expand Left Panel"
              >
                <ChevronRight className="w-5 h-5 stroke-[2.5]" />
              </button>

              <div className="w-6 h-px bg-white/10 my-1" />

              <button
                type="button"
                onClick={() => setIsLeftCollapsed(false)}
                className="w-8 h-8 rounded-lg bg-dark-850 border border-white/10 flex items-center justify-center text-cyan-400 hover:border-cyan-500/40 hover:bg-cyan-500/10 transition-colors cursor-pointer"
                title={`Raw LiDAR (${rawPoints.length > 0 ? `${rawPoints.length} Pts` : 'No Frame'}) - Click to expand`}
              >
                <Eye className="w-4 h-4" />
              </button>

              <button
                type="button"
                onClick={() => setIsLeftCollapsed(false)}
                className="w-8 h-8 rounded-lg bg-dark-850 border border-white/10 flex items-center justify-center text-purple-400 hover:border-purple-500/40 hover:bg-purple-500/10 transition-colors cursor-pointer"
                title="Semantic Fast-FRNet - Click to expand"
              >
                <Cpu className="w-4 h-4" />
              </button>
            </div>

            <div className="flex flex-col items-center gap-2 py-4">
              <span
                style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}
                className="text-[10px] font-mono font-bold tracking-widest text-cyan-400/80 uppercase whitespace-nowrap cursor-pointer hover:text-cyan-300 transition-colors"
                onClick={() => setIsLeftCollapsed(false)}
              >
                Perception & LiDAR
              </span>
            </div>

            <button
              type="button"
              onClick={() => setIsLeftCollapsed(false)}
              className="w-8 h-8 flex items-center justify-center rounded-lg text-gray-500 hover:text-cyan-400 hover:bg-white/5 transition-colors cursor-pointer"
              title="Expand panel"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </aside>
        ) : (
          <aside className="flex flex-col gap-4 min-w-0 w-full transition-all">
            {/* Top Collapse Action Bar for Left Column */}
            <div className="flex items-center justify-between px-3 py-1.5 rounded-xl bg-dark-900/60 border border-white/5 text-[11px] font-mono text-gray-400">
              <span className="flex items-center gap-1.5 font-bold uppercase tracking-wider text-cyan-400 text-[10px]">
                <Radio className="w-3.5 h-3.5" />
                <span>Perception & LiDAR</span>
              </span>
              <button
                type="button"
                onClick={() => setIsLeftCollapsed(true)}
                className="flex items-center gap-1 px-1.5 py-0.5 rounded text-gray-400 hover:text-cyan-300 hover:bg-white/5 transition-colors cursor-pointer text-[10px]"
                title="Collapse left panel to expand map view"
                aria-label="Collapse Left Panel"
              >
                <span>Collapse</span>
                <ChevronLeft className="w-3.5 h-3.5 stroke-[2]" />
              </button>
            </div>
          {/* PANEL ONE: Raw LiDAR Point Cloud */}
          <div className="tech-panel rounded-2xl p-4 border border-white/10 flex flex-col space-y-3">
            <div className="flex items-center justify-between border-b border-white/10 pb-2">
              <div className="flex items-center gap-2">
                <Eye className="w-4 h-4 text-cyan-400" />
                <h3 className="text-xs font-bold text-white uppercase tracking-wider font-mono">
                  Raw LiDAR Point Cloud
                </h3>
              </div>
              <span className="text-[10px] font-mono text-cyan-400">
                {rawPoints.length > 0 ? `${rawPoints.length} Pts` : 'No Frame'}
              </span>
            </div>

            {/* View Mode & Intensity Controls */}
            <div className="flex items-center justify-between gap-1 text-[10px] font-mono">
              <div className="flex items-center gap-1 bg-dark-850 p-1 rounded-lg border border-white/5">
                <button
                  onClick={() => setRawViewMode('perspective')}
                  className={`px-2 py-0.5 rounded transition-colors ${
                    rawViewMode === 'perspective' ? 'bg-cyan-500/20 text-cyan-300 font-bold' : 'text-gray-400 hover:text-white'
                  }`}
                >
                  Perspective
                </button>
                <button
                  onClick={() => setRawViewMode('top')}
                  className={`px-2 py-0.5 rounded transition-colors ${
                    rawViewMode === 'top' ? 'bg-cyan-500/20 text-cyan-300 font-bold' : 'text-gray-400 hover:text-white'
                  }`}
                >
                  Top-Down
                </button>
              </div>

              <button
                onClick={() => setIsGrayscale(!isGrayscale)}
                className={`px-2 py-1 rounded-lg border text-[10px] font-mono transition-colors ${
                  isGrayscale
                    ? 'bg-dark-800 text-gray-200 border-white/20'
                    : 'bg-dark-850 text-cyan-400 border-white/5'
                }`}
                title="Toggle Grayscale vs False-Color Intensity"
              >
                {isGrayscale ? 'Grayscale' : 'Intensity'}
              </button>
            </div>

            {/* Canvas 3D */}
            <div className="w-full h-[250px] rounded-xl overflow-hidden border border-white/10 bg-dark-950 relative shadow-inner">
              <LidarCanvas3D
                points={rawPoints}
                viewMode={isGrayscale ? 'raw' : 'intensity'}
                grayscale={isGrayscale}
                pointSize={pointSize}
                emptyMessage="Upload a LiDAR frame or sequence to render the raw point cloud."
                className="w-full h-full"
              />
            </div>
          </div>

          {/* PANEL TWO: Semantic Segmentation (Fast-FRNet) */}
          <div className="tech-panel rounded-2xl p-4 border border-white/10 flex flex-col space-y-3">
            <div className="flex items-center justify-between border-b border-white/10 pb-2">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-purple-400" />
                <h3 className="text-xs font-bold text-white uppercase tracking-wider font-mono">
                  Semantic Segmentation
                </h3>
              </div>
              <button
                onClick={() => handleRunStage('salsanext')}
                className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 border border-purple-500/40 transition-colors"
                title="Execute Fast-FRNet deep neural inference"
              >
                Run Fast-FRNet
              </button>
            </div>

            {/* Semantic Category Filter Chips - Uniform 3x2 Grid */}
            <div className="grid grid-cols-3 gap-1 text-[10px] font-mono">
              <button
                onClick={() => toggleCategoryFilter('drivable')}
                className={`px-1.5 py-1 rounded border transition-colors flex items-center justify-center gap-1 text-[9.5px] ${
                  categoryFilters.drivable ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40 font-semibold' : 'bg-dark-850 text-gray-500 border-white/5 opacity-50'
                }`}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-semantic-drivable shrink-0" />
                <span className="truncate">Drivable</span>
              </button>
              <button
                onClick={() => toggleCategoryFilter('nondrivable')}
                className={`px-1.5 py-1 rounded border transition-colors flex items-center justify-center gap-1 text-[9.5px] ${
                  categoryFilters.nondrivable ? 'bg-gray-500/20 text-gray-300 border-gray-500/40 font-semibold' : 'bg-dark-850 text-gray-500 border-white/5 opacity-50'
                }`}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-semantic-nondrivable shrink-0" />
                <span className="truncate">Non-Drive</span>
              </button>
              <button
                onClick={() => toggleCategoryFilter('staticObstacle')}
                className={`px-1.5 py-1 rounded border transition-colors flex items-center justify-center gap-1 text-[9.5px] ${
                  categoryFilters.staticObstacle ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 font-semibold' : 'bg-dark-850 text-gray-500 border-white/5 opacity-50'
                }`}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-semantic-staticObstacle shrink-0" />
                <span className="truncate">Static</span>
              </button>
              <button
                onClick={() => toggleCategoryFilter('dynamicObstacle')}
                className={`px-1.5 py-1 rounded border transition-colors flex items-center justify-center gap-1 text-[9.5px] ${
                  categoryFilters.dynamicObstacle ? 'bg-red-500/20 text-red-300 border-red-500/40 font-semibold' : 'bg-dark-850 text-gray-500 border-white/5 opacity-50'
                }`}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-semantic-dynamicObstacle shrink-0" />
                <span className="truncate">Dynamic</span>
              </button>
              <button
                onClick={() => toggleCategoryFilter('vegetation')}
                className={`px-1.5 py-1 rounded border transition-colors flex items-center justify-center gap-1 text-[9.5px] ${
                  categoryFilters.vegetation ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 font-semibold' : 'bg-dark-850 text-gray-500 border-white/5 opacity-50'
                }`}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-semantic-vegetation shrink-0" />
                <span className="truncate">Foliage</span>
              </button>
              <button
                onClick={() => toggleCategoryFilter('infrastructure')}
                className={`px-1.5 py-1 rounded border transition-colors flex items-center justify-center gap-1 text-[9.5px] ${
                  categoryFilters.infrastructure ? 'bg-purple-500/20 text-purple-300 border-purple-500/40 font-semibold' : 'bg-dark-850 text-gray-500 border-white/5 opacity-50'
                }`}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-semantic-infrastructure shrink-0" />
                <span className="truncate">Structure</span>
              </button>
            </div>

            {/* Semantic Canvas 3D */}
            <div className="w-full h-[250px] rounded-xl overflow-hidden border border-white/10 bg-dark-950 relative shadow-inner">
              <LidarCanvas3D
                points={semanticPoints}
                viewMode="semantic"
                categoryFilters={categoryFilters}
                highlightBox={selectedObject ? selectedObject.bounding_box : null}
                emptyMessage="Semantic output is not available for this frame."
                className="w-full h-full"
              />
            </div>
          </div>
        </aside>
      )}

      {/* ================= CENTER COLUMN: Large Adaptive 2.5D Map ================= */}
      <section className="flex flex-col gap-4 min-w-0 flex-1 w-full">
        {/* PANEL FOUR: Adaptive 2.5D Map (Largest Center Panel) */}
        <div className="tech-panel rounded-2xl p-4 border border-white/10 flex flex-col space-y-3 min-w-0 w-full">
          <div className="flex items-center justify-between border-b border-white/10 pb-2.5">
            <div className="flex items-center gap-2">
              <Grid className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-bold text-white uppercase tracking-wider font-mono">
                Adaptive 2.5D Foveated Map
              </h3>
            </div>

            {/* Center Map Quick Layout Expand / Theater Controls */}
            <div className="flex items-center gap-2">
              {/* Toggle Left Panel from Center */}
              <button
                type="button"
                onClick={() => setIsLeftCollapsed(!isLeftCollapsed)}
                className={`hidden sm:flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] font-mono border transition-all cursor-pointer ${
                  isLeftCollapsed
                    ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40 hover:bg-cyan-500/30'
                    : 'bg-dark-850 text-gray-400 hover:text-white border-white/10'
                }`}
                title={isLeftCollapsed ? 'Show Perception Panel' : 'Hide Perception Panel'}
              >
                {isLeftCollapsed ? <ChevronRight className="w-3 h-3" /> : <ChevronLeft className="w-3 h-3" />}
                <span>{isLeftCollapsed ? '+ Perception' : 'Hide Left'}</span>
              </button>

              {/* Theater / Fullscreen Width Toggle */}
              <button
                type="button"
                onClick={() => {
                  if (isLeftCollapsed && isRightCollapsed) {
                    setIsLeftCollapsed(false);
                    setIsRightCollapsed(false);
                  } else {
                    setIsLeftCollapsed(true);
                    setIsRightCollapsed(true);
                  }
                }}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] font-mono font-semibold border transition-all cursor-pointer ${
                  isLeftCollapsed && isRightCollapsed
                    ? 'bg-cyan-500 text-dark-950 border-cyan-400 shadow-[0_0_12px_rgba(0,240,255,0.35)]'
                    : 'bg-dark-850 text-cyan-400 hover:text-cyan-300 border-white/10 hover:border-cyan-500/30'
                }`}
                title={isLeftCollapsed && isRightCollapsed ? 'Restore 3-Column Layout' : 'Maximize Map (Collapse Side Panels)'}
              >
                <Maximize2 className="w-3.5 h-3.5" />
                <span>{isLeftCollapsed && isRightCollapsed ? 'Restore View' : 'Maximize Map'}</span>
              </button>

              {/* Toggle Right Panel from Center */}
              <button
                type="button"
                onClick={() => setIsRightCollapsed(!isRightCollapsed)}
                className={`hidden sm:flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] font-mono border transition-all cursor-pointer ${
                  isRightCollapsed
                    ? 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40 hover:bg-indigo-500/30'
                    : 'bg-dark-850 text-gray-400 hover:text-white border-white/10'
                }`}
                title={isRightCollapsed ? 'Show Terrain Panel' : 'Hide Terrain Panel'}
              >
                <span>{isRightCollapsed ? '+ Terrain' : 'Hide Right'}</span>
                {isRightCollapsed ? <ChevronLeft className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              </button>
            </div>
          </div>

          {/* 2.5D Adaptive Grid Canvas */}
          <div className="w-full min-h-[580px] h-[640px] xl:h-[680px] rounded-xl overflow-hidden border border-white/10 bg-dark-950 relative shadow-2xl min-w-0">
              {isLoading ? (
                <div className="w-full h-full flex flex-col items-center justify-center space-y-3">
                  <div className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                  <span className="text-xs font-mono text-gray-400">Synthesizing 2.5D Adaptive Grid...</span>
                </div>
              ) : (
                <MapCanvas2D
                  cells={cells}
                  layers={mapLayers}
                  objects={activeObjectDetection?.instances || []}
                  tracks={activeTracks || []}
                  points={semanticPoints}
                  dataset={activeDataset}
                  onCellSelect={(c) => setSelectedCell(c)}
                  onObjectSelect={(id) => setSelectedInstanceId(id)}
                  selectedInstanceId={selectedInstanceId}
                  className="w-full h-full"
                />
              )}
            </div>

            {/* Quick Semantic Legend Strip */}
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 p-2.5 rounded-xl bg-dark-900/70 border border-white/5 text-[11px] font-mono text-gray-400">
              <span className="text-gray-500 uppercase tracking-wider text-[10px]">Legend:</span>
              <span className="flex items-center gap-1 text-cyan-300">
                <span className="w-2 h-2 rounded-xs bg-semantic-drivable" />
                <span>Drivable</span>
              </span>
              <span className="flex items-center gap-1 text-emerald-300">
                <span className="w-2 h-2 rounded-xs bg-semantic-vegetation" />
                <span>Vegetation</span>
              </span>
              <span className="flex items-center gap-1 text-amber-300">
                <span className="w-2 h-2 rounded-xs bg-semantic-staticObstacle" />
                <span>Static Hazard</span>
              </span>
              <span className="flex items-center gap-1 text-red-300">
                <span className="w-2 h-2 rounded-xs bg-semantic-dynamicObstacle" />
                <span>Dynamic Actor</span>
              </span>
              <span className="flex items-center gap-1 text-purple-300">
                <span className="w-2 h-2 rounded-xs bg-semantic-infrastructure" />
                <span>Infrastructure</span>
              </span>
              <span className="flex items-center gap-1 text-gray-500">
                <span className="w-2 h-2 rounded-xs bg-semantic-unknown" />
                <span>Unobserved</span>
              </span>
            </div>

            {/* Interactive Cell Inspector */}
            {selectedCell && (
              <div className="p-3.5 rounded-xl bg-dark-900/90 border border-cyan-500/30 space-y-2.5">
                <div className="flex items-center justify-between border-b border-white/10 pb-2">
                  <div className="flex items-center gap-2">
                    <Crosshair className="w-4 h-4 text-cyan-400" />
                    <span className="text-xs font-bold text-white font-mono">
                      Cell Inspector: <span className="text-cyan-300">{selectedCell.id}</span>
                    </span>
                  </div>
                  <Badge variant={ac ? 'emerald' : tc ? 'purple' : 'cyan'} className="text-[10px]">
                    {ac ? `${ac.level.toUpperCase()} (${ac.size_m}m)` : tc ? 'TERRAIN GEOMETRY' : 'SELECTED'}
                  </Badge>
                </div>

                {ac ? (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono">
                    <div className="p-2 rounded bg-dark-850 border border-white/5">
                      <span className="text-gray-400 block text-[10px]">Dominant Class</span>
                      <span className="text-white font-bold capitalize">{ac.dominant_semantic_class}</span>
                    </div>
                    <div className="p-2 rounded bg-dark-850 border border-white/5">
                      <span className="text-gray-400 block text-[10px]">Traversability</span>
                      <span className="text-cyan-300 font-bold uppercase">{ac.traversability_state.replace(/_/g, ' ')}</span>
                    </div>
                    <div className="p-2 rounded bg-dark-850 border border-white/5">
                      <span className="text-gray-400 block text-[10px]">Slope Angle</span>
                      <span className="text-amber-300 font-bold">{ac.slope_summary.toFixed(1)}°</span>
                    </div>
                    <div className="p-2 rounded bg-dark-850 border border-white/5">
                      <span className="text-gray-400 block text-[10px]">Elevation (Mean / ΔZ)</span>
                      <span className="text-gray-200 font-bold">{ac.elevation_mean.toFixed(2)}m (Δ{ac.elevation_variation.toFixed(2)}m)</span>
                    </div>
                  </div>
                ) : tc ? (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono">
                    <div className="p-2 rounded bg-dark-850 border border-white/5">
                      <span className="text-gray-400 block text-[10px]">Drivability</span>
                      <span className="text-cyan-300 font-bold uppercase">{tc.drivability_state.replace(/_/g, ' ')}</span>
                    </div>
                    <div className="p-2 rounded bg-dark-850 border border-white/5">
                      <span className="text-gray-400 block text-[10px]">Roughness (RMS)</span>
                      <span className="text-purple-300 font-bold">{tc.roughness_m.toFixed(3)}m</span>
                    </div>
                    <div className="p-2 rounded bg-dark-850 border border-white/5">
                      <span className="text-gray-400 block text-[10px]">Slope</span>
                      <span className="text-amber-300 font-bold">{tc.slope_deg.toFixed(1)}°</span>
                    </div>
                    <div className="p-2 rounded bg-dark-850 border border-white/5">
                      <span className="text-gray-400 block text-[10px]">Step / Curb</span>
                      <span className={`font-bold ${tc.has_step ? 'text-red-400' : 'text-gray-400'}`}>{tc.step_height_m.toFixed(2)}m</span>
                    </div>
                  </div>
                ) : (
                  <div className="text-xs font-mono text-gray-400">
                    Position: ({selectedCell.worldX}m, {selectedCell.worldY}m) • Traversability: {selectedCell.traversability}
                  </div>
                )}
              </div>
            )}
          </div>
      </section>

      {/* ================= RIGHT COLUMN: Elevation & Traversability ================= */}
      {isRightCollapsed ? (
        <aside className="tech-panel rounded-2xl p-2 border border-white/10 flex flex-col items-center justify-between py-4 bg-dark-900/80 hover:bg-dark-900/95 transition-all w-full min-w-0 shadow-lg select-none group min-h-[520px]">
          <div className="flex flex-col items-center gap-3 w-full">
            <button
              type="button"
              onClick={() => setIsRightCollapsed(false)}
              className="w-9 h-9 flex items-center justify-center rounded-xl bg-indigo-500/10 hover:bg-indigo-500/25 text-indigo-400 border border-indigo-500/30 transition-all hover:scale-105 active:scale-95 shadow-md cursor-pointer"
              title="Expand Terrain & Traversability Maps"
              aria-label="Expand Right Panel"
            >
              <ChevronLeft className="w-5 h-5 stroke-[2.5]" />
            </button>

            <div className="w-6 h-px bg-white/10 my-1" />

            <button
              type="button"
              onClick={() => setIsRightCollapsed(false)}
              className="w-8 h-8 rounded-lg bg-dark-850 border border-white/10 flex items-center justify-center text-indigo-400 hover:border-indigo-500/40 hover:bg-indigo-500/10 transition-colors cursor-pointer"
              title="Elevation Map (2D Relief) - Click to expand"
            >
              <Mountain className="w-4 h-4" />
            </button>

            <button
              type="button"
              onClick={() => setIsRightCollapsed(false)}
              className="w-8 h-8 rounded-lg bg-dark-850 border border-white/10 flex items-center justify-center text-emerald-400 hover:border-emerald-500/40 hover:bg-emerald-500/10 transition-colors cursor-pointer"
              title="Traversability Map (Drivable Boundaries) - Click to expand"
            >
              <Shield className="w-4 h-4" />
            </button>
          </div>

          <div className="flex flex-col items-center gap-2 py-4">
            <span
              style={{ writingMode: 'vertical-rl' }}
              className="text-[10px] font-mono font-bold tracking-widest text-indigo-400/80 uppercase whitespace-nowrap cursor-pointer hover:text-indigo-300 transition-colors"
              onClick={() => setIsRightCollapsed(false)}
            >
              Terrain & Relief
            </span>
          </div>

          <button
            type="button"
            onClick={() => setIsRightCollapsed(false)}
            className="w-8 h-8 flex items-center justify-center rounded-lg text-gray-500 hover:text-indigo-400 hover:bg-white/5 transition-colors cursor-pointer"
            title="Expand panel"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
        </aside>
        ) : (
          <aside className="flex flex-col gap-4 min-w-0 w-full transition-all">
            {/* Top Collapse Action Bar for Right Column */}
            <div className="flex items-center justify-between px-3 py-1.5 rounded-xl bg-dark-900/60 border border-white/5 text-[11px] font-mono text-gray-400">
              <button
                type="button"
                onClick={() => setIsRightCollapsed(true)}
                className="flex items-center gap-1 px-1.5 py-0.5 rounded text-gray-400 hover:text-indigo-300 hover:bg-white/5 transition-colors cursor-pointer text-[10px]"
                title="Collapse right panel to expand map view"
                aria-label="Collapse Right Panel"
              >
                <ChevronRight className="w-3.5 h-3.5 stroke-[2]" />
                <span>Collapse</span>
              </button>
              <span className="flex items-center gap-1.5 font-bold uppercase tracking-wider text-indigo-400 text-[10px]">
                <Mountain className="w-3.5 h-3.5" />
                <span>Terrain & Relief</span>
              </span>
            </div>

            {/* PANEL FIVE: Elevation Map */}
            <div className="tech-panel rounded-2xl p-4 border border-white/10 flex flex-col space-y-3">
              <div className="flex items-center justify-between border-b border-white/10 pb-2">
                <div className="flex items-center gap-2">
                  <Mountain className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-xs font-bold text-white uppercase tracking-wider font-mono">
                    Elevation Map (2D Relief)
                  </h3>
                </div>
                <span className="text-[10px] font-mono text-indigo-300">
                  {activeMapResponse?.metadata?.bounds
                    ? `ΔZ ${(activeMapResponse.metadata.bounds.max_z - activeMapResponse.metadata.bounds.min_z).toFixed(1)}m`
                    : '2D Relief'}
                </span>
              </div>

              {/* Top Mode & Status Row matching Panel 1 */}
              <div className="flex items-center justify-between gap-1 text-[10px] font-mono">
                <div className="flex items-center gap-1 bg-dark-850 p-1 rounded-lg border border-white/5">
                  <span className="px-2 py-0.5 text-indigo-300 font-semibold">Terrain Relief</span>
                  <span className="text-gray-600">•</span>
                  <span className="px-1 text-gray-400">Height Field</span>
                </div>
                <span className="px-2 py-1 rounded-lg border bg-dark-850 border-white/5 text-gray-300 text-[10px] font-mono">
                  Continuous Z
                </span>
              </div>

              <div className="w-full h-[250px] rounded-xl overflow-hidden border border-white/10 bg-dark-950 relative">
                <ElevationMap2D
                  cells={cells}
                  selectedCellId={selectedCell?.id}
                  onCellSelect={(c) => setSelectedCell(c)}
                  className="w-full h-full"
                  layers={mapLayers}
                  useAbsoluteScale
                  absoluteRange={activeMapResponse?.metadata.bounds ? {
                    min: activeMapResponse.metadata.bounds.min_z,
                    max: activeMapResponse.metadata.bounds.max_z,
                  } : undefined}
                />
              </div>
            </div>

            {/* PANEL SIX: Traversability Map */}
            <div className="tech-panel rounded-2xl p-4 border border-white/10 flex flex-col space-y-3">
              <div className="flex items-center justify-between border-b border-white/10 pb-2">
                <div className="flex items-center gap-2">
                  <Shield className="w-4 h-4 text-emerald-400" />
                  <h3 className="text-xs font-bold text-white uppercase tracking-wider font-mono">
                    Traversability Map
                  </h3>
                </div>
                <span className="text-[10px] font-mono text-emerald-300">Drivable Boundaries</span>
              </div>

              {/* Top Mode & Status Row matching Panel 2 */}
              <div className="flex items-center justify-between gap-1 text-[10px] font-mono">
                <div className="flex items-center gap-1 bg-dark-850 p-1 rounded-lg border border-white/5">
                  <span className="px-2 py-0.5 text-emerald-300 font-semibold">Obstacle Clearance</span>
                  <span className="text-gray-600">•</span>
                  <span className="px-1 text-gray-400">5-Zone</span>
                </div>
                <span className="px-2 py-1 rounded-lg border bg-dark-850 border-white/5 text-emerald-400 text-[10px] font-mono">
                  Active Safety
                </span>
              </div>

              <div className="w-full h-[250px] rounded-xl overflow-hidden border border-white/10 bg-dark-950 relative">
                <TraversabilityMap2D
                  cells={cells}
                  selectedCellId={selectedCell?.id}
                  onCellSelect={(c) => setSelectedCell(c)}
                  className="w-full h-full"
                  layers={mapLayers}
                />
              </div>
            </div>
        </aside>
      )}
      </div>

      {/* 4. Real-Time Performance & Validation Section */}
      <PerformanceValidationPanel
        currentFrameId={selectedFrameId || activeLidarFrame?.frameId}
        sessionId={replaySessionId}
        sequenceName={replaySequenceName}
        isReplayActive={replayPlaybackState === 'playing'}
        liveFps={realtimeFps}
        activePerformance={activePerformance}
      />
    </div>
  );
};
