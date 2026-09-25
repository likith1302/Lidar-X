import React, { useState } from 'react';
import {
  Activity,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ChevronDown,
  ChevronUp,
  Cpu,
  Navigation,
  Database,
  Box,
  Car,
} from 'lucide-react';
import {
  ObjectDetectionResponse,
  TrackedObject,
  SemanticFrameResponse,
} from '../../types/objects';
import { Badge } from '../common/Badge';

interface ObjectPipelineDiagnosticsProps {
  frameId: string;
  hasPredictions: boolean;
  predictionFileName?: string;
  totalPoints: number;
  totalLabels: number;
  semanticFrame?: SemanticFrameResponse | null;
  detectionResult?: ObjectDetectionResponse | null;
  tracks?: TrackedObject[];
  frameCount?: number;
  isClusteringLoading?: boolean;
  className?: string;
}

export const ObjectPipelineDiagnostics: React.FC<ObjectPipelineDiagnosticsProps> = ({
  frameId,
  hasPredictions,
  predictionFileName,
  totalPoints,
  totalLabels,
  semanticFrame,
  detectionResult,
  tracks = [],
  frameCount = 1,
  isClusteringLoading = false,
  className = '',
}) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(false);

  // 1. Validation calculation
  const isLabelCountValid = hasPredictions && totalPoints > 0 && totalLabels === totalPoints;
  const isLabelCountMismatch = hasPredictions && totalPoints > 0 && totalLabels > 0 && totalLabels !== totalPoints;

  // 2. Dynamic points calculation
  const categoryCounts = semanticFrame?.project_category_counts || {};
  const dynamicPointsCount = categoryCounts['dynamic_object'] || 0;
  const dynamicPointPercent = totalPoints > 0 ? ((dynamicPointsCount / totalPoints) * 100).toFixed(1) : '0.0';

  // 3. Instances and clusters
  const totalInstances = detectionResult?.total_instances ?? 0;
  const dynamicInstances = detectionResult?.dynamic_instances_count ?? 0;
  const staticInstances = detectionResult?.static_instances_count ?? 0;

  // 4. Tracking state determination
  let trackingStateText = 'Tracking requires consecutive LiDAR frames';
  let trackingBadgeVariant: 'emerald' | 'cyan' | 'warning' | 'neutral' = 'neutral';

  if (frameCount >= 2 && tracks.length > 0) {
    const activeCount = tracks.filter((t) => t.lifecycle_state === 'active').length;
    trackingStateText = `Active Kalman MOT (${activeCount} active, ${tracks.length} total)`;
    trackingBadgeVariant = 'cyan';
  } else if (frameCount === 1 && tracks.length > 0) {
    trackingStateText = `Initial frame registered (${tracks.length} new tracks, awaiting next frame for motion vectors)`;
    trackingBadgeVariant = 'warning';
  } else if (!hasPredictions) {
    trackingStateText = 'Awaiting semantic predictions to initialize object tracks';
    trackingBadgeVariant = 'neutral';
  }

  return (
    <div className={`bg-dark-900/80 backdrop-blur-md border border-white/10 rounded-2xl p-3.5 text-xs font-mono transition-all ${className}`}>
      {/* Accordion Toggle Bar */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between cursor-pointer select-none"
      >
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-purple-400" />
          <span className="font-bold text-white uppercase tracking-wider text-xs">
            Object Pipeline Diagnostics
          </span>
          <Badge
            variant={totalInstances > 0 ? 'purple' : hasPredictions ? 'warning' : 'neutral'}
            className="text-[10px]"
          >
            {totalInstances > 0
              ? `${totalInstances} INSTANCES`
              : hasPredictions
              ? 'LABELS LOADED'
              : 'PREDICTIONS REQUIRED'}
          </Badge>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-[11px] text-gray-400 hidden sm:inline">
            Frame: <span className="text-cyan-300 font-bold">{frameId || 'N/A'}</span>
          </span>
          <button className="text-gray-400 hover:text-white p-1 rounded-lg hover:bg-white/5 transition-colors">
            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Expanded Diagnostics Details */}
      {isExpanded && (
        <div className="mt-3.5 pt-3 border-t border-white/10 space-y-3">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2.5">
            {/* 1. Selected Frame */}
            <div className="bg-dark-950/60 p-2.5 rounded-xl border border-white/5 space-y-1">
              <span className="text-[10px] text-gray-500 uppercase flex items-center gap-1">
                <Database className="w-3 h-3 text-cyan-400" />
                Selected Frame ID
              </span>
              <p className="font-bold text-cyan-300 truncate text-[11px]">{frameId || 'None'}</p>
              <p className="text-[10px] text-gray-400">{totalPoints > 0 ? `${totalPoints.toLocaleString()} points` : '0 points'}</p>
            </div>

            {/* 2. Fast-FRNet Prediction File */}
            <div className="bg-dark-950/60 p-2.5 rounded-xl border border-white/5 space-y-1">
              <span className="text-[10px] text-gray-500 uppercase flex items-center gap-1">
                <Cpu className="w-3 h-3 text-purple-400" />
                Matching Prediction File
              </span>
              <div className="flex items-center gap-1.5">
                {hasPredictions ? (
                  <>
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    <span className="text-emerald-300 font-bold truncate text-[11px]">
                      {predictionFileName || 'Found (.label)'}
                    </span>
                  </>
                ) : (
                  <>
                    <XCircle className="w-3.5 h-3.5 text-red-400 shrink-0" />
                    <span className="text-red-400 font-bold text-[11px]">Missing Prediction</span>
                  </>
                )}
              </div>
              <p className="text-[10px] text-gray-400">
                {hasPredictions ? `${totalLabels.toLocaleString()} labels loaded` : 'Matching .label required'}
              </p>
            </div>

            {/* 3. Label Count Validation */}
            <div className="bg-dark-950/60 p-2.5 rounded-xl border border-white/5 space-y-1">
              <span className="text-[10px] text-gray-500 uppercase flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3 text-cyan-400" />
                Label Validation State
              </span>
              <div className="flex items-center gap-1.5">
                {isLabelCountValid ? (
                  <>
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    <span className="text-emerald-300 font-bold text-[11px]">1:1 Count Matched</span>
                  </>
                ) : isLabelCountMismatch ? (
                  <>
                    <AlertTriangle className="w-3.5 h-3.5 text-red-400 shrink-0" />
                    <span className="text-red-400 font-bold text-[11px]">Count Mismatch</span>
                  </>
                ) : (
                  <>
                    <span className="w-2 h-2 rounded-full bg-gray-500 shrink-0" />
                    <span className="text-gray-400 text-[11px]">Awaiting Labels</span>
                  </>
                )}
              </div>
              <p className="text-[10px] text-gray-400">
                {totalPoints > 0 && totalLabels > 0
                  ? `${totalPoints} pts = ${totalLabels} lbls`
                  : 'Requires .bin + .label'}
              </p>
            </div>

            {/* 4. Dynamic Point Availability */}
            <div className="bg-dark-950/60 p-2.5 rounded-xl border border-white/5 space-y-1">
              <span className="text-[10px] text-gray-500 uppercase flex items-center gap-1">
                <Car className="w-3 h-3 text-amber-400" />
                Dynamic-Point Availability
              </span>
              <p className={`font-bold text-[11px] ${dynamicPointsCount > 0 ? 'text-amber-300' : 'text-gray-400'}`}>
                {dynamicPointsCount > 0
                  ? `${dynamicPointsCount.toLocaleString()} pts (${dynamicPointPercent}%)`
                  : '0 Dynamic Points'}
              </p>
              <p className="text-[10px] text-gray-400 truncate">
                {dynamicPointsCount > 0 ? 'Car, Truck, Person, etc.' : 'Static terrain & infra'}
              </p>
            </div>
          </div>

          {/* Row 2: Clustering Result & Tracking State */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
            {/* Clustering Summary */}
            <div className="bg-dark-950/60 p-3 rounded-xl border border-white/5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-gray-500 uppercase flex items-center gap-1">
                  <Box className="w-3 h-3 text-purple-400" />
                  DBSCAN Instance Clustering Result
                </span>
                <Badge variant={totalInstances > 0 ? 'purple' : 'neutral'} className="text-[9px]">
                  {isClusteringLoading ? 'PROCESSING' : totalInstances > 0 ? 'CLUSTERS FOUND' : 'NO CLUSTERS'}
                </Badge>
              </div>

              {totalInstances > 0 ? (
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-white font-bold">
                    {totalInstances} Real Instances Formed
                  </span>
                  <span className="text-gray-400">
                    <span className="text-purple-300 font-semibold">{dynamicInstances} Dynamic</span> •{' '}
                    <span className="text-amber-300 font-semibold">{staticInstances} Static</span>
                  </span>
                </div>
              ) : hasPredictions ? (
                <p className="text-[11px] text-gray-400">
                  No dynamic object instances detected in this frame.
                </p>
              ) : (
                <p className="text-[11px] text-gray-500">
                  Matching Fast-FRNet prediction file is required for object detection.
                </p>
              )}
            </div>

            {/* Tracking State */}
            <div className="bg-dark-950/60 p-3 rounded-xl border border-white/5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-gray-500 uppercase flex items-center gap-1">
                  <Navigation className="w-3 h-3 text-cyan-400" />
                  Multi-Frame Kalman Tracking State
                </span>
                <Badge variant={trackingBadgeVariant} className="text-[9px]">
                  FRAME {frameCount}
                </Badge>
              </div>

              <div className="flex items-center justify-between text-[11px]">
                <span className="text-white font-medium truncate max-w-[280px]">
                  {trackingStateText}
                </span>
                <span className="text-cyan-400 font-bold shrink-0">
                  {tracks.length} Active Tracks
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
