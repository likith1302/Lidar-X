import React, { useState } from 'react';
import {
  ShieldCheck,
  ShieldAlert,
  CheckCircle2,
  AlertCircle,
  Clock,
  Layers,
  Cpu,
  Mountain,
  Box,
  Grid,
  ChevronDown,
  ChevronUp,
  Terminal,
  Play,
  Loader2,
  RefreshCw,
} from 'lucide-react';
import { PipelineStatusResponse } from '../../types/pipeline';
import { Badge } from '../common/Badge';

interface PipelineGateBannerProps {
  status: PipelineStatusResponse | null;
  isLoading?: boolean;
  isProcessing?: boolean;
  onRefresh?: () => void;
  onRunPipeline?: () => void;
  className?: string;
}

export const PipelineGateBanner: React.FC<PipelineGateBannerProps> = ({
  status,
  isLoading = false,
  isProcessing = false,
  onRefresh,
  onRunPipeline,
  className = '',
}) => {
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  const frameId = status?.frame_id || '—';
  const pointCloudOk = Boolean(status?.point_cloud_available && (status?.point_count ?? 0) > 0);
  const labelsOk = Boolean(status?.semantic_labels_available && status?.labels_match_point_count);
  const terrainOk = Boolean(status?.terrain_available);
  const objectsOk = Boolean(status?.object_detection_available);
  const mapOk = Boolean(status?.adaptive_map_available);

  const allGatesPassed = pointCloudOk && labelsOk && terrainOk && objectsOk && mapOk;

  return (
    <div className={`tech-panel rounded-2xl p-3.5 border border-white/10 bg-dark-900/90 backdrop-blur-md space-y-3 font-mono text-xs ${className}`}>
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 border-b border-white/10 pb-2.5">
        <div className="flex items-center gap-2.5 flex-wrap">
          <div className="flex items-center gap-2">
            {allGatesPassed ? (
              <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0" />
            ) : (
              <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0" />
            )}
            <span className="font-bold text-white uppercase tracking-wider text-xs">
              Live Pipeline Gate
            </span>
          </div>

          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-dark-850 border border-white/10 text-cyan-300">
            <span className="text-[10px] text-gray-400">Canonical Frame:</span>
            <span className="font-bold">{frameId}</span>
          </div>

          <Badge variant={allGatesPassed ? 'emerald' : 'warning'} className="text-[10px]">
            {allGatesPassed ? 'READY / VERIFIED' : 'GATED / PREREQUISITES PENDING'}
          </Badge>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          {onRefresh && (
            <button
              onClick={onRefresh}
              disabled={isLoading || isProcessing}
              className="px-2.5 py-1 rounded-lg bg-dark-850 hover:bg-dark-800 text-gray-300 hover:text-white border border-white/10 transition-colors flex items-center gap-1 text-[11px] disabled:opacity-50 cursor-pointer"
              title="Refresh Pipeline Gate Status"
            >
              <RefreshCw className={`w-3 h-3 ${isLoading ? 'animate-spin text-cyan-400' : ''}`} />
              <span>Verify</span>
            </button>
          )}

          {onRunPipeline && (
            <button
              onClick={onRunPipeline}
              disabled={isProcessing || !pointCloudOk}
              className="px-3 py-1 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-black font-bold transition-all shadow-glow-cyan flex items-center gap-1.5 text-[11px] disabled:opacity-40 cursor-pointer"
              title="Execute full perception pipeline sequentially"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Run Sequential Pipeline</span>
                </>
              )}
            </button>
          )}

          <button
            onClick={() => setShowDiagnostics(!showDiagnostics)}
            className="px-2.5 py-1 rounded-lg bg-dark-850 hover:bg-dark-800 text-gray-300 hover:text-white border border-white/10 transition-colors flex items-center gap-1 text-[11px] cursor-pointer"
            title="Toggle Developer Diagnostics"
          >
            <Terminal className="w-3.5 h-3.5 text-purple-400" />
            <span>Diagnostics</span>
            {showDiagnostics ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>
        </div>
      </div>

      {/* 5 Sequential Gate Chips */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
        {/* Gate 1: Point Cloud */}
        <div className={`p-2 rounded-xl border flex items-center justify-between ${
          pointCloudOk ? 'bg-emerald-950/20 border-emerald-500/30 text-emerald-300' : 'bg-dark-850/60 border-white/5 text-gray-400'
        }`}>
          <div className="flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 shrink-0" />
            <div>
              <div className="text-[10px] text-gray-400 uppercase">1. Point Cloud</div>
              <div className="font-bold text-[11px] text-white">
                {pointCloudOk ? `${status?.point_count} pts` : 'Missing'}
              </div>
            </div>
          </div>
          {pointCloudOk ? <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" /> : <AlertCircle className="w-4 h-4 text-gray-600 shrink-0" />}
        </div>

        {/* Gate 2: Semantic Labels */}
        <div className={`p-2 rounded-xl border flex items-center justify-between ${
          labelsOk ? 'bg-emerald-950/20 border-emerald-500/30 text-emerald-300' : 'bg-dark-850/60 border-white/5 text-gray-400'
        }`}>
          <div className="flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 shrink-0" />
            <div>
              <div className="text-[10px] text-gray-400 uppercase">2. Fast-FRNet</div>
              <div className="font-bold text-[11px] text-white">
                {labelsOk ? `${status?.semantic_label_count} labels` : status?.semantic_labels_available ? 'Count Mismatch' : 'Not Run'}
              </div>
            </div>
          </div>
          {labelsOk ? <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" /> : <AlertCircle className="w-4 h-4 text-amber-500 shrink-0" />}
        </div>

        {/* Gate 3: Geometric Terrain */}
        <div className={`p-2 rounded-xl border flex items-center justify-between ${
          terrainOk ? 'bg-emerald-950/20 border-emerald-500/30 text-emerald-300' : 'bg-dark-850/60 border-white/5 text-gray-400'
        }`}>
          <div className="flex items-center gap-1.5">
            <Mountain className="w-3.5 h-3.5 shrink-0" />
            <div>
              <div className="text-[10px] text-gray-400 uppercase">3. Terrain</div>
              <div className="font-bold text-[11px] text-white">
                {terrainOk ? 'Analyzed' : 'Pending'}
              </div>
            </div>
          </div>
          {terrainOk ? <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" /> : <Clock className="w-4 h-4 text-gray-600 shrink-0" />}
        </div>

        {/* Gate 4: Object Detection */}
        <div className={`p-2 rounded-xl border flex items-center justify-between ${
          objectsOk ? 'bg-emerald-950/20 border-emerald-500/30 text-emerald-300' : 'bg-dark-850/60 border-white/5 text-gray-400'
        }`}>
          <div className="flex items-center gap-1.5">
            <Box className="w-3.5 h-3.5 shrink-0" />
            <div>
              <div className="text-[10px] text-gray-400 uppercase">4. 3D Objects</div>
              <div className="font-bold text-[11px] text-white">
                {objectsOk ? `${status?.object_instance_count} clusters` : !labelsOk ? 'Needs Labels' : 'Pending'}
              </div>
            </div>
          </div>
          {objectsOk ? <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" /> : <AlertCircle className="w-4 h-4 text-gray-600 shrink-0" />}
        </div>

        {/* Gate 5: 2.5D Adaptive Grid */}
        <div className={`p-2 rounded-xl border flex items-center justify-between ${
          mapOk ? 'bg-emerald-950/20 border-emerald-500/30 text-emerald-300' : 'bg-dark-850/60 border-white/5 text-gray-400'
        }`}>
          <div className="flex items-center gap-1.5">
            <Grid className="w-3.5 h-3.5 shrink-0" />
            <div>
              <div className="text-[10px] text-gray-400 uppercase">5. 2.5D Grid</div>
              <div className="font-bold text-[11px] text-white">
                {mapOk ? `${status?.adaptive_grid_cell_count} cells` : 'Pending'}
              </div>
            </div>
          </div>
          {mapOk ? <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" /> : <Clock className="w-4 h-4 text-gray-600 shrink-0" />}
        </div>
      </div>

      {/* Expandable Developer Diagnostics Panel */}
      {showDiagnostics && (
        <div className="p-3 rounded-xl bg-dark-950 border border-purple-500/30 space-y-2.5 animate-fadeIn">
          <div className="flex items-center justify-between text-purple-300 font-bold border-b border-white/5 pb-1.5">
            <span className="flex items-center gap-1.5">
              <Terminal className="w-4 h-4" />
              <span>Developer Artifact Diagnostics</span>
            </span>
            <span className="text-[10px] text-gray-500">Strict Canonical Identity Enforcement</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[11px]">
            <div className="p-2 rounded bg-dark-900 border border-white/5 space-y-1">
              <div className="text-gray-400">Canonical Frame ID:</div>
              <div className="text-cyan-300 font-bold">{frameId}</div>
              <div className="text-[10px] text-gray-500">
                Point Path: <code className="text-gray-300">backend/data/frames/{frameId}/points.npz</code> ({status?.point_count ?? 0} pts)
              </div>
              <div className="text-[10px] text-gray-500">
                Labels Path: <code className="text-gray-300">backend/data/labels/{frameId}_labels.npz</code> ({status?.semantic_label_count ?? 0} labels)
              </div>
            </div>

            <div className="p-2 rounded bg-dark-900 border border-white/5 space-y-1">
              <div className="text-gray-400">Downstream Artifacts:</div>
              <div className="text-[10px] text-gray-500">
                Prediction File: <code className="text-gray-300">backend/data/predictions/{frameId}.label</code>
              </div>
              <div className="text-[10px] text-gray-500">
                Terrain Path: <code className="text-gray-300">backend/data/terrain/{frameId}.json</code>
              </div>
              <div className="text-[10px] text-gray-500">
                Objects Path: <code className="text-gray-300">backend/data/objects/{frameId}.json</code> ({status?.object_instance_count ?? 0} objects)
              </div>
              <div className="text-[10px] text-gray-500">
                Adaptive Map Path: <code className="text-gray-300">backend/data/maps/{frameId}.json</code> ({status?.adaptive_grid_cell_count ?? 0} cells)
              </div>
            </div>
          </div>

          {status?.last_error && (
            <div className="p-2 rounded bg-red-950/30 border border-red-500/40 text-red-300 text-[11px] flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>Gate Diagnostic Warning: {status.last_error}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
