import React, { useRef } from 'react';
import {
  Play,
  Pause,
  Square,
  SkipBack,
  SkipForward,
  Radio,
  Loader2,
  FileArchive,
  Zap,
} from 'lucide-react';
import { ReplayState, ReplayDataMode } from '../../types/replay';
import { Badge } from '../common/Badge';

export type ObjectDetectionActionState =
  | 'idle'
  | 'checking_labels'
  | 'mapping_classes'
  | 'clustering'
  | 'instances_found'
  | 'no_dynamic_points'
  | 'no_valid_clusters'
  | 'error';

interface SequenceReplayBarProps {
  sessionId: string | null;
  sequenceName: string;
  playbackState: ReplayState;
  currentFrameIndex: number;
  totalFrames: number;
  fps: number;
  liveFps?: number;
  dataMode: ReplayDataMode;
  semanticSource?: 'LIVE GEOMETRIC' | 'LIVE GEOMETRIC — FAST-FRNET PROCESSING' | 'LIVE FAST-FRNET' | 'LIVE GEOMETRIC \u2014 SALSANEXT PROCESSING' | 'LIVE SALSANEXT' | 'GROUND TRUTH' | 'PRECOMPUTED';
  processingTimeMs?: number;
  isUploading: boolean;
  hasPredictions?: boolean;
  detectionState?: ObjectDetectionActionState;
  detectedCount?: number;
  detectionMessage?: string;
  onRunObjectDetection?: () => void;
  onUploadZip: (file: File) => void;
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  onStepBackward: () => void;
  onStepForward: () => void;
  onSeek: (frameIndex: number) => void;
  onSetFps: (fps: number) => void;
  precompute?: {
    isRunning: boolean;
    isComplete: boolean;
    percentComplete: number;
    processedFrames: number;
    totalFrames: number;
    etaSeconds: number;
    currentStage?: string;
    failedCount?: number;
    errorMessage?: string | null;
  } | null;
  onTriggerPrecompute?: () => void;
  onCancelPrecompute?: () => void;
  isWebSocketConnected?: boolean;
}

export const SequenceReplayBar: React.FC<SequenceReplayBarProps> = ({
  sessionId,
  sequenceName,
  playbackState,
  currentFrameIndex,
  totalFrames,
  fps,
  liveFps,
  dataMode,
  semanticSource = 'LIVE GEOMETRIC',
  processingTimeMs = 0,
  isUploading,
  hasPredictions = true,
  detectionState = 'idle',
  detectedCount = 0,
  detectionMessage,
  onRunObjectDetection,
  onUploadZip,
  onPlay,
  onPause,
  onStop,
  onStepBackward,
  onStepForward,
  onSeek,
  onSetFps,
  precompute,
  onTriggerPrecompute,
  onCancelPrecompute,
  isWebSocketConnected,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onUploadZip(file);
    }
  };

  const isPlaying = playbackState === 'playing';
  const hasFrames = totalFrames > 0;

  const getSemanticSourceBadge = (source?: string) => {
    switch (source) {
      case 'GROUND TRUTH':
        return (
          <Badge variant="emerald" className="text-[10px] font-mono font-bold tracking-wider">
            GROUND TRUTH
          </Badge>
        );
      case 'PRECOMPUTED':
        return null;
      case 'LIVE FAST-FRNET':
      case 'LIVE SALSANEXT':
        return (
          <Badge variant="purple" className="text-[10px] font-mono font-bold tracking-wider">
            LIVE FAST-FRNET
          </Badge>
        );
      case 'LIVE GEOMETRIC — FAST-FRNET PROCESSING':
      case 'LIVE GEOMETRIC \u2014 SALSANEXT PROCESSING':
        return (
          <Badge variant="warning" className="text-[10px] font-mono font-bold tracking-wider animate-pulse">
            GEOMETRIC — FAST-FRNET &#9654;
          </Badge>
        );
      case 'LIVE GEOMETRIC':
      default:
        return (
          <Badge variant="warning" className="text-[10px] font-mono font-bold tracking-wider">
            LIVE GEOMETRIC (CPU)
          </Badge>
        );
    }
  };

  const getStateBadge = (state: ReplayState) => {
    // Dynamic real measured FPS: preference to live delivery rate, fallback to frame pipeline compute rate
    const currentRealFps =
      liveFps && liveFps > 0
        ? liveFps
        : processingTimeMs && processingTimeMs > 0
        ? 1000.0 / processingTimeMs
        : null;

    switch (state) {
      case 'playing':
        return (
          <Badge variant="cyan" dot className="text-[10px] animate-pulse font-mono font-medium">
            LIVE STREAMING • {currentRealFps !== null ? `${currentRealFps.toFixed(1)} FPS` : 'STREAMING...'}
          </Badge>
        );
      case 'paused':
        return (
          <Badge variant="warning" className="text-[10px]">
            PAUSED
          </Badge>
        );
      case 'completed':
        return (
          <Badge variant="purple" className="text-[10px]">
            COMPLETED
          </Badge>
        );
      case 'failed':
        return (
          <Badge variant="neutral" className="text-[10px] text-red-400 border-red-500/40">
            FAILED
          </Badge>
        );
      default:
        return (
          <Badge variant="emerald" className="text-[10px]">
            READY
          </Badge>
        );
    }
  };

  return (
    <div className="bg-dark-900/90 backdrop-blur-md border border-cyan-500/30 rounded-2xl p-3.5 md:p-4 space-y-3 shadow-glow-cyan">
      {/* Top Header & Sequence Meta */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 border-b border-white/10 pb-2.5">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-8 h-8 rounded-xl bg-cyan-500/15 border border-cyan-500/40 flex items-center justify-center shrink-0 shadow-glow-cyan">
            <Radio className="w-4 h-4 text-cyan-400 animate-pulse" />
          </div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-xs md:text-sm font-bold text-white uppercase tracking-wider font-mono truncate">
                LiDAR Sequence Video Feed Replay
              </h3>
              {getStateBadge(playbackState)}
              {getSemanticSourceBadge(semanticSource)}
            </div>
            <p className="text-[11px] font-mono text-gray-400 truncate">
              {sessionId
                ? `Sequence: "${sequenceName}"`
                : 'Upload a SemanticKITTI sequence ZIP (velodyne/*.bin + predictions/*.label) for continuous feed replay.'}
            </p>
          </div>
        </div>

        {/* Upload Sequence ZIP button */}
        <div className="flex flex-wrap items-center gap-2 self-start sm:self-auto shrink-0">
          <input
            ref={fileInputRef}
            type="file"
            accept=".zip"
            onChange={handleFileChange}
            className="hidden"
          />

          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="text-xs font-mono px-3.5 py-1.5 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 transition-colors flex items-center gap-1.5 font-bold shadow-glow-cyan"
            title="Upload SemanticKITTI sequence package (.zip)"
          >
            {isUploading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Ingesting ZIP...</span>
              </>
            ) : (
              <>
                <FileArchive className="w-3.5 h-3.5 text-cyan-400" />
                <span>Upload Sequence ZIP</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Real Precomputation Progress Bar */}
      {precompute?.isRunning && (
        <div className="bg-dark-850/90 border border-cyan-500/40 rounded-xl p-3 space-y-2 shadow-glow-cyan">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
            <span className="flex items-center gap-2 text-cyan-300 font-bold">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-cyan-400" />
              <span>{precompute.currentStage || 'Processing frames with Fast-FRNet...'}</span>
            </span>
            <div className="flex items-center gap-3 text-gray-400">
              <span>
                <strong className="text-white font-bold">{precompute.processedFrames.toLocaleString()}</strong> / {precompute.totalFrames.toLocaleString()} frames ({precompute.percentComplete.toFixed(1)}%)
              </span>
              {precompute.etaSeconds > 0 && (
                <span className="text-cyan-400 font-medium">ETA: {Math.ceil(precompute.etaSeconds)}s</span>
              )}
              {onCancelPrecompute && (
                <button
                  onClick={onCancelPrecompute}
                  className="px-2 py-0.5 rounded bg-red-500/20 hover:bg-red-500/30 text-red-300 border border-red-500/40 text-[10px] font-bold"
                >
                  Cancel
                </button>
              )}
            </div>
          </div>
          <div className="w-full bg-dark-700 rounded-full h-2 overflow-hidden">
            <div
              className="bg-gradient-to-r from-cyan-500 to-blue-500 h-2 rounded-full transition-all duration-300 shadow-glow-cyan"
              style={{ width: `${Math.min(100, Math.max(0, precompute.percentComplete))}%` }}
            />
          </div>
        </div>
      )}

      {/* Media Player Controls & Frame Scrubber */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pt-1">
        {/* Play / Pause / Stop / Step buttons */}
        <div className="flex items-center gap-1.5 shrink-0">
          <button
            onClick={onStepBackward}
            disabled={!hasFrames || currentFrameIndex <= 0}
            className="p-2 rounded-lg bg-dark-850 hover:bg-dark-750 text-gray-300 border border-white/10 disabled:opacity-40 transition-colors"
            title="Previous Frame"
          >
            <SkipBack className="w-4 h-4" />
          </button>

          {isPlaying ? (
            <button
              onClick={onPause}
              disabled={!hasFrames}
              className="px-4 py-2 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 font-mono text-xs font-bold transition-colors flex items-center gap-1.5 shadow-glow-amber"
              title="Pause Replay"
            >
              <Pause className="w-4 h-4 fill-current" />
              <span>Pause</span>
            </button>
          ) : (
            <button
              onClick={onPlay}
              disabled={!hasFrames}
              className="px-4 py-2 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/50 font-mono text-xs font-bold transition-colors flex items-center gap-1.5 shadow-glow-cyan disabled:opacity-40"
              title="Start / Resume Replay"
            >
              <Play className="w-4 h-4 fill-current" />
              <span>Play Feed</span>
            </button>
          )}

          <button
            onClick={onStepForward}
            disabled={!hasFrames || currentFrameIndex >= totalFrames - 1}
            className="p-2 rounded-lg bg-dark-850 hover:bg-dark-750 text-gray-300 border border-white/10 disabled:opacity-40 transition-colors"
            title="Next Frame"
          >
            <SkipForward className="w-4 h-4" />
          </button>

          <button
            onClick={onStop}
            disabled={!hasFrames}
            className="p-2 rounded-lg bg-dark-850 hover:bg-dark-750 text-gray-400 hover:text-white border border-white/10 disabled:opacity-40 transition-colors"
            title="Stop & Rewind to Frame 0"
          >
            <Square className="w-4 h-4" />
          </button>
        </div>

        {/* Frame Seek Scrubber Slider */}
        <div className="flex-1 flex items-center gap-3 min-w-0 bg-dark-850/80 px-3 py-1.5 rounded-xl border border-white/5">
          <span className="text-[11px] font-mono text-cyan-400 font-bold shrink-0">
            {hasFrames ? (currentFrameIndex + 1).toLocaleString() : '0'}
          </span>

          <input
            type="range"
            min={0}
            max={Math.max(0, totalFrames - 1)}
            value={currentFrameIndex}
            onChange={(e) => onSeek(parseInt(e.target.value, 10))}
            disabled={!hasFrames}
            className="flex-1 h-1.5 bg-dark-700 rounded-lg appearance-none cursor-pointer accent-cyan-400 disabled:opacity-40"
          />

          <span className="text-[11px] font-mono text-gray-500 shrink-0">
            / {hasFrames ? totalFrames.toLocaleString() : '0'}
          </span>
        </div>

        {/* Speed Controls */}
        <div className="flex items-center gap-1 bg-dark-850 px-2.5 py-1 rounded-xl border border-white/5 shrink-0">
          <span className="text-[10px] font-mono text-gray-400 mr-1 uppercase">Speed:</span>
          {[
            { label: '0.5x', value: 5 },
            { label: '1x', value: 10 },
            { label: '2x', value: 20 },
            { label: '30 FPS', value: 30 },
            { label: '60 MAX', value: 60 },
          ].map((preset) => (
            <button
              key={preset.value}
              onClick={() => onSetFps(preset.value)}
              className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold transition-colors ${
                fps === preset.value
                  ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-500/50'
                  : 'bg-dark-800 text-gray-400 hover:text-white border border-transparent'
              }`}
              title={`Set playback speed to ${preset.value} FPS`}
            >
              {preset.label}
            </button>
          ))}
        </div>

        {isWebSocketConnected === false && (
          <div className="flex items-center gap-2 shrink-0 text-xs font-mono">
            <div
              data-testid="ws-disconnected"
              className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-amber-500/10 border border-amber-500/40 text-[10px] text-amber-300"
              title="WebSocket disconnected - reconnecting with backoff"
            >
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              <span>WS reconnecting…</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
