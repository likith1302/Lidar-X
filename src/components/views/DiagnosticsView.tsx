import React, { useState, useEffect } from 'react';
import { useAppState } from '../../context/AppStateContext';
import { PipelineGateBanner } from '../console/PipelineGateBanner';
import { ObjectPipelineDiagnostics } from '../console/ObjectPipelineDiagnostics';
import { PipelineService } from '../../services/pipelineService';
import { PipelineStatusResponse } from '../../types/pipeline';
import { SYSTEM_CONNECTIONS } from '../../mocks/mockArchitecture';
import { Badge } from '../common/Badge';
import { Activity, ShieldCheck, Cpu } from 'lucide-react';

export const DiagnosticsView: React.FC = () => {
  const {
    activeLidarFrame,
    activeSemanticFrame,
    activeObjectDetection,
    activeTracks,
    selectedFrameId,
  } = useAppState();

  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const fetchStatus = async () => {
    setIsLoading(true);
    const frameId = selectedFrameId || activeLidarFrame?.frameId || '000000.bin';
    try {
      const res = await PipelineService.getPipelineStatus(frameId);
      if (res.success && res.data) {
        setPipelineStatus(res.data);
      }
    } catch {
      // Fallback gracefully
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, [selectedFrameId]);

  return (
    <div className="max-w-6xl mx-auto space-y-6 pb-12 font-sans text-gray-200">
      {/* Header */}
      <header className="border-b border-white/10 pb-4 flex flex-col sm:flex-row sm:items-end justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <ShieldCheck className="w-5 h-5 text-emerald-400" />
            <h1 className="text-xl font-bold tracking-wider text-white uppercase font-mono">
              SYSTEM & PIPELINE DIAGNOSTICS
            </h1>
          </div>
          <p className="text-xs text-gray-400 font-mono">
            Subsystem Health, Architectural Gate Integrity & Perception Diagnostics
          </p>
        </div>

        <Badge variant="emerald" className="text-[11px] font-mono">
          SYSTEM ACTIVE
        </Badge>
      </header>

      {/* 1. Pipeline Gate Banner */}
      <section className="space-y-2">
        <h2 className="text-xs font-mono font-semibold tracking-wider text-gray-400 uppercase">
          Pipeline Gate Verification
        </h2>
        <PipelineGateBanner
          status={pipelineStatus}
          isLoading={isLoading}
          onRefresh={fetchStatus}
        />
      </section>

      {/* 2. Object Pipeline Diagnostics */}
      <section className="space-y-2">
        <h2 className="text-xs font-mono font-semibold tracking-wider text-gray-400 uppercase">
          Perception Object & MOT Tracking Diagnostics
        </h2>
        <ObjectPipelineDiagnostics
          frameId={selectedFrameId || activeLidarFrame?.frameId || '000000.bin'}
          hasPredictions={Boolean(activeSemanticFrame && activeSemanticFrame.sample_labeled_points.length > 0)}
          predictionFileName="Fast-FRNet (SemanticKITTI/RELLIS)"
          totalPoints={activeLidarFrame?.points.length || 122829}
          totalLabels={activeSemanticFrame?.sample_labeled_points.length || 122829}
          semanticFrame={activeSemanticFrame}
          detectionResult={activeObjectDetection}
          tracks={activeTracks}
          frameCount={activeTracks.length > 0 ? 5 : 1}
        />
      </section>

      {/* 3. Subsystem Health Connections */}
      <section className="space-y-3">
        <h2 className="text-xs font-mono font-semibold tracking-wider text-gray-400 uppercase">
          Subsystem Service Health
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {SYSTEM_CONNECTIONS.map((conn) => (
            <div
              key={conn.id}
              className="bg-[#10141d] border border-white/10 rounded-xl p-4 space-y-2 font-mono"
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-white text-xs">{conn.name}</span>
                <Badge variant={conn.badgeVariant} className="text-[10px]">
                  {conn.statusText}
                </Badge>
              </div>
              <p className="text-[11px] text-gray-400 font-sans leading-relaxed">
                {conn.description}
              </p>
              <div className="text-[10px] text-gray-500 truncate pt-1 border-t border-white/5">
                {conn.endpointPlaceholder}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};
