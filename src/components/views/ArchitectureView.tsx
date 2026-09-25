import React from 'react';

interface StageNodeProps {
  stage: string;
  title: string;
  subtitle: string;
  meta?: string;
  tooltip: {
    component: string;
    file: string;
  };
  accent?: boolean;
  children?: React.ReactNode;
}

const StageNode: React.FC<StageNodeProps> = ({
  stage,
  title,
  subtitle,
  meta,
  tooltip,
  accent = false,
  children,
}) => {
  return (
    <div className="relative group w-full">
      {/* Hover Tooltip */}
      <div className="pointer-events-none absolute -top-8 left-4 opacity-0 group-hover:opacity-100 transition-opacity duration-150 z-30 bg-dark-900 border border-white/15 rounded-md px-2.5 py-1 text-[11px] font-mono text-gray-300 shadow-xl whitespace-nowrap hidden sm:flex items-center gap-2">
        <span className="text-cyan-400 font-semibold">{tooltip.component}</span>
        <span className="text-white/20">|</span>
        <span className="text-gray-400">{tooltip.file}</span>
      </div>

      {/* Main Node Card */}
      <div
        className={`w-full rounded-xl bg-dark-900/90 border transition-all duration-200 p-4 relative text-left border-l-2 ${
          accent
            ? 'border-cyan-500/60 border-l-cyan-400 shadow-glow-cyan'
            : 'border-white/10 border-l-cyan-500/80 hover:border-cyan-500/40 hover:shadow-glow-cyan'
        }`}
      >
        <div className="flex items-center justify-between gap-2 mb-1.5">
          <span className="text-[10px] font-mono font-bold text-cyan-400 tracking-wider uppercase">
            STAGE {stage}
          </span>
          {meta && (
            <span className="text-[10px] font-mono text-gray-400 bg-dark-950/80 border border-white/10 px-2 py-0.5 rounded">
              {meta}
            </span>
          )}
        </div>

        <h3 className="text-sm md:text-base font-bold text-white tracking-tight mb-1">
          {title}
        </h3>

        <p className="text-xs text-gray-300 font-sans leading-relaxed">
          {subtitle}
        </p>

        {children}
      </div>
    </div>
  );
};

const FlowArrow: React.FC<{ label?: string }> = ({ label }) => (
  <div className="flex flex-col items-center my-2 select-none">
    {label && (
      <span className="text-[10px] font-mono text-gray-400 mb-1 px-2 py-0.5 rounded bg-dark-900 border border-white/10">
        {label}
      </span>
    )}
    <div className="w-[2px] h-5 bg-cyan-500/60" />
    <div className="w-0 h-0 border-l-[4px] border-l-transparent border-r-[4px] border-r-transparent border-t-[6px] border-t-cyan-400" />
  </div>
);

const BranchSplit: React.FC<{ label?: string }> = ({ label }) => (
  <div className="flex flex-col items-center my-2 w-full select-none">
    {label && (
      <span className="text-[10px] font-mono text-gray-400 mb-1.5 px-2 py-0.5 rounded bg-dark-900 border border-white/10">
        {label}
      </span>
    )}
    <div className="w-[2px] h-4 bg-cyan-500/60" />
    <div className="w-[80%] max-w-md h-[2px] bg-cyan-500/60" />
    <div className="w-[80%] max-w-md flex justify-between">
      <div className="flex flex-col items-center">
        <div className="w-[2px] h-4 bg-cyan-500/60" />
        <div className="w-0 h-0 border-l-[4px] border-l-transparent border-r-[4px] border-r-transparent border-t-[6px] border-t-cyan-400" />
      </div>
      <div className="flex flex-col items-center">
        <div className="w-[2px] h-4 bg-cyan-500/60" />
        <div className="w-0 h-0 border-l-[4px] border-l-transparent border-r-[4px] border-r-transparent border-t-[6px] border-t-cyan-400" />
      </div>
    </div>
  </div>
);

const BranchMerge: React.FC<{ label?: string }> = ({ label }) => (
  <div className="flex flex-col items-center my-2 w-full select-none">
    <div className="w-[80%] max-w-md flex justify-between">
      <div className="w-[2px] h-4 bg-cyan-500/60" />
      <div className="w-[2px] h-4 bg-cyan-500/60" />
    </div>
    <div className="w-[80%] max-w-md h-[2px] bg-cyan-500/60" />
    <div className="w-[2px] h-4 bg-cyan-500/60" />
    {label && (
      <span className="text-[10px] font-mono text-gray-400 my-1.5 px-2 py-0.5 rounded bg-dark-900 border border-white/10">
        {label}
      </span>
    )}
    <div className="w-0 h-0 border-l-[4px] border-l-transparent border-r-[4px] border-r-transparent border-t-[6px] border-t-cyan-400" />
  </div>
);

export const ArchitectureView: React.FC = () => {
  return (
    <div className="min-h-full w-full py-6 md:py-8 px-4 font-sans flex flex-col items-center">
      {/* Header */}
      <div className="text-center mb-8 max-w-xl">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-mono font-medium mb-3">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span>System Pipeline</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight text-white font-sans">
          ARCHITECTURE
        </h1>
        <p className="text-xs md:text-sm text-gray-400 font-mono mt-1">
          LiDAR-X Perception & Foveated Mapping Pipeline
        </p>
      </div>

      {/* Main Flowchart Canvas */}
      <div className="w-full max-w-2xl flex flex-col items-center">
        {/* STAGE 01: Ingestion */}
        <StageNode
          stage="01"
          title="LidarParser + SequenceReplayService"
          subtitle="Raw LiDAR ingestion & calibrated sequence replay"
          meta="Input: .bin / .pcd → Output: [x,y,z,i]"
          tooltip={{
            component: 'LidarParser & SequenceReplay',
            file: 'backend/app/services/lidar_parser.py',
          }}
        />

        <FlowArrow />

        {/* STAGE 02: Scene Analysis */}
        <StageNode
          stage="02"
          title="SceneAnalysisEngine"
          subtitle="Sensor beam ring count & scene identification (< 15 ms)"
          meta="VLP-32C vs HDL-64E"
          tooltip={{
            component: 'SceneAnalysisEngine',
            file: 'backend/app/services/scene_analysis.py',
          }}
        />

        {/* STAGE 03: Dual Model Branching */}
        <BranchSplit label="Disambiguated Sensor Route" />

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full">
          <StageNode
            stage="03A"
            title="RELLIS-3D Branch"
            subtitle="Off-road rugged domain (mud, water, foliage)"
            meta="VLP-32C · 32 beams"
            tooltip={{
              component: 'FastFRNet RELLIS-3D',
              file: 'backend/app/models/weights/best_frnet_rellis.pth',
            }}
          >
            <div className="mt-2 text-[10px] font-mono text-gray-400 bg-dark-950/80 p-2 rounded-lg border border-white/10 space-y-0.5">
              <div>Weight: <span className="text-gray-200">best_frnet_rellis.pth</span></div>
              <div>FOV: <span className="text-gray-200">+15° to -25° (32×512)</span></div>
            </div>
          </StageNode>

          <StageNode
            stage="03B"
            title="SemanticKITTI Branch"
            subtitle="Structured urban domain (road, vehicle, sidewalk)"
            meta="HDL-64E · 64 beams"
            tooltip={{
              component: 'FastFRNet SemanticKITTI',
              file: 'backend/app/models/weights/best_frnet_semantickitti.pth',
            }}
          >
            <div className="mt-2 text-[10px] font-mono text-gray-400 bg-dark-950/80 p-2 rounded-lg border border-white/10 space-y-0.5">
              <div>Weight: <span className="text-gray-200">best_frnet_semantickitti.pth</span></div>
              <div>FOV: <span className="text-gray-200">+3° to -25° (64×512)</span></div>
            </div>
          </StageNode>
        </div>

        <BranchMerge label="Convergence to Inference" />

        {/* STAGE 04: Fast-FRNet Inference */}
        <StageNode
          stage="04"
          title="Fast-FRNet Inference"
          subtitle="Spherical range projection → Frustum-Residual ConvNet → Point back-projection"
          meta="H×W: 32×512 / 64×512"
          accent={true}
          tooltip={{
            component: 'FastFRNetInferenceService',
            file: 'backend/app/models/fast_frnet.py',
          }}
        />

        <FlowArrow />

        {/* STAGE 05: Semantic Point Cloud */}
        <StageNode
          stage="05"
          title="Semantic Point Cloud"
          subtitle="Per-point semantic labels back-projected to 3D Cartesian coordinates"
          meta="20 Semantic Classes"
          tooltip={{
            component: 'SemanticClassRegistry',
            file: 'backend/app/services/semantic_registry.py',
          }}
        />

        {/* STAGE 06: Parallel Processing */}
        <BranchSplit label="Parallel Processing Split" />

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full">
          <StageNode
            stage="06A"
            title="Instance Clustering"
            subtitle="DBSCAN spatial grouping + 2D PCA yaw heading decomposition"
            meta="3D OBBs"
            tooltip={{
              component: 'InstanceClusteringService',
              file: 'backend/app/services/instance_clustering.py',
            }}
          >
            <div className="mt-2 text-[10px] font-mono text-gray-400 bg-dark-950/80 p-2 rounded-lg border border-white/10 space-y-0.5">
              <div>Params: <span className="text-gray-200">ε = 0.8m · min_samples = 5</span></div>
              <div>Yaw: <span className="text-gray-200">θ = arctan2(v_1y, v_1x)</span></div>
            </div>
          </StageNode>

          <StageNode
            stage="06B"
            title="Terrain Analysis"
            subtitle="Local surface normal SVD, slope angles, and elevation roughness"
            meta="Traversability Cost"
            tooltip={{
              component: 'TerrainAnalysisEngine',
              file: 'backend/app/services/terrain_analysis.py',
            }}
          >
            <div className="mt-2 text-[10px] font-mono text-gray-400 bg-dark-950/80 p-2 rounded-lg border border-white/10 space-y-0.5">
              <div>Normals: <span className="text-gray-200">3×3 Covariance SVD</span></div>
              <div>Metrics: <span className="text-gray-200">Slope (25° max) · Step (0.15m)</span></div>
            </div>
          </StageNode>
        </div>

        <BranchMerge label="Fused Perception States" />

        {/* STAGE 07: Foveated 2.5D Adaptive Grid (Main Innovation) */}
        <StageNode
          stage="07"
          title="AdaptiveGridService (Foveated 2.5D Grid)"
          subtitle="Variable-resolution foveated hierarchy with strict 2:1 integer quadtree alignment"
          meta="grid_policy.yaml"
          accent={true}
          tooltip={{
            component: 'AdaptiveGridService',
            file: 'backend/app/services/adaptive_grid.py',
          }}
        >
          {/* Nested Resolution Zones Representation */}
          <div className="mt-3 p-3.5 rounded-xl bg-dark-950/80 border border-white/10 text-left">
            <div className="flex items-center justify-between text-[10px] font-mono text-gray-400 mb-2.5 uppercase">
              <span>Resolution Hierarchy</span>
              <span className="text-cyan-400 font-bold">2:1 Integer Ratio</span>
            </div>

            {/* FAR ZONE */}
            <div className="p-3 rounded-lg border border-white/10 bg-dark-850/60">
              <div className="flex items-center justify-between text-[11px] font-mono text-gray-400 mb-2">
                <span className="font-bold text-white">FAR ZONE: 28 – 80 m</span>
                <span className="px-1.5 py-0.5 rounded bg-dark-950 border border-white/10 text-[10px]">
                  3.00 m cells · Coarse Horizon
                </span>
              </div>

              {/* MID ZONE */}
              <div className="p-2.5 rounded-lg border border-white/10 bg-dark-900/80">
                <div className="flex items-center justify-between text-[11px] font-mono text-gray-400 mb-2">
                  <span className="font-bold text-white">MID ZONE: 12 – 28 m</span>
                  <span className="px-1.5 py-0.5 rounded bg-dark-950 border border-white/10 text-[10px]">
                    1.50 m cells · Transition
                  </span>
                </div>

                {/* NEAR ZONE */}
                <div className="p-2 rounded-lg border border-cyan-500/40 bg-cyan-500/10">
                  <div className="flex items-center justify-between text-[11px] font-mono">
                    <span className="font-bold text-cyan-300">NEAR ZONE: 0 – 12 m</span>
                    <span className="px-1.5 py-0.5 rounded bg-cyan-500/20 border border-cyan-500/40 text-cyan-300 text-[10px] font-bold">
                      0.75 m cells · High Detail
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </StageNode>

        <FlowArrow />

        {/* STAGE 08: WebSocket Replay Server */}
        <StageNode
          stage="08"
          title="WebSocket Replay Server"
          subtitle="Real-time frame broadcast via asynchronous WebSocket stream"
          meta="/replay/{session_id}/stream"
          tooltip={{
            component: 'WebSocket Route',
            file: 'backend/app/api/routes/replay.py',
          }}
        />

        <FlowArrow />

        {/* STAGE 09: Final Perception Output */}
        <div className="w-full relative group">
          <div className="pointer-events-none absolute -top-8 left-4 opacity-0 group-hover:opacity-100 transition-opacity duration-150 z-30 bg-dark-900 border border-white/15 rounded-md px-2.5 py-1 text-[11px] font-mono text-gray-300 shadow-xl whitespace-nowrap hidden sm:flex items-center gap-2">
            <span className="text-emerald-400 font-semibold">LiDAR-X Frontend Client</span>
            <span className="text-white/20">|</span>
            <span className="text-gray-400">src/components/views/MappingConsoleView.tsx</span>
          </div>

          <div className="w-full rounded-xl bg-dark-900/90 border border-emerald-500/40 p-5 relative text-left border-l-2 border-l-emerald-400 shadow-[0_0_20px_-5px_rgba(16,185,129,0.2)]">
            <div className="flex items-center justify-between gap-2 mb-2">
              <span className="text-[10px] font-mono font-bold text-emerald-400 tracking-wider uppercase">
                STAGE 09 · FINAL OUTPUT
              </span>
              <span className="text-[10px] font-mono text-emerald-300 bg-emerald-500/10 border border-emerald-500/30 px-2 py-0.5 rounded font-bold">
                SYNCHRONIZED CLIENT
              </span>
            </div>

            <h3 className="text-base md:text-lg font-bold text-white tracking-tight mb-1">
              LiDAR-X Perception Output
            </h3>

            <p className="text-xs text-gray-300 font-sans mb-3">
              Unified 3D sensor and 2.5D spatial autonomy visualization in Three.js and Canvas 2D.
            </p>

            {/* Output Components Badges */}
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 pt-2.5 border-t border-white/10 text-center">
              <div className="p-2 rounded-lg bg-dark-950/80 border border-white/10 text-[11px] font-mono text-gray-200">
                3D Point Cloud
              </div>
              <div className="p-2 rounded-lg bg-dark-950/80 border border-white/10 text-[11px] font-mono text-gray-200">
                2.5D Foveated Map
              </div>
              <div className="p-2 rounded-lg bg-dark-950/80 border border-white/10 text-[11px] font-mono text-gray-200">
                Elevation (Z)
              </div>
              <div className="p-2 rounded-lg bg-dark-950/80 border border-white/10 text-[11px] font-mono text-gray-200">
                Traversability
              </div>
              <div className="p-2 rounded-lg bg-dark-950/80 border border-white/10 text-[11px] font-mono text-gray-200 col-span-2 sm:col-span-1">
                Detected Objects
              </div>
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="mt-8 pt-4 border-t border-white/10 w-full flex flex-wrap items-center justify-center gap-6 text-[11px] font-mono text-gray-400">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-[2px] bg-cyan-400" />
            <span>Primary Pipeline Flow</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-[2px] bg-cyan-500/40" />
            <span>Branching & Convergence</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500/20 border border-emerald-400" />
            <span>Perception Output</span>
          </div>
        </div>
      </div>
    </div>
  );
};
