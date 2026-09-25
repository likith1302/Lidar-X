import React from 'react';

/**
 * STATIC BENCHMARK CONSTANTS
 * Grounded in official project benchmark dataset:
 * - Dataset: SemanticKITTI HDL-64E / Sequence 01 (foveamap_sequence01_replay.json)
 * - Volume: 156.8m x 148.6m x 9.4m (218,150 m³)
 * - Input Points: 122,829 points / scan
 * - Grid Policy: Near (0-12m, 0.75m), Mid (12-28m, 1.5m), Far (28-80m, 3.0m)
 */
const BENCHMARK = {
  memory: {
    uniformMb: 7.11,
    lidarxMb: 0.23,
    reductionPct: 96.8,
  },
  representation: {
    uniformVoxels: 1777868,
    lidarxCells: 3601,
    reductionPct: 99.8,
  },
  latency: {
    uniformMs: 112,
    lidarxMs: 38,
    reductionPct: 66.1,
  },
  fps: {
    uniformFps: 8.9,
    lidarxFps: 26.3,
    gainPct: 195.5,
  },
  breakdown: [
    { name: 'Inference', ms: 14.5, pct: 38.2, color: '#00E5FF' },      // Website Cyan
    { name: 'Foveated Grid', ms: 9.5, pct: 25.0, color: '#0070F3' },  // Brand Blue
    { name: 'Terrain', ms: 8.2, pct: 21.6, color: '#A855F7' },        // Brand Purple
    { name: 'Objects', ms: 5.8, pct: 15.3, color: '#10B981' },        // Brand Emerald
  ],
  totalMs: 38.0,
};

export const PerformanceLabView: React.FC = () => {
  return (
    <div className="max-w-6xl mx-auto space-y-5 pb-12 font-sans text-gray-200 select-none">
      {/* HEADER */}
      <header className="border-b border-white/10 pb-3 flex flex-col sm:flex-row sm:items-end justify-between gap-2">
        <div>
          <h1 className="text-xl font-bold tracking-wider text-white uppercase font-mono">
            PERFORMANCE LAB
          </h1>
          <p className="text-xs text-cyan-400 font-mono tracking-wide mt-0.5">
            Uniform 3D vs LiDAR-X Foveated 2.5D
          </p>
        </div>

        <div className="text-[11px] font-mono text-gray-500">
          BENCHMARK: <span className="text-gray-300">SemanticKITTI HDL-64E</span> • 122.8k PTS • 80m RANGE
        </div>
      </header>

      {/* 2-COLUMN CHART GRID */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        {/* 1. MEMORY COMPARISON (Vertical Bar Chart) */}
        <div className="bg-[#10141d] border border-white/10 rounded-sm p-4 flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <div className="font-mono">
              <span className="text-xs font-bold text-white uppercase tracking-wider block">
                1. Memory Comparison
              </span>
              <span className="text-[10px] text-gray-500">Allocated RAM per scan volume</span>
            </div>
            <div className="flex items-center gap-3 text-[10px] font-mono">
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 bg-slate-500 inline-block rounded-none" />
                <span className="text-gray-400">Uniform 3D</span>
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 bg-cyan-400 inline-block rounded-none shadow-[0_0_6px_rgba(0,229,255,0.4)]" />
                <span className="text-cyan-400 font-semibold">LiDAR-X</span>
              </span>
            </div>
          </div>

          <div className="w-full h-56">
            <svg viewBox="0 0 380 210" className="w-full h-full font-mono text-[11px] overflow-visible">
              {/* Y Axis Grid & Labels (0 to 8 MB) */}
              {[0, 2, 4, 6, 8].map((val) => {
                const y = 170 - (val / 8) * 140;
                return (
                  <g key={val}>
                    <line
                      x1="55"
                      y1={y}
                      x2="360"
                      y2={y}
                      stroke="#21262d"
                      strokeWidth="1"
                      strokeDasharray={val === 0 ? undefined : '2,2'}
                      shapeRendering="crispEdges"
                    />
                    <text x="45" y={y + 4} fill="#64748b" textAnchor="end" fontSize="10">
                      {val} MB
                    </text>
                  </g>
                );
              })}

              {/* Y Axis Title */}
              <text
                x="-100"
                y="14"
                transform="rotate(-90)"
                fill="#475569"
                textAnchor="middle"
                fontSize="9"
                letterSpacing="1"
              >
                MEMORY (MB)
              </text>

              {/* Axes Lines */}
              <line x1="55" y1="30" x2="55" y2="170" stroke="#334155" strokeWidth="1" shapeRendering="crispEdges" />
              <line x1="55" y1="170" x2="360" y2="170" stroke="#334155" strokeWidth="1" shapeRendering="crispEdges" />

              {/* Bar 1: Uniform 3D (7.11 MB) -> height = (7.11/8)*140 = 124.4px */}
              <rect
                x="110"
                y={170 - (7.11 / 8) * 140}
                width="60"
                height={(7.11 / 8) * 140}
                fill="#475569"
                className="hover:fill-slate-500 transition-colors"
              />
              <text
                x="140"
                y={170 - (7.11 / 8) * 140 - 6}
                fill="#cbd5e1"
                textAnchor="middle"
                fontWeight="bold"
                fontSize="11"
              >
                7.11 MB
              </text>

              {/* Bar 2: LiDAR-X (0.23 MB) -> height = (0.23/8)*140 = 4.0px */}
              <rect
                x="245"
                y={170 - (0.23 / 8) * 140}
                width="60"
                height={Math.max(4, (0.23 / 8) * 140)}
                fill="#00E5FF"
                className="hover:fill-cyan-300 transition-colors"
              />
              <text
                x="275"
                y={170 - (0.23 / 8) * 140 - 6}
                fill="#00E5FF"
                textAnchor="middle"
                fontWeight="bold"
                fontSize="11"
              >
                0.23 MB
              </text>
              <text
                x="275"
                y={170 - (0.23 / 8) * 140 - 20}
                fill="#22c55e"
                textAnchor="middle"
                fontSize="9"
                fontWeight="bold"
              >
                -96.8%
              </text>

              {/* X Axis Labels */}
              <text x="140" y="190" fill="#94a3b8" textAnchor="middle" fontSize="11">
                Uniform 3D
              </text>
              <text x="275" y="190" fill="#00E5FF" textAnchor="middle" fontSize="11" fontWeight="bold">
                LiDAR-X 2.5D
              </text>
            </svg>
          </div>
        </div>

        {/* 2. REPRESENTATION SIZE (Vertical Bar Chart with Logarithmic Scale) */}
        <div className="bg-[#10141d] border border-white/10 rounded-sm p-4 flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <div className="font-mono">
              <span className="text-xs font-bold text-white uppercase tracking-wider block">
                2. Representation Size
              </span>
              <span className="text-[10px] text-gray-500">Spatial element count (Logarithmic Scale)</span>
            </div>
            <div className="flex items-center gap-3 text-[10px] font-mono">
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 bg-slate-500 inline-block rounded-none" />
                <span className="text-gray-400">Voxels</span>
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 bg-cyan-400 inline-block rounded-none shadow-[0_0_6px_rgba(0,229,255,0.4)]" />
                <span className="text-cyan-400 font-semibold">Cells</span>
              </span>
            </div>
          </div>

          <div className="w-full h-56">
            <svg viewBox="0 0 380 210" className="w-full h-full font-mono text-[11px] overflow-visible">
              {/* Y Axis Log Grid (10^3 to 10^7) */}
              {[
                { log: 3, label: '1k (10³)' },
                { log: 4, label: '10k (10⁴)' },
                { log: 5, label: '100k (10⁵)' },
                { log: 6, label: '1M (10⁶)' },
                { log: 7, label: '10M (10⁷)' },
              ].map((item) => {
                const y = 170 - ((item.log - 3) / 4) * 140;
                return (
                  <g key={item.log}>
                    <line
                      x1="65"
                      y1={y}
                      x2="360"
                      y2={y}
                      stroke="#21262d"
                      strokeWidth="1"
                      strokeDasharray="2,2"
                      shapeRendering="crispEdges"
                    />
                    <text x="55" y={y + 4} fill="#64748b" textAnchor="end" fontSize="9">
                      {item.label}
                    </text>
                  </g>
                );
              })}

              {/* Y Axis Title */}
              <text
                x="-100"
                y="14"
                transform="rotate(-90)"
                fill="#475569"
                textAnchor="middle"
                fontSize="9"
                letterSpacing="1"
              >
                ELEMENTS (LOG SCALE)
              </text>

              {/* Axes Lines */}
              <line x1="65" y1="30" x2="65" y2="170" stroke="#334155" strokeWidth="1" shapeRendering="crispEdges" />
              <line x1="65" y1="170" x2="360" y2="170" stroke="#334155" strokeWidth="1" shapeRendering="crispEdges" />

              {/* Bar 1: Uniform 3D (1,777,868 voxels, log10=6.250) */}
              {(() => {
                const logVal = Math.log10(1777868);
                const h = ((logVal - 3) / 4) * 140;
                return (
                  <>
                    <rect
                      x="115"
                      y={170 - h}
                      width="60"
                      height={h}
                      fill="#475569"
                      className="hover:fill-slate-500 transition-colors"
                    />
                    <text x="145" y={170 - h - 6} fill="#cbd5e1" textAnchor="middle" fontWeight="bold" fontSize="11">
                      1,777,868
                    </text>
                  </>
                );
              })()}

              {/* Bar 2: LiDAR-X (3,601 cells, log10=3.556) */}
              {(() => {
                const logVal = Math.log10(3601);
                const h = ((logVal - 3) / 4) * 140;
                return (
                  <>
                    <rect
                      x="250"
                      y={170 - h}
                      width="60"
                      height={h}
                      fill="#00E5FF"
                      className="hover:fill-cyan-300 transition-colors"
                    />
                    <text x="280" y={170 - h - 6} fill="#00E5FF" textAnchor="middle" fontWeight="bold" fontSize="11">
                      3,601
                    </text>
                    <text x="280" y={170 - h - 20} fill="#22c55e" textAnchor="middle" fontSize="9" fontWeight="bold">
                      -99.8% (493x)
                    </text>
                  </>
                );
              })()}

              {/* X Axis Labels */}
              <text x="145" y="190" fill="#94a3b8" textAnchor="middle" fontSize="11">
                Uniform 3D
              </text>
              <text x="280" y="190" fill="#00E5FF" textAnchor="middle" fontSize="11" fontWeight="bold">
                LiDAR-X
              </text>
            </svg>
          </div>
        </div>

        {/* 3. LATENCY COMPARISON (Vertical Bar Chart) */}
        <div className="bg-[#10141d] border border-white/10 rounded-sm p-4 flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <div className="font-mono">
              <span className="text-xs font-bold text-white uppercase tracking-wider block">
                3. Pipeline Latency
              </span>
              <span className="text-[10px] text-gray-500">End-to-end frame processing time</span>
            </div>
            <div className="flex items-center gap-3 text-[10px] font-mono">
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 bg-slate-500 inline-block rounded-none" />
                <span className="text-gray-400">Uniform 3D</span>
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 bg-cyan-400 inline-block rounded-none shadow-[0_0_6px_rgba(0,229,255,0.4)]" />
                <span className="text-cyan-400 font-semibold">LiDAR-X</span>
              </span>
            </div>
          </div>

          <div className="w-full h-56">
            <svg viewBox="0 0 380 210" className="w-full h-full font-mono text-[11px] overflow-visible">
              {/* Y Axis Grid & Labels (0 to 120 ms) */}
              {[0, 30, 60, 90, 120].map((val) => {
                const y = 170 - (val / 120) * 140;
                return (
                  <g key={val}>
                    <line
                      x1="55"
                      y1={y}
                      x2="360"
                      y2={y}
                      stroke="#21262d"
                      strokeWidth="1"
                      strokeDasharray={val === 0 ? undefined : '2,2'}
                      shapeRendering="crispEdges"
                    />
                    <text x="45" y={y + 4} fill="#64748b" textAnchor="end" fontSize="10">
                      {val} ms
                    </text>
                  </g>
                );
              })}

              {/* Y Axis Title */}
              <text
                x="-100"
                y="14"
                transform="rotate(-90)"
                fill="#475569"
                textAnchor="middle"
                fontSize="9"
                letterSpacing="1"
              >
                LATENCY (MS)
              </text>

              {/* Axes Lines */}
              <line x1="55" y1="30" x2="55" y2="170" stroke="#334155" strokeWidth="1" shapeRendering="crispEdges" />
              <line x1="55" y1="170" x2="360" y2="170" stroke="#334155" strokeWidth="1" shapeRendering="crispEdges" />

              {/* Bar 1: Uniform 3D (112 ms) -> height = (112/120)*140 = 130.6px */}
              <rect
                x="110"
                y={170 - (112 / 120) * 140}
                width="60"
                height={(112 / 120) * 140}
                fill="#475569"
                className="hover:fill-slate-500 transition-colors"
              />
              <text
                x="140"
                y={170 - (112 / 120) * 140 - 6}
                fill="#cbd5e1"
                textAnchor="middle"
                fontWeight="bold"
                fontSize="11"
              >
                112 ms
              </text>

              {/* Bar 2: LiDAR-X (38 ms) -> height = (38/120)*140 = 44.3px */}
              <rect
                x="245"
                y={170 - (38 / 120) * 140}
                width="60"
                height={(38 / 120) * 140}
                fill="#00E5FF"
                className="hover:fill-cyan-300 transition-colors"
              />
              <text
                x="275"
                y={170 - (38 / 120) * 140 - 6}
                fill="#00E5FF"
                textAnchor="middle"
                fontWeight="bold"
                fontSize="11"
              >
                38 ms
              </text>
              <text
                x="275"
                y={170 - (38 / 120) * 140 - 20}
                fill="#22c55e"
                textAnchor="middle"
                fontSize="9"
                fontWeight="bold"
              >
                -66.1% (2.95x)
              </text>

              {/* X Axis Labels */}
              <text x="140" y="190" fill="#94a3b8" textAnchor="middle" fontSize="11">
                Uniform 3D
              </text>
              <text x="275" y="190" fill="#00E5FF" textAnchor="middle" fontSize="11" fontWeight="bold">
                LiDAR-X
              </text>
            </svg>
          </div>
        </div>

        {/* 4. FPS / THROUGHPUT (Vertical Bar Chart) */}
        <div className="bg-[#10141d] border border-white/10 rounded-sm p-4 flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <div className="font-mono">
              <span className="text-xs font-bold text-white uppercase tracking-wider block">
                4. Throughput (FPS)
              </span>
              <span className="text-[10px] text-gray-500">Autonomous processing update frequency</span>
            </div>
            <div className="flex items-center gap-3 text-[10px] font-mono">
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 bg-slate-500 inline-block rounded-none" />
                <span className="text-gray-400">Uniform 3D</span>
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 bg-cyan-400 inline-block rounded-none shadow-[0_0_6px_rgba(0,229,255,0.4)]" />
                <span className="text-cyan-400 font-semibold">LiDAR-X</span>
              </span>
            </div>
          </div>

          <div className="w-full h-56">
            <svg viewBox="0 0 380 210" className="w-full h-full font-mono text-[11px] overflow-visible">
              {/* Y Axis Grid & Labels (0 to 30 FPS) */}
              {[0, 10, 20, 30].map((val) => {
                const y = 170 - (val / 30) * 140;
                return (
                  <g key={val}>
                    <line
                      x1="55"
                      y1={y}
                      x2="360"
                      y2={y}
                      stroke="#21262d"
                      strokeWidth="1"
                      strokeDasharray={val === 0 ? undefined : '2,2'}
                      shapeRendering="crispEdges"
                    />
                    <text x="45" y={y + 4} fill="#64748b" textAnchor="end" fontSize="10">
                      {val}
                    </text>
                  </g>
                );
              })}

              {/* Y Axis Title */}
              <text
                x="-100"
                y="14"
                transform="rotate(-90)"
                fill="#475569"
                textAnchor="middle"
                fontSize="9"
                letterSpacing="1"
              >
                THROUGHPUT (FPS)
              </text>

              {/* Axes Lines */}
              <line x1="55" y1="30" x2="55" y2="170" stroke="#334155" strokeWidth="1" shapeRendering="crispEdges" />
              <line x1="55" y1="170" x2="360" y2="170" stroke="#334155" strokeWidth="1" shapeRendering="crispEdges" />

              {/* Bar 1: Uniform 3D (8.9 FPS) -> height = (8.9/30)*140 = 41.5px */}
              <rect
                x="110"
                y={170 - (8.9 / 30) * 140}
                width="60"
                height={(8.9 / 30) * 140}
                fill="#475569"
                className="hover:fill-slate-500 transition-colors"
              />
              <text
                x="140"
                y={170 - (8.9 / 30) * 140 - 6}
                fill="#cbd5e1"
                textAnchor="middle"
                fontWeight="bold"
                fontSize="11"
              >
                8.9 FPS
              </text>

              {/* Bar 2: LiDAR-X (26.3 FPS) -> height = (26.3/30)*140 = 122.7px */}
              <rect
                x="245"
                y={170 - (26.3 / 30) * 140}
                width="60"
                height={(26.3 / 30) * 140}
                fill="#00E5FF"
                className="hover:fill-cyan-300 transition-colors"
              />
              <text
                x="275"
                y={170 - (26.3 / 30) * 140 - 6}
                fill="#00E5FF"
                textAnchor="middle"
                fontWeight="bold"
                fontSize="11"
              >
                26.3 FPS
              </text>
              <text
                x="275"
                y={170 - (26.3 / 30) * 140 - 20}
                fill="#22c55e"
                textAnchor="middle"
                fontSize="9"
                fontWeight="bold"
              >
                +195%
              </text>

              {/* X Axis Labels */}
              <text x="140" y="190" fill="#94a3b8" textAnchor="middle" fontSize="11">
                Uniform 3D
              </text>
              <text x="275" y="190" fill="#00E5FF" textAnchor="middle" fontSize="11" fontWeight="bold">
                LiDAR-X
              </text>
            </svg>
          </div>
        </div>

        {/* 5. PROCESSING BREAKDOWN (Donut Chart) */}
        <div className="bg-[#10141d] border border-white/10 rounded-sm p-4 flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <div className="font-mono">
              <span className="text-xs font-bold text-white uppercase tracking-wider block">
                5. Processing Breakdown
              </span>
              <span className="text-[10px] text-gray-500">LiDAR-X pipeline execution time by module</span>
            </div>
            <span className="text-[11px] font-mono text-cyan-300 font-bold px-2 py-0.5 bg-cyan-500/10 border border-cyan-500/30">
              Total: 38.0 ms
            </span>
          </div>

          <div className="w-full h-56 flex items-center justify-center">
            <svg viewBox="0 0 380 200" className="w-full h-full font-mono text-[10px] overflow-visible">
              {/* Donut Geometry: center (115, 100), R=64, r=40 */}
              {(() => {
                const cx = 115;
                const cy = 100;
                const R = 64;
                const r = 40;
                let cumulativeAngle = -Math.PI / 2;

                return (
                  <g>
                    {BENCHMARK.breakdown.map((item) => {
                      const angle = (item.pct / 100) * 2 * Math.PI;
                      const startAngle = cumulativeAngle;
                      const endAngle = cumulativeAngle + angle;
                      cumulativeAngle = endAngle;

                      // Outer Arc points
                      const x1 = cx + R * Math.cos(startAngle);
                      const y1 = cy + R * Math.sin(startAngle);
                      const x2 = cx + R * Math.cos(endAngle);
                      const y2 = cy + R * Math.sin(endAngle);

                      // Inner Arc points
                      const x3 = cx + r * Math.cos(endAngle);
                      const y3 = cy + r * Math.sin(endAngle);
                      const x4 = cx + r * Math.cos(startAngle);
                      const y4 = cy + r * Math.sin(startAngle);

                      const largeArcFlag = angle > Math.PI ? 1 : 0;

                      const pathData = [
                        `M ${x1} ${y1}`,
                        `A ${R} ${R} 0 ${largeArcFlag} 1 ${x2} ${y2}`,
                        `L ${x3} ${y3}`,
                        `A ${r} ${r} 0 ${largeArcFlag} 0 ${x4} ${y4}`,
                        'Z',
                      ].join(' ');

                      return (
                        <path
                          key={item.name}
                          d={pathData}
                          fill={item.color}
                          stroke="#10141d"
                          strokeWidth="2"
                          className="hover:opacity-85 transition-opacity cursor-default"
                        />
                      );
                    })}

                    {/* Center Text */}
                    <text x={cx} y={cy - 2} fill="#ffffff" textAnchor="middle" fontWeight="bold" fontSize="15">
                      38.0
                    </text>
                    <text x={cx} y={cy + 13} fill="#00E5FF" textAnchor="middle" fontSize="9" letterSpacing="1" fontWeight="bold">
                      MS TOTAL
                    </text>
                  </g>
                );
              })()}

              {/* Legend & Percentages (Right side) */}
              <g transform="translate(210, 30)">
                {BENCHMARK.breakdown.map((item, idx) => {
                  const y = idx * 34;
                  return (
                    <g key={item.name} transform={`translate(0, ${y})`}>
                      <rect x="0" y="2" width="10" height="10" fill={item.color} rx="1" />
                      <text x="18" y="11" fill="#e2e8f0" fontSize="11" fontWeight="500">
                        {item.name}
                      </text>
                      <text x="18" y="23" fill="#64748b" fontSize="10">
                        {item.ms} ms
                      </text>
                      <text x="150" y="11" fill={item.color} fontSize="11" fontWeight="bold" textAnchor="end">
                        {item.pct}%
                      </text>
                    </g>
                  );
                })}
              </g>
            </svg>
          </div>
        </div>

        {/* 6. FOVEATED RESOLUTION (Line / Area Graph) */}
        <div className="bg-[#10141d] border border-white/10 rounded-sm p-4 flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <div className="font-mono">
              <span className="text-xs font-bold text-white uppercase tracking-wider block">
                6. Foveated Resolution Curve
              </span>
              <span className="text-[10px] text-gray-500">Distance vs. Cell Coarseness</span>
            </div>
            <div className="flex items-center gap-2 text-[10px] font-mono text-gray-400">
              <span className="text-cyan-400 font-semibold">Fine (Near)</span>
              <span>→</span>
              <span className="text-gray-300">Coarse (Far)</span>
            </div>
          </div>

          <div className="w-full h-56">
            <svg viewBox="0 0 380 210" className="w-full h-full font-mono text-[11px] overflow-visible">
              {/* Distance Zones Backgrounds: Near (0-12m), Mid (12-28m), Far (28-80m) */}
              {/* x-scale: 55 to 355 (300px total). 0m=55, 12m=100, 28m=160, 80m=355 */}
              <rect x="55" y="30" width="45" height="140" fill="rgba(0, 229, 255, 0.08)" />
              <rect x="100" y="30" width="60" height="140" fill="rgba(0, 112, 243, 0.06)" />
              <rect x="160" y="30" width="195" height="140" fill="rgba(168, 85, 247, 0.04)" />

              {/* Y Axis Grid & Labels (Cell Size 0 to 3.5m) */}
              {[0, 1.0, 2.0, 3.0].map((val) => {
                const y = 170 - (val / 3.5) * 140;
                return (
                  <g key={val}>
                    <line
                      x1="55"
                      y1={y}
                      x2="355"
                      y2={y}
                      stroke="#21262d"
                      strokeWidth="1"
                      strokeDasharray={val === 0 ? undefined : '2,2'}
                      shapeRendering="crispEdges"
                    />
                    <text x="45" y={y + 4} fill="#64748b" textAnchor="end" fontSize="10">
                      {val.toFixed(1)}m
                    </text>
                  </g>
                );
              })}

              {/* Y Axis Title */}
              <text
                x="-100"
                y="14"
                transform="rotate(-90)"
                fill="#475569"
                textAnchor="middle"
                fontSize="9"
                letterSpacing="1"
              >
                CELL SIZE (M)
              </text>

              {/* Zone Boundary Dashed Lines */}
              <line x1="100" y1="30" x2="100" y2="170" stroke="#00E5FF" strokeWidth="1" strokeDasharray="3,3" opacity="0.4" />
              <line x1="160" y1="30" x2="160" y2="170" stroke="#0070F3" strokeWidth="1" strokeDasharray="3,3" opacity="0.3" />

              {/* Zone Labels Top */}
              <text x="77" y="44" fill="#00E5FF" textAnchor="middle" fontSize="9" fontWeight="bold">NEAR</text>
              <text x="77" y="55" fill="#00E5FF" textAnchor="middle" fontSize="8" opacity="0.8">0.75m</text>

              <text x="130" y="44" fill="#0070F3" textAnchor="middle" fontSize="9" fontWeight="bold">MID</text>
              <text x="130" y="55" fill="#0070F3" textAnchor="middle" fontSize="8" opacity="0.8">1.5m</text>

              <text x="257" y="44" fill="#A855F7" textAnchor="middle" fontSize="9" fontWeight="bold">FAR</text>
              <text x="257" y="55" fill="#A855F7" textAnchor="middle" fontSize="8" opacity="0.8">3.0m</text>

              {/* Shaded Area Under Curve */}
              {(() => {
                const yFine = 170 - (0.75 / 3.5) * 140;
                const yMed = 170 - (1.5 / 3.5) * 140;
                const yCoarse = 170 - (3.0 / 3.5) * 140;

                const areaPath = [
                  `M 55 170`,
                  `L 55 ${yFine}`,
                  `L 100 ${yFine}`,
                  `L 100 ${yMed}`,
                  `L 160 ${yMed}`,
                  `L 160 ${yCoarse}`,
                  `L 355 ${yCoarse}`,
                  `L 355 170`,
                  'Z',
                ].join(' ');

                const linePath = [
                  `M 55 ${yFine}`,
                  `L 100 ${yFine}`,
                  `L 100 ${yMed}`,
                  `L 160 ${yMed}`,
                  `L 160 ${yCoarse}`,
                  `L 355 ${yCoarse}`,
                ].join(' ');

                return (
                  <>
                    <path d={areaPath} fill="rgba(0, 229, 255, 0.12)" />
                    <path d={linePath} fill="none" stroke="#00E5FF" strokeWidth="2.5" />

                    {/* Step Anchor Points */}
                    <circle cx="55" cy={yFine} r="3" fill="#00E5FF" />
                    <circle cx="100" cy={yFine} r="3" fill="#00E5FF" />
                    <circle cx="100" cy={yMed} r="3" fill="#00E5FF" />
                    <circle cx="160" cy={yMed} r="3" fill="#00E5FF" />
                    <circle cx="160" cy={yCoarse} r="3" fill="#00E5FF" />
                    <circle cx="355" cy={yCoarse} r="3" fill="#00E5FF" />
                  </>
                );
              })()}

              {/* Axes Lines */}
              <line x1="55" y1="30" x2="55" y2="170" stroke="#334155" strokeWidth="1" shapeRendering="crispEdges" />
              <line x1="55" y1="170" x2="355" y2="170" stroke="#334155" strokeWidth="1" shapeRendering="crispEdges" />

              {/* X Axis Ticks & Labels */}
              <text x="55" y="188" fill="#00E5FF" textAnchor="middle" fontSize="10" fontWeight="bold">0m</text>
              <text x="100" y="188" fill="#00E5FF" textAnchor="middle" fontSize="10">12m</text>
              <text x="160" y="188" fill="#0070F3" textAnchor="middle" fontSize="10">28m</text>
              <text x="355" y="188" fill="#64748b" textAnchor="middle" fontSize="10">80m</text>

              {/* X Axis Title */}
              <text x="205" y="204" fill="#64748b" textAnchor="middle" fontSize="9" letterSpacing="1">
                RADIAL DISTANCE FROM VEHICLE (METERS)
              </text>
            </svg>
          </div>
        </div>

      </div>

      {/* BOTTOM COMPARISON STATEMENT */}
      <footer className="pt-2 border-t border-white/10 text-center">
        <p className="text-xs font-mono text-gray-400 tracking-wide">
          "LiDAR-X reduces spatial representation while maintaining higher processing efficiency."
        </p>
      </footer>
    </div>
  );
};
