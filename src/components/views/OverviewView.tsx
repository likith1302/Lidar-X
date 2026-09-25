import React from 'react';
import {
  Cpu,
  Grid,
  ArrowRight,
  ShieldCheck,
} from 'lucide-react';
import { useAppState } from '../../context/AppStateContext';
import { HeroTransitionVisual } from '../visualizers/HeroTransitionVisual';
import { Badge } from '../common/Badge';

export const OverviewView: React.FC = () => {
  const { setCurrentTab } = useAppState();

  return (
    <div className="space-y-10 max-w-7xl mx-auto pb-12">
      {/* 1. HERO SECTION */}
      <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-b from-dark-900 via-dark-900/90 to-dark-950 p-6 md:p-10 shadow-2xl">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          {/* Left Column: Headline & Description */}
          <div className="lg:col-span-6 space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-mono font-medium">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
              <span>Autonomous Navigation Research Foundation</span>
            </div>

            <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-white tracking-tight leading-[1.15]">
              Adaptive LiDAR Mapping for Safer Autonomous Navigation
            </h1>

            <p className="text-gray-300 text-base sm:text-lg leading-relaxed font-normal">
              LiDAR-X transforms raw 3D LiDAR data into semantic, elevation-aware maps designed for intelligent autonomous navigation.
            </p>

            {/* Action Buttons */}
            <div className="flex flex-wrap items-center gap-4 pt-2">
              <button
                onClick={() => setCurrentTab('mapping-console')}
                className="flex items-center gap-2 px-6 py-3.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-black font-semibold text-sm transition-all shadow-glow-cyan transform hover:-translate-y-0.5"
              >
                <span>Open Mapping Console</span>
                <ArrowRight className="w-4 h-4 stroke-[2.5]" />
              </button>
            </div>
          </div>

          {/* Right Column: Premium Visual Illustration */}
          <div className="lg:col-span-6 w-full h-[360px] md:h-[420px]">
            <HeroTransitionVisual className="w-full h-full" />
          </div>
        </div>
      </section>

      {/* 2. THREE ELEGANT FEATURE CARDS */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">
              Core Architectural Pillars
            </h2>
            <p className="text-xs text-gray-400 mt-0.5 font-mono">
              Three-tiered spatial representation for robust vehicle motion planning
            </p>
          </div>
          <Badge variant="cyan" className="text-[10px]">ACTIVE PERCEPTION STACK</Badge>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Card 1: Semantic Perception */}
          <div className="tech-panel rounded-2xl p-6 border border-white/10 space-y-4 relative overflow-hidden group hover:border-purple-500/40 hover:shadow-glow-purple transition-all duration-300">
            <div className="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/30 flex items-center justify-center text-purple-400 group-hover:scale-105 transition-transform">
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <h3 className="text-lg font-bold text-white">Semantic Perception</h3>
                <Badge variant="purple" className="text-[10px] py-0 px-2">AI MODEL</Badge>
              </div>
              <p className="text-xs text-purple-300 font-mono">Point-wise Neural Classification</p>
            </div>
            <p className="text-sm text-gray-300 leading-relaxed">
              The integrated Fast-FRNet PyTorch deep learning models (RELLIS-3D off-road primary and SemanticKITTI urban) perform frustum-point fusion segmentation across 20 scene categories, identifying ground, terrain, obstacles, vegetation, and vehicles.
            </p>
          </div>

          {/* Card 2: Adaptive 2.5D Mapping */}
          <div className="tech-panel rounded-2xl p-6 border border-white/10 space-y-4 relative overflow-hidden group hover:border-cyan-500/40 hover:shadow-glow-cyan transition-all duration-300">
            <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 group-hover:scale-105 transition-transform">
              <Grid className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <h3 className="text-lg font-bold text-white">Adaptive 2.5D Mapping</h3>
                <Badge variant="cyan" className="text-[10px] py-0 px-2">FOVEATED GRID</Badge>
              </div>
              <p className="text-xs text-cyan-300 font-mono">Variable Resolution Allocation</p>
            </div>
            <p className="text-sm text-gray-300 leading-relaxed">
              The foveated grid engine projects classified 3D points into a hierarchical height-aware grid (5 cm near-field up to 50 cm at 100 m) with strict 2:1 integer nested alignment and feature-driven refinement.
            </p>
          </div>

          {/* Card 3: Navigation Intelligence */}
          <div className="tech-panel rounded-2xl p-6 border border-white/10 space-y-4 relative overflow-hidden group hover:border-amber-500/40 hover:shadow-glow-orange transition-all duration-300">
            <div className="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400 group-hover:scale-105 transition-transform">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <h3 className="text-lg font-bold text-white">Navigation Intelligence</h3>
                <Badge variant="warning" className="text-[10px] py-0 px-2">SAFETY LAYER</Badge>
              </div>
              <p className="text-xs text-amber-300 font-mono">Traversability & Obstacle Buffer</p>
            </div>
            <p className="text-sm text-gray-300 leading-relaxed">
              Real geometric PCA slope estimation, surface roughness analysis, curb detection, and Kalman multi-object tracking continuously evaluate terrain traversability and collision hazards.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
