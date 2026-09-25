import React, { useEffect, useRef } from 'react';

export const HeroTransitionVisual: React.FC<{ className?: string }> = ({ className = '' }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let time = 0;

    const resize = () => {
      canvas.width = canvas.parentElement?.clientWidth || 550;
      canvas.height = canvas.parentElement?.clientHeight || 400;
    };
    resize();
    window.addEventListener('resize', resize);

    const render = () => {
      time += 0.015;
      const w = canvas.width;
      const h = canvas.height;

      // Clear
      ctx.fillStyle = '#070A0F';
      ctx.fillRect(0, 0, w, h);

      // Transition wave position (oscillates from left 20% to right 80%)
      const waveX = w * (0.3 + 0.4 * (Math.sin(time * 0.8) * 0.5 + 0.5));

      // 1. Draw Left Side: Raw 3D Point Cloud Representation
      // 2. Draw Right Side: Semantic 2.5D Grid Map Representation
      
      const numRays = 24;
      const numCols = 32;

      for (let i = 0; i < numCols; i++) {
        for (let j = 0; j < numRays; j++) {
          const normX = i / numCols;
          const normY = j / numRays;

          // Pseudo 3D perspective projection
          const px = w * 0.1 + normX * (w * 0.8);
          const py = h * 0.15 + normY * (h * 0.7) + (Math.sin(normX * 4 + time) * 8);

          const isSemanticSide = px > waveX;

          // Determine class / feature based on coordinates
          const isRoad = normY >= 0.35 && normY <= 0.65;
          const isObstacle = (normX > 0.65 && normX < 0.75 && normY > 0.45 && normY < 0.58) ||
                             (normX > 0.25 && normX < 0.35 && normY > 0.25 && normY < 0.35);
          const isFoliage = (normY < 0.25 || normY > 0.75) && Math.sin(normX * 10) > 0.2;

          ctx.beginPath();
          if (!isSemanticSide) {
            // Raw LiDAR (Grayscale points, ring structure)
            const intensity = (Math.sin(normX * 12 + normY * 8) * 0.5 + 0.5);
            const gray = Math.floor(100 + intensity * 120);
            ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
            ctx.arc(px, py, 1.8, 0, Math.PI * 2);
            ctx.fill();
          } else {
            // Semantic 2.5D Foveated Cells
            let color = '#64748B'; // Non-drivable
            let size = 5;

            // Distance to center (foveation)
            const distToCenter = Math.hypot(normX - 0.5, normY - 0.5);
            if (distToCenter < 0.2) size = 3;
            else if (distToCenter < 0.35) size = 5;
            else size = 8;

            if (isObstacle) color = '#EF4444'; // Red dynamic obstacle
            else if (isRoad) color = '#00E5FF'; // Cyan drivable surface
            else if (isFoliage) color = '#10B981'; // Emerald vegetation
            else color = '#8B5CF6'; // Infrastructure

            ctx.fillStyle = color;
            ctx.shadowColor = color;
            ctx.shadowBlur = 4;
            ctx.fillRect(px - size / 2, py - size / 2, size, size);
            ctx.shadowBlur = 0;
          }
        }
      }

      // 3. Transformation Scanline Wave
      const waveGrad = ctx.createLinearGradient(waveX - 40, 0, waveX + 40, 0);
      waveGrad.addColorStop(0, 'rgba(0, 229, 255, 0)');
      waveGrad.addColorStop(0.5, 'rgba(0, 229, 255, 0.75)');
      waveGrad.addColorStop(1, 'rgba(138, 43, 226, 0)');

      ctx.fillStyle = waveGrad;
      ctx.fillRect(waveX - 30, h * 0.1, 60, h * 0.8);

      // Laser line
      ctx.strokeStyle = '#00E5FF';
      ctx.lineWidth = 2;
      ctx.shadowColor = 'rgba(0, 229, 255, 0.9)';
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.moveTo(waveX, h * 0.1);
      ctx.lineTo(waveX, h * 0.9);
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Labels on top
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.fillText('RAW 3D LIDAR STREAM', w * 0.1, h * 0.08);

      ctx.fillStyle = '#00E5FF';
      ctx.fillText('SEMANTIC 2.5D GRID', w * 0.65, h * 0.08);

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <div className={`relative rounded-2xl overflow-hidden border border-white/10 tech-panel ${className}`}>
      <canvas ref={canvasRef} className="w-full h-full block" />
      <div className="absolute bottom-3 left-4 right-4 flex items-center justify-between pointer-events-none text-[11px] font-mono text-gray-400 bg-dark-900/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/5">
        <span className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span>Continuous Point-to-Grid Transformation</span>
        </span>
        <span className="text-purple-400">Deep Learning Pipeline</span>
      </div>
    </div>
  );
};
