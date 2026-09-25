import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Plus, Minus } from 'lucide-react';
import { GridCell, MapLayersState } from '../../types/map';

interface TraversabilityMap2DProps {
  cells: GridCell[];
  onCellSelect?: (cell: GridCell) => void;
  selectedCellId?: string | null;
  className?: string;
  layers?: MapLayersState;
}

type TravKey = 'drivable' | 'caution' | 'hazard' | 'non_drivable' | 'uncertain';

const TRAV_COLORS: Record<TravKey, { color: string; label: string }> = {
  drivable: { color: '#00E5FF', label: 'Drivable' },
  caution: { color: '#F59E0B', label: 'Caution' },
  hazard: { color: '#EF4444', label: 'Hazard' },
  non_drivable: { color: '#475569', label: 'Non-Drivable' },
  uncertain: { color: '#7C3AED', label: 'Uncertain' },
};

/** Normalise any backend / frontend traversability string to a single key. */
function normaliseTrav(cell: GridCell): TravKey {
  const t =
    cell.backendAdaptiveCell?.traversability_state ||
    cell.backendTerrain?.drivability_state ||
    cell.traversability ||
    '';

  const s = String(t).toLowerCase();
  if (s === 'drivable' || s === 'drivable_candidate' || s === 'drivable / safe') {
    return 'drivable';
  }
  if (s === 'caution_irregular' || s === 'caution / irregular') {
    return 'caution';
  }
  if (
    s === 'collision_hazard' ||
    s === 'obstacle_hazard' ||
    s === 'collision hazard' ||
    s === 'obstacle barrier'
  ) {
    return 'hazard';
  }
  if (s === 'non_traversable' || s === 'non_drivable_candidate' || s === 'non-traversable') {
    return 'non_drivable';
  }
  return 'uncertain';
}

export const TraversabilityMap2D: React.FC<TraversabilityMap2DProps> = ({
  cells,
  onCellSelect,
  selectedCellId,
  className = 'w-full h-full min-h-[220px]',
  layers,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [zoom, setZoom] = useState(2.3);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const userInteractedRef = useRef(false);
  const isInitializedRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0 });
  const panStartRef = useRef({ x: 0, y: 0 });

  // Calculate auto-fit zoom based on visible cells bounding envelope
  const calculateAutoFitZoom = useCallback((cellList: GridCell[], width: number, height: number): number => {
    if (!cellList || cellList.length === 0) return 2.3;
    let maxExtent = 35.0;
    for (const c of cellList) {
      const d = Math.max(Math.abs(c.worldX), Math.abs(c.worldY));
      if (d > maxExtent && d < 120.0) {
        maxExtent = d;
      }
    }
    const minDim = Math.min(width, height);
    const fit = (minDim * 0.44) / maxExtent;
    return Math.max(1.0, Math.min(10.0, fit));
  }, []);

  const cellsRef = useRef(cells);
  cellsRef.current = cells;

  // Auto-fit view once on initial cells load without stomping subsequent user or resize view states
  useEffect(() => {
    if (!isInitializedRef.current && cells.length > 0 && containerRef.current) {
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight || 240;
      if (w >= 150 && h > 0) {
        const fit = calculateAutoFitZoom(cells, w, h);
        setZoom(fit);
        isInitializedRef.current = true;
      }
    }
  }, [cells, calculateAutoFitZoom]);

  const autoFitView = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    const w = container.clientWidth;
    const h = container.clientHeight || 240;
    const newZoom = calculateAutoFitZoom(cells, w, h);
    setZoom(newZoom);
    setPanOffset({ x: 0, y: 0 });
    userInteractedRef.current = false;
  }, [cells, calculateAutoFitZoom]);

  // Standard AV Ego Frame:
  //   +X = Forward  -> Maps to top of screen (centerY - wx * zoom)
  //   +Y = Left     -> Maps to left of screen (centerX - wy * zoom)
  //   -Y = Right    -> Maps to right of screen (centerX + |wy| * zoom)
  const worldToCanvas = useCallback(
    (wx: number, wy: number, width: number, height: number) => {
      const centerX = width / 2 + panOffset.x;
      const centerY = height / 2 + panOffset.y;
      const px = centerX - wy * zoom;
      const py = centerY - wx * zoom;
      return { px, py };
    },
    [zoom, panOffset]
  );

  const canvasToWorld = useCallback(
    (px: number, py: number, width: number, height: number) => {
      const centerX = width / 2 + panOffset.x;
      const centerY = height / 2 + panOffset.y;
      const wy = -(px - centerX) / zoom;
      const wx = -(py - centerY) / zoom;
      return { wx, wy };
    },
    [zoom, panOffset]
  );

  const drawTraversabilityMap = useCallback((overrideWidth?: number, overrideHeight?: number) => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = overrideWidth ?? container.clientWidth ?? 300;
    const height = overrideHeight ?? container.clientHeight ?? 240;
    if (width <= 0 || height <= 0) return;

    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = '#080B11';
    ctx.fillRect(0, 0, width, height);

    const center = worldToCanvas(0, 0, width, height);

    // Range rings at 10m, 25m, 45m
    [10, 25, 45].forEach((r) => {
      const rPx = r * zoom;
      ctx.beginPath();
      ctx.arc(center.px, center.py, rPx, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Range label
      ctx.fillStyle = 'rgba(255, 255, 255, 0.22)';
      ctx.font = '8px monospace';
      ctx.fillText(`${r}m`, center.px + 4, center.py - rPx + 10);
    });

    // Draw axis compass at the top-left (so it never collides with the bottom legend bar)
    const compassX = 14;
    const compassY = 32;
    const compassSize = 12;
    ctx.save();
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    ctx.fillRect(compassX - 4, compassY - compassSize - 4, compassSize * 2 + 8, compassSize * 2 + 8);
    ctx.font = '9px monospace';

    // Forward arrow (Up)
    ctx.strokeStyle = '#00E5FF';
    ctx.beginPath();
    ctx.moveTo(compassX + compassSize, compassY - compassSize + 2);
    ctx.lineTo(compassX + compassSize, compassY - 2);
    ctx.stroke();
    ctx.fillStyle = '#00E5FF';
    ctx.beginPath();
    ctx.moveTo(compassX + compassSize - 3, compassY - compassSize + 6);
    ctx.lineTo(compassX + compassSize, compassY - compassSize + 2);
    ctx.lineTo(compassX + compassSize + 3, compassY - compassSize + 6);
    ctx.closePath();
    ctx.fill();
    ctx.fillText('F', compassX + compassSize + 5, compassY - compassSize + 6);

    // Left arrow (Left)
    ctx.strokeStyle = '#A855F7';
    ctx.beginPath();
    ctx.moveTo(compassX + compassSize, compassY);
    ctx.lineTo(compassX + 2, compassY);
    ctx.stroke();
    ctx.fillStyle = '#A855F7';
    ctx.beginPath();
    ctx.moveTo(compassX + 6, compassY - 3);
    ctx.lineTo(compassX + 2, compassY);
    ctx.lineTo(compassX + 6, compassY + 3);
    ctx.closePath();
    ctx.fill();
    ctx.fillText('L', compassX + 2, compassY + 11);
    ctx.restore();

    // Layer-driven visibility
    const showDrivable = layers?.semanticClasses !== false && layers?.traversability !== false;
    const showCaution = layers?.traversability !== false;
    const showHazard = layers?.staticObstacles !== false;
    const showNonDrivable = layers?.traversability !== false;
    const showUncertain = layers?.unknownAreas !== false;

    // Draw all multi-resolution adaptive grid cells with seamless spatial coverage
    cells.forEach((cell) => {
      const { px, py } = worldToCanvas(cell.worldX, cell.worldY, width, height);
      const cellSizePx = Math.max(1.5, cell.size * zoom);
      const halfSize = cellSizePx / 2;

      const key = normaliseTrav(cell);
      const isActive =
        (key === 'drivable' && showDrivable) ||
        (key === 'caution' && showCaution) ||
        (key === 'hazard' && showHazard) ||
        (key === 'non_drivable' && showNonDrivable) ||
        (key === 'uncertain' && showUncertain);

      ctx.save();
      if (isActive) {
        ctx.fillStyle = TRAV_COLORS[key].color;
        ctx.globalAlpha = 0.88;
        ctx.fillRect(px - halfSize, py - halfSize, cellSizePx, cellSizePx);
      } else {
        ctx.fillStyle = 'rgba(40, 50, 70, 0.30)';
        ctx.fillRect(px - halfSize, py - halfSize, cellSizePx, cellSizePx);
      }
      ctx.globalAlpha = 1.0;
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.35)';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(px - halfSize, py - halfSize, cellSizePx, cellSizePx);
      ctx.restore();
    });

    // Selection outline last so it wins
    if (selectedCellId) {
      const sel = cells.find((c) => c.id === selectedCellId);
      if (sel) {
        const { px, py } = worldToCanvas(sel.worldX, sel.worldY, width, height);
        const cellSizePx = Math.max(2, sel.size * zoom);
        const halfSize = cellSizePx / 2;
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        ctx.strokeRect(px - halfSize - 1, py - halfSize - 1, cellSizePx + 2, cellSizePx + 2);
      }
    }

    // Ego Origin
    ctx.beginPath();
    ctx.arc(center.px, center.py, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = '#00E5FF';
    ctx.shadowColor = '#00E5FF';
    ctx.shadowBlur = 6;
    ctx.fill();
    ctx.shadowBlur = 0;
  }, [cells, zoom, panOffset, selectedCellId, worldToCanvas, layers]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    let rafId: number | null = null;
    let transitionTimer: ReturnType<typeof setTimeout> | null = null;

    const applyResize = () => {
      if (!container || !canvas) return;
      const w = container.clientWidth;
      const h = container.clientHeight || 220;
      if (w <= 0 || h <= 0) return;

      if (!userInteractedRef.current && cellsRef.current.length > 0 && w >= 150) {
        const fit = calculateAutoFitZoom(cellsRef.current, w, h);
        setZoom(fit);
      }

      const dpr = window.devicePixelRatio || 1;
      const targetW = Math.round(w * dpr);
      const targetH = Math.round(h * dpr);

      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }
      drawTraversabilityMap(w, h);
    };

    const scheduleResize = () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(applyResize);
      if (transitionTimer !== null) clearTimeout(transitionTimer);
      transitionTimer = setTimeout(applyResize, 350);
    };

    const resizeObserver = new ResizeObserver(() => {
      scheduleResize();
    });
    resizeObserver.observe(container);
    window.addEventListener('resize', scheduleResize);
    scheduleResize();

    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      if (transitionTimer !== null) clearTimeout(transitionTimer);
      resizeObserver.disconnect();
      window.removeEventListener('resize', scheduleResize);
    };
  }, [drawTraversabilityMap]);

  useEffect(() => {
    drawTraversabilityMap();
  }, [drawTraversabilityMap]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const clickPx = e.clientX - rect.left;
    const clickPy = e.clientY - rect.top;
    const { wx, wy } = canvasToWorld(clickPx, clickPy, rect.width, rect.height);

    for (const cell of cells) {
      const halfSize = cell.size / 2;
      if (
        wx >= cell.worldX - halfSize &&
        wx <= cell.worldX + halfSize &&
        wy >= cell.worldY - halfSize &&
        wy <= cell.worldY + halfSize
      ) {
        if (onCellSelect) onCellSelect(cell);
        break;
      }
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button === 0 && (e.shiftKey || e.ctrlKey)) {
      setIsPanning(true);
      userInteractedRef.current = true;
      dragStartRef.current = { x: e.clientX, y: e.clientY };
      panStartRef.current = { ...panOffset };
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isPanning) return;
    const dx = e.clientX - dragStartRef.current.x;
    const dy = e.clientY - dragStartRef.current.y;
    setPanOffset({ x: panStartRef.current.x + dx, y: panStartRef.current.y + dy });
  };

  const handleMouseUp = () => setIsPanning(false);

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    userInteractedRef.current = true;
    const zoomDelta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom((prev) => Math.max(0.8, Math.min(25, prev * zoomDelta)));
  };

  return (
    <div ref={containerRef} className={`relative overflow-hidden w-full h-full min-w-0 min-h-0 bg-dark-950 rounded-xl border border-white/10 ${className}`}>
      {cells.length === 0 ? (
        <div className="w-full h-full flex flex-col items-center justify-center p-4 text-center">
          <p className="text-xs text-gray-500 font-mono">No traversability map data available.</p>
          <p className="text-[10px] text-gray-600 mt-1">Run Terrain Analysis or 2.5D Adaptive Grid synthesis.</p>
        </div>
      ) : (
        <>
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onWheel={handleWheel}
            className="w-full h-full cursor-crosshair block"
          />

          <div className="absolute bottom-2 left-2 right-2 bg-dark-900/90 backdrop-blur-md px-2 py-1.5 rounded-lg border border-white/10 grid grid-cols-5 gap-1 text-[9px] font-mono pointer-events-none text-center">
            <span className="flex items-center justify-center gap-1 text-cyan-300 truncate">
              <span className="w-1.5 h-1.5 rounded-xs bg-[#00E5FF] shrink-0" />
              <span className="truncate">Drivable</span>
            </span>
            <span className="flex items-center justify-center gap-1 text-amber-300 truncate">
              <span className="w-1.5 h-1.5 rounded-xs bg-[#F59E0B] shrink-0" />
              <span className="truncate">Caution</span>
            </span>
            <span className="flex items-center justify-center gap-1 text-red-300 truncate">
              <span className="w-1.5 h-1.5 rounded-xs bg-[#EF4444] shrink-0" />
              <span className="truncate">Hazard</span>
            </span>
            <span className="flex items-center justify-center gap-1 text-gray-300 truncate">
              <span className="w-1.5 h-1.5 rounded-xs bg-[#475569] shrink-0" />
              <span className="truncate">Blocked</span>
            </span>
            <span className="flex items-center justify-center gap-1 text-purple-300 truncate">
              <span className="w-1.5 h-1.5 rounded-xs bg-[#7C3AED] shrink-0" />
              <span className="truncate">Unknown</span>
            </span>
          </div>

          <div className="absolute top-2.5 right-2.5 flex items-center gap-1.5 z-10">
            <div className="flex items-center bg-dark-900/90 rounded border border-white/10 p-0.5 font-mono shadow-md">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  userInteractedRef.current = true;
                  setZoom((z) => Math.min(25, z * 1.25));
                }}
                className="w-5 h-5 flex items-center justify-center rounded hover:bg-white/10 text-cyan-400 hover:text-cyan-300 transition-colors cursor-pointer"
                title="Zoom In (+)"
                aria-label="Zoom In"
              >
                <Plus className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  userInteractedRef.current = true;
                  setZoom((z) => Math.max(0.8, z * 0.8));
                }}
                className="w-5 h-5 flex items-center justify-center rounded hover:bg-white/10 text-cyan-400 hover:text-cyan-300 transition-colors cursor-pointer"
                title="Zoom Out (-)"
                aria-label="Zoom Out"
              >
                <Minus className="w-3.5 h-3.5" />
              </button>
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                autoFitView();
              }}
              className="px-2 py-0.5 rounded bg-dark-900/90 hover:bg-white/10 text-[10px] font-mono text-cyan-400 border border-white/10 cursor-pointer"
              title="Reset View"
            >
              Reset View
            </button>
          </div>
        </>
      )}
    </div>
  );
};
