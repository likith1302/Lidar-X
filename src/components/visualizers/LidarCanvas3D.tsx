import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { Plus, Minus } from 'lucide-react';
import { Point3D, ViewMode } from '../../types/lidar';
import { SEMANTIC_CLASSES, SemanticClassId } from '../../types/semantic';
import { BoundingBox3D } from '../../types/terrain';

interface LidarCanvas3DProps {
  points: Point3D[];
  viewMode?: ViewMode;
  showVehicleMarker?: boolean;
  pointSize?: number;
  grayscale?: boolean;
  categoryFilters?: Record<string, boolean>;
  highlightBox?: BoundingBox3D | null;
  highlightPoints?: Point3D[];
  emptyMessage?: string;
  className?: string;
  isInteractive?: boolean;
}

export const LidarCanvas3D: React.FC<LidarCanvas3DProps> = ({
  points,
  viewMode = 'perspective',
  showVehicleMarker = true,
  pointSize = 0.18,
  grayscale = false,
  categoryFilters,
  highlightBox = null,
  highlightPoints = [],
  emptyMessage,
  className = 'w-full h-full min-h-[300px]',
  isInteractive = true,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const pointsObjRef = useRef<THREE.Points | null>(null);
  const highlightBoxMeshRef = useRef<THREE.LineSegments | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Mouse interaction state
  const isDraggingRef = useRef(false);
  const previousMousePositionRef = useRef({ x: 0, y: 0 });
  const cameraSphericalRef = useRef({ radius: 32, theta: Math.PI / 4, phi: Math.PI / 3.2 });
  const cameraTargetRef = useRef(new THREE.Vector3(5, 0, 0));

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    // 1. Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07090e);
    scene.fog = new THREE.FogExp2(0x07090e, 0.012);
    sceneRef.current = scene;

    // 2. Camera setup
    const width = mount.clientWidth || 300;
    const height = mount.clientHeight || 260;
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 200);
    cameraRef.current = camera;

    // 3. Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height, false);
    renderer.domElement.style.position = 'absolute';
    renderer.domElement.style.top = '0';
    renderer.domElement.style.left = '0';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    renderer.domElement.style.display = 'block';
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Vehicle Origin Marker
    if (showVehicleMarker) {
      const egoGroup = new THREE.Group();

      // Chassis wireframe
      const boxGeo = new THREE.BoxGeometry(4.2, 1.8, 1.4);
      const boxMat = new THREE.MeshBasicMaterial({
        color: 0x00e5ff,
        wireframe: true,
        transparent: true,
        opacity: 0.6,
      });
      const boxMesh = new THREE.Mesh(boxGeo, boxMat);
      boxMesh.position.set(0, 0, -0.9);
      egoGroup.add(boxMesh);

      // Origin sensor beacon
      const beaconGeo = new THREE.SphereGeometry(0.2, 16, 16);
      const beaconMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
      const beacon = new THREE.Mesh(beaconGeo, beaconMat);
      beacon.position.set(0, 0, 0);
      egoGroup.add(beacon);

      // Concentric ground range rings
      const ringMat = new THREE.LineBasicMaterial({ color: 0x1c2331, transparent: true, opacity: 0.6 });
      [10, 20, 30].forEach((r) => {
        const ringGeo = new THREE.BufferGeometry();
        const pts: THREE.Vector3[] = [];
        for (let a = 0; a <= 64; a++) {
          const angle = (a / 64) * Math.PI * 2;
          pts.push(new THREE.Vector3(Math.cos(angle) * r, Math.sin(angle) * r, -1.73));
        }
        ringGeo.setFromPoints(pts);
        egoGroup.add(new THREE.Line(ringGeo, ringMat));
      });

      scene.add(egoGroup);
    }

    // 5. Render loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);

      const { radius, theta, phi } = cameraSphericalRef.current;
      const target = cameraTargetRef.current;
      camera.position.x = target.x + radius * Math.sin(phi) * Math.cos(theta);
      camera.position.y = target.y + radius * Math.sin(phi) * Math.sin(theta);
      camera.position.z = target.z + radius * Math.cos(phi);
      camera.lookAt(target);

      renderer.render(scene, camera);
    };
    animate();

    let rafId: number | null = null;
    let transitionTimer: ReturnType<typeof setTimeout> | null = null;

    const resizeRenderer = () => {
      if (!mount || !renderer || !camera || !scene) return;
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (w <= 0 || h <= 0) return;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();

      renderer.setPixelRatio(dpr);
      renderer.setSize(w, h, false);

      renderer.domElement.style.width = '100%';
      renderer.domElement.style.height = '100%';

      renderer.render(scene, camera);
    };

    const scheduleResize = () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        rafId = null;
        resizeRenderer();
      });

      if (transitionTimer !== null) clearTimeout(transitionTimer);
      transitionTimer = setTimeout(() => {
        transitionTimer = null;
        resizeRenderer();
      }, 350);
    };

    const resizeObserver = new ResizeObserver(() => {
      scheduleResize();
    });
    resizeObserver.observe(mount);
    window.addEventListener('resize', scheduleResize);
    requestAnimationFrame(resizeRenderer);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener('resize', scheduleResize);
      if (rafId !== null) cancelAnimationFrame(rafId);
      if (transitionTimer !== null) clearTimeout(transitionTimer);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      if (renderer.domElement && mount.contains(renderer.domElement)) {
        mount.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  // Update point cloud geometry
  const updatePointCloud = () => {
    const scene = sceneRef.current;
    if (!scene) return;

    if (pointsObjRef.current) {
      scene.remove(pointsObjRef.current);
      pointsObjRef.current.geometry.dispose();
      (pointsObjRef.current.material as THREE.Material).dispose();
      pointsObjRef.current = null;
    }

    if (!points || points.length === 0) return;

    // Filter points based on categoryFilters
    const filteredPoints = points.filter((p) => {
      if (!categoryFilters) return true;
      const clsId = p.semanticClass || 'unknown';
      if (categoryFilters[clsId] === false) return false;
      return true;
    });

    if (filteredPoints.length === 0) return;

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(filteredPoints.length * 3);
    const colors = new Float32Array(filteredPoints.length * 3);

    filteredPoints.forEach((p, idx) => {
      positions[idx * 3] = p.x;
      positions[idx * 3 + 1] = p.y;
      positions[idx * 3 + 2] = p.z;

      const color = new THREE.Color(0x8892b0);

      if (grayscale || viewMode === 'raw') {
        const intensity = p.intensity ?? 0.5;
        const shade = 0.25 + intensity * 0.7;
        color.setRGB(shade, shade, shade);
      } else if (viewMode === 'intensity') {
        const intensity = p.intensity ?? 0.5;
        if (intensity > 0.8) color.setHex(0x00e5ff);
        else if (intensity > 0.5) color.setHex(0xf59e0b);
        else color.setHex(0x64748b);
      } else {
        const clsId = (p.semanticClass || 'unknown') as SemanticClassId;
        const clsDef = SEMANTIC_CLASSES[clsId] || SEMANTIC_CLASSES.unknown;
        color.setStyle(clsDef.color);
      }

      colors[idx * 3] = color.r;
      colors[idx * 3 + 1] = color.g;
      colors[idx * 3 + 2] = color.b;
    });

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      sizeAttenuation: true,
    });

    const pointsObj = new THREE.Points(geometry, material);
    scene.add(pointsObj);
    pointsObjRef.current = pointsObj;
  };

  // Update Highlight Bounding Box
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    if (highlightBoxMeshRef.current) {
      scene.remove(highlightBoxMeshRef.current);
      highlightBoxMeshRef.current.geometry.dispose();
      (highlightBoxMeshRef.current.material as THREE.Material).dispose();
      highlightBoxMeshRef.current = null;
    }

    if (highlightBox) {
      const { min_x, max_x, min_y, max_y, min_z, max_z } = highlightBox;
      const dx = max_x - min_x || 1.0;
      const dy = max_y - min_y || 1.0;
      const dz = max_z - min_z || 1.0;
      const cx = (min_x + max_x) / 2;
      const cy = (min_y + max_y) / 2;
      const cz = (min_z + max_z) / 2;

      const boxGeo = new THREE.BoxGeometry(dx, dy, dz);
      const wireframe = new THREE.WireframeGeometry(boxGeo);
      const line = new THREE.LineSegments(
        wireframe,
        new THREE.LineBasicMaterial({ color: 0x00e5ff, linewidth: 2 })
      );
      line.position.set(cx, cy, cz);
      scene.add(line);
      highlightBoxMeshRef.current = line;
    }
  }, [highlightBox]);

  const applyCameraPreset = (mode: ViewMode) => {
    if (mode === 'top') {
      cameraSphericalRef.current = { radius: 36, theta: 0, phi: 0.01 };
      cameraTargetRef.current = new THREE.Vector3(8, 0, 0);
    } else if (mode === 'intensity') {
      cameraSphericalRef.current = { radius: 28, theta: Math.PI / 4.5, phi: Math.PI / 3.0 };
      cameraTargetRef.current = new THREE.Vector3(7, 0, 0);
    } else {
      cameraSphericalRef.current = { radius: 32, theta: Math.PI / 3.8, phi: Math.PI / 3.2 };
      cameraTargetRef.current = new THREE.Vector3(6, 0, 0);
    }
  };

  useEffect(() => {
    updatePointCloud();
  }, [points, viewMode, pointSize, grayscale, categoryFilters]);

  useEffect(() => {
    applyCameraPreset(viewMode);
  }, [viewMode]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!isInteractive) return;
    isDraggingRef.current = true;
    previousMousePositionRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingRef.current || !isInteractive) return;
    const deltaX = e.clientX - previousMousePositionRef.current.x;
    const deltaY = e.clientY - previousMousePositionRef.current.y;

    cameraSphericalRef.current.theta -= deltaX * 0.008;
    cameraSphericalRef.current.phi = Math.max(
      0.05,
      Math.min(Math.PI / 2 - 0.02, cameraSphericalRef.current.phi + deltaY * 0.008)
    );

    previousMousePositionRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (!isInteractive) return;
    e.preventDefault();
    const zoomFactor = e.deltaY * 0.03;
    cameraSphericalRef.current.radius = Math.max(
      5,
      Math.min(70, cameraSphericalRef.current.radius + zoomFactor)
    );
  };

  const handleResetCamera = () => {
    applyCameraPreset(viewMode);
  };

  return (
    <div
      ref={mountRef}
      className={`relative overflow-hidden w-full h-full min-w-0 min-h-0 cursor-grab active:cursor-grabbing ${className}`}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {(!points || points.length === 0) && (
        <div className="absolute inset-0 flex flex-col items-center justify-center p-4 text-center bg-dark-950/80 pointer-events-none">
          <p className="text-xs text-gray-400 font-mono">
            {emptyMessage || 'Upload a LiDAR frame to render the raw point cloud.'}
          </p>
        </div>
      )}

      {/* Floating Controls */}
      <div className="absolute top-2 right-2 flex items-center gap-1.5 z-10">
        <div className="flex items-center bg-dark-900/90 rounded border border-white/10 p-0.5 font-mono shadow-md">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              cameraSphericalRef.current.radius = Math.max(5, cameraSphericalRef.current.radius * 0.8);
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
              cameraSphericalRef.current.radius = Math.min(70, cameraSphericalRef.current.radius * 1.25);
            }}
            className="w-5 h-5 flex items-center justify-center rounded hover:bg-white/10 text-cyan-400 hover:text-cyan-300 transition-colors cursor-pointer"
            title="Zoom Out (-)"
            aria-label="Zoom Out"
          >
            <Minus className="w-3.5 h-3.5" />
          </button>
        </div>
        <button
          onClick={handleResetCamera}
          className="px-2 py-0.5 rounded bg-dark-900/90 hover:bg-white/10 text-[10px] font-mono text-cyan-400 border border-white/10 cursor-pointer"
          title="Reset Camera View"
        >
          Reset View
        </button>
      </div>
    </div>
  );
};
