import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import * as THREE from 'three';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { Plus, Minus } from 'lucide-react';
import { GridCell, MapLayersState } from '../../types/map';
import { SEMANTIC_CLASSES, SemanticClassId } from '../../types/semantic';
import { ObjectInstance, TrackedObject, OrientedBoundingBox3D } from '../../types/objects';
import { Point3D } from '../../types/lidar';
import { useAppState } from '../../context/AppStateContext';

export interface MapCanvas2DProps {
  cells: GridCell[];
  layers: MapLayersState;
  objects?: ObjectInstance[];
  tracks?: TrackedObject[];
  points?: Point3D[];
  dataset?: 'rellis' | 'semantickitti';
  onCellSelect?: (cell: GridCell) => void;
  onObjectSelect?: (instanceId: string) => void;
  selectedInstanceId?: string | null;
  className?: string;
}

// ──────────────────────────────────────────────────────────────────────────────
// Semantic Color System (RELLIS-3D + SemanticKITTI Unified Color Mapping)
// ──────────────────────────────────────────────────────────────────────────────

interface ClassStyle {
  color: number;
  hex: string;
  name: string;
}

function normalizeKey(str: string): string {
  return (str || '').toLowerCase().replace(/_/g, '-').trim();
}

const CLASS_STYLES: Record<string, ClassStyle> = {
  // RELLIS-3D Off-Road Classes
  dirt: { color: 0x8d6e63, hex: '#8d6e63', name: 'Dirt' },
  grass: { color: 0x7cb342, hex: '#7cb342', name: 'Grass' },
  tree: { color: 0x2e7d32, hex: '#2e7d32', name: 'Tree' },
  water: { color: 0x0284c7, hex: '#0284c7', name: 'Water' },
  sky: { color: 0x38bdf8, hex: '#38bdf8', name: 'Sky' },
  log: { color: 0xa16207, hex: '#a16207', name: 'Log' },
  bush: { color: 0x10b981, hex: '#10b981', name: 'Bush' },
  concrete: { color: 0x64748b, hex: '#64748b', name: 'Concrete' },
  barrier: { color: 0xef4444, hex: '#ef4444', name: 'Barrier' },
  puddle: { color: 0x06b6d4, hex: '#06b6d4', name: 'Puddle' },
  mud: { color: 0x713f12, hex: '#713f12', name: 'Mud' },
  rubble: { color: 0x78716c, hex: '#78716c', name: 'Rubble' },
  asphalt: { color: 0x475569, hex: '#475569', name: 'Asphalt' },

  // Dynamic Vehicles & Actors
  vehicle: { color: 0x00f0ff, hex: '#00f0ff', name: 'Vehicle' },
  'moving-vehicle': { color: 0x00f0ff, hex: '#00f0ff', name: 'Vehicle' },
  car: { color: 0x00f0ff, hex: '#00f0ff', name: 'Car' },
  'moving-car': { color: 0x00f0ff, hex: '#00f0ff', name: 'Moving Car' },
  truck: { color: 0x06b6d4, hex: '#06b6d4', name: 'Truck' },
  'moving-truck': { color: 0x06b6d4, hex: '#06b6d4', name: 'Moving Truck' },
  bus: { color: 0x14b8a6, hex: '#14b8a6', name: 'Bus' },
  'moving-bus': { color: 0x14b8a6, hex: '#14b8a6', name: 'Moving Bus' },
  'other-vehicle': { color: 0x38bdf8, hex: '#38bdf8', name: 'Other Vehicle' },
  'moving-other-vehicle': { color: 0x38bdf8, hex: '#38bdf8', name: 'Moving Vehicle' },
  'moving-on-rails': { color: 0x818cf8, hex: '#818cf8', name: 'Rail Vehicle' },

  // Pedestrians & VRUs
  person: { color: 0xf43f5e, hex: '#f43f5e', name: 'Person' },
  'moving-person': { color: 0xf43f5e, hex: '#f43f5e', name: 'Person' },
  pedestrian: { color: 0xf43f5e, hex: '#f43f5e', name: 'Pedestrian' },
  bicyclist: { color: 0xfb7185, hex: '#fb7185', name: 'Bicyclist' },
  'moving-bicyclist': { color: 0xfb7185, hex: '#fb7185', name: 'Bicyclist' },
  motorcyclist: { color: 0xf97316, hex: '#f97316', name: 'Motorcyclist' },
  'moving-motorcyclist': { color: 0xf97316, hex: '#f97316', name: 'Motorcyclist' },
  bicycle: { color: 0xa855f7, hex: '#a855f7', name: 'Bicycle' },
  'moving-bicycle': { color: 0xa855f7, hex: '#a855f7', name: 'Bicycle' },
  motorcycle: { color: 0xc084fc, hex: '#c084fc', name: 'Motorcycle' },
  'moving-motorcycle': { color: 0xc084fc, hex: '#c084fc', name: 'Motorcycle' },

  // Obstacles & Infrastructure
  pole: { color: 0xf97316, hex: '#f97316', name: 'Pole' },
  'traffic-sign': { color: 0xfbbf24, hex: '#fbbf24', name: 'Sign' },
  building: { color: 0xef4444, hex: '#ef4444', name: 'Building' },
  wall: { color: 0xdc2626, hex: '#dc2626', name: 'Wall' },
  fence: { color: 0xf59e0b, hex: '#f59e0b', name: 'Fence' },
  object: { color: 0xfb923c, hex: '#fb923c', name: 'Obstacle' },
  'other-object': { color: 0xfb923c, hex: '#fb923c', name: 'Obstacle' },
  'other-structure': { color: 0xb91c1c, hex: '#b91c1c', name: 'Structure' },

  // Vegetation
  vegetation: { color: 0x10b981, hex: '#10b981', name: 'Vegetation' },
  trunk: { color: 0x059669, hex: '#059669', name: 'Trunk' },
  terrain: { color: 0x16a34a, hex: '#16a34a', name: 'Terrain' },

  // Ground & Drivable
  road: { color: 0x0ea5e9, hex: '#0ea5e9', name: 'Road' },
  parking: { color: 0x38bdf8, hex: '#38bdf8', name: 'Parking' },
  sidewalk: { color: 0x64748b, hex: '#64748b', name: 'Sidewalk' },
  'other-ground': { color: 0x475569, hex: '#475569', name: 'Ground' },

  // Generic Categories
  drivable: { color: 0x0ea5e9, hex: '#0ea5e9', name: 'Drivable' },
  nondrivable: { color: 0x475569, hex: '#475569', name: 'Non-Drivable' },
  staticobstacle: { color: 0xf59e0b, hex: '#f59e0b', name: 'Obstacle' },
  'static-obstacle': { color: 0xf59e0b, hex: '#f59e0b', name: 'Obstacle' },
  dynamicobstacle: { color: 0x00f0ff, hex: '#00f0ff', name: 'Dynamic Obstacle' },
  'dynamic-obstacle': { color: 0x00f0ff, hex: '#00f0ff', name: 'Dynamic Obstacle' },
  infrastructure: { color: 0xef4444, hex: '#ef4444', name: 'Infrastructure' },

  // Unknown / Unlabeled
  unknown: { color: 0x64748b, hex: '#64748b', name: 'Unknown' },
  unlabeled: { color: 0x475569, hex: '#475569', name: 'Unlabeled' },
  outlier: { color: 0x334155, hex: '#334155', name: 'Outlier' },
  void: { color: 0x1e293b, hex: '#1e293b', name: 'Void' },
};

function getClassStyle(type: string): ClassStyle {
  const key = normalizeKey(type);
  return CLASS_STYLES[key] ?? CLASS_STYLES.unknown;
}

const MAX_CELL_INSTANCES = 16384;
const dummyMatrix = new THREE.Matrix4();
const dummyPosition = new THREE.Vector3();
const dummyQuaternion = new THREE.Quaternion();
const dummyScale = new THREE.Vector3();
const tempColor = new THREE.Color();

// ──────────────────────────────────────────────────────────────────────────────
// Reusable static geometries to eliminate GPU buffer allocation and GC churn on every frame
const UNIT_BOX_GEO = new THREE.BoxGeometry(1, 1, 1);
const UNIT_EDGES_GEO = new THREE.EdgesGeometry(UNIT_BOX_GEO);
const UNIT_CONE_GEO = new THREE.ConeGeometry(1, 1, 12);
UNIT_CONE_GEO.rotateZ(-Math.PI / 2);

const UNIT_FOOTPRINT_GEO = new THREE.BufferGeometry().setFromPoints([
  new THREE.Vector3(-0.5, -0.5, 0),
  new THREE.Vector3(0.5, -0.5, 0),
  new THREE.Vector3(0.5, 0.5, 0),
  new THREE.Vector3(-0.5, 0.5, 0),
  new THREE.Vector3(-0.5, -0.5, 0),
]);

const UNIT_HEADING_LINE_GEO = new THREE.BufferGeometry().setFromPoints([
  new THREE.Vector3(0, 0, 0),
  new THREE.Vector3(1, 0, 0),
]);

// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export const MapCanvas2D: React.FC<MapCanvas2DProps> = ({
  cells,
  layers,
  objects = [],
  tracks = [],
  points = [],
  dataset = 'semantickitti',
  onCellSelect,
  onObjectSelect,
  selectedInstanceId,
  className = 'w-full h-full min-h-[500px]',
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const labelRendererRef = useRef<CSS2DRenderer | null>(null);

  // Instanced Meshes for zero-allocation cell rendering
  const instancedPrismsRef = useRef<THREE.InstancedMesh | null>(null);
  const visibleCellsRef = useRef<GridCell[]>([]);

  // 3D Scene Groups
  const pointsGroupRef = useRef<THREE.Group | null>(null);
  const pointsMeshRef = useRef<THREE.Points | null>(null);
  const objectsGroupRef = useRef<THREE.Group | null>(null);
  const labelsGroupRef = useRef<THREE.Group | null>(null);
  const egoGroupRef = useRef<THREE.Group | null>(null);
  const floorGroupRef = useRef<THREE.Group | null>(null);
  const selectedOutlineMeshRef = useRef<THREE.Group | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const labelElementsRef = useRef<Map<string, HTMLDivElement>>(new Map());

  const { selectedCell, setSelectedCell, activeGridPolicy } = useAppState();

  const [viewAngle, setViewAngle] = useState<'3d' | 'top'>('3d');
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);

  // Camera Orbit & Pan State
  const isDraggingRef = useRef(false);
  const isPanningRef = useRef(false);
  const mouseStartRef = useRef({ x: 0, y: 0 });
  const cameraSphericalRef = useRef({ radius: 38, theta: -Math.PI / 2, phi: Math.PI / 4.8 });
  const cameraTargetRef = useRef(new THREE.Vector3(12, 0, -1.2));

  const nearThreshold = activeGridPolicy?.near_zone_max_distance_m || 12.0;
  const midThreshold = activeGridPolicy?.mid_zone_max_distance_m || 28.0;
  const farThreshold = activeGridPolicy?.far_zone_max_distance_m || 80.0;

  // ──────────────────────────────────────────────────────────────────────────────
  // Elevation & Ground Metrics
  // ──────────────────────────────────────────────────────────────────────────────

  const { minZ, maxZ, groundZ } = useMemo(() => {
    let min = Infinity;
    let max = -Infinity;
    let groundSum = 0;
    let groundCount = 0;

    cells.forEach((c) => {
      const z = c.backendAdaptiveCell?.elevation_mean ?? c.backendTerrain?.mean_z ?? 0;
      if (z < min) min = z;
      if (z > max) max = z;

      const semType = (c.semanticType || '') as string;
      const isDrivable =
        semType === 'drivable' ||
        semType === 'road' ||
        semType === 'dirt' ||
        c.backendAdaptiveCell?.dominant_category === 'drivable';

      if (isDrivable) {
        groundSum += z;
        groundCount += 1;
      }
    });

    if (min === Infinity) return { minZ: -2.5, maxZ: 2.5, groundZ: -1.6 };
    if (min === max) return { minZ: min - 1.0, maxZ: max + 1.0, groundZ: min };

    const estimatedGround = groundCount > 0 ? groundSum / groundCount : min + 0.2;
    return { minZ: min, maxZ: max, groundZ: estimatedGround };
  }, [cells]);

  // ──────────────────────────────────────────────────────────────────────────────
  // Scene Initialization
  // ──────────────────────────────────────────────────────────────────────────────

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    // 1. Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060911);
    scene.fog = new THREE.FogExp2(0x060911, 0.005);
    sceneRef.current = scene;

    // 2. Camera (Z-up coordinate system for ISO-8855 compliance)
    const width = mount.clientWidth || 800;
    const height = mount.clientHeight || 520;
    const camera = new THREE.PerspectiveCamera(46, width / height, 0.1, 750);
    camera.up.set(0, 0, 1);
    cameraRef.current = camera;

    // 3. WebGL Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height, false);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    renderer.domElement.style.position = 'absolute';
    renderer.domElement.style.top = '0';
    renderer.domElement.style.left = '0';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    renderer.domElement.style.display = 'block';
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. CSS2D Label Renderer
    const labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(width, height);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0';
    labelRenderer.domElement.style.left = '0';
    labelRenderer.domElement.style.width = '100%';
    labelRenderer.domElement.style.height = '100%';
    labelRenderer.domElement.style.overflow = 'hidden';
    labelRenderer.domElement.style.pointerEvents = 'none';
    mount.appendChild(labelRenderer.domElement);
    labelRendererRef.current = labelRenderer;

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.3);
    dirLight.position.set(30, 40, 55);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 140;
    dirLight.shadow.camera.left = -70;
    dirLight.shadow.camera.right = 70;
    dirLight.shadow.camera.top = 70;
    dirLight.shadow.camera.bottom = -70;
    dirLight.shadow.bias = -0.0003;
    scene.add(dirLight);

    const fillLight = new THREE.DirectionalLight(0x38bdf8, 0.45);
    fillLight.position.set(-30, -30, 25);
    scene.add(fillLight);

    const hemiLight = new THREE.HemisphereLight(0x38bdf8, 0x0f172a, 0.4);
    scene.add(hemiLight);

    // 6. Ego Vehicle Model & Sensor Origin
    const egoGroup = new THREE.Group();
    egoGroupRef.current = egoGroup;

    // Chassis Box
    const chassisGeo = new THREE.BoxGeometry(4.2, 1.9, 1.4);
    const chassisEdges = new THREE.EdgesGeometry(chassisGeo);
    const chassisWireframe = new THREE.LineSegments(
      chassisEdges,
      new THREE.LineBasicMaterial({ color: 0x00f0ff, linewidth: 2, transparent: true, opacity: 0.95 })
    );
    chassisWireframe.position.set(0, 0, 0.7);
    egoGroup.add(chassisWireframe);

    const chassisFill = new THREE.Mesh(
      chassisGeo,
      new THREE.MeshStandardMaterial({
        color: 0x08101e,
        roughness: 0.3,
        metalness: 0.85,
        transparent: true,
        opacity: 0.92,
      })
    );
    chassisFill.position.set(0, 0, 0.7);
    chassisFill.castShadow = true;
    egoGroup.add(chassisFill);

    // Forward Direction Arrow (Heading along +X)
    const arrowGeo = new THREE.ConeGeometry(0.5, 1.2, 16);
    arrowGeo.rotateZ(-Math.PI / 2);
    const arrowMesh = new THREE.Mesh(arrowGeo, new THREE.MeshBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.95 }));
    arrowMesh.position.set(2.8, 0, 0.7);
    egoGroup.add(arrowMesh);

    // LiDAR Sensor Dome Beacon
    const beacon = new THREE.Mesh(
      new THREE.CylinderGeometry(0.18, 0.18, 0.3, 16),
      new THREE.MeshBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.95 })
    );
    beacon.position.set(0, 0, 1.55);
    beacon.rotateX(Math.PI / 2);
    egoGroup.add(beacon);

    scene.add(egoGroup);

    // 7. Dynamic Groups & Shared Instanced Meshes
    const floorGroup = new THREE.Group();
    scene.add(floorGroup);
    floorGroupRef.current = floorGroup;

    // Layer 1: InstancedMesh for 2.5D elevation cells
    const unitBoxGeo = new THREE.BoxGeometry(1, 1, 1);
    const prismMat = new THREE.MeshStandardMaterial({
      roughness: 0.45,
      metalness: 0.1,
      transparent: true,
      opacity: 0.94,
    });
    const instancedPrisms = new THREE.InstancedMesh(unitBoxGeo, prismMat, MAX_CELL_INSTANCES);
    instancedPrisms.count = 0;
    instancedPrisms.castShadow = true;
    instancedPrisms.receiveShadow = true;
    instancedPrisms.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    scene.add(instancedPrisms);
    instancedPrismsRef.current = instancedPrisms;

    // Layer 2: Real LiDAR Semantic Points Cloud
    const pointsGroup = new THREE.Group();
    scene.add(pointsGroup);
    pointsGroupRef.current = pointsGroup;

    // Layer 3: 3D Objects & Tracks
    const objectsGroup = new THREE.Group();
    scene.add(objectsGroup);
    objectsGroupRef.current = objectsGroup;

    // Layer 4: Dynamic CSS2D Floating Labels
    const labelsGroup = new THREE.Group();
    scene.add(labelsGroup);
    labelsGroupRef.current = labelsGroup;

    // 8. Animation & Render Loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);

      const { radius, theta, phi } = cameraSphericalRef.current;
      const target = cameraTargetRef.current;
      camera.position.x = target.x + radius * Math.sin(phi) * Math.cos(theta);
      camera.position.y = target.y + radius * Math.sin(phi) * Math.sin(theta);
      camera.position.z = target.z + radius * Math.cos(phi);
      camera.lookAt(target);

      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    };
    animate();

    let rafId: number | null = null;
    let transitionTimer: ReturnType<typeof setTimeout> | null = null;

    const resizeRenderer = () => {
      if (!mount || !renderer || !camera || !labelRenderer || !scene) return;
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (w <= 0 || h <= 0) return;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);

      // Camera aspect ratio update without altering camera position, target, or mode
      camera.aspect = w / h;
      camera.updateProjectionMatrix();

      // Pass updateStyle: false so Three.js does NOT overwrite CSS width/height: 100%!
      renderer.setPixelRatio(dpr);
      renderer.setSize(w, h, false);

      // Guarantee canvas style stays strictly 100%
      renderer.domElement.style.width = '100%';
      renderer.domElement.style.height = '100%';

      // Update CSS2D label renderer and immediately reset style width/height to 100%
      labelRenderer.setSize(w, h);
      labelRenderer.domElement.style.width = '100%';
      labelRenderer.domElement.style.height = '100%';
      labelRenderer.domElement.style.overflow = 'hidden';

      // Render immediately to eliminate any blank frame or visual lag
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    };

    const scheduleResize = () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        rafId = null;
        resizeRenderer();
      });

      // Handle CSS grid transitions settling (e.g. 300ms transition-all)
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
      if (labelRenderer.domElement && mount.contains(labelRenderer.domElement)) {
        mount.removeChild(labelRenderer.domElement);
      }
      renderer.dispose();
      unitBoxGeo.dispose();
      prismMat.dispose();
    };
  }, []);

  // ──────────────────────────────────────────────────────────────────────────────
  // Update Floor Grid & Range Rings based on Ground Level & Grid Policy
  // ──────────────────────────────────────────────────────────────────────────────

  useEffect(() => {
    const floorGroup = floorGroupRef.current;
    const egoGroup = egoGroupRef.current;
    if (!floorGroup || !egoGroup) return;

    while (floorGroup.children.length > 0) {
      const obj = floorGroup.children[0];
      floorGroup.remove(obj);
      if (obj instanceof THREE.Mesh || obj instanceof THREE.Line || obj instanceof THREE.LineSegments) {
        obj.geometry.dispose();
        if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
        else obj.material.dispose();
      }
    }

    // Position ego vehicle on real road ground surface
    egoGroup.position.set(0, 0, groundZ);

    // Deep simulation void floor grid
    const floorZ = Math.min(-3.2, minZ - 0.8);
    const baseGrid = new THREE.GridHelper(300, 75, 0x00f0ff, 0x111c30);
    baseGrid.position.set(0, 0, floorZ);
    baseGrid.rotateX(Math.PI / 2);
    (baseGrid.material as THREE.Material).transparent = true;
    (baseGrid.material as THREE.Material).opacity = 0.22;
    floorGroup.add(baseGrid);

    // Concentric Foveation Zone Rings
    const ringSpecs = [
      { r: nearThreshold, color: 0x00f0ff, opacity: 0.65 },
      { r: midThreshold, color: 0x818cf8, opacity: 0.45 },
      { r: Math.min(50.0, farThreshold), color: 0xf59e0b, opacity: 0.30 },
      { r: 10.0, color: 0x38bdf8, opacity: 0.22 },
      { r: 25.0, color: 0x475569, opacity: 0.18 },
      { r: 40.0, color: 0x334155, opacity: 0.14 },
    ];

    ringSpecs.forEach(({ r, color, opacity }) => {
      const ringGeo = new THREE.BufferGeometry();
      const pts: THREE.Vector3[] = [];
      const segments = 96;
      for (let a = 0; a <= segments; a++) {
        const angle = (a / segments) * Math.PI * 2;
        pts.push(new THREE.Vector3(Math.cos(angle) * r, Math.sin(angle) * r, groundZ + 0.02));
      }
      ringGeo.setFromPoints(pts);
      const ringLine = new THREE.Line(
        ringGeo,
        new THREE.LineBasicMaterial({ color, transparent: true, opacity, linewidth: 1.5 })
      );
      floorGroup.add(ringLine);
    });
  }, [groundZ, minZ, nearThreshold, midThreshold, farThreshold]);

  // ──────────────────────────────────────────────────────────────────────────────
  // Layer 1: Seamless Continuous 2.5D Elevation Grid Cells
  // ──────────────────────────────────────────────────────────────────────────────

  useEffect(() => {
    const instancedPrisms = instancedPrismsRef.current;
    if (!instancedPrisms) return;

    if (cells.length === 0) {
      instancedPrisms.count = 0;
      visibleCellsRef.current = [];
      return;
    }

    const zSpan = Math.max(1.0, maxZ - minZ);
    const visibleCells: GridCell[] = [];

    let idx = 0;
    const totalCells = cells.length;

    for (let c = 0; c < totalCells; c++) {
      const cell = cells[c];

      // Layer filters
      if (!layers.staticObstacles && cell.isObstacle && !cell.isDynamic) continue;
      if (!layers.dynamicObstacles && cell.isDynamic) continue;
      if (!layers.unknownAreas && cell.semanticType === 'unknown') continue;

      if (idx >= MAX_CELL_INSTANCES) break;

      visibleCells.push(cell);

      const size = cell.size;
      const zMean = cell.backendAdaptiveCell?.elevation_mean ?? cell.backendTerrain?.mean_z ?? groundZ;
      const zMin = cell.backendAdaptiveCell?.elevation_min ?? cell.backendTerrain?.min_z ?? (zMean - 0.15);
      const zMax = cell.backendAdaptiveCell?.elevation_max ?? cell.backendTerrain?.max_z ?? (zMean + 0.15);
      const zVariation = cell.backendAdaptiveCell?.elevation_variation ?? cell.backendTerrain?.elevation_range ?? (zMax - zMin);

      const semType = (cell.semanticType || '') as string;
      const domCat = cell.backendAdaptiveCell?.dominant_category || '';
      const isVerticalObstacle =
        cell.isObstacle ||
        semType === 'vegetation' ||
        semType === 'infrastructure' ||
        semType === 'tree' ||
        semType === 'building' ||
        semType === 'pole' ||
        domCat === 'static_obstacle' ||
        (zVariation > 0.45 && semType !== 'drivable' && semType !== 'dirt' && semType !== 'road');

      let prismHeight: number;
      let centerZ: number;

      if (isVerticalObstacle) {
        const zBase = Math.max(minZ - 0.2, zMin);
        const zTop = zMax;
        prismHeight = Math.max(0.35, zTop - zBase);
        centerZ = zBase + prismHeight / 2;
      } else {
        // Continuous, seamless terrain ground tiles anchored to local elevation
        prismHeight = Math.max(0.08, Math.min(zVariation, 0.35));
        centerZ = zMean - prismHeight / 2;
      }

      // Base color according to active layer
      tempColor.setHex(0x0ea5e9);

      if (layers.semanticClasses) {
        const style = getClassStyle(cell.semanticType);
        tempColor.setHex(style.color);

        const normZ = Math.max(0, Math.min(1, (zMean - minZ) / zSpan));
        const lumOffset = (normZ - 0.5) * 0.14;
        tempColor.offsetHSL(0, 0, lumOffset);

        const slope = cell.backendAdaptiveCell?.slope_summary ?? 0;
        if (slope > 12.0) {
          tempColor.offsetHSL(0, 0.05, -0.06);
        }
      } else if (layers.elevation) {
        const t = Math.max(0, Math.min(1, (zMean - minZ) / zSpan));
        if (t < 0.25) {
          tempColor.lerpColors(new THREE.Color(0x1e3a8a), new THREE.Color(0x00e5ff), t / 0.25);
        } else if (t < 0.5) {
          tempColor.lerpColors(new THREE.Color(0x00e5ff), new THREE.Color(0x10b981), (t - 0.25) / 0.25);
        } else if (t < 0.75) {
          tempColor.lerpColors(new THREE.Color(0x10b981), new THREE.Color(0xf59e0b), (t - 0.5) / 0.25);
        } else {
          tempColor.lerpColors(new THREE.Color(0xf59e0b), new THREE.Color(0xef4444), (t - 0.75) / 0.25);
        }
      } else if (layers.traversability) {
        const trav = cell.backendAdaptiveCell?.traversability_state || cell.backendTerrain?.drivability_state || cell.traversability;
        if (trav === 'drivable' || trav === 'drivable_candidate' || trav === 'Drivable / Safe') tempColor.setHex(0x00e5ff);
        else if (trav === 'collision_hazard' || trav === 'obstacle_hazard' || trav === 'Collision Hazard') tempColor.setHex(0xef4444);
        else if (trav === 'caution_irregular' || trav === 'Caution / Irregular') tempColor.setHex(0xf59e0b);
        else if (trav === 'non_traversable' || trav === 'non_drivable_candidate' || trav === 'Non-Traversable') tempColor.setHex(0x475569);
        else tempColor.setHex(0x7c3aed);
      }

      // Seamless tile width (size * 1.0 eliminates artificial Minecraft-style grid gaps)
      const tileWidth = size * 1.0;
      dummyPosition.set(cell.worldX, cell.worldY, centerZ);
      dummyScale.set(tileWidth, tileWidth, prismHeight);
      dummyMatrix.compose(dummyPosition, dummyQuaternion, dummyScale);

      instancedPrisms.setMatrixAt(idx, dummyMatrix);
      instancedPrisms.setColorAt(idx, tempColor);

      idx++;
    }

    visibleCellsRef.current = visibleCells;

    instancedPrisms.count = idx;
    instancedPrisms.instanceMatrix.needsUpdate = true;
    if (instancedPrisms.instanceColor) instancedPrisms.instanceColor.needsUpdate = true;
  }, [cells, layers, minZ, maxZ, groundZ]);

  // ──────────────────────────────────────────────────────────────────────────────
  // Layer 2: Real Classified LiDAR Point Cloud (Overlaid Directly on the 2.5D Map)
  // ──────────────────────────────────────────────────────────────────────────────

  useEffect(() => {
    const pointsGroup = pointsGroupRef.current;
    if (!pointsGroup) return;

    if (!points || points.length === 0) {
      if (pointsMeshRef.current) {
        pointsMeshRef.current.visible = false;
      }
      return;
    }

    const N = points.length;
    let mesh = pointsMeshRef.current;

    // Allocate persistent buffer with capacity headroom once
    const posAttrExisting = mesh ? (mesh.geometry.getAttribute('position') as THREE.BufferAttribute | undefined) : undefined;
    if (!mesh || !posAttrExisting || posAttrExisting.count < N) {
      const capacity = Math.max(N * 2, 8000);
      if (mesh) {
        pointsGroup.remove(mesh);
        mesh.geometry.dispose();
        if (Array.isArray(mesh.material)) mesh.material.forEach((m) => m.dispose());
        else mesh.material.dispose();
      }

      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(capacity * 3);
      const colors = new Float32Array(capacity * 3);
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      const material = new THREE.PointsMaterial({
        size: 0.18,
        vertexColors: true,
        transparent: true,
        opacity: 0.92,
        sizeAttenuation: true,
      });

      mesh = new THREE.Points(geometry, material);
      pointsGroup.add(mesh);
      pointsMeshRef.current = mesh;
    }

    mesh.visible = true;
    const posAttr = mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
    const colAttr = mesh.geometry.getAttribute('color') as THREE.BufferAttribute;
    const posArray = posAttr.array as Float32Array;
    const colArray = colAttr.array as Float32Array;

    for (let i = 0; i < N; i++) {
      const pt = points[i];
      posArray[i * 3] = pt.x;
      posArray[i * 3 + 1] = pt.y;
      posArray[i * 3 + 2] = pt.z;

      const style = getClassStyle(pt.semanticClass || 'unknown');
      const hex = Number(style.color);
      colArray[i * 3] = ((hex >> 16) & 255) / 255;
      colArray[i * 3 + 1] = ((hex >> 8) & 255) / 255;
      colArray[i * 3 + 2] = (hex & 255) / 255;
    }

    mesh.geometry.setDrawRange(0, N);
    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
  }, [points, dataset]);

  // ──────────────────────────────────────────────────────────────────────────────
  // Layer 3 & 4: PCA Yaw Oriented Bounding Boxes & Realistic 3D Object Proxies
  // ──────────────────────────────────────────────────────────────────────────────

  useEffect(() => {
    const objectsGroup = objectsGroupRef.current;
    const labelsGroup = labelsGroupRef.current;
    if (!objectsGroup || !labelsGroup) return;

    while (objectsGroup.children.length > 0) {
      const obj = objectsGroup.children[0];
      objectsGroup.remove(obj);
      obj.traverse((child) => {
        if (child instanceof THREE.Mesh || child instanceof THREE.LineSegments || child instanceof THREE.Line) {
          if (
            child.geometry !== UNIT_BOX_GEO &&
            child.geometry !== UNIT_EDGES_GEO &&
            child.geometry !== UNIT_CONE_GEO &&
            child.geometry !== UNIT_FOOTPRINT_GEO &&
            child.geometry !== UNIT_HEADING_LINE_GEO
          ) {
            child.geometry.dispose();
          }
          if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
          else child.material.dispose();
        }
      });
    }

    while (labelsGroup.children.length > 0) {
      const obj = labelsGroup.children[0];
      labelsGroup.remove(obj);
    }
    labelElementsRef.current.clear();

    // 1. Build unified render items list
    interface RenderItem {
      id: string;
      objectType: string;
      isDynamic: boolean;
      bbox: { min_x: number; max_x: number; min_y: number; max_y: number; min_z: number; max_z: number };
      centroid: [number, number, number];
      velocity?: [number, number, number];
      trackId?: string;
      lifecycleState?: string;
      isTracked: boolean;
      distance: number;
      velMag: number;
      yaw?: number;
      oriented_bounding_box?: OrientedBoundingBox3D;
      confidence?: number;
      dimensions?: [number, number, number];
      pointCount?: number;
    }

    const trackedInstanceIds = new Set(tracks.map((t) => t.associated_instance_id).filter(Boolean));
    const renderItems: RenderItem[] = [];

    // Tracks (Kalman-filtered dynamic/static tracks)
    tracks.forEach((tr) => {
      const [cx, cy, cz] = tr.current_position;
      const dist = Math.hypot(cx, cy, cz);
      const [vx, vy, vz] = tr.estimated_velocity || [0, 0, 0];
      const velMag = Math.hypot(vx, vy, vz);
      const isDyn = tr.lifecycle_state !== 'expired' && (velMag > 0.4 || tr.object_type.startsWith('moving-'));

      renderItems.push({
        id: tr.track_id,
        objectType: tr.object_type,
        isDynamic: isDyn,
        bbox: tr.bounding_box,
        centroid: tr.current_position,
        velocity: tr.estimated_velocity,
        trackId: tr.track_id,
        lifecycleState: tr.lifecycle_state,
        isTracked: true,
        distance: dist,
        velMag: velMag,
        yaw: tr.yaw,
        oriented_bounding_box: tr.oriented_bounding_box,
        confidence: tr.confidence,
      });
    });

    // Untracked cluster instances
    objects.forEach((obj) => {
      if (!trackedInstanceIds.has(obj.instance_id)) {
        const [cx, cy, cz] = obj.centroid;
        const dist = Math.hypot(cx, cy, cz);
        renderItems.push({
          id: obj.instance_id,
          objectType: obj.object_type,
          isDynamic: obj.is_dynamic,
          bbox: obj.bounding_box,
          centroid: obj.centroid,
          isTracked: false,
          distance: dist,
          velMag: 0,
          yaw: obj.yaw,
          oriented_bounding_box: obj.oriented_bounding_box,
          confidence: obj.confidence,
          dimensions: obj.dimensions,
          pointCount: obj.point_count,
        });
      }
    });

    // 2. Render 3D Oriented Bounding Boxes for each item
    renderItems.forEach((item) => {
      const obb = item.oriented_bounding_box;
      let sizeX: number, sizeY: number, sizeZ: number;
      let centerX: number, centerY: number, centerZ: number;
      let yaw = item.yaw || 0.0;

      if (obb) {
        if (Array.isArray(obb.size) && obb.size.length === 3 && Array.isArray(obb.center) && obb.center.length === 3) {
          sizeX = Math.max(0.35, obb.size[0]);
          sizeY = Math.max(0.35, obb.size[1]);
          sizeZ = Math.max(0.35, obb.size[2]);
          centerX = obb.center[0];
          centerY = obb.center[1];
          centerZ = obb.center[2];
        } else if (obb.length !== undefined && obb.width !== undefined && obb.height !== undefined && obb.center_x !== undefined) {
          sizeX = Math.max(0.35, obb.length);
          sizeY = Math.max(0.35, obb.width);
          sizeZ = Math.max(0.35, obb.height);
          centerX = obb.center_x;
          centerY = obb.center_y ?? 0;
          centerZ = obb.center_z ?? 0;
        } else {
          const b = item.bbox;
          sizeX = Math.max(0.4, b.max_x - b.min_x);
          sizeY = Math.max(0.4, b.max_y - b.min_y);
          sizeZ = Math.max(0.4, b.max_z - b.min_z);
          centerX = (b.min_x + b.max_x) / 2;
          centerY = (b.min_y + b.max_y) / 2;
          centerZ = (b.min_z + b.max_z) / 2;
        }
        if (obb.yaw !== undefined) {
          yaw = obb.yaw;
        }
      } else {
        const b = item.bbox;
        sizeX = Math.max(0.4, b.max_x - b.min_x);
        sizeY = Math.max(0.4, b.max_y - b.min_y);
        sizeZ = Math.max(0.4, b.max_z - b.min_z);
        centerX = (b.min_x + b.max_x) / 2;
        centerY = (b.min_y + b.max_y) / 2;
        centerZ = (b.min_z + b.max_z) / 2;
      }

      // True 3D Euclidean distance from the sensor/ego origin (0, 0, 0)
      const distanceToOrigin = Math.hypot(centerX, centerY, centerZ);
      const isNearZone = distanceToOrigin <= nearThreshold;

      const isSelected = selectedInstanceId === item.id;
      const isHovered = hoveredObjectId === item.id;
      const style = getClassStyle(item.objectType);
      const isUnknown = item.objectType === 'unknown' || item.objectType === 'unlabeled';

      // Visual styling: Near zone objects are always prominent; hover adds highlight; outside near zone follows subtle default
      let wireColor = style.color;
      if (isSelected) {
        wireColor = 0x00ffff;
      } else if (isHovered) {
        wireColor = 0xffffff;
      } else if (item.isDynamic && item.velMag > 0.4) {
        wireColor = 0x00f0ff;
      }

      const wireOpacity = isSelected ? 1.0 : isHovered ? 1.0 : isNearZone ? 0.92 : isUnknown ? 0.25 : 0.55;
      const wireLineWidth = isSelected || isHovered ? 2.5 : isNearZone ? 2.0 : 1.5;

      // Master Oriented Object Group: Rotated by PCA Yaw in-place
      const itemGroup = new THREE.Group();
      itemGroup.position.set(centerX, centerY, centerZ);
      itemGroup.rotation.z = yaw;
      itemGroup.userData = { objectId: item.id, item, isNearZone, distance: distanceToOrigin };

      // ── Layer 3: PCA Yaw Oriented Bounding Box (OBB) Wireframe ──
      const wireframe = new THREE.LineSegments(
        UNIT_EDGES_GEO,
        new THREE.LineBasicMaterial({
          color: wireColor,
          linewidth: wireLineWidth,
          transparent: true,
          opacity: wireOpacity,
        })
      );
      wireframe.scale.set(sizeX, sizeY, sizeZ);
      wireframe.userData = { objectId: item.id };
      itemGroup.add(wireframe);

      // Subtle holographic fill inside box (crisp and visible for near zone objects)
      const fillOpacity = isSelected ? 0.28 : isHovered ? 0.22 : isNearZone ? 0.14 : isUnknown ? 0.02 : 0.04;
      const fillMat = new THREE.MeshBasicMaterial({
        color: wireColor,
        transparent: true,
        opacity: fillOpacity,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const fillMesh = new THREE.Mesh(UNIT_BOX_GEO, fillMat);
      fillMesh.scale.set(sizeX, sizeY, sizeZ);
      fillMesh.userData = { objectId: item.id };
      itemGroup.add(fillMesh);

      // Ground footprint projection at bottom face (z = -sizeZ / 2)
      const hwX = sizeX / 2;
      const groundOffsetZ = -sizeZ / 2;
      const footOpacity = isSelected ? 1.0 : isHovered ? 0.95 : isNearZone ? 0.85 : 0.45;
      const footLine = new THREE.Line(
        UNIT_FOOTPRINT_GEO,
        new THREE.LineBasicMaterial({ color: wireColor, transparent: true, opacity: footOpacity })
      );
      footLine.scale.set(sizeX, sizeY, 1);
      footLine.position.set(0, 0, groundOffsetZ);
      footLine.userData = { objectId: item.id };
      itemGroup.add(footLine);

      // Forward heading indicator arrow (along local +X)
      const arrowLen = Math.min(0.55, sizeX * 0.28);
      const arrowRadius = Math.min(0.22, sizeY * 0.2);
      const arrowMesh = new THREE.Mesh(
        UNIT_CONE_GEO,
        new THREE.MeshBasicMaterial({ color: wireColor, transparent: true, opacity: isNearZone || isSelected || isHovered ? 0.95 : 0.6 })
      );
      arrowMesh.scale.set(arrowLen, arrowRadius, arrowRadius);
      arrowMesh.position.set(hwX + Math.min(0.25, sizeX * 0.12), 0, groundOffsetZ + 0.08);
      arrowMesh.userData = { objectId: item.id };
      itemGroup.add(arrowMesh);

      // Heading axis line from center to front
      const headingLen = hwX + Math.min(0.25, sizeX * 0.12);
      const headingLine = new THREE.Line(
        UNIT_HEADING_LINE_GEO,
        new THREE.LineBasicMaterial({ color: wireColor, transparent: true, opacity: isNearZone || isSelected || isHovered ? 0.85 : 0.5 })
      );
      headingLine.scale.set(headingLen, 1, 1);
      headingLine.position.set(0, 0, groundOffsetZ + 0.08);
      headingLine.userData = { objectId: item.id };
      itemGroup.add(headingLine);

      // Dynamic velocity vector arrow in world coordinates
      if (item.velocity && item.velMag > 0.35) {
        const [vx, vy, vz] = item.velocity;
        const arrowLen = Math.min(item.velMag * 1.4, 6.0);
        const worldDir = new THREE.Vector3(vx, vy, vz).normalize();
        const arrow = new THREE.ArrowHelper(worldDir, new THREE.Vector3(centerX, centerY, centerZ), arrowLen, 0x00f0ff, 0.4, 0.25);
        objectsGroup.add(arrow);
      }

      objectsGroup.add(itemGroup);

      // ── Layer 4: Object Marker & Interactive Floating Labels ──
      // Near-zone objects are always marked and visible. Hover or selection enhances with full detailed metrics.
      const shouldShowLabel = isNearZone || isSelected || isHovered;

      if (shouldShowLabel) {
        const labelDiv = document.createElement('div');
        const isDynamicMove = item.isDynamic && item.velMag > 0.4;
        const accentHex = isSelected ? '#00ffff' : isDynamicMove ? '#00f0ff' : style.hex;
        const trackNum = item.trackId ? item.trackId.replace(/[^0-9]/g, '') || item.trackId : null;
        const confPct = Math.round((item.confidence ?? 0.88) * 100);
        const dimsStr = `${sizeX.toFixed(1)}×${sizeY.toFixed(1)}×${sizeZ.toFixed(1)}m`;

        let statusText = '';
        if (isDynamicMove) statusText = `${item.velMag.toFixed(1)} m/s`;
        else if (item.isDynamic) statusText = 'parked';

        if (isSelected || isHovered) {
          // Detailed Inspection Card (on hover or click selection)
          labelDiv.style.cssText = `
            background: rgba(8, 12, 22, 0.94);
            border: 1.5px solid ${accentHex};
            box-shadow: 0 4px 18px ${accentHex}55, 0 0 10px rgba(0,0,0,0.8);
            color: #f8fafc;
            padding: 5px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-family: "JetBrains Mono", "Fira Code", monospace;
            white-space: nowrap;
            pointer-events: none;
            transform: translate(-50%, -100%);
            display: flex;
            flex-direction: column;
            gap: 3px;
            backdrop-filter: blur(6px);
            z-index: 100;
          `;

          labelDiv.innerHTML = `
            <div style="display:flex; align-items:center; justify-content:space-between; gap:8px; border-bottom:1px solid rgba(255,255,255,0.12); padding-bottom:3px;">
              <div style="display:flex; align-items:center; gap:5px;">
                <span style="width:7px; height:7px; border-radius:50%; background:${accentHex}; box-shadow:0 0 6px ${accentHex}; flex-shrink:0;"></span>
                <span style="color:#ffffff; font-weight:700;">${style.name}</span>
                ${trackNum ? `<span style="color:${accentHex}; opacity:0.85; font-size:10px;">#${trackNum}</span>` : ''}
              </div>
              <span style="font-size:9px; background:${isNearZone ? 'rgba(0,240,255,0.2)' : 'rgba(255,255,255,0.1)'}; color:${isNearZone ? '#00f0ff' : '#94a3b8'}; border:1px solid ${isNearZone ? 'rgba(0,240,255,0.4)' : 'rgba(255,255,255,0.2)'}; border-radius:3px; padding:0 3px; font-weight:700;">
                ${isNearZone ? 'NEAR ZONE' : 'MID/FAR'}
              </span>
            </div>
            <div style="display:grid; grid-template-columns:auto auto; gap:2px 8px; font-size:10px; color:#94a3b8;">
              <div>Dist: <span style="color:#f8fafc; font-weight:600;">${distanceToOrigin.toFixed(1)}m</span></div>
              <div>Conf: <span style="color:#34d399; font-weight:600;">${confPct}%</span></div>
              <div>Dim: <span style="color:#cbd5e1;">${dimsStr}</span></div>
              <div>State: <span style="color:${isDynamicMove ? '#00f0ff' : '#cbd5e1'}; font-weight:600;">${statusText || 'static'}</span></div>
            </div>
          `;
        } else {
          // Permanent Near-Zone Object Marker Pill (clean, non-obtrusive, always visible without hover)
          labelDiv.style.cssText = `
            background: rgba(8, 12, 20, 0.88);
            border: 1px solid ${accentHex}99;
            box-shadow: 0 2px 10px rgba(0,0,0,0.6);
            color: #f8fafc;
            padding: 2.5px 6px;
            border-radius: 4px;
            font-size: 10.5px;
            font-family: "JetBrains Mono", "Fira Code", monospace;
            font-weight: 600;
            white-space: nowrap;
            pointer-events: none;
            transform: translate(-50%, -100%);
            display: flex;
            align-items: center;
            gap: 5px;
            backdrop-filter: blur(4px);
          `;

          labelDiv.innerHTML = `
            <span style="width:6px; height:6px; border-radius:50%; background:${accentHex}; box-shadow:0 0 5px ${accentHex}; flex-shrink:0;"></span>
            <span style="color:#ffffff; font-weight:700;">${style.name}</span>
            ${trackNum ? `<span style="color:${accentHex}; opacity:0.85; font-size:10px;">#${trackNum}</span>` : ''}
            <span style="color:#94a3b8; font-size:10px;">${distanceToOrigin.toFixed(1)}m</span>
          `;
        }

        const label = new CSS2DObject(labelDiv);
        label.position.set(centerX, centerY, centerZ + sizeZ / 2 + 0.35);
        label.userData = { objectId: item.id };
        labelsGroup.add(label);
        labelElementsRef.current.set(item.id, labelDiv);
      }
    });
  }, [objects, tracks, selectedInstanceId, hoveredObjectId, nearThreshold]);

  // ──────────────────────────────────────────────────────────────────────────────
  // Selected Grid Cell Holographic Cursor
  // ──────────────────────────────────────────────────────────────────────────────

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    if (selectedOutlineMeshRef.current) {
      scene.remove(selectedOutlineMeshRef.current);
      selectedOutlineMeshRef.current.traverse((child) => {
        if (child instanceof THREE.Mesh || child instanceof THREE.LineSegments) {
          child.geometry.dispose();
          if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
          else child.material.dispose();
        }
      });
      selectedOutlineMeshRef.current = null;
    }

    if (selectedCell) {
      const cursorGroup = new THREE.Group();
      const size = selectedCell.size;
      const zMean = selectedCell.backendAdaptiveCell?.elevation_mean ?? selectedCell.backendTerrain?.mean_z ?? groundZ;
      const zVariation = selectedCell.backendAdaptiveCell?.elevation_variation ?? selectedCell.backendTerrain?.elevation_range ?? 0.2;
      const prismHeight = Math.max(0.25, zVariation) + 0.1;

      // Outer glowing wireframe box
      const boxGeo = new THREE.BoxGeometry(size * 1.05, size * 1.05, prismHeight);
      const edges = new THREE.EdgesGeometry(boxGeo);
      const outline = new THREE.LineSegments(
        edges,
        new THREE.LineBasicMaterial({ color: 0x00ffff, linewidth: 2.5, transparent: true, opacity: 1.0 })
      );
      cursorGroup.add(outline);

      // Top face holographic fill
      const topFill = new THREE.Mesh(
        new THREE.PlaneGeometry(size * 1.05, size * 1.05),
        new THREE.MeshBasicMaterial({ color: 0x00ffff, transparent: true, opacity: 0.25, side: THREE.DoubleSide })
      );
      topFill.position.set(0, 0, prismHeight / 2 + 0.01);
      cursorGroup.add(topFill);

      cursorGroup.position.set(selectedCell.worldX, selectedCell.worldY, zMean);
      scene.add(cursorGroup);
      selectedOutlineMeshRef.current = cursorGroup;
    }
  }, [selectedCell, groundZ]);

  // ──────────────────────────────────────────────────────────────────────────────
  // Camera & Mouse Interactions (True Top-Down + 3D Orbital Modes)
  // ──────────────────────────────────────────────────────────────────────────────

  const handleToggleView = (mode: '3d' | 'top') => {
    setViewAngle(mode);
    const camera = cameraRef.current;
    if (!camera) return;

    if (mode === 'top') {
      // Align ISO-8855 Forward (+X) to Screen UP, Left (+Y) to Screen LEFT
      camera.up.set(1, 0, 0);
      cameraSphericalRef.current = { radius: 48, theta: -Math.PI / 2, phi: 0.001 };
      cameraTargetRef.current = new THREE.Vector3(12, 0, groundZ);
    } else {
      // Align Z-axis to camera UP for orbital rotation around scene
      camera.up.set(0, 0, 1);
      cameraSphericalRef.current = { radius: 38, theta: -Math.PI / 2, phi: Math.PI / 4.8 };
      cameraTargetRef.current = new THREE.Vector3(12, 0, groundZ + 0.5);
    }
  };

  const handleResetView = () => {
    handleToggleView(viewAngle);
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button === 2 || e.shiftKey) {
      isPanningRef.current = true;
    } else {
      isDraggingRef.current = true;
    }
    mouseStartRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const dx = e.clientX - mouseStartRef.current.x;
    const dy = e.clientY - mouseStartRef.current.y;
    mouseStartRef.current = { x: e.clientX, y: e.clientY };

    if (isDraggingRef.current) {
      if (viewAngle === 'top') {
        // In top-down view, left-drag pans directly in screen coordinates
        const factor = cameraSphericalRef.current.radius * 0.0009;
        cameraTargetRef.current.x -= dy * factor;
        cameraTargetRef.current.y -= dx * factor;
      } else {
        cameraSphericalRef.current.theta -= dx * 0.006;
        cameraSphericalRef.current.phi = Math.max(0.04, Math.min(Math.PI / 2.05, cameraSphericalRef.current.phi - dy * 0.006));
      }
    } else if (isPanningRef.current) {
      const factor = cameraSphericalRef.current.radius * 0.0009;
      if (viewAngle === 'top') {
        cameraTargetRef.current.x -= dy * factor;
        cameraTargetRef.current.y -= dx * factor;
      } else {
        const cosTheta = Math.cos(cameraSphericalRef.current.theta);
        const sinTheta = Math.sin(cameraSphericalRef.current.theta);
        cameraTargetRef.current.x += (sinTheta * dx - cosTheta * dy) * factor;
        cameraTargetRef.current.y += (-cosTheta * dx - sinTheta * dy) * factor;
      }
    } else {
      // Hover Raycasting for objects
      const mount = mountRef.current;
      const camera = cameraRef.current;
      if (mount && camera && objectsGroupRef.current) {
        const rect = mount.getBoundingClientRect();
        const mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        const mouseY = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(new THREE.Vector2(mouseX, mouseY), camera);

        const hits = raycaster.intersectObjects(objectsGroupRef.current.children, true);
        if (hits.length > 0) {
          let cur: THREE.Object3D | null = hits[0].object;
          while (cur && !cur.userData?.objectId && cur.parent) {
            cur = cur.parent;
          }
          if (cur?.userData?.objectId) {
            setHoveredObjectId(cur.userData.objectId);
            return;
          }
        }
        setHoveredObjectId(null);
      }
    }
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
    isPanningRef.current = false;
  };

  const handleZoomIn = () => {
    cameraSphericalRef.current.radius = Math.max(5, cameraSphericalRef.current.radius * 0.8);
  };

  const handleZoomOut = () => {
    cameraSphericalRef.current.radius = Math.min(220, cameraSphericalRef.current.radius * 1.25);
  };

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const mount = mountRef.current;
    const camera = cameraRef.current;
    if (!mount || !camera) return;

    const rect = mount.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(x, y), camera);

    // 1. Check Objects Group first
    if (objectsGroupRef.current) {
      const objectHits = raycaster.intersectObjects(objectsGroupRef.current.children, true);
      if (objectHits.length > 0) {
        let current: THREE.Object3D | null = objectHits[0].object;
        while (current && !current.userData?.objectId && current.parent) {
          current = current.parent;
        }
        if (current?.userData?.objectId) {
          if (onObjectSelect) onObjectSelect(current.userData.objectId);
          return;
        }
      }
    }

    // 2. Check Instanced Cells Mesh
    if (instancedPrismsRef.current && instancedPrismsRef.current.count > 0) {
      const cellHits = raycaster.intersectObject(instancedPrismsRef.current, false);
      if (cellHits.length > 0 && cellHits[0].instanceId !== undefined) {
        const hitIdx = cellHits[0].instanceId;
        const hitCell = visibleCellsRef.current[hitIdx];
        if (hitCell) {
          setSelectedCell(hitCell);
          if (onCellSelect) onCellSelect(hitCell);
        }
      }
    }
  };

  // ──────────────────────────────────────────────────────────────────────────────
  // Render JSX
  // ──────────────────────────────────────────────────────────────────────────────

  return (
    <div
      ref={mountRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onClick={handleClick}
      onContextMenu={(e) => e.preventDefault()}
      className={`relative overflow-hidden w-full h-full min-w-0 min-h-0 bg-dark-950 rounded-xl border border-white/10 select-none ${
        hoveredObjectId ? 'cursor-pointer' : 'cursor-grab active:cursor-grabbing'
      } ${className}`}
    >
      {/* Empty State Overlay Message (if no cells and no points) */}
      {cells.length === 0 && points.length === 0 && (
        <div className="absolute inset-0 z-0 flex flex-col items-center justify-center p-6 text-center pointer-events-none bg-dark-950/40 backdrop-blur-[1px]">
          <p className="text-sm font-semibold text-gray-300 font-mono">No 2.5D Adaptive Grid cells synthesized.</p>
          <p className="text-xs text-gray-500 mt-1 max-w-md leading-relaxed font-mono">
            Execute the 2.5D Adaptive Grid perception pipeline or replay a sequence to synthesize cells.
          </p>
        </div>
      )}


      {/* Top-Right Camera & Zoom Controls Toolbar */}
      <div className="absolute top-3 right-3 flex items-center gap-2 z-20">
        {/* Real-Map Zoom (+ / -) Control Bar */}
        <div className="flex items-center bg-dark-900/95 backdrop-blur-md rounded-lg border border-white/15 p-0.5 shadow-2xl font-mono">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              handleZoomIn();
            }}
            className="w-7 h-7 flex items-center justify-center rounded hover:bg-white/15 text-cyan-400 hover:text-cyan-300 transition-colors active:scale-95 cursor-pointer font-bold"
            title="Zoom In (+)"
            aria-label="Zoom In"
          >
            <Plus className="w-4 h-4 stroke-[2.5]" />
          </button>
          <span className="text-gray-600 text-xs mx-0.5">|</span>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              handleZoomOut();
            }}
            className="w-7 h-7 flex items-center justify-center rounded hover:bg-white/15 text-cyan-400 hover:text-cyan-300 transition-colors active:scale-95 cursor-pointer font-bold"
            title="Zoom Out (-)"
            aria-label="Zoom Out"
          >
            <Minus className="w-4 h-4 stroke-[2.5]" />
          </button>
        </div>

        {/* View Angle & Reset Mode */}
        <div className="flex items-center gap-1.5 bg-dark-900/95 backdrop-blur-md px-2.5 py-1 rounded-lg border border-white/15 text-xs font-mono shadow-2xl">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              handleToggleView(viewAngle === '3d' ? 'top' : '3d');
            }}
            className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-colors cursor-pointer ${
              viewAngle === 'top'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                : 'bg-dark-800 text-gray-300 hover:text-white'
            }`}
          >
            {viewAngle === 'top' ? 'Top-Down 2D' : '3D Perspective'}
          </button>
          <span className="text-gray-600 mx-1">|</span>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              handleResetView();
            }}
            className="px-2.5 py-1 rounded hover:bg-white/10 text-cyan-400 font-semibold text-[11px] transition-colors cursor-pointer"
            title="Reset View Position"
          >
            Reset
          </button>
        </div>
      </div>


      {/* Bottom Status Bar */}
      {(cells.length > 0 || points.length > 0) && (
        <div className="absolute bottom-3 left-3 flex flex-wrap items-center gap-3 bg-dark-900/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 text-[11px] font-mono text-gray-300 pointer-events-none z-10 shadow-lg">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span className="text-white font-semibold">2.5D Adaptive Grid</span>
          </span>
          <span className="text-gray-600">|</span>
          <span>Cells: <strong className="text-cyan-300">{cells.length}</strong></span>
          <span className="text-gray-600">|</span>
          <span>Points: <strong className="text-emerald-300">{points.length}</strong></span>
          <span className="text-gray-600">|</span>
          <span>Objects: <strong className="text-amber-300">{objects.length}</strong></span>
          <span className="text-gray-600">|</span>
          <span>Tracks: <strong className="text-purple-300">{tracks.length}</strong></span>
          <span className="text-gray-600">|</span>
          <span>Z: <strong className="text-gray-200">{minZ.toFixed(1)}m ~ +{maxZ.toFixed(1)}m</strong></span>
        </div>
      )}
    </div>
  );
};