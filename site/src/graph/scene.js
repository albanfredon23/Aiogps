/**
 * Scène 3D Aiogps (Three.js) : le pipeline AIOTECH44 en motion design.
 *
 * Requête → porte de calcul adaptative → graphe de connaissances (nœuds
 * activés selon la complexité) → faisceau TAP (top-k) → sphère SCG (élagage
 * géométrique) → mémoire différentielle (k documents sur 15) → agents
 * spécialisés → décision et trace auditable.
 *
 * - Les unités de calcul sont des particules : une partie est économisée dès
 *   la porte adaptative (elles s'élèvent en vert, « calcul évité »), une autre
 *   est élaguée par le SCG (elles rougissent et tombent).
 * - La sphère SCG porte le vecteur de contraintes : les trajectoires qui
 *   aboutissent dans l'hémisphère opposé violent la contrainte et rougissent.
 *
 * Performance : pixel ratio adaptatif (cible 60 FPS), rendu suspendu hors
 * écran ou onglet masqué, particules réduites sur mobile.
 * Mémoire : dispose() libère géométries, matériaux, textures, environnement
 * et contexte WebGL.
 */
import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';

const C = {
  bg: 0x060c10,
  query: 0x7dd3fc,
  gate: 0xa78bfa,
  graph: 0x94a3b8,
  graphOn: 0x67e8f9,
  tap: 0x2dd4bf,
  scg: 0xfbbf24,
  memory: 0x93c5fd,
  agent: 0xc4b5fd,
  decision: 0x34d399,
  trace: 0xe2e8f0,
  saved: 0x6ee7b7,
  reject: 0xff5a5a,
  edge: 0x334155,
};

export const NODES = [
  { id: 'query', label: 'Requête', pos: [-8, 0, 0], color: C.query, size: 0.55, stage: 1, target: '#etape-porte',
    desc: 'Embedding de la requête utilisateur.' },
  { id: 'gate', label: 'Porte adaptative', pos: [-5.6, 0, 0], color: C.gate, size: 0.62, stage: 1, kind: 'ring',
    target: '#etape-porte', desc: "Évalue la complexité et module l'effort de calcul (30 à 100 %)." },
  { id: 'tap', label: 'Faisceau TAP', pos: [-0.4, 0, 0], color: C.tap, size: 0.55, stage: 2, target: '#etape-tap',
    desc: 'Sélection différentiable des k meilleures trajectoires (Gumbel-Softmax).' },
  { id: 'scg', label: 'Sphère SCG', pos: [2.7, 0, 0], color: C.scg, size: 0.3, stage: 3, target: '#etape-scg',
    desc: 'Élague les trajectoires qui violent les contraintes latentes.' },
  { id: 'memory', label: 'Mémoire différentielle', pos: [5.3, 0, 0], color: C.memory, size: 0.5, stage: 4, kind: 'stack',
    target: '#etape-memoire', desc: 'Ne garde que k documents de contexte, de 2 à 15.' },
  { id: 'reasoning', label: 'Raisonnement', pos: [7.1, 1.9, -0.5], color: C.agent, size: 0.24, stage: 5, target: '#etape-agents',
    desc: 'Agent déductif.' },
  { id: 'math', label: 'Math', pos: [7.1, 0.65, 0.8], color: C.agent, size: 0.24, stage: 5, target: '#etape-agents',
    desc: 'Agent de calcul numérique.' },
  { id: 'logic', label: 'Logique', pos: [7.1, -0.65, 0.8], color: C.agent, size: 0.24, stage: 5, target: '#etape-agents',
    desc: 'Agent symbolique.' },
  { id: 'critic', label: 'Critique', pos: [7.1, -1.9, -0.5], color: C.agent, size: 0.24, stage: 5, target: '#etape-agents',
    desc: 'Agent de validation.' },
  { id: 'decision', label: 'Décision', pos: [9, 0, 0], color: C.decision, size: 0.7, stage: 6, target: '#etape-decision',
    desc: 'Réponse produite, avec sa trace auditable.' },
];

const GRAPH_CENTER = [-3.1, 0, 0];
const AGENT_IDS = ['reasoning', 'math', 'logic', 'critic'];
const ROUTE = ['query', 'gate', 'graph', 'tap', 'scg', 'memory', null, 'decision'];
const SEGMENTS = ROUTE.length - 1;
const T_GATE = 1 / SEGMENTS;
const T_SCG = 4 / SEGMENTS;

// Vues caméra par étape : [position, cible]
const VIEWS = [
  [[-0.6, 0.6, 28], [-0.6, -0.6, 0]],
  [[-5.4, 1.2, 10.5], [-5.2, 0, 0]],
  [[-1.6, 1.2, 10], [-1.5, 0, 0]],
  [[2.6, 1.6, 7.5], [2.6, 0, 0]],
  [[5.2, 1.0, 8], [5.2, 0, 0]],
  [[6.8, 0.6, 9], [6.8, 0, 0]],
  [[7.0, -0.6, 12], [7.0, -1.5, 0]],
];

function vec(p) {
  return new THREE.Vector3(p[0], p[1], p[2]);
}

function makeLabelTexture(text) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const font = '600 44px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
  ctx.font = font;
  const width = Math.ceil(ctx.measureText(text).width) + 56;
  canvas.width = width;
  canvas.height = 84;
  ctx.font = font;
  ctx.fillStyle = 'rgba(6, 12, 16, 0.8)';
  ctx.beginPath();
  ctx.roundRect(0, 0, width, 84, 24);
  ctx.fill();
  ctx.fillStyle = '#eef4f2';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 28, 44);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return { texture, aspect: width / 84 };
}

function makeDotTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext('2d');
  const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.35, 'rgba(255,255,255,0.85)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export function createGraph({ canvas, tooltip, reducedMotion = false, onContextLost, onNodeActivate }) {
  const isSmall = window.matchMedia('(max-width: 720px)').matches;
  const maxDpr = Math.min(window.devicePixelRatio || 1, 2);
  let dpr = isSmall ? Math.min(maxDpr, 1.5) : maxDpr;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(dpr);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.setClearColor(C.bg, 0);

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(C.bg, 30, 60);

  // Éclairage réaliste : environnement studio pré-filtré (PBR) + lumières douces.
  const pmrem = new THREE.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  const envTexture = pmrem.fromScene(room, 0.04).texture;
  scene.environment = envTexture;
  room.traverse((o) => {
    o.geometry?.dispose();
    o.material?.dispose?.();
  });
  pmrem.dispose();
  scene.add(new THREE.HemisphereLight(0xc8f0e4, 0x081014, 0.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(4, 8, 6);
  scene.add(key);

  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 90);
  const camPos = vec(VIEWS[0][0]);
  const camLook = vec(VIEWS[0][1]);
  camera.position.copy(camPos);

  const root = new THREE.Group();
  scene.add(root);
  const rand = mulberry32(44);
  const posOf = (id) => (id === 'graph' ? vec(GRAPH_CENTER) : vec(NODES.find((n) => n.id === id).pos));

  /* ------------------------------------------------------------- nœuds */
  const nodeMeshes = new Map();
  const labels = [];
  const sphereGeo = new THREE.IcosahedronGeometry(1, 4);
  const geometries = {
    ring: () => new THREE.TorusGeometry(1, 0.14, 24, 96),
    stack: () => new THREE.BoxGeometry(0.1, 0.1, 0.1), // pivot invisible : les feuillets sont ajoutés à part
  };
  NODES.forEach((node) => {
    const geometry = node.kind ? geometries[node.kind]() : sphereGeo;
    const material = new THREE.MeshStandardMaterial({
      color: node.color,
      emissive: node.color,
      emissiveIntensity: 0.35,
      metalness: 0.35,
      roughness: 0.28,
      transparent: node.kind === 'stack',
      opacity: node.kind === 'stack' ? 0 : 1,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(vec(node.pos));
    mesh.scale.setScalar(node.size);
    if (node.kind === 'ring') mesh.rotation.y = Math.PI / 2;
    if (node.kind === 'stack') mesh.scale.set(14, 22, 10); // zone de survol couvrant la pile
    mesh.userData = { node, baseScale: node.size, emissive: 0.35, hover: 0, flash: 0 };
    root.add(mesh);
    nodeMeshes.set(node.id, mesh);

    const { texture, aspect } = makeLabelTexture(node.label);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false }));
    const h = isSmall ? 0.42 : node.size < 0.3 ? 0.3 : 0.36;
    sprite.scale.set(h * aspect, h, 1);
    let above = node.size + 0.45;
    if (node.id === 'critic') above = -(node.size + 0.42);
    if (node.id === 'scg') above = 1.75;
    if (node.id === 'memory') above = 1.45;
    sprite.position.copy(vec(node.pos)).add(new THREE.Vector3(0, above, 0));
    sprite.renderOrder = 10;
    sprite.userData = { node };
    root.add(sprite);
    labels.push(sprite);
  });

  // Disque de la porte adaptative (s'illumine quand l'effort est réduit).
  const gateDisc = new THREE.Mesh(
    new THREE.CircleGeometry(0.55, 48),
    new THREE.MeshBasicMaterial({ color: C.gate, transparent: true, opacity: 0.06, side: THREE.DoubleSide, depthWrite: false }),
  );
  gateDisc.position.copy(posOf('gate'));
  gateDisc.rotation.y = Math.PI / 2;
  root.add(gateDisc);

  /* ------------------------------------- graphe de connaissances (50 nœuds) */
  const GRAPH_N = 50;
  const graphNodes = [];
  const graphGeo = new THREE.IcosahedronGeometry(0.075, 2);
  // Matériau non éclairé : la couleur par instance (allumé / éteint) reste lisible sous tous les angles.
  const graphMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const graphMesh = new THREE.InstancedMesh(graphGeo, graphMat, GRAPH_N);
  const m4 = new THREE.Matrix4();
  for (let i = 0; i < GRAPH_N; i += 1) {
    // Répartition de Fibonacci dans une boule : nuage régulier, rendu identique à chaque visite.
    const y = 1 - (2 * (i + 0.5)) / GRAPH_N;
    const r = Math.sqrt(1 - y * y);
    const th = i * 2.399963;
    const radius = 1.1 + rand() * 0.55;
    const p = vec(GRAPH_CENTER).add(new THREE.Vector3(Math.cos(th) * r * 0.75, y, Math.sin(th) * r).multiplyScalar(radius));
    graphNodes.push({ p, activation: rand(), level: 1 });
    m4.makeTranslation(p.x, p.y, p.z);
    graphMesh.setMatrixAt(i, m4);
    graphMesh.setColorAt(i, new THREE.Color(C.graphOn));
  }
  root.add(graphMesh);
  // Arêtes entre voisins proches.
  const edgePts = [];
  graphNodes.forEach((a, i) => {
    graphNodes.slice(i + 1).forEach((b) => {
      if (a.p.distanceTo(b.p) < 0.78) edgePts.push(a.p, b.p);
    });
  });
  const graphEdgeMat = new THREE.LineBasicMaterial({ color: C.graph, transparent: true, opacity: 0.18, depthWrite: false });
  root.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(edgePts), graphEdgeMat));

  /* ------------------------------------------------------------ liaisons */
  const edgeMaterial = new THREE.MeshStandardMaterial({ color: C.edge, emissive: 0x13232b, emissiveIntensity: 0.6, roughness: 0.6 });
  const tube = (points, radius = 0.025) => {
    const curve = new THREE.CatmullRomCurve3(points);
    root.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 48, radius, 8, false), edgeMaterial));
    return curve;
  };
  tube([posOf('query'), posOf('gate')]);
  tube([posOf('gate'), new THREE.Vector3(-4.6, 0, 0)]);
  tube([new THREE.Vector3(-1.6, 0, 0), posOf('tap')]);
  tube([new THREE.Vector3(4.0, 0, 0), posOf('memory'), new THREE.Vector3(5.9, 0, 0)]);
  AGENT_IDS.forEach((id) => {
    tube([new THREE.Vector3(5.9, 0, 0), posOf(id)], 0.016);
    tube([posOf(id), posOf('decision')], 0.016);
  });

  // Top-k : 4 chemins lumineux du graphe vers le TAP.
  const beamMat = new THREE.LineBasicMaterial({ color: C.tap, transparent: true, opacity: 0.25, depthWrite: false, blending: THREE.AdditiveBlending });
  const topk = [3, 17, 29, 41].map((i) => graphNodes[i].p);
  topk.forEach((p) => {
    const curve = new THREE.CatmullRomCurve3([p, p.clone().lerp(posOf('tap'), 0.5).add(new THREE.Vector3(0, 0, 0.3)), posOf('tap')]);
    root.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(30)), beamMat));
  });

  /* --------------------------------------------- sphère de contraintes SCG */
  const SCG_R = 1.25;
  const scgCenter = posOf('scg');
  const shell = new THREE.Mesh(
    new THREE.IcosahedronGeometry(SCG_R, 3),
    new THREE.MeshStandardMaterial({ color: C.scg, emissive: C.scg, emissiveIntensity: 0.3, wireframe: true, transparent: true, opacity: 0.28, depthWrite: false }),
  );
  shell.position.copy(scgCenter);
  root.add(shell);
  const shellGlass = new THREE.Mesh(
    new THREE.SphereGeometry(SCG_R * 0.98, 48, 32),
    new THREE.MeshPhysicalMaterial({ color: 0x1a1406, metalness: 0, roughness: 0.15, transmission: 0, transparent: true, opacity: 0.18, depthWrite: false }),
  );
  shellGlass.position.copy(scgCenter);
  root.add(shellGlass);
  // Vecteur de contraintes c (normalisé sur la sphère).
  const cDir = new THREE.Vector3(-0.55, 0.62, 0.56).normalize();
  const arrowMat = new THREE.MeshStandardMaterial({ color: C.scg, emissive: C.scg, emissiveIntensity: 0.9 });
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, SCG_R * 1.25, 12), arrowMat);
  const headCone = new THREE.Mesh(new THREE.ConeGeometry(0.1, 0.26, 20), arrowMat);
  const arrow = new THREE.Group();
  shaft.position.y = (SCG_R * 1.25) / 2;
  headCone.position.y = SCG_R * 1.25 + 0.13;
  arrow.add(shaft, headCone);
  arrow.position.copy(scgCenter);
  arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), cDir);
  root.add(arrow);

  /* ------------------------- trajectoires TAP → sphère, élaguées par le SCG */
  const dotTexture = makeDotTexture();
  const admissibleMat = new THREE.LineBasicMaterial({ color: C.tap, transparent: true, opacity: 0.32, depthWrite: false, blending: THREE.AdditiveBlending });
  const prunedMat = new THREE.LineBasicMaterial({ color: C.tap, transparent: true, opacity: 0.2, depthWrite: false, blending: THREE.AdditiveBlending });
  const nLines = isSmall ? 16 : 28;
  const bundle = [];
  const ends = [];
  const endColors = [];
  for (let i = 0; i < nLines; i += 1) {
    const dir = new THREE.Vector3(rand() - 0.5, rand() - 0.5, rand() - 0.5).normalize();
    // Violation angulaire : relu(-t·c) ; l'hémisphère opposé au vecteur c est élagué.
    const isPruned = dir.dot(cDir) < -0.15;
    const end = scgCenter.clone().addScaledVector(dir, SCG_R);
    const start = posOf('tap');
    const mid = start.clone().lerp(end, 0.5).add(new THREE.Vector3(0, (rand() - 0.5) * 0.8, (rand() - 0.5) * 0.8));
    const curve = new THREE.CatmullRomCurve3([start, mid, end]);
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(40)), isPruned ? prunedMat : admissibleMat);
    line.userData = { count: 41, delay: rand() * 0.6, pruned: isPruned };
    root.add(line);
    bundle.push(line);
    ends.push(end.x, end.y, end.z);
    endColors.push(isPruned);
  }
  const endGeo = new THREE.BufferGeometry();
  endGeo.setAttribute('position', new THREE.Float32BufferAttribute(ends, 3));
  endGeo.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(ends.length), 3));
  const endMat = new THREE.PointsMaterial({ size: 0.22, map: dotTexture, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  const endPoints = new THREE.Points(endGeo, endMat);
  root.add(endPoints);

  /* ----------------------------------------- mémoire différentielle : k sur 15 */
  const SLABS = 15;
  const slabGeo = new THREE.BoxGeometry(0.95, 0.07, 0.65);
  const slabs = [];
  for (let i = 0; i < SLABS; i += 1) {
    const slab = new THREE.Mesh(
      slabGeo,
      new THREE.MeshStandardMaterial({ color: C.memory, emissive: C.memory, emissiveIntensity: 0.2, metalness: 0.3, roughness: 0.35, transparent: true, opacity: 0.9 }),
    );
    slab.position.copy(posOf('memory')).add(new THREE.Vector3(0, -1 + (i * 2) / (SLABS - 1), 0));
    slab.rotation.y = 0.5;
    root.add(slab);
    slabs.push(slab);
  }

  /* ------------------------------------------------ trace auditable (blocs) */
  const blockGeo = new THREE.BoxGeometry(0.34, 0.34, 0.34);
  const chain = [];
  const CHAIN = 6;
  for (let i = 0; i < CHAIN; i += 1) {
    const p = new THREE.Vector3(4.2 + i * 0.95, -3.3, 0.5);
    const block = new THREE.Mesh(
      blockGeo,
      new THREE.MeshStandardMaterial({ color: C.trace, emissive: C.trace, emissiveIntensity: 0.15, metalness: 0.5, roughness: 0.3 }),
    );
    block.position.copy(p);
    block.rotation.set(0.4, 0.5, 0);
    root.add(block);
    chain.push(block);
    if (i > 0) tube([chain[i - 1].position, p], 0.012);
  }
  const traceMat = new THREE.LineDashedMaterial({ color: C.trace, dashSize: 0.18, gapSize: 0.14, transparent: true, opacity: 0.2 });
  const traceCurve = new THREE.CatmullRomCurve3([posOf('decision'), new THREE.Vector3(9.2, -2.2, 0.4), chain[CHAIN - 1].position]);
  const traceLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(traceCurve.getPoints(80)), traceMat);
  traceLine.computeLineDistances();
  root.add(traceLine);

  /* ---------------------------------------------- particules (unités de calcul) */
  const routeCurves = AGENT_IDS.map((agent) => new THREE.CatmullRomCurve3(ROUTE.map((id) => posOf(id ?? agent))));
  const COUNT = isSmall ? 110 : 220;
  const particles = Array.from({ length: COUNT }, () => ({ alive: false }));
  const pGeometry = new THREE.BufferGeometry();
  const pPositions = new Float32Array(COUNT * 3);
  const pColors = new Float32Array(COUNT * 3);
  pPositions.fill(9999); // hors champ tant qu'aucune particule n'est émise
  pGeometry.setAttribute('position', new THREE.BufferAttribute(pPositions, 3));
  pGeometry.setAttribute('color', new THREE.BufferAttribute(pColors, 3));
  const pMaterial = new THREE.PointsMaterial({
    size: isSmall ? 0.2 : 0.17, map: dotTexture, vertexColors: true, transparent: true,
    depthWrite: false, blending: THREE.AdditiveBlending,
  });
  const points = new THREE.Points(pGeometry, pMaterial);
  points.frustumCulled = false;
  root.add(points);

  const colFlow = new THREE.Color(C.tap);
  const colOut = new THREE.Color(C.decision);
  const colReject = new THREE.Color(C.reject);
  const colSaved = new THREE.Color(C.saved);
  const colScg = new THREE.Color(C.scg);
  const colGraphOn = new THREE.Color(C.graphOn);
  const colGraphOff = new THREE.Color(0x1e293b);
  const tmp = new THREE.Vector3();
  const tmpColor = new THREE.Color();

  /* --------------------------------------------------------------- état */
  const state = {
    stage: 0,
    running: false,
    disposed: false,
    time: 0,
    stageTime: 0,
    spawnAcc: 0,
    complexity: 0.6, // 0..1, complexité estimée de la requête courante
    k: 7,
    traceFlash: 0,
    pointer: new THREE.Vector2(0, 0),
    parallax: new THREE.Vector2(0, 0),
    hovered: null,
  };

  function spawn(p) {
    p.alive = true;
    p.route = Math.floor(rand() * routeCurves.length);
    p.t = 0;
    p.speed = 0.075 + rand() * 0.05;
    // Requête simple : la porte évite une large part du calcul.
    p.saved = rand() < 0.7 * (1 - state.complexity);
    p.rejected = !p.saved && rand() < (state.stage === 3 ? 0.4 : 0.2);
    p.mode = 'flow';
    p.fade = 1;
    p.jitter = new THREE.Vector3((rand() - 0.5) * 0.25, (rand() - 0.5) * 0.25, (rand() - 0.5) * 0.25);
    p.vel = new THREE.Vector3();
  }

  function updateParticles(dt) {
    state.spawnAcc += (isSmall ? 18 : 34) * dt;
    for (const p of particles) {
      if (state.spawnAcc < 1) break;
      if (!p.alive) {
        spawn(p);
        state.spawnAcc -= 1;
      }
    }
    state.spawnAcc = Math.min(state.spawnAcc, 2);

    let flash = 0;
    particles.forEach((p, i) => {
      const o = i * 3;
      if (!p.alive) {
        pPositions[o] = pPositions[o + 1] = pPositions[o + 2] = 9999;
        return;
      }
      if (p.mode === 'fall' || p.mode === 'rise') {
        if (p.mode === 'fall') p.vel.y -= 4.5 * dt;
        else p.vel.y += 1.2 * dt;
        tmp.set(pPositions[o], pPositions[o + 1], pPositions[o + 2]).addScaledVector(p.vel, dt);
        p.fade -= dt * 0.9;
        tmpColor.copy(p.mode === 'fall' ? colReject : colSaved).multiplyScalar(Math.max(p.fade, 0));
        if (p.fade <= 0) p.alive = false;
      } else {
        p.t += p.speed * dt;
        if (p.saved && p.t >= T_GATE) {
          p.mode = 'rise'; // calcul évité par la porte adaptative
          p.vel.set((rand() - 0.5) * 0.4, 0.4 + rand() * 0.5, (rand() - 0.5) * 0.6);
        } else if (p.rejected && p.t >= T_SCG) {
          p.mode = 'fall'; // trajectoire élaguée par le SCG
          p.vel.set((rand() - 0.2) * 0.8, 0.6 + rand() * 0.6, (rand() - 0.5) * 1.2);
          flash = 1;
        }
        if (p.t >= 1) {
          p.alive = false;
          state.traceFlash = 1; // chaque décision laisse une trace
          return;
        }
        routeCurves[p.route].getPoint(p.t, tmp).add(p.jitter);
        const k = THREE.MathUtils.smoothstep(p.t, T_SCG, 1);
        tmpColor.copy(colFlow).lerp(colOut, k).multiplyScalar(Math.max(p.fade, 0));
      }
      pPositions[o] = tmp.x;
      pPositions[o + 1] = tmp.y;
      pPositions[o + 2] = tmp.z;
      pColors[o] = tmpColor.r;
      pColors[o + 1] = tmpColor.g;
      pColors[o + 2] = tmpColor.b;
    });
    pGeometry.attributes.position.needsUpdate = true;
    pGeometry.attributes.color.needsUpdate = true;
    if (flash) nodeMeshes.get('scg').userData.flash = 1;
  }

  const isFocused = (node, s) => s === 0 || node.stage === s;

  function updateNodes(dt) {
    const s = state.stage;
    const ease = (rate) => Math.min(1, dt * rate);

    // Complexité : alterne requêtes simples et complexes (porte adaptative).
    const cycle = reducedMotion ? 0.35 : 0.5 + 0.5 * Math.sin(state.time * 0.55);
    const targetComplexity = s === 1 ? cycle : 0.6;
    state.complexity += (targetComplexity - state.complexity) * ease(2);

    nodeMeshes.forEach((mesh) => {
      const { node } = mesh.userData;
      if (node.kind === 'stack') return;
      const focused = isFocused(node, s);
      const targetEmissive = (s === 0 ? 0.75 : focused ? 1.3 : 0.12) + mesh.userData.hover * 0.8;
      mesh.userData.emissive += (targetEmissive - mesh.userData.emissive) * ease(4);
      let emissive = mesh.userData.emissive;
      if (node.id === 'scg' && mesh.userData.flash > 0) {
        mesh.userData.flash = Math.max(0, mesh.userData.flash - dt * 3);
        emissive += mesh.userData.flash * 0.9;
      }
      if (node.id === 'decision') emissive += state.traceFlash * 0.4;
      mesh.material.emissiveIntensity = emissive;
      const breathe = 1 + Math.sin(state.time * 1.6 + node.pos[0]) * 0.03;
      let scale = mesh.userData.baseScale * breathe * (1 + mesh.userData.hover * 0.18 + (focused && s !== 0 ? 0.08 : 0));
      if (node.id === 'query') scale *= 0.85 + state.complexity * 0.3;
      if (AGENT_IDS.includes(node.id)) {
        // Poids des agents : se réallouent selon la requête.
        const w = 0.75 + 0.5 * (0.5 + 0.5 * Math.sin(state.time * 0.7 + AGENT_IDS.indexOf(node.id) * 1.7));
        scale *= s === 5 && !reducedMotion ? w : 1;
      }
      mesh.scale.setScalar(scale);
      if (node.kind === 'ring') mesh.rotation.x += dt * (0.25 + (1 - state.complexity) * 0.8);
      mesh.userData.hover += ((state.hovered === mesh ? 1 : 0) - mesh.userData.hover) * ease(10);
    });
    labels.forEach((sprite) => {
      const target = isFocused(sprite.userData.node, s) ? 1 : 0.35;
      sprite.material.opacity += (target - sprite.material.opacity) * ease(5);
    });
    gateDisc.material.opacity = 0.05 + (s === 1 ? (1 - state.complexity) * 0.3 : 0);

    // Graphe : seuls les nœuds dont l'activation dépasse le seuil restent allumés.
    const threshold = 1 - (0.3 + 0.7 * state.complexity);
    const graphFocus = s === 1 || s === 2 ? 1 : s === 0 ? 0.7 : 0.35;
    graphNodes.forEach((n, i) => {
      const on = n.activation >= threshold ? 1 : 0;
      n.level += (on - n.level) * ease(5);
      tmpColor.copy(colGraphOff).lerp(colGraphOn, n.level * graphFocus);
      graphMesh.setColorAt(i, tmpColor);
    });
    graphMesh.instanceColor.needsUpdate = true;
    graphEdgeMat.opacity = 0.08 + graphFocus * 0.14;
    beamMat.opacity = s === 2 ? 0.7 : s === 0 ? 0.3 : 0.12;

    // Faisceau : se dessine à l'étape TAP ; les trajectoires en violation rougissent à l'étape SCG.
    bundle.forEach((line) => {
      const local = s === 2 ? THREE.MathUtils.clamp((state.stageTime - line.userData.delay) / 1.2, 0, 1) : 1;
      line.geometry.setDrawRange(0, Math.max(2, Math.floor(line.userData.count * local)));
    });
    const bundleTarget = s === 2 || s === 3 ? 0.55 : s === 0 ? 0.28 : 0.14;
    admissibleMat.opacity += (bundleTarget - admissibleMat.opacity) * ease(4);
    const prunedTarget = s === 3 ? 0.75 : bundleTarget * 0.7;
    prunedMat.opacity += (prunedTarget - prunedMat.opacity) * ease(4);
    const redness = s === 3 ? Math.min(1, state.stageTime / 1.2) : 0;
    prunedMat.color.setHex(C.tap).lerp(colReject, redness);
    const endCol = endGeo.attributes.color;
    endColors.forEach((pruned, i) => {
      tmpColor.copy(pruned ? colFlow.clone().lerp(colReject, redness) : colScg);
      endCol.setXYZ(i, tmpColor.r, tmpColor.g, tmpColor.b);
    });
    endCol.needsUpdate = true;
    const scgFocus = s === 3 ? 1 : s === 0 ? 0.6 : 0.3;
    const scgFlash = nodeMeshes.get('scg').userData.flash;
    shell.material.opacity = 0.12 + scgFocus * 0.25 + scgFlash * 0.15;
    shell.material.emissiveIntensity = 0.2 + scgFocus * 0.5;
    shell.rotation.y += dt * 0.12;
    arrowMat.emissiveIntensity = 0.3 + scgFocus * 0.9;

    // Mémoire différentielle : k documents retenus, de 2 à 15.
    const kTarget = s === 4 && !reducedMotion ? 2 + 13 * (0.5 + 0.5 * Math.sin(state.stageTime * 0.8 - 1.2)) : 7;
    state.k += (kTarget - state.k) * ease(3);
    const memFocus = s === 4 ? 1 : s === 0 ? 0.6 : 0.25;
    slabs.forEach((slab, i) => {
      const kept = i < Math.round(state.k);
      const target = kept ? 0.25 + memFocus * 1.1 : 0.04;
      slab.material.emissiveIntensity += (target - slab.material.emissiveIntensity) * ease(6);
      slab.material.opacity += ((kept ? 0.95 : 0.25) - slab.material.opacity) * ease(6);
    });

    // Trace : une impulsion parcourt la chaîne de blocs à chaque décision.
    state.traceFlash = Math.max(0, state.traceFlash - dt * 2.5);
    const traceFocus = s === 6 ? 1 : s === 0 ? 0.5 : 0.15;
    const head = (state.time * 1.5) % CHAIN;
    chain.forEach((block, i) => {
      const pulse = Math.max(0, 1 - Math.abs(head - i));
      block.material.emissiveIntensity = 0.12 + traceFocus * (0.35 + pulse * 0.9);
      block.rotation.y += dt * 0.3;
    });
    traceMat.opacity = 0.12 + traceFocus * 0.4;
  }

  function updateCamera(dt, immediate = false) {
    const [p, l] = VIEWS[state.stage];
    const k = immediate ? 1 : 1 - Math.exp(-dt * 2.2);
    state.parallax.lerp(state.pointer, immediate ? 1 : Math.min(1, dt * 3));
    const idle = reducedMotion ? 0 : 1;
    tmp.set(
      p[0] + state.parallax.x * 0.6 + Math.sin(state.time * 0.21) * 0.25 * idle,
      p[1] + state.parallax.y * 0.4 + Math.sin(state.time * 0.17) * 0.18 * idle,
      p[2],
    );
    camPos.lerp(tmp, k);
    camLook.lerp(vec(l), k);
    camera.position.copy(camPos);
    camera.lookAt(camLook);
  }

  /* --------------------------------------------------------- redimension */
  function resize() {
    const w = canvas.clientWidth || window.innerWidth;
    const h = canvas.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // Grand écran : le texte occupe la gauche, la scène est décalée vers la droite.
    // Mobile : les cartes de texte occupent le centre, la scène remonte dans le tiers haut.
    if (w >= 960) camera.setViewOffset(w, h, -w * 0.2, 0, w, h);
    else camera.setViewOffset(w, h, 0, h * 0.24, w, h);
    camera.fov = w < 720 ? 55 : 42;
    camera.updateProjectionMatrix();
    if (!state.running) renderOnce();
  }
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(canvas);

  /* ----------------------------------------------------------- survol */
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const pickables = [...nodeMeshes.values()];
  const BLOCKERS = '.card, a, button, input, select, textarea, label, .site-header, .consent, .hero-copy';

  function pick(event) {
    const rect = canvas.getBoundingClientRect();
    ndc.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    return raycaster.intersectObjects(pickables, false)[0]?.object ?? null;
  }

  function onPointerMove(event) {
    state.pointer.set((event.clientX / window.innerWidth) * 2 - 1, -(event.clientY / window.innerHeight) * 2 + 1);
    if (event.pointerType === 'touch' || !state.running) return;
    const blocked = event.target instanceof Element && event.target.closest(BLOCKERS);
    const hit = blocked ? null : pick(event);
    if (hit !== state.hovered) {
      state.hovered = hit;
      document.documentElement.classList.toggle('graph-hover', Boolean(hit));
    }
    if (tooltip) {
      if (hit) {
        const { node } = hit.userData;
        tooltip.innerHTML = `<strong>${node.label}</strong><span>${node.desc}</span>`;
        tooltip.style.transform = `translate(${event.clientX + 16}px, ${event.clientY + 16}px)`;
        tooltip.hidden = false;
      } else {
        tooltip.hidden = true;
      }
    }
  }

  function onClick(event) {
    if (!state.hovered) return;
    const blocked = event.target instanceof Element && event.target.closest(BLOCKERS);
    if (blocked) return;
    onNodeActivate?.(state.hovered.userData.node);
  }

  window.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('click', onClick);

  function onContextLostEvent(event) {
    event.preventDefault();
    onContextLost?.();
  }
  canvas.addEventListener('webglcontextlost', onContextLostEvent);

  /* ------------------------------------------------------------- boucle */
  let lastTime = performance.now();
  let raf = 0;
  let frames = 0;
  let frameTime = 0;
  let goodStreak = 0;

  function adaptQuality(dt) {
    frames += 1;
    frameTime += dt;
    if (frames < 90) return;
    const fps = frames / frameTime;
    frames = 0;
    frameTime = 0;
    if (fps < 52 && dpr > 0.75) {
      dpr = Math.max(0.75, dpr - 0.25);
      renderer.setPixelRatio(dpr);
      goodStreak = 0;
    } else if (fps > 58.5 && dpr < maxDpr) {
      goodStreak += 1;
      if (goodStreak >= 4) {
        dpr = Math.min(maxDpr, dpr + 0.25);
        renderer.setPixelRatio(dpr);
        goodStreak = 0;
      }
    }
  }

  function frame(now) {
    raf = requestAnimationFrame(frame);
    const dt = Math.min(Math.max((now - lastTime) / 1000, 0), 1 / 20);
    lastTime = now;
    state.time += dt;
    state.stageTime += dt;
    updateParticles(dt);
    updateNodes(dt);
    updateCamera(dt);
    renderer.render(scene, camera);
    adaptQuality(dt);
  }

  function renderOnce() {
    if (state.disposed) return;
    // Mouvement réduit : un rendu statique et complet de l'étape courante.
    state.stageTime = 10;
    for (let i = 0; i < 8; i += 1) updateNodes(1);
    updateCamera(0, true);
    renderer.render(scene, camera);
  }

  function setActive(active) {
    if (state.disposed) return;
    const shouldRun = active && !reducedMotion && document.visibilityState === 'visible';
    if (shouldRun === state.running) return;
    state.running = shouldRun;
    if (shouldRun) {
      lastTime = performance.now();
      raf = requestAnimationFrame(frame);
    } else {
      cancelAnimationFrame(raf);
      if (tooltip) tooltip.hidden = true;
      state.hovered = null;
      document.documentElement.classList.remove('graph-hover');
    }
  }

  function setStage(stage) {
    const next = THREE.MathUtils.clamp(stage | 0, 0, VIEWS.length - 1);
    if (next === state.stage) return;
    state.stage = next;
    state.stageTime = 0;
    if (reducedMotion) renderOnce();
  }

  function dispose() {
    if (state.disposed) return;
    setActive(false);
    state.disposed = true;
    resizeObserver.disconnect();
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('click', onClick);
    canvas.removeEventListener('webglcontextlost', onContextLostEvent);
    const textures = new Set();
    scene.traverse((obj) => {
      obj.geometry?.dispose();
      const materials = Array.isArray(obj.material) ? obj.material : obj.material ? [obj.material] : [];
      materials.forEach((m) => {
        if (m.map) textures.add(m.map);
        m.dispose();
      });
    });
    graphMesh.dispose();
    textures.forEach((t) => t.dispose());
    sphereGeo.dispose();
    envTexture.dispose();
    scene.clear();
    renderer.dispose();
    renderer.forceContextLoss();
  }

  resize();
  renderOnce();
  return { setStage, setActive, dispose, get stage() { return state.stage; } };
}

/** Générateur pseudo-aléatoire déterministe (rendu identique à chaque visite). */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
