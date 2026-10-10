// scene.js: renderer, camera, controls, bloom, finding markers, labels and camera flights.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { buildBody, buildFloor } from './body.js';
import { buildSpine, loadRegion, slotForRegion, materialFor } from './spine.js';
import { loadSkeleton } from './skeleton.js';
import { sev, normLevel, levelList, levelStatusMap, el, reportFlaggedDiscs } from './data.js';
import { prefersReducedMotion } from './motion.js';

const HOME_TARGET = new THREE.Vector3(0, 930, 0);
const HOME_DIR = new THREE.Vector3(0.42, 0.1, 1).normalize();
// Oblique from the patient's front-left: shows the side-on curve of the spine and its back edge.
const SIDE_DIR = new THREE.Vector3(0.9, 0.16, 0.42).normalize();
const LABEL_DISTANCE = 1100;
// Closer than this to the orbit target, bones and glass between the camera and the target fade out.
const FADE_DISTANCE = 1600;
const FADED = 0.12;
// Marker sprite size as a fraction of the view's smaller side (floor MARKER_MIN_PX CSS px); the disc is
// pulled MARKER_NUDGE (mm) toward the camera so a marker sitting on or just under a surface still reads
// as in front of it.
const MARKER_SIZE = 0.035;
const MARKER_MIN_PX = 22;
const MARKER_NUDGE = 8;

function backgroundTexture() {
  const cv = document.createElement('canvas');
  cv.width = 16;
  cv.height = 512;
  const c = cv.getContext('2d');
  const g = c.createLinearGradient(0, 0, 0, 512);
  g.addColorStop(0, '#181715');
  g.addColorStop(0.55, '#121110');
  g.addColorStop(1, '#0d0d0c');
  c.fillStyle = g;
  c.fillRect(0, 0, 16, 512);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/**
 * Flat marker disc: severity fill, dark outline, thin light inner ring (thicker when selected).
 * Drawn at 128 px for a sprite of ~30 px on screen, so 8 px here is ~2 px there.
 */
const markerTextures = new Map();
function markerTexture(color, selected) {
  const key = `${color}:${selected}`;
  if (markerTextures.has(key)) return markerTextures.get(key);
  const cv = document.createElement('canvas');
  cv.width = cv.height = 128;
  const c = cv.getContext('2d');
  c.beginPath();
  c.arc(64, 64, 58, 0, Math.PI * 2);
  c.fillStyle = color;
  c.fill();
  c.lineWidth = 8;
  c.strokeStyle = '#111';
  c.stroke();
  c.beginPath();
  c.arc(64, 64, selected ? 38 : 41, 0, Math.PI * 2);
  c.lineWidth = selected ? 10 : 4;
  c.strokeStyle = '#f4efe6';
  c.stroke();
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  markerTextures.set(key, tex);
  return tex;
}

// Ring for a disc the radiologist's report flagged: one colour no severity uses (violet), hollow.
const RING_COLOR = '#b58cf0';
let ringTex = null;
function ringTexture() {
  if (ringTex) return ringTex;
  const cv = document.createElement('canvas');
  cv.width = cv.height = 128;
  const c = cv.getContext('2d');
  c.beginPath();
  c.arc(64, 64, 50, 0, Math.PI * 2);
  c.lineWidth = 20;
  c.strokeStyle = '#111';
  c.stroke();
  c.lineWidth = 11;
  c.strokeStyle = RING_COLOR;
  c.stroke();
  ringTex = new THREE.CanvasTexture(cv);
  ringTex.colorSpace = THREE.SRGBColorSpace;
  return ringTex;
}

/** Worst severity per level, from report.levels and from findings at exactly that level. */
function makeStatusFn(report) {
  const map = levelStatusMap(report);
  return (n) => map.get(normLevel(n)) || 'normal';
}

const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export function createScene({ canvas, labelLayer, tooltip, getInset }) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  } catch {
    return null;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  scene.background = backgroundTexture();
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.4;
  pmrem.dispose();

  scene.add(new THREE.HemisphereLight(0xf2eee6, 0x14120f, 0.7));
  const key = new THREE.DirectionalLight(0xfff4e6, 1.15);
  key.position.set(900, 2200, 1600);
  const rim = new THREE.DirectionalLight(0xd9dcd6, 0.45);
  rim.position.set(-1200, 1400, -1600);
  scene.add(key, rim);

  const camera = new THREE.PerspectiveCamera(30, 1, 5, 30000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 120;
  controls.maxDistance = 20000; // phones with a short visible band need ~13 m to fit the whole body
  controls.target.copy(HOME_TARGET);
  controls.autoRotate = !prefersReducedMotion();
  controls.autoRotateSpeed = 0.45;

  const body = buildBody();
  scene.add(body.group, buildFloor());
  const spine = buildSpine();
  scene.add(spine.group);

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  if (Math.min(window.innerWidth, window.innerHeight) >= 420) {
    composer.addPass(new UnrealBloomPass(new THREE.Vector2(512, 512), 0.15, 0.3, 0.92));
  }
  composer.addPass(new OutputPass());

  const labelRenderer = new CSS2DRenderer({ element: labelLayer });

  let view = { w: 0, h: 0, ox: 0, oy: 0 };
  function resize() {
    const w = canvas.clientWidth || window.innerWidth;
    const h = canvas.clientHeight || window.innerHeight;
    const inset = getInset ? getInset() : { right: 0, bottom: 0 };
    const ox = Math.round(Math.min(inset.right || 0, w * 0.6));
    const oy = Math.round(Math.min(inset.bottom || 0, h * 0.7));
    const ot = Math.round(Math.min(inset.top || 0, h * 0.35));
    if (w !== view.w || h !== view.h) {
      renderer.setSize(w, h, false);
      composer.setSize(w, h);
      labelRenderer.setSize(w, h);
    }
    // Shift the projection centre so the model sits in the part of the canvas not covered by the
    // panel (right or bottom) or, on phones, by the header and dock (top).
    const d = oy - ot;
    const H = h + Math.abs(d);
    camera.aspect = (w + ox) / H;
    camera.setViewOffset(w + ox, H, ox, Math.max(d, 0), w, h);
    camera.updateProjectionMatrix();
    view = { w, h, ox, oy, ot, H };
  }

  function fitDistance() {
    // A page loaded in a hidden tab has no size yet; measure now, and never fly into the model.
    if (!view.w || !view.h) resize();
    if (!view.w || !view.h) return 5200;
    const tanH = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const FH = view.H;
    const vh = Math.max(120, view.h - view.oy - view.ot);
    const vw = Math.max(120, view.w - view.ox);
    return Math.max((1950 * FH) / (2 * tanH * vh), (900 * FH) / (2 * tanH * vw));
  }

  let flight = null;
  function flyTo(target, dist, dir) {
    const d = (dir || camera.position.clone().sub(controls.target)).clone().normalize();
    const toPos = target.clone().addScaledVector(d, dist);
    controls.autoRotate = false;
    if (prefersReducedMotion()) {
      flight = null;
      controls.target.copy(target);
      camera.position.copy(toPos);
      return;
    }
    const fromOff = camera.position.clone().sub(controls.target);
    flight = {
      t0: performance.now(), dur: 1150,
      fromT: controls.target.clone(), toT: target.clone(),
      fromDir: fromOff.clone().normalize(), toDir: d, fromLen: fromOff.length(), toLen: dist,
    };
  }
  function stepFlight(now) {
    const k = ease(Math.min(1, (now - flight.t0) / flight.dur));
    controls.target.lerpVectors(flight.fromT, flight.toT, k);
    const dir = flight.fromDir.clone().lerp(flight.toDir, k);
    if (dir.lengthSq() < 1e-6) dir.copy(flight.toDir);
    dir.normalize();
    camera.position.copy(controls.target).addScaledVector(dir, THREE.MathUtils.lerp(flight.fromLen, flight.toLen, k));
    if (k >= 1) flight = null;
  }
  // Zoom buttons: dolly toward or away from the target by a factor, eased (instant under reduced motion).
  let zoomAnim = null;
  function zoomBy(factor) {
    const cur = camera.position.distanceTo(controls.target);
    const from = zoomAnim ? zoomAnim.to : cur;
    const to = THREE.MathUtils.clamp(from / factor, controls.minDistance, controls.maxDistance);
    flight = null;
    controls.autoRotate = false;
    if (prefersReducedMotion()) {
      zoomAnim = null;
      camera.position.sub(controls.target).setLength(to).add(controls.target);
      return;
    }
    zoomAnim = { from: cur, to, t0: performance.now() };
  }
  function stepZoom(now) {
    const k = Math.min(1, (now - zoomAnim.t0) / 240);
    const dir = camera.position.clone().sub(controls.target).normalize();
    camera.position.copy(controls.target).addScaledVector(dir, THREE.MathUtils.lerp(zoomAnim.from, zoomAnim.to, ease(k)));
    if (k >= 1) zoomAnim = null;
  }
  controls.addEventListener('start', () => {
    zoomAnim = null;
    flight = null;
    controls.autoRotate = false;
  });

  const regionState = new Map();
  const markers = [];
  let selectedId = null;
  let hoveredId = null;

  function addMarker(f, parent, localPos, parentScale) {
    const hex = sev(f.severity).color;
    const maps = { normal: markerTexture(hex, false), selected: markerTexture(hex, true) };
    // Markers inside a scaled region group: undo the parent's world scale so sizes are in world mm.
    parent.updateWorldMatrix(true, false);
    const ws = parent.getWorldScale(new THREE.Vector3()).x || parentScale || 1;
    const group = new THREE.Group();
    group.position.copy(localPos);
    const core = new THREE.Mesh(
      new THREE.SphereGeometry(3, 16, 12),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(hex) }),
    );
    core.scale.setScalar(1 / ws);
    // Two copies of the disc: a solid one that bone in front can hide, and a faint one that always
    // draws, so a marker behind bone is still findable but clearly reads as behind.
    const front = new THREE.Sprite(new THREE.SpriteMaterial({
      map: maps.normal, sizeAttenuation: false, transparent: true, depthTest: true, depthWrite: false,
    }));
    const behind = new THREE.Sprite(new THREE.SpriteMaterial({
      map: maps.normal, sizeAttenuation: false, transparent: true, opacity: 0.45, depthTest: false, depthWrite: false,
    }));
    const hit = new THREE.Mesh(new THREE.SphereGeometry(14, 10, 8), new THREE.MeshBasicMaterial({ visible: false }));
    behind.renderOrder = 29;
    core.renderOrder = 30;
    front.renderOrder = 31;
    group.add(behind, core, front, hit);
    parent.add(group);
    const m = { id: f.id, finding: f, group, core, front, behind, hit, maps, parentScale: ws, phase: markers.length * 1.3 };
    hit.userData.marker = m;
    markers.push(m);
  }

  function placeMarker(f, i, all) {
    const st = regionState.get(f.region);
    const p = f.anchor.point;
    if (st?.ok && p.length === 3 && p.every(Number.isFinite)) {
      addMarker(f, st.inner, new THREE.Vector3(...p), st.scale);
      return;
    }
    if (st?.ok && st.meshByName.has(f.anchor.mesh)) {
      const mesh = st.meshByName.get(f.anchor.mesh);
      mesh.geometry.computeBoundingBox();
      addMarker(f, st.inner, mesh.geometry.boundingBox.getCenter(new THREE.Vector3()), st.scale);
      return;
    }
    const pos = spine.anchorFor(levelList(f.level)[0] || '', st?.slot);
    const twins = all.slice(0, i).filter((g) => normLevel(g.level) === normLevel(f.level)).length;
    pos.x += twins * 14 * (twins % 2 ? 1 : -1);
    addMarker(f, scene, pos, 1);
  }

  // Rings on the discs the radiologist's report flagged. Always drawn (no depth test) so bone cannot hide them.
  const rings = [];
  let ringsOn = false;
  let markersOn = true;
  let mode = 'ai';
  let modeReport = null;
  function addRing({ level, text }) {
    let pos = null;
    for (const st of regionState.values()) {
      if (st.ok && st.meshByLevel.has(level)) pos = new THREE.Box3().setFromObject(st.meshByLevel.get(level)).getCenter(new THREE.Vector3());
    }
    pos ||= spine.levels.get(level)?.center.clone();
    pos ||= spine.anchorFor(level);
    const group = new THREE.Group();
    group.position.copy(pos);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: ringTexture(), sizeAttenuation: false, transparent: true, depthTest: false, depthWrite: false,
    }));
    sprite.renderOrder = 32;
    const hit = new THREE.Mesh(new THREE.SphereGeometry(14, 10, 8), new THREE.MeshBasicMaterial({ visible: false }));
    group.add(sprite, hit);
    group.visible = ringsOn;
    scene.add(group);
    const r = { id: `ring:${level}`, level, text, group, sprite, hit };
    hit.userData.ring = r;
    rings.push(r);
  }

  const labels = new Map();
  function buildLabels() {
    for (const e of spine.levels.values()) {
      if (e.kind !== 'vertebra' && e.kind !== 'sacrum') continue;
      const div = el('div', { class: 'level-label' }, e.kind === 'sacrum' ? 'Sacrum' : e.name);
      const obj = new CSS2DObject(div);
      obj.position.copy(e.center).addScaledVector(e.anterior, e.D.d / 2 + 18);
      scene.add(obj);
      labels.set(e.name, div);
    }
  }
  buildLabels();

  function setActiveLabel(level) {
    const parts = levelList(level).flatMap((n) => n.split('-'));
    for (const [name, div] of labels) div.classList.toggle('is-active', parts.includes(name));
  }

  let labelsOn = true;
  let bodyTarget = 0;
  let model = 'skeleton';
  let ghostPref = true;
  let skeleton = null;
  let skeletonLoading = null;
  function ensureSkeleton() {
    if (skeleton) return Promise.resolve(skeleton);
    const shown = [];
    for (const st of regionState.values()) if (st.ok) shown.push(...st.meshByLevel.keys());
    skeletonLoading ||= loadSkeleton(spine, shown).then((s) => {
      skeleton = s;
      scene.add(s.group);
      s.updateBoxes();
      return s;
    }).catch((e) => { skeletonLoading = null; throw e; });
    return skeletonLoading;
  }
  const tmp = new THREE.Vector3();
  const toCam = new THREE.Vector3();
  const sightRay = new THREE.Ray();
  const clock = new THREE.Clock();

  /**
   * Fades every non-spinal skeleton part whose box lies on the line of sight from the camera to the
   * orbit target (pelvis in front of L5-S1, ribs in front of the thoracic spine, ...), when close.
   */
  function fadeOccluders(dist) {
    if (!skeleton) return;
    const near = dist < FADE_DISTANCE;
    if (near) sightRay.set(camera.position, toCam.copy(controls.target).sub(camera.position).normalize());
    for (const p of skeleton.parts) {
      if (!p.occluder) continue;
      let goal = 1;
      if (near && (p.box.containsPoint(camera.position)
        || (sightRay.intersectBox(p.box, tmp) && tmp.distanceTo(camera.position) < dist))) goal = FADED;
      if (p.fade === goal) continue;
      const next = p.fade + (goal - p.fade) * 0.15;
      skeleton.setFade(p, Math.abs(goal - next) < 0.005 ? goal : next);
    }
  }

  function frame() {
    const t = clock.getElapsedTime();
    if (flight) stepFlight(performance.now());
    if (zoomAnim) stepZoom(performance.now());
    controls.update();

    const dist = camera.position.distanceTo(controls.target);
    fadeOccluders(dist);
    // The glass body surrounds the spine, so it always sits in the line of sight: thin it out close up.
    const near = THREE.MathUtils.smoothstep(dist, 500, FADE_DISTANCE);
    const u = body.material.uniforms.uOpacity;
    u.value += (bodyTarget * (0.3 + 0.7 * near) - u.value) * 0.14;
    body.group.visible = u.value > 0.01;

    const still = prefersReducedMotion();
    // Sprite scale (sizeAttenuation off) is in view units: this gives MARKER_SIZE of the view height.
    // Smaller when the whole body is in view (markers would crowd the spine), full size close up.
    const far = THREE.MathUtils.smoothstep(dist, 800, 4000);
    // MARKER_SIZE follows the smaller side of the view (never below MARKER_MIN_PX), so a tall phone
    // gets about the marker size of a desktop window instead of a height-scaled one.
    const markerFrac = Math.max(MARKER_MIN_PX, MARKER_SIZE * Math.min(view.w, view.h)) / (view.h || 1);
    const unit = (2 * markerFrac * (1 - 0.45 * far)) / camera.projectionMatrix.elements[5];
    for (const m of markers) {
      if (!markersOn) break;
      m.group.getWorldPosition(tmp);
      const k = THREE.MathUtils.clamp(tmp.distanceTo(camera.position) / 1100, 0.3, 1.8) / m.parentScale;
      const selected = m.id === selectedId;
      const hovered = m.id === hoveredId;
      m.hit.scale.setScalar(k * (selected ? 1.35 : hovered ? 1.2 : 1));
      const pulse = still ? 1 : 1.05 - 0.05 * Math.cos(t * Math.PI * 2 + m.phase);
      const size = (unit * (selected ? 1.45 : hovered ? 1.2 : 1) * pulse) / m.parentScale;
      const map = selected ? m.maps.selected : m.maps.normal;
      m.group.worldToLocal(toCam.copy(camera.position)).setLength(MARKER_NUDGE / m.parentScale);
      for (const s of [m.front, m.behind]) {
        s.scale.set(size, size, 1);
        s.material.map = map;
        s.position.copy(toCam);
      }
    }

    if (ringsOn) {
      for (const r of rings) {
        r.group.getWorldPosition(tmp);
        const k = THREE.MathUtils.clamp(tmp.distanceTo(camera.position) / 1100, 0.3, 1.8);
        r.hit.scale.setScalar(k * 1.1);
        const size = unit * 1.45 * (r.id === hoveredId ? 1.15 : 1);
        r.sprite.scale.set(size, size, 1);
      }
    }

    labelLayer.classList.toggle('is-visible', labelsOn && camera.position.distanceTo(controls.target) < LABEL_DISTANCE);
    composer.render();
    labelRenderer.render(scene, camera);
  }

  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  function pick(e) {
    if (!markers.length || !markersOn) return null;
    const r = canvas.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObjects(markers.map((m) => m.hit), false);
    return hits.length ? hits[0].object.userData.marker : null;
  }

  function pickRing(e) {
    if (!ringsOn || !rings.length) return null;
    const r = canvas.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObjects(rings.map((x) => x.hit), false);
    return hits.length ? hits[0].object.userData.ring : null;
  }

  function showTooltip(m, x, y) {
    if (m.text != null) {
      tooltip.replaceChildren(el('div', { class: 'tt-title' }, `${m.level}: the radiologist's report says ${m.text}`));
    } else {
      const s = sev(m.finding.severity);
      tooltip.replaceChildren(
        el('div', { class: 'tt-level' },
          el('span', { class: 'sev-chip', style: { '--sev': s.color } }, s.label),
          m.finding.level),
        el('div', { class: 'tt-title' }, m.finding.title),
      );
    }
    tooltip.hidden = false;
    const pad = 14;
    const w = tooltip.offsetWidth;
    const h = tooltip.offsetHeight;
    tooltip.style.left = `${Math.min(x + pad, window.innerWidth - w - 8)}px`;
    tooltip.style.top = `${Math.min(y + pad, window.innerHeight - h - 8)}px`;
  }

  const api = {
    onFindingClick: null,
    resize,
    ready: false,

    async load(report) {
      const statusOf = makeStatusFn(report);
      await Promise.all((report?.regions || []).map(async (region) => {
        const slot = slotForRegion(region);
        let res = null;
        if (slot && region.meshes.length) {
          try { res = await loadRegion(region, spine, slot); } catch { res = null; }
        }
        if (res?.ok) {
          regionState.set(region.id, { ...res, slot });
          if (res.failedLevels.length) spine.light(res.failedLevels, statusOf);
        } else {
          regionState.set(region.id, { ok: false, slot });
          if (slot) spine.light(spine.regionLevels(region, slot), statusOf);
        }
      }));
      // Real cord meshes sit close to the procedural one; fade the procedural cord so it does not read as a double.
      if ([...regionState.values()].some((s) => s.ok)) spine.cord.material.opacity = 0.18;
      const findings = report?.findings || [];
      findings.forEach((f, i, all) => placeMarker(f, i, all));
      reportFlaggedDiscs(report).forEach(addRing);
      modeReport = report;
      api.setMode(mode);
      api.ready = true;
    },

    reset() {
      setActiveLabel(null);
      flyTo(HOME_TARGET, fitDistance(), HOME_DIR);
    },

    intro() {
      const d = fitDistance();
      controls.target.copy(HOME_TARGET);
      camera.position.copy(HOME_TARGET).addScaledVector(HOME_DIR, d * (prefersReducedMotion() ? 1 : 1.9));
      if (!prefersReducedMotion()) {
        const keepSpin = controls.autoRotate;
        flyTo(HOME_TARGET, d, HOME_DIR);
        controls.autoRotate = keepSpin;
      }
    },

    focusSlot(slot) {
      let box = null;
      for (const st of regionState.values()) {
        if (st.ok && st.slot === slot) box = new THREE.Box3().setFromObject(st.outer);
      }
      if (!box || box.isEmpty()) box = spine.slotBox(slot);
      const size = box.getSize(new THREE.Vector3());
      setActiveLabel(null);
      flyTo(box.getCenter(new THREE.Vector3()), Math.max(260, Math.max(size.y, size.x) * 2.3), SIDE_DIR);
    },

    focusLevel(level, regionId) {
      const n = normLevel(level);
      let pos = null;
      const st = regionState.get(regionId);
      if (st?.ok && st.meshByLevel.has(n)) {
        pos = new THREE.Box3().setFromObject(st.meshByLevel.get(n)).getCenter(new THREE.Vector3());
      }
      pos ||= spine.anchorFor(n, st?.slot);
      setActiveLabel(n);
      flyTo(pos, 300, SIDE_DIR);
    },

    focusFinding(id) {
      const m = markers.find((x) => x.id === id);
      if (!m) return;
      selectedId = id;
      setActiveLabel(m.finding.level);
      flyTo(m.group.getWorldPosition(new THREE.Vector3()), 280, SIDE_DIR);
    },

    select(id) { selectedId = id; },
    setBody(v) { bodyTarget = v ? 1 : 0; },
    setGhost(v) {
      ghostPref = v;
      if (model === 'glass' || model === 'spine') spine.setGhost(v);
    },

    /** 'glass' | 'skeleton' | 'xray' | 'spine'. The skeleton downloads the first time it is needed. */
    async setModel(m) {
      model = m;
      const wantSkeleton = m === 'skeleton' || m === 'xray';
      bodyTarget = m === 'glass' ? 1 : m === 'xray' ? 0.4 : 0;
      spine.setGhost(m === 'glass' || m === 'spine' ? ghostPref : false);
      if (skeleton) skeleton.group.visible = wantSkeleton;
      if (!wantSkeleton) return true;
      let s = null;
      try { s = await ensureSkeleton(); } catch { s = null; }
      if (!s) {
        if (model === m) api.setModel('glass');
        return false;
      }
      if (model === m) {
        s.setStyle(m === 'xray' ? 'xray' : 'solid');
        s.group.visible = true;
      }
      return true;
    },
    setLabels(v) { labelsOn = v; },
    /** Dolly the camera in (factor > 1) or out (< 1) from the current target. */
    zoomBy,
    /** Show or hide the rings on the discs the radiologist's report flagged. */
    setRings(v) {
      ringsOn = v;
      for (const r of rings) r.group.visible = v;
      if (!v) tooltip.hidden = true;
    },

    /**
     * 'ai' | 'radiologist' | 'compare'. Repaints the discs (and the bones, which only carry the AI's tint)
     * and shows what belongs to the mode: ai = the AI's colours and finding markers; radiologist = the
     * radiologist's colours only (no markers, no incidental blue); compare = the AI plus the violet rings.
     */
    setMode(m) {
      mode = m;
      const rad = mode === 'radiologist';
      const levels = modeReport?.comparison?.report?.levels;
      const radStatus = (n) => { const v = levels?.get(normLevel(n)); return v && v !== 'incidental' ? v : 'normal'; };
      for (const st of regionState.values()) {
        if (!st.ok) continue;
        for (const mesh of st.inner.children) {
          const { kind, level, status } = mesh.userData;
          if (!kind) continue;
          mesh.material = materialFor(kind, rad ? (kind === 'disc' ? radStatus(level) : 'normal') : status);
        }
      }
      const aiStatus = modeReport ? makeStatusFn(modeReport) : () => 'normal';
      spine.repaint(rad ? (n) => (n.includes('-') ? radStatus(n) : 'normal') : aiStatus);
      markersOn = !rad;
      for (const mk of markers) mk.group.visible = markersOn;
      selectedId = null;
      hoveredId = null;
      api.setRings(mode === 'compare');
      tooltip.hidden = true;
    },
  };

  let down = null;
  canvas.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; });
  canvas.addEventListener('pointerup', (e) => {
    if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 6) {
      const m = pick(e);
      const ring = m ? null : pickRing(e);
      if (m) {
        api.focusFinding(m.id);
        api.onFindingClick?.(m.id);
      } else if (ring) {
        // a tap on a phone has no hover, so show the report's words now
        showTooltip(ring, e.clientX, e.clientY);
        api.focusLevel(ring.level);
      } else if (e.pointerType === 'touch') {
        tooltip.hidden = true;
      }
    }
    down = null;
  });
  canvas.addEventListener('pointermove', (e) => {
    if (e.pointerType === 'touch' || e.buttons) return;
    const m = pick(e) || pickRing(e);
    hoveredId = m?.id ?? null;
    canvas.style.cursor = m ? 'pointer' : '';
    if (m) showTooltip(m, e.clientX, e.clientY);
    else tooltip.hidden = true;
  });
  canvas.addEventListener('pointerleave', () => {
    hoveredId = null;
    tooltip.hidden = true;
  });

  resize();
  renderer.setAnimationLoop(frame);
  return api;
}
