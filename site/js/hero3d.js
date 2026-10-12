// hero3d.js: the home page's live 3D stage. A glass body with the generic reference skeleton inside
// (BodyParts3D, CC BY 4.0, the only anatomy this module names, always tagged "Reference anatomy"),
// turning slowly in a black lightbox. A teal scan plane follows the sample CT: ctdemo.js announces
// every slice on window as `raidiology:slice`.
//
//   mountHero3D(stage, opts?) -> { setSlice({ index, count, level }), dispose(), ready }
//
// This file is small on purpose: three.js, the body and the skeleton load after the first paint and
// only where WebGL works. Until the first frame is drawn the stage shows its poster (data-poster, or
// data-poster-phone on stages under 600 px wide). tools/render_hero_poster.py renders the posters
// with this camera, and the camera fits its frame the way background-size: contain fits the poster,
// so the swap from poster to canvas does not show. The page writes the stage's caption and the skeleton's
// credit (BodyParts3D, CC BY 4.0) itself, so both show before this loads and where WebGL does not.
// opts: { autoTurn: false } holds the turn (the poster renderer uses it); { mode: 'skeleton' };
// { avoid: element } names something laid over the stage (the home page's sample CT card): while it covers
// part of the stage, the body and the poster move sideways to stand in the room beside it.
import { prefersReducedMotion } from './motion.js';

const HERO_GLB = 'data/models/skeleton-hero.glb';
const STAGE_LABEL = 'A 3D glass body with a reference skeleton inside';
const LIGHTBOX = 0x05090e;
const PHONE_MAX = 600;
const TURN_RATE = (Math.PI * 2) / 40; // one turn in about 40 s
const YAW0 = -0.55;
const CAM_DIST = 5600;
// The box each poster shows, in world mm at the target's depth, with the poster's own aspect.
const FRAMES = {
  wide: { aspect: 1280 / 1100, height: 2340, y: 800, elev: 0.2 },
  phone: { aspect: 780 / 880, height: 1640, y: 1020, elev: 0.18 },
};
const MODES = [['glass', 'Glass'], ['skeleton', 'Skeleton'], ['xray', 'X-ray']];
const MODE_LINE = { glass: ['Glass', ': a clear shell around the bones.'], skeleton: ['Skeleton', ': the bones alone.'], xray: ['X-ray', ': a faded body, bones showing through.'] };
const BODY_OPACITY = { glass: 1, skeleton: 0, xray: 0.32 };
const SCAN_COLOR = 0x4fe0d2;
const REST_LEVEL = 'T12';
const BAND = ['L3', 'T8']; // slice 0 at the bottom of the band, the last slice at the top
const LEVEL_RE = /^(C[1-7]|T(?:1[0-2]|[1-9])|L[1-5]|S1)$/;
const ORDINAL = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th', '12th'];

// Light on the bones and the glass where the scan plane cuts them: a thin line and a soft halo.
const SCAN_GLSL = /* glsl */ `
uniform float uScanY;
uniform vec3 uScanColor;
uniform float uScanGain;
varying float vScanY;
vec3 scanGlow(float y) {
  float d = y - uScanY;
  float w = max(1.6, fwidth(y) * 1.2);
  float line = exp(-d * d / (w * w));
  float halo = exp(-d * d / (w * w * 40.0)) * 0.2;
  return uScanColor * (line + halo) * uScanGain;
}
`;

function el(tag, cls, attrs) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
  return e;
}

/** Connected pieces of one merged skeleton part: per vertex its piece, per piece its centre and height. */
function pieces(geo) {
  const pos = geo.attributes.position;
  const n = pos.count;
  const idx = geo.index ? geo.index.array : null;
  const parent = new Int32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  const find = (a) => {
    while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; }
    return a;
  };
  const union = (a, b) => {
    a = find(a);
    b = find(b);
    if (a !== b) parent[b] = a;
  };
  const len = idx ? idx.length : n;
  for (let t = 0; t + 2 < len; t += 3) {
    const a = idx ? idx[t] : t;
    union(a, idx ? idx[t + 1] : t + 1);
    union(a, idx ? idx[t + 2] : t + 2);
  }
  const of = new Int32Array(n);
  const byRoot = new Map();
  const list = [];
  for (let i = 0; i < n; i++) {
    const r = find(i);
    let k = byRoot.get(r);
    if (k === undefined) {
      k = list.length;
      byRoot.set(r, k);
      list.push({ n: 0, cx: 0, cy: 0, cz: 0, minY: Infinity, maxY: -Infinity });
    }
    of[i] = k;
    const c = list[k];
    const y = pos.getY(i);
    c.n++;
    c.cx += pos.getX(i);
    c.cy += y;
    c.cz += pos.getZ(i);
    if (y < c.minY) c.minY = y;
    if (y > c.maxY) c.maxY = y;
  }
  for (const c of list) {
    c.cx /= c.n;
    c.cy /= c.n;
    c.cz /= c.n;
    c.side = c.cx >= 0 ? 'Left' : 'Right'; // +X is the patient's left
  }
  return { of, list };
}

/**
 * Plain-English names for the pieces of one skeleton part. The parts are the reference skeleton's
 * groups (skeleton.json), in its own frame: mm, +Y up, +Z front, +X the patient's left.
 */
function nameBones(part, list) {
  const names = list.map(() => 'Bone');
  const each = (fn) => list.forEach((c, i) => { names[i] = fn(c, i); });
  const sides = (keep = () => true) => {
    const s = { Left: [], Right: [] };
    list.forEach((c, i) => { if (keep(c)) s[c.side].push(i); });
    return Object.entries(s);
  };
  const byTop = (a, b) => list[b].maxY - list[a].maxY;
  const byOut = (a, b) => Math.abs(list[b].cx) - Math.abs(list[a].cx);

  if (/^(C[1-7]|T(?:1[0-2]|[1-9])|L[1-5])$/.test(part)) {
    const extra = { C1: ' (atlas)', C2: ' (axis)' }[part] || '';
    each(() => `${part} vertebra${extra}`);
  } else if (part === 'sacrum') each(() => 'Sacrum');
  else if (part === 'coccyx') each(() => 'Coccyx');
  else if (part === 'pelvis') each((c) => `${c.side} hip bone`);
  else if (part === 'hands') each((c) => `${c.side} hand bones`);
  else if (part === 'feet' || part === 'other') each((c) => (c.maxY < 100 ? `${c.side} foot bones` : 'Bone'));
  else if (part === 'skull') {
    each(() => 'Skull');
    let jaw = -1;
    list.forEach((c, i) => { if (c.n > 300 && (jaw < 0 || c.minY < list[jaw].minY)) jaw = i; });
    if (jaw >= 0) {
      names[jaw] = 'Lower jaw';
      list.forEach((c, i) => { if (c.n < 300 && c.maxY - c.minY < 30 && c.cy < list[jaw].cy) names[i] = 'Hyoid bone'; });
    }
  } else if (part === 'ribcage') {
    const cartilage = (c) => c.n < 300 && c.cz > 120;
    each((c) => (cartilage(c) ? `${c.side} rib cartilage` : `${c.side} rib`));
    for (const [side, ribs] of sides((c) => !cartilage(c) && c.n >= 100)) {
      if (ribs.length === 12) ribs.sort(byTop).forEach((i, k) => { names[i] = `${side} ${ORDINAL[k]} rib`; });
    }
  } else if (part === 'shoulders') {
    each((c) => `${c.side} shoulder blade`);
    for (const [side, own] of sides()) {
      if (own.length > 1) names[own.sort((a, b) => list[b].cz - list[a].cz)[0]] = `${side} collarbone`;
    }
  } else if (part === 'arms') {
    for (const [side, own] of sides()) {
      const [upper, ...fore] = own.sort(byTop);
      if (upper !== undefined) names[upper] = `${side} humerus`;
      fore.sort(byOut).forEach((i, k) => {
        names[i] = fore.length === 2 ? `${side} ${k ? 'ulna' : 'radius'}` : `${side} forearm bone`;
      });
    }
  } else if (part === 'legs') {
    for (const [side, own] of sides()) {
      const [femur, ...rest] = own.sort(byTop);
      if (femur !== undefined) names[femur] = `${side} femur`;
      const lower = [];
      for (const i of rest) {
        if (list[i].maxY - list[i].minY < 80) names[i] = `${side} kneecap`;
        else lower.push(i);
      }
      lower.sort(byOut).forEach((i, k) => {
        names[i] = lower.length === 2 ? `${side} ${k ? 'tibia' : 'fibula'}` : `${side} shin bone`;
      });
    }
  }
  return names;
}

function normLevel(level) {
  const s = String(level || '').trim().toUpperCase().split(/[-\s/]/)[0];
  if (s === 'SACRUM') return 'S1';
  return LEVEL_RE.test(s) ? s : null;
}

export function mountHero3D(stage, opts = {}) {
  const ac = new AbortController();
  const on = (target, type, fn, o) => target.addEventListener(type, fn, { signal: ac.signal, ...o });
  const reduced = prefersReducedMotion();
  const autoTurn = opts.autoTurn !== false && !reduced;
  let mode = MODES.some(([m]) => m === opts.mode) ? opts.mode : 'glass';
  let dead = false;
  let booted = false;
  let core = null;
  // `ready` settles once: true when the 3D has drawn its first frame, false when it never will (no WebGL, a failed load, dispose).
  const readySignal = new EventTarget();
  const ready = new Promise((resolve) => { readySignal.addEventListener('ready', (e) => resolve(e.detail), { once: true }); });
  const settleReady = (ok) => readySignal.dispatchEvent(new CustomEvent('ready', { detail: ok }));
  const before = { style: stage.getAttribute('style') };

  stage.classList.add('hero3d');
  Object.assign(stage.style, { backgroundColor: '#05090e', backgroundPosition: 'center', backgroundRepeat: 'no-repeat', backgroundSize: 'contain' });

  // The picture's name sits on the canvas, not the stage: an img role hides its children from some screen readers,
  // and the stage holds the View buttons.
  const canvas = el('canvas', 'hero3d__canvas', { role: 'img', 'aria-label': STAGE_LABEL });
  canvas.style.touchAction = 'pan-y';
  const label = el('div', 'hero3d__label hero3d__ui', { 'aria-hidden': 'true' });
  label.hidden = true;
  const dot = el('span', 'hero3d__dot');
  const pill = el('span', 'hero3d__pill');
  const labelName = el('span', 'hero3d__name');
  const labelRef = el('span', 'hero3d__ref');
  labelRef.textContent = 'Reference anatomy';
  pill.append(labelName, labelRef);
  label.append(dot, pill);
  const modeBar = el('div', 'hero3d__modes hero3d__ui', { role: 'group', 'aria-label': 'View' });
  const chips = MODES.map(([m, text]) => {
    const b = el('button', 'hero3d__mode', { type: 'button', 'data-mode': m, 'aria-pressed': String(m === mode) });
    b.textContent = text;
    on(b, 'click', () => setMode(m));
    modeBar.append(b);
    return b;
  });
  const viewLine = el('p', 'hero3d__view', { 'aria-live': 'polite' });
  const showView = () => {
    viewLine.replaceChildren(...MODES.map(([m]) => {
      const [name, rest] = MODE_LINE[m]; const row = el('span', 'hero3d__row'); const b = document.createElement('strong'); b.textContent = name;
      if (m === mode) row.dataset.on = '';
      row.append(b, rest); return row;
    }));
  };
  showView();
  const made = [canvas, label, modeBar, viewLine];
  stage.append(canvas, label, modeBar);
  stage.append(viewLine);

  const width = () => stage.clientWidth || 0;
  // The page may say which framing fits its layout (--hero3d-frame: phone or wide); otherwise the stage's width decides.
  const isPhone = () => {
    const f = getComputedStyle(stage).getPropertyValue('--hero3d-frame').trim();
    return f ? f === 'phone' : width() < PHONE_MAX;
  };
  // How far the body moves sideways (px, positive toward the right) to clear opts.avoid while it covers part of the
  // stage: half the width it covers, so the body stands in the middle of the room left beside it.
  function shiftX() {
    const a = opts.avoid;
    if (!a || !a.isConnected) return 0;
    const s = stage.getBoundingClientRect();
    const r = a.getBoundingClientRect();
    if (!r.width || !s.width || r.top >= s.bottom || r.bottom <= s.top || r.left >= s.right || r.right <= s.left) return 0;
    return r.left - s.left <= s.right - r.right ? (r.right - s.left) / 2 : -(s.right - r.left) / 2;
  }
  let shift = 0;
  let posterUrl = '';
  function setPoster() {
    const url = (isPhone() && stage.dataset.posterPhone) || stage.dataset.poster || '';
    if (url === posterUrl) return;
    posterUrl = url;
    stage.style.backgroundImage = url ? `url("${new URL(url, document.baseURI).href}")` : '';
  }
  function layoutClasses() {
    // Narrow View buttons follow the stage's own width (a docked desktop stage can be narrow too); the framing follows isPhone().
    stage.classList.toggle('hero3d--narrow', width() < PHONE_MAX);
    setPoster();
    shift = shiftX();
    stage.style.backgroundPosition = shift ? `calc(50% + ${shift.toFixed(2)}px) 50%` : 'center';
  }
  layoutClasses();

  function goStatic() {
    stage.classList.add('hero3d--static');
    stage.classList.remove('is-live');
    settleReady(false);
  }

  function setMode(m) {
    mode = m;
    showView();
    for (const b of chips) b.setAttribute('aria-pressed', String(b.dataset.mode === m));
    core?.applyMode(false);
  }

  // Scan plane target, kept without three.js so slices that arrive early are not lost.
  const known = new Map(); // slice index -> vertebra level
  let lastSlice = null;
  function setSlice(detail) {
    if (!detail || typeof detail !== 'object') return;
    const index = Number(detail.index);
    const count = Number(detail.count);
    const level = normLevel(detail.level);
    if (Number.isFinite(index) && level) known.set(index, level);
    lastSlice = { index, count, level };
    core?.slice();
  }
  on(window, 'raidiology:slice', (e) => setSlice(e.detail));

  let onScreen = false;
  let io = null;
  const ro = 'ResizeObserver' in window ? new ResizeObserver(() => { layoutClasses(); core?.resize(); }) : null;
  ro?.observe(stage);
  if (ro && opts.avoid) ro.observe(opts.avoid);
  if (!ro) on(window, 'resize', () => { layoutClasses(); core?.resize(); });

  if (typeof WebGLRenderingContext === 'undefined') goStatic();
  else {
    // Heavy work starts after the first paint, and only once the stage is near the screen.
    const start = () => requestAnimationFrame(() => setTimeout(() => { if (onScreen || !io) boot(); }, 0));
    if ('IntersectionObserver' in window) {
      io = new IntersectionObserver((entries) => {
        onScreen = entries[entries.length - 1].isIntersecting;
        if (onScreen && !booted) start();
        core?.kick();
      }, { rootMargin: '120px 0px' });
      io.observe(stage);
    } else {
      onScreen = true;
      start();
    }
    on(document, 'visibilitychange', () => core?.kick());
  }

  async function boot() {
    if (booted || dead) return;
    booted = true;
    let THREE;
    let bodyMod;
    let spineMod;
    let skeletonMod;
    let RoomEnvironment;
    try {
      [THREE, bodyMod, spineMod, skeletonMod, { RoomEnvironment }] = await Promise.all([
        import('three'), import('./body.js'), import('./spine.js'), import('./skeleton.js'),
        import('three/addons/environments/RoomEnvironment.js'),
      ]);
    } catch {
      goStatic();
      return;
    }
    if (dead) return;
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas, antialias: !isPhone(), powerPreference: 'high-performance' });
    } catch {
      goStatic();
      return;
    }
    renderer.setClearColor(LIGHTBOX, 1);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;

    const scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(renderer);
    const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();
    scene.environment = envTex;
    scene.environmentIntensity = 0.25;
    scene.add(new THREE.HemisphereLight(0xdfe9f5, 0x0b0d10, 0.35));
    const key = new THREE.DirectionalLight(0xfff2e2, 2.1);
    key.position.set(1500, 2600, 2400);
    const rim = new THREE.DirectionalLight(0xa9c8ff, 2.4);
    rim.position.set(-1800, 1700, -2300);
    const fill = new THREE.DirectionalLight(0xd4e3ff, 0.35);
    fill.position.set(-2000, 600, 1500);
    scene.add(key, rim, fill);

    const U = {
      scanY: { value: 0 },
      scanColor: { value: new THREE.Color(SCAN_COLOR) },
      scanGlass: { value: 1.0 },
      scanBone: { value: 1.2 },
    };
    // body.js's fresnel glass, plus the scan line and studio reflections (two tall softboxes, a top light).
    const glassScan = (mat, spec = 0) => {
      Object.assign(mat.uniforms, { uScanY: U.scanY, uScanColor: U.scanColor, uScanGain: U.scanGlass, uSpec: { value: spec } });
      mat.onBeforeCompile = (sh) => {
        sh.vertexShader = sh.vertexShader.replace('void main() {',
          'varying float vScanY;\nvoid main() {\n  vScanY = (modelMatrix * vec4(position, 1.0)).y;');
        sh.fragmentShader = sh.fragmentShader.replace('void main() {', `${SCAN_GLSL}\nuniform float uSpec;\nvoid main() {`)
          .replace(/\}\s*$/, `  vec3 gn = normalize(vN);
  vec3 gv = normalize(vV);
  vec3 gr = reflect(-gv, gn);
  float tall = smoothstep(-0.6, 0.0, gr.y) * (1.0 - smoothstep(0.35, 0.6, gr.y));
  float studio = smoothstep(0.9, 0.985, gr.x) * tall
    + 0.55 * smoothstep(0.9, 0.985, -gr.x) * tall
    + 0.45 * smoothstep(0.8, 0.95, gr.y);
  float gs = studio * uSpec;
  gl_FragColor = vec4(gl_FragColor.rgb * gl_FragColor.a + (uColor * gs + scanGlow(vScanY)) * uOpacity, 1.0);
}
`);
      };
      mat.customProgramCacheKey = () => 'hero3d-glass-scan';
      mat.needsUpdate = true;
      return mat;
    };
    const boneScan = (mat) => {
      mat.onBeforeCompile = (sh) => {
        Object.assign(sh.uniforms, { uScanY: U.scanY, uScanColor: U.scanColor, uScanGain: U.scanBone });
        sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying float vScanY;')
          .replace('#include <project_vertex>', '#include <project_vertex>\n  vScanY = (modelMatrix * vec4(transformed, 1.0)).y;');
        sh.fragmentShader = sh.fragmentShader.replace('#include <common>', `#include <common>\n${SCAN_GLSL}`)
          .replace('#include <opaque_fragment>', 'outgoingLight += scanGlow(vScanY);\n#include <opaque_fragment>');
      };
      mat.customProgramCacheKey = () => 'hero3d-bone-scan';
      mat.needsUpdate = true;
      return mat;
    };

    const figure = new THREE.Group();
    scene.add(figure);
    const body = bodyMod.buildBody();
    const glass = body.material.uniforms;
    glass.uColor.value.set(0xe6f0fc);
    glass.uBase.value = 0.004;
    glass.uRim.value = 1.4;
    glass.uPower.value = 3.0;
    glassScan(body.material, 1.1);
    figure.add(body.group);
    // The body is built from overlapping pieces. A depth-only copy drawn just before the glass leaves
    // only the outermost surface to glow, so the pieces read as one smooth shell, not bubbles.
    // It draws after the bones and the scan plane, so neither is hidden by it.
    const shellMat = new THREE.MeshBasicMaterial({ colorWrite: false, transparent: true, depthWrite: true });
    const shell = body.group.clone();
    shell.traverse((o) => {
      if (!o.isMesh) return;
      o.material = shellMat;
      o.renderOrder = 9;
    });
    figure.add(shell);
    const floor = bodyMod.buildFloor();
    floor.material.opacity = 0.55;
    scene.add(floor);

    // The scan plane: a thin teal sheet with a bright rim. It belongs to the scanner, so it does not turn.
    const PW = 330;
    const PD = 250;
    const planeMat = new THREE.ShaderMaterial({
      uniforms: { uHalf: { value: new THREE.Vector2(PW, PD) }, uColor: U.scanColor, uOpacity: { value: 1 } },
      vertexShader: /* glsl */ `
        varying vec2 vP;
        void main() {
          vP = position.xz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform vec2 uHalf;
        uniform vec3 uColor;
        uniform float uOpacity;
        varying vec2 vP;
        float box(vec2 p, vec2 b, float r) {
          vec2 q = abs(p) - b + r;
          return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
        }
        void main() {
          float d = box(vP, uHalf, 36.0);
          float aa = max(fwidth(d), 0.5);
          float inside = 1.0 - smoothstep(-aa, aa, d);
          float rim = 1.0 - smoothstep(aa * 0.6, aa * 2.2 + 1.2, abs(d));
          float glow = exp(-abs(d) / 16.0) * 0.32;
          float fill = inside * (0.06 + 0.12 * exp(d / 60.0));
          gl_FragColor = vec4(uColor * (rim * 1.1 + glow + fill) * uOpacity, 1.0);
        }`,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    const planeGeo = new THREE.PlaneGeometry(PW * 2 + 120, PD * 2 + 120);
    planeGeo.rotateX(-Math.PI / 2);
    const plane = new THREE.Mesh(planeGeo, planeMat);
    plane.position.z = -10;
    plane.renderOrder = 8;
    scene.add(plane);

    // A faint glow behind the figure: seeing it through the body is what makes the glass read as clear.
    // It fades out inside the poster's frame, so the poster and the canvas still match at the edges.
    const backdrop = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.ShaderMaterial({
      uniforms: { uColor: { value: new THREE.Color(0.11, 0.187, 0.275) } },
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uColor;
        varying vec2 vUv;
        void main() {
          float f = length((vUv - 0.5) * 2.0);
          gl_FragColor = vec4(uColor * pow(1.0 - smoothstep(0.0, 1.0, f), 1.7), 1.0);
        }`,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
    }));
    backdrop.renderOrder = -1;
    scene.add(backdrop);

    const camera = new THREE.PerspectiveCamera(20, 1, 100, 20000);

    let skeleton;
    let spine;
    try {
      spine = spineMod.buildSpine();
      skeleton = await skeletonMod.loadSkeleton(spine, [], { url: HERO_GLB });
    } catch {
      renderer.dispose();
      goStatic();
      return;
    }
    if (dead) {
      renderer.dispose();
      return;
    }
    figure.add(skeleton.group);
    for (const p of skeleton.parts) {
      boneScan(p.solid);
      glassScan(p.xray, 0.12);
      p.xray.uniforms.uColor.value.set(0xe4eefb);
      p.xray.uniforms.uBase.value = 0.045;
      p.xray.uniforms.uRim.value = 0.95;
    }
    const partOf = new Map(skeleton.parts.map((p) => [p.mesh, p]));
    const anatomy = new Map();
    const analyse = (p) => {
      if (!anatomy.has(p)) {
        const { of, list } = pieces(p.mesh.geometry);
        anatomy.set(p, { of, names: nameBones(p.name, list), overlays: new Map() });
      }
      return anatomy.get(p);
    };

    // The bone under the pointer, drawn again on top in a lit-up material.
    const hiSolid = boneScan(new THREE.MeshStandardMaterial({
      color: 0xfff3dc, roughness: 0.45, metalness: 0, emissive: 0x2ab7c6, emissiveIntensity: 0.42,
      polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
    }));
    const hiGlass = glassScan(bodyMod.glassMaterial({ color: 0x8fe6ff, base: 0.1, rim: 1.1, power: 1.7 }), 0.2);
    const hi = new THREE.Mesh(new THREE.BufferGeometry(), hiSolid);
    hi.visible = false;
    hi.raycast = () => {};
    skeleton.group.add(hi);

    const levelY = (name) => spine.levels.get(name)?.center.y ?? null;
    const restY = levelY(REST_LEVEL) ?? 1180;
    let planeY = restY;
    let planeTarget = restY;
    let yaw = YAW0;
    let spin = 0;
    let rampT = 0;
    let resumeAt = 0;
    let started = false;
    let resetAnim = null;
    let bodyOpacity = BODY_OPACITY[mode];
    let sel = null; // { local, name, pinned, part, piece }
    let ptr = null;
    let lastTap = null;
    let raf = 0;
    let last = 0;
    let lastDraw = 0;
    let lost = false;
    let size = { w: 0, h: 0 };
    let frame = FRAMES.wide;
    let tray = false; // the View buttons sit under the canvas, not over its bottom edge

    function resize() {
      // The canvas can stop short of the stage's bottom: a page may keep a tray there for the View buttons
      // (--hero3d-tray in hero3d.css), and the poster fits the same box.
      const w = canvas.clientWidth || width();
      const h = canvas.clientHeight || 0;
      tray = stage.clientHeight - h > 1;
      if (!w || !h) return;
      frame = isPhone() ? FRAMES.phone : FRAMES.wide;
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, w < 500 ? 1.5 : 2));
      renderer.setSize(w, h, false);
      size = { w, h };
      // Fit the frame's box like background-size: contain fits the poster.
      const visible = frame.height * Math.max(1, (frame.aspect * h) / w);
      camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(visible / 2 / CAM_DIST));
      camera.aspect = w / h;
      camera.position.set(0, frame.y + Math.sin(frame.elev) * CAM_DIST, Math.cos(frame.elev) * CAM_DIST);
      camera.lookAt(0, frame.y, 0);
      // The same sideways move as the poster (layoutClasses), so the swap still does not show.
      if (shift) camera.setViewOffset(w, h, -shift, 0, w, h);
      else camera.clearViewOffset();
      camera.updateProjectionMatrix();
      backdrop.position.set(0, frame.y + 40, -650);
      backdrop.scale.set(frame.height * 0.95, frame.height, 1);
      backdrop.lookAt(camera.position);
      kick(true);
    }

    function sliceTarget() {
      const s = lastSlice;
      if (!s) return restY;
      const groups = new Map();
      for (const [i, lv] of known) {
        if (levelY(lv) === null) continue;
        const g = groups.get(lv) || { sum: 0, n: 0 };
        g.sum += i;
        g.n++;
        groups.set(lv, g);
      }
      const anchors = [...groups].map(([lv, g]) => [g.sum / g.n, levelY(lv)]).sort((a, b) => a[0] - b[0]);
      const ok = Number.isFinite(s.index);
      let y;
      if (!ok || !anchors.length) {
        if (s.level && levelY(s.level) !== null) return levelY(s.level);
        if (!ok) return planeTarget;
        const f = Number.isFinite(s.count) && s.count > 1 ? THREE.MathUtils.clamp(s.index / (s.count - 1), 0, 1) : 0.5;
        return THREE.MathUtils.lerp(levelY(BAND[0]), levelY(BAND[1]), f);
      }
      if (anchors.length === 1) {
        const per = Number.isFinite(s.count) && s.count > 1 ? (levelY(BAND[1]) - levelY(BAND[0])) / (s.count - 1) : 0;
        y = anchors[0][1] + (s.index - anchors[0][0]) * per;
      } else {
        let k = 0;
        while (k < anchors.length - 2 && s.index > anchors[k + 1][0]) k++;
        const [i0, y0] = anchors[k];
        const [i1, y1] = anchors[k + 1];
        y = y0 + ((s.index - i0) / (i1 - i0 || 1)) * (y1 - y0);
      }
      // Never further than the named vertebra's own height.
      if (s.level && levelY(s.level) !== null) {
        const order = [...spine.levels.values()].filter((e) => e.kind === 'vertebra' || e.kind === 'sacrum')
          .map((e) => e.center.y).sort((a, b) => a - b);
        const c = levelY(s.level);
        const i = order.indexOf(c);
        const lo = i > 0 ? (order[i - 1] + c) / 2 : c - 15;
        const hi2 = i >= 0 && i < order.length - 1 ? (order[i + 1] + c) / 2 : c + 15;
        y = THREE.MathUtils.clamp(y, lo, hi2);
      }
      return THREE.MathUtils.clamp(y, levelY('S1') ?? 0, levelY('C1') ?? 2000);
    }

    function applyMode(instant) {
      skeleton.setStyle(mode === 'xray' ? 'xray' : 'solid');
      hi.material = mode === 'xray' ? hiGlass : hiSolid;
      hi.renderOrder = mode === 'xray' ? 6 : 1;
      if (instant || reduced) bodyOpacity = BODY_OPACITY[mode];
      kick(true);
    }

    function showLabel() {
      if (!sel) {
        label.hidden = true;
        hi.visible = false;
        return;
      }
      labelName.textContent = sel.name;
      label.hidden = false;
      placeLabel();
    }

    const tmp = new THREE.Vector3();
    function placeLabel() {
      if (!sel || label.hidden) return;
      tmp.copy(sel.local).applyMatrix4(figure.matrixWorld).project(camera);
      const x = ((tmp.x + 1) / 2) * size.w;
      const y = THREE.MathUtils.clamp(((1 - tmp.y) / 2) * size.h, 24, size.h - (tray ? 24 : 84));
      const left = x + pill.offsetWidth + 30 > size.w;
      label.classList.toggle('is-left', left);
      label.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) translate(${left ? 'calc(-100% + 5px)' : '-5px'}, -50%)`;
    }

    function select(hit, pinned) {
      if (!hit) {
        sel = null;
        showLabel();
        kick(true);
        return;
      }
      const p = partOf.get(hit.object);
      const a = analyse(p);
      const piece = a.of[hit.face.a];
      if (!a.overlays.has(piece)) {
        const src = p.mesh.geometry;
        const idx = src.index.array;
        const out = [];
        for (let t = 0; t < idx.length; t += 3) if (a.of[idx[t]] === piece) out.push(idx[t], idx[t + 1], idx[t + 2]);
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', src.attributes.position);
        g.setAttribute('normal', src.attributes.normal);
        g.setIndex(out);
        a.overlays.set(piece, g);
      }
      hi.geometry = a.overlays.get(piece);
      hi.visible = true;
      sel = { local: figure.worldToLocal(hit.point.clone()), name: a.names[piece], pinned };
      showLabel();
      kick(true);
    }

    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    function pick(clientX, clientY) {
      const r = canvas.getBoundingClientRect();
      if (!r.width || !r.height) return null;
      ndc.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
      raycaster.setFromCamera(ndc, camera);
      figure.updateMatrixWorld(true);
      const meshes = skeleton.parts.filter((p) => p.mesh.visible).map((p) => p.mesh);
      return raycaster.intersectObjects(meshes, false)[0] || null;
    }

    // The loop stops when nothing moves, so a later resume needs its own timer to wake it.
    let resumeTimer = 0;
    const pause = () => {
      rampT = 0;
      resumeAt = Infinity;
      clearTimeout(resumeTimer);
    };
    const resumeSoon = (ms = 3500) => {
      if (sel?.pinned || ptr) return;
      resumeAt = performance.now() + ms;
      clearTimeout(resumeTimer);
      resumeTimer = setTimeout(() => kick(true), ms + 16);
    };

    function reset() {
      spin = 0;
      sel = null;
      showLabel();
      const goal = YAW0 + Math.round((yaw - YAW0) / (Math.PI * 2)) * Math.PI * 2;
      if (reduced) yaw = goal;
      else resetAnim = { from: yaw, to: goal, t0: performance.now(), dur: 900 };
      resumeAt = performance.now() + (reduced ? 0 : 900);
      kick(true);
    }

    // Sideways drags turn the body; vertical swipes stay with the page (touch-action: pan-y).
    on(canvas, 'pointerdown', (e) => {
      if (e.button > 0 || lost) return;
      ptr = { id: e.pointerId, x: e.clientX, y: e.clientY, lx: e.clientX, lt: performance.now(), drag: false, vel: 0 };
      spin = 0;
      resetAnim = null;
    });
    on(canvas, 'pointermove', (e) => {
      if (ptr && e.pointerId === ptr.id) {
        const dx = e.clientX - ptr.x;
        if (!ptr.drag) {
          if (Math.abs(dx) < 6 || Math.abs(dx) < Math.abs(e.clientY - ptr.y)) return;
          ptr.drag = true;
          try { canvas.setPointerCapture(e.pointerId); } catch { /* the pointer is gone */ }
          stage.classList.add('is-dragging');
          pause();
        }
        const now = performance.now();
        const d = (e.clientX - ptr.lx) * ((1.25 * Math.PI) / Math.max(320, size.w));
        yaw += d;
        ptr.vel = ptr.vel * 0.6 + (d / Math.max(0.008, (now - ptr.lt) / 1000)) * 0.4;
        ptr.lx = e.clientX;
        ptr.lt = now;
        kick(true);
        return;
      }
      if (e.pointerType === 'mouse' && !e.buttons) hover(e.clientX, e.clientY);
    });
    const endPointer = (e, cancelled) => {
      if (!ptr || e.pointerId !== ptr.id) return;
      const p = ptr;
      ptr = null;
      stage.classList.remove('is-dragging');
      if (p.drag) {
        if (!reduced && !cancelled && performance.now() - p.lt < 80) spin = THREE.MathUtils.clamp(p.vel, -6, 6);
        resumeSoon();
        kick(true);
        return;
      }
      if (cancelled) return;
      const now = performance.now();
      if (lastTap && now - lastTap.t < 330 && Math.hypot(e.clientX - lastTap.x, e.clientY - lastTap.y) < 30) {
        lastTap = null;
        reset();
        return;
      }
      lastTap = { t: now, x: e.clientX, y: e.clientY };
      const hit = pick(e.clientX, e.clientY);
      select(hit, true);
      if (hit) pause();
      else resumeSoon(1200);
    };
    on(canvas, 'pointerup', (e) => endPointer(e, false));
    on(canvas, 'pointercancel', (e) => endPointer(e, true));
    on(canvas, 'pointerleave', (e) => {
      if (e.pointerType !== 'mouse' || ptr) return;
      stage.classList.remove('is-pointing');
      if (sel && !sel.pinned) {
        select(null);
        resumeSoon(1200);
      }
    });

    let hoverAt = null;
    let hoverRaf = 0;
    function hover(x, y) {
      hoverAt = { x, y };
      if (hoverRaf) return;
      hoverRaf = requestAnimationFrame(() => {
        hoverRaf = 0;
        if (!hoverAt || ptr || dead) return;
        const hit = pick(hoverAt.x, hoverAt.y);
        stage.classList.toggle('is-pointing', !!hit);
        if (sel?.pinned) return;
        if (hit) {
          select(hit, false);
          pause();
        } else if (sel) {
          select(null);
          resumeSoon(1200);
        }
      });
    }

    function step(dt, now) {
      let moving = false;
      if (resetAnim) {
        const k = Math.min(1, (now - resetAnim.t0) / resetAnim.dur);
        const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
        yaw = THREE.MathUtils.lerp(resetAnim.from, resetAnim.to, e);
        if (k >= 1) resetAnim = null;
        moving = true;
      } else if (!ptr?.drag) {
        if (Math.abs(spin) > 0.02) {
          yaw += spin * dt;
          spin *= Math.exp(-3.2 * dt);
          moving = true;
        } else spin = 0;
        if (autoTurn && started && now >= resumeAt) {
          rampT = Math.min(1, rampT + dt / 1.4);
          yaw += TURN_RATE * rampT * rampT * (3 - 2 * rampT) * dt;
          moving = true;
        }
      }
      figure.rotation.y = yaw;

      const dy = planeTarget - planeY;
      if (Math.abs(dy) > 0.05) {
        planeY += dy * (1 - Math.exp(-dt * 6.5));
        moving = true;
      } else planeY = planeTarget;
      plane.position.y = planeY;
      U.scanY.value = planeY;

      const goal = BODY_OPACITY[mode];
      const db = goal - bodyOpacity;
      if (Math.abs(db) > 0.004) {
        bodyOpacity += db * (1 - Math.exp(-dt * 7));
        moving = true;
      } else bodyOpacity = goal;
      body.material.uniforms.uOpacity.value = bodyOpacity;
      body.group.visible = bodyOpacity > 0.004;
      shell.visible = body.group.visible;
      return moving;
    }

    function tick() {
      raf = 0;
      if (!running()) return;
      // One clock for every timestamp here (the rAF time can be earlier than performance.now()).
      const now = performance.now();
      const dt = Math.min(0.05, Math.max(0, (now - last) / 1000));
      // Small stages turning on their own draw at about 30 frames a second.
      const slow = size.w < 500 && !ptr && !resetAnim && Math.abs(spin) < 0.02;
      if (slow && now - lastDraw < 30) {
        raf = requestAnimationFrame(tick);
        return;
      }
      last = now;
      lastDraw = now;
      const moving = step(dt, now);
      renderer.render(scene, camera);
      placeLabel();
      if (moving) raf = requestAnimationFrame(tick);
    }

    function running() {
      return !dead && !lost && onScreen && !document.hidden && size.w > 0;
    }

    function kick(fresh) {
      if (raf || !running()) return;
      if (fresh || !last) last = performance.now();
      raf = requestAnimationFrame(tick);
    }

    // A lost context shows the poster again and stops; nothing is restored.
    on(canvas, 'webglcontextlost', () => {
      lost = true;
      cancelAnimationFrame(raf);
      raf = 0;
      sel = null;
      label.hidden = true;
      goStatic();
    });

    core = {
      kick,
      resize,
      applyMode,
      slice() {
        if (!started) return;
        planeTarget = sliceTarget();
        if (reduced) planeY = planeTarget;
        kick(true);
      },
      dispose() {
        cancelAnimationFrame(raf);
        cancelAnimationFrame(hoverRaf);
        clearTimeout(resumeTimer);
        scene.traverse((o) => {
          if (o.geometry) o.geometry.dispose();
          const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
          for (const m of mats) { m.map?.dispose(); m.dispose(); }
        });
        for (const a of anatomy.values()) for (const g of a.overlays.values()) g.dispose();
        for (const p of skeleton.parts) { p.solid.dispose(); p.xray.dispose(); }
        hiSolid.dispose();
        hiGlass.dispose();
        envTex.dispose();
        renderer.dispose();
      },
    };

    // First frame: the poster's exact view (turn held, plane at rest), then the canvas fades in over it.
    resize();
    applyMode(true);
    step(0, performance.now());
    try { await renderer.compileAsync(scene, camera); } catch { /* compiled on first draw instead */ }
    if (dead || lost) return;
    renderer.render(scene, camera);
    // Reveal on the next frame. A hidden tab never fires requestAnimationFrame, so a timer reveals it too;
    // whichever comes first runs the reveal and the other is a no-op.
    let revealed = false;
    let revealTimer = 0;
    const reveal = () => {
      if (revealed || dead || lost) return;
      revealed = true;
      clearTimeout(revealTimer);
      stage.classList.add('is-live');
      settleReady(true);
      setTimeout(() => {
        if (dead || lost) return;
        started = true;
        resumeAt = performance.now();
        if (lastSlice) core.slice();
        kick(true);
      }, reduced ? 0 : 750);
    };
    requestAnimationFrame(reveal);
    revealTimer = setTimeout(reveal, 1000);
  }

  return {
    ready,
    setSlice,
    dispose() {
      if (dead) return;
      dead = true;
      ac.abort();
      io?.disconnect();
      ro?.disconnect();
      core?.dispose();
      for (const n of made) n.remove();
      stage.classList.remove('hero3d--static', 'hero3d--narrow', 'is-live', 'is-dragging', 'is-pointing');
      if (before.style === null) stage.removeAttribute('style'); else stage.setAttribute('style', before.style);
      settleReady(false);
    },
  };
}
