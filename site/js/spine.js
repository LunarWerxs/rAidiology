// spine.js: procedural spine (C1 to coccyx) inside the glass body, plus loading the real
// per-region .glb meshes and fitting them into their anatomical slot.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { glassMaterial } from './body.js';
import { sev, normLevel, LEVEL_ORDER } from './data.js';

const UP = new THREE.Vector3(0, 1, 0);
const V = (x, y, z) => new THREE.Vector3(x, y, z);
const range = (p, a, b) => Array.from({ length: b - a + 1 }, (_, i) => `${p}${a + i}`);

export const SLOTS = {
  neck: [...range('C', 1, 7), 'T1'],
  lower_back: ['T12', ...range('L', 1, 5), 'S1'],
};

export function slotForRegion(region) {
  if (SLOTS[region.anchor]) return region.anchor;
  if (region.id === 'cervical') return 'neck';
  if (region.id === 'lumbar') return 'lower_back';
  return null;
}

/** Vertebrae plus the discs between consecutive ones: [L4, L5] -> [L4, L5, L4-L5]. C1-C2 has no disc. */
export function withDiscs(vertebrae) {
  const idx = (v) => LEVEL_ORDER.indexOf(v);
  const vs = [...new Set(vertebrae)].filter((v) => idx(v) >= 0).sort((a, b) => idx(a) - idx(b));
  const out = [...vs];
  for (let i = 1; i < vs.length; i++) {
    const a = vs[i - 1];
    const b = vs[i];
    if (idx(b) - idx(a) === 1 && a !== 'C1' && b !== 'CO') out.push(`${a}-${b}`);
  }
  return out;
}

/** Natural vertebra proportions in mm (body width/depth/height, canal radius, arch thickness, processes). */
function dims(name) {
  const r = name[0];
  const n = Number(name.slice(1));
  if (r === 'C') {
    const D = { w: 17 + n * 1.3, d: 15 + n * 0.4, h: 13, canalR: 8.5, t: 3.2, sp: 12 + n * 2.6, tilt: 0.35, tp: 9 };
    if (n === 1) Object.assign(D, { h: 10, atlas: true, sp: 6, tp: 12 });
    if (n === 2) Object.assign(D, { h: 17, dens: true, sp: 22 });
    if (n === 7) D.sp = 36;
    return D;
  }
  if (r === 'T') {
    return {
      w: 26 + n * 1.3, d: 17 + n * 1.3, h: 16 + n * 0.7, canalR: 7.5, t: 4.6,
      sp: n < 9 ? 34 + n : 42 - (n - 8) * 3,
      tilt: n <= 3 ? 0.55 : n <= 9 ? 1.0 : 0.45,
      tp: 20 - Math.max(0, n - 9) * 3,
    };
  }
  return { w: 42 + n * 1.6, d: 31 + n * 0.9, h: 26 + n * 0.3, canalR: 9, t: 6.5, sp: 28, tilt: 0.08, tp: n === 3 ? 28 : 22 };
}

function discHeight(upper) {
  const n = Number(upper.slice(1));
  if (upper[0] === 'C') return 5;
  if (upper[0] === 'T') return 4 + n * 0.3;
  return 9.5 + n * 0.35;
}

function rod(r0, r1, len, from, dir, sx = 1, sz = 1) {
  const g = new THREE.CylinderGeometry(r1, r0, len, 10, 1);
  g.scale(sx, 1, sz);
  g.translate(0, len / 2, 0);
  g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(UP, dir.clone().normalize()));
  g.translate(from.x, from.y, from.z);
  return g;
}

function roundedDrum(w, d, h, waist, segments = 28) {
  const e = Math.min(2, h * 0.12);
  const prof = [[0.01, -h / 2], [0.9, -h / 2], [1, -h / 2 + e], [waist, 0], [1, h / 2 - e], [0.9, h / 2], [0.01, h / 2]]
    .map(([x, y]) => new THREE.Vector2(x, y));
  const g = new THREE.LatheGeometry(prof, segments);
  g.scale(w / 2, 1, d / 2);
  return g;
}

/** Vertebral body + posterior arch + spinous and transverse processes, merged into one geometry. */
function vertebraGeometry(D) {
  const { w, d, h, canalR, t } = D;
  const parts = [];
  const canalZ = D.atlas ? -10 : -(d / 2 + canalR * 0.75);
  const R = canalR + t * 0.5;
  if (D.atlas) {
    const ring = new THREE.TorusGeometry(15, 3.6, 10, 36);
    ring.rotateX(Math.PI / 2);
    ring.scale(1.05, 1.5, 1);
    ring.translate(0, 0, -8);
    parts.push(ring);
    for (const s of [-1, 1]) {
      const mass = new THREE.SphereGeometry(7, 14, 10);
      mass.scale(1, 1.1, 1.2);
      mass.translate(s * 15, 0, -5);
      parts.push(mass);
    }
  } else {
    parts.push(roundedDrum(w, d, h, 0.9));
    const arch = new THREE.TorusGeometry(R, t * 0.55, 8, 22, Math.PI * 1.3);
    arch.rotateZ(-Math.PI * 0.15);
    arch.rotateX(-Math.PI / 2);
    arch.scale(1, Math.max(1, (h * 0.55) / (t * 1.1)), 1);
    arch.translate(0, 0, canalZ);
    parts.push(arch);
  }
  const back = V(0, 0, D.atlas ? -23 : canalZ - R);
  parts.push(rod(t * 0.7, t * 0.35, D.sp, back, V(0, -Math.sin(D.tilt), -Math.cos(D.tilt)), 0.6, 1.8));
  const tpX = D.atlas ? 19 : R * 0.85;
  for (const s of [-1, 1]) {
    parts.push(rod(t * 0.5, t * 0.3, D.tp, V(s * tpX, 0, canalZ + R * 0.25), V(s * Math.cos(0.35), 0.05, -Math.sin(0.35))));
  }
  if (D.dens) parts.push(rod(5, 3.8, 14, V(0, h / 2 - 1, 0), UP));
  const geo = mergeGeometries(parts.map((g) => (g.index ? g.toNonIndexed() : g)));
  return { geo, canalZ };
}

function discGeometry(D) {
  const prof = [[0.01, -D.h / 2], [0.88, -D.h / 2], [1, 0], [0.88, D.h / 2], [0.01, D.h / 2]]
    .map(([x, y]) => new THREE.Vector2(x, y));
  const g = new THREE.LatheGeometry(prof, 28);
  g.scale(D.w * 0.47, 1, D.d * 0.47);
  return g;
}

function sacrumGeometry(h) {
  const top = 50;
  const bot = 12;
  const s = new THREE.Shape();
  s.moveTo(-top, h / 2);
  s.quadraticCurveTo(0, h / 2 + 8, top, h / 2);
  s.quadraticCurveTo(top * 0.85, 0, bot, -h / 2);
  s.quadraticCurveTo(0, -h / 2 - 6, -bot, -h / 2);
  s.quadraticCurveTo(-top * 0.85, 0, -top, h / 2);
  const g = new THREE.ExtrudeGeometry(s, { depth: 24, bevelEnabled: true, bevelThickness: 5, bevelSize: 5, bevelSegments: 3, curveSegments: 16 });
  g.translate(0, 0, -12);
  return g;
}

const BASE = { bone: 0xd9cfb9, disc: 0x9fb0bf, cord: 0xd6b25e, csf: 0x9cc4b8 };
const KIND_GROUP = { vertebra: 'bone', sacrum: 'bone', coccyx: 'bone', disc: 'disc', cord: 'cord', canal: 'csf', csf: 'csf' };
const materialCache = new Map();

/** Solid material by mesh kind, tinted by status with the contract's severity colours. */
export function materialFor(kind, status = 'normal') {
  const g = KIND_GROUP[kind] || 'bone';
  const key = `${g}:${status}`;
  if (materialCache.has(key)) return materialCache.get(key);
  const color = new THREE.Color(BASE[g]);
  const tinted = !!status && status !== 'normal';
  const tint = new THREE.Color(sev(status).color);
  if (tinted) color.lerp(tint, g === 'bone' ? 0.55 : 0.65);
  const m = new THREE.MeshPhysicalMaterial({
    color,
    roughness: g === 'bone' ? 0.68 : 0.35,
    metalness: 0,
    clearcoat: g === 'bone' ? 0.12 : 0.45,
    clearcoatRoughness: 0.35,
    emissive: tinted ? tint : color,
    emissiveIntensity: tinted ? 0.35 : g === 'cord' ? 0.18 : 0.0,
  });
  if (g === 'csf') Object.assign(m, { transparent: true, opacity: 0.18, depthWrite: false, emissiveIntensity: 0.3 });
  if (g === 'disc') Object.assign(m, { transparent: true, opacity: 0.92 });
  materialCache.set(key, m);
  return m;
}

// Body-centre line in the mid-sagittal plane, coccyx tip to C1 as (y, z): sacral curve, lumbar lordosis,
// thoracic kyphosis, cervical lordosis. Segments are laid along it by arc length.
const CENTRE_LINE = [
  [895, -62], [925, -92], [960, -98], [1000, -72], [1030, -45], [1065, -28], [1100, -22], [1140, -26],
  [1175, -34], [1210, -42], [1265, -52], [1340, -60], [1410, -55], [1455, -45], [1485, -36], [1525, -24],
  [1560, -22], [1600, -32],
];

export function buildSpine() {
  const group = new THREE.Group();
  group.name = 'spine';
  const ghost = {
    bone: glassMaterial({ color: 0xf5e9cf, base: 0.07, rim: 0.9, power: 1.7 }),
    disc: glassMaterial({ color: 0x9fb0bf, base: 0.1, rim: 0.8, power: 1.6 }),
  };

  const seq = [{ name: 'CO', kind: 'coccyx', h: 26 }, { gap: 3 }, { name: 'S1', kind: 'sacrum', h: 100 }];
  const upward = [...range('L', 1, 5).reverse(), ...range('T', 1, 12).reverse(), ...range('C', 1, 7).reverse()];
  let below = 'S1';
  for (const v of upward) {
    const D = dims(v);
    if (v === 'C1') seq.push({ gap: 3 });
    else {
      const Db = dims(below === 'S1' ? 'L5' : below);
      seq.push({ name: `${v}-${below}`, kind: 'disc', h: discHeight(v), D: { w: (D.w + Db.w) / 2, d: (D.d + Db.d) / 2 } });
    }
    seq.push({ name: v, kind: 'vertebra', h: D.h, D });
    below = v;
  }

  const curve = new THREE.CatmullRomCurve3(CENTRE_LINE.map(([y, z]) => V(0, y, z)), false, 'centripetal');
  const Lc = curve.getLength();
  const k = Lc / seq.reduce((a, s) => a + (s.h ?? s.gap), 0);

  const levels = new Map();
  let acc = 0;
  for (const s of seq) {
    const len = (s.h ?? s.gap) * k;
    if (s.name) {
      const u = (acc + len / 2) / Lc;
      const p = curve.getPointAt(u);
      const tan = curve.getTangentAt(u);
      const a = Math.atan2(tan.z, tan.y);
      let geo;
      let canalZ = 0;
      let D;
      if (s.kind === 'vertebra') {
        D = { ...s.D, h: s.D.h * k };
        ({ geo, canalZ } = vertebraGeometry(D));
      } else if (s.kind === 'disc') {
        D = { ...s.D, h: len };
        geo = discGeometry(D);
      } else if (s.kind === 'sacrum') {
        D = { w: 104, d: 34, h: len };
        geo = sacrumGeometry(len);
        canalZ = -20;
      } else {
        D = { w: 14, d: 14, h: len };
        geo = new THREE.CylinderGeometry(7, 3, len, 12);
      }
      const mesh = new THREE.Mesh(geo, ghost[s.kind === 'disc' ? 'disc' : 'bone']);
      mesh.name = s.name;
      mesh.position.copy(p);
      mesh.rotation.x = a;
      mesh.renderOrder = 5;
      group.add(mesh);
      levels.set(s.name, {
        name: s.name, kind: s.kind, mesh, D, canalZ,
        center: p.clone(),
        anterior: V(0, -Math.sin(a), Math.cos(a)),
        lit: false, replaced: false,
      });
    }
    acc += len;
  }
  group.updateMatrixWorld(true);

  const canalPoint = (e) => e.center.clone().addScaledVector(e.anterior, e.canalZ);
  const byOrder = [...levels.values()]
    .filter((e) => e.kind === 'vertebra' || e.kind === 'sacrum')
    .sort((a, b) => LEVEL_ORDER.indexOf(a.name) - LEVEL_ORDER.indexOf(b.name));
  const brainstem = canalPoint(levels.get('C1')).add(V(0, 30, 6));
  const canalPts = [brainstem, ...byOrder.map(canalPoint)];
  const cordPts = [brainstem, ...byOrder.filter((e) => LEVEL_ORDER.indexOf(e.name) <= LEVEL_ORDER.indexOf('L1')).map(canalPoint)];
  cordPts.push(canalPoint(levels.get('L1')).lerp(canalPoint(levels.get('L2')), 0.55));

  const cord = new THREE.Mesh(
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3(cordPts), 180, 4.2, 12),
    new THREE.MeshStandardMaterial({ color: 0xd6b25e, emissive: 0xd6b25e, emissiveIntensity: 0.2, roughness: 0.4, transparent: true, opacity: 0.55, depthWrite: false }),
  );
  cord.renderOrder = 4;
  const canal = new THREE.Mesh(
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3(canalPts), 220, 7.5, 16),
    glassMaterial({ color: 0x9cc4b8, base: 0.03, rim: 0.45, power: 2 }),
  );
  canal.renderOrder = 6;
  group.add(cord, canal);

  let ghostOn = true;
  const refresh = (e) => { e.mesh.visible = !e.replaced && (ghostOn || e.lit); };

  return {
    group,
    levels,
    cord,
    slotNames: (slot) => withDiscs(SLOTS[slot] || []),

    slotBox(slot) {
      const box = new THREE.Box3();
      for (const n of withDiscs(SLOTS[slot] || [])) {
        const e = levels.get(n);
        if (e) box.expandByObject(e.mesh);
      }
      return box;
    },

    /** Levels a region covers: its levels_present (plus discs between), else the whole slot. */
    regionLevels(region, slot) {
      const present = region.levels_present.map(normLevel).filter((n) => levels.has(n) && levels.get(n).kind !== 'disc');
      return withDiscs(present.length ? present : SLOTS[slot] || []).filter((n) => levels.has(n));
    },

    /** Mesh fallback: turn ghost segments into solid, status-tinted ones. */
    light(names, statusOf) {
      for (const n of names) {
        const e = levels.get(n);
        if (!e || e.replaced) continue;
        e.mesh.material = materialFor(e.kind, statusOf(n));
        e.mesh.renderOrder = 0;
        e.lit = true;
        refresh(e);
      }
    },

    replace(name) {
      const e = levels.get(name);
      if (!e) return;
      e.replaced = true;
      refresh(e);
    },

    setGhost(show) {
      ghostOn = show;
      levels.forEach(refresh);
      cord.visible = show;
      canal.visible = show;
    },

    /** World position for a finding with no usable mesh anchor. Disc findings sit at the back of the disc. */
    anchorFor(level, slot) {
      const n = normLevel(level);
      const e = levels.get(n);
      if (e) return e.kind === 'disc' ? e.center.clone().addScaledVector(e.anterior, -e.D.d * 0.42) : e.center.clone();
      const [a, b] = n.split('-');
      if (a && b && levels.has(a) && levels.has(b)) return levels.get(a).center.clone().lerp(levels.get(b).center, 0.5);
      if (slot) return this.slotBox(slot).getCenter(new THREE.Vector3());
      return V(0, 1250, -40);
    },
  };
}

let gltfLoader = null;

async function loadGeometry(url) {
  gltfLoader ||= new GLTFLoader();
  const gltf = await gltfLoader.loadAsync(url);
  gltf.scene.updateMatrixWorld(true);
  let geo = null;
  gltf.scene.traverse((o) => {
    if (!geo && o.isMesh && o.geometry) {
      geo = o.geometry.clone();
      geo.applyMatrix4(o.matrixWorld);
    }
  });
  if (!geo) throw new Error(`no mesh in ${url}`);
  if (!geo.attributes.normal) geo.computeVertexNormals();
  geo.computeBoundingBox();
  geo.computeBoundingSphere();
  return geo;
}

const FIT_KINDS = new Set(['vertebra', 'disc', 'sacrum']);

/**
 * Loads a region's meshes into a group and fits it (uniform scale, orientation kept) onto the
 * procedural segments of the same levels, falling back to the whole anatomical slot.
 * Returns { ok:false } when nothing loaded so the caller can light up the procedural levels instead.
 */
export async function loadRegion(region, spine, slot) {
  const results = await Promise.allSettled(region.meshes.map((m) => loadGeometry(m.file)));
  const outer = new THREE.Group();
  const inner = new THREE.Group();
  outer.name = `region-${region.id}`;
  outer.add(inner);
  const meshByName = new Map();
  const meshByLevel = new Map();
  const failed = new Set();
  const covered = new Set();

  results.forEach((r, i) => {
    const def = region.meshes[i];
    const level = normLevel(def.level || def.name);
    if (r.status !== 'fulfilled') {
      if (level) failed.add(level);
      return;
    }
    const mesh = new THREE.Mesh(r.value, materialFor(def.kind, def.status));
    mesh.name = def.name;
    mesh.userData = { level, kind: def.kind, status: def.status };
    if (def.kind === 'canal' || def.kind === 'csf') mesh.renderOrder = 3;
    inner.add(mesh);
    if (def.name) meshByName.set(def.name, mesh);
    if (FIT_KINDS.has(def.kind) && level) {
      covered.add(level);
      if (!meshByLevel.has(level)) meshByLevel.set(level, mesh);
    }
  });
  if (!inner.children.length) return { ok: false };

  outer.updateMatrixWorld(true);
  const slotNames = new Set(spine.slotNames(slot));
  const matched = inner.children.filter((m) => FIT_KINDS.has(m.userData.kind) && slotNames.has(m.userData.level) && spine.levels.has(m.userData.level));
  const realBox = new THREE.Box3();
  const procBox = new THREE.Box3();
  if (matched.length) {
    for (const m of matched) {
      realBox.expandByObject(m);
      procBox.expandByObject(spine.levels.get(m.userData.level).mesh);
    }
  } else {
    realBox.setFromObject(inner);
    procBox.copy(spine.slotBox(slot));
  }
  const realH = realBox.getSize(new THREE.Vector3()).y;
  const procH = procBox.getSize(new THREE.Vector3()).y;
  const scale = THREE.MathUtils.clamp(realH > 1e-3 ? procH / realH : 1, 0.4, 2.5);
  outer.scale.setScalar(scale);
  outer.position.copy(procBox.getCenter(new THREE.Vector3())).addScaledVector(realBox.getCenter(new THREE.Vector3()), -scale);
  spine.group.add(outer);
  outer.updateMatrixWorld(true);

  for (const n of covered) spine.replace(n);
  const failedLevels = [...failed].filter((n) => !covered.has(n) && spine.levels.has(n));
  return { ok: true, outer, inner, scale, meshByName, meshByLevel, failedLevels };
}
