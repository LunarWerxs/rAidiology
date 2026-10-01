// body.js: a procedural translucent "glass" adult male (~178 cm), built from lathes and spheres.
// Units are millimetres, feet on y = 0, +Z = the patient's front, +X = the patient's left.
import * as THREE from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const UP = V(0, 1, 0);

const GLASS_VERT = /* glsl */ `
varying vec3 vN;
varying vec3 vV;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = normalize(-mv.xyz);
  gl_Position = projectionMatrix * mv;
}`;

const GLASS_FRAG = /* glsl */ `
uniform vec3 uColor;
uniform float uBase;
uniform float uRim;
uniform float uPower;
uniform float uOpacity;
varying vec3 vN;
varying vec3 vV;
void main() {
  float facing = clamp(abs(dot(normalize(vN), normalize(vV))), 0.0, 1.0);
  float fres = pow(1.0 - facing, uPower);
  float a = (uBase + fres * uRim) * uOpacity;
  gl_FragColor = vec4(uColor * (0.55 + fres * 1.1), a);
}`;

/** Fresnel-rim glass: nearly clear face-on, glowing at the silhouette. Additive, so draw order does not matter. */
export function glassMaterial({ color = 0xe6e0d4, base = 0.015, rim = 0.75, power = 2.2, opacity = 1 } = {}) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: new THREE.Color(color) },
      uBase: { value: base },
      uRim: { value: rim },
      uPower: { value: power },
      uOpacity: { value: opacity },
    },
    vertexShader: GLASS_VERT,
    fragmentShader: GLASS_FRAG,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
}

/** Welds the lathe seam and poles so the fresnel rim has no visible crease. */
function smooth(geo) {
  geo.deleteAttribute('uv');
  geo.deleteAttribute('normal');
  const merged = mergeVertices(geo, 1e-3);
  merged.computeVertexNormals();
  return merged;
}

/** Lathe through (radius, y) stations with rounded caps; stations are smoothed with a spline. */
function taper(radii, length, segments = 40) {
  const n = radii.length;
  const stations = radii.map((r, i) => new THREE.Vector2(r, (i / (n - 1)) * length));
  const body = new THREE.SplineCurve(stations).getPoints(36);
  const pts = [];
  const r0 = radii[0];
  const r1 = radii[n - 1];
  for (let k = 0; k < 8; k++) {
    const a = -Math.PI / 2 + (k / 8) * (Math.PI / 2);
    pts.push(new THREE.Vector2(Math.max(0.01, r0 * Math.cos(a)), r0 * Math.sin(a)));
  }
  pts.push(...body);
  for (let k = 1; k <= 8; k++) {
    const a = (k / 8) * (Math.PI / 2);
    pts.push(new THREE.Vector2(Math.max(0.01, r1 * Math.cos(a)), length + r1 * Math.sin(a)));
  }
  return smooth(new THREE.LatheGeometry(pts, segments));
}

/** Torso + pelvis: one lathe with a per-height depth ratio and front/back offset. */
function torsoGeometry() {
  // [y, half width, half depth, z centre]
  const P = [
    [800, 1, 1, -10],
    [818, 118, 88, -14],
    [860, 158, 106, -20],
    [915, 174, 116, -26],
    [970, 168, 110, -20],
    [1030, 156, 100, -10],
    [1085, 144, 94, -4],
    [1150, 150, 100, 0],
    [1230, 160, 114, -6],
    [1300, 168, 122, -16],
    [1370, 176, 120, -22],
    [1425, 178, 104, -28],
    [1458, 132, 80, -30],
    [1482, 78, 64, -26],
    [1500, 62, 58, -20],
  ];
  const prof = new THREE.SplineCurve(P.map(([y, w]) => new THREE.Vector2(w, y))).getPoints(90);
  const geo = new THREE.LatheGeometry(prof, 72);
  const pos = geo.attributes.position;
  const at = (y) => {
    if (y <= P[0][0]) return P[0];
    for (let i = 1; i < P.length; i++) {
      if (y <= P[i][0]) {
        const a = P[i - 1];
        const b = P[i];
        const t = (y - a[0]) / (b[0] - a[0]);
        return [y, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t];
      }
    }
    return P[P.length - 1];
  };
  for (let i = 0; i < pos.count; i++) {
    const [, w, d, zc] = at(pos.getY(i));
    pos.setZ(i, zc + pos.getZ(i) * (d / Math.max(w, 1)));
  }
  return smooth(geo);
}

function headGeometry() {
  const geo = new THREE.SphereGeometry(1, 56, 40);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    let x = pos.getX(i);
    const y = pos.getY(i);
    let z = pos.getZ(i);
    if (y < 0) {
      // narrow the jaw and pull the back of the skull in toward the neck
      x *= 1 - 0.32 * y * y;
      z *= z < 0 ? 1 - 0.42 * y * y : 1 - 0.08 * y * y;
    }
    pos.setXYZ(i, x * 78, y * 116, z * 98);
  }
  return smooth(geo);
}

function orient(mesh, from, to) {
  const dir = to.clone().sub(from).normalize();
  mesh.position.copy(from);
  mesh.quaternion.setFromUnitVectors(UP, dir);
  return mesh;
}

export function buildBody() {
  const material = glassMaterial({ color: 0xe6e0d4, base: 0.012, rim: 0.42, power: 2.6 });
  const group = new THREE.Group();
  group.name = 'body';
  const add = (geo) => {
    const m = new THREE.Mesh(geo, material);
    m.renderOrder = 10;
    group.add(m);
    return m;
  };

  add(torsoGeometry());
  add(headGeometry()).position.set(0, 1668, 4);
  const neck = add(taper([62, 56, 54, 57], 175, 48));
  neck.position.set(0, 1420, -12);
  neck.scale.set(1, 1, 1.02);

  for (const s of [1, -1]) {
    const shoulder = V(s * 186, 1424, -28);
    const elbow = shoulder.clone().add(V(s * 0.2, -1, 0.03).normalize().multiplyScalar(300));
    const wrist = elbow.clone().add(V(s * 0.12, -1, 0.12).normalize().multiplyScalar(255));
    const tip = wrist.clone().add(V(s * 0.06, -1, 0.05).normalize().multiplyScalar(170));

    const deltoid = add(smooth(new THREE.SphereGeometry(1, 40, 28)));
    deltoid.scale.set(56, 62, 54);
    deltoid.position.copy(shoulder).add(V(s * 6, -8, 0));

    group.add(orient(new THREE.Mesh(taper([46, 44, 40, 36], 300), material), shoulder, elbow));
    group.add(orient(new THREE.Mesh(taper([37, 36, 31, 26], 255), material), elbow, wrist));
    const handGeo = taper([27, 31, 27, 17], 170, 32);
    handGeo.scale(0.55, 1, 1);
    group.add(orient(new THREE.Mesh(handGeo, material), wrist, tip));

    const hip = V(s * 92, 900, -12);
    const knee = V(s * 86, 505, 4);
    const ankle = V(s * 78, 88, -22);
    group.add(orient(new THREE.Mesh(taper([86, 80, 70, 60, 52], hip.distanceTo(knee)), material), hip, knee));
    group.add(orient(new THREE.Mesh(taper([50, 57, 50, 38, 30], knee.distanceTo(ankle)), material), knee, ankle));
    const footGeo = taper([34, 38, 36, 27], 215, 32);
    footGeo.scale(1, 1, 0.62);
    group.add(orient(new THREE.Mesh(footGeo, material), V(s * 80, 38, -62), V(s * 88, 30, 160)));
  }
  group.traverse((o) => { if (o.isMesh) o.renderOrder = 10; });
  return { group, material };
}

/** Soft neutral floor disc with a few rings, drawn once into a canvas texture. */
export function buildFloor() {
  const size = 512;
  const cv = document.createElement('canvas');
  cv.width = cv.height = size;
  const c = cv.getContext('2d');
  const g = c.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, 'rgba(190,184,172,0.10)');
  g.addColorStop(0.45, 'rgba(150,145,136,0.04)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  c.fillStyle = g;
  c.fillRect(0, 0, size, size);
  c.strokeStyle = 'rgba(210,204,192,0.12)';
  for (const r of [0.18, 0.3, 0.42]) {
    c.lineWidth = r === 0.3 ? 1.6 : 1;
    c.beginPath();
    c.arc(size / 2, size / 2, r * size, 0, Math.PI * 2);
    c.stroke();
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(
    new THREE.CircleGeometry(900, 64),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }),
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 1;
  return mesh;
}
