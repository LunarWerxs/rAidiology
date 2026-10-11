// skeleton.js: the anatomical reference skeleton (BodyParts3D, CC BY 4.0), loaded on demand,
// fitted onto the procedural spine so the patient's real regions sit inside its spine.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { glassMaterial } from './body.js';

const SOLID = new THREE.MeshStandardMaterial({ color: 0xd8ccb2, roughness: 0.64, metalness: 0 });
const XRAY = glassMaterial({ color: 0xf1ece2, base: 0.03, rim: 0.7, power: 2.1 });
const SPINAL = /^(C[1-7]|T([1-9]|1[0-2])|L[1-5]|sacrum)$/;

/**
 * @param spine    the procedural spine (its level centres are the fitting target)
 * @param hidden   level names shown by the patient's own meshes (C1..L5, S1); the reference
 *                 skeleton's copies of those are hidden so the real ones read clearly
 */
export async function loadSkeleton(spine, hidden = [], { url = 'data/models/skeleton.glb' } = {}) {
  const loader = new GLTFLoader();
  loader.setMeshoptDecoder(MeshoptDecoder);
  const [info, gltf] = await Promise.all([
    fetch('data/models/skeleton.json').then((r) => { if (!r.ok) throw new Error('skeleton.json'); return r.json(); }),
    loader.loadAsync(url),
  ]);
  const group = new THREE.Group();
  group.name = 'skeleton';
  gltf.scene.updateMatrixWorld(true);
  const parts = [];
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    // gltfpack keeps the name on the wrapper node, so fall back to the parent's.
    const name = (/^mesh_\d+$/.test(o.name) && o.parent.name) || o.name || o.parent.name;
    const geo = o.geometry.clone();
    // The hero copy stores positions as quantized integers (gltfpack). Bake the node transform in
    // floats, or every negative coordinate wraps around inside the integer array.
    for (const [key, a] of Object.entries(geo.attributes)) {
      if (a.array instanceof Float32Array && !a.isInterleavedBufferAttribute) continue;
      const f = new Float32Array(a.count * a.itemSize);
      for (let i = 0; i < a.count; i++) for (let c = 0; c < a.itemSize; c++) f[i * a.itemSize + c] = a.getComponent(i, c);
      geo.setAttribute(key, new THREE.BufferAttribute(f, a.itemSize));
    }
    geo.applyMatrix4(o.matrixWorld);
    if (!geo.attributes.normal) geo.computeVertexNormals();
    // Materials are cloned per part so each one can fade on its own when it hides the focus.
    const solid = SOLID.clone();
    const xray = XRAY.clone();
    const mesh = new THREE.Mesh(geo, solid);
    mesh.name = name;
    parts.push({ name, mesh, solid, xray, box: new THREE.Box3(), occluder: !SPINAL.test(name), fade: 1 });
    group.add(mesh);
  });

  // Uniform scale + translation (least squares) from the skeleton's vertebra centres to the procedural ones.
  const P = [];
  const Q = [];
  for (const [name, c] of Object.entries(info.levels || {})) {
    const e = spine.levels.get(name === 'sacrum' ? 'S1' : name);
    if (e && e.kind !== 'disc') {
      P.push(new THREE.Vector3(...c));
      Q.push(e.center.clone());
    }
  }
  if (P.length >= 3) {
    const pm = P.reduce((a, p) => a.add(p), new THREE.Vector3()).divideScalar(P.length);
    const qm = Q.reduce((a, q) => a.add(q), new THREE.Vector3()).divideScalar(Q.length);
    let num = 0;
    let den = 0;
    P.forEach((p, i) => {
      const dp = p.clone().sub(pm);
      num += dp.dot(Q[i].clone().sub(qm));
      den += dp.lengthSq();
    });
    const s = den > 0 ? num / den : 1;
    group.scale.setScalar(s);
    group.position.copy(qm).addScaledVector(pm, -s);
  }

  const hide = new Set(hidden.map((n) => (n === 'S1' ? 'sacrum' : n)));
  for (const p of parts) p.mesh.visible = !hide.has(p.name);

  const api = {
    group,
    attribution: info.attribution || '',
    /** Every part: { name, mesh, box (world Box3), occluder (not a vertebra or the sacrum), fade }. */
    parts,

    /** World bounding boxes; call once the fitted group is in the scene. */
    updateBoxes() {
      group.updateWorldMatrix(true, true);
      for (const p of parts) p.box.setFromObject(p.mesh);
    },

    /** Opacity 0..1 for one part, in either style. Below 1 it stops writing depth so what is behind shows. */
    setFade(part, opacity) {
      part.fade = opacity;
      const faded = opacity < 0.995;
      const s = part.solid;
      if (s.transparent !== faded) {
        s.transparent = faded;
        s.depthWrite = !faded;
        s.needsUpdate = true;
      }
      s.opacity = faded ? opacity : 1;
      part.xray.uniforms.uOpacity.value = opacity;
    },

    setStyle(style) {
      const xray = style === 'xray';
      for (const p of parts) {
        p.mesh.material = xray ? p.xray : p.solid;
        p.mesh.renderOrder = xray ? 5 : 0;
      }
    },
  };
  return api;
}
