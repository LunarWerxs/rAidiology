"""Turn TotalSpineSeg labels into smooth per-structure .glb meshes in the app's frame
(+X patient left, +Y superior, +Z anterior, millimetres) and write anchors.json with
centroids and disc posterior-margin points in the same frame.
Usage: python build_meshes.py <labels.nii.gz> <region_id> <site_dir>"""
import json
import os
import sys

import nibabel as nib
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from labels import NAMES, kind, level_of

ISO = 0.7  # mm, target isotropic grid for surface extraction
RAS_TO_APP = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float)


def to_app(ijk, affine):
    ras = (affine[:3, :3] @ ijk.T).T + affine[:3, 3]
    return (RAS_TO_APP @ ras.T).T


def surface(mask, zooms, affine, offset, target):
    f = np.array(zooms) / ISO
    m = ndimage.zoom(mask.astype(np.float32), f, order=1)
    m = ndimage.gaussian_filter(m, sigma=0.9)
    m = np.pad(m, 2)
    if m.max() < 0.5:
        return None
    verts, faces, _, _ = measure.marching_cubes(m, 0.5, step_size=1)
    verts = (verts - 2) / f + offset  # back to original voxel index space
    mesh = trimesh.Trimesh(to_app(verts, affine), faces[:, ::-1], process=True)
    trimesh.smoothing.filter_taubin(mesh, iterations=12)
    if len(mesh.faces) > target:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=target)
        except Exception:
            pass
    mesh.fix_normals()
    return mesh


def main(lab_p, region, site):
    n = nib.load(lab_p)
    lab = np.asarray(n.dataobj).astype(np.int32)
    zooms, affine = n.header.get_zooms()[:3], n.affine
    out_dir = os.path.join(site, "data", "meshes", region)
    os.makedirs(out_dir, exist_ok=True)
    anchors, meshes = {}, []
    for v in sorted(int(x) for x in np.unique(lab) if x):
        name = NAMES.get(v)
        if not name:
            continue
        k = kind(name)
        mask = lab == v
        if mask.sum() < 50:
            continue
        sl = ndimage.find_objects(mask.astype(np.int8))[0]
        pad = [slice(max(s.start - 3, 0), min(s.stop + 3, lab.shape[i])) for i, s in enumerate(sl)]
        sub = mask[tuple(pad)]
        offset = np.array([p.start for p in pad], float)
        mesh = surface(sub, zooms, affine, offset, 12000 if k in ("canal", "cord", "sacrum") else 6000)
        if mesh is None:
            continue
        fn = f"{name}.glb"
        mesh.export(os.path.join(out_dir, fn))
        ijk = np.argwhere(mask).astype(float)
        a = {"centroid": [round(float(x), 1) for x in to_app(ijk.mean(0, keepdims=True), affine)[0]],
             "faces": int(len(mesh.faces))}
        if k == "disc":
            pts = to_app(ijk, affine)
            back = pts[pts[:, 2] <= np.percentile(pts[:, 2], 10)]  # most posterior 10% (smallest Z)
            a["posterior"] = [round(float(x), 1) for x in back.mean(0)]
            a["posterior_left"] = [round(float(x), 1) for x in (back.mean(0) + [8, 0, 0])]
            a["posterior_right"] = [round(float(x), 1) for x in (back.mean(0) - [8, 0, 0])]
        anchors[name] = a
        meshes.append({"name": name, "kind": k, "level": level_of(name),
                       "file": f"data/meshes/{region}/{fn}", "status": "normal"})
        print(region, name, len(mesh.faces), a["centroid"])
    json.dump({"meshes": meshes, "anchors": anchors}, open(os.path.join(out_dir, "anchors.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
