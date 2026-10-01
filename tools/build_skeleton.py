"""Build the anatomical reference skeleton for the app from BodyParts3D 4.0
(DBCLS, CC BY 4.0): every "bone organ" element plus costal cartilages, grouped into
one mesh per vertebra (C1..L5, sacrum, coccyx) and one per body area, decimated for phones,
in the app frame (+X patient left, +Y up, +Z front, mm). Writes models/skeleton.glb and
models/skeleton.json (vertebra centres, per-group face counts, attribution).
Usage: python build_skeleton.py <bp3d_dir> <site_dir> [face_budget]"""
import csv
import collections
import glob
import json
import os
import re
import sys

import numpy as np
import trimesh

ORD = {w: i + 1 for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split())}
GROUPS = [  # (group, keywords) checked in order on the element's most specific name
    ("skull", ["skull", "cranial", "frontal bone", "parietal bone", "occipital bone", "temporal bone", "sphenoid",
               "ethmoid", "mandible", "maxilla", "zygomatic", "nasal bone", "lacrimal", "palatine", "vomer",
               "nasal concha", "hyoid", "incus", "malleus", "stapes", "auditory ossicle"]),
    ("ribcage", ["rib", "sternum", "manubrium", "xiphoid", "costal cartilage"]),
    ("shoulders", ["scapula", "clavicle"]),
    ("hands", ["carpal", "scaphoid", "lunate", "triquetr", "pisiform", "trapezium", "trapezoid", "capitate",
               "hamate", "metacarp", "of hand", "finger", "thumb",
               "sesamoid bone of hand"]),
    ("arms", ["humerus", "radius", "ulna"]),
    ("pelvis", ["hip bone", "ilium", "ischium", "pubis"]),
    ("feet", ["talus", "calcaneus", "navicular", "cuboid", "cuneiform", "metatars", "of foot", "toe",
              "sesamoid bone of foot", "tarsal"]),
    ("legs", ["femur", "patella", "tibia", "fibula"]),
]
BUDGET_SHARE = {"vertebra": 0.22, "sacrum": 0.02, "skull": 0.16, "ribcage": 0.14, "shoulders": 0.05, "arms": 0.07,
                "hands": 0.08, "pelvis": 0.07, "legs": 0.09, "feet": 0.08, "other": 0.02}


def vertebra_level(name):
    n = name.lower()
    if n in ("atlas", "first cervical vertebra"):
        return "C1"
    if n in ("axis", "second cervical vertebra"):
        return "C2"
    m = re.match(r"(\w+) (cervical|thoracic|lumbar) vertebra$", n)
    if m and m.group(1) in ORD:
        return {"cervical": "C", "thoracic": "T", "lumbar": "L"}[m.group(2)] + str(ORD[m.group(1)])
    if n == "sacrum":
        return "sacrum"
    if n == "coccyx":
        return "coccyx"
    return None


def group_of(name):
    lv = vertebra_level(name)
    if lv:
        return lv
    n = name.lower()
    for g, kws in GROUPS:
        if any(k in n for k in kws):
            return g
    return "other"


def read_obj(path):
    v, f = [], []
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(p.split("/")[0]) for p in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                f.append([idx[0] - 1, idx[k] - 1, idx[k + 1] - 1])
    return np.array(v, float), np.array(f, int)


def main(bp3d, site, budget=160000):
    budget = int(budget)
    rows = list(csv.reader(open(os.path.join(bp3d, "isa_element_parts.txt"), encoding="utf-8"), delimiter="\t"))[1:]
    elems, names = collections.defaultdict(set), {}
    for c, n, e in rows:
        elems[c].add(e)
        names[c] = n
    wanted = set(elems["FMA5018"])  # bone organ
    for c, n in names.items():
        if "costal cartilage" in n:
            wanted |= elems[c]
    # most specific concept naming each element
    best = {}
    for c, es in elems.items():
        for e in es & wanted:
            if e not in best or len(es) < len(elems[best[e]]):
                best[e] = c
    files = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(bp3d, "**", "*.obj"), recursive=True)}
    parts = collections.defaultdict(list)
    missing = 0
    for e in sorted(wanted):
        if e not in files:
            missing += 1
            continue
        v, f = read_obj(files[e])
        if not len(f):
            continue
        app = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)  # BP3D (x left, y back, z up) -> app (x left, y up, z front)
        parts[group_of(names[best[e]])].append(trimesh.Trimesh(app, f, process=False))
    scene = trimesh.Scene()
    info = {"levels": {}, "groups": {}, "missing_files": missing,
            "attribution": "BodyParts3D, (c) The Database Center for Life Science, licensed under CC Attribution 4.0 International (https://creativecommons.org/licenses/by/4.0/); modified (parts merged and simplified) by LunarWerx"}
    lo = min(m.vertices[:, 1].min() for ms in parts.values() for m in ms)
    hi = max(m.vertices[:, 1].max() for ms in parts.values() for m in ms)
    info["height_mm"] = round(float(hi - lo), 1)
    info["floor_y"] = round(float(lo), 1)
    n_vert = sum(1 for g in parts if re.match(r"^[CTL]\d+$", g))
    for g, ms in sorted(parts.items()):
        mesh = trimesh.util.concatenate(ms)
        mesh.merge_vertices()
        kind = "vertebra" if re.match(r"^[CTL]\d+$", g) else g if g in BUDGET_SHARE else "other"
        share = BUDGET_SHARE[kind] / (n_vert if kind == "vertebra" else 1)
        target = max(400, int(budget * share))
        if len(mesh.faces) > target:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=target)
            except Exception as ex:
                print("simplify failed", g, ex)
        mesh.fix_normals()
        scene.add_geometry(mesh, node_name=g, geom_name=g)
        info["groups"][g] = int(len(mesh.faces))
        if kind == "vertebra" or g in ("sacrum", "coccyx"):
            info["levels"][g] = [round(float(x), 1) for x in mesh.bounding_box.centroid]
    out = os.path.join(site, "data", "models")
    os.makedirs(out, exist_ok=True)
    scene.export(os.path.join(out, "skeleton.glb"))
    json.dump(info, open(os.path.join(out, "skeleton.json"), "w"), indent=1)
    print("groups", len(info["groups"]), "faces", sum(info["groups"].values()), "missing", missing,
          "height", info["height_mm"], "levels", sorted(info["levels"]))
    print({k: v for k, v in info["groups"].items() if not re.match(r"^[CTL]\d+$", k)})


if __name__ == "__main__":
    main(*sys.argv[1:4])
