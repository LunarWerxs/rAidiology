"""The shipped X-ray reading without torch: the five folds as one ONNX graph on ONNX Runtime.

The MCP server reads with this. Importing torch alone took 2 to 3.8 s of a 4.7 to 5.2 s cold reading
(ledger, 2026-10-01), and torch is a 2.5 GB install; ONNX Runtime is about 15 MB. The graph comes
from tools/export_detector_onnx.py and holds exactly the weights in the fold files: their sha256 is
in its metadata, and a graph whose weights have changed since is refused rather than read.

Same answers as `detector.score_images` and `detector.score_images_with_maps`, in the same form:
each fold's probability comes out of the graph and the average is taken here, in fold order and in
double precision, exactly as the torch path takes it.
"""
import hashlib
import json
import os

import numpy as np
import onnxruntime as ort

import datasets
import prepare
from core import paths

READ_BATCH = 8
_SESSIONS = {}


def model_dir(version):
    return os.path.join(paths.MODELS, "xray-fracture-detector-v%d" % version)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def session(version):
    """The graph and the detector's meta.json, loaded once per process."""
    if version in _SESSIONS:
        return _SESSIONS[version]
    d = model_dir(version)
    graph = os.path.join(d, "ensemble.onnx")
    if not os.path.exists(graph):
        raise SystemExit("no %s: make it with python tools/export_detector_onnx.py --version %d" % (graph, version))
    s = ort.InferenceSession(graph, providers=["CPUExecutionProvider"])
    made_from = s.get_modelmeta().custom_metadata_map
    for name, digest in made_from.items():
        source = os.path.join(d, name)
        if name.endswith(".pt") and os.path.exists(source) and _sha256(source) != digest:
            raise SystemExit("%s was exported from other weights than %s holds now; export it again" % (graph, name))
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    _SESSIONS[version] = (s, meta)
    return s, meta


def _runs(paths_, pictures):
    """Runs of up to READ_BATCH pictures of one shape, with their paths and how each was read."""
    run = []
    for p, (pix, kind) in zip(paths_, pictures):
        if run and (len(run) == READ_BATCH or run[0][1].shape != pix.shape):
            yield run
            run = []
        run.append((p, pix, kind))
    if run:
        yield run


def _read(paths_, version, dataset, with_maps):
    s, meta = session(version)
    one = np.zeros(len(datasets.NAMES), dtype=np.float32)
    one[datasets.NAMES.index(dataset)] = 1.0
    out = []
    for run in _runs(paths_, prepare.read_many(paths_)):
        pixels = (np.stack([pix for _, pix, _ in run]).astype(np.float32) / np.float32(255.0))[:, None]
        probs, maps = s.run(None, {"pixels": pixels, "which": np.repeat(one[None], len(run), 0)})
        for i, (p, _, kind) in enumerate(run):
            total = 0.0
            for k in range(probs.shape[1]):          # fold order, in double, as the torch path adds
                total += float(probs[i, k])
            score = total / probs.shape[1]
            row = {"path": p, "read_as": kind, "score": round(score, 6),
                   "fracture": bool(score >= meta["thresholds"][dataset]),
                   "threshold": meta["thresholds"][dataset], "dataset_profile": dataset, "version": version}
            if with_maps:
                m = maps[i].astype(np.float32).sum(axis=0) / np.float32(probs.shape[1])
                row["map"] = [float(v) for v in m.flatten().tolist()]
                row["map_size"] = [int(m.shape[1]), int(m.shape[0])]
            out.append(row)
    return out, meta


def score_images(paths_, version, dataset="grazpedwri"):
    return _read(paths_, version, dataset, with_maps=False)


def score_images_with_maps(paths_, version, dataset="grazpedwri"):
    return _read(paths_, version, dataset, with_maps=True)
