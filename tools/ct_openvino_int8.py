"""The CT network as an OpenVINO graph saved to disk, and its int8 copy, for the reading on a processor.

    python tools/ct_openvino_int8.py export [--version 1] [--folds 0]       (main environment)
    python tools/ct_openvino_int8.py calib  [--version 1] [--folds 0] [--scans 16] [--per-scan 4]
    ../_tools/venv-nncf/Scripts/python.exe tools/ct_openvino_int8.py quantize [--version 1] [--folds 0]

`export` writes `work/ov/v<V>/fold<K>.xml` (+ .bin): the network and its softmax (segmenter.Softmaxed)
in fp32, with the sha256 of the fold's weights in the graph's own metadata, so a reader can refuse a
graph whose weights have changed since (segmenter.GraphReader does). `calib` cuts calibration windows,
exactly as a reading cuts them, from scans the fold's model TRAINED on (development scans of the
other four folds), never from its held-out fold, so a later Dice on that fold is not flattered; it
writes `work/ov/v<V>/calib_fold<K>.npy`. `quantize` runs NNCF's post-training quantisation over those
windows and writes `fold<K>.int8.xml`; it needs only openvino, nncf and numpy, so it runs in its own
environment (`../_tools/venv-nncf`) and leaves the main environment's numpy alone.

A reading uses a saved graph with RAIDIOLOGY_CPU_ENGINE=openvino RAIDIOLOGY_OV_GRAPH=fp32|int8.
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "work", "ov")


def folder(version):
    return os.path.join(OUT, "v%d" % version)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def segmenter():
    os.environ.setdefault("RAIDIOLOGY_DEVICE", "cpu")
    os.environ["RAIDIOLOGY_CPU_ENGINE"] = "torch"      # export and calibrate from the PyTorch weights
    sys.path[:0] = [os.path.join(ROOT, "modalities", "ct", "eval"), ROOT]
    import segmenter as s  # noqa: PLC0415  main environment only
    return s


def export(a):
    import openvino as ov  # noqa: PLC0415
    import torch  # noqa: PLC0415

    s = segmenter()
    os.makedirs(folder(a.version), exist_ok=True)
    for k in a.folds:
        models, meta = s.load_models(a.version, [k])
        patch = meta["cfg"]["patch"]
        graph = ov.convert_model(s.Softmaxed(models[0]).eval(), example_input=torch.zeros(1, 1, patch, patch, patch))
        graph.set_rt_info(sha256(os.path.join(s.model_dir(a.version), "fold%d.pt" % k)), ["raidiology", "weights_sha256"])
        graph.set_rt_info(str(patch), ["raidiology", "patch"])
        path = os.path.join(folder(a.version), "fold%d.xml" % k)
        ov.save_model(graph, path, compress_to_fp16=False)
        print("fold %d -> %s" % (k, path))


def calib(a):
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    s = segmenter()
    import prepare  # noqa: PLC0415  importable once segmenter() has set the path

    split = s.load_split()
    skipped = prepare.unprepared("dev")
    with open(os.path.join(s.model_dir(a.version), "meta.json"), encoding="utf-8") as f:
        meta_patch = json.load(f)["cfg"]["patch"]
    os.makedirs(folder(a.version), exist_ok=True)
    for k in a.folds:
        trained_on = sorted(n for n, f in split["dev_folds"].items() if f != k and n not in skipped)
        step = max(1, len(trained_on) // a.scans)
        picked = trained_on[::step][:a.scans]
        wins = []
        for scan in picked:
            x = s.to_input(torch.as_tensor(np.ascontiguousarray(np.load(prepare.lo_files(scan)[0]))))
            pad = [max(meta_patch - v, 0) for v in x.shape]
            if any(pad):
                x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]), value=s.LOW_HU / s.HU_SCALE)
            starts = [(i, j, l) for i in s._starts(x.shape[0], meta_patch) for j in s._starts(x.shape[1], meta_patch)
                      for l in s._starts(x.shape[2], meta_patch)]
            # Spread through the scan, so the windows show body parts from top to bottom.
            for i, j, l in [starts[t] for t in np.linspace(0, len(starts) - 1, min(a.per_scan, len(starts))).astype(int)]:
                wins.append(x[i:i + meta_patch, j:j + meta_patch, l:l + meta_patch][None].numpy())
        path = os.path.join(folder(a.version), "calib_fold%d.npy" % k)
        np.save(path, np.stack(wins).astype(np.float32))
        print("fold %d: %d windows from %d of its %d training scans -> %s" % (k, len(wins), len(picked), len(trained_on), path))


def quantize(a):
    import nncf  # noqa: PLC0415  the venv-nncf environment
    import numpy as np  # noqa: PLC0415
    import openvino as ov  # noqa: PLC0415

    core = ov.Core()
    for k in a.folds:
        graph = core.read_model(os.path.join(folder(a.version), "fold%d.xml" % k))
        windows = np.load(os.path.join(folder(a.version), "calib_fold%d.npy" % k))
        data = nncf.Dataset([windows[i:i + 1] for i in range(len(windows))])
        # The softmax always stays in float; --keep-norm keeps the per-scan normalisation (MVN, what
        # InstanceNorm3d becomes) in float too.
        kept = ["SoftMax", "Softmax"] + (["MVN"] if a.keep_norm else [])
        q = nncf.quantize(graph, data, subset_size=len(windows), preset=nncf.QuantizationPreset.MIXED,
                          fast_bias_correction=not a.full_bias_correction,
                          ignored_scope=nncf.IgnoredScope(types=kept, validate=False))
        for key in ("weights_sha256", "patch"):
            q.set_rt_info(graph.get_rt_info(["raidiology", key]).astype(str), ["raidiology", key])
        q.set_rt_info("int8 nncf %s, %d calibration windows, float: %s" % (nncf.__version__, len(windows), ", ".join(kept)),
                      ["raidiology", "graph"])
        path = os.path.join(folder(a.version), "fold%d.%s.xml" % (k, a.name))
        ov.save_model(q, path, compress_to_fp16=False)
        print("fold %d -> %s" % (k, path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "calib", "quantize"])
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--folds", type=lambda v: [int(x) for x in v.split(",")], default=[0])
    ap.add_argument("--scans", type=int, default=16)
    ap.add_argument("--per-scan", type=int, default=4)
    ap.add_argument("--keep-norm", action="store_true")
    ap.add_argument("--full-bias-correction", action="store_true")
    ap.add_argument("--name", default="int8", help="the graph's name: work/ov/v<V>/fold<K>.<name>.xml")
    a = ap.parse_args()
    {"export": export, "calib": calib, "quantize": quantize}[a.cmd](a)


if __name__ == "__main__":
    main()
