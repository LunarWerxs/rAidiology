"""The X-ray detector's five folds as one ONNX graph, so a reading needs ONNX Runtime and not torch.

    python tools/export_detector_onnx.py [--version 1]

Writes ensemble.onnx beside the fold weights. In: `pixels`, pictures scaled to [0, 1] as
(n, 1, 384, 384), and `which`, the collection one-hot (n, 2). Out: `probs`, each fold's probability
(n, 5), and `maps`, each fold's 6 x 6 map of where the score came from (n, 5, 6, 6). The averaging
stays outside the graph, in modalities/xray/eval/onnx_reader.py, done in the same order and precision
as the torch path, so the two agree to the rounding of the score. The sha256 of every fold file goes
into the graph's metadata, and the reader refuses a graph whose weights have changed since.
"""
import argparse
import hashlib
import importlib
import os
import sys

# idle OpenMP threads sleep after 1 ms instead of spinning for 200 (docs/efficiency, 2026-10-03); read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import onnx
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "modalities", "xray", "eval")


def xray_modules():
    """datasets, detector and prepare from the X-ray folder, the detector on the processor: the graph
    is exported from the weights as they are, with no card in the way."""
    sys.path[:0] = [EVAL, ROOT]
    os.environ.setdefault("RAIDIOLOGY_DEVICE", "cpu")
    return tuple(importlib.import_module(m) for m in ("datasets", "detector", "prepare"))


class Ensemble(nn.Module):
    def __init__(self, folds):
        super().__init__()
        self.nets = nn.ModuleList([net for net, _, _ in folds])
        self.register_buffer("mean", torch.tensor([float(m) for _, m, _ in folds]))
        self.register_buffer("sd", torch.tensor([float(s) for _, _, s in folds]))

    def forward(self, pixels, which):
        probs, maps = [], []
        for k, net in enumerate(self.nets):
            feat = net.trunk((pixels - self.mean[k]) / self.sd[k])
            w = net.head.weight[0, :feat.shape[1]]
            maps.append(torch.einsum("c,bchw->bhw", w, feat))
            probs.append(torch.sigmoid(net.logit_from(feat, which)))
        return torch.stack(probs, 1), torch.stack(maps, 1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=1)
    a = ap.parse_args()
    datasets, detector, prepare = xray_modules()
    n_datasets = len(datasets.NAMES)
    _, folds = detector.load_folds(a.version, n_datasets)
    model = Ensemble(folds).eval()
    out_dir = detector.model_dir(a.version)
    out = os.path.join(out_dir, "ensemble.onnx")
    x = torch.rand(2, 1, prepare.SIZE, prepare.SIZE)
    which = torch.nn.functional.one_hot(torch.tensor([0, 1]), n_datasets).float()
    with torch.no_grad():
        torch.onnx.export(model, (x, which), out, input_names=["pixels", "which"], output_names=["probs", "maps"],
                          dynamic_axes={"pixels": {0: "n"}, "which": {0: "n"}, "probs": {0: "n"}, "maps": {0: "n"}},
                          opset_version=18, dynamo=False)
    graph = onnx.load(out)
    for k in range(len(folds)):
        entry = graph.metadata_props.add()
        entry.key, entry.value = "fold%d.pt" % k, sha256(os.path.join(out_dir, "fold%d.pt" % k))
    entry = graph.metadata_props.add()
    entry.key, entry.value = "version", str(a.version)
    onnx.save(graph, out)
    print("wrote %s (%.1f MB)" % (out, os.path.getsize(out) / 2 ** 20))


if __name__ == "__main__":
    main()
