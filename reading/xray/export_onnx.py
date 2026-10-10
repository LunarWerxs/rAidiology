"""Turn one X-ray fracture detector folder (fold0.pt to fold4.pt, meta.json) into the files that are published:
fold<N>.safetensors (the tensors, no pickle) and fold<N>.onnx (one graph per fold).

    python reading/xray/export_onnx.py ../Models/xray-fracture-detector-v140

The .pt files are our own pickles and are read here, nowhere else. A graph takes `pixels`, pictures scaled to
[0, 1] as (n, 1, 384, 384), and `which`, the collection one-hot (n, 2) in the order of meta.json's `datasets`;
it returns `prob`, the fold's probability (n). The fold's pixel mean and spread are inside the graph.
"""
import argparse
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("RAIDIOLOGY_DEVICE", "cpu")
import torch
import torch.nn as nn
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "modalities", "xray", "eval"), ROOT]
import detector  # noqa: E402
import prepare  # noqa: E402


class FoldGraph(nn.Module):
    """One fold as the graph runs it: scale by the fold's mean and spread, the network, the sigmoid."""

    def __init__(self, net, mean, sd):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("sd", torch.tensor(float(sd)))

    def forward(self, pixels, which):
        return torch.sigmoid(self.net((pixels - self.mean) / self.sd, which))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="a folder holding fold0.pt to fold4.pt")
    a = ap.parse_args()
    n_datasets = 2
    for k in range(detector.FOLDS):
        pt = os.path.join(a.folder, "fold%d.pt" % k)
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        c = ck["cfg"]
        net = detector.Net(c["widths"], c["dropout"], c["stem_stride"], n_datasets, loc="loc.weight" in ck["state"])
        net.load_state_dict(ck["state"])
        net.eval()
        save_file({n: t.contiguous() for n, t in ck["state"].items()}, os.path.join(a.folder, "fold%d.safetensors" % k))
        graph = FoldGraph(net, ck["info"]["mean"], ck["info"]["sd"]).eval()
        x = torch.rand(2, 1, prepare.SIZE, prepare.SIZE)
        which = nn.functional.one_hot(torch.tensor([0, 1]), n_datasets).float()
        with torch.no_grad():
            torch.onnx.export(graph, (x, which), os.path.join(a.folder, "fold%d.onnx" % k),
                              input_names=["pixels", "which"], output_names=["prob"],
                              dynamic_axes={"pixels": {0: "n"}, "which": {0: "n"}, "prob": {0: "n"}},
                              opset_version=18, dynamo=False)
        print("fold %d: safetensors and onnx written" % k)


if __name__ == "__main__":
    main()
