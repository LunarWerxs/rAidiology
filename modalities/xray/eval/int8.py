"""X-ray speed trial: each fold as an int8 ONNX graph, calibrated on its own training pictures only.

    python modalities/xray/eval/int8.py [--calibrate 512]

Out-of-fold on development, on the processor: fold k's fp32 graph, fold k's int8 graph, and the
int8 graph reading the picture and its mirror (mirror.py) read fold k's held-out pictures. Each int8
arm against today's plain fp32 reading, AUROC per collection with score.py's patient bootstrap, and
the ensemble's time for a batch of eight on four threads. Calibration uses fold k's training pictures
only, so its held-out scores stay honest.

Measured 2026-10-02 (Ryzen 9 7950X): the ensemble 0.0598 s a picture in fp32, 0.0165 in int8, 3.63x.
int8 alone lost AUROC on GRAZPEDWRI-DX, -0.0007 (95% CI -0.0010 to -0.0004), and -0.0008 on
FracAtlas (-0.0027 to +0.0011): dropped. int8 with the mirror gained on both, +0.0020 (+0.0012 to
+0.0028) and +0.0053 (+0.0011 to +0.0098), at 1.8x the speed of today's reading. int8 moves a score
by up to 0.35, so a shipped int8 reading needs its thresholds chosen again from its own scores.
"""
import argparse
import os
import sys
import time

# idle OpenMP threads sleep after 1 ms instead of spinning for 200 (docs/efficiency, 2026-10-03); read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [HERE, ROOT]
os.environ["RAIDIOLOGY_DEVICE"] = "cpu"
import datasets  # noqa: E402  arkitect-allow: test-floor-loosened - after the device and sys.path lines, as in every eval script here
import detector  # noqa: E402  arkitect-allow: test-floor-loosened - after the device and sys.path lines, as in every eval script here
import mirror  # noqa: E402  arkitect-allow: test-floor-loosened - after the device and sys.path lines, as in every eval script here
from core import ledger, paths  # noqa: E402  arkitect-allow: test-floor-loosened - after the device and sys.path lines, as in every eval script here

OUT = os.path.join(paths.WORK, "eval", "xray", "int8")


class One(nn.Module):
    def __init__(self, net, mean, sd):
        super().__init__()
        self.net, self.mean, self.sd = net, float(mean), float(sd)

    def forward(self, pixels, which):
        return torch.sigmoid(self.net((pixels - self.mean) / self.sd, which))


def pictures(df, arrays, rows):
    part = df.iloc[rows]
    pix = np.stack([arrays[d][0][r] for d, r in zip(part.dataset, part.cache_row)]).astype(np.float32) / np.float32(255.0)
    which = np.eye(len(datasets.NAMES), dtype=np.float32)[part.dataset_index.values]
    return pix[:, None], which


class Feed(CalibrationDataReader):
    def __init__(self, df, arrays, rows, batch=16):
        self.batches = [rows[a:a + batch] for a in range(0, len(rows), batch)]
        self.df, self.arrays = df, arrays

    def get_next(self):
        if not self.batches:
            return None
        pix, which = pictures(self.df, self.arrays, self.batches.pop(0))
        return {"pixels": pix, "which": which}


def session(path, threads):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def score(s, df, arrays, rows, batch=64, mirrored=False):
    out = np.zeros(len(rows))
    for a in range(0, len(rows), batch):
        pix, which = pictures(df, arrays, rows[a:a + batch])
        if mirrored:
            pix = np.ascontiguousarray(pix[..., ::-1])
        out[a:a + batch] = s.run(None, {"pixels": pix, "which": which})[0].astype(np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", type=int, default=512)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(8)
    n_datasets = len(datasets.NAMES)
    _, nets = detector.reading_folds(1, n_datasets)
    df, arrays, _ = detector.table(("dev",))
    df = df[df.readable].reset_index(drop=True)
    fp = np.full(len(df), np.nan)
    q8 = np.full(len(df), np.nan)
    q8m = np.full(len(df), np.nan)
    rng = np.random.default_rng(datasets.SEED)
    timing = {"fp32": 0.0, "int8": 0.0}
    t0 = time.time()
    for k, (net, mean, sd) in enumerate(nets):
        f32 = os.path.join(OUT, "fold%d.onnx" % k)
        i8 = os.path.join(OUT, "fold%d.int8.onnx" % k)
        x = torch.rand(2, 1, 384, 384)
        w = torch.eye(n_datasets)[[0, 1]]
        with torch.no_grad():
            torch.onnx.export(One(net, mean, sd).eval(), (x, w), f32, input_names=["pixels", "which"], output_names=["prob"],
                              dynamic_axes={"pixels": {0: "n"}, "which": {0: "n"}, "prob": {0: "n"}}, opset_version=18, dynamo=False)
        train = np.flatnonzero(df.fold.values != k)
        calib = np.sort(rng.choice(train, size=min(a.calibrate, len(train)), replace=False))
        quantize_static(f32, i8, Feed(df, arrays, calib), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        held = np.flatnonzero(df.fold.values == k)
        s32, s8 = session(f32, 8), session(i8, 8)
        fp[held] = score(s32, df, arrays, held)
        q8[held] = score(s8, df, arrays, held)
        q8m[held] = (q8[held] + score(s8, df, arrays, held, mirrored=True)) / 2.0
        pix, which = pictures(df, arrays, held[:8])
        for name, s in (("fp32", session(f32, 4)), ("int8", session(i8, 4))):
            s.run(None, {"pixels": pix, "which": which})
            t1 = time.perf_counter()
            for _ in range(5):
                s.run(None, {"pixels": pix, "which": which})
            timing[name] += (time.perf_counter() - t1) / 5 / 8
        print("fold %d: %d held out, %.0f s" % (k, len(held), time.time() - t0), flush=True)
    numbers = {"seconds_per_picture_fp32_5folds_4threads": (timing["fp32"], "s"),
               "seconds_per_picture_int8_5folds_4threads": (timing["int8"], "s")}
    print("five folds on four threads, a batch of eight: fp32 %.4f s a picture, int8 %.4f (%.2fx)"
          % (timing["fp32"], timing["int8"], timing["fp32"] / timing["int8"]))
    for arm, scores in (("int8", q8), ("int8_mirrored", q8m)):
        for d in datasets.NAMES:
            m = (df.dataset == d).values
            t = df.truth.values[m].astype(int)
            a_auc, b_auc, diff, lo, hi, _ = mirror.paired_auroc(t, scores[m], fp[m], df.patient.values[m].astype(str))
            print("%s, %s: AUROC fp32 %.4f, %s %.4f, difference %+.4f (95%% CI %+.4f to %+.4f), largest score change %.4f"
                  % (arm, d, a_auc, arm, b_auc, diff, lo, hi, float(np.max(np.abs(scores[m] - fp[m])))))
            numbers.update({"fp32.%s.auroc" % d: (a_auc, ""), "%s.%s.auroc_b" % (arm, d): (b_auc, ""), "%s.%s.auroc_minus_reference" % (arm, d): (diff, ""),
                            "%s.%s.auroc_minus_reference_low" % (arm, d): (lo, ""),
                            "%s.%s.auroc_minus_reference_high" % (arm, d): (hi, "")})
    pd.DataFrame({"dataset": df.dataset, "image": df.image, "patient": df.patient, "fold": df.fold, "truth": df.truth,
                  "fp32": fp, "int8": q8, "int8_mirrored": q8m}).to_csv(os.path.join(OUT, "oof_dev_int8_v1.csv"), index=False)
    ledger.write("experiment", "xray reading: each fold as an int8 ONNX graph calibrated on its own training pictures, plain and mirrored, against plain fp32 (dev out-of-fold, processor)",
                 time.time() - t0, numbers, settings={"calibrate": str(a.calibrate)}, note="modalities/xray/eval/int8.py")


if __name__ == "__main__":
    main()
