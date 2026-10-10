"""X-ray accuracy trial: average each fold's score with its score on the mirrored picture.

    python modalities/xray/eval/mirror.py [--threads 8]

Out-of-fold on development only: fold k's model reads fold k's held-out pictures, plain and
mirrored, on the processor in fp32, both arms in this one run, so the only difference between them
is the mirror. Then the AUROC difference per collection with score.py's patient bootstrap, and the
verdict into the run ledger. Training already mirrors half its pictures, so the mirrored view is one
the network knows. Measured 2026-10-02: FracAtlas +0.0061 (95% CI +0.0024 to +0.0102), GRAZPEDWRI-DX
+0.0025 (+0.0018 to +0.0032), both kept on development; the plain arm on the processor matched the
scored GPU run to the fourth decimal (0.8680 against 0.8682, 0.9654 against 0.9654).
"""
import argparse
import json
import os
import sys
import time

# idle OpenMP threads sleep after 1 ms instead of spinning for 200 (docs/efficiency, 2026-10-03); read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [HERE, ROOT]
os.environ["RAIDIOLOGY_DEVICE"] = "cpu"
import datasets  # noqa: E402  arkitect-allow: test-floor-loosened - after the device and sys.path lines, as in every eval script here
import detector  # noqa: E402  arkitect-allow: test-floor-loosened - after the device and sys.path lines, as in every eval script here
import score  # noqa: E402  arkitect-allow: test-floor-loosened - after the device and sys.path lines, as in every eval script here
from core import ledger, paths  # noqa: E402  arkitect-allow: test-floor-loosened - after the device and sys.path lines, as in every eval script here

OUT = os.path.join(paths.WORK, "eval", "xray")


def scores(threads, batch=64):
    torch.set_num_threads(threads)
    n_datasets = len(datasets.NAMES)
    meta, nets = detector.reading_folds(1, n_datasets)
    df, arrays, _ = detector.table(("dev",))
    df = df[df.readable].reset_index(drop=True)
    plain = np.full(len(df), np.nan)
    flip = np.full(len(df), np.nan)
    t0 = time.time()
    with torch.no_grad():
        for k, (net, mean, sd) in enumerate(nets):
            rows = np.flatnonzero(df.fold.values == k)
            for a in range(0, len(rows), batch):
                sl = rows[a:a + batch]
                part = df.iloc[sl]
                pix = np.stack([arrays[d][0][r] for d, r in zip(part.dataset, part.cache_row)])
                x = torch.from_numpy(pix).float().div_(255.0).unsqueeze(1)
                x = ((x - mean) / sd).contiguous(memory_format=torch.channels_last)
                which = F.one_hot(torch.as_tensor(part.dataset_index.values, dtype=torch.long), n_datasets).float()
                both = torch.cat([x, x.flip(-1)]).contiguous(memory_format=torch.channels_last)
                p = torch.sigmoid(net(both, torch.cat([which, which]))).double().numpy()
                plain[sl] = p[:len(sl)]
                flip[sl] = p[len(sl):]
            print("fold %d: %d pictures, %.0f s" % (k, len(rows), time.time() - t0), flush=True)
    out = df[["dataset", "image", "patient", "fold", "truth"]].copy()
    out["plain"], out["mirrored"] = plain, flip
    out["both"] = (plain + flip) / 2.0
    return out, meta, time.time() - t0


def paired_auroc(t, s_b, s_a, patients):
    P, unit = score.patient_index(list(patients))
    M = score.multiplicity(len(P))
    per = []
    for s in (s_b, s_a):
        vals, bins = np.unique(s, return_inverse=True)
        Sp = sparse.csr_matrix((np.ones((t == 1).sum(), np.float32), (unit[t == 1], bins[t == 1])), shape=(len(P), len(vals)))
        Sn = sparse.csr_matrix((np.ones((t == 0).sum(), np.float32), (unit[t == 0], bins[t == 0])), shape=(len(P), len(vals)))
        full = score.auroc_from_bins(np.asarray(Sp.sum(0)).ravel(), np.asarray(Sn.sum(0)).ravel())
        boots = np.concatenate([score.auroc_from_bins(np.asarray(Sp.T.dot(M[i:i + score.CHUNK].T).T),
                                                      np.asarray(Sn.T.dot(M[i:i + score.CHUNK].T).T))
                                for i in range(0, score.BOOT, score.CHUNK)])
        per.append((float(full), boots))
    diff = per[0][1] - per[1][1]
    lo, hi = np.percentile(diff[~np.isnan(diff)], [2.5, 97.5])
    return per[1][0], per[0][0], per[0][0] - per[1][0], float(lo), float(hi), len(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out, meta, secs = scores(a.threads)
    f = os.path.join(OUT, "oof_dev_flip_v1.csv")
    out.to_csv(f, index=False)
    oof = pd.read_csv(os.path.join(detector.model_dir(1), "oof_dev.csv"), dtype={"patient": str})
    numbers = {}
    for d in datasets.NAMES:
        m = out[out.dataset == d]
        t = m.truth.values.astype(int)
        ref = oof[oof.dataset == d].set_index("image").score.reindex(m.image.values).values
        a_auc, b_auc, diff, lo, hi, n_p = paired_auroc(t, m.both.values, m.plain.values, m.patient.astype(str).values)
        verdict = "kept" if lo > 0 else "dropped" if hi < 0 else "inconclusive"
        print("%s, %d pictures of %d patients: AUROC plain %.4f (GPU bf16 run: %.4f), plain+mirrored %.4f, "
              "difference %+.4f (95%% CI %+.4f to %+.4f): %s"
              % (d, len(m), n_p, a_auc, detector.auroc(ref, t), b_auc, diff, lo, hi, verdict.upper()), flush=True)
        numbers.update({"%s.auroc_a" % d: (a_auc, ""), "%s.auroc_b" % d: (b_auc, ""),
                        "%s.auroc_minus_reference" % d: (diff, ""), "%s.auroc_minus_reference_low" % d: (lo, ""),
                        "%s.auroc_minus_reference_high" % d: (hi, ""), "%s.verdict_kept" % d: (int(verdict == "kept"), "count")})
    ledger.write("experiment", "xray reading: each fold also reads the mirrored picture, scores averaged (dev out-of-fold, processor fp32)",
                 secs, numbers, settings={"file": os.path.relpath(f, ROOT), "threads": str(a.threads)},
                 note="modalities/xray/eval/mirror.py")


if __name__ == "__main__":
    main()
