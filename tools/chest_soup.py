"""Make a chest X-ray reader "version" out of readers already trained, with no training.

  python tools/chest_soup.py soup   --versions 103 106 --out 190
      Uniformly average the fold 0 weights of the listed versions (a "model soup"), read fold 0's
      held-out pictures with the averaged net, and write work/eval/chest/reader-v190/ in the format
      reader.py writes (fold0.pt, meta.json, oof_logits.npz). Needs the nets to share their layout,
      picture size and normalisation. Reads the card only (no_grad, bfloat16 autocast, as predict does).
      RAIDIOLOGY_SOUP_CARD_GB (default 1.5) is the card share leased for the reading.

  python tools/chest_soup.py logits --versions 103 106 --out 192
      Average the saved held-out logits of the listed versions (they must hold the same rows and
      labels). No net, no card.

  python tools/chest_soup.py logits --versions 118 119 --out 210 --fold 0
      Same, keeping only fold 0's held-out rows of every version (for a version trained on fold 0 alone).

  python tools/chest_soup.py tta --version 103 --out 194
      Read fold 0's held-out pictures with one version's fold 0 net five times (as they are; shifted
      by +5% and -5% of the width; turned +5 and -5 degrees, on the card, the way training's own
      nudges are made) and average the five logit sets. Writes work/eval/chest/reader-v194/ like
      the soup mode, prints the mean AUROC of the 14 findings per look and for the average, and the
      seconds for the five looks against one. Same card share as the soup mode.

  python tools/chest_soup.py merge --versions 119 211 212 213 214 --out 215 --like 118
      Join versions that each hold some of the five folds (the masked-autoencoder reader is trained one
      fold per version) into one five-fold version. Each version's fold<k>.pt is copied into
      work/eval/chest/reader-v215/, the held-out rows are put in --like's row order (a finished five-fold
      version), and meta.json gets the mean AUROC of the 14 findings on the merged rows. The versions must
      share settings and seed (and mae and cfg when present); their folds must be disjoint and cover 0-4.
      No net, no card.

compare.py and chestxdet_eval.py read any of these results like any trained version.
compare.py, chest_soup.py logits and final_eval.py read a merged result like any five-fold version.
"""
import argparse
import json
import os
import shutil
import sys
import time

# idle OpenMP threads sleep after 1 ms instead of spinning for 200; read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "modalities", "chest_xray", "eval"))
import reader  # noqa: E402  (also binds core, datasets, prepare, split as reader.py does)
from core import card, ledger  # noqa: E402

datasets = reader.datasets
xd = reader.xd
FOLD = 0


def load_version(v):
    """(checkpoint, meta, fold 0's info) of a trained version; refuses a version without fold 0."""
    d = reader.out_dir(v)
    pt, mp = os.path.join(d, "fold%d.pt" % FOLD), os.path.join(d, "meta.json")
    for p in (pt, mp):
        if not os.path.exists(p):
            raise SystemExit("v%d: %s is missing" % (v, p))
    with open(mp, encoding="utf-8") as f:
        meta = json.load(f)
    infos = [i for i in meta.get("folds", []) if i.get("fold") == FOLD]
    if not infos:
        raise SystemExit("v%d: meta.json has no fold %d entry" % (v, FOLD))
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    if "crop" in (ck.get("settings") or {}):
        raise SystemExit("v%d was trained on a crop cache; reader.table() would show it the wrong pictures" % v)
    return ck, meta, infos[0]


def check_same(versions, cks, metas, infos):
    first = cks[0]["state"]
    shapes = {k: tuple(t.shape) for k, t in first.items()}
    for v, ck in zip(versions[1:], cks[1:]):
        got = {k: tuple(t.shape) for k, t in ck["state"].items()}
        if got != shapes:
            diff = sorted(set(got) ^ set(shapes)) or [k for k in shapes if got[k] != shapes[k]]
            raise SystemExit("v%d and v%d differ in state_dict keys or shapes (e.g. %s)" % (versions[0], v, diff[:3]))
    caches = [reader.cache_size_of(ck.get("settings")) for ck in cks]
    if len(set(caches)) != 1:
        raise SystemExit("picture caches differ: %s" % dict(zip(versions, caches)))
    sizes = [int((ck.get("settings") or {}).get("size", c)) for ck, c in zip(cks, caches)]
    if len(set(sizes)) != 1:
        raise SystemExit("picture sizes differ: %s" % dict(zip(versions, sizes)))
    norms = [(info["mean"], info["sd"]) for info in infos]
    if len(set(norms)) != 1:
        raise SystemExit("normalisation (mean, sd) differs: %s" % dict(zip(versions, norms)))
    return sizes[0], norms[0]


def average_states(cks):
    """Uniform mean of every floating-point tensor; other buffers (num_batches_tracked) from the first net."""
    out, n_avg = {}, 0
    for k, t0 in cks[0]["state"].items():
        if t0.is_floating_point():
            acc = torch.zeros_like(t0, dtype=torch.float64)
            for ck in cks:
                acc += ck["state"][k].to(torch.float64)
            out[k] = (acc / len(cks)).to(t0.dtype)
            n_avg += 1
        else:
            out[k] = t0.clone()
    return out, n_avg


def auroc_meta(logit, y):
    n_lab = len(datasets.LABELS)
    per = [xd.auroc(logit[:, i], y[:, i].astype(int)) for i in range(n_lab)]
    return dict(zip(datasets.LABELS, per)), float(np.nanmean(per[:14]))


def write_npz(out, z, logit):
    np.savez(os.path.join(out, "oof_logits.npz"), image=z["image"], patient=z["patient"], fold=z["fold"],
             y=z["y"], logit=logit.astype(np.float32))


def write_meta(out, meta):
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


def check_out(version, versions):
    if version in versions:
        raise SystemExit("--out %d is one of the inputs" % version)
    out = reader.out_dir(version)
    if os.path.exists(os.path.join(out, "meta.json")):
        raise SystemExit("%s already holds a version; pick another --out" % out)
    os.makedirs(out, exist_ok=True)
    return out


def soup(a):
    versions = a.versions
    out = check_out(a.out, versions)
    # 2026-10-03: leased before the checkpoints and data are loaded (waiting jobs that had loaded them held
    # 2.1-6.9 GB; the machine ran out of committed memory at 23:21 and a training job crashed)
    gb = float(os.environ.get("RAIDIOLOGY_SOUP_CARD_GB", "1.5"))
    card.require_card(reader.DEV, torch)
    # 2026-10-04: one lease, taken inside limit_card_memory (a second one waited on this process's own share)
    xd.limit_card_memory(gb=gb, name="chest soup", short_min=10)
    loaded = []
    for v in versions:
        loaded.append(load_version(v))
        print("loaded v%d: %s" % (v, reader.out_dir(v)), flush=True)
    cks, metas, infos = [x[0] for x in loaded], [x[1] for x in loaded], [x[2] for x in loaded]
    px, (mean_v, sd_v) = check_same(versions, cks, metas, infos)
    state, n_avg = average_states(cks)
    print("averaged %d floating-point tensors of %d (others from v%d)" % (n_avg, len(state), versions[0]), flush=True)

    settings = cks[0].get("settings") or {}
    cfg = cks[0]["cfg"]
    # fold 0's held-out rows, as reader.train finds them for a resumed fold (reader.py:437)
    df, pix, cache_row, labels = reader.table(reader.cache_size_of(settings))
    rows = np.flatnonzero(df.fold.to_numpy() == FOLD)
    net = reader.ChestNet(reader.widths_of(settings, cfg), cfg["dropout"], cfg["stem_stride"], len(datasets.LABELS),
                          renderings=int(settings.get("renderings", 1))).to(reader.DEV)
    net.load_state_dict(state)
    mean = torch.tensor(mean_v, dtype=torch.float32, device=reader.DEV)
    sd = torch.tensor(sd_v, dtype=torch.float32, device=reader.DEV)
    with ledger.run("chest_soup", "soup v%d of %s" % (a.out, "+".join(str(v) for v in versions))) as led:
        led.settings_from({"soup_of": versions, "batch": a.batch})
        t0 = time.time()
        logit = reader.held_out_logits(net, pix, cache_row, labels, rows, mean, sd, a.batch, settings)
        took = time.time() - t0
        y = labels[rows].astype(np.int8)
        per, mean14 = auroc_meta(logit, y)
        led.number("read.seconds", took, "s").number("auroc.mean14", mean14).number("tensors", n_avg, "count")
        for name, v in per.items():
            led.number("auroc.%s" % name, v)
    print("read %d held-out pictures in %.1f s" % (len(rows), took), flush=True)

    got = np.full(len(rows), FOLD, np.int8)
    z = {"image": df.image.to_numpy(str)[rows], "patient": df.patient.to_numpy()[rows], "fold": got, "y": y}
    write_npz(out, z, logit)
    info = {**infos[0], "predicted_images": int(len(rows)), "soup_of": versions}
    meta = {"version": a.out, "cfg": cfg, "settings": settings, "seed": metas[0].get("seed", reader.SEED),
            "folds": [info], "auroc": per, "auroc_mean14": mean14, "soup_of": versions,
            "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": torch.cuda.get_device_name(0) if reader.DEV.type == "cuda" else "cpu"}
    write_meta(out, meta)
    torch.save({"state": state, "cfg": cfg, "info": info, "settings": settings,
                "fingerprint": {"soup_of": versions, "fold": FOLD, "version": a.out}, "soup_of": versions},
               os.path.join(out, "fold%d.pt" % FOLD))
    print("fold-0 mean AUROC of 14 findings %.4f -> %s" % (mean14, out))


def logits(a):
    versions = a.versions
    # a caller without --fold (mae.py's selftest) has no attribute: read every row
    fold = getattr(a, "fold", None)
    out = check_out(a.out, versions)
    zs, metas = [], []
    for v in versions:
        d = reader.out_dir(v)
        with np.load(os.path.join(d, "oof_logits.npz")) as f:
            z = {k: f[k] for k in f.files}
        if fold is not None:
            # one fold's rows: the same mask on every per-row array (first dimension = the row count)
            keep = z["fold"] == fold
            if not keep.any():
                raise SystemExit("v%d has no row of fold %d" % (v, fold))
            z = {k: x[keep] if x.ndim and x.shape[0] == len(keep) else x for k, x in z.items()}
            print("v%d: %d of %d rows are fold %d" % (v, keep.sum(), len(keep), fold), flush=True)
        zs.append(z)
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            metas.append(json.load(f))
        print("loaded v%d: %d rows" % (v, len(zs[-1]["image"])), flush=True)
    for v, z in zip(versions[1:], zs[1:]):
        for k in ("image", "patient", "fold", "y"):
            if z[k].shape != zs[0][k].shape or not np.array_equal(z[k], zs[0][k]):
                raise SystemExit("v%d and v%d differ in saved '%s' (not the same rows and labels)" % (versions[0], v, k))
    logit = np.mean([z["logit"].astype(np.float64) for z in zs], axis=0)
    write_npz(out, zs[0], logit)
    per, mean14 = auroc_meta(logit, zs[0]["y"])
    m0 = metas[0]
    folds = m0.get("folds", [])
    if fold is not None:
        folds = [i for i in folds if i.get("fold") == fold]
    meta = {**m0, "version": a.out, "folds": folds, "auroc": per, "auroc_mean14": mean14,
            "ensemble_of": versions, "trained": time.strftime("%Y-%m-%d %H:%M:%S")}
    if fold is not None:
        meta["fold_only"] = fold
    meta.pop("soup_of", None)
    write_meta(out, meta)
    print("averaged %d logit sets (%d rows): mean AUROC of 14 findings %.4f -> %s"
          % (len(zs), len(zs[0]["image"]), mean14, out))


# the five looks: (name, horizontal shift as a share of the width, turn in degrees)
LOOKS = [("as_is", 0.0, 0.0), ("shift+5%", 0.05, 0.0), ("shift-5%", -0.05, 0.0),
         ("turn+5deg", 0.0, 5.0), ("turn-5deg", 0.0, -5.0)]


def nudge(x, pad, shift, deg):
    """Shift and turn a normalised batch the way detector.augment does it: one affine_grid and a
    bilinear grid_sample, the same centre, the same aspect factors. Training pads with 0 in [0, 1]
    units (black); on a normalised batch that is `pad`, so the picture is warped about it."""
    if shift == 0.0 and deg == 0.0:
        return x
    n = x.shape[0]
    hw, wh = x.shape[-2] / x.shape[-1], x.shape[-1] / x.shape[-2]
    ang = torch.full((n,), deg * np.pi / 180.0, device=x.device, dtype=x.dtype)
    tx = torch.full((n,), shift * 2, device=x.device, dtype=x.dtype)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin * hw, tx
    theta[:, 1, 0], theta[:, 1, 1] = sin * wh, cos
    grid = reader.F.affine_grid(theta, list(x.shape), align_corners=False)
    return reader.F.grid_sample(x - pad, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + pad


def read_looks(net, pix, cache_row, labels, rows, mean, sd, batch, px, looks):
    """One logit set per look for `rows`, each batch read from disk once and nudged on the card
    (predict's no_grad and bfloat16 autocast)."""
    net.eval()
    out = [np.zeros((len(rows), len(datasets.LABELS)), np.float32) for _ in looks]
    parts = [rows[a:a + batch] for a in range(0, len(rows), batch)]
    pad = (0.0 - mean) / sd
    at = 0
    with torch.no_grad():
        for sl, x, _y in reader.prefetch(pix, cache_row, labels, parts):
            x = (reader.to_float(x, px) - mean) / sd
            for o, (_name, shift, deg) in zip(out, looks):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(reader.DEV.type == "cuda")):
                    z = net(nudge(x, pad, shift, deg))
                o[at:at + len(sl)] = z.float().cpu().numpy()
            at += len(sl)
    return out


def tta(a):
    out = check_out(a.out, [a.version])
    # 2026-10-03: leased before the checkpoint and data are loaded (see soup)
    gb = float(os.environ.get("RAIDIOLOGY_SOUP_CARD_GB", "1.5"))
    card.require_card(reader.DEV, torch)
    # 2026-10-04: one lease, taken inside limit_card_memory (a second one waited on this process's own share)
    xd.limit_card_memory(gb=gb, name="chest soup", short_min=10)
    ck, meta0, info0 = load_version(a.version)
    print("loaded v%d: %s" % (a.version, reader.out_dir(a.version)), flush=True)
    settings = ck.get("settings") or {}
    cfg = ck["cfg"]
    px = int(settings.get("size", reader.cache_size_of(settings)))
    df, pix, cache_row, labels = reader.table(reader.cache_size_of(settings))
    rows = np.flatnonzero(df.fold.to_numpy() == FOLD)
    net = reader.ChestNet(reader.widths_of(settings, cfg), cfg["dropout"], cfg["stem_stride"], len(datasets.LABELS),
                          renderings=int(settings.get("renderings", 1))).to(reader.DEV)
    net.load_state_dict(ck["state"])
    mean = torch.tensor(info0["mean"], dtype=torch.float32, device=reader.DEV)
    sd = torch.tensor(info0["sd"], dtype=torch.float32, device=reader.DEV)
    y = labels[rows].astype(np.int8)
    nudges = [{"look": n, "shift_share_of_width": s, "turn_degrees": d} for n, s, d in LOOKS]
    with ledger.run("chest_soup", "tta v%d of v%d" % (a.out, a.version)) as led:
        led.settings_from({"tta_of": a.version, "batch": a.batch, "looks": nudges})
        # the first look alone, through the reader's own path, is the cost to set the five against
        t0 = time.time()
        first = reader.held_out_logits(net, pix, cache_row, labels, rows, mean, sd, a.batch, settings)
        t_one = time.time() - t0
        t0 = time.time()
        sets = read_looks(net, pix, cache_row, labels, rows, mean, sd, a.batch, px, LOOKS)
        t_five = time.time() - t0
        _, m_first = auroc_meta(first, y)
        print("first look alone (reader's path): %.4f mean AUROC of 14, %.1f s" % (m_first, t_one), flush=True)
        for (name, _s, _d), z in zip(LOOKS, sets):
            _, m = auroc_meta(z, y)
            print("look %-10s mean AUROC of 14 %.4f" % (name, m), flush=True)
            led.number("auroc.mean14.%s" % name, m)
        logit = np.mean([z.astype(np.float64) for z in sets], axis=0)
        per, mean14 = auroc_meta(logit, y)
        led.number("read.seconds.one", t_one, "s").number("read.seconds.five", t_five, "s")
        led.number("auroc.mean14", mean14)
        for name, v in per.items():
            led.number("auroc.%s" % name, v)
    print("read %d held-out pictures: five looks %.1f s, first look alone %.1f s (%.2fx)"
          % (len(rows), t_five, t_one, t_five / t_one), flush=True)

    got = np.full(len(rows), FOLD, np.int8)
    z = {"image": df.image.to_numpy(str)[rows], "patient": df.patient.to_numpy()[rows], "fold": got, "y": y}
    write_npz(out, z, logit)
    tta_meta = {"source_version": a.version, "looks": nudges, "average": "mean of logits"}
    info = {**info0, "predicted_images": int(len(rows)), "tta": tta_meta}
    meta = {"version": a.out, "cfg": cfg, "settings": settings, "seed": meta0.get("seed", reader.SEED),
            "folds": [info], "auroc": per, "auroc_mean14": mean14, "mode": "tta", "tta": tta_meta,
            "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": torch.cuda.get_device_name(0) if reader.DEV.type == "cuda" else "cpu"}
    write_meta(out, meta)
    # the net is the source's; the copy is what makes the folder load like any version
    torch.save({**ck, "info": info, "fingerprint": {"tta_of": a.version, "fold": FOLD, "version": a.out}},
               os.path.join(out, "fold%d.pt" % FOLD))
    print("fold-0 mean AUROC of 14 findings %.4f -> %s" % (mean14, out))


def merge(a):
    """Join versions that each hold some of folds 0-4 into one five-fold version. No net, no card."""
    versions = a.versions
    if len(versions) < 2 or len(set(versions)) != len(versions):
        raise SystemExit("--versions needs two or more different versions")
    out = check_out(a.out, versions)
    metas, zs = [], []
    for v in versions:
        d = reader.out_dir(v)
        mp = os.path.join(d, "meta.json")
        if not os.path.exists(mp):
            raise SystemExit("v%d: %s is missing" % (v, mp))
        with open(mp, encoding="utf-8") as f:
            metas.append(json.load(f))
        with np.load(os.path.join(d, "oof_logits.npz")) as f:
            zs.append({k: f[k] for k in f.files})
    m0 = metas[0]
    for v, m in zip(versions[1:], metas[1:]):
        for k in ("settings", "seed"):
            if m.get(k) != m0.get(k):
                raise SystemExit("v%d and v%d differ in '%s'" % (versions[0], v, k))
        # only when a version has the key
        for k in ("mae", "cfg"):
            if (k in m or k in m0) and m.get(k) != m0.get(k):
                raise SystemExit("v%d and v%d differ in '%s'" % (versions[0], v, k))
    # which folds each version holds: meta's fold infos must match its saved rows and its checkpoints
    seen = {}
    for v, m, z in zip(versions, metas, zs):
        listed = sorted(int(i["fold"]) for i in m.get("folds", []))
        held = sorted(set(int(k) for k in z["fold"].tolist()))
        if listed != held:
            raise SystemExit("v%d: meta.json folds %s and oof_logits.npz folds %s differ" % (v, listed, held))
        for k in listed:
            if k in seen:
                raise SystemExit("fold %d is in both v%d and v%d" % (k, seen[k], v))
            if not os.path.exists(os.path.join(reader.out_dir(v), "fold%d.pt" % k)):
                raise SystemExit("v%d: fold%d.pt is missing" % (v, k))
            seen[k] = v
        print("v%d: folds %s, %d rows" % (v, listed, len(z["image"])), flush=True)
    if sorted(seen) != list(range(5)):
        raise SystemExit("the versions cover folds %s, not 0-4" % sorted(seen))
    with np.load(os.path.join(reader.out_dir(a.like), "oof_logits.npz")) as f:
        like = {k: f[k] for k in f.files}
    if sorted(set(int(k) for k in like["fold"].tolist())) != list(range(5)):
        raise SystemExit("--like v%d is not a five-fold version" % a.like)
    # the rows in --like's order: the same pictures once each, then patient, fold and labels equal
    img = np.concatenate([z["image"] for z in zs])
    if len(set(img.tolist())) != len(img):
        raise SystemExit("the versions share a picture (duplicate rows)")
    if set(img.tolist()) != set(like["image"].tolist()) or len(img) != len(like["image"]):
        raise SystemExit("the merged pictures (%d) are not v%d's pictures (%d)" % (len(img), a.like, len(like["image"])))
    pos = {x: i for i, x in enumerate(img.tolist())}
    order = np.array([pos[x] for x in like["image"].tolist()], dtype=np.int64)
    merged = {k: np.concatenate([z[k] for z in zs])[order] for k in ("image", "patient", "fold", "y", "logit")}
    for k in ("patient", "fold", "y"):
        if not np.array_equal(merged[k], like[k]):
            raise SystemExit("merged '%s' differs from v%d's (not the same rows and labels)" % (k, a.like))
    for k, v in sorted(seen.items()):
        src = os.path.join(reader.out_dir(v), "fold%d.pt" % k)
        shutil.copy2(src, os.path.join(out, "fold%d.pt" % k))
    write_npz(out, merged, merged["logit"])
    per, mean14 = auroc_meta(merged["logit"], merged["y"])
    folds = sorted((i for m in metas for i in m.get("folds", [])), key=lambda i: int(i["fold"]))
    meta = {**m0, "version": a.out, "folds": folds, "auroc": per, "auroc_mean14": mean14,
            "merged_from": versions, "trained": time.strftime("%Y-%m-%d %H:%M:%S")}
    for k in ("soup_of", "ensemble_of", "fold_only"):
        meta.pop(k, None)
    write_meta(out, meta)
    print("merged %d versions (%d rows, folds 0-4): mean AUROC of 14 findings %.4f -> %s"
          % (len(versions), len(img), mean14, out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("soup", "logits"):
        p = sub.add_parser(name)
        p.add_argument("--versions", type=int, nargs="+", required=True)
        p.add_argument("--out", type=int, required=True)
        if name == "soup":
            p.add_argument("--batch", type=int, default=reader.CFG["batch"])
        if name == "logits":
            p.add_argument("--fold", type=int, default=None)
    p = sub.add_parser("tta")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--out", type=int, required=True)
    p.add_argument("--batch", type=int, default=reader.CFG["batch"])
    p = sub.add_parser("merge")
    p.add_argument("--versions", type=int, nargs="+", required=True)
    p.add_argument("--out", type=int, required=True)
    p.add_argument("--like", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "tta":
        return tta(a)
    if a.cmd == "merge":
        return merge(a)
    if len(a.versions) < 2 or len(set(a.versions)) != len(a.versions):
        raise SystemExit("--versions needs two or more different versions")
    soup(a) if a.cmd == "soup" else logits(a)


if __name__ == "__main__":
    main()
