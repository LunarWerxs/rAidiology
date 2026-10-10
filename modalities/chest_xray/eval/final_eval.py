"""NIH ChestX-ray14's own test list, opened ONCE, scoring the chest reader that was fixed on development.

  python modalities/chest_xray/eval/final_eval.py --versions 103,106 [--read 192] [--batch 64]
         [--device cpu|card] [--card-gb 1.5]
  python modalities/chest_xray/eval/final_eval.py --second-look --versions 106,118 --read 197 [--batch 64]
         [--device cpu|card] [--card-gb 1.5]
  python modalities/chest_xray/eval/final_eval.py --selftest

The reading is reader-v192: chest reader versions 103 and 106 (each five fold nets), a picture's logits
the mean over the ten nets (each version's five-fold mean, then the mean of the two versions: equal
weight per net). v192 is the same read kept on development ('ensemble_of': [103, 106]); this script
refuses a read whose meta.json says otherwise. A version may be a mae.py fine-tuned ViT-S (settings arch 'vit-s16'):
it reads the same 384 cache at its own px, and the gates and the numbers are the same.

Why each choice:
- The marker work/eval/chest/nih_test_opened.json is written (open 'x') before any test picture is read,
  so from then on the test counts as opened, whatever happens. A second run refuses.
- Numbers follow taix.py: per label AUROC with a 95% interval from a PATIENT bootstrap (2,000 resamples,
  seed 0, a drawn patient's pictures counted as many times as drawn). The mean over the 14 findings takes
  its interval from the SAME resamples (each finding's weighted AUROC per resample, then the mean).
- A second look (--second-look): a read kept on development with no test number may read the spent test list
  once more, only when compare.compare says its development gain over the read the test last read is BETTER
  (paired patient bootstrap of the out-of-fold logits), checked before anything touches the test. Once per
  read; a look that comes out lower is reported, never re-run. work/eval/chest/nih_test_looks.json lists the
  looks (the first look counts as look 1) and the gate refusals; each look has its own marker
  nih_test_look<L>_v<read>.json (open 'x', before any picture is read) and its own folder
  nih_test/look<L>-v<read>/. A look opened but not finished is resumed by the same command, without a new
  gate. The first look's files are never touched.
- Default device is the processor (the card is shared); `--device card` takes the card lease as
  tools/chest_soup.py does. torch and reader are imported inside the functions, so --selftest needs
  neither data nor card."""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
from core import paths  # noqa: E402

FOLDS = 5
BOOT, BOOT_SEED = 2000, 0
N_FINDINGS = 14
EVAL_DIR = os.path.join(paths.WORK, "eval", "chest")


def say(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def vdir(eval_dir, v):
    return os.path.join(eval_dir, "reader-v%d" % v)


def marker_path(eval_dir):
    return os.path.join(eval_dir, "nih_test_opened.json")


# ---------------------------------------------------------------- step 1: the gates

def check_gates(eval_dir, versions, read):
    """Stops unless the test is unopened, the read was fixed on development as these versions, and every
    version has its five fold nets and meta.json."""
    marker = marker_path(eval_dir)
    if os.path.exists(marker):
        try:
            m = json.load(open(marker, encoding="utf-8"))
            who = "versions %s, read %s, at %s" % (m.get("versions"), m.get("read"), m.get("time"))
        except (OSError, ValueError):
            who = "unknown"
        raise SystemExit("the NIH test split was already opened (%s; %s); it opens once" % (who, marker))
    rp = os.path.join(vdir(eval_dir, read), "meta.json")
    if not os.path.exists(rp):
        raise SystemExit("read v%d has no meta.json (%s)" % (read, rp))
    ens = json.load(open(rp, encoding="utf-8")).get("ensemble_of")
    if ens is None or sorted(ens) != sorted(versions):
        raise SystemExit("read v%d is the ensemble of %s, not of %s: the read must be the one fixed on development"
                         % (read, ens, sorted(versions)))
    for v in versions:
        for name in ["fold%d.pt" % k for k in range(FOLDS)] + ["meta.json"]:
            p = os.path.join(vdir(eval_dir, v), name)
            if not os.path.exists(p):
                raise SystemExit("v%d: %s is missing" % (v, p))


def load_checked(eval_dir, versions):
    """Loads every checkpoint on the processor and refuses a crop-cache version, a checkpoint without its
    normalisation, or versions whose cache differs, or CNN versions whose picture size differs. A mae.py ViT reads
    the shared cache at its own px, so its px is not compared. Returns (cache side, CNN picture side or None)."""
    import torch
    import reader
    caches, sizes = set(), set()
    for v in versions:
        for k in range(FOLDS):
            ck = torch.load(os.path.join(vdir(eval_dir, v), "fold%d.pt" % k), map_location="cpu", weights_only=False)
            s = ck.get("settings") or {}
            if "crop" in s:
                raise SystemExit("v%d fold %d was trained on a crop cache; the table would show it the wrong pictures" % (v, k))
            info = ck.get("info") or {}
            if "mean" not in info or "sd" not in info:
                raise SystemExit("v%d fold %d: checkpoint has no info mean/sd" % (v, k))
            c = reader.cache_size_of(s)
            caches.add(c)
            if s.get("arch") != "vit-s16":
                sizes.add(int(s.get("size", c)))
            del ck
    if len(caches) != 1 or len(sizes) > 1:
        raise SystemExit("versions differ in picture size (caches %s, sizes %s)" % (sorted(caches), sorted(sizes)))
    return caches.pop(), (sizes.pop() if sizes else None)


# ---------------------------------------------------------------- step 3: the test table

def test_table(size):
    """Like reader.table, for the test split: (df, pix, cache_row, labels, unreadable count)."""
    import reader
    datasets, splitmod, prepare = reader.datasets, reader.splitmod, reader.prepare
    df = splitmod.frame("test").reset_index(drop=True)
    pix, order, _ = prepare.load(size)
    bad = prepare.unreadable(size)
    keep = np.array([str(i) not in bad and str(i) in order for i in df.image])
    n_bad = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    cache_row = np.array([order[str(i)] for i in df.image], dtype=np.int64)
    labels = df[datasets.LABELS].to_numpy(np.float32)
    return df, pix, cache_row, labels, n_bad


# ---------------------------------------------------------------- step 4: the read

def equal_weight_mean(per_version):
    """per_version: for each version, a list of fold logit arrays. Each version's fold mean, then the mean
    over versions (equal weight per net when every version has the same number of folds)."""
    return np.mean([np.mean([np.asarray(z, np.float64) for z in folds], axis=0) for folds in per_version], axis=0)


def build_reader_net(ck, n_labels, side):
    """(net, px) a checkpoint was trained as, its state loaded strictly. A mae.py fine-tuned ViT-S (settings arch
    'vit-s16') is rebuilt at its own px; a reader ChestNet at its settings' size, else the cache side `side`."""
    import reader
    settings = ck.get("settings") or {}
    if settings.get("arch") == "vit-s16":
        import mae
        px = int(settings["px"])
        enc = mae.ENC
        net = mae.ChestViT(mae.make_vit(px, enc["dim"], enc["depth"], enc["heads"]), n_labels)
    else:
        cfg = ck["cfg"]
        px = int(settings.get("size", side))
        net = reader.ChestNet(reader.widths_of(settings, cfg), cfg["dropout"], cfg["stem_stride"], n_labels,
                              renderings=int(settings.get("renderings", 1)))
    net = net.to(reader.DEV)
    net.load_state_dict(ck["state"])
    return net, px


def read_all(eval_dir, versions, table, batch):
    import torch
    import reader
    datasets = reader.datasets
    df, pix, cache_row, labels, _ = table
    rows = np.arange(len(df))
    per_version = []
    for v in versions:
        folds = []
        for k in range(FOLDS):
            t0 = time.time()
            ck = torch.load(os.path.join(vdir(eval_dir, v), "fold%d.pt" % k), map_location="cpu", weights_only=False)
            info = ck["info"]
            net, px = build_reader_net(ck, len(datasets.LABELS), pix.shape[-1])
            mean = torch.tensor(info["mean"], dtype=torch.float32, device=reader.DEV)
            sd = torch.tensor(info["sd"], dtype=torch.float32, device=reader.DEV)
            folds.append(reader.predict(net, pix, cache_row, labels, rows, mean, sd, batch, px))
            del net, ck
            if reader.DEV.type == "cuda":
                torch.cuda.empty_cache()
            say("v%d fold %d read: %.1f s" % (v, k, time.time() - t0))
        per_version.append(folds)
    return equal_weight_mean(per_version)


# ---------------------------------------------------------------- step 6: the numbers

def _auroc_weighted(order, gid, y, w):
    """AUROC of the pictures counted w times each (order: ascending score; gid: tie group of each sorted picture)."""
    ys, ws = y[order], w[order]
    ng = int(gid[-1]) + 1
    gp = np.bincount(gid, weights=ws * ys, minlength=ng)
    gn = np.bincount(gid, weights=ws * (1 - ys), minlength=ng)
    wp, wn = gp.sum(), gn.sum()
    if wp == 0 or wn == 0:
        return float("nan")
    below = np.cumsum(gn) - gn
    return float((gp * (below + 0.5 * gn)).sum() / (wp * wn))


def auroc_points(logits, truth):
    out = []
    for j in range(logits.shape[1]):
        order = np.argsort(logits[:, j], kind="mergesort")
        s = logits[order, j]
        gid = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])])
        out.append(_auroc_weighted(order, gid, truth[:, j].astype(float), np.ones(len(s))))
    return out


def patient_counts(patients, n=BOOT, seed=BOOT_SEED):
    """(pid per picture, n resamples x patients draw counts): each resample draws the patients with
    replacement; a drawn patient's pictures all count as many times as the patient was drawn."""
    _, pid = np.unique(np.asarray(patients), return_inverse=True)
    n_pat = int(pid.max()) + 1
    rng = np.random.default_rng(seed)
    return pid, [np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat) for _ in range(n)]


def bootstrap(logits, truth, patients, n=BOOT, seed=BOOT_SEED, n_findings=N_FINDINGS):
    """{'points': [(point, lo, hi) per label], 'mean': (point, lo, hi) over the first n_findings labels}; the
    mean's interval is the percentiles of the per-resample mean over findings, from the same resamples."""
    logits, truth = np.asarray(logits, float), np.asarray(truth)
    pid, counts = patient_counts(patients, n, seed)
    boots = np.full((n, logits.shape[1]), np.nan)
    points = []
    for j in range(logits.shape[1]):
        order = np.argsort(logits[:, j], kind="mergesort")
        s = logits[order, j]
        gid = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])])
        y = truth[:, j].astype(float)
        points.append(_auroc_weighted(order, gid, y, np.ones(len(y))))
        for b, c in enumerate(counts):
            boots[b, j] = _auroc_weighted(order, gid, y, c[pid].astype(float))
    out = []
    for j in range(logits.shape[1]):
        col = boots[:, j]
        lo, hi = np.nanpercentile(col, [2.5, 97.5]) if np.isfinite(col).any() else (float("nan"),) * 2
        out.append((points[j], float(lo), float(hi)))
    mb = boots[:, :n_findings].mean(axis=1)
    lo, hi = np.nanpercentile(mb, [2.5, 97.5])
    return {"points": out, "mean": (float(np.mean(points[:n_findings])), float(lo), float(hi))}


def dev_mean(eval_dir, versions):
    """The read's development mean AUROC over the 14 findings, from the versions' out-of-fold logits averaged
    on the rows all of them hold (matched by image). Returns (mean, rows used)."""
    zs = []
    for v in versions:
        with np.load(os.path.join(vdir(eval_dir, v), "oof_logits.npz")) as f:
            zs.append({k: f[k] for k in ("image", "y", "logit")})
    common = set(zs[0]["image"].astype(str))
    for z in zs[1:]:
        common &= set(z["image"].astype(str))
    names = sorted(common)
    picked = []
    for z in zs:
        at = {n: i for i, n in enumerate(z["image"].astype(str))}
        picked.append(np.array([at[n] for n in names], dtype=np.int64))
    logit = np.mean([z["logit"][ix].astype(np.float64) for z, ix in zip(zs, picked)], axis=0)
    y = zs[0]["y"][picked[0]]
    per = auroc_points(logit[:, :N_FINDINGS], y[:, :N_FINDINGS])
    return float(np.nanmean(per)), len(names)


def write_results(out, labels, res, facts, look=None):
    r = {"labels": {n: {"auroc": p, "lo": lo, "hi": hi, "positives": int(facts["positives"][n])}
                    for n, (p, lo, hi) in zip(labels, res["points"])},
         "mean14": {"auroc": res["mean"][0], "lo": res["mean"][1], "hi": res["mean"][2]}, **facts["rest"]}
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1)
    title = "# NIH ChestX-ray14 test: reader v%s (the ten nets of v%s)\n" % (facts["rest"]["read"], facts["rest"]["versions"])
    if look:
        title = ("# NIH ChestX-ray14 test, second look (look %d): reader v%s (the nets of v%s)\n"
                 % (look["look"], facts["rest"]["read"], facts["rest"]["versions"]))
    lines = [title,
             "%d pictures, %d patients, %.1f abnormal per 100; %d unreadable pictures left out. "
             "Patient bootstrap, %d resamples, seed %d.\n" % (
                 facts["rest"]["pictures"], facts["rest"]["patients"], facts["rest"]["abnormal_per_100"],
                 facts["rest"]["unreadable"], BOOT, BOOT_SEED),
             "| label | AUROC | 95% interval | positives |", "|---|---|---|---|"]
    for n, (p, lo, hi) in zip(labels, res["points"]):
        lines.append("| %s | %.4f | %.4f to %.4f | %d |" % (n, p, lo, hi, facts["positives"][n]))
    m = res["mean"]
    lines.append("| **mean of the 14 findings** | %.4f | %.4f to %.4f | |" % m)
    lines.append("\nDevelopment mean of the 14 (out-of-fold, %d pictures both versions hold): %.4f\n"
                 % (facts["rest"]["dev_rows"], facts["rest"]["dev_mean14"]))
    if look:
        g = look["gate"]
        lines.append("First look (reference): mean of the 14 findings %.4f (read v%s).\n" % (look["first_mean14"], look["first_read"]))
        lines.append("Development gate: v%s against v%s, %s, mean of 14 gain %+.4f (%.4f to %.4f).\n"
                     % (g["against"], facts["rest"]["read"], g["verdict"], g["gain"], g["low"], g["high"]))
    with open(os.path.join(out, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return r


# ---------------------------------------------------------------- the run

def run(a, eval_dir=EVAL_DIR, load_fn=load_checked, table_fn=test_table, read_fn=read_all, numbers_fn=None):
    versions = sorted(a.versions)
    say("1. gates: marker, read v%d, checkpoints of %s" % (a.read, versions))
    check_gates(eval_dir, versions, a.read)
    if a.device == "cpu":
        os.environ["RAIDIOLOGY_DEVICE"] = "cpu"          # before reader is imported: the card is busy
    size, _px = load_fn(eval_dir, versions)
    if a.device != "cpu":
        import torch
        import reader
        reader.card.require_card(reader.DEV, torch)
        reader.xd.limit_card_memory(gb=a.card_gb, name="chest nih test")
    say("2. marker, then the test opens")
    marker = marker_path(eval_dir)
    os.makedirs(eval_dir, exist_ok=True)
    with open(marker, "x", encoding="utf-8") as f:           # from here the test counts as opened
        json.dump({"versions": versions, "read": a.read, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
    os.environ["RAIDIOLOGY_FINAL_EVAL"] = "1"
    say("3. the test table")
    table = table_fn(size)
    df, labels, n_bad = table[0], table[3], table[4]
    say("   %d pictures, %d patients, %d unreadable left out" % (len(df), df.patient.nunique(), n_bad))
    say("4. the read: %d nets" % (FOLDS * len(versions)))
    logits = read_fn(eval_dir, versions, table, a.batch)
    out = os.path.join(eval_dir, "nih_test")
    os.makedirs(out, exist_ok=True)
    say("5. saving test_logits.npz")
    np.savez(os.path.join(out, "test_logits.npz"), image=df.image.to_numpy(str), patient=df.patient.to_numpy(),
             logits=logits.astype(np.float32), truth=labels, versions=np.array(versions))
    say("6. numbers")
    return (numbers_fn or numbers)(eval_dir, out, versions, a.read, df, logits, labels, n_bad)


def numbers(eval_dir, out, versions, read, df, logits, labels, n_bad, look=None):
    from core import ledger
    import reader
    names = reader.datasets.LABELS
    res = bootstrap(logits, labels, df.patient.to_numpy())
    dm, dev_rows = dev_mean(eval_dir, versions)
    facts = {"positives": {n: labels[:, j].sum() for j, n in enumerate(names)},
             "rest": {"versions": versions, "read": read, "pictures": int(len(df)),
                      "patients": int(df.patient.nunique()), "unreadable": n_bad,
                      "abnormal_per_100": float(100 * labels[:, names.index("Abnormal")].mean()),
                      "dev_mean14": dm, "dev_rows": dev_rows,
                      "boot": BOOT, "seed": BOOT_SEED, "time": time.strftime("%Y-%m-%d %H:%M:%S")}}
    r = write_results(out, names, res, facts, look)
    name = "nih test v%s" % "+v".join(str(v) for v in versions) + (" look %d" % look["look"] if look else "")
    with ledger.run("experiment", name) as led:
        led.settings_from({"versions": versions, "read": read, "boot": BOOT, "seed": BOOT_SEED})
        for n, v in r["labels"].items():
            led.number("test.auroc.%s" % n, v["auroc"]).number("test.lo.%s" % n, v["lo"]) \
                .number("test.hi.%s" % n, v["hi"]).number("test.positives.%s" % n, v["positives"], "count")
        led.number("test.auroc.mean14", r["mean14"]["auroc"]).number("test.lo.mean14", r["mean14"]["lo"]) \
            .number("test.hi.mean14", r["mean14"]["hi"])
        led.number("test.pictures", r["pictures"], "count").number("test.patients", r["patients"], "count")
        led.number("test.unreadable", n_bad, "count").number("test.abnormal_per_100", r["abnormal_per_100"])
        led.number("dev.mean14", dm).number("dev.rows", dev_rows, "count")
    say("test mean AUROC of the 14 findings %.4f (%.4f to %.4f); development %.4f" % (
        r["mean14"]["auroc"], r["mean14"]["lo"], r["mean14"]["hi"], dm))
    for n, v in r["labels"].items():
        say("   %-18s %.4f (%.4f to %.4f)" % (n, v["auroc"], v["lo"], v["hi"]))
    return r


# ---------------------------------------------------------------- the second look

def looks_path(eval_dir):
    return os.path.join(eval_dir, "nih_test_looks.json")


def look_marker_path(eval_dir, look, read):
    return os.path.join(eval_dir, "nih_test_look%d_v%d.json" % (look, read))


def read_looks(eval_dir):
    """The looks record: {'looks': [...second looks...], 'gate_refusals': [...]}; empty when absent."""
    p = looks_path(eval_dir)
    rec = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    rec.setdefault("looks", [])
    rec.setdefault("gate_refusals", [])
    return rec


def write_looks(eval_dir, rec):
    os.makedirs(eval_dir, exist_ok=True)
    with open(looks_path(eval_dir), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)


def first_look(eval_dir):
    """Look 1 from the marker and nih_test/results.json, or None while the first look is not finished."""
    res = os.path.join(eval_dir, "nih_test", "results.json")
    if not (os.path.exists(marker_path(eval_dir)) and os.path.exists(res)):
        return None
    m = json.load(open(marker_path(eval_dir), encoding="utf-8"))
    r = json.load(open(res, encoding="utf-8"))
    return {"look": 1, "versions": m.get("versions"), "read": m.get("read", 192), "opened": m.get("time"),
            "finished": r.get("time", "done"), "mean14": r["mean14"]["auroc"]}


def compare_gate(a, b):
    """The development gate: compare.compare(a, b) on the out-of-fold logits, as the record the looks file keeps."""
    import compare
    r = compare.compare(a, b)
    _name, va, vb, gain, low, high = r["mean14"]
    return {"against": a, "verdict": r["verdict"], "gain": float(gain), "low": float(low), "high": float(high),
            "a": float(va), "b": float(vb)}


def check_second_gates(eval_dir, versions, read, first, finished):
    """Gates that need no development numbers: the first look finished, the read is the versions' ensemble, every
    version has its nets, and no finished look read this read already."""
    if first is None:
        raise SystemExit("the first look is not finished (marker and nih_test/results.json both needed); no second look")
    ens = None
    rp = os.path.join(vdir(eval_dir, read), "meta.json")
    if os.path.exists(rp):
        ens = json.load(open(rp, encoding="utf-8")).get("ensemble_of")
    if ens is None or sorted(ens) != sorted(versions):
        raise SystemExit("read v%d is the ensemble of %s, not of %s: the read must be the one fixed on development"
                         % (read, ens, sorted(versions)))
    for v in versions:
        for name in ["fold%d.pt" % k for k in range(FOLDS)] + ["meta.json"]:
            p = os.path.join(vdir(eval_dir, v), name)
            if not os.path.exists(p):
                raise SystemExit("v%d: %s is missing" % (v, p))
    for lk in finished:
        if lk["read"] == read:
            raise SystemExit("read v%d was already read by look %d; a read gets one look per test list" % (read, lk["look"]))


def run_second(a, eval_dir=EVAL_DIR, load_fn=load_checked, table_fn=test_table, read_fn=read_all, numbers_fn=None,
               gate_fn=compare_gate):
    """A second look at the spent test list: gates, development gate, look marker, then the read and numbers."""
    versions, read = sorted(a.versions), a.read
    first = first_look(eval_dir)
    rec = read_looks(eval_dir)
    done = ([first] if first else []) + [lk for lk in rec["looks"] if lk.get("finished")]
    say("1. second look gates: first look, read v%d, checkpoints of %s" % (read, versions))
    check_second_gates(eval_dir, versions, read, first, done)
    open_look = next((lk for lk in rec["looks"] if lk["read"] == read and not lk.get("finished")), None)
    if open_look:
        say("   resuming look %d (opened %s): no new gate" % (open_look["look"], open_look["opened"]))
        look = open_look
    else:
        latest = max(done, key=lambda lk: lk["look"])
        g = gate_fn(latest["read"], read)
        say("   development gate: v%s -> v%s: %.4f -> %.4f, gain %+.4f (%.4f to %.4f), %s"
            % (latest["read"], read, g.get("a", float("nan")), g.get("b", float("nan")), g["gain"], g["low"], g["high"], g["verdict"]))
        if g["verdict"] != "BETTER":
            rec["gate_refusals"].append({"read": read, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "gate": g})
            write_looks(eval_dir, rec)
            raise SystemExit("development gate says %s for v%d over v%s (gain %+.4f, %.4f to %.4f); no second look"
                             % (g["verdict"], read, latest["read"], g["gain"], g["low"], g["high"]))
        look = {"look": latest["look"] + 1, "versions": versions, "read": read,
                "opened": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None, "gate": g}
    if a.device == "cpu":
        os.environ["RAIDIOLOGY_DEVICE"] = "cpu"
    size, _px = load_fn(eval_dir, versions)
    if a.device != "cpu":
        import torch
        import reader
        reader.card.require_card(reader.DEV, torch)
        reader.xd.limit_card_memory(gb=a.card_gb, name="chest nih test")
    say("2. look %d marker, then the test opens" % look["look"])
    if not open_look:
        rec["looks"].append(look)
        write_looks(eval_dir, rec)
        with open(look_marker_path(eval_dir, look["look"], read), "x", encoding="utf-8") as f:
            json.dump({"look": look["look"], "versions": versions, "read": read, "time": look["opened"]}, f)
    os.environ["RAIDIOLOGY_FINAL_EVAL"] = "1"
    table = table_fn(size)
    df, labels, n_bad = table[0], table[3], table[4]
    say("3. the read: %d nets, %d pictures" % (FOLDS * len(versions), len(df)))
    logits = read_fn(eval_dir, versions, table, a.batch)
    out = os.path.join(eval_dir, "nih_test", "look%d-v%d" % (look["look"], read))
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "test_logits.npz"), image=df.image.to_numpy(str), patient=df.patient.to_numpy(),
             logits=logits.astype(np.float32), truth=labels, versions=np.array(versions))
    info = {"look": look["look"], "gate": look["gate"], "first_mean14": first["mean14"], "first_read": first["read"]}
    r = (numbers_fn or numbers)(eval_dir, out, versions, read, df, logits, labels, n_bad, look=info)
    look["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_looks(eval_dir, rec)
    return r


# ---------------------------------------------------------------- selftest

def selftest():
    import tempfile
    from types import SimpleNamespace

    def fake_eval(root, ens=(103, 106), read=192):
        for v in (103, 106):
            os.makedirs(vdir(root, v), exist_ok=True)
            json.dump({}, open(os.path.join(vdir(root, v), "meta.json"), "w"))
            for k in range(FOLDS):
                open(os.path.join(vdir(root, v), "fold%d.pt" % k), "wb").close()
        os.makedirs(vdir(root, read), exist_ok=True)
        json.dump({"ensemble_of": list(ens)}, open(os.path.join(vdir(root, read), "meta.json"), "w"))

    def args(**kw):
        return SimpleNamespace(**{"versions": [103, 106], "read": 192, "batch": 8, "device": "cpu", "card_gb": 1.5, **kw})

    def refuses(fn):
        try:
            fn()
        except SystemExit as e:
            return str(e)
        return None

    with tempfile.TemporaryDirectory() as root:
        # (a) an existing marker refuses
        fake_eval(root)
        check_gates(root, [103, 106], 192)
        json.dump({"versions": [1], "read": 1, "time": "t"}, open(marker_path(root), "w"))
        msg = refuses(lambda: check_gates(root, [103, 106], 192))
        assert msg and "already opened" in msg, msg
    # (b) a read whose ensemble_of differs refuses
    with tempfile.TemporaryDirectory() as root:
        fake_eval(root, ens=(103, 107))
        msg = refuses(lambda: check_gates(root, [103, 106], 192))
        assert msg and "not of" in msg, msg
    # (c) the marker is on disk before the read function is called
    with tempfile.TemporaryDirectory() as root:
        fake_eval(root)
        seen = {}

        class Stop(Exception):
            pass

        def read_stub(*_a):
            seen["marker"] = os.path.exists(marker_path(root))
            raise Stop()

        import pandas as pd
        df = pd.DataFrame({"image": ["a", "b"], "patient": [1, 2]})
        table = (df, None, None, np.zeros((2, 15), np.float32), 0)
        old = os.environ.get("RAIDIOLOGY_FINAL_EVAL")
        try:
            run(args(), eval_dir=root, load_fn=lambda *_a: (384, 384), table_fn=lambda s: table, read_fn=read_stub)
            raise AssertionError("read stub was not reached")
        except Stop:
            pass
        finally:
            if old is None:
                os.environ.pop("RAIDIOLOGY_FINAL_EVAL", None)
            else:
                os.environ["RAIDIOLOGY_FINAL_EVAL"] = old
        assert seen["marker"], "marker not on disk before the read"
        assert json.load(open(marker_path(root)))["versions"] == [103, 106]
        assert refuses(lambda: check_gates(root, [103, 106], 192)), "a second run must refuse"
    # (d) equal-weight average
    g = np.random.default_rng(0)
    fake = [[g.normal(size=(6, 15)) for _ in range(5)] for _ in range(2)]
    want = (sum(fake[0]) / 5 + sum(fake[1]) / 5) / 2
    assert np.allclose(equal_weight_mean(fake), want), "equal-weight mean"
    assert np.allclose(equal_weight_mean(fake), sum(fake[0] + fake[1]) / 10), "equal weight per net"
    # (e) bootstrap mean over findings
    n_pat, per = 120, 3
    pats = np.repeat(np.arange(n_pat), per)
    n = len(pats)
    truth = (g.random((n, 15)) < 0.3).astype(np.float32)
    logits = truth * 0.8 + g.normal(size=(n, 15))
    res = bootstrap(logits, truth, pats, n=200, seed=0)
    pts = auroc_points(logits, truth)
    assert np.isclose(res["mean"][0], np.mean(pts[:N_FINDINGS])), "mean of per-finding points"
    assert res["mean"][1] <= res["mean"][0] <= res["mean"][2], "interval contains the point"
    assert all(np.isclose(a_[0], b) for a_, b in zip(res["points"], pts)), "per-label points"
    # (f) a drawn patient's pictures count together
    pid, counts = patient_counts(pats, n=50, seed=0)
    for c in counts:
        w = c[pid]
        for p in range(n_pat):
            ws = w[pats == p]
            assert (ws == ws[0]).all() and ws[0] == c[p], "pictures of one patient share a weight"
    assert any((c == 0).any() for c in counts) and any((c > 1).any() for c in counts), "draw is with replacement"
    # (g) the second look
    second_look_selftest(fake_eval, args, refuses)
    print("selftest ok")


def second_look_selftest(fake_eval, args, refuses):
    import tempfile
    import pandas as pd

    def stub_read(*_a):
        return np.zeros((2, 15))

    def stub_numbers(eval_dir, out, versions, read, df, logits, labels, n_bad, look=None):
        r = {"mean14": {"auroc": 0.8}, "look": look["look"], "time": "t"}
        json.dump(r, open(os.path.join(out, "results.json"), "w"))
        return r

    def gate(verdict):
        return lambda a, b: {"against": a, "verdict": verdict, "gain": 0.01, "low": 0.005, "high": 0.015}

    def boom(a, b):
        raise AssertionError("a resumed look must not run the gate")

    table = (pd.DataFrame({"image": ["a", "b"], "patient": [1, 2]}), None, None, np.zeros((2, 15), np.float32), 0)
    kw = dict(load_fn=lambda *_a: (384, 384), table_fn=lambda s: table, read_fn=stub_read, numbers_fn=stub_numbers)
    sa = args(versions=[106, 118], read=197)

    def go(root, gate_fn, a=sa):
        old = os.environ.get("RAIDIOLOGY_FINAL_EVAL")
        try:
            return run_second(a, eval_dir=root, gate_fn=gate_fn, **kw)
        finally:
            if old is None:
                os.environ.pop("RAIDIOLOGY_FINAL_EVAL", None)
            else:
                os.environ["RAIDIOLOGY_FINAL_EVAL"] = old

    def fake(root, first=True):
        fake_eval(root)
        for v, ens in ((106, None), (118, None), (197, (106, 118))):
            os.makedirs(vdir(root, v), exist_ok=True)
            for k in range(FOLDS):
                open(os.path.join(vdir(root, v), "fold%d.pt" % k), "wb").close()
            json.dump({"ensemble_of": list(ens)} if ens else {}, open(os.path.join(vdir(root, v), "meta.json"), "w"))
        if first:
            json.dump({"versions": [103, 106], "read": 192, "time": "t0"}, open(marker_path(root), "w"))
            os.makedirs(os.path.join(root, "nih_test"), exist_ok=True)
            json.dump({"mean14": {"auroc": 0.79}, "time": "t1"}, open(os.path.join(root, "nih_test", "results.json"), "w"))

    # (g1) no finished first look refuses
    with tempfile.TemporaryDirectory() as root:
        fake(root, first=False)
        msg = refuses(lambda: go(root, gate("BETTER")))
        assert msg and "first look" in msg, msg
        assert not os.path.exists(look_marker_path(root, 2, 197))
    # (g2) the read of the last finished look refuses (the first look read 192)
    with tempfile.TemporaryDirectory() as root:
        fake(root)
        fake_eval(root, ens=(106, 118), read=192)
        msg = refuses(lambda: go(root, gate("BETTER"), args(versions=[106, 118], read=192)))
        assert msg and "already read" in msg, msg
    # (g3) an INCONCLUSIVE gate refuses, writes no look marker, records one gate refusal
    with tempfile.TemporaryDirectory() as root:
        fake(root)
        msg = refuses(lambda: go(root, gate("INCONCLUSIVE")))
        assert msg and "INCONCLUSIVE" in msg, msg
        assert not os.path.exists(look_marker_path(root, 2, 197))
        assert len(read_looks(root)["gate_refusals"]) == 1 and not read_looks(root)["looks"]
    # (g4) a pass writes marker, look folder results and finished; the same read again refuses
    with tempfile.TemporaryDirectory() as root:
        fake(root)
        go(root, gate("BETTER"))
        assert os.path.exists(look_marker_path(root, 2, 197))
        assert os.path.exists(os.path.join(root, "nih_test", "look2-v197", "results.json"))
        lk = read_looks(root)["looks"]
        assert len(lk) == 1 and lk[0]["look"] == 2 and lk[0]["finished"], lk
        msg = refuses(lambda: go(root, gate("BETTER")))
        assert msg and "already read" in msg, msg
    # (g5) an unfinished look is resumed without the gate and without a new marker
    with tempfile.TemporaryDirectory() as root:
        fake(root)
        json.dump({"look": 2, "versions": [106, 118], "read": 197, "time": "t2"}, open(look_marker_path(root, 2, 197), "w"))
        write_looks(root, {"looks": [{"look": 2, "versions": [106, 118], "read": 197, "opened": "t2", "finished": None,
                                      "gate": {"against": 192, "verdict": "BETTER", "gain": 0.01, "low": 0.0, "high": 0.02}}],
                           "gate_refusals": []})
        go(root, boom)
        assert read_looks(root)["looks"][0]["finished"]
        assert os.path.exists(os.path.join(root, "nih_test", "look2-v197", "results.json"))


def main():
    ap = argparse.ArgumentParser(description="Open NIH's test list once and score the reader fixed on development.")
    ap.add_argument("--versions", type=lambda s: [int(x) for x in s.split(",")], default=None)
    ap.add_argument("--read", type=int, default=192)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", choices=["cpu", "card"], default="cpu")
    ap.add_argument("--card-gb", type=float, default=1.5)
    ap.add_argument("--second-look", action="store_true", help="read the spent test list once more (gated on development)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.versions:
        raise SystemExit("--versions 103,106 is required")
    if a.second_look:
        sys.path.insert(0, HERE)
        run_second(a)
        return
    run(a)


if __name__ == "__main__":
    main()
