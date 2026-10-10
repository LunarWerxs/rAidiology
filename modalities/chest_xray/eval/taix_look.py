"""The labelled second look at TAIX-Ray's test split: one more opening, for a read that beats, on val and beyond noise, the
read the test split last saw. taix.test() refuses any second opening, so this file does it, reusing taix.py's functions.

  1. gate, before any test picture: the look-1 marker names --last's read, no look-2 marker exists, and taix.compare_sides
     on val says BETTER for --read against --last
  2. the look-2 marker (taix_test_look2.json) is written before any test picture is read
  3. each version of --read is read on the test split as taix.test reads it; mae.py's ViTs are rebuilt by taix.make_net
  4. the read is the mean of the versions' probabilities, as compare does; its logits are saved, and each version's too
  5. the printed result against the paper, then the paired comparison against look 1's saved test logits (reported only)
  6. a ledger row

  second-look --read "1,2,3,4,6,7|14" --last "1,2,3,4,6,7" [--margin 0.01]
  selftest                 CPU, temp folders; never touches work/ or the ledger

--read and --last take taix.parse_side's syntax (groups split by '|', versions by ',').
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import taix  # noqa: E402  (read, never edited: a training queue imports it)
import reader  # noqa: E402
from core import ledger  # noqa: E402


def look2_path(cache):
    return os.path.join(cache, "taix_test_look2.json")


def look_dir(versions, cache):
    """Where one version's, or a read's, test logits are kept under cache: taix.test's own rule."""
    if len(versions) == 1:
        return os.path.join(cache, "taix-v%d" % versions[0])
    return os.path.join(cache, "taix-read-%s" % taix.read_label(versions))


def load_npz(path):
    """An npz's arrays as a dict, the file closed (an open one blocks a temp folder's cleanup on Windows)."""
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def flat(groups):
    return [v for g in groups for v in g]


def print_compare(b, a, rows, verdict, n_pic, n_pat, margin, n, seed):
    """compare_run's printout for the gate: side B against side A on val, the rows, the verdict line."""
    print("val, B (%s) against A (%s): %d pictures, %d patients; paired patient bootstrap, %d resamples, seed %d\n"
          % (b, a, n_pic, n_pat, n, seed))
    print("| label | A | B | B - A (95% CI) |\n|---|---|---|---|")
    for lab, va, vb, d, lo, hi in rows:
        print("| %s | %.4f | %.4f | %+.4f (%+.4f to %+.4f) |" % (lab, va, vb, d, lo, hi))
    print("\nVerdict (mean of the eight, margin %.3f): %s" % (margin, verdict), flush=True)


def second_look(read, last, margin=0.01, cache=taix.CACHE, marker=taix.MARKER, look2=None, n=taix.BOOT, seed=taix.BOOT_SEED):
    """Opens TAIX-Ray's test split once more for read, when the val gate against last says BETTER. The look-2 folder is
    cache/taix-look2-<label>; stops before any test picture is read unless every check passes."""
    import torch
    look2 = look2 or look2_path(cache)
    groups, last_groups = taix.parse_side(read), taix.parse_side(last)
    read_v, last_v = flat(groups), flat(last_groups)
    label = "_".join("-".join(str(v) for v in g) for g in groups)
    # 1a. the look-1 marker names the read given as --last
    if not os.path.exists(marker):
        raise SystemExit("no look 1: %s is missing, so the test split was not opened for %s" % (marker, taix.read_label(last_v)))
    with open(marker, encoding="utf-8") as f:
        opened = str(json.load(f).get("version"))
    if opened != taix.read_label(last_v):
        raise SystemExit("the look-1 marker names %s, not %s (--last): the second look is for the read that opened the test split"
                         % (opened, taix.read_label(last_v)))
    # 1b. one second look per test split, whatever read it was for
    if os.path.exists(look2):
        with open(look2, encoding="utf-8") as f:
            held = json.load(f).get("read")
        raise SystemExit("a second look is already on record (read %s, %s); one per test split" % (held, look2))
    # 1c. the val gate, side B (--read) against side A (--last), as compare does it
    rows, verdict, n_pic, n_pat = taix.compare_sides(last, read, margin, cache=cache, n=n, seed=seed)
    print_compare(read, last, rows, verdict, n_pic, n_pat, margin, n, seed)
    if verdict != "BETTER":
        raise SystemExit("no look: the gate says %s" % verdict)
    print("gate: BETTER", flush=True)
    for v in read_v:
        if not os.path.exists(os.path.join(look_dir([v], cache), "model.pt")):
            raise SystemExit("no trained model at %s" % os.path.join(look_dir([v], cache), "model.pt"))
    # 2. the look-2 marker goes down before any test picture is read ('x' fails if one exists)
    gate = rows[-1]
    os.makedirs(os.path.dirname(look2), exist_ok=True)
    with open(look2, "x", encoding="utf-8") as f:
        json.dump({"read": read, "last": last, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "gate": {"mean8_diff": float(gate[3]), "lo": float(gate[4]), "hi": float(gate[5])}}, f)
    print("look-2 marker written: %s" % look2, flush=True)
    # 3. each version's test rows, read as taix.test reads them
    taix.setup_card(reader, "taix second look v%s" % label)
    tables, per, ids = {}, {}, None
    for v in read_v:
        ck = torch.load(os.path.join(look_dir([v], cache), "model.pt"), map_location="cpu", weights_only=False)
        settings, info = ck["settings"], ck["info"]
        cs = taix.cache_size_of(settings)
        if cs not in tables:
            tables[cs] = taix.load_table(settings)
        t, pix = tables[cs]
        test_rows = np.flatnonzero(t["split"] == "test")
        net = taix.make_net(reader, ck["cfg"], settings).to(reader.DEV)
        net.load_state_dict(ck["state"])
        mean = torch.tensor(info["mean"], dtype=torch.float32, device=reader.DEV)
        sd = torch.tensor(info["sd"], dtype=torch.float32, device=reader.DEV)
        z = taix.predict(reader, net, pix, t["truth"], test_rows, mean, sd, ck["cfg"].get("batch", 64), taix.read_px(settings),
                         taix.n_outputs(settings))
        per[v] = taix.present_logits(z, bool(settings.get("ordinal")))
        now = (t["uid"][test_rows], t["patient"][test_rows], t["truth"][test_rows])
        if ids is None:
            ids = now
        elif not all(np.array_equal(a, b) for a, b in zip(ids, now)):
            raise SystemExit("version %d's test rows differ from version %d's: not the same pictures" % (v, read_v[0]))
        del net
    # 4. the read: the mean of the versions' probabilities, as compare's side does it
    uid, patient, truth = ids
    logits = taix.prob_logit(taix.side_probs(groups, {v: {"logits": per[v]} for v in read_v}))
    # 5. saved: the read's logits, and each version's present logits in the same UID order
    out = os.path.join(cache, "taix-look2-" + label)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "test_logits.npz"), UID=uid, PatientID=patient, logits=logits, truth=truth)
    for v in read_v:
        np.savez(os.path.join(out, "test_logits_v%d.npz" % v), UID=uid, PatientID=patient, logits=per[v], truth=truth)
    # 6. the result against the paper, as taix.test prints it
    res = taix.bootstrap_auroc(logits, truth, patient, n, seed)
    tally = {"BEATEN": 0, "LEVEL": 0, "BEHIND": 0}
    print("look 2 (labelled second look) of TAIX-Ray's test split: read %s" % read)
    print("test AUROC against the paper (Truhn et al., Sci Data 2026, Table 4; patient bootstrap, %d resamples, seed %d)"
          % (n, seed))
    for name, (p, lo, hi), paper in zip(taix.NAMES, res, taix.PAPER):
        j = taix.judge(lo, hi, paper)
        tally[j] += 1
        print("  %-17s ours %.4f (95%% interval %.4f to %.4f)  paper %.2f  %s" % (name, p, lo, hi, paper, j), flush=True)
    m = float(np.mean([r[0] for r in res]))
    line = "of eight: beaten %d, level %d, behind %d; mean ours %.4f against the paper's %.4f" % (
        tally["BEATEN"], tally["LEVEL"], tally["BEHIND"], m, taix.PAPER.mean())
    print(line, flush=True)
    # 7. reported, never a verdict: look 2's read against look 1's saved test logits (the same pictures, in the same order)
    prev = load_npz(os.path.join(look_dir(last_v, cache), "test_logits.npz"))
    if not (np.array_equal(prev["UID"], uid) and np.array_equal(prev["truth"], truth)):
        raise SystemExit("look 1's saved test rows differ from look 2's (UID order or truth): the paired comparison is not made")
    pts_a, pts_b, diffs = taix.paired_auroc(prev["logits"], logits, truth, patient, n, seed)
    lo8, hi8 = np.nanpercentile(diffs, [2.5, 97.5], axis=0)
    print("\nreported: look 2's read against look 1's read on the test split")
    print("| label | look 1 | look 2 | look 2 - look 1 (95% CI) |\n|---|---|---|---|")
    for lab, va, vb, lo, hi in zip(taix.NAMES + ["mean of the eight"], pts_a, pts_b, lo8, hi8):
        print("| %s | %.4f | %.4f | %+.4f (%+.4f to %+.4f) |" % (lab, va, vb, vb - va, lo, hi))
    # 8. the ledger row, as taix.test writes it
    with ledger.run("experiment", "taix test look 2 v" + label) as led:
        led.settings_from({"read": read, "last": last, "margin": margin})
        for name, (p, lo, hi) in zip(taix.NAMES, res):
            led.number("test.auroc.%s" % name, p).number("test.lo.%s" % name, lo).number("test.hi.%s" % name, hi)
        led.number("test.auroc.mean8", m)
        for k, c in tally.items():
            led.number("test.%s" % k.lower(), c, "count")
        led.number("look1.diff.mean8", float(pts_b[-1] - pts_a[-1]))
        led.number("look1.lo.mean8", float(lo8[-1])).number("look1.hi.mean8", float(hi8[-1]))
        led.note_is(line)


# ---------------------------------------------------------------- selftest

class _FakeRun:
    """core.ledger.run's stand-in in the selftest: the numbers go to a dict, nothing to the repository's ledger."""

    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def number(self, key, value, unit=""):
        self.log[key] = float(value)
        return self

    def settings_from(self, mapping):
        self.log["settings"] = mapping
        return self

    def note_is(self, text):
        self.log["note"] = text
        return self


def selftest():
    import contextlib
    import io
    import tempfile
    import torch

    class Net:
        """taix.make_net's stand-in: nothing to build or load."""

        def to(self, dev):
            return self

        def load_state_dict(self, state):
            pass

    rng = np.random.default_rng(7)
    uid = np.array(["u%d" % i for i in range(400)])
    patient = np.array(["v%d" % (i // 4) for i in range(200)] + ["t%d" % (i // 4) for i in range(200)])
    split = np.array(["val"] * 200 + ["test"] * 200)
    truth = (rng.random((400, 8)) < 0.3).astype(np.int8)
    val, test = np.flatnonzero(split == "val"), np.flatnonzero(split == "test")
    table = {"uid": uid, "patient": patient, "split": split, "truth": truth}

    def logits_of(v, rows):
        """v3 reads the truth (a perfect net), v4 reads it backwards (a worse one), the rest are noise seeded by v."""
        if v == 3:
            return truth[rows] * 6.0 - 3.0
        if v == 4:
            return 3.0 - truth[rows] * 6.0
        return np.random.default_rng(100 + v).normal(size=(len(rows), 8))

    def fixture(d, marker_version="1-2"):
        """A cache folder: versions 1-4's val logits and model files (info mean = the version, the predict stub's key), and
        the look-1 marker and look-1 test logits of read 1,2. Returns the marker's path."""
        for v in (1, 2, 3, 4):
            os.makedirs(os.path.join(d, "taix-v%d" % v))
            np.savez(os.path.join(d, "taix-v%d" % v, "val_logits.npz"), UID=uid[val], PatientID=patient[val],
                     logits=logits_of(v, val), truth=truth[val])
            torch.save({"state": {}, "cfg": {}, "info": {"mean": float(v), "sd": 1.0}, "settings": {}},
                       os.path.join(d, "taix-v%d" % v, "model.pt"))
        os.makedirs(os.path.join(d, "taix-read-1-2"))
        np.savez(os.path.join(d, "taix-read-1-2", "test_logits.npz"), UID=uid[test], PatientID=patient[test],
                 logits=rng.normal(size=(len(test), 8)), truth=truth[test])
        marker = os.path.join(d, "taix_test_opened.json")
        with open(marker, "w", encoding="utf-8") as f:
            json.dump({"version": marker_version, "time": "2026-10-06 12:00:00"}, f)
        return marker

    def refuses(text, **kw):
        try:
            second_look(**kw)
        except SystemExit as e:
            assert text in str(e), (text, str(e))
        else:
            raise AssertionError("second_look let through: " + text)

    log = {}
    calls = []
    saved = {k: getattr(taix, k) for k in ("load_table", "make_net", "setup_card", "predict")}
    saved_run = ledger.run

    def predict_for(look2):
        def predict(reader_, net, pix, truth_, rows, mean, sd, batch, px, n_out=8):
            assert os.path.exists(look2), "a test picture was read before the look-2 marker was written"
            calls.append(float(mean))
            return logits_of(int(round(float(mean))), rows)
        return predict

    try:
        taix.load_table = lambda settings=None: (table, None)
        taix.make_net = lambda reader_, cfg, settings: Net()
        taix.setup_card = lambda reader_, name: None
        ledger.run = lambda kind, name, note="": _FakeRun(log)
        # (i) no look-1 marker: refused, nothing written
        with tempfile.TemporaryDirectory() as d:
            fixture(d)
            look2 = look2_path(d)
            taix.predict = predict_for(look2)
            refuses("no look 1", read="1,2|3", last="1,2", cache=d, marker=os.path.join(d, "missing.json"), n=60)
            assert not os.path.exists(look2)
        # (ii) the look-1 marker names another read: refused
        with tempfile.TemporaryDirectory() as d:
            marker = fixture(d, marker_version="1-3")
            look2 = look2_path(d)
            taix.predict = predict_for(look2)
            refuses("names 1-3, not 1-2", read="1,2|3", last="1,2", cache=d, marker=marker, n=60)
            assert not os.path.exists(look2)
        # (iii) the gate says WORSE: refused, nothing written
        with tempfile.TemporaryDirectory() as d:
            marker = fixture(d)
            look2 = look2_path(d)
            taix.predict = predict_for(look2)
            refuses("no look: the gate says", read="1,2|4", last="1,2", cache=d, marker=marker, n=60)
            assert not os.path.exists(look2)
        # (iv) a look-2 marker exists: refused, and it is not overwritten
        with tempfile.TemporaryDirectory() as d:
            marker = fixture(d)
            look2 = look2_path(d)
            with open(look2, "w", encoding="utf-8") as f:
                json.dump({"read": "1,2,3,4|9", "last": "1,2", "time": "2026-10-07 09:00:00", "gate": {}}, f)
            taix.predict = predict_for(look2)
            refuses("already on record (read 1,2,3,4|9", read="1,2|3", last="1,2", cache=d, marker=marker, n=60)
            with open(look2, encoding="utf-8") as f:
                assert json.load(f)["read"] == "1,2,3,4|9", "the look-2 marker was overwritten"
        # (v) and (vi): a BETTER gate writes the marker before any test picture, saves the logits, and reports the paired look
        with tempfile.TemporaryDirectory() as d:
            marker = fixture(d)
            look2 = look2_path(d)
            taix.predict = predict_for(look2)
            log.clear()
            calls.clear()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                second_look(read="1,2|3", last="1,2", cache=d, marker=marker, n=60)
            text = buf.getvalue()
            assert sorted(calls) == [1.0, 2.0, 3.0], calls
            with open(look2, encoding="utf-8") as f:
                m_json = json.load(f)
            assert m_json["read"] == "1,2|3" and m_json["last"] == "1,2" and set(m_json["gate"]) == {"mean8_diff", "lo", "hi"}
            assert m_json["gate"]["lo"] > 0, m_json["gate"]
            out = os.path.join(d, "taix-look2-1-2_3")
            z = load_npz(os.path.join(out, "test_logits.npz"))
            assert set(z) == {"UID", "PatientID", "logits", "truth"}, sorted(z)
            per = {v: load_npz(os.path.join(out, "test_logits_v%d.npz" % v))["logits"] for v in (1, 2, 3)}
            want = taix.prob_logit(taix.side_probs([[1, 2], [3]], {v: {"logits": per[v]} for v in (1, 2, 3)}))
            assert np.allclose(z["logits"], want), "the read's logits are not the combined per-version logits"
            assert "look 2 (labelled second look) of TAIX-Ray's test split: read 1,2|3" in text, text
            assert "gate: BETTER" in text and "reported: look 2's read against look 1's read on the test split" in text
            paired = taix.paired_auroc(load_npz(os.path.join(d, "taix-read-1-2", "test_logits.npz"))["logits"], z["logits"],
                                       z["truth"], z["PatientID"], 60, taix.BOOT_SEED)
            assert abs(log["look1.diff.mean8"] - (paired[1][-1] - paired[0][-1])) < 1e-9, (log.get("look1.diff.mean8"), paired[0][-1])
            assert "test.auroc.mean8" in log and "test.beaten" in log and "look1.hi.mean8" in log, sorted(log)
    finally:
        for k, v in saved.items():
            setattr(taix, k, v)
        ledger.run = saved_run
    print("taix_look selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("second-look")
    s.add_argument("--read", required=True, help='the read to look at again, e.g. "1,2,3,4,6,7|14"')
    s.add_argument("--last", required=True, help='the read the test split last saw, e.g. "1,2,3,4,6,7"')
    s.add_argument("--margin", type=float, default=0.01)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        second_look(a.read, a.last, a.margin)


if __name__ == "__main__":
    main()
