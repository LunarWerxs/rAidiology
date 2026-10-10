"""CT reading on a processor, one window at a time: the network and its softmax in PyTorch, as
`segmenter.py` reads, against the same fold's network as an OpenVINO graph with the softmax inside.

    python tools/ct_openvino_window.py [--version 1] [--fold 0] [--scans 2] [--windows 24]
                                       [--arms torch_bf16,ov_bf16,ov_f32,torch_f32]

The windows are real ones, cut exactly as `probabilities` cuts them, from the first `--scans`
held-out development scans of the fold, up to `--windows` of them. Every arm reads every window,
the arms taking turns window by window (a machine that gets busier slows them alike), after one
untimed read each to warm up. Each arm is compared with PyTorch in fp32 on the same window: the share
of voxels whose likeliest class differs, and the largest probability difference. A window is the
unit the 10x plan names (docs/efficiency/README.md, CT reading on a processor); the whole reading
and its Dice are `tools/ct_cpu_reading.py`'s job, once an arm here is worth it.

The PyTorch arms read in segmenter.py's processor layout (CPU_LAYOUT, plain unless set).
"""
import argparse
import os
import sys
import time

os.environ.setdefault("RAIDIOLOGY_DEVICE", "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "modalities", "ct", "eval"), ROOT]
# idle OpenMP threads sleep after 1 ms instead of spinning for 200 (docs/efficiency, 2026-10-03); read as torch loads
os.environ.setdefault("KMP_BLOCKTIME", "1")

import numpy as np  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every script here
import torch  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every script here
import torch.nn.functional as F  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every script here
import datasets  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every script here
import prepare  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every script here
import segmenter as s  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every script here
from core import ledger  # noqa: E402  arkitect-allow: test-floor-loosened - after the sys.path line, as in every script here

# ov_int8 and ov_int8_bf16 read the fold's int8 graph (tools/ct_openvino_int8.py), the rest of the
# graph in f32 or bf16; they are not in the default list because the graph has to be made first.
ARMS = ("torch_bf16", "ov_bf16", "ov_f32", "torch_f32", "ov_int8", "ov_int8_bf16")
DEFAULT_ARMS = "torch_bf16,ov_bf16,ov_f32,torch_f32"


def windows_of(scans, patch, limit):
    """The windows `probabilities` would read, as (1, 1, p, p, p) float32 arrays, scan by scan."""
    out = []
    for scan in scans:
        ct = torch.as_tensor(np.ascontiguousarray(np.load(prepare.lo_files(scan)[0])))
        x = s.to_input(ct)
        pad = [max(patch - v, 0) for v in x.shape]
        if any(pad):
            x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]), value=s.LOW_HU / s.HU_SCALE)
        for a in s._starts(x.shape[0], patch):
            for b in s._starts(x.shape[1], patch):
                for c in s._starts(x.shape[2], patch):
                    out.append(x[a:a + patch, b:b + patch, c:c + patch][None, None].contiguous().numpy())
                    if len(out) >= limit:
                        return out
    return out


def torch_arm(net, half):
    def read(w):
        x = torch.from_numpy(w).contiguous(memory_format=net.layout)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=half):
            logits = net(x)[0]
        return F.softmax(logits, dim=1, dtype=torch.float32)[0]
    return read


def ov_arm(net, patch, precision, saved=None, weights=None):
    """The reading's own GraphReader, so this bench and `probabilities` run the same graph."""
    t0 = time.time()
    reader = s.GraphReader(net, patch, precision, saved, s.sha256(weights) if saved else None)
    built = time.time() - t0
    about = {"convert_and_compile_s": built,
             "precision_used": str(reader.compiled.get_property("INFERENCE_PRECISION_HINT")),
             "threads": int(reader.compiled.get_property("INFERENCE_NUM_THREADS"))}
    return (lambda w: reader(torch.from_numpy(w))), about


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--scans", type=int, default=2)
    ap.add_argument("--windows", type=int, default=24)
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--graph", default="int8", help="the saved graph the ov_int8 arms read: fold<K>.<graph>.xml")
    a = ap.parse_args()
    arms = [v for v in a.arms.split(",") if v]
    if "torch_f32" not in arms:
        arms.append("torch_f32")              # the reference every arm is compared with
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit("unknown arm(s): %s; known: %s" % (", ".join(sorted(unknown)), ", ".join(ARMS)))
    datasets.guard_test("dev")
    s.steady_inference()
    models, meta = s.load_models(a.version, [a.fold])
    net, patch = models[0], meta["cfg"]["patch"]
    split = s.load_split()
    skipped = prepare.unprepared("dev")
    scans = sorted(k for k, f in split["dev_folds"].items() if f == a.fold and k not in skipped)[:a.scans]
    wins = windows_of(scans, patch, a.windows)
    readers, about = {}, {}
    weights = os.path.join(s.model_dir(a.version), "fold%d.pt" % a.fold)
    int8 = os.path.join(ROOT, "work", "ov", "v%d" % a.version, "fold%d.%s.xml" % (a.fold, a.graph))
    for arm in arms:
        if arm.startswith("torch"):
            readers[arm] = torch_arm(net, arm == "torch_bf16")
        elif arm.startswith("ov_int8"):
            readers[arm], about[arm] = ov_arm(net, patch, "bf16" if arm.endswith("bf16") else "f32", int8, weights)
        else:
            readers[arm], about[arm] = ov_arm(net, patch, arm.split("_")[1])
    with ledger.run("segmenter", "window on a processor v%d fold %d" % (a.version, a.fold)) as led:
        led.settings_from({"layout": s.CPU_LAYOUT, "arms": ",".join(arms), "windows": len(wins),
                           "threads": torch.get_num_threads()})
        for arm in arms:
            readers[arm](wins[0])             # warm-up, untimed
        seconds = {arm: [] for arm in arms}
        differ = {arm: [] for arm in arms}
        worst = {arm: 0.0 for arm in arms}
        for i, w in enumerate(wins):
            order = arms[i % len(arms):] + arms[:i % len(arms)]
            got = {}
            for arm in order:
                t0 = time.perf_counter()
                got[arm] = readers[arm](w)
                seconds[arm].append(time.perf_counter() - t0)
            ref = got["torch_f32"]
            best = ref.argmax(dim=0)
            for arm in arms:
                differ[arm].append(float((got[arm].argmax(dim=0) != best).float().mean()))
                worst[arm] = max(worst[arm], float((got[arm].float() - ref).abs().max()))
        base = float(np.median(seconds["torch_bf16"])) if "torch_bf16" in arms else None
        print("%d windows of %d scan(s), %s layout for PyTorch, %d threads" % (
            len(wins), len(scans), s.CPU_LAYOUT, torch.get_num_threads()))
        for arm in arms:
            med = float(np.median(seconds[arm]))
            led.number("%s_s_per_window" % arm, med, "s")
            led.number("%s_voxels_differing_from_fp32" % arm, float(np.mean(differ[arm])))
            led.number("%s_largest_probability_difference" % arm, worst[arm])
            extra = about.get(arm, {})
            print("  %-10s %.3f s a window (median; mean %.3f)%s, likeliest class differs from fp32 on %.5f%% "
                  "of voxels, largest probability difference %.4f%s" % (
                      arm, med, float(np.mean(seconds[arm])),
                      "" if base is None or arm == "torch_bf16" else ", %.2fx torch_bf16" % (base / med),
                      100 * float(np.mean(differ[arm])), worst[arm],
                      "" if not extra else "  [%s]" % ", ".join("%s %s" % (k, ("%.1f" % v) if isinstance(v, float) else v)
                                                              for k, v in extra.items())))


if __name__ == "__main__":
    main()
