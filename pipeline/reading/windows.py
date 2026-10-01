"""Display windows for reader images, computed from the full 16-bit pixel data.

- standard: the whole series at its 0.5th-99.7th percentile (what render.py always wrote).
- fluid:    per slice, the body's 60th-99.9th percentile. Stretches the bright end, so fluid,
            a hydrated nucleus and a small bright annular fissure separate from each other.
- marrow:   per slice, the body's 5th-80th percentile. Stretches the middle, so bone marrow,
            endplates and Modic-type signal changes show their differences.
"Body" is the pixels above a tenth of the series' 99th percentile, so air does not decide the
window. Every window is a pure function of the pixels: no labels, no reading, nothing else."""
import numpy as np

NAMES = ("standard", "fluid", "marrow")


def to8(a, lo, hi):
    return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def series_window(vol):
    return float(np.percentile(vol, 0.5)), float(np.percentile(vol, 99.7))


def slice_windows(sl, vol_p99):
    body = sl[sl > 0.1 * vol_p99]
    if body.size < 100:
        body = sl.ravel()
    return {"fluid": (float(np.percentile(body, 60)), float(np.percentile(body, 99.9))),
            "marrow": (float(np.percentile(body, 5)), float(np.percentile(body, 80)))}


def render_all(vol):
    """{window name: [uint8 slice, ...]} and the windows used, for a (slices, rows, cols) volume."""
    vol = np.asarray(vol, np.float32)
    lo, hi = series_window(vol)
    p99 = float(np.percentile(vol, 99))
    out = {n: [] for n in NAMES}
    used = {"standard": [lo, hi], "fluid": [], "marrow": []}
    for sl in vol:
        out["standard"].append(to8(sl, lo, hi))
        w = slice_windows(sl, p99)
        for n in ("fluid", "marrow"):
            out[n].append(to8(sl, *w[n]))
            used[n].append([round(w[n][0], 1), round(w[n][1], 1)])
    return out, used
