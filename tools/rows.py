"""A trial's comparison rows as the markdown table docs/efficiency writes, from the ledger.

Reads ledger/experiment.jsonl: the latest row whose name contains each given text (seed 0's, then
seed 1's comparison, as compare.py wrote them, spine or X-ray). Per measure: A, B and B - A with
its 95% interval for each seed, the seeds' mean difference, and the noise over the square root of
two that the keep rules measure it against; then each comparison's verdict. One text gives a
one-seed table. Nothing is judged here: the rule written before the run decides.
Usage: python tools/rows.py "<text in seed 0's name>" ["<text in seed 1's name>"]"""
import json
import math
import os
import re
import sys

LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ledger", "experiment.jsonl")
DIFF = re.compile(r"^(.*?)(_b_minus_a|_minus_reference)$")


def find(text):
    rows = [json.loads(line) for line in open(LEDGER, encoding="utf-8") if line.strip()]
    hits = [r for r in rows if text in r["name"]]
    if not hits:
        raise SystemExit(f"no ledger row's name contains {text!r}")
    return hits[-1]


def measures(row):
    """{measure: (a, b, diff, low, high)} for every measure the comparison has both sides of."""
    n = {k: v["value"] for k, v in row["numbers"].items()}
    out = {}
    for k in n:
        m = DIFF.match(k)
        if m and f"{m.group(1)}_a" in n:
            p = m.group(1)
            out[p] = (n[f"{p}_a"], n[f"{p}_b"], n[k], n[f"{k}_low"], n[f"{k}_high"])
    return out


def noise(row):
    text = (row.get("settings") or {}).get("noise", "")
    return {k: float(v) for k, v in (kv.split("=") for kv in text.split(",") if "=" in kv)}


def main(texts):
    rows = [find(t) for t in texts]
    ms = [measures(r) for r in rows]
    nz = noise(rows[0])
    head = "| measure |"
    rule = "|---|"
    for i in range(len(rows)):
        head += f" A{i} | B{i} | B - A (95% CI) |"
        rule += "---:|---:|---|"
    if len(rows) > 1:
        head += " mean | noise / sqrt 2 |"
        rule += "---:|---:|"
    print("\n".join("- " + r["name"] for r in rows) + "\n")
    print(head + "\n" + rule)
    for m in ms[0]:
        line = f"| {m} |"
        for d in ms:
            a, b, diff, lo, hi = d[m]
            line += f" {a:.4f} | {b:.4f} | {diff:+.4f} ({lo:+.4f} to {hi:+.4f}) |"
        if len(rows) > 1:
            key = m.split(".")[0]
            bar = nz.get(key, nz.get(m))
            line += f" {sum(d[m][2] for d in ms) / len(ms):+.4f} | " + (f"{bar / math.sqrt(2):.4f} |" if bar is not None else "n/a |")
        print(line)
    print("\n" + "; ".join(f"seed {i}: {(r.get('settings') or {}).get('verdict', '?')}" for i, r in enumerate(rows)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
