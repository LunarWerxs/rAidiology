"""Write the research-results table in README.md from site/data/stats.json.

    python release/site/build_readme_table.py          rewrite the table between the markers
    python release/site/build_readme_table.py --check  exit 1 if README.md is out of date
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"


def result(h):
    unit = "%" if h["metric"] == "percent" else ""
    text = f"{h['value']}{unit}"
    if h.get("low") is not None and h.get("high") is not None:
        return f"{text} (95% interval {h['low']} to {h['high']}{unit})"
    return f"{text} (no interval recorded)"


def table():
    stats = json.loads((ROOT / "site/data/stats.json").read_text(encoding="utf-8"))
    rows = ["| Model | Scan | Measure | Result | Test set | Held-out or development |", "|---|---|---|---|---|---|"]
    for m in stats["models"]:
        h = m["headline"][0]
        measure = "share matched" if h["metric"] == "percent" else h["metric"]
        kind = "held-out" if h["held_out"] else "development"
        rows.append(f"| {m['name']} | {m['scan']} | {measure}: {h['label']} | {result(h)} | {h['set']} | {kind} |")
    return "\n".join(rows)


def render(text):
    pat = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pat.search(text):
        sys.exit("README.md has no results markers")
    return pat.sub(lambda _: f"{START}\n{table()}\n{END}", text)


text = README.read_text(encoding="utf-8")
new = render(text)
if "--check" in sys.argv:
    sys.exit(0 if new == text else "README.md results table is out of date")
README.write_text(new, encoding="utf-8")
