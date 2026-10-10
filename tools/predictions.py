"""Score the predictions in docs/efficiency/PREDICTIONS.md against their results.

Each bar writes a point, an 80% range and a chance it is kept before it runs. This reads the table and prints:
one line per row (where the result fell against its range), how many judged results landed inside their
ranges (aim: about 8 in 10), the Brier score of the chances, and the ideas still open.

Run from anywhere (standard library only):
    python tools/predictions.py [--file PATH]
    python tools/predictions.py --selftest
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = ROOT / "tools" / "efficiency" / "PREDICTIONS.md"
COLUMNS = ["idea", "written", "measure", "point", "range", "chance", "result", "judged", "kept"]
NUM = r"[+-]?\d+(?:\.\d+)?"
RANGE_RE = re.compile(rf"^({NUM}) to ({NUM})$")


class BadCell(ValueError):
    pass


def _bad(idea, cell, text):
    return BadCell(f"idea {idea}: bad {cell} cell {text!r}")


def _number(idea, cell, text):
    if text == "":
        return None
    if not re.fullmatch(NUM, text):
        raise _bad(idea, cell, text)
    return float(text)


def parse(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(COLUMNS) or cells[0] == "idea" or set(cells[0]) <= set("-: "):
            continue
        raw = dict(zip(COLUMNS, cells))
        idea = raw["idea"]
        m = RANGE_RE.match(raw["range"])
        if not m:
            raise _bad(idea, "80% range", raw["range"])
        low, high = float(m.group(1)), float(m.group(2))
        chance = _number(idea, "chance kept", raw["chance"])
        if chance is not None and not 0 <= chance <= 1:
            raise _bad(idea, "chance kept", raw["chance"])
        if raw["kept"] not in ("yes", "no", ""):
            raise _bad(idea, "kept", raw["kept"])
        rows.append({
            "idea": idea,
            "measure": raw["measure"],
            "point": _number(idea, "point", raw["point"]),
            "range_text": raw["range"],
            "low": low,
            "high": high,
            "chance": chance,
            "result": _number(idea, "result", raw["result"]),
            "result_text": raw["result"],
            "judged": raw["judged"],
            "kept": raw["kept"],
        })
    return rows


def where(row):
    r = row["result"]
    if r is None:
        return "closed" if row["judged"] else "open"
    if r < row["low"]:
        return "below"
    if r > row["high"]:
        return "above"
    return "inside"


def brier(rows):
    pairs = [(r["chance"], 1.0 if r["kept"] == "yes" else 0.0)
             for r in rows if r["result"] is not None and r["chance"] is not None and r["kept"]]
    if not pairs:
        return None, 0
    return sum((c - k) ** 2 for c, k in pairs) / len(pairs), len(pairs)


def report(rows):
    out = []
    for r in rows:
        measure = r["measure"][:60]
        result = r["result_text"] or "-"
        out.append(f"{r['idea']}: {measure} | range {r['range_text']} | result {result} | {where(r)}")
    judged = [r for r in rows if r["result"] is not None]
    inside = sum(1 for r in judged if where(r) == "inside")
    out.append(f"ranges: {inside} of {len(judged)} judged results inside their 80% ranges (aim: about 8 in 10)")
    b, k = brier(rows)
    if k:
        out.append(f"chances: Brier {b:.3f} over {k} judged rows (0 is perfect; always 50% scores 0.25)")
    else:
        out.append("chances: none judged yet")
    open_ideas = [r["idea"] for r in rows if where(r) == "open"]
    out.append("open: " + (", ".join(open_ideas) if open_ideas else "none"))
    closed_ideas = [r["idea"] for r in rows if where(r) == "closed"]
    out.append("closed before measurement: " + (", ".join(closed_ideas) if closed_ideas else "none"))
    return "\n".join(out)


TOY = """| idea | written | measure | point | 80% range | chance kept | result | judged | kept |
|---|---|---|---|---|---|---|---|---|
| a | d | m | | +0.002 to +0.010 | 0.4 | +0.002 | d | no |
| b | d | m | | 0 to +0.010 | 0.8 | +0.010 | d | yes |
| c | d | m | | -0.02 to +0.05 | | -0.03 | d | |
| d | d | m | 1.1 | 0.90 to 1.40 | | 1.5 | d | |
| e | d | m | | 0.90 to 1.40 | 0.5 | | | |
| f | d | m | | 0.90 to 1.40 | 0.5 | | not opened: x | |
"""


def selftest():
    rows = {r["idea"]: r for r in parse(TOY)}
    assert (rows["a"]["low"], rows["a"]["high"]) == (0.002, 0.010)
    assert (rows["b"]["low"], rows["b"]["high"]) == (0.0, 0.010)
    assert (rows["c"]["low"], rows["c"]["high"]) == (-0.02, 0.05)
    assert (rows["d"]["low"], rows["d"]["high"]) == (0.90, 1.40)
    assert where(rows["a"]) == "inside" and where(rows["b"]) == "inside"
    assert where(rows["c"]) == "below" and where(rows["d"]) == "above"
    assert where(rows["e"]) == "open"
    assert where(rows["f"]) == "closed"
    b, k = brier(list(rows.values()))
    assert k == 2 and abs(b - 0.1) < 1e-9, (b, k)
    try:
        parse(TOY.replace("| 0.5 |", "| 1.2 |"))
    except BadCell as e:
        assert "idea e" in str(e), str(e)
    else:
        raise AssertionError("chance 1.2 was accepted")
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", default=str(DEFAULT_FILE))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return 0
    path = Path(args.file)
    if not path.is_absolute():
        path = ROOT / path
    try:
        rows = parse(path.read_text(encoding="utf-8"))
    except BadCell as e:
        print(e, file=sys.stderr)
        return 1
    print(report(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
