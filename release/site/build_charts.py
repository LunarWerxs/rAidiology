"""Build the charts and the stats band of the public pages from site/data/stats.json.

    python release/site/build_charts.py          write every <!-- chart:ID --> ... <!-- /chart:ID --> block in site/*.html
    python release/site/build_charts.py --check  write nothing; exit 1 if any page would change

Ids: band, overview, <model key>-headline, and every breakdown id in stats.json.
Output is plain inline SVG plus a <details> table, the same every run."""
import glob
import html
import json
import os
import re
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SITE = os.path.join(ROOT, "site")
STATS = json.load(open(os.path.join(SITE, "data", "stats.json"), encoding="utf-8"))

# Axis per metric: (min, max, ticks, panel title, note under the axis at the minimum)
AXES = {
    "AUROC": (0.5, 1.0, [0.5, 0.75, 1.0], "AUROC, 0.5 to 1", "coin toss (0.5)"),
    "Dice": (0.0, 1.0, [0, 0.5, 1.0], "Dice overlap, 0 to 1", None),
    "kappa": (0.0, 1.0, [0, 0.5, 1.0], "Kappa agreement, 0 to 1", "0 = chance agreement"),
    "percent": (0.0, 100.0, [0, 50, 100], "Percent, 0 to 100", None),
}
W = 440
X0, X1 = 18, 422
MARK = "#0e8f8f"
MIN_HALF = 9
INK = "#0b1a2b"
MUTED = "#536274"
GRID = "#cbd5e1"
LINE = "#e2e8f0"
FONT = "Inter, system-ui, sans-serif"
HALO = 'paint-order="stroke" stroke="#ffffff" stroke-width="4" stroke-linejoin="round"'


def esc(s):
    return html.escape(str(s), quote=True)


def fmt(metric, v):
    if v is None:
        return ""
    return f"{v:.1f}" if metric == "percent" else f"{v:.3f}"


def interval(metric, r):
    if r.get("low") is None or r.get("high") is None:
        return "not recorded"
    return f"{fmt(metric, r['low'])} to {fmt(metric, r['high'])}"


def value_text(metric, r):
    """Value with its interval beside it: (main, interval part or '')."""
    pct = "%" if metric == "percent" else ""
    main = fmt(metric, r["value"]) + pct
    if r.get("low") is None or r.get("high") is None:
        return main, ""
    return main, f" ({fmt(metric, r['low'])} to {fmt(metric, r['high'])}{pct})"


# Two layouts per chart. Narrow: 400 units wide, shown at phone width, text 19 units (13 CSS px at 375 px).
# Wide: 760 units, label left, track middle, value right.
LAYOUT = {
    "n": dict(W=400, fs=19, ax0=44, ax1=356, frame=(18, 382), wrap=33),
    "w": dict(W=760, fs=20, ax0=262, ax1=462, frame=(240, 484), wrap=22),
}


def panel(pid, metric, rows, title_extra="", mode="n"):
    """rows: dicts with label, value, low, high, held_out. One SVG in the given layout."""
    L = LAYOUT[mode]
    W, fs, ax0, ax1 = L["W"], L["fs"], L["ax0"], L["ax1"]
    lo_a, hi_a, ticks, ptitle, note = AXES[metric]
    sx = lambda v: ax0 + (min(max(v, lo_a), hi_a) - lo_a) / (hi_a - lo_a) * (ax1 - ax0)
    narrow = mode == "n"
    lh = fs + 6
    groups = [("Held-out test", [r for r in rows if r["held_out"]]), ("Development", [r for r in rows if not r["held_out"]])]
    laid, y = [], 44
    for gname, grows in groups:
        if not grows:
            continue
        laid.append(("g", gname, y))
        y += lh + 6
        for r in grows:
            lines = textwrap.wrap(r["label"], L["wrap"]) or [""]
            rh = lh * len(lines) + (lh + 30 if narrow else 14)
            if not narrow:
                rh = max(rh, 44)
            laid.append(("r", (r, lines), y))
            y += rh
    top, bottom = 34, y
    out = []
    for t in ticks:
        out.append(f'<line x1="{sx(t):.1f}" y1="{top}" x2="{sx(t):.1f}" y2="{bottom}" stroke="{GRID}" stroke-width="1"'
                   + (' stroke-dasharray="4 4"' if note and t == lo_a else "") + "/>")
    for kind, data, ry in laid:
        if kind == "g":
            out.append(f'<text x="{L["frame"][0] if narrow else 0}" y="{ry + fs}" font-size="{fs}" font-weight="700" fill="{MUTED}">{esc(data)}</text>')
            continue
        r, lines = data
        for i, ln in enumerate(lines):
            out.append(f'<text x="{0 if not narrow else L["frame"][0]}" y="{ry + fs + lh * i}" font-size="{fs}" fill="{INK}" {HALO}>{esc(ln)}</text>')
        main, iv = value_text(metric, r)
        tspan = f'<tspan font-weight="400" fill="{MUTED}">{esc(iv)}</tspan>' if iv else ""
        vtxt = f'<text @@ font-size="{fs}" font-weight="600" fill="{INK}" {HALO}>{esc(main)}{tspan}</text>'
        if narrow:
            vy = ry + lh * len(lines) + fs
            out.append(vtxt.replace('@@', f'x="{L["frame"][0]}" y="{vy}"'))
            cy = vy + 20
        else:
            cy = ry + 20
            out.append(vtxt.replace('@@', f'x="{L["frame"][1] + 14}" y="{cy + 6}"'))
        out.append(f'<line x1="{ax0}" y1="{cy}" x2="{ax1}" y2="{cy}" stroke="{LINE}" stroke-width="1"/>')
        v, lo, hi = r["value"], r.get("low"), r.get("high")
        if lo is not None and hi is not None:
            # Drawn ends sit at least MIN_HALF units from the dot so the caps stay visible; printed numbers stay exact.
            dl, dh = min(sx(lo), sx(v) - MIN_HALF), max(sx(hi), sx(v) + MIN_HALF)
            out.append(f'<line x1="{dl:.1f}" y1="{cy}" x2="{dh:.1f}" y2="{cy}" stroke="{MARK}" stroke-width="2.5"/>')
            for e in (dl, dh):
                out.append(f'<line x1="{e:.1f}" y1="{cy - 8}" x2="{e:.1f}" y2="{cy + 8}" stroke="{MARK}" stroke-width="2.5"/>')
        fill = MARK if r["held_out"] else "#ffffff"
        out.append(f'<circle cx="{sx(v):.1f}" cy="{cy}" r="3.2" fill="{fill}" stroke="{MARK}" stroke-width="2"/>')
    ay = bottom + 6
    out.append(f'<line x1="{L["frame"][0]}" y1="{ay}" x2="{L["frame"][1]}" y2="{ay}" stroke="{MUTED}" stroke-width="1"/>')
    for t in ticks:
        out.append(f'<text x="{sx(t):.1f}" y="{ay + fs + 4}" font-size="{fs}" fill="{MUTED}" text-anchor="middle">{t:g}</text>')
    yy = ay + fs + 4
    if note:
        out.append(f'<text x="{L["frame"][0] if narrow else 0}" y="{yy + lh}" font-size="{fs}" fill="{MUTED}">{esc(note)}</text>')
        yy += lh
    height = yy + 12
    out.insert(0, f'<text x="{L["frame"][0] if narrow else 0}" y="{fs + 2}" font-size="{fs}" font-weight="700" fill="{INK}">{esc(title_extra or ptitle)}</text>')
    desc = "Dot and interval chart. Filled dot: held-out test. Hollow dot: development. " + "; ".join(
        f"{r['label']}: {fmt(metric, r['value'])}" + (f" (95% interval {interval(metric, r)})" if r.get("low") is not None else "")
        for r in rows) + ". Research results on public test data."
    sid = pid if narrow else pid + "-w"
    return (f'<svg class="dc-{mode}" viewBox="0 0 {W} {height}" role="img" aria-labelledby="{sid}-t {sid}-d" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="{FONT}"><title id="{sid}-t">{esc(title_extra or ptitle)}</title><desc id="{sid}-d">{esc(desc)}</desc>'
            + "".join(out) + "</svg>")


def both(pid, metric, rows, title_extra=""):
    return panel(pid, metric, rows, title_extra, "n") + panel(pid, metric, rows, title_extra, "w")


def table(metric_rows, extra_col="Test set"):
    """metric_rows: list of (metric, row dict, set text)."""
    body = []
    for metric, r, setx in metric_rows:
        body.append("<tr><th scope=\"row\">" + esc(r["label"]) + f"</th><td>{esc(metric)}</td><td>{fmt(metric, r['value'])}</td>"
                    f"<td>{esc(interval(metric, r))}</td><td>{esc(setx)}</td><td>{'held-out test' if r['held_out'] else 'development'}</td></tr>")
    return ('<details class="chart-table"><summary>Show the numbers</summary><div class="chart-table-wrap"><table>'
            f"<thead><tr><th scope=\"col\">Result</th><th scope=\"col\">Metric</th><th scope=\"col\">Score</th>"
            f"<th scope=\"col\">95% interval</th><th scope=\"col\">{esc(extra_col)}</th><th scope=\"col\">Kind</th></tr></thead>"
            "<tbody>" + "".join(body) + "</tbody></table></div></details>")


def group_panels(cid, items):
    """items: list of (metric, row, set). One panel per metric, in a fixed order, then one table."""
    parts = []
    for metric in ("AUROC", "Dice", "kappa", "percent"):
        rows = [r for m, r, _ in items if m == metric]
        if rows:
            parts.append(f'<div class="chart-panel">{both(f"{cid}-{metric.lower()}", metric, rows)}</div>')
    n = len(parts)
    return f'<figure class="chart chart-n{n}">' + "".join(parts) + "</figure>" + table(items)


def overview():
    items = []
    for m in STATS["models"]:
        for h in m["headline"]:
            items.append((h["metric"], dict(h, label=f"{m['name']}: {h['label']}"), h["set"]))
    return group_panels("overview", items)


def home_headlines():
    """One headline result per model, for the home page."""
    items = []
    for m in STATS["models"]:
        h = m["headline"][0]
        items.append((h["metric"], dict(h, label=f"{m['name']}: {h['label']}"), h["set"]))
    return group_panels("home", items)


def headline(m):
    items = [(h["metric"], h, h["set"]) for h in m["headline"]]
    return group_panels(f"{m['key']}-headline", items)


def breakdown(b):
    rows = [dict(r, held_out=b["held_out"]) for r in b["rows"]]
    items = [(b["metric"], r, f"{b['set']}; {r['n']}" if r.get("n") else b["set"]) for r in rows]
    svg = both(b["id"], b["metric"], rows, b["title"])
    return f'<figure class="chart chart-n1"><div class="chart-panel">{svg}</div></figure>' + table(items, "Test set")


def band():
    cards = []
    for b in STATS["band"]:
        src = (b.get("source") or {}).get("file", "")
        key = os.path.splitext(os.path.basename(src))[0] if src.startswith("release/cards/") else ""
        href = f"models.html#{key}" if key else "models.html"
        cards.append(f'<li class="stat"><p class="stat-n">{esc(b["value"])}</p><p class="stat-l">{esc(b["label"])}</p>'
                     f'<p class="stat-d">{esc(b["detail"])}</p><a class="stat-src" href="{href}">Where this comes from</a></li>')
    return '<ul class="stats">' + "".join(cards) + "</ul>"


def build():
    out = {"band": band(), "overview": overview(), "home-headlines": home_headlines()}
    for m in STATS["models"]:
        out[f"{m['key']}-headline"] = headline(m)
        for b in m.get("breakdowns", []):
            out[b["id"]] = breakdown(b)
    return out


MARK_RE = re.compile(r"(<!-- chart:([\w-]+) -->)(.*?)(<!-- /chart:\2 -->)", re.S)


def main():
    check = "--check" in sys.argv
    blocks = build()
    changed = []
    for p in sorted(glob.glob(os.path.join(SITE, "*.html"))):
        text = open(p, encoding="utf-8", newline="").read()

        def sub(m):
            if m.group(2) not in blocks:
                raise SystemExit(f"{os.path.basename(p)}: unknown chart id {m.group(2)}")
            return f"{m.group(1)}\n{blocks[m.group(2)]}\n{m.group(4)}"

        new = MARK_RE.sub(sub, text)
        if new != text:
            changed.append(os.path.basename(p))
            if not check:
                open(p, "w", encoding="utf-8", newline="").write(new)
    if check and changed:
        print("would change:", ", ".join(changed))
        sys.exit(1)
    print("charts ok" + (f", wrote {', '.join(changed)}" if changed else ", nothing to change"))


if __name__ == "__main__":
    main()
