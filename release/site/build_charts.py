"""Build the result charts, the result tiles and the stats band of the public pages from site/data/stats.json.

    python release/site/build_charts.py          write every <!-- chart:ID --> ... <!-- /chart:ID --> block in site/*.html
    python release/site/build_charts.py --check  write nothing; exit 1 if any page would change

Ids: band, tiles, <model key>-headline, <model key>-numbers, and every breakdown id in stats.json.
Chart text is HTML at fixed sizes (css/results.css); only the dot-and-interval tracks stretch with the card.
Scores print exactly as stats.json holds them (at least three decimals, percents to one), the same every run."""
import glob
import html
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SITE = os.path.join(ROOT, "site")
STATS = json.load(open(os.path.join(SITE, "data", "stats.json"), encoding="utf-8"))

# Axis per metric: (min, max, ticks, name, what the scale means, chance level drawn dashed or None)
AXES = {
    "AUROC": (0.5, 1.0, [0.5, 0.75, 1.0], "AUROC", "0.5 is a coin toss, 1 is perfect", 0.5),
    "Dice": (0.0, 1.0, [0, 0.5, 1.0], "Dice", "overlap with the dataset's own outlines, 1 is a perfect match", None),
    "kappa": (0.0, 1.0, [0, 0.5, 1.0], "Kappa", "0 is chance agreement, 1 is full agreement", 0.0),
    "percent": (0.0, 100.0, [0, 50, 100], "Percent", "0 to 100", None),
}
ORDER = ("AUROC", "Dice", "kappa", "percent")
# The dot only repeats the words beside it ("Held-out test", "Development"), so screen readers skip it.
GLYPH = {
    True: '<svg class="dot-key" viewBox="0 0 12 12" width="12" height="12" aria-hidden="true" focusable="false">'
          '<circle cx="6" cy="6" r="4.5"/></svg>',
    False: '<svg class="dot-key hollow" viewBox="0 0 12 12" width="12" height="12" aria-hidden="true" focusable="false">'
           '<circle cx="6" cy="6" r="4.5"/></svg>',
}
KIND = {True: "Held-out test", False: "Development"}


def esc(s):
    return html.escape(str(s), quote=True)


def text(s):
    """Escaped text in which a hyphenated word (X-ray, GRAZPEDWRI-DX's) never breaks at its hyphen."""
    return re.sub(r"[\w'’]+(?:-[\w'’]+)+", lambda m: f'<span class="nb">{m.group(0)}</span>',
                  html.escape(str(s), quote=False))


def webp_size(rel):
    """(width, height) of a WebP under site/, read from its header."""
    b = open(os.path.join(SITE, rel), "rb").read(30)
    if b[12:16] == b"VP8X":
        return 1 + int.from_bytes(b[24:27], "little"), 1 + int.from_bytes(b[27:30], "little")
    if b[12:16] == b"VP8L":
        bits = int.from_bytes(b[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if b[12:16] == b"VP8 ":
        return int.from_bytes(b[26:28], "little") & 0x3FFF, int.from_bytes(b[28:30], "little") & 0x3FFF
    raise SystemExit(f"{rel}: not a WebP this script can read")


def decimals(v):
    return len(repr(float(v)).partition(".")[2]) if v is not None else 0


def places(metric, rows):
    """Decimals for a set of results: percents to one; other scores to the most any of them carries, at least three,
    so a column of scores lines up the way the model cards print them (0.7610, not 0.761, beside 0.7459)."""
    if metric == "percent":
        return 1
    return max([3] + [decimals(r.get(k)) for r in rows for k in ("value", "low", "high")])


def fmt(metric, v, dp=None):
    """The number as stats.json holds it, padded with zeros to dp decimals."""
    if v is None:
        return ""
    dp = places(metric, [{"value": v}]) if dp is None else dp
    return f"{v:.{dp}f}"


def has_interval(r):
    return r.get("low") is not None and r.get("high") is not None


def interval(metric, r, dp=None):
    return f"{fmt(metric, r['low'], dp)} to {fmt(metric, r['high'], dp)}" if has_interval(r) else "not recorded"


def value_text(metric, r, dp=None):
    """(score, interval in brackets or ''), to dp decimals (default: as many as the row itself needs)."""
    dp = places(metric, [r]) if dp is None else dp
    pct = "%" if metric == "percent" else ""
    main = fmt(metric, r["value"], dp) + pct
    return main, (f"({fmt(metric, r['low'], dp)} to {fmt(metric, r['high'], dp)}{pct})" if has_interval(r) else "")


def pos(metric, v):
    lo_a, hi_a = AXES[metric][0], AXES[metric][1]
    return round((min(max(v, lo_a), hi_a) - lo_a) / (hi_a - lo_a) * 100, 2)


def num(x):
    return f"{x:g}"


def chart_grid(metric, rows, note_key):
    """One metric's rows: a header (scale, ticks, value heading), then a group per kind (held-out first).
    rows: dicts with label, value, low, high, held_out, and a note under the label (note_key)."""
    lo_a, hi_a, ticks, name, scale, chance = AXES[metric]
    dp = places(metric, rows)
    out = ['<div class="rc-grid">', '<div class="rc-hd">',
           f'<p class="rc-scale"><strong>{esc(name)}</strong> {text(scale)}</p>',
           '<p class="rc-ax" aria-hidden="true">' + "".join(
               f'<span style="left:{num(pos(metric, t))}%">{num(t)}</span>' for t in ticks) + "</p>",
           '<p class="rc-vh">Score <span>(95% interval)</span></p>', "</div>"]
    dashed = ' class="c"'
    grid = "".join(f'<i{dashed if chance is not None and t == chance else ""} style="left:{num(pos(metric, t))}%"></i>'
                   for t in ticks)
    for held in (True, False):
        group = [r for r in rows if r["held_out"] == held]
        if not group:
            continue
        out.append(f'<p class="rc-grp">{GLYPH[held]}{KIND[held]}</p>')
        for r in group:
            main, iv = value_text(metric, r, dp)
            note = r.get(note_key)
            note_html = f'<span class="{"rc-set" if note_key == "set" else "rc-n"}">{text(note)}</span>' if note else ""
            var = f"--v:{num(pos(metric, r['value']))}"
            if has_interval(r):
                var += f";--lo:{num(pos(metric, r['low']))};--hi:{num(pos(metric, r['high']))}"
            out.append(f'<div class="rc-row{"" if held else " dev"}{"" if has_interval(r) else " no-iv"}">'
                       f'<p class="rc-l">{text(r["label"])}{note_html}</p>'
                       f'<span class="rc-t" style="{var}" aria-hidden="true">{grid}</span>'
                       f'<p class="rc-v"><strong>{esc(main)}</strong>{(" <span>" + esc(iv) + "</span>") if iv else ""}</p></div>')
    out.append("</div>")
    return "".join(out)


def chart(title, sub, items, note_key):
    """items: rows carrying their own metric. One card, one grid per metric in a fixed order."""
    grids = [chart_grid(m, [r for r in items if r["metric"] == m], note_key) for m in ORDER
             if any(r["metric"] == m for r in items)]
    cap = f"<h3>{text(title)}</h3>" + (f"<p>{text(sub)}</p>" if sub else "")
    return f'<figure class="rc"><figcaption class="rc-cap">{cap}</figcaption>' + "".join(grids) + "</figure>"


def headline(m):
    return chart("Main results", "", [dict(h) for h in m["headline"]], "set")


def breakdown(b):
    rows = [dict(r, metric=b["metric"], held_out=b["held_out"]) for r in b["rows"]]
    return chart(b["title"], f"Test set: {b['set']}", rows, "n")


def numbers(m):
    """Every result of one model as a table, behind "Show the numbers"."""
    head = ('<thead><tr><th scope="col">Result</th><th scope="col">Metric</th><th scope="col">Score</th>'
            '<th scope="col">95% interval</th><th scope="col">Kind</th><th scope="col">Test set</th></tr></thead>')

    def row(metric, r, setx, held, dp):
        return (f'<tr><th scope="row">{esc(r["label"])}</th><td>{esc(AXES[metric][3])}</td>'
                f'<td>{fmt(metric, r["value"], dp)}{"%" if metric == "percent" else ""}</td>'
                f'<td>{esc(interval(metric, r, dp))}</td><td>{KIND[held]}</td><td>{esc(setx)}</td></tr>')

    hl = m["headline"]
    dps = {mt: places(mt, [h for h in hl if h["metric"] == mt]) for mt in ORDER}
    bodies = ['<tbody><tr class="grp"><th scope="colgroup" colspan="6">Main results</th></tr>'
              + "".join(row(h["metric"], h, h["set"], h["held_out"], dps[h["metric"]]) for h in hl) + "</tbody>"]
    for b in m.get("breakdowns", []):
        dp = places(b["metric"], b["rows"])
        bodies.append(f'<tbody><tr class="grp"><th scope="colgroup" colspan="6">{esc(b["title"])}</th></tr>'
                      + "".join(row(b["metric"], r, f"{b['set']}; {r['n']}" if r.get("n") else b["set"], b["held_out"], dp)
                                for r in b["rows"]) + "</tbody>")
    return ('<details class="rnums"><summary>Show the numbers</summary><div class="rnums-wrap"><table>'
            f'<caption class="sr">{esc(m["name"])}: every result</caption>' + head + "".join(bodies)
            + "</table></div></details>")


PLAIN = {
    "chest-xray": "Scores one chest X-ray for 14 kinds of label from a public dataset.",
    # 117, not the headline's {n}: the model outlines 117 structures and its mean covers the 115 common enough in the test
    # scans to score (release/cards/ct-segmenter.md line 30; models.html says both, side by side).
    "ct-segmenter": "Outlines and names 117 body structures on CT.",
    "disc-grader": "Grades the discs of a spine MRI on a public five-step scale.",
    "fracture-detector": "Scores a child's wrist X-ray for a fracture label from a public dataset.",
    "lung-nodules": "Marks small lung nodules on CT.",
    "mammography": "Scores one mammogram picture as benign or malignant against a public dataset.",
    "spine-xray-corners": "Marks the corners of the lower back's vertebrae on a side-view spine X-ray.",
}
# The size of the set each headline was measured on, read from its own description ("25,596 pictures").
SET_SIZE = re.compile(r"([\d,]+)\s+(?:[\w-]+\s+){0,2}?(pictures|scans|discs|patients)\b")


def tiles():
    """One tile per model, in plain words (no metric names, intervals or scores; the size of the set it was measured on);
    each opens that model's section on models.html."""
    out = []
    for m in STATS["models"]:
        h = m["headline"][0]
        n = re.search(r"\d+", h["label"])
        what = PLAIN[m["key"]].format(n=n.group(0) if n else "")
        size = SET_SIZE.search(h["set"])
        # A held-out result says in plain words what the tag means; a development one is only "measured".
        head = ((f"Tested on {size.group(1)} {size.group(2)} it never saw" if h["held_out"]
                 else f"Measured on {size.group(1)} {size.group(2)}") if size else m["scan"])
        tag = (f'<span class="tag">{GLYPH[True]}{KIND[True]}</span>' if h["held_out"]
               else f'<span class="tag dev">{GLYPH[False]}Development</span>')
        w, h_ = webp_size(m["picture"])
        out.append(f'<a class="result-tile" href="models.html#{esc(m["key"])}">'
                   f'<img src="{esc(m["picture"])}" width="{w}" height="{h_}" alt="" loading="lazy">'
                   f'<span class="pic-chip">{esc(m["scan"])}</span>'
                   f'<h3>{text(m["name"])}</h3>'
                   f'<p class="score">{esc(head)}</p>'
                   f'<p class="what">{text(what)}</p>{tag}</a>')
    return '<div class="result-tiles">' + "".join(out) + "</div>"


def band():
    """The numbers band, with one source link under it (a link on every card read as clutter in the design review)."""
    cards = [f'<li class="stat"><p class="stat-n">{esc(b["value"])}</p><p class="stat-l">{text(b["label"])}</p>'
             f'<p class="stat-d">{text(b["detail"])}</p></li>' for b in STATS["band"]]
    return ('<ul class="stats">' + "".join(cards) + '</ul>'
            '<p class="band-src"><a href="models.html">Where these numbers come from</a></p>')


def build():
    for metric, (lo_a, hi_a, ticks, *_rest) in AXES.items():
        assert ticks[0] == lo_a and ticks[-1] == hi_a, f"{metric}: ticks must start and end at the axis ends"
    out = {"band": band(), "tiles": tiles()}
    for m in STATS["models"]:
        out[f"{m['key']}-headline"] = headline(m)
        out[f"{m['key']}-numbers"] = numbers(m)
        for b in m.get("breakdowns", []):
            out[b["id"]] = breakdown(b)
    return out


MARK_RE = re.compile(r"(<!-- chart:([\w-]+) -->)(.*?)(<!-- /chart:\2 -->)", re.S)


def main():
    check = "--check" in sys.argv
    blocks = build()
    pages = {}
    for p in sorted(glob.glob(os.path.join(SITE, "*.html"))):
        text_ = open(p, encoding="utf-8", newline="").read()

        def sub(m):
            if m.group(2) not in blocks:
                raise SystemExit(f"{os.path.basename(p)}: unknown chart id {m.group(2)}")
            return f"{m.group(1)}\n{blocks[m.group(2)]}\n{m.group(4)}"

        new = MARK_RE.sub(sub, text_)
        if new != text_:
            pages[p] = new
    if check and pages:
        print("would change:", ", ".join(os.path.basename(p) for p in pages))
        sys.exit(1)
    if not check:
        for p, new in pages.items():
            open(p, "w", encoding="utf-8", newline="").write(new)
    print("charts ok" + (f", wrote {', '.join(os.path.basename(p) for p in pages)}" if pages else ", nothing to change"))


if __name__ == "__main__":
    main()
