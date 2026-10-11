"""Write the "Built in the open" numbers on the home page from the run ledger.

    python tools/build_momentum.py            write site/index.html between <!-- momentum --> and <!-- /momentum -->
    python tools/build_momentum.py --check    exit 1 when the page is out of date

Reads ledger/*.jsonl (one JSON record per run) and keeps only counts, hours and one date:
runs recorded, hours of recorded runs, ideas tried (distinct "idea N" numbers named in a record) and the date of the
newest record. No dataset name, host name, path or note text ever reaches the page.
"""
import datetime
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "site", "index.html")
MARK = re.compile(r"(<!-- momentum -->)(.*?)(<!-- /momentum -->)", re.S)
IDEA = re.compile(r"\bidea (\d+)", re.I)


def numbers():
    runs, seconds, newest, ideas = 0, 0.0, "", set()
    for path in sorted(glob.glob(os.path.join(ROOT, "ledger", "*.jsonl"))):
        if os.path.basename(path) == "visitors.jsonl":  # simulated-visitor reviews of the site, not training or test runs
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict):
                    continue
                runs += 1
                s = rec.get("seconds")
                if isinstance(s, (int, float)) and s > 0:
                    seconds += s
                at = rec.get("at")
                if isinstance(at, str):
                    try:
                        newest = max(newest, datetime.date.fromisoformat(at[:10]).isoformat())
                    except ValueError:
                        pass
                for field in ("name", "note"):
                    v = rec.get(field)
                    if isinstance(v, str):
                        ideas.update(int(n) for n in IDEA.findall(v))
    return runs, round(seconds / 3600), len(ideas), newest


def block():
    runs, hours, ideas, newest = numbers()
    if not (runs and newest):
        raise SystemExit("ledger/*.jsonl holds no records")
    d = datetime.date.fromisoformat(newest)
    shown = f"{d.strftime('%b')} {d.day}, {d.year}"
    items = [
        (f"{runs:,}", "runs recorded"),
        (f"{hours:,}", "hours of recorded runs"),
        (f"{ideas:,}", "ideas tried"),
        (f'<time datetime="{newest}">{shown}</time>', "newest entry"),
    ]
    rows = "\n".join(f'          <li><p class="m-n">{n}</p><p class="m-l">{label}</p></li>' for n, label in items)
    return f'\n        <ul class="momentum-list">\n{rows}\n        </ul>\n        '


def main():
    with open(PAGE, encoding="utf-8", newline="") as f:
        old = f.read()
    if not MARK.search(old):
        raise SystemExit("site/index.html has no <!-- momentum --> block")
    new = MARK.sub(lambda m: m.group(1) + block() + m.group(3), old)
    if "--check" in sys.argv:
        if new != old:
            print("site/index.html momentum numbers are out of date: run tools/build_momentum.py")
            return 1
        print("momentum ok")
        return 0
    if new != old:
        with open(PAGE, "w", encoding="utf-8", newline="") as f:
            f.write(new)
    print("momentum ok" + (", wrote site/index.html" if new != old else ", nothing to change"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
