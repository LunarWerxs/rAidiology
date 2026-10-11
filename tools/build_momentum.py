"""Write the "Built in the open" numbers on the home page, and the run counts on the support page, from the run ledger.

    python tools/build_momentum.py            write site/index.html between <!-- momentum --> and <!-- /momentum -->,
                                              and every data-log="runs" / data-log="hours" count in site/support.html
    python tools/build_momentum.py --check    exit 1 when either page is out of date

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
SUPPORT = os.path.join(ROOT, "site", "support.html")
# Any element on either page carrying data-log="runs" or data-log="hours" gets that count as its text, and a
# <time data-log="newest"> gets the newest record's date.
LOG = re.compile(r'(<(\w+)[^>]*\bdata-log="(runs|hours)"[^>]*>)[^<]*(</\2>)')
NEWEST = re.compile(r'<time data-log="newest" datetime="[^"]*">[^<]*</time>')
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


def shown_date(newest):
    d = datetime.date.fromisoformat(newest)
    return f"{d.strftime('%b')} {d.day}, {d.year}"


def block(runs, hours, ideas, newest):
    items = [
        (f"{runs:,}", "runs recorded"),
        (f"{hours:,}", "hours of recorded runs"),
        (f"{ideas:,}", "ideas tried"),
        (f'<time datetime="{newest}">{shown_date(newest)}</time>', "newest entry"),
    ]
    rows = "\n".join(f'          <li><p class="m-n">{n}</p><p class="m-l">{label}</p></li>' for n, label in items)
    return f'\n        <ul class="momentum-list">\n{rows}\n        </ul>\n        '


def log_counts(old, nums):
    counts = {"runs": f"{nums[0]:,}", "hours": f"{nums[1]:,}"}
    new = LOG.sub(lambda m: m.group(1) + counts[m.group(3)] + m.group(4), old)
    when = f'<time data-log="newest" datetime="{nums[3]}">{shown_date(nums[3])}</time>'
    return NEWEST.sub(lambda _m: when, new)


def home_page(old, nums):
    if not MARK.search(old):
        raise SystemExit("site/index.html has no <!-- momentum --> block")
    return log_counts(MARK.sub(lambda m: m.group(1) + block(*nums) + m.group(3), old), nums)


def support_page(old, nums):
    if not LOG.search(old):
        raise SystemExit('site/support.html has no data-log="runs" or data-log="hours" count')
    return log_counts(old, nums)


def main():
    nums = numbers()
    if not (nums[0] and nums[3]):
        raise SystemExit("ledger/*.jsonl holds no records")
    stale, wrote = [], []
    for page, render in ((PAGE, home_page), (SUPPORT, support_page)):
        name = "site/" + os.path.basename(page)
        with open(page, encoding="utf-8", newline="") as f:
            old = f.read()
        new = render(old, nums)
        if new == old:
            continue
        stale.append(name)
        if "--check" not in sys.argv:
            with open(page, "w", encoding="utf-8", newline="") as f:
                f.write(new)
            wrote.append(name)
    if "--check" in sys.argv and stale:
        print(f"{', '.join(stale)}: momentum numbers are out of date: run tools/build_momentum.py")
        return 1
    print("momentum ok" + (f", wrote {', '.join(wrote)}" if wrote else ", nothing to change"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
