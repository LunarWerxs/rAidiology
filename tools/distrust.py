"""The next thing to re-check, from the register in docs/DISTRUST.md (the owner's standing rule to
distrust our own past work).

Reads the markdown table under '## The register' (columns: # | What is trusted | What rests on it |
Load | Last checked | How | Verdict | Fix). Load is 1 (most load-bearing) to 4; Last checked is
'YYYY-MM-DD HH:MM', or 'never' / empty for a row not yet checked. Markdown ** and backticks are
stripped from the cells for display.

Order of the next re-check: never checked first, then Load ascending, then Last checked oldest first
(row number breaks ties). A row is "due" when never checked or checked over 14 days ago.
Exits 1 with a message when the table is missing, a row has the wrong number of cells, or a date or
load is bad.

Usage: python tools/distrust.py [--top N] [--due] [--selftest]
       (--top N: the next N items, default 5; --due: only rows never checked or checked over 14 days
       ago; --selftest: parses a small in-memory register, touches nothing)
"""
import argparse
import os
import re
import sys
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTER = os.path.join(REPO, "tools", "DISTRUST.md")
HEADING = "## The register"
CELLS = 8
DUE_DAYS = 14


class RegisterError(Exception):
    pass


def clean(cell):
    return cell.replace("**", "").replace("`", "").strip()


def split_row(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def parse(text):
    """Rows of the register as dicts: n, what, load, last (datetime or None), verdict."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip().lower() == HEADING.lower()), None)
    if start is None:
        raise RegisterError("no '%s' heading found" % HEADING)
    table = []
    for l in lines[start + 1:]:
        if l.startswith("#"):
            break
        if l.strip().startswith("|"):
            table.append(l)
        elif table:
            break
    if len(table) < 2:
        raise RegisterError("no table under '%s'" % HEADING)
    rows = []
    for l in table[2:]:  # header and separator first
        cells = split_row(l)
        if len(cells) != CELLS:
            raise RegisterError("row has %d cells, expected %d: %s" % (len(cells), CELLS, l.strip()[:80]))
        n, what, _rests, load, last, _how, verdict, _fix = [clean(c) for c in cells]
        try:
            num = int(n.lstrip("#"))
        except ValueError:
            raise RegisterError("bad row number %r" % n)
        try:
            load_n = int(load)
        except ValueError:
            raise RegisterError("row %s: bad load %r" % (n, load))
        if last.lower() in ("", "never"):
            when = None
        else:
            try:
                when = datetime.strptime(last, "%Y-%m-%d %H:%M")
            except ValueError:
                raise RegisterError("row %s: bad date %r (want YYYY-MM-DD HH:MM or never)" % (n, last))
        rows.append({"n": num, "what": what, "load": load_n, "last": when, "verdict": verdict})
    return rows


def days_ago(row, now):
    return None if row["last"] is None else (now - row["last"]).days


def is_due(row, now):
    d = days_ago(row, now)
    return d is None or d > DUE_DAYS


def order(rows):
    return sorted(rows, key=lambda r: (r["last"] is not None, r["load"],
                                       r["last"] or datetime.min, r["n"]))


def line(row, now):
    d = days_ago(row, now)
    when = "never" if d is None else "%s (%d days ago)" % (row["last"].strftime("%Y-%m-%d"), d)
    return "#%d load %d, last checked %s  %s  [%s]" % (
        row["n"], row["load"], when, row["what"], row["verdict"] or "-")


def summary(rows, now):
    never = sum(1 for r in rows if r["last"] is None)
    due = sum(1 for r in rows if r["last"] is not None and is_due(r, now))
    return "register: %d rows, %d never checked, %d checked over %d days ago" % (
        len(rows), never, due, DUE_DAYS)


def selftest():
    now = datetime(2026, 10, 4, 23, 0)
    text = "\n".join([
        "# x", "", HEADING, "",
        "| # | What | Rests | Load | Last checked | How | Verdict | Fix |",
        "|---|---|---|---|---|---|---|---|",
        "| 1 | recent one | r | 1 | 2026-10-03 10:00 | h | **HOLDS** | none |",
        "| 2 | never three | r | 3 | never | h | | |",
        "| 3 | old one | r | 1 | 2026-09-01 08:30 | h | `WRONG` | f |",
        "| 4 | load two | r | 2 | 2026-10-01 09:00 | h | PARTLY | f |",
        "", "## Next", "| not | a | row |"])
    rows = parse(text)
    assert len(rows) == 4 and rows[0]["verdict"] == "HOLDS" and rows[2]["verdict"] == "WRONG"
    assert [r["n"] for r in order(rows)] == [2, 3, 1, 4], [r["n"] for r in order(rows)]
    assert [r["n"] for r in rows if is_due(r, now)] == [2, 3]
    assert summary(rows, now) == "register: 4 rows, 1 never checked, 1 checked over 14 days ago"
    assert line(rows[1], now) == "#2 load 3, last checked never  never three  [-]"
    assert line(rows[2], now).startswith("#3 load 1, last checked 2026-09-01 (33 days ago)")
    for bad in (text.replace("| 4 | load two | r | 2 |", "| 4 | load two | 2 |"),
                text.replace("2026-10-01 09:00", "yesterday"), "# nothing here"):
        try:
            parse(bad)
        except RegisterError:
            pass
        else:
            raise AssertionError("bad register accepted")
    print("distrust selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--due", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    try:
        with open(REGISTER, encoding="utf-8") as f:
            rows = parse(f.read())
    except (OSError, RegisterError) as e:
        print("distrust: %s: %s" % (REGISTER, e), file=sys.stderr)
        return 1
    now = datetime.now()
    ordered = order(rows)
    shown = [r for r in ordered if is_due(r, now)] if a.due else ordered[:max(a.top, 0)]
    for r in shown:
        print(line(r, now))
    print(summary(rows, now))
    return 0


if __name__ == "__main__":
    sys.exit(main())
