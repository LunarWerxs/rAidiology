"""One database of every training data set: where it lives, how big it is, what was done to it and when, and why.

Built from what the jobs already record, never typed by hand, so it can be rebuilt at any time:
- E:/rAidiology/archive/<set>/fetch.json (fetch_public.py: source, licence, download, unpack, lossless shrink, moves)
  and train.json (train_copy.py: the training copies made from TCIA/IDC series);
- the junctions in Datasets/public, which say where each working copy is now, and the download archives still kept
  beside an unpacked copy (an <archive>.extracted marker, left by modalities/*/fetch_data.py);
- the milestone lines of every log in E:/rAidiology/logs (done, unpacked, moved, shrunk, deleted, paused, failed);
- each drive's size and free space when it ran;
- records/datasets/decisions.json: the decisions behind all of it, each with the words it rests on.

Writes E:/rAidiology/catalog/datasets.sqlite and records/datasets/catalog.sql (the same database as SQL text, rows
in a fixed order, so git shows what changed between two builds). Reads only; touches no data.

    python tools/dataset_catalog.py            # build both
    python tools/dataset_catalog.py --md t.md  # and the tables of the CrAInium doc docs/DATA_STORE.md
    sqlite3 E:/rAidiology/catalog/datasets.sqlite "select name, drive, state from sets order by drive"
"""
import datetime
import glob
import json
import os
import re
import shutil
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = "E:/rAidiology/archive"
LOGS = "E:/rAidiology/logs"
LINKS = os.path.join(os.path.dirname(ROOT), "Datasets", "public")
DB = "E:/rAidiology/catalog/datasets.sqlite"
SQL = os.path.join(ROOT, "records", "datasets", "catalog.sql")
DECISIONS = os.path.join(ROOT, "records", "datasets", "decisions.json")
DRIVES = "DEFGH"
ARCHIVE_EXT = (".zip", ".tar.gz", ".tgz", ".tar", ".7z", ".rar")
LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ([A-Za-z0-9_.+-]+): (.*)$")
MILESTONE = re.compile(r"\b(done|unpacked|now at|removed|deleted|re-encoded|ended|paused|resumed|failed|FAILED|"
                       r"gives up|blocked|junction made|moved|finished|this run|started|STOP|kept|held|thawed|"
                       r"frozen|has its turn)\b")
PROGRESS = re.compile(r"^\s*\d+ of \d+ |\d+ of \d+ GB through|MB/s unpacked|waits its turn")

SCHEMA = """
create table built (at text, by text);
create table drives (drive text primary key, total_gb real, free_gb real, used_pct real, line_pct real, note text);
create table sets (
  name text primary key, kind text, source text, link text, licence text, status text, error text,
  started text, finished text, download_gb real, download_files integer, unpacked_at text, download_removed_at text,
  shrink_codec text, shrink_files integer, shrink_shrunk integer, shrink_left_alone integer,
  shrink_gb_before real, shrink_gb_after real, working_copy text, drive text, junction text, on_disk integer,
  kept_archives_gb real, state text, note text, record text);
create table moves (name text, from_path text, to_path text, at text);
create table train_copies (
  name text primary key, collection text, method_ct text, modality text, series integer, files integer,
  gb_in real, gb_out real, finished text, held text, deleted text, note text, working_copy text, drive text,
  on_disk integer);
create table kept_archives (name text, archive text, gb real, marker text);
create table events (at text, log text, subject text, text text);
create table decisions (id text primary key, date text, decided_by text, decision text, why text, words text,
  source text);
"""


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def drive_of(path):
    return (path or "")[:1].upper() or None


def target(link):
    try:
        return os.readlink(link).replace("\\\\?\\", "").replace("\\", "/")
    except OSError:
        return None


def gb(n):
    return round(n / 1e9, 3) if n is not None else None


def build(db):
    db.executescript(SCHEMA)
    db.execute("insert into built values (?, ?)", (now(), "tools/dataset_catalog.py"))
    for d in DRIVES:
        root = d + ":/"
        if not os.path.exists(root):
            continue
        u = shutil.disk_usage(root)
        line = 85.0 if d == "H" else 80.0
        db.execute("insert into drives values (?,?,?,?,?,?)",
                   (d, round(u.total / 1e9, 1), round(u.free / 1e9, 1), round(100 * u.used / u.total, 1), line,
                    "may reach 95% while sets pass through, never stay over 85%" if d == "H" else None))

    names = {n for n in os.listdir(LINKS) if os.path.isdir(os.path.join(LINKS, n))}
    names |= {n for n in os.listdir(ARCHIVE) if not n.startswith("_") and (
        os.path.exists(os.path.join(ARCHIVE, n, "fetch.json")) or os.path.exists(os.path.join(ARCHIVE, n, "train.json")))}
    for name in sorted(n for n in names if not n.startswith("_")):
        link = os.path.join(LINKS, name)
        where = target(link) if os.path.lexists(link) else None
        if where is None and os.path.isdir(link):
            where = link.replace("\\", "/")
        kept = []
        if where and os.path.isdir(where):
            for f in sorted(os.listdir(where)):
                p = os.path.join(where, f)
                if f.lower().endswith(ARCHIVE_EXT) and os.path.isfile(p) and os.path.exists(p + ".extracted"):
                    kept.append((f, os.path.getsize(p)))
        for f, b in kept:
            db.execute("insert into kept_archives values (?,?,?,?)", (name, f, gb(b), f + ".extracted"))

        t = load(os.path.join(ARCHIVE, name, "train.json"))
        if t:
            s = t.get("series", {})
            wc = t.get("working_copy")
            db.execute("insert into train_copies values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                name, t.get("collection"), (t.get("method") or {}).get("CT"), t.get("modality"), len(s),
                sum(len(v.get("files", [])) for v in s.values()),
                gb(t.get("bytes_in", sum(v.get("bytes_in", 0) for v in s.values()))),
                gb(t.get("bytes_out", sum(v.get("bytes_out", 0) for v in s.values()))),
                t.get("finished"), t.get("held"), t.get("deleted"), t.get("note"), wc, drive_of(wc),
                int(bool(wc) and os.path.isdir(wc))))
            continue

        r = load(os.path.join(ARCHIVE, name, "fetch.json"))
        if r is None:
            marker = next((m for m in ("figshare.json", "mendeley.json") if where and os.path.exists(os.path.join(where, m))), None)
            db.execute("insert into sets (name, kind, working_copy, drive, junction, on_disk, kept_archives_gb, state, "
                       "note) values (?,?,?,?,?,?,?,?,?)", (
                           name, "modalities fetcher" if (marker or kept) else "no record", where, drive_of(where),
                           link.replace("\\", "/") if os.path.lexists(link) else None,
                           int(bool(where) and os.path.isdir(where)), gb(sum(b for _, b in kept)) if kept else None,
                           "in use" if where and os.path.isdir(where) else "missing",
                           "fetched by modalities/*/fetch_data.py (%s); no fetch_public record" % marker if marker else
                           "no fetch_public or train_copy record"))
            continue
        sh = r.get("shrunk") or {}
        wc = r.get("working_copy") or where
        files = r.get("files") or {}
        if r.get("status") == "blocked":
            state = "blocked"
        elif r.get("status") != "done":
            state = "downloading"
        elif not r.get("unpacked"):
            state = "downloaded, not unpacked"
        else:
            state = "in use" if wc and os.path.isdir(wc) else "unpacked, working copy missing"
        db.execute("insert into sets values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            name, "download", r.get("archive"), r.get("link"), r.get("licence_seen"), r.get("status"), r.get("error"),
            r.get("started"), r.get("finished"), gb(r.get("bytes")), len(files) or None, r.get("unpacked"),
            r.get("removed"), sh.get("codec"), sh.get("files"), sh.get("shrunk"), sh.get("left_alone"),
            sh.get("gb_before"), sh.get("gb_after"), wc, drive_of(wc), r.get("junction"),
            int(bool(wc) and os.path.isdir(wc)), gb(sum(b for _, b in kept)) if kept else None, state,
            "; ".join(str(n) for n in r.get("notes") or []) or None,
            os.path.join(ARCHIVE, name, "fetch.json").replace("\\", "/")))
        for m in r.get("moved") or []:
            db.execute("insert into moves values (?,?,?,?)", (name, m.get("from"), m.get("to"), m.get("at")))

    for path in sorted(glob.glob(os.path.join(LOGS, "*.log"))):
        log = os.path.basename(path)
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = LINE.match(line.rstrip("\n"))
                if not m or PROGRESS.search(m.group(3)) or not MILESTONE.search(m.group(3)):
                    continue
                db.execute("insert into events values (?,?,?,?)", (m.group(1), log, m.group(2), m.group(3)[:600]))

    for d in load(DECISIONS) or []:
        db.execute("insert into decisions values (?,?,?,?,?,?,?)", (
            d["id"], d["date"], d["by"], d["decision"], d.get("why"), d.get("words"), d.get("source")))


def dump(db, path):
    """The database as SQL, each table's rows sorted, so two builds diff line by line."""
    tables = [r[0] for r in db.execute("select name from sqlite_master where type='table' order by name")]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("-- rAidiology data set catalog, built by tools/dataset_catalog.py; load with: sqlite3 x.db < catalog.sql\n")
        for t in tables:
            f.write(db.execute("select sql from sqlite_master where name=?", (t,)).fetchone()[0] + ";\n")
        for t in tables:
            cols = [c[1] for c in db.execute("pragma table_info(%s)" % t)]
            for row in db.execute("select * from %s order by %s" % (t, ", ".join(cols))):
                vals = ", ".join("NULL" if v is None else str(v) if isinstance(v, (int, float))
                                 else "'" + str(v).replace("'", "''") + "'" for v in row)
                f.write("insert into %s values (%s);\n" % (t, vals))


def cell(v):
    return "-" if v is None else str(v).replace("|", "/").replace("\n", " ")


def table(db, sql, head):
    rows = db.execute(sql).fetchall()
    out = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    out += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def markdown(db, path):
    """The tables of the CrAInium doc docs/DATA_STORE.md, from the database just built ("-" is not recorded)."""
    built = db.execute("select at from built").fetchone()[0]
    parts = [
        "## Drives (%s)\n" % built[:16].replace("T", " "),
        table(db, "select drive, total_gb, free_gb, used_pct, line_pct, coalesce(note, '') from drives order by drive",
              ["drive", "size GB", "free GB", "used %", "line %", "note"]),
        "\n## Decisions\n",
        table(db, "select id, date, decided_by, decision, coalesce(words, ''), source from decisions order by id",
              ["id", "date", "by", "decision", "their words", "source"]),
        "\n## Every set\n\nGB are the download's size; the lossless shrink columns are the DICOM it re-encoded. "
        "kept GB is a download still kept beside its unpacked copy.\n",
        table(db, "select name, coalesce(drive, '-'), state, kind, download_gb, shrink_gb_before, shrink_gb_after, "
                  "kept_archives_gb, licence from sets order by drive, name",
              ["set", "drive", "state", "kind", "GB", "shrunk from GB", "to GB", "kept GB", "licence"]),
        "\n## Training copies (train_copy.py)\n",
        table(db, "select name, drive, method_ct, series, gb_in, gb_out, coalesce(deleted, finished, held) from "
                  "train_copies order by drive, name",
              ["copy", "drive", "CT form", "series", "GB in", "GB kept", "deleted / finished / held"]),
        "\n## Moves between drives\n",
        table(db, "select at, name, from_path, to_path from moves order by at", ["at", "set", "from", "to"]),
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(parts) + "\n")


def main():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    os.makedirs(os.path.dirname(SQL), exist_ok=True)
    tmp = DB + ".new"
    if os.path.exists(tmp):
        os.remove(tmp)
    db = sqlite3.connect(tmp)
    build(db)
    db.commit()
    dump(db, SQL)
    if "--md" in sys.argv:
        markdown(db, sys.argv[sys.argv.index("--md") + 1])
    counts = {t: db.execute("select count(*) from %s" % t).fetchone()[0]
              for t in ("sets", "train_copies", "moves", "kept_archives", "events", "decisions")}
    db.close()
    os.replace(tmp, DB)
    print("catalog: %s -> %s and %s" % (counts, DB, os.path.relpath(SQL, ROOT)))


if __name__ == "__main__":
    sys.exit(main())
