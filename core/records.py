"""Durable, queryable records of every queue, audit, swarm job, model, idea and claim, so none of it
dies with the work folder and a question about any of it is one SELECT.

`core/ledger.py` already keeps the numbers of every run. This keeps the rest, the same two-copy way:
- `records/` in the repository, committed, the durable record. Queue scripts and logs
  (`records/queues/<name>/`, with an `index.jsonl`), the audit files (`records/audit/`), and the
  HSwarm jobs that worked on this project (`records/hswarm/`). Every copy is gzipped
  deterministically (no name, no time stamp inside), so an unchanged file gives the same bytes and git
  sees no change. A copy is only rewritten when its source's sha256 changed. A source that later
  disappears from `work/` keeps its copy: that is the point.
- `work/records.db`, a SQLite file rebuilt from scratch on every `sync` from those copies and from
  the repository's own documents, so the files stay the truth. `work/ledger.db` is attached as
  `ledger`, so a run's numbers and a queue's log lines join in one query.

Tables: `models` and `model_fields` (every `Models/*/meta.json` and `work/eval/**/meta.json`, with
`grader.json` beside it), `queues` and `queue_lines`, `ideas` and `bars` (docs/efficiency/README.md:
the table rows, and the written-before-any-code sections), `register` (docs/DISTRUST.md),
`comparators` (docs/VS_DOCTORS.md), `datasets` (LICENSING.md, with a cached count of files and
bytes on disk for each folder it names), `audits` (the audit and swarm indexes), `meta` (when it ran,
at which commit, how many rows).

Walking the dataset folders is the slow part, so their counts are cached in `work/records_disk.json`
by the folder's own modification time; `sync --disk` walks them all again. Nothing here writes
LICENSING.md, reads a secret, or touches a drive other than D: and E:.

Usage: python core/records.py sync [--queue NAME] [--disk]
       python core/records.py status
       python core/records.py query "<SQL>" [--all]
       python core/records.py selftest             (its promises, checked in a throwaway folder)
"""
import argparse
import datetime
import gzip
import hashlib
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile

ROOT =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import paths  # noqa: E402

RECORDS = os.path.join(ROOT, "records")
DB = os.path.join(paths.WORK, "records.db")
DISK_CACHE = os.path.join(paths.WORK, "records_disk.json")
MODELS = os.path.join(paths.BESIDE, "Models")
DATASET_DIRS = (os.path.join(paths.BESIDE, "Datasets", "public"), "E:/rAidiology/archive")
HSWARM_JOBS = os.path.join(os.path.expanduser("~"), ".hswarm", "jobs")
HSWARM_CWD = "d:/newprojects/active/raidiology"     # a job counts when its cwd is under this
TEXT_EXT = (".txt", ".json", ".md", ".py", ".csv", ".jsonl")
PICTURE_EXT = (".png",)
AUDIT_MAX = 1 << 20          # an audit file bigger than this is indexed, not copied
HSWARM_MAX = 2 << 20
STAMP = re.compile(r"\b(\d\d:\d\d:\d\d)\b")
ROW_SEP = re.compile(r"^\|[\s:|-]+\|\s*$")
IDEA_AT = re.compile(r"\(idea (\d+)|\bidea (\d+):")
WRITTEN = re.compile(r"written (?:(\d{4}-\d\d-\d\d)(?: (\d\d:\d\d))?|(\d\d:\d\d))")
LIST_ITEM = re.compile(r"^( *)(?:[-*]|\d+\.) \*\*")
FULL_STAMP = re.compile(r"written (\d{4}-\d\d-\d\d)")
BAR_HEAD = re.compile(r"(?=.*written)(?=.*(?:before any code|before it ran|before any change|before the run))",
                      re.I)
_NUM = (r"\d+|(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[- ]"
        r"(?:one|two|three|four|five|six|seven|eight|nine))?|zero|one|two|three|four|five|six|seven|"
        r"eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
        r"|(?:one )?hundred)")
QUEUE_REF = re.compile(rf"\bqueue (?:{_NUM})\b|\bq\d+\b", re.I)
OUTCOME = re.compile(r"\b(?:NOT KEPT|KEPT|DROPPED|NOT CONFIRMED|CONFIRMED|BETTER|WORSE|INCONCLUSIVE|"
                     r"NOT WORSE)\b|(?i:gate passed)")
FOLDER = re.compile(r"(?:Datasets/public|archive)/([\w.\-]+)")
TABLES = ("models", "model_fields", "queues", "queue_lines", "ideas", "bars", "register",
          "comparators", "datasets", "audits", "meta")
SCHEMA = """
CREATE TABLE models (path TEXT, area TEXT, kind TEXT, version TEXT, has_grader INTEGER);
CREATE TABLE model_fields (path TEXT, key TEXT, value TEXT);
CREATE TABLE queues (name TEXT, state TEXT, first_line TEXT, last_line TEXT, first_stamp TEXT,
  last_stamp TEXT, files TEXT);
CREATE TABLE queue_lines (queue TEXT, line_no INTEGER, stamp TEXT, text TEXT);
CREATE TABLE ideas (idea INTEGER, title TEXT, why TEXT, cost TEXT, status TEXT, verdict TEXT);
CREATE TABLE bars (idea INTEGER, trial INTEGER, queues TEXT, written TEXT, title TEXT, text TEXT,
  outcome TEXT);
CREATE TABLE register (num TEXT, heading TEXT, trusted TEXT, rests_on TEXT, load TEXT, checked TEXT,
  how TEXT, verdict_text TEXT, fix TEXT, verdict TEXT);
CREATE TABLE comparators (heading TEXT, what TEXT, cells TEXT, status_text TEXT, status TEXT);
CREATE TABLE datasets (heading TEXT, source TEXT, licence TEXT, checked TEXT, used_in TEXT,
  status TEXT, cells TEXT, folder TEXT, disk_path TEXT, disk_files INTEGER, disk_bytes INTEGER);
CREATE TABLE audits (source TEXT, path TEXT, sha256 TEXT, bytes INTEGER, label TEXT, created TEXT,
  tasks INTEGER);
CREATE TABLE meta (key TEXT, value TEXT);
CREATE INDEX queue_lines_q ON queue_lines (queue, line_no);
"""


def gz_bytes(data):
    """The same bytes in give the same bytes out: no file name, no time stamp."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as f:
        f.write(data)
    return buf.getvalue()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


def copy_gz(src_data, dst, was_sha):
    """Write the gzip copy unless the source is unchanged. Returns (sha, gz bytes)."""
    digest = sha256(src_data)
    if digest != was_sha or not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(gz_bytes(src_data))
    return digest, os.path.getsize(dst)


# ---- queues ------------------------------------------------------------------------------------

def queue_name(filename):
    """`queue85.log` and `after_queue85.log` both belong to queue85; `queue_speed3.log` to queue_speed3."""
    base = filename[len("after_"):] if filename.startswith("after_") else filename
    return os.path.splitext(base)[0]


def queue_files(work):
    out = {}
    if not os.path.isdir(work):
        return out
    for fn in sorted(os.listdir(work)):
        p = os.path.join(work, fn)
        if (fn.startswith("queue") or fn.startswith("after_queue")) and os.path.isfile(p) \
                and not fn.endswith(".gz"):
            out.setdefault(queue_name(fn), []).append(fn)
    return out


def log_facts(text):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    stamps = [m.group(1) for ln in lines for m in STAMP.finditer(ln)]
    state = "open"
    if any("queue done" in ln for ln in lines):
        state = "done"
    if any("FAILED" in ln for ln in lines):
        state = "failed"
    return {"first_line": lines[0] if lines else "", "last_line": lines[-1] if lines else "",
            "first_stamp": stamps[0] if stamps else "", "last_stamp": stamps[-1] if stamps else "",
            "state": state}


def sync_queues(work, records, only=None):
    """Copy queue files to records/queues/<name>/ and rewrite the sorted index. Returns the names."""
    qdir = os.path.join(records, "queues")
    index_path = os.path.join(qdir, "index.jsonl")
    index = {r["name"]: r for r in read_jsonl(index_path)}
    done = []
    for name, files in queue_files(work).items():
        if only and name != only:
            continue
        was = {f["file"]: f["sha256"] for f in index.get(name, {}).get("files", [])}
        entry = {"name": name, "files": [], "first_line": "", "last_line": "", "first_stamp": "",
                 "last_stamp": "", "state": "open"}
        for fn in files:
            with open(os.path.join(work, fn), "rb") as f:
                data = f.read()
            digest, gz_size = copy_gz(data, os.path.join(qdir, name, fn + ".gz"), was.get(fn))
            entry["files"].append({"file": fn, "sha256": digest, "bytes": len(data), "gz_bytes": gz_size})
            if fn == name + ".log":
                entry.update(log_facts(data.decode("utf-8", "replace")))
        index[name] = entry
        done.append(name)
    write_jsonl(index_path, [index[k] for k in sorted(index)])
    return done


# ---- audit and swarm ---------------------------------------------------------------------------

def sync_audit(work, records):
    adir = os.path.join(records, "audit")
    index_path = os.path.join(adir, "index.jsonl")
    index = {r["path"]: r for r in read_jsonl(index_path)}
    src = os.path.join(work, "audit")
    for base, _dirs, names in os.walk(src):
        for fn in sorted(names):
            p = os.path.join(base, fn)
            rel = os.path.relpath(p, src).replace(os.sep, "/")
            ext = os.path.splitext(fn)[1].lower()
            size = os.path.getsize(p)
            if ext in PICTURE_EXT or (ext in TEXT_EXT and size < AUDIT_MAX):
                with open(p, "rb") as f:
                    data = f.read()
                row = {"path": rel, "bytes": size, "sha256": sha256(data)}
                if ext in TEXT_EXT:
                    copy_gz(data, os.path.join(adir, rel + ".gz"), index.get(rel, {}).get("sha256"))
                index[rel] = row
    write_jsonl(index_path, [index[k] for k in sorted(index)])


def norm_path(p):
    return re.sub(r"/+", "/", str(p).replace("\\", "/")).lower()


def sync_hswarm(jobs, records, cwd_prefix=HSWARM_CWD):
    """Index the jobs run from this project. Reads only job.json and results.jsonl."""
    hdir = os.path.join(records, "hswarm")
    index_path = os.path.join(hdir, "index.jsonl")
    index = {r["id"]: r for r in read_jsonl(index_path)}
    if os.path.isdir(jobs):
        for jid in sorted(os.listdir(jobs)):
            jp = os.path.join(jobs, jid, "job.json")
            if not os.path.isfile(jp):
                continue
            try:
                with open(jp, encoding="utf-8") as f:
                    job = json.load(f)
            except (OSError, ValueError):
                continue
            tasks = job.get("tasks") or []
            cwds = [(job.get("caller") or {}).get("cwd", "")]
            cwds += [t.get("cwd", "") for t in (tasks.values() if isinstance(tasks, dict) else tasks)
                     if isinstance(t, dict)]
            if not any(c and norm_path(c).startswith(cwd_prefix) for c in cwds):
                continue
            summary = job.get("summary") or {}
            row = {"id": jid, "label": summary.get("label") or (job.get("caller") or {}).get("label", ""),
                   "created": summary.get("created", ""), "tasks": len(tasks), "bytes": 0,
                   "sha256": ""}
            rp = os.path.join(jobs, jid, "results.jsonl")
            if os.path.isfile(rp) and os.path.getsize(rp) < HSWARM_MAX:
                with open(rp, "rb") as f:
                    data = f.read()
                row["bytes"] = len(data)
                row["sha256"], _ = copy_gz(data, os.path.join(hdir, jid + ".jsonl.gz"),
                                           index.get(jid, {}).get("sha256"))
            index[jid] = row
    write_jsonl(index_path, [index[k] for k in sorted(index)])


# ---- markdown ----------------------------------------------------------------------------------

def read_text(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def table_rows(text):
    """Yield (heading, cells, is_header) for every markdown table row; separator rows are skipped."""
    heading, prev_row = "", False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            heading = s.lstrip("#").strip()
        if s.startswith("|") and s.endswith("|") and len(s) > 1:
            if not ROW_SEP.match(s):
                cells = [c.strip() for c in re.split(r"(?<!\\)\|", s[1:-1])]
                yield heading, cells, not prev_row
            prev_row = True
        else:
            prev_row = False


def pick(text, rules, default="other"):
    low = text.lower()
    for word, verdict in rules:
        if word in low:
            return verdict
    return default


IDEA_RULES = (("not kept", "not kept"), ("worse", "not kept"), ("closed", "not kept"),
              ("gate passed", "gate passed"), ("bar written", "bar written"), ("queued", "queued"),
              ("kept", "kept"), ("better", "kept"), ("ships", "kept"))
COMPARATOR_RULES = (("not started", "not started"), ("ahead", "ahead not like for like"),
                    ("beaten", "beaten"), ("behind", "behind"), ("level", "level"))


def parse_ideas(text):
    rows = []
    for _h, cells, _hd in table_rows(text):
        if len(cells) >= 5 and cells[0].isdigit():
            rows.append((int(cells[0]), cells[1].replace("**", ""), cells[2], cells[3], cells[4],
                         pick(cells[4], IDEA_RULES)))
    return rows


def parse_bars(text):
    """Every heading or bold-opened paragraph that says it was written before the code, the run or
    the change, with its text up to the next heading or the next bar."""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines)
              if ((re.match(r"^#{1,6} ", ln) or ln.startswith("**")) and BAR_HEAD.search(ln))
              or (LIST_ITEM.match(ln) and BAR_HEAD.search(ln[:200]))]
    heads = [i for i, ln in enumerate(lines) if re.match(r"^#{1,6} ", ln)]
    stamps = [(i, m.group(1)) for i, ln in enumerate(lines) for m in FULL_STAMP.finditer(ln)]

    def stamp_of(i):
        w = WRITTEN.search(lines[i])
        if not w:
            return ""
        date = w.group(1) or next((d for n, d in reversed(stamps) if n <= i), "")
        return f"{date} {w.group(2) or w.group(3) or ''}".strip()

    def title_of(i):
        t = lines[i].lstrip("#").strip()
        if LIST_ITEM.match(lines[i]):
            t = re.sub(r"^(?:[-*]|\d+\.) +", "", t).replace("**", "")[:200].strip()
        return t.strip("*").strip()

    rows = []
    for k, i in enumerate(starts):
        j = min([x for x in heads + starts if x > i] or [len(lines)])
        item = LIST_ITEM.match(lines[i])
        if item:
            same = re.compile(r"^ {%d}(?:[-*]|\d+\.) " % len(item.group(1)))
            j = min([x for x in range(i + 1, j) if same.match(lines[x])] or [j])
        head = lines[i]
        body = "\n".join(lines[i:j]).strip()
        written, title, named = stamp_of(i), title_of(i), head
        if not written:
            # No stamp of its own: the nearest earlier heading or bar in this `##` section has one,
            # and its idea and trial numbers too.
            for n in range(i - 1, -1, -1):
                if (n in heads or n in starts) and stamp_of(n):
                    written, title, named = stamp_of(n), title_of(n) + " (the bar)", lines[n]
                    break
                if lines[n].startswith("## "):
                    break
        idea, trial = re.search(r"\bidea (\d+)", named, re.I), re.search(r"\bTrial (\d+)", named)
        queues = []
        for m in QUEUE_REF.finditer(head + "\n" + body):
            q = m.group(0).lower()
            if q not in queues:
                queues.append(q)
        outcome = next((ln.strip()[:300] for ln in lines[i + 1:j] if OUTCOME.search(ln)), None)
        rows.append((int(idea.group(1)) if idea else None, int(trial.group(1)) if trial else None,
                     ", ".join(queues), written, title, body, outcome))
    return rows


def parse_register(text):
    rows, in_register = [], False
    for h, cells, header in table_rows(text):
        if header:
            in_register = cells[0] == "#"
        elif in_register and len(cells) >= 8:
            cells = cells[:8]
            first = min(((cells[6].find(w), w) for w in ("HOLDS", "PARTLY", "WRONG", "CANNOT TELL")
                         if w in cells[6]), default=(0, "CANNOT TELL"))[1]
            rows.append((cells[0], h, *cells[1:8], first))
    return rows


def parse_comparators(text):
    return [(h, cells[0], json.dumps(cells, ensure_ascii=False), cells[-1],
             pick(cells[-1], COMPARATOR_RULES))
            for h, cells, header in table_rows(text) if not header and cells]


def parse_datasets(text):
    rows = []
    for h, cells, header in table_rows(text):
        if header or not cells:
            continue
        five = (cells + [""] * 5)[:5] if len(cells) == 5 else [cells[0], "", "", "", ""]
        m = FOLDER.search(cells[4]) if len(cells) == 5 else None
        rows.append((h, *five, json.dumps(cells, ensure_ascii=False), m.group(1) if m else ""))
    return rows


# ---- disk counts -------------------------------------------------------------------------------

def disk_count(folders, cache_path, force=False):
    """{folder: (path, files, bytes)}, walking a folder only when its own mtime changed."""
    try:
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    out = {}
    for name in sorted(set(folders)):
        for base in folders_roots(folders):
            p = os.path.join(base, name)
            if not os.path.isdir(p):
                continue
            key = p.replace("\\", "/")
            mtime = os.stat(p).st_mtime
            hit = cache.get(key)
            if force or not hit or hit["mtime"] != mtime:
                files = size = 0
                for b, _d, names in os.walk(p):
                    for fn in names:
                        try:
                            size += os.path.getsize(os.path.join(b, fn))
                            files += 1
                        except OSError:
                            pass
                hit = cache[key] = {"mtime": mtime, "files": files, "bytes": size}
            out[name] = (key, hit["files"], hit["bytes"])
            break
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, sort_keys=True, indent=1)
    return out


_ROOTS = list(DATASET_DIRS)


def folders_roots(_folders):
    return _ROOTS


# ---- the database ------------------------------------------------------------------------------

def scalars(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, (str, int, float, bool)) or v is None} \
        if isinstance(data, dict) else {}


def model_rows(models, eval_dir):
    found = []
    if os.path.isdir(models):
        for d in sorted(os.listdir(models)):
            p = os.path.join(models, d, "meta.json")
            if os.path.isfile(p):
                found.append((p, d, "models"))
    for base, _dirs, names in os.walk(eval_dir):
        if "meta.json" in names:
            rel = os.path.relpath(base, eval_dir)
            found.append((os.path.join(base, "meta.json"), rel.split(os.sep)[0], "eval"))
    models_rows, fields = [], []
    for p, area, kind in sorted(found):
        meta = scalars(p)
        grader = os.path.join(os.path.dirname(p), "grader.json")
        g = scalars(grader) if os.path.isfile(grader) else {}
        key = p.replace("\\", "/")
        models_rows.append((key, area, kind, str(meta.get("version", "")), int(os.path.isfile(grader))))
        fields += [(key, k, json.dumps(v) if v is None or isinstance(v, bool) else str(v))
                   for k, v in meta.items()]
        fields += [(key, "grader." + k, json.dumps(v) if v is None or isinstance(v, bool) else str(v))
                   for k, v in g.items()]
    return models_rows, fields


def git_head(cwd):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def build_db(db, records, repo, models, eval_dir, ledger_db=None, cache_path=DISK_CACHE,
             disk=False, disk_roots=None, head=""):
    """Rebuild every table from the committed copies and the repository's documents."""
    global _ROOTS
    if disk_roots is not None:
        _ROOTS = list(disk_roots)
    os.makedirs(os.path.dirname(db), exist_ok=True)
    if os.path.exists(db):
        os.remove(db)
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    if ledger_db and os.path.exists(ledger_db):
        con.execute("ATTACH DATABASE ? AS ledger", (ledger_db,))

    def put(table, rows):
        rows = list(rows)
        if rows:
            con.executemany(f"INSERT INTO {table} VALUES ({','.join('?' * len(rows[0]))})", rows)

    m, f = model_rows(models, eval_dir)
    put("models", m)
    put("model_fields", f)

    qdir = os.path.join(records, "queues")
    lines = []
    for q in read_jsonl(os.path.join(qdir, "index.jsonl")):
        put("queues", [(q["name"], q["state"], q["first_line"], q["last_line"], q["first_stamp"],
                        q["last_stamp"], json.dumps(q["files"]))])
        gz = os.path.join(qdir, q["name"], q["name"] + ".log.gz")
        if os.path.exists(gz):
            with gzip.open(gz, "rb") as fh:
                for n, ln in enumerate(fh.read().decode("utf-8", "replace").splitlines(), 1):
                    s = STAMP.search(ln)
                    lines.append((q["name"], n, s.group(1) if s else "", ln))
    put("queue_lines", lines)

    readme = read_text(os.path.join(repo, "tools", "efficiency", "README.md"))
    put("ideas", parse_ideas(readme))
    put("bars", parse_bars(readme))
    put("register", parse_register(read_text(os.path.join(repo, "tools", "DISTRUST.md"))))
    put("comparators", parse_comparators(read_text(os.path.join(repo, "tools", "VS_DOCTORS.md"))))

    ds = parse_datasets(read_text(os.path.join(repo, "LICENSING.md")))
    counts = disk_count({r[7] for r in ds if r[7]}, cache_path, force=disk)
    put("datasets", [(*r, *(counts.get(r[7], ("", None, None)) if r[7] else ("", None, None)))
                     for r in ds])

    audits = [("audit", r["path"], r["sha256"], r["bytes"], "", "", None)
              for r in read_jsonl(os.path.join(records, "audit", "index.jsonl"))]
    audits += [("hswarm", r["id"], r["sha256"], r["bytes"], r["label"], r["created"], r["tasks"])
               for r in read_jsonl(os.path.join(records, "hswarm", "index.jsonl"))]
    put("audits", audits)

    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES if t != "meta"}
    put("meta", [("synced", datetime.datetime.now().isoformat(timespec="seconds")), ("head", head)]
        + [("rows." + t, str(n)) for t, n in counts.items()])
    con.commit()
    con.close()
    return counts


# ---- commands ----------------------------------------------------------------------------------

def stale_queues(work, records):
    index = {r["name"]: r for r in read_jsonl(os.path.join(records, "queues", "index.jsonl"))}
    stale = []
    for name, files in queue_files(work).items():
        was = {f["file"]: f["sha256"] for f in index.get(name, {}).get("files", [])}
        for fn in files:
            with open(os.path.join(work, fn), "rb") as fh:
                if was.get(fn) != sha256(fh.read()):
                    stale.append(name)
                    break
    return sorted(stale)


def cmd_sync(args):
    work = paths.WORK
    if args.queue:
        done = sync_queues(work, RECORDS, only=args.queue)
        print(f"queue copies: {', '.join(done) or 'none found for ' + args.queue}")
        return 0
    n = len(sync_queues(work, RECORDS))
    sync_audit(work, RECORDS)
    sync_hswarm(HSWARM_JOBS, RECORDS)
    from core import ledger
    ledger.database().close()
    counts = build_db(DB, RECORDS, ROOT, MODELS, os.path.join(work, "eval"), ledger.DB,
                      disk=args.disk, head=git_head(ROOT))
    print(f"{n} queues copied; " + ", ".join(f"{t} {c}" for t, c in counts.items()))
    return 0


def cmd_status(_args):
    if not os.path.exists(DB):
        print("no records.db yet: run `sync`")
    else:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        meta = dict(con.execute("SELECT key, value FROM meta"))
        print(f"last sync {meta.get('synced', '?')} at {meta.get('head', '')[:10]}")
        for t in TABLES:
            print(f"{t}\t{con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]}")
        con.close()
    stale = stale_queues(paths.WORK, RECORDS)
    print("queue copies older than their source: " + (", ".join(stale) or "none"))
    return 0


def cmd_query(args):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    from core import ledger
    if os.path.exists(ledger.DB):
        con.execute("ATTACH DATABASE ? AS ledger", (f"file:{ledger.DB}?mode=ro",))
    cur = con.execute(args.sql)
    print("\t".join(d[0] for d in cur.description or []))
    for n, row in enumerate(cur):
        if n >= 200 and not args.all:
            print("... (200 rows shown; --all for the rest)")
            break
        print("\t".join("" if v is None else str(v) for v in row))
    return 0


def selftest():
    """Everything built from small fixtures in a throwaway folder; nothing outside it is touched."""
    global _ROOTS
    with tempfile.TemporaryDirectory() as tmp:
        def w(rel, text):
            p = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            return p
        w("work/queue7.log", "10:00:01 start\n10:05:09 step\n10:09:30 queue done\n")
        w("work/queue7.sh", "echo hi\n")
        w("work/after_queue7.log", "11:00:00 after\n")
        w("work/queue_speed3.log", "09:00:00 FAILED here\n")
        w("work/audit/a.txt", "audit text\n")
        w("work/audit/p.png", "not really a picture")
        w("work/eval/chest/v1/meta.json", json.dumps({"version": "v1", "auroc": 0.9, "nested": {"a": 1}}))
        w("work/eval/chest/v1/grader.json", json.dumps({"kappa": 0.5}))
        w("Models/spine/meta.json", json.dumps({"version": "v3"}))
        w("repo/tools/efficiency/README.md",
          "# Ideas\n\n| # | Idea | Why | Cost | Status |\n|---|---|---|---|---|\n"
          "| 1 | **Soup** | merge | minutes | **WORSE, closed** (22:22) |\n\n"
          "### Soup the nets (idea 1, written 2026-10-04 21:48, before any code)\nThe bar.\n\n"
          "**Trial 10, idea 4 (written 22:14, before any code): three renderings, chest v114.**\n"
          "Queue seventy-nine and q70 run it.\nBETTER than the last.\n\n"
          "**Trial 8 on five folds (queue eighty, written 2026-10-04 22:09, before it ran).**\nPlain.\n\n"
          "### A card lease per fold (RSI; written 2026-10-05 00:11, before any change)\nNOT KEPT yet.\n\n"
          "## Later\n\n"
          "### More data: OrthoFrac-XR (queue eighteen, written 2026-10-03 before it ran)\nText.\n\n"
          "**The bar, written before it ran,** is queue eight's: x\n\n"
          "- **Bar for `tta_top`** (written 17:54, before any code): y\n"
          "3. **v104 (written 2026-10-03 13:05, before it ran): inverted pictures and gamma**, z\n\n"
          "## Next\n")
        w("repo/tools/DISTRUST.md",
          "# Register\n\n| # | What | Rests | Load | Last | How | Verdict | Fix |\n|---|---|---|---|---|---|---|---|\n"
          "| 1 | labels | grader | 1 | 2026-10-04 | read | **WRONG** three ways | idea 21 |\n")
        w("repo/tools/VS_DOCTORS.md",
          "## Spine\n\n| Measure | Ours | Bar | Source | Same? | Status |\n|---|---|---|---|---|---|\n"
          "| kappa | 0.78 | 0.51 | paper | yes | **beaten** |\n")
        w("repo/LICENSING.md",
          "## Data\n\n| Source | Licence | Checked | Used in | Status |\n|---|---|---|---|---|\n"
          "| [BUU](https://x) | EULA | 2026-10-03 | nowhere | Usable, in `Datasets/public/buu_lspine` |\n")
        w("Datasets/public/buu_lspine/a.bin", "12345")
        w("jobs/j1/job.json", json.dumps({"summary": {"label": "L", "created": "2026-10-04T06:51:41+00:00"},
                                          "caller": {"cwd": "D:\\\\NewProjects\\\\Active\\\\rAidiology\\\\x"},
                                          "tasks": [{}, {}]}))
        w("jobs/j1/results.jsonl", '{"id": "a"}\n')
        w("jobs/j2/job.json", json.dumps({"caller": {"cwd": "C:\\\\elsewhere"}, "tasks": []}))
        work, rec = os.path.join(tmp, "work"), os.path.join(tmp, "records")
        assert gz_bytes(b"same") == gz_bytes(b"same")
        assert sorted(sync_queues(work, rec)) == ["queue7", "queue_speed3"]
        before = os.path.getmtime(os.path.join(rec, "queues", "queue7", "queue7.log.gz"))
        sync_queues(work, rec, only="queue7")
        assert os.path.getmtime(os.path.join(rec, "queues", "queue7", "queue7.log.gz")) == before
        sync_audit(work, rec)
        sync_hswarm(os.path.join(tmp, "jobs"), rec)
        counts = build_db(os.path.join(work, "records.db"), rec, os.path.join(tmp, "repo"),
                          os.path.join(tmp, "Models"), os.path.join(work, "eval"),
                          cache_path=os.path.join(work, "records_disk.json"), head="abc",
                          disk_roots=[os.path.join(tmp, "Datasets", "public")])
        assert counts["queues"] == 2 and counts["queue_lines"] == 4, counts
        assert counts["models"] == 2 and counts["ideas"] == 1 and counts["bars"] == 8, counts
        assert counts["register"] == 1 and counts["comparators"] == 1 and counts["datasets"] == 1, counts
        assert counts["audits"] == 3, counts       # a.txt, p.png, one swarm job
        con = sqlite3.connect(os.path.join(work, "records.db"))
        one = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
        assert one("SELECT verdict FROM ideas") == "not kept"
        assert one("SELECT written FROM bars WHERE idea=1") == "2026-10-04 21:48"
        assert one("SELECT written FROM bars WHERE trial=10") == "2026-10-04 22:14"
        assert one("SELECT idea FROM bars WHERE trial=10") == 4
        assert one("SELECT outcome FROM bars WHERE trial=10") == "BETTER than the last."
        assert one("SELECT queues FROM bars WHERE trial=10") == "queue seventy-nine, q70"
        assert one("SELECT queues FROM bars WHERE trial=8") == "queue eighty"
        assert one("SELECT written FROM bars WHERE title LIKE 'A card lease%'") == "2026-10-05 00:11"
        assert one("SELECT outcome FROM bars WHERE title LIKE 'A card lease%'") == "NOT KEPT yet."
        assert one("SELECT written FROM bars WHERE title LIKE 'More data%' AND title NOT LIKE '%(the bar)'") == "2026-10-03"
        assert one("SELECT written FROM bars WHERE title LIKE 'More data%(the bar)'") == "2026-10-03"
        assert one("SELECT written FROM bars WHERE title LIKE 'Bar for `tta_top`%'") == "2026-10-03 17:54"
        assert one("SELECT written FROM bars WHERE title LIKE 'v104 %'") == "2026-10-03 13:05"
        assert one("SELECT verdict FROM register") == "WRONG"
        assert one("SELECT status FROM comparators") == "beaten"
        assert one("SELECT folder FROM datasets") == "buu_lspine"
        assert one("SELECT disk_bytes FROM datasets") == 5
        assert one("SELECT state FROM queues WHERE name='queue7'") == "done"
        assert one("SELECT state FROM queues WHERE name='queue_speed3'") == "failed"
        assert one("SELECT last_stamp FROM queues WHERE name='queue7'") == "10:09:30"
        assert one("SELECT COUNT(*) FROM model_fields WHERE key='grader.kappa'") == 1
        con.close()
        assert stale_queues(work, rec) == []
    _ROOTS = list(DATASET_DIRS)
    print("selftest ok")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sync")
    p.add_argument("--queue")
    p.add_argument("--disk", action="store_true")
    sub.add_parser("status")
    p = sub.add_parser("query")
    p.add_argument("sql")
    p.add_argument("--all", action="store_true")
    sub.add_parser("selftest")
    args = ap.parse_args()
    return {"sync": cmd_sync, "status": cmd_status, "query": cmd_query,
            "selftest": lambda a: selftest()}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
