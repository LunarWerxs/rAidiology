"""Herniation labels for LSMA-PQR's lumbar discs from the radiologists' own free-text notes.

Source: `Radiologists Report.xlsx` (the Mendeley Data "lumbar-mri-radiologist-notes" set, the same
patients as LSMA-PQR; columns "Patient ID" and "Clinician's Notes", 575 rows, ids 1-575 = LSMA-PQR's
Patient_ID, 60 notes empty). Path: <DATASETS>/public/lumbar-mri-radiologist-notes/, or LSMA_NOTES_XLSX.
Why: LSMA-PQR's structured_notes.csv reads "no evidence of disc herniation" as a herniation and types
only L4-L5 and L5-S1 (lsma.note_items), so its herniation labels are wrong in a known way.

Rule (`rule_read`, deterministic): level spellings are normalised (L4-5, L4/5, L4-L5, L4 L5, L4_L5, L5S1,
D12-L1 = T12-L1, lumbosacral / LS junction = L5-S1, "last two levels" = L4-L5 and L5-S1; a range such
as "L3-4 to L5-S1" or "L3-L5" covers every level in it). A line "L4-5:" or "L4/L5 level:" (levels, no
finding term, then a colon) is a heading: its levels belong to the lines after it until the next
heading. The note is cut into clauses at sentence stops, at "but/however/while/whereas" and at a comma
before "no/not/without". A herniation term (protrusion, extrusion, herniation, HNP, prolapse,
sequestration) counts unless "no, not, without, negative for, free of, absence of, rule out, r/o"
stands before it in its clause. A counted term belongs to the levels its clause names, else to its
section heading's; with neither the note reports a herniation it does not place (herniation_unplaced).
Bulge terms work the same way. A negated herniation term with no level, in a note with no counted
one, is global_negative. Lines under a cervical, thoracic or dorsal heading are skipped.

merge(rule, swarm, hand): a second reading (a swarm of cheap models, work/lsma_text/swarm.json) and
the hand readings of the notes they differ on (lsma_text_hand.json). Same herniation_levels and
herniation_unplaced = 'agree'; else the hand reading when there is one ('hand'); else 'unresolved'.
Labels per disc: herniation 1 where a herniation is placed at the level, else blank (unknown) when
the note reports one it does not place, is empty or is unresolved, else 0; bulge 1 / 0, blank for an
empty note.

Usage: python lsma_text.py build      (swarm.json + hand -> lsma_text_labels.csv, work/lsma_text/disagree.csv;
                                       exits 2 while notes are unresolved)
       python lsma_text.py gate       (every 'Central Protrusion' in structured_notes.csv must be a 1 here; 95%)
       python lsma_text.py slipgate   (lsma.slip_items counts per level; 20 discs to read by hand: work/lsma_text/slip_gate.csv)
       python lsma_text.py selftest   (synthetic notes, no file, no GPU)"""
import csv
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository root
from core import paths  # noqa: E402

XLSX = os.environ.get("LSMA_NOTES_XLSX", os.path.join(
    paths.DATASETS, "public", "lumbar-mri-radiologist-notes", "Radiologists Report.xlsx"))
WORK = os.path.join(paths.WORK, "lsma_text")
SWARM = os.path.join(WORK, "swarm.json")
SWARM_JOB = "20261005-033649-4e43"
HAND = os.path.join(HERE, "lsma_text_hand.json")
LABELS = os.path.join(HERE, "lsma_text_labels.csv")
READER = "rule-v1+"
VERT = ["T12", "L1", "L2", "L3", "L4", "L5", "S1"]
LEVELS = [f"{a}-{b}" for a, b in zip(VERT, VERT[1:])]
DISC_LEVELS = LEVELS[1:]                                  # L1-L2 .. L5-S1
IVD = {"L5-S1": 1, "L4-L5": 2, "L3-L4": 3}                 # lsma.NOTE_LEVEL, with '-'
KEYS = ("herniation_levels", "bulge_levels", "mentioned_levels", "herniation_unplaced", "global_negative")


def notes():
    """{patient id: note} from the xlsx (empty notes are ''): xl/sharedStrings.xml and the first sheet."""
    m = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(XLSX)
    strings = ["".join(t.text or "" for t in si.iter(m + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
    out = {}
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(m + "row"):
        cells = {}
        for c in row:
            v = c.findtext(m + "v")
            cells[re.match(r"[A-Z]+", c.get("r")).group()] = (
                strings[int(v)] if c.get("t") == "s" and v is not None else
                "".join(t.text or "" for t in c.iter(m + "t")) if c.get("t") == "inlineStr" else v or "")
        if re.fullmatch(r"\d+(\.0)?", cells.get("A", "")):        # the header row has text in A
            out[int(float(cells["A"]))] = cells.get("B", "").strip()
    return out


PAIR = re.compile(r"(?<![A-Za-z0-9])(?P<a>[TD]12|L[1-5])(?:\s*[-/_]\s*(?P<b>L?[1-5]|S1)|\s*(?P<c>L[1-5]|S1))(?!\d|\s*mm)", re.I)
LUMBOSACRAL = re.compile(r"\blumbo-?sacral(?!\s+(?:MRI|spine|region|area))|\bL-?S\s+junction", re.I)
LAST_TWO = re.compile(r"\b(?:last|lower|lowest|bottom)\s+two(?:\s+(?:lumbar\s+)?(?:levels|discs))?|\btwo\s+(?:lower|lowest|last)\s+(?:lumbar\s+)?(?:levels|discs)", re.I)
TO = re.compile(r"\s*(?:to|through|thru|until)\s*", re.I)
HERN = re.compile(r"\b(?:protru|extru|hernia|prolap|sequest)\w*|\bHNP\b", re.I)
BULGE = re.compile(r"\bbulg\w*", re.I)
NEG = re.compile(r"\b(?:no|not|without|negative for|free of|absence of|rule[sd]? out)\b|(?<!\w)r/o(?!\w)", re.I)
CLAUSE = re.compile(r"\b(?:but|however|while|whereas)\b|,(?=\s*(?:no|not|without)\b)", re.I)
HEADING = re.compile(r"^[^:]{0,70}:")
NOT_LUMBAR = re.compile(r"^\W*(?:c\.?\s*spine|cervical|thoracic|dorsal|d\.?\s*spine)", re.I)
LUMBAR = re.compile(r"^\W*(?:lss|lumbar|lumbosacral|ls\s+spine)", re.I)


def vertebra(s):
    s = s.upper()
    if s in ("T12", "D12"):
        return 0
    return VERT.index(s if s[0] in "LS" else "L" + s)


def levels_in(text):
    """Sorted level names the text names, ranges expanded."""
    spans = []                                             # (start, end, first level index, last level index)
    for m in PAIR.finditer(text):
        a, b = vertebra(m.group("a")), vertebra(m.group("b") or m.group("c"))
        if b > a:
            spans.append((m.start(), m.end(), a, b - 1))
    for m in LUMBOSACRAL.finditer(text):
        spans.append((m.start(), m.end(), 5, 5))
    for m in LAST_TWO.finditer(text):
        spans.append((m.start(), m.end(), 4, 5))
    spans.sort()
    found = set()
    for i, (s, e, lo, hi) in enumerate(spans):
        found.update(range(lo, hi + 1))
        if i + 1 < len(spans) and TO.fullmatch(text[e:spans[i + 1][0]]):
            found.update(range(lo, spans[i + 1][3] + 1))
    return sorted(LEVELS[i] for i in found)


def counted(pattern, clause):
    """Whether the clause has a term of the pattern that no negation precedes: (counted, negated)."""
    hits = [not NEG.search(clause[:m.start()]) for m in pattern.finditer(clause)]
    return any(hits), not all(hits) and bool(hits)


def rule_read(note):
    """The deterministic reading described in the module docstring."""
    herni, bulge, mention = set(), set(), set()
    unplaced = negated_unplaced = skipping = False
    section = []
    for line in note.split("\n"):
        if NOT_LUMBAR.match(line):
            skipping = True
        elif LUMBAR.match(line):
            skipping = False
        head = HEADING.match(line)
        if head and not (HERN.search(head.group()) or BULGE.search(head.group())):
            found = levels_in(head.group())
            if found or len(head.group()) <= 30:                 # "LSS MRI :" or "NB :" ends the section
                section = found
            mention.update(found)
            line = line[head.end():]
        if skipping:
            continue
        for sentence in re.split(r"\.(?!\d)|;", line):
            for clause in CLAUSE.split(sentence):
                lv = levels_in(clause)
                mention.update(lv)
                place = lv or section
                h, h_neg = counted(HERN, clause)
                if h:
                    if place:
                        herni.update(place)
                    else:
                        unplaced = True
                elif h_neg and not place:
                    negated_unplaced = True
                if place and counted(BULGE, clause)[0]:
                    bulge.update(place)
    return {"herniation_levels": sorted(herni, key=LEVELS.index), "bulge_levels": sorted(bulge, key=LEVELS.index),
            "mentioned_levels": sorted(mention, key=LEVELS.index), "herniation_unplaced": unplaced,
            "global_negative": negated_unplaced and not herni and not unplaced}


SLIP = re.compile(r"\b\w*listhes\w*|\bslip(?:s|ped|page)?\b(?!\s+disc)", re.I)   # not spondylosis; "slipped disc" is a herniation
OVER = re.compile(r"\b(L[1-5])\s+(?:over|on)\s+(L[1-5]|S1)\b", re.I)
ONE_VERTEBRA = re.compile(r"(?<![A-Za-z0-9])L([1-5])(?![\d/\-_]|\s*mm)", re.I)


def slip_read(note):
    """(reported, levels): whether the free text reports a slip (spondylolisthesis, listhesis,
    anterolisthesis, retrolisthesis, slip, slipped; a negated term does not count; spondylosis is not
    one) and the set of levels ('L4-L5' style) the clauses that mention it name. 'L4 over L5' is
    L4-L5; a lone vertebra ('anterolisthesis of L4') is the disc below it; a clause with no level
    takes the section heading's. Reported with no levels means the note does not place the slip."""
    reported, found, section = False, set(), []
    for line in note.split("\n"):
        head = HEADING.match(line)
        if head and not SLIP.search(head.group()):
            lv = levels_in(head.group())
            if lv or len(head.group()) <= 30:
                section = lv
            line = line[head.end():]
        for sentence in re.split(r"\.(?!\d)|;", line):
            for clause in CLAUSE.split(sentence):
                if not counted(SLIP, clause)[0]:
                    continue
                reported = True
                clause = OVER.sub(r"\1-\2", clause)
                lv = levels_in(clause)
                if not lv:
                    lv = [LEVELS[int(m.group(1))] for m in ONE_VERTEBRA.finditer(clause)]   # LEVELS[k] is Lk-L(k+1)
                found.update(lv or section)
    return reported, found


def union(a, b):
    return sorted(set(a) | set(b), key=LEVELS.index)


def merge(rule, swarm, hand):
    """{pid: reading + 'agreement'} for every pid in `rule` ({pid: rule_read, or None for an empty note}).
    `swarm`: a list of readings with pid (or {pid: reading}); `hand`: {"<pid>": {herniation_levels,
    herniation_unplaced, why}}. bulge_levels and mentioned_levels are the union of rule and swarm."""
    swarm = {int(r["pid"]): r for r in swarm} if isinstance(swarm, list) else {int(p): r for p, r in swarm.items()}
    hand = {int(p): r for p, r in (hand or {}).items()}
    out = {}
    for pid, r in rule.items():
        if r is None:
            out[pid] = {"herniation_levels": [], "bulge_levels": [], "mentioned_levels": [], "herniation_unplaced": False,
                        "agreement": "empty"}
            continue
        s = swarm.get(pid)
        merged = {"herniation_levels": r["herniation_levels"], "herniation_unplaced": r["herniation_unplaced"],
                  "bulge_levels": union(r["bulge_levels"], (s or {}).get("bulge_levels", [])),
                  "mentioned_levels": union(r["mentioned_levels"], (s or {}).get("mentioned_levels", []))}
        if s is not None and sorted(s["herniation_levels"]) == r["herniation_levels"] \
                and bool(s["herniation_unplaced"]) == r["herniation_unplaced"]:
            merged["agreement"] = "agree"
        elif pid in hand:
            merged.update(herniation_levels=sorted(hand[pid]["herniation_levels"], key=LEVELS.index),
                          herniation_unplaced=bool(hand[pid]["herniation_unplaced"]), agreement="hand")
        else:
            merged.update(herniation_levels=[], herniation_unplaced=False, agreement="unresolved")
        out[pid] = merged
    return out


def labels(merged):
    """Rows (patient, level, herniation, bulge, agreement) for L1-L2..L5-S1 of every pid; '' is unknown."""
    rows = []
    for pid in sorted(merged):
        m = merged[pid]
        for lv in DISC_LEVELS:
            unknown = m["agreement"] in ("empty", "unresolved") or m["herniation_unplaced"]
            rows.append({"patient": pid, "level": lv,
                         "herniation": 1 if lv in m["herniation_levels"] else "" if unknown else 0,
                         "bulge": "" if m["agreement"] == "empty" else int(lv in m["bulge_levels"]),
                         "agreement": m["agreement"]})
    return rows


def load_swarm():
    """(readings, job id) from swarm.json: a list, or {"job": id, "readings": [...]}."""
    d = json.load(open(SWARM, encoding="utf-8"))
    if isinstance(d, dict):
        return d.get("readings") or d.get("items") or d.get("results") or [], d.get("job") or d.get("job_id") or SWARM_JOB
    return d, SWARM_JOB


def build():
    n = notes()
    rule = {pid: (rule_read(t) if t else None) for pid, t in n.items()}
    swarm, job = load_swarm()
    hand = json.load(open(HAND, encoding="utf-8")) if os.path.exists(HAND) else {}
    merged = merge(rule, swarm, hand)
    by_swarm = {int(r["pid"]): r for r in swarm}
    show = lambda r: json.dumps({k: r.get(k) for k in KEYS}) if r else ""
    os.makedirs(WORK, exist_ok=True)
    with open(os.path.join(WORK, "disagree.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pid", "rule reading", "swarm reading", "swarm evidence", "note"])
        for pid, m in sorted(merged.items()):
            if m["agreement"] not in ("agree", "empty"):
                w.writerow([pid, show(rule[pid]), show(by_swarm.get(pid)), (by_swarm.get(pid) or {}).get("evidence", ""), n[pid]])
    rows = labels(merged)
    with open(LABELS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient", "level", "ivd", "herniation", "bulge", "agreement", "reader"])
        for r in rows:
            w.writerow([r["patient"], r["level"], IVD.get(r["level"], ""), r["herniation"], r["bulge"], r["agreement"],
                        READER + job])
    count = lambda s: sum(m["agreement"] == s for m in merged.values())
    print(f"notes {len(n)}, empty {count('empty')}, agree {count('agree')}, hand {count('hand')}, unresolved {count('unresolved')}")
    print("patients with any herniation", sum(bool(m["herniation_levels"]) for m in merged.values()))
    for ivd in (1, 2, 3):
        h = [r["herniation"] for r in rows if IVD.get(r["level"]) == ivd]
        print(f"ivd {ivd}: herniation 1 {h.count(1)}, 0 {h.count(0)}, unknown {h.count('')}")
    if count("unresolved"):
        print("UNRESOLVED", count("unresolved"))
        sys.exit(2)


def gate():
    """Every 'Central Protrusion' LSMA-PQR's structured notes type must be a 1 in the labels."""
    import pandas as pd
    import lsma
    s = pd.read_csv(os.path.join(lsma.ROOT, "structured_notes.csv"))
    got = {(int(r["patient"]), r["level"]): r["herniation"] for r in csv.DictReader(open(LABELS, encoding="utf-8"))}
    n = notes()
    k = total = 0
    for lv in ("L4_L5", "L5_S1"):
        for pid in s.Patient_ID[s[f"{lv}_Disc_Type"] == "Central Protrusion"]:
            total += 1
            if got.get((int(pid), lv.replace("_", "-"))) == "1":
                k += 1
            else:
                print(f"miss {int(pid)} {lv}: {n.get(int(pid), '')[:300]!r}")
    print(f"gate: {k} of {total} ({100 * k / max(total, 1):.1f}%)")
    sys.exit(0 if total and k / total >= 0.95 else 1)


def selftest():
    def read(t):
        return rule_read(t)
    r = read("No evidence of disc herniation.")
    assert r["herniation_levels"] == [] and not r["herniation_unplaced"] and r["global_negative"], r
    r = read("L4-5: diffuse disc bulge with superimposed central protrusion.")
    assert r["herniation_levels"] == ["L4-L5"] and r["bulge_levels"] == ["L4-L5"] and not r["global_negative"], r
    r = read("Small posterior disc protrusion is seen.")
    assert r["herniation_levels"] == [] and r["herniation_unplaced"], r
    r = read("Disc bulges at L3-4 and L4-5, no herniation.")
    assert r["herniation_levels"] == [] and r["bulge_levels"] == ["L3-L4", "L4-L5"] and not r["herniation_unplaced"], r
    assert read("Broad-based disc herniation at L5/S1")["herniation_levels"] == ["L5-S1"]
    r = read("LSS MRI :\nL3-L4: Mild disc bulge noted\nL4-L5: broad based, central and Lt paracentral disc protrusion, "
             "compressing the thecal sac.\nL5-S1: Central disc protrusion.")
    assert r["herniation_levels"] == ["L4-L5", "L5-S1"] and r["bulge_levels"] == ["L3-L4"] and not r["herniation_unplaced"], r
    assert read("Central disc protrusion at the lumbosacral junction.")["herniation_levels"] == ["L5-S1"]
    r = read("Lumbosacral MRI\nNo disc herniation.")
    assert r["mentioned_levels"] == [] and r["global_negative"], r
    assert read("Disc protrusions from L3-4 to L5-S1.")["herniation_levels"] == ["L3-L4", "L4-L5", "L5-S1"]
    assert read("L3-L4, L4-L5 and L5-S1: disc protrusions.")["herniation_levels"] == ["L3-L4", "L4-L5", "L5-S1"]
    assert read("Small central disc protrusion, noted at D12/L1 level, indenting the cord.")["herniation_levels"] == ["T12-L1"]
    assert read("Protrusion at the last two levels.")["herniation_levels"] == ["L4-L5", "L5-S1"]
    r = read("L4/L5& L5/S1levels: \ndiffuse disc bulge , compressing thecal sac")
    assert r["bulge_levels"] == ["L4-L5", "L5-S1"] and r["herniation_levels"] == [], r
    assert read("L5S1 extrusion. L4_L5 herniated disc.")["herniation_levels"] == ["L4-L5", "L5-S1"]
    r = read("C.SPINE MRI :\nDisc protrusion at C5/C6.\nLSS MRI :\nDisc bulge at L4/L5.")
    assert r["herniation_levels"] == [] and not r["herniation_unplaced"] and r["bulge_levels"] == ["L4-L5"], r
    r = read("L4-L5 level:\nCentral disc protrusion, compressing the thecal sac\nL5-S1 level:\nNo disc protrusion or extrusion.")
    assert r["herniation_levels"] == ["L4-L5"] and not r["global_negative"], r
    r = read("No thecal sac compression. No disc herniation or protrusion at L5/S1.")
    assert r["herniation_levels"] == [] and not r["global_negative"], r

    # merge and labels, on in-memory readings
    a, b, c, d = read("L5-S1: central protrusion."), read("Small protrusion seen."), read("L4-L5 protrusion."), read("")
    rule = {1: a, 2: b, 3: c, 4: None, 5: read("L3-L4 bulge.")}
    swarm = [dict(a, pid=1, evidence="x"), dict(read("Protrusion at L4-L5."), pid=2), dict(read("Bulge."), pid=3),
             dict(read("L3-L4 bulge."), pid=5)]
    hand = {"2": {"herniation_levels": ["L4-L5"], "herniation_unplaced": False, "why": "the report says L4-L5 further on"}}
    m = merge(rule, swarm, hand)
    assert [m[i]["agreement"] for i in (1, 2, 3, 4, 5)] == ["agree", "hand", "unresolved", "empty", "agree"], m
    lab = {(r["patient"], r["level"]): r for r in labels(m)}
    assert lab[(1, "L5-S1")]["herniation"] == 1 and lab[(1, "L4-L5")]["herniation"] == 0
    assert lab[(2, "L4-L5")]["herniation"] == 1 and lab[(2, "L5-S1")]["herniation"] == 0
    assert lab[(3, "L4-L5")]["herniation"] == "" and lab[(4, "L4-L5")]["herniation"] == "" and lab[(4, "L4-L5")]["bulge"] == ""
    assert lab[(5, "L3-L4")]["bulge"] == 1 and lab[(5, "L3-L4")]["herniation"] == 0
    p = merge({1: b}, [dict(b, pid=1)], {})                     # an unplaced herniation stays unknown at every level
    assert [r["herniation"] for r in labels(p)] == [""] * 5
    assert slip_read("Grade I anterolisthesis of L4 over L5.") == (True, {"L4-L5"})
    assert slip_read("Spondylotic changes at L4-L5.") == (False, set())
    assert slip_read("L5/S1 spondylolisthesis.") == (True, {"L5-S1"})
    assert slip_read("No spondylolisthesis at L4-5.") == (False, set())
    assert slip_read("Grade 1 retrolisthesis of L3 vertebra.") == (True, {"L3-L4"})
    assert slip_read("L3-4: disc bulge.\nL4-5: grade I listhesis, bulge.") == (True, {"L4-L5"})
    assert slip_read("Spondylolisthesis noted.") == (True, set())
    assert slip_read("Slipped disc at L4-5.") == (False, set())
    print("lsma_text selftest ok")


def slipgate():
    """Counts of the slip label (1 / unknown / 0) per graded level and in all, then 20 discs drawn at
    random (seed 22) from the 1s and unknowns into work/lsma_text/slip_gate.csv, to read by hand."""
    import numpy as np
    import lsma
    s = lsma.slip_items()
    s = s[s.ivd.isin(lsma.NOTE_LEVEL)].assign(level=lambda t: t.ivd.map(lsma.NOTE_LEVEL).str.replace("_", "-"))
    lab = s.t_spondylolisthesis
    s = s.assign(label=np.where(lab.isna(), "unknown", np.where(lab == 1, "1", "0")))
    for name, part in [(lv, s[s.level == lv]) for lv in ("L3-L4", "L4-L5", "L5-S1")] + [("all", s)]:
        print(f"{name}: 1 {int((part.label == '1').sum())}, unknown {int((part.label == 'unknown').sum())}, 0 {int((part.label == '0').sum())}")
    print("patients with a 1:", s[s.label == "1"].patient.nunique())
    pool = s[s.label != "0"].sort_values(["patient", "ivd"]).reset_index(drop=True)
    pick = pool.iloc[np.random.default_rng(22).choice(len(pool), size=min(20, len(pool)), replace=False)]
    covers, n = lsma.slip_covers(), notes()
    structured = lsma.structured_slip()
    os.makedirs(WORK, exist_ok=True)
    with open(os.path.join(WORK, "slip_gate.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient", "level", "label", "max region cover", "structured Spondylolisthesis", "note"])
        for r in pick.itertuples():
            w.writerow([r.patient, r.level, r.label, round(covers.get((r.patient, r.ivd), 0.0), 3),
                        structured.get(r.patient, 0), n.get(r.patient, "")])
    print(f"wrote {os.path.join(WORK, 'slip_gate.csv')} ({len(pick)} discs)")
    sys.exit(0)


if __name__ == "__main__":
    {"build": build, "gate": gate, "selftest": selftest, "slipgate": slipgate}.get(sys.argv[1] if sys.argv[1:] else "", lambda: sys.exit(__doc__))()
