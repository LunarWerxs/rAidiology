"""Merge the synthesized reading (reading.json) with the pipeline's meshes and anchors into
the app's data/report.json.

reading.json is report.json minus the geometry: each finding carries
  "anchor_ref": {"mesh": "disc_L4_L5", "point": "posterior" | "centroid" | "posterior_left" | "posterior_right",
                 "offset": [dx, dy, dz]}
and each region omits "meshes". This script resolves the anchor to millimetres in the app frame,
lists the region meshes, and colours each disc mesh by its level status.
A view may be {"stack": id, "at": "posterior" | "centroid" | ..., "mark": "circle", "r": 0.05,
"arrow": true, "caption": "..."}: the finding's anchor (or the named anchor point) is projected onto
the nearest slice of that stack using the DICOM geometry sidecar, giving slice and mark position.
Usage: python make_report.py <reading.json> <site_dir> <geometry.json>"""
import glob
import json
import os
import sys

RANK = {"normal": 0, "normal-variant": 0, "incidental": 0, "mild": 1, "moderate": 2, "severe": 3}
STATUS = {0: "normal", 1: "mild", 2: "moderate", 3: "severe"}


def place_views(f, anchor_pts, stacks, geo):
    from geometry import project
    views = []
    for v in f.get("views", []):
        if "slice" in v:
            views.append(v)
            continue
        st = stacks.get(v["stack"])
        pt = anchor_pts.get(v.get("at", "anchor"))
        if not st or pt is None:
            print("WARN view skipped", f["id"], v)
            continue
        k, x, y, dist = project(pt, geo[st["source_series"]])
        r = v.get("r", 0.05)
        marks = [{"type": v.get("mark", "circle"), "x": x, "y": y, "r": r}]
        if v.get("arrow", True):
            marks.append({"type": "arrow", "x": x + r * 0.7, "y": y - r * 0.7, "dx": 0.07, "dy": -0.07})
        views.append({"stack": v["stack"], "slice": k, "caption": v.get("caption", st["label"]), "marks": marks})
    f["views"] = views


def _load_meshes(r, site, level_status):
    anchors = {}
    for reg in r["regions"]:
        p = os.path.join(site, "data", "meshes", reg["id"], "anchors.json")
        if not os.path.exists(p):
            reg["meshes"] = []
            continue
        a = json.load(open(p, encoding="utf-8"))
        anchors[reg["id"]] = a["anchors"]
        meshes = []
        for m in a["meshes"]:
            m = dict(m)
            if m["kind"] == "disc":
                m["status"] = level_status.get((reg["id"], m["name"][5:]), "normal")
            meshes.append(m)
        reg["meshes"] = meshes
    return anchors


def _place_findings(r, anchors, stacks, geo):
    for f in r["findings"]:
        ref = f.pop("anchor_ref", None)
        if not ref:
            continue
        a = anchors.get(f["region"], {}).get(ref["mesh"])
        if not a:
            print("WARN no anchor", f["id"], ref)
            continue
        pt = list(a.get(ref.get("point", "centroid"), a["centroid"]))
        pt = [round(pt[i] + ref.get("offset", [0, 0, 0])[i], 1) for i in range(3)]
        f["anchor"] = {"mesh": ref["mesh"], "point": pt}
        pts = {k2: v2 for k2, v2 in a.items() if isinstance(v2, list)}
        pts["anchor"] = pt
        place_views(f, pts, stacks, geo)
        # a finding at a vertebra or elsewhere also tints that mesh
        for reg in r["regions"]:
            if reg["id"] != f["region"]:
                continue
            for m in reg["meshes"]:
                if m["name"] == ref["mesh"] and f["severity"] in RANK:
                    m["status"] = STATUS[max(RANK.get(m["status"], 0), RANK[f["severity"]])]


def _attach_measurements(r, work):
    # automated per-level measurements (metrics.py) ride along on each level row
    codes = {"cervical": "C", "thoracic": "T", "lumbar": "L"}
    for region, code in codes.items():
        found = sorted(glob.glob(os.path.join(work, f"metrics_{code}_*SAG_T2*.json")))
        if not found:
            continue
        by_level = {m["level"]: m for m in json.load(open(found[0]))["levels"]}
        for lv in r.get("levels", []):
            m = by_level.get(lv["level"]) if lv["region"] == region else None
            if m:
                lv["measure"] = {k: m[k] for k in ("disc_height_mm", "disc_to_csf_signal", "canal_ap_at_disc_mm",
                                                   "cord_ap_at_disc_mm") if m.get(k) is not None}


def _attach_sct(r, work):
    # cervical cord vs healthy adults (modalities/spine_mri/sct/cord_compression.py), only where it measured one
    sct_p = os.path.join(work, "sct_compression.json")
    if os.path.exists(sct_p):
        for level, v in json.load(open(sct_p)).get("levels", {}).items():
            for lv in r.get("levels", []):
                if lv["region"] == "cervical" and lv["level"] == level and v.get("cord_compression_ratio_hc") is not None:
                    lv.setdefault("measure", {})["cord_compression_ratio_hc"] = v["cord_compression_ratio_hc"]


def _attach_reading_extras(r, reading_p):
    # optional: blind reading vs an outside report vs the re-check (see comparison.json)
    cmp_p = os.path.join(os.path.dirname(os.path.abspath(reading_p)), "comparison.json")
    if os.path.exists(cmp_p):
        r["comparison"] = json.load(open(cmp_p, encoding="utf-8"))
    # optional: the AI's own report in radiologist form (cases/<id>/ai_report.json)
    air_p = os.path.join(os.path.dirname(os.path.abspath(reading_p)), "ai_report.json")
    if os.path.exists(air_p):
        r["ai_report"] = json.load(open(air_p, encoding="utf-8"))


def _attach_model(r, m, lv_key, card_key, card_src, label_key, short_key):
    for lv in r.get("levels", []):
        g = m["levels"].get(lv["level"]) if lv["region"] == "lumbar" else None
        if g:
            lv[lv_key] = {"model": m["method"], "not_blind": m["not_blind"], **g}
    r[card_key] = m[card_src]
    if "comparison" in r and m.get("comparison"):
        r["comparison"][label_key] = m["comparison"]["model_label"]
        for row in r["comparison"]["rows"]:
            short = m["comparison"]["rows"].get(row["level"]) if row["region"] == "lumbar" else None
            if short and row.get("topic", "").lower().startswith("disc"):
                row[short_key] = short


def main(reading_p, site, geo_p):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "core", "dicom"))
    geo = json.load(open(geo_p))
    stacks = {e["id"]: e for e in json.load(open(os.path.join(site, "data", "stacks", "index.json")))}
    r = json.load(open(reading_p, encoding="utf-8"))
    level_status = {(lv["region"], lv["level"].replace("-", "_")): lv["status"] for lv in r.get("levels", [])}
    anchors = _load_meshes(r, site, level_status)
    _place_findings(r, anchors, stacks, geo)
    work = os.path.dirname(os.path.abspath(geo_p))
    _attach_measurements(r, work)
    _attach_sct(r, work)
    _attach_reading_extras(r, reading_p)
    # the specialist model, only when one passed the ship rule (modalities/spine_mri/specialist.py writes this)
    spec_p = os.path.join(work, "specialist.json")
    if os.path.exists(spec_p):
        spec = json.load(open(spec_p, encoding="utf-8"))
        _attach_model(r, spec, "specialist", "scorecard", "scorecard", "model_label", "model_short")
    # a development model beside it (specialist.py --dev-version N): no ship rule, never held-out numbers
    dev_p = os.path.join(work, "specialist_dev.json")
    if os.path.exists(dev_p):
        dev = json.load(open(dev_p, encoding="utf-8"))
        _attach_model(r, dev, "specialist_dev", "dev_scorecard", "dev_scorecard", "model_dev_label", "model_dev_short")
    out = os.path.join(site, "data", "report.json")
    json.dump(r, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", out, len(r["findings"]), "findings")


if __name__ == "__main__":
    main(*sys.argv[1:4])
