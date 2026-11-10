"""Check the live Hugging Face repositories against a release record, as a stranger and as the owner.

For each repository in the record (default: the newest records/models/manifest-*-huggingface.jsonl): it is public,
its gate is manual for a paid card (Schedule A) and automatic for a free one, it holds exactly the record's files
with the same bytes and, for every large file, the same sha256, and a download without a login is refused (401/403).
Exit 1 if any repository fails. Run after every upload:

  python release/huggingface/verify_live.py [CARD ...] [--record PATH]
"""
import glob
import os
import json, sys, urllib.request, urllib.error
from huggingface_hub import HfApi
api = HfApi()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
args = sys.argv[1:]
record = None
if "--record" in args:
    i = args.index("--record"); record = args[i + 1]; del args[i:i + 2]
record = record or sorted(glob.glob(os.path.join(ROOT, "records", "models", "manifest-*-huggingface.jsonl")))[-1]
want = {}
for l in open(record, encoding="utf-8"):
    r = json.loads(l); want.setdefault(r["repo"], {})[r["path"]] = (r["bytes"], r["sha256"])
sys.path.insert(0, HERE)
from publish import paid_cards  # Schedule A is the one list
paid = paid_cards()
cards = args or sorted(k.split("rAidiology-")[1] for k in want)
bad = 0
for card in cards:
    repo = f"LunarWerx/rAidiology-{card}"
    try:
        info = api.model_info(repo, files_metadata=True)
    except Exception as e:
        print(f"{card:<20} NOT UP: {type(e).__name__}"); bad += 1; continue
    have = {s.rfilename: (s.size, s.lfs.sha256 if s.lfs else None) for s in info.siblings}
    weights = {k: v for k, v in have.items() if k not in ("README.md", "LICENSE", ".gitattributes")}
    exp = want[repo]
    missing = sorted(set(exp) - set(weights)); extra = sorted(set(weights) - set(exp))
    wrong = [p for p in exp if p in weights and (weights[p][0] != exp[p][0] or (weights[p][1] and weights[p][1] != exp[p][1]))]
    unhashed = [p for p in exp if p in weights and not weights[p][1]]
    gate_ok = info.gated == ("manual" if card in paid else "auto")
    probe = next(p for p in exp if not p.endswith(".json"))
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/{probe}", method="HEAD")); stranger = "OPEN"
    except urllib.error.HTTPError as e:
        stranger = str(e.code)
    ok = not (missing or extra or wrong) and gate_ok and not info.private and stranger in ("401", "403")
    bad += not ok
    print(f"{card:<20} {'OK ' if ok else 'BAD'} public={not info.private} gate={info.gated} files={len(weights)}/{len(exp)} "
          f"same-fingerprint={len(exp) - len(wrong) - len(missing)} (json by size: {len(unhashed)}) stranger={stranger}"
          + (f" missing={missing[:3]} extra={extra[:3]} wrong={wrong[:3]}" if (missing or extra or wrong) else ""))
sys.exit(1 if bad else 0)
