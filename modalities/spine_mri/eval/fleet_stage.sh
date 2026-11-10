#!/usr/bin/env bash
# Runs ON A FLEET BOX only (never on the owner's PC): stages SPIDER (Zenodo 10159290, CC BY 4.0) and builds the
# development crops the disc grader reads, unless the drive already holds them. Usage: fleet_stage.sh <python>
# NO_TSS=1: the commercial chain (spine_finder_box.sh). Nothing installs, downloads, imports or reads TotalSpineSeg:
# prepare.py, then scope by nominal level (match.py under RAIDIOLOGY_NO_TSS=1, which fails unless 1004 discs of 168
# patients are in scope), then the spider_masks crops alone; a marker says the drive's crops were built this way.
set -euo pipefail
P=${1:-/mnt/raid/venv/bin/python}
export SPIDER_ROOT=${SPIDER_ROOT:-/mnt/raid/Datasets/public/spider}
cd /mnt/raid/code/modalities/spine_mri/eval
CROPS=/mnt/raid/work/eval/spider/dev/crops.npz
BARE=/mnt/raid/work/eval/spider/dev/.staged-no-tss
# The box venv has no pip: uv installs, and it only gains what prepare.py and crops.py read (torch stays as it is).
[ -x /tmp/uvbin/uv ] || curl -fsSL https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/tmp/uvbin UV_NO_MODIFY_PATH=1 sh
U=/tmp/uvbin/uv
"$P" -c "import SimpleITK, nibabel" 2>/dev/null || $U pip install --python "$P" SimpleITK nibabel
[ "${DEPS_ONLY:-0}" != 1 ] || { echo "deps ready"; exit 0; }   # DEPS_ONLY=1: only these, for a smoke before staging
if [ "${NO_TSS:-0}" = 1 ]; then
  export RAIDIOLOGY_NO_TSS=1
  if [ -f "$CROPS" ] && [ -f "$BARE" ]; then echo "no-TotalSpineSeg crops present: $CROPS"; exit 0; fi
elif [ "${DATA_ONLY:-0}" != 1 ] && [ -f "$CROPS" ]; then echo "crops present: $CROPS"; exit 0; fi
if ! ls "$SPIDER_ROOT"/images/*.mha >/dev/null 2>&1 || ! ls "$SPIDER_ROOT"/masks/*.mha >/dev/null 2>&1; then   # not overview.csv: a takeback after it left no images (spine171e-data)
  mkdir -p "$SPIDER_ROOT"
  echo "== downloading SPIDER from Zenodo $(date -u +%T)"
  for f in images.zip masks.zip overview.csv radiological_gradings.csv; do
    # download to .part and rename only when whole: a machine taken back mid-download left a partial images.zip (spine171d-data)
    [ -f "$SPIDER_ROOT/$f" ] || { curl -fsSL --retry 5 -C - -o "$SPIDER_ROOT/$f.part" "https://zenodo.org/records/10159290/files/$f?download=1" && mv "$SPIDER_ROOT/$f.part" "$SPIDER_ROOT/$f"; }
  done
  (cd "$SPIDER_ROOT" && unzip -tq images.zip && unzip -tq masks.zip && unzip -qno images.zip && unzip -qno masks.zip)
  ls "$SPIDER_ROOT" | head
fi
# TotalSpineSeg runs in its own Python 3.12 venv: under the box venv's Python 3.14 its worker pools (forkserver start
# method) die with "SemLock._rebuild FileNotFoundError" -> BrokenProcessPool (spine171c, spine171g-data, 2026-11-07).
if [ "${NO_TSS:-0}" = 1 ]; then
  echo "== SPIDER dev without TotalSpineSeg: prepare, scope by nominal level, spider_masks crops $(date -u +%T)"
  rm -f "$BARE"
  "$P" -u prepare.py dev
  "$P" -u match.py dev          # exits non-zero unless 1004 discs of 168 patients are in scope
  "$P" -u crops.py dev
  "$P" - <<'PYEOF'
import numpy as np, pandas as pd
b = "/mnt/raid/work/eval/spider/dev/"
m = pd.read_csv(b + "matches.csv", keep_default_na=False)
s = m[m.in_scope == 1]
i = pd.read_csv(b + "crops_index.csv")
print("in scope", len(s), "discs of", s.patient.nunique(), "patients; crops by localisation", i.localisation.value_counts().to_dict())
assert (len(s), s.patient.nunique()) == (1004, 168), "in-scope set is not the bar's 1004 discs of 168 patients"
assert set(i.localisation) == {"spider_masks"} and len(i) == 1004, "crops.npz must hold the 1004 spider_masks crops alone"
assert len(np.load(b + "crops.npz")["x"]) == len(i)
PYEOF
  test ! -e /mnt/raid/tssvenv/bin/totalspineseg || echo "note: a TotalSpineSeg venv is on the drive; this chain did not use it"
  touch "$BARE"
  exit 0
fi
TV=/mnt/raid/tssvenv
if [ ! -x "$TV/bin/totalspineseg" ]; then
  rm -rf "$TV"; UV_PYTHON_INSTALL_DIR=/mnt/raid/uv-python $U venv --python 3.12 "$TV"
  $U pip install --python "$TV/bin/python" totalspineseg nnunetv2 "kornia<0.8" SimpleITK nibabel
fi
# On these boxes TotalSpineSeg's tqdm process_map pools die (SemLock FileNotFoundError / a stall) after the model runs, with the
# default start method (fork) and with --no-stalling (spine171h-data, spine171i-data, 2026-11-07). Every one of its steps maps
# files through tqdm.contrib.concurrent.process_map, so that name (set in a .pth file) runs threads instead (thread_map takes the same arguments).
SP=$("$TV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
# a .pth line, not sitecustomize.py: Ubuntu's own /usr/lib/python3.12/sitecustomize.py shadows a venv one (spine171j-data)
echo "import tqdm.contrib.concurrent as _c; _c.process_map = _c.thread_map" > "$SP/zz_tqdm_threads.pth"
"$TV/bin/python" -c "import tqdm.contrib.concurrent as c; assert c.process_map is c.thread_map; print('process_map -> thread_map')"
export PATH="$TV/bin:$PATH"   # tss.executable() finds totalspineseg first on PATH
"$TV/bin/python" -c "import torch, totalspineseg; print('tss venv torch', torch.__version__, torch.cuda.is_available())"
# a takeback mid-download leaves a partial weights zip that TotalSpineSeg then trusts (BadZipFile, spine171k-data): drop bad ones
for z in $(ls /mnt/raid/work/tss_data/*.zip /mnt/raid/work/tss_data/*/*.zip 2>/dev/null); do
  unzip -tq "$z" >/dev/null 2>&1 || { echo "removing incomplete $z"; rm -f "$z"; }
done
if [ "${DATA_ONLY:-0}" = 1 ]; then   # a CPU rehearsal: everything but training, then stop
  timeout 3300 "$P" -u fleet_data_check.py
  df -h /mnt/raid | tail -1; exit 0
fi
echo "== SPIDER dev: prepare, TotalSpineSeg, match, crops $(date -u +%T)"
"$P" -u run_split.py dev
"$P" -u crops.py dev
test -f "$CROPS"
