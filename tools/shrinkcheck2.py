"""Second own check of the data session's lossless re-encoding (core/store.py), for what it shrank after the first
check (register 41, CT only): mammograms (CMMD 8-bit, CBIS-DDSM 16-bit), MR (Firat), more CT.

Part A, files ALREADY shrunk in place (their originals are gone): store.py verified each against its original with
pydicom's own JPEG-LS decoder at write time. An independent decoder (GDCM, through SimpleITK) must decode the same
stored values, so the stream is valid JPEG-LS for any reader, not only for the library that wrote it. Compared after
the modality LUT (rescale) on both sides, and the raw stored values where there is no rescale.
Part B, CMMD files the shrink LEFT alone: their reason (store._reason_to_leave), and for any still uncompressed, the
round trip on a scratch COPY compared with the original by both decoders.
Reads live files only; writes only under the scratch folder.

Usage: python tools/shrinkcheck2.py <scratch folder> [files per set, default 25]   (register rows 41 and 47)"""
import os, random, shutil, sys, time, collections
import numpy as np, pydicom, SimpleITK as sitk
from pydicom.pixels import apply_modality_lut
sys.path.insert(0, '.')
from core import store

SP = sys.argv[1]
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 25
PUB = 'D:/NEWProjects/active/rAidiology/Datasets/public'
SETS = ['cmmd', 'cbis-ddsm', 'lumbar-axial-stenosis-firat', 'pancreas-ct', 'mediastinal-lymph-node-seg',
        'prostate-3t', 'cptac-ccrcc-tumor-annotations', 'adrenal-acc-ki67-seg', 'pediatric-ct-seg']
LS = {'1.2.840.10008.1.2.4.80', '1.2.840.10008.1.2.4.81'}
rng = random.Random(0)
out = os.path.join(SP, 'shrinkcheck2'); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)


def sample_files(root, k, want_ls=True, cap=4000):
    """Up to k files of root whose transfer syntax is (want_ls) or is not JPEG-LS, drawn from the first `cap` DICOMs."""
    pool = []
    for i, p in enumerate(store.dicom_files(root)):
        if i >= cap:
            break
        pool.append(p)
    rng.shuffle(pool)
    got = []
    for p in pool:
        try:
            ts = str(pydicom.dcmread(p, stop_before_pixels=True).file_meta.TransferSyntaxUID)
        except Exception:
            continue
        if (ts in LS) == want_ls:
            got.append(p)
        if len(got) >= k:
            break
    return got


def both(path):
    ds = pydicom.dcmread(path)
    a = ds.pixel_array
    s = sitk.GetArrayFromImage(sitk.ReadImage(path))
    if s.ndim == 3 and s.shape[0] == 1 and a.ndim == 2:
        s = s[0]
    return ds, a, s


total = collections.Counter()
for name in SETS:
    root = os.path.join(PUB, name)
    if not os.path.isdir(root):
        print('%-32s not on disk' % name); continue
    files = sample_files(root, PER)
    ok = bad = 0; t_py = t_gd = 0.0; bits = collections.Counter(); notes = []
    for p in files:
        try:
            t0 = time.perf_counter(); ds = pydicom.dcmread(p); a = ds.pixel_array; t_py += time.perf_counter() - t0
            t0 = time.perf_counter(); s = sitk.GetArrayFromImage(sitk.ReadImage(p)); t_gd += time.perf_counter() - t0
            if s.ndim == 3 and s.shape[0] == 1 and a.ndim == 2:
                s = s[0]
            bits[(int(ds.BitsAllocated), int(ds.BitsStored), int(ds.PixelRepresentation), str(ds.PhotometricInterpretation))] += 1
            m = np.asarray(apply_modality_lut(a, ds), np.float64)
            same = s.shape == m.shape and np.array_equal(s.astype(np.float64), m)
            if not same and s.shape == a.shape:      # GDCM may leave a no-op rescale out, or apply it: either agreement counts
                same = np.array_equal(s.astype(np.float64), a.astype(np.float64))
            ok += same; bad += (not same)
            if not same and len(notes) < 3:
                notes.append('%s: shapes %s %s, max |diff| %s' % (os.path.relpath(p, root)[:60], s.shape, a.shape,
                             float(np.abs(s.astype(np.float64) - m).max()) if s.shape == m.shape else 'n/a'))
        except Exception as e:  # noqa: BLE001
            bad += 1; notes.append('%s: %s %s' % (os.path.relpath(p, root)[:60], type(e).__name__, str(e)[:80]))
    n = ok + bad
    total['files'] += n; total['bad'] += bad
    print('%-32s A: %2d shrunk files, GDCM equals pydicom on %2d, differs %d; decode %.0f ms pydicom, %.0f ms GDCM; %s'
          % (name, n, ok, bad, 1e3 * t_py / max(n, 1), 1e3 * t_gd / max(n, 1), dict(bits)))
    for x in notes:
        print('      ', x)

# Part B: CMMD's files left alone
root = os.path.join(PUB, 'cmmd')
left = sample_files(root, PER, want_ls=False, cap=6000)
why = collections.Counter()
rt = rt_bad = 0
for i, p in enumerate(left):
    ds = pydicom.dcmread(p)
    r = store._reason_to_leave(ds)
    why[r or 'none (uncompressed, would shrink)'] += 1
    if r is None and rt < 8:
        a_path = os.path.join(out, 'cmmd_%d.dcm' % i); b_path = os.path.join(out, 'cmmd_%d_ls.dcm' % i)
        shutil.copy2(p, a_path); shutil.copy2(p, b_path)
        outcome, old, new = store.shrink_file(b_path)
        _, A, SA = both(a_path); _, B, SB = both(b_path)
        same = np.array_equal(A, B) and A.dtype == B.dtype and np.array_equal(SA, SB)
        rt += 1; rt_bad += (not same)
        print('cmmd left-alone copy %d: %s, %.0f KB -> %.0f KB, pydicom and GDCM identical to the original: %s'
              % (i, outcome[:24], old / 1e3, new / 1e3, same))
print('B: CMMD files not JPEG-LS, sampled %d: reasons %s; round trips %d, differing %d' % (len(left), dict(why), rt, rt_bad))
print('TOTAL part A: %d shrunk files read by both decoders, %d differ' % (total['files'], total['bad']))
shutil.rmtree(out, ignore_errors=True)
