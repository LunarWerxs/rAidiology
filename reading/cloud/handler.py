"""rAidiology's CT reading as a RunPod Serverless worker: a scan in, what it found out, the scan deleted.

The worker loads the segmenter's five fold models once, when it starts; that is most of a cold start. It keeps them for
every request it serves until the endpoint scales back to zero. A request names its scan by an https URL (a short-lived
link the visitor's browser uploaded the scan to, on a host in RAID_FETCH_HOSTS; redirects are refused) or, for the speed
test, by a file in the volume's samples/ folder. The
scan is read the way `modalities/ct/eval/segmenter.py scan` reads one (read_scan there is the reference; this file
repeats its steps without the printing) and is deleted when the read ends, whether it worked or not. Nothing about a
scan is logged.

Input:  {"url": "https://..."} or {"sample": "s0654.nii.gz"}
Output: {"seconds": {...}, "found": [{"structure": ..., "millilitres": ...}], "gpu": ..., "not_a_medical_device": true}

Layout on the network volume (RAID_SERVE_HOME, default /runpod-volume/raid-serve): repo/ (core/ and modalities/ct/eval/),
Models/ct-segmenter-v<N>/, samples/, and pylib/ (the packages the base image lacks, on PYTHONPATH).
"""
import os
import shutil
import sys
import tempfile
import time
import urllib.parse
import urllib.request

BOOT = time.time()
HOME = os.environ.get("RAID_SERVE_HOME", "/runpod-volume/raid-serve")
REPO = os.path.join(HOME, "repo")
sys.path[:0] = [os.path.join(REPO, "modalities", "ct", "eval"), REPO]
os.environ.setdefault("RAIDIOLOGY_MODELS", os.path.join(HOME, "Models"))
os.environ.setdefault("RAIDIOLOGY_WORK", "/tmp/raid-work")
os.environ.setdefault("RAIDIOLOGY_DATASETS", "/tmp/raid-datasets")
os.environ.setdefault("RAIDIOLOGY_DEVICE", "cuda")
MAX_BYTES = int(os.environ.get("RAID_MAX_SCAN_BYTES", str(2 << 30)))
# The only hosts a scan link may name (comma-separated; a leading '.' means any subdomain): the storage the browser
# uploads to, Cloudflare R2's signed links by default. Anything else is refused before a request is made.
FETCH_HOSTS = tuple(h.strip().lower() for h in os.environ.get("RAID_FETCH_HOSTS", ".r2.cloudflarestorage.com").split(",")
                    if h.strip())

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import runpod  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

IMPORTED = time.time()
import prepare  # noqa: E402
import segmenter  # noqa: E402

VERSION = int(os.environ.get("RAID_CT_VERSION", "2"))
segmenter.steady_inference()
MODELS, META = segmenter.load_models(VERSION, range(segmenter.FOLDS))
NAMES = META["structures"]
READING = segmenter.reading_of(META)
if torch.cuda.is_available():
    torch.cuda.synchronize()
READY = time.time()
SERVED = 0


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """A signed link points straight at the scan, so a redirect is an error, never a second request."""

    def redirect_request(self, *args, **kwargs):
        return None


OPENER = urllib.request.build_opener(NoRedirect)


def allowed(url):
    """True for an https link to one of FETCH_HOSTS (an entry starting with '.' allows that domain's subdomains)."""
    u = urllib.parse.urlsplit(url)
    host = (u.hostname or "").lower()
    return u.scheme == "https" and bool(host) and any(
        host.endswith(h) if h.startswith(".") else host == h for h in FETCH_HOSTS)


def fetch(url, path):
    """The scan from an https link on the upload storage, refused past MAX_BYTES; the worker requests nothing else."""
    if not allowed(url):
        raise ValueError("only https links to the upload storage are read")
    got = 0
    with OPENER.open(url, timeout=120) as r, open(path, "wb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            got += len(block)
            if got > MAX_BYTES:
                raise ValueError("scan larger than %d bytes" % MAX_BYTES)
            f.write(block)


@torch.no_grad()
def read(path):
    img = nib.as_closest_canonical(nib.load(path))
    ct = np.asanyarray(img.dataobj).astype(np.float32)
    zooms = tuple(float(v) for v in img.header.get_zooms()[:3])
    measured = all(abs(v - prepare.SPACING) < 1e-3 for v in zooms)
    fine = ct
    if not measured:
        size = [max(2, int(round(s * v / prepare.SPACING))) for s, v in zip(ct.shape, zooms)]
        fine = F.interpolate(torch.as_tensor(ct)[None, None], size=size, mode="trilinear",
                             align_corners=False)[0, 0].numpy()
    lo = prepare.average_down(np.rint(fine).astype(np.int16))
    acc, weight = segmenter.read_volume(MODELS, lo, META["cfg"]["patch"], READING, NAMES)
    pred = segmenter.under_the_floor(segmenter.labels_on_grid(acc, weight, tuple(fine.shape)), READING["floor"], NAMES)
    if not measured:
        pred = F.interpolate(pred[None, None].float(), size=ct.shape, mode="nearest")[0, 0].to(torch.uint8)
    found = torch.bincount(pred.reshape(-1).long(), minlength=len(NAMES) + 1)[1:].cpu().numpy()
    voxel_ml = float(np.prod(zooms)) / 1000.0
    listing = [{"structure": n, "millilitres": round(float(found[i]) * voxel_ml, 1)}
               for i, n in enumerate(NAMES) if found[i]]
    return listing, measured, list(ct.shape)


def handler(job):
    global SERVED
    inp = job.get("input") or {}
    first = SERVED == 0
    SERVED += 1
    t0 = time.time()
    work = tempfile.mkdtemp(prefix="scan-")
    try:
        path = os.path.join(work, "scan.nii.gz")
        if inp.get("url"):
            fetch(inp["url"], path)
        elif inp.get("sample"):
            shutil.copyfile(os.path.join(HOME, "samples", os.path.basename(inp["sample"])), path)
        else:
            return {"error": "give url or sample"}
        t1 = time.time()
        found, measured, shape = read(path)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.time()
        return {"seconds": {"fetch": round(t1 - t0, 3), "read": round(t2 - t1, 3),
                            "imports": round(IMPORTED - BOOT, 3), "load_models": round(READY - IMPORTED, 3)},
                "first_request_on_this_worker": first, "shape": shape, "measured_spacing": measured,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "torch": torch.__version__, "version": VERSION, "found": found, "not_a_medical_device": True}
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
