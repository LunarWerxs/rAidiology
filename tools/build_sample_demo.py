"""Build the home page's compact live CT demo from the open viewer's sample.

Reads site/sample/sample-ct.zip and site/sample/sample-ct.raidiology.json.gz and writes
site/sample/demo/s00.webp .. sNN.webp plus demo.json. Slices are ordered and windowed the way
site/js/open/viewer.js does it (position along the plane normal; the file's own window centre and
width), outline paths are decoded the way findings.js decodePath does, and colours are viewer.js
colourOf. Run `python tools/build_sample_demo.py` to build, `--check` to re-verify (exit 1 on a miss).
"""
import base64
import gzip
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / 'site' / 'sample'
OUT = SAMPLE / 'demo'
WIDTH = 384
FRAME_LIMIT = 650 * 1024
JSON_LIMIT = 120 * 1024
TOLERANCE = 1.5
CREDIT = 'Sample CT from TotalSegmentator (CC BY 4.0). Outlines by an open research model.'


def decode_path(text):
    """findings.js decodePath: base64 of zigzag varints, steps in half pixels."""
    raw = base64.b64decode(text)
    pts, z, scale, x, y, odd = [], 0, 1, 0, 0, False
    for b in raw:
        z += (b & 0x7F) * scale
        scale *= 128
        if b >= 0x80:
            continue
        d = -(z + 1) // 2 if z % 2 == 1 else z // 2
        if odd:
            y += d
            pts.append((x / 2, y / 2))
        else:
            x += d
        odd = not odd
        z, scale = 0, 1
    return pts


def colour_of(s):
    return f"hsl({round((s * 137.508) % 360)} 85% {[60, 44, 74][s % 3]}%)"


def simplify(pts, tol):
    """Douglas-Peucker on a closed ring."""
    if len(pts) < 4:
        return pts
    a = np.array(pts, float)
    far = int(np.argmax(np.hypot(*(a - a[0]).T)))
    if far == 0:
        return pts

    def dp(p):
        if len(p) < 3:
            return p
        s, e = p[0], p[-1]
        d = e - s
        n = np.hypot(*d)
        if n == 0:
            dist = np.hypot(*(p - s).T)
        else:
            dist = np.abs(d[0] * (p[:, 1] - s[1]) - d[1] * (p[:, 0] - s[0])) / n
        i = int(np.argmax(dist))
        if dist[i] <= tol:
            return np.array([s, e])
        return np.vstack([dp(p[: i + 1])[:-1], dp(p[i:])])

    ring = np.vstack([dp(a[: far + 1])[:-1], dp(np.vstack([a[far:], a[:1]]))[:-1]])
    return [tuple(p) for p in ring]


def area(pts):
    a = np.array(pts)
    x, y = a[:, 0], a[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def path_d(pts):
    r = [(round(x), round(y)) for x, y in pts]
    out = [r[0]]
    for p in r[1:]:
        if p != out[-1]:
            out.append(p)
    if len(out) < 3:
        return None
    # relative lineto, numbers packed (a minus sign separates), to keep demo.json small
    nums = [v for (x0, y0), (x1, y1) in zip(out, out[1:]) for v in (x1 - x0, y1 - y0)]
    body = ''
    for v in nums:
        body += (str(v) if v < 0 or not body else ' ' + str(v))
    return f'M{out[0][0]} {out[0][1]}l{body}z'


def load_slices():
    z = zipfile.ZipFile(SAMPLE / 'sample-ct.zip')
    ds = [pydicom.dcmread(io.BytesIO(z.read(n))) for n in sorted(z.namelist()) if not n.endswith('/')]
    o = [float(v) for v in ds[0].ImageOrientationPatient]
    n = [o[1] * o[5] - o[2] * o[4], o[2] * o[3] - o[0] * o[5], o[0] * o[4] - o[1] * o[3]]
    ds.sort(key=lambda d: sum(float(p) * q for p, q in zip(d.ImagePositionPatient, n)))
    return ds


def fingerprint(uid):
    return hashlib.sha256(str(uid).strip().encode()).hexdigest()[:16]


def window(d):
    centre, width = float(d.WindowCenter), float(d.WindowWidth)
    a = d.pixel_array.astype(np.float32) * float(d.RescaleSlope) + float(d.RescaleIntercept)
    g = np.clip((a - (centre - width / 2)) / width, 0, 1) * 255
    return Image.fromarray(g.astype(np.uint8), 'L')


def encode(images, q):
    bufs = []
    for im in images:
        b = io.BytesIO()
        im.save(b, 'WEBP', quality=q, method=6)
        bufs.append(b.getvalue())
    return bufs


def build():
    ds = load_slices()
    doc = json.load(gzip.open(SAMPLE / 'sample-ct.raidiology.json.gz'))
    ol = doc['outlines']
    names, about = ol['structures'], ol['about']
    prints = [fingerprint(d.SOPInstanceUID) for d in ds]
    cols, rows = int(ds[0].Columns), int(ds[0].Rows)
    k = WIDTH / cols
    size = [WIDTH, round(rows * k)]
    images = [window(d).resize(tuple(size), Image.LANCZOS) for d in ds]
    q = 90
    while q > 20:
        bufs = encode(images, q)
        if sum(map(len, bufs)) <= FRAME_LIMIT:
            break
        q -= 2
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob('s*.webp'):
        f.unlink()
    frames = []
    for i, b in enumerate(bufs):
        (OUT / f's{i:02d}.webp').write_bytes(b)
        frames.append(f's{i:02d}.webp')
    slices, used = [], set()
    for i in range(len(ds)):
        row = next((r for r in ol['slices'] if r.get('image_fingerprint') == prints[i]), None)
        if row is None:
            row = next((r for r in ol['slices'] if r.get('image_index') == i), {'paths': []})
        paths, best = [], (0, None)
        for s, text in row['paths']:
            pts = [(x * k, y * k) for x, y in decode_path(text)]
            a = area(pts) if len(pts) > 2 else 0
            d = path_d(simplify(pts, TOLERANCE))
            if d is None:
                continue
            paths.append({'s': s, 'd': d})
            used.add(s)
            m = re.fullmatch(r'vertebra (\w+)', names[s])
            if m and a > best[0]:
                best = (a, m.group(1))
        slices.append({'paths': paths, 'level': best[1]})
    ids = sorted(used)
    out = {
        'count': len(ds), 'size': size, 'frames': frames,
        'structures': [{'id': s, 'name': names[s], 'color': colour_of(s)} for s in ids],
        'slices': slices, 'credit': CREDIT, 'about': {str(i): about[i] for i in ids},
    }
    (OUT / 'demo.json').write_text(json.dumps(out, separators=(',', ':')), encoding='utf-8')
    print(f'quality {q}, frames {sum(map(len, bufs))} bytes')


def check():
    bad = []
    p = OUT / 'demo.json'
    if not p.exists():
        print('missing demo.json')
        return 1
    d = json.loads(p.read_text(encoding='utf-8'))
    if p.stat().st_size > JSON_LIMIT:
        bad.append(f'demo.json {p.stat().st_size} > {JSON_LIMIT}')
    total = 0
    for f in d['frames']:
        fp = OUT / f
        if not fp.exists():
            bad.append(f'missing {f}')
        else:
            total += fp.stat().st_size
    if total > FRAME_LIMIT:
        bad.append(f'frames {total} > {FRAME_LIMIT}')
    if not (d['count'] == len(d['frames']) == len(d['slices']) == 46):
        bad.append('count mismatch')
    ids = {s['id'] for s in d['structures']}
    for sl in d['slices']:
        if any(x['s'] not in ids for x in sl['paths']):
            bad.append('path with unknown structure')
            break
    if chr(0x2014) in p.read_text(encoding='utf-8'):
        bad.append('em dash')
    print('levels', sum(1 for s in d['slices'] if s['level']), 'frames', total, 'json', p.stat().st_size)
    for b in bad:
        print('FAIL', b)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(check() if '--check' in sys.argv else build())
