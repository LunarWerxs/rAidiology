"""Copy the pinned three.js build and the addons the site imports into site/js/vendor/three/.
Re-runnable. Usage: python tools/vendor_three.py <path to node_modules/three at 0.170.0>"""
import json, re, shutil, sys
from pathlib import Path

# The three package to copy from: any node_modules/three at version 0.170.0, passed as the first argument
# (for example `npm pack three@0.170.0` unpacked, or a project's node_modules/three).
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else None
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'site/js/vendor/three'
VERSION = '0.170.0'
IMPORT = re.compile(r"""(?:from\s*|import\s*\(?\s*)['"]([^'"]+)['"]""")


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print('copied', dst.relative_to(ROOT).as_posix())


def main():
    if SRC is None or not (SRC / 'package.json').is_file():
        raise SystemExit('usage: python tools/vendor_three.py <path to node_modules/three at 0.170.0>')
    ver = json.loads((SRC / 'package.json').read_text(encoding='utf-8'))['version']
    if ver != VERSION:
        sys.exit(f'refusing: three is {ver}, need {VERSION}')
    jsm = SRC / 'examples/jsm'
    todo = ['libs/meshopt_decoder.module.js']
    files = list((ROOT / 'site/js').glob('*.js'))
    for f in files:
        for m in IMPORT.findall(f.read_text(encoding='utf-8')):
            if m.startswith('three/addons/'):
                todo.append(m[len('three/addons/'):])
    seen = set()
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        src = (jsm / rel).resolve()
        for m in IMPORT.findall(src.read_text(encoding='utf-8')):
            if m.startswith('.'):
                todo.append((src.parent / m).resolve().relative_to(jsm.resolve()).as_posix())
    copy(SRC / 'build/three.module.min.js', OUT / 'three.module.min.js')
    copy(SRC / 'LICENSE', OUT / 'LICENSE')
    for rel in sorted(seen):
        copy(jsm / rel, OUT / 'addons' / rel)


main()
