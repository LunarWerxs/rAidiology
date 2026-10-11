"""Make site/data/models/skeleton-hero.glb (meshopt + quantized) from skeleton.glb with gltfpack."""
import json, shutil, struct, subprocess, sys, tempfile
from pathlib import Path

M = Path(__file__).resolve().parent.parent / 'site/data/models'
SRC, OUT = M / 'skeleton.glb', M / 'skeleton-hero.glb'


def nodes(p):
    b = p.read_bytes()
    n = struct.unpack('<I', b[12:16])[0]
    j = json.loads(b[20:20 + n])
    return len(j.get('nodes', [])), len(j.get('meshes', [])), {x.get('name') for x in j['nodes']}


def main():
    with tempfile.TemporaryDirectory() as t:
        cmd = f'npx --yes gltfpack -i "{SRC}" -o "{OUT}" -cc -kn -km'
        subprocess.run(cmd, cwd=t, shell=True, check=True)
    a, b = nodes(SRC), nodes(OUT)
    print('note: gltfpack adds one named wrapper node per mesh, so node totals differ; mesh count must match')
    print('nodes/meshes', a[:2], b[:2], 'names kept:', a[2] <= b[2])
    print('bytes', SRC.stat().st_size, OUT.stat().st_size)
    if a[1] != b[1] or not a[2] <= b[2] or OUT.stat().st_size > 900_000:
        sys.exit('check failed')


main()
