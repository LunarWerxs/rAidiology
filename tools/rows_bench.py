"""
Idea 53 bench: core.rows.FileRows (file reads) against np.memmap on a uint8 row cache.
Processor only: never touches CUDA.

Usage:
  python tools/rows_bench.py [--cache work/eval/chest/taix_512.u8] [--rows N --size N]
                             [--batch 64] [--batches 200] [--rounds 4] [--check 50] [--seed 0]
  python tools/rows_bench.py --selftest
  (internal) python tools/rows_bench.py arm --mode memmap|file --seed S ...

The shape is (rows, size, size) uint8, taken from the json beside the cache unless given.
First one process checks that both readers return equal bytes. Then each round runs a
memmap arm and a file arm, each in a fresh child process (order alternates by round),
and the arm reports its end-of-run peak working set and private bytes from psapi.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import contextlib
import ctypes
import io
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import zlib
from ctypes import wintypes

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.rows import FileRows  # noqa: E402

GB = 2 ** 30
NO_WINDOW = 0x08000000 if os.name == 'nt' else 0


class PMCEX(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
        ('PrivateUsage', ctypes.c_size_t),
    ]


def memory_gb():
    """(peak working set, private usage) of this process in GiB, from psapi."""
    if os.name != 'nt':
        return 0.0, 0.0
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMCEX), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    c = PMCEX()
    c.cb = ctypes.sizeof(c)
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return c.PeakWorkingSetSize / GB, c.PrivateUsage / GB


def shape_of(args):
    rows, size = args.rows, args.size
    if rows is None or size is None:
        with open(os.path.splitext(args.cache)[0] + '.json') as f:
            info = json.load(f)
        rows = rows if rows is not None else info['rows']
        size = size if size is not None else info['size']
    return int(rows), int(size)


def draw(rows, batch, batches, seed):
    return np.random.default_rng(seed).integers(0, rows, (batches, batch))


def arm(mode, cache, rows, size, batch, batches, seed):
    idx = draw(rows, batch, batches, seed)
    shape = (rows, size, size)
    reader = np.memmap(cache, dtype=np.uint8, mode='r', shape=shape) if mode == 'memmap' \
        else FileRows(cache, shape)
    crc = 0
    t0 = time.perf_counter()
    for b in range(batches):
        crc = zlib.crc32(np.ascontiguousarray(reader[idx[b]]).tobytes(), crc)
    dt = time.perf_counter() - t0
    peak, private = memory_gb()
    return {'mode': mode, 'seed': seed, 'batches': batches,
            'batches_per_s': batches / dt if dt > 0 else 0.0,
            'peak_ws_gb': peak, 'private_gb': private, 'crc': crc}


def check(cache, rows, size, batch, n, seed=12345):
    shape = (rows, size, size)
    mm = np.memmap(cache, dtype=np.uint8, mode='r', shape=shape)
    fr = FileRows(cache, shape)
    idx = draw(rows, batch, n, seed)
    for b in range(n):
        if not np.array_equal(mm[idx[b]], fr[idx[b]]):
            print(f'check: MISMATCH at batch {b}')
            return False
    fr.close()
    del mm
    print(f'check: {n} batches byte-identical')
    return True


def run_all(args):
    rows, size = shape_of(args)
    if not check(args.cache, rows, size, args.batch, args.check):
        return 1
    ratios = []
    for r in range(args.rounds):
        modes = ['memmap', 'file'] if r % 2 == 0 else ['file', 'memmap']
        res = {}
        for p, mode in enumerate(modes):
            cmd = [sys.executable, os.path.abspath(__file__), 'arm', '--mode', mode,
                   '--seed', str(args.seed + 1000 * r + p), '--cache', args.cache,
                   '--rows', str(rows), '--size', str(size),
                   '--batch', str(args.batch), '--batches', str(args.batches)]
            cp = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW)
            if cp.returncode != 0:
                print(f'arm {mode} failed:\n{cp.stderr}')
                return 1
            res[mode] = json.loads(cp.stdout.strip().splitlines()[-1])
            print(json.dumps(res[mode]))
        ratio = res['file']['batches_per_s'] / res['memmap']['batches_per_s']
        mapped = res['file']['peak_ws_gb'] - res['file']['private_gb']
        ratios.append((ratio, mapped))
        print(f'round {r}: file/memmap throughput {ratio:.2f}; '
              f'file arm mapped working set {mapped:.2f} GB (peak - private)')
    median = statistics.median(x for x, _ in ratios)
    print(f'median file/memmap throughput {median:.2f}')
    kept = all(m < 1.0 for _, m in ratios) and median >= 0.9
    print("Verdict (idea 53, equal bytes, the file arm's mapped working set under 1 GB "
          f"in every round, median throughput at least 0.9x): {'KEPT' if kept else 'NOT KEPT'}")
    return 0


def selftest():
    d = tempfile.mkdtemp(prefix='rows_bench_')
    try:
        cache = os.path.join(d, 'cache.u8')
        np.random.default_rng(0).integers(0, 256, (3000, 64, 64), dtype=np.uint8).tofile(cache)
        with open(os.path.join(d, 'cache.json'), 'w') as f:
            json.dump({'rows': 3000, 'size': 64}, f)
        args = parser().parse_args(['--cache', cache, '--batch', '16', '--batches', '20',
                                    '--rounds', '2', '--check', '5'])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = run_all(args)
        out = buf.getvalue()
        print(out, end='')
        assert rc == 0, 'run_all failed'
        assert 'byte-identical' in out
        assert sum(1 for ln in out.splitlines() if ln.startswith('round ')) == 2
        assert any('median' in ln for ln in out.splitlines())
        assert 'Verdict (idea 53' in out
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print('rows_bench selftest ok')
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', default='work/eval/chest/taix_512.u8')
    p.add_argument('--rows', type=int)
    p.add_argument('--size', type=int)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--batches', type=int, default=200)
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--check', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--selftest', action='store_true')
    return p


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'arm':
        p = parser()
        p.add_argument('--mode', choices=['memmap', 'file'], required=True)
        a = p.parse_args(sys.argv[2:])
        rows, size = shape_of(a)
        print(json.dumps(arm(a.mode, a.cache, rows, size, a.batch, a.batches, a.seed)))
        return 0
    a = parser().parse_args()
    return selftest() if a.selftest else run_all(a)


if __name__ == '__main__':
    sys.exit(main())
