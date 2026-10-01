"""The TotalSpineSeg step, shared by the patient pipeline (run_pipeline.py) and the SPIDER
evaluation (eval/), so both segment the same way.

TotalSpineSeg's released weights are "permission pending" (LICENSING.md): anything that depends
on them is internal development and evaluation only until NeuroPoly agrees in writing."""
import glob
import os
import shutil
import subprocess
import sys

PERMISSION = "TSS weights: permission pending"


def executable():
    return shutil.which("totalspineseg") or os.path.join(os.path.dirname(sys.executable), "totalspineseg")


def weights_release(data_dir):
    """The release tag of the installed weights, e.g. 'r20260730', or None."""
    tags = sorted(os.path.basename(p) for p in glob.glob(os.path.join(data_dir, "nnUNet", "results", "r*")))
    return tags[-1] if tags else None


def segment(in_path, out_dir, data_dir, keep_only=None, quiet=False, workers=None):
    """Run TotalSpineSeg on a .nii.gz or a folder of them; returns the step-2 label folder, whose
    files have the input names and the input geometry. `workers` caps its process pools (by
    default one per core, which on a shared box can run it out of memory)."""
    args = [executable(), in_path, out_dir, "--data-dir", data_dir]
    if workers:
        args += ["--max-workers", str(workers), "--max-workers-nnunet", "1"]
    if keep_only:
        args += ["--keep-only", *keep_only]
    if quiet:
        args.append("--quiet")
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)
    return os.path.join(out_dir, "step2_output")


def chunk_workers(out_dir, default=1):
    """Worker count for the next chunk: <out_dir>/workers.txt if present (so it can be turned up or
    down while a long run goes on, as the shared box's free memory changes), else `default`."""
    try:
        return max(1, int(open(os.path.join(out_dir, "workers.txt")).read().strip()))
    except (OSError, ValueError):
        return default


def segment_chunked(in_dir, out_dir, data_dir, chunk=10, retries=4, pause_s=90):
    """segment() over a folder in chunks of `chunk` volumes, skipping volumes that already have a
    step-2 output. Workers per chunk come from chunk_workers(); a chunk that fails (on a shared box,
    usually a memory dip) is retried after a pause with one worker; finished chunks are never redone."""
    import shutil
    import time
    final = os.path.join(out_dir, "step2_output")
    os.makedirs(final, exist_ok=True)
    todo = sorted(f for f in os.listdir(in_dir) if f.endswith(".nii.gz") and not os.path.exists(os.path.join(final, f)))
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        tmp_in, tmp_out = os.path.join(out_dir, "_chunk_in"), os.path.join(out_dir, "_chunk_out")
        for attempt in range(retries):
            shutil.rmtree(tmp_in, ignore_errors=True)
            shutil.rmtree(tmp_out, ignore_errors=True)
            os.makedirs(tmp_in)
            for f in part:
                shutil.copy(os.path.join(in_dir, f), tmp_in)
            try:
                segment(tmp_in, tmp_out, data_dir, keep_only=["step2_output"], quiet=True,
                        workers=chunk_workers(out_dir) if attempt == 0 else 1)
                break
            except subprocess.CalledProcessError:
                print(f"chunk {i // chunk} failed (attempt {attempt + 1}), retrying in {pause_s}s", flush=True)
                time.sleep(pause_s)
        else:
            raise RuntimeError(f"TotalSpineSeg failed {retries} times on {part}")
        for f in part:
            shutil.move(os.path.join(tmp_out, "step2_output", f), os.path.join(final, f))
        print(f"TotalSpineSeg: {min(i + chunk, len(todo))}/{len(todo)} volumes", flush=True)
    return final
