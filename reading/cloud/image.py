"""Builds, checks and pushes the private container image of the CT reader (reading/cloud/Dockerfile): handler.py, the
repo files it imports, the version 2 weights and three development scans, with runpod and the other packages installed
in the image, so a RunPod Serverless worker starts with no network volume and installs nothing.

  python reading/cloud/image.py stage [--version 2] [--samples s0654.nii.gz,...]
      repo/ holds the .json files of runpod_serverless.repo_file_list() and only the .py files handler.py imports
      rebuilds work/cloud-image/context/ and refuses, naming the file, if a staged .json or .py carries a path of this
      machine or a staged path names the owner's private scan data; prints file count, size and each fold's sha256
  python reading/cloud/image.py build --tag TAG
      docker build of that context; prints the image id and size
  python reading/cloud/image.py test --tag TAG [--sample s0654.nii.gz]
      runs the handler once in the image on the CPU and checks its output: anatomy names only, no pathology classes,
      no measurements, not_a_medical_device true
  python reading/cloud/image.py login --registry ghcr.io --user USER --token-file PATH
      docker login with the token piped through stdin; the token is never printed
  python reading/cloud/image.py push --tag TAG
      docker push; prints the pushed digest
"""
import argparse
import ast
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import runpod_serverless as rs  # noqa: E402

common = rs.common
DOCKERFILE = os.path.join(HERE, "Dockerfile")
CONTEXT = os.path.join(common.paths.WORK, "cloud-image", "context")
LOCAL_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/Users/|/home/", re.I)
PRIVATE_PATH = ("report.json", "meshes", "stacks")
PATHOLOGY = ("cyst", "lesion", "tumor", "tumour", "nodule", "stone", "calcif", "fracture")
MEASURE = re.compile(r"volume|(^|_)ml($|_)|size|score", re.I)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy(src, rel):
    dst = os.path.join(CONTEXT, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def import_closure():
    """The repo .py files handler.py reaches through imports (any import, also those inside functions), relative to
    the repo root. Other core/ modules carry paths of the owner's machine and have no place in the image."""
    roots = [os.path.join(ROOT, "modalities", "ct", "eval"), ROOT]
    seen, todo = set(), ["reading/cloud/handler.py"]
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                names = [x.name for x in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                names = [n.module] + [n.module + "." + x.name for x in n.names]
            else:
                continue
            for name in names:
                for base in roots:
                    for cand in (name.replace(".", "/") + ".py", name.replace(".", "/") + "/__init__.py"):
                        if os.path.exists(os.path.join(base, cand)):
                            todo.append(os.path.relpath(os.path.join(base, cand), ROOT).replace("\\", "/"))
    return seen


def check_context():
    """Exits naming the first staged file that must not go into an image."""
    for d, _, files in os.walk(CONTEXT):
        for f in files:
            path = os.path.join(d, f)
            rel = os.path.relpath(path, CONTEXT).replace("\\", "/")
            hit = next((w for w in PRIVATE_PATH if w in rel.lower()), None)
            if hit:
                raise SystemExit("refused: %s names private scan data (%s)" % (rel, hit))
            if f.endswith((".json", ".py")):
                with open(path, encoding="utf-8", errors="replace") as fh:
                    m = LOCAL_PATH.search(fh.read())
                if m:
                    raise SystemExit("refused: %s carries a path of this machine (%r)" % (rel, m.group(0)))


def stage(a):
    """Wipes and rebuilds the build context, checks it, and prints what is in it."""
    models = os.path.join(common.paths.MODELS, "ct-segmenter-v%d" % a.version)
    shutil.rmtree(CONTEXT, ignore_errors=True)
    os.makedirs(CONTEXT)
    needed = import_closure()
    for rel in rs.repo_file_list():
        if rel.endswith(".py") and rel not in needed:
            continue
        copy(os.path.join(ROOT, rel), "repo/" + rel)
    copy(os.path.join(HERE, "handler.py"), "handler.py")
    dest = "Models/ct-segmenter-v%d/" % a.version
    folds = sorted(glob.glob(os.path.join(models, "fold*.pt")))
    if not folds or not os.path.exists(os.path.join(models, "meta.json")):
        raise SystemExit("no fold*.pt and meta.json in %s" % models)
    for p in folds + [os.path.join(models, "meta.json")]:
        copy(p, dest + os.path.basename(p))
    for s in a.samples.split(","):
        copy(os.path.join(a.sample_root, s.split(".")[0], "ct.nii.gz"), "samples/" + s)
    check_context()
    count = size = 0
    for d, _, files in os.walk(CONTEXT):
        count += len(files)
        size += sum(os.path.getsize(os.path.join(d, f)) for f in files)
    print("%d files, %.1f MB in %s" % (count, size / 1e6, CONTEXT))
    for p in folds:
        print("%s  %s" % (sha256(p), os.path.basename(p)))
    print("Dockerfile: %s" % DOCKERFILE)


def docker(args, **kw):
    return subprocess.run(["docker"] + args, **common.no_window(), **kw)


def build(a):
    r = docker(["build", "-f", DOCKERFILE, CONTEXT, "-t", a.tag])
    if r.returncode:
        raise SystemExit("docker build failed (exit %d)" % r.returncode)
    out = docker(["image", "inspect", a.tag, "--format", "{{.Id}} {{.Size}}"], capture_output=True, text=True)
    image_id, size = out.stdout.split()
    print("image %s, %.2f GB" % (image_id, int(size) / 1e9))


def walk_keys(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from walk_keys(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk_keys(v)


def find_output(text):
    """The handler's output from runpod's local test log: the last "Job result: " line holds the Python repr of
    {'output': {...}}, or of {'error': ...} when the job failed."""
    lines = [x for x in text.splitlines() if "Job result: " in x]
    if not lines:
        return None
    try:
        result = ast.literal_eval(lines[-1].split("Job result: ", 1)[1])
    except (ValueError, SyntaxError):
        return None
    if isinstance(result, dict) and result.get("error"):
        raise SystemExit("the handler returned an error: %s" % str(result["error"])[:500])
    return result.get("output") if isinstance(result, dict) else None


def test(a):
    """One reading inside the image, on the CPU, and the checks on what it returned."""
    inp = json.dumps({"input": {"sample": a.sample}})
    r = docker(["run", "--rm", "-e", "RAIDIOLOGY_DEVICE=cpu", a.tag, "python", "-u", "/opt/raid-serve/handler.py",
                "--test_input", inp], capture_output=True, text=True)
    out = find_output(r.stdout + "\n" + r.stderr)
    if out is None:
        raise SystemExit("no handler output found (docker exit %d); last lines:\n%s"
                         % (r.returncode, (r.stdout + r.stderr)[-1500:]))
    structs = out["structures"]
    assert isinstance(structs, list) and structs, "structures is empty"
    assert all(set(s) == {"label", "name"} for s in structs), "a structure is not {label, name}"
    bad = [s["name"] for s in structs if any(w in s["name"].lower() for w in PATHOLOGY)]
    assert not bad, "pathology names listed: %s" % bad[:5]
    keys = [k for k in walk_keys(out) if MEASURE.search(str(k))]
    assert not keys, "measurement-like keys: %s" % sorted(set(keys))
    assert out.get("not_a_medical_device") is True, "not_a_medical_device is not true"
    print("ok: %d structures, read in %s s" % (len(structs), (out.get("seconds") or {}).get("read")))


def login(a):
    with open(a.token_file, "rb") as f:
        token = f.read().strip()
    r = docker(["login", a.registry, "-u", a.user, "--password-stdin"], input=token, capture_output=True)
    text = (r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")).replace(
        token.decode("utf-8", "replace"), "***")
    lines = text.splitlines()
    print(next((x for x in lines if "Login" in x), lines[-1] if lines else ""))
    if r.returncode:
        raise SystemExit("docker login failed (exit %d)" % r.returncode)


def push(a):
    r = docker(["push", a.tag], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("docker push failed (exit %d): %s" % (r.returncode, r.stderr[-500:]))
    m = re.search(r"digest: (sha256:[0-9a-f]+)", r.stdout)
    print("pushed %s %s" % (a.tag, m.group(1) if m else "(digest not found in docker's output)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--version", type=int, default=2)
    s.add_argument("--samples", default="s0654.nii.gz,s0574.nii.gz,s0011.nii.gz")
    s.add_argument("--sample-root", default=os.path.join(common.paths.DATASETS, "public", "totalsegmentator"))
    b = sub.add_parser("build")
    b.add_argument("--tag", required=True)
    t = sub.add_parser("test")
    t.add_argument("--tag", required=True)
    t.add_argument("--sample", default="s0654.nii.gz")
    lg = sub.add_parser("login")
    lg.add_argument("--registry", default="ghcr.io")
    lg.add_argument("--user", required=True)
    lg.add_argument("--token-file", required=True)
    p = sub.add_parser("push")
    p.add_argument("--tag", required=True)
    a = ap.parse_args()
    {"stage": stage, "build": build, "test": test, "login": login, "push": push}[a.cmd](a)


if __name__ == "__main__":
    main()
