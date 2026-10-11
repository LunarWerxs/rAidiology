"""Write the precache list and version into site/sw.js. `--check` exits 1 when sw.js is out of date."""
import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
PAGES = ["index.html", "open.html", "run.html", "donation.html", "health-data-privacy.html", "models.html", "support.html", "pricing.html", "thanks.html"]
SKIP = {"img/og.png", "img/screenshot-home.png"}
DIRS = ["css", "js", "fonts", "img", "data/models"]
FILES = ["manifest.webmanifest", "data/stats.json", "data/scan-pictures.json"]
NEVER = ("data/meshes/", "data/stacks/", "data/report", "report", "api/", "launcher/", "sample/")


def collect():
    out = [p for p in PAGES if (SITE / p).is_file()]
    out += [f for f in FILES if (SITE / f).is_file()]
    for d in DIRS:
        base = SITE / d
        if base.is_dir():
            out += [f.relative_to(SITE).as_posix() for f in base.rglob("*") if f.is_file()]
    return sorted({p for p in out if p not in SKIP and not p.startswith(NEVER)})


def render(sw_text):
    files = collect()
    h = hashlib.sha256()
    for p in files:
        h.update(p.encode() + b"\0" + (SITE / p).read_bytes() + b"\0")
    lines = ["// precache:start", f"const VERSION = '{h.hexdigest()[:12]}';", "const PRECACHE = ["]
    lines += [f"  './{p}'," for p in files]
    lines += ["];", "// precache:end"]
    return re.sub(r"// precache:start.*?// precache:end", lambda m: "\n".join(lines), sw_text, flags=re.S)


def main():
    path = SITE / "sw.js"
    old = path.read_text(encoding="utf-8")
    new = render(old)
    if "--check" in sys.argv:
        if new != old:
            print("sw.js is out of date: run release/site/build_sw.py")
            return 1
        return 0
    path.write_text(new, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
