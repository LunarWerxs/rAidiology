"""Render the home page hero's posters from the live 3D (site/js/hero3d.js).

Serves site/ on a free local port (8840-8849), opens a throwaway page (site/_hero-test.html, deleted
afterwards) in headless Chromium on SwiftShader, mounts the hero with its turn held, waits for the
skeleton and the first frame, and screenshots the stage without its controls:
  site/img/hero-poster.webp        desktop stage 640 x 550 at 2x (1280 x 1100), at most 70 KB
  site/img/hero-poster-phone.webp  phone stage 390 x 440 at 2x (780 x 880), at most 45 KB
The posters match the canvas's first frame, so the page can swap one for the other unseen.

Usage: python tools/render_hero_poster.py [--shots DIR]
--shots also saves review PNGs to DIR: the stage with its controls, a hovered bone, each mode, and the
scan plane moved by a slice event.
"""
import argparse
import io
import os
import pathlib
import socket
import subprocess
import sys
import time

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PAGE = SITE / "_hero-test.html"
STAGES = [  # name, css width, css height, poster path, byte limit
    ("desktop", 640, 550, SITE / "img/hero-poster.webp", 70_000),
    ("phone", 390, 440, SITE / "img/hero-poster-phone.webp", 45_000),
]
CHROME_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>hero poster</title>
<link rel="stylesheet" href="css/brand.css">
<link rel="stylesheet" href="css/hero3d.css">
<style>
  html, body { margin: 0; background: #05090e; }
  #stage { border-radius: 0; aspect-ratio: auto; }
  .hero3d .hero3d__canvas { transition: none; }
  .hero3d.hero3d--poster .hero3d__ui { display: none !important; }
</style>
<script type="importmap">
{"imports":{"three":"./js/vendor/three/three.module.min.js","three/addons/":"./js/vendor/three/addons/"}}
</script>
</head><body>
<div id="stage"></div>
<script type="module">
import { mountHero3D } from './js/hero3d.js';
const q = new URLSearchParams(location.search);
const stage = document.getElementById('stage');
stage.style.width = q.get('w') + 'px';
stage.style.height = q.get('h') + 'px';
const api = mountHero3D(stage, { autoTurn: false });
window.__hero = api;
api.ready.then((ok) => { window.__heroState = ok ? 'live' : 'static'; });
</script>
</body></html>
"""


def free_port():
    for port in range(8840, 8850):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit("no free port in 8840-8849")


def wait_for_server(port, seconds=15):
    end = time.time() + seconds
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise SystemExit(f"the local server on {port} did not start")


def webp(png, path, limit):
    img = Image.open(io.BytesIO(png)).convert("RGB")
    for q in range(88, 30, -4):
        buf = io.BytesIO()
        img.save(buf, "WEBP", quality=q, method=6)
        if buf.tell() <= limit:
            path.write_bytes(buf.getvalue())
            return q, buf.tell()
    raise SystemExit(f"{path.name}: cannot get under {limit} bytes")


def settle(page, ms=250):
    page.evaluate("() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))")
    page.wait_for_timeout(ms)


def review(page, stage, name, w, h, shots):
    def shot(tag):
        settle(page, 300)
        stage.screenshot(path=str(shots / f"{name}-{tag}.png"))

    shot("ui")
    # Hover until a bone answers (a mouse over the thigh, the ribs, the skull, ...).
    for fx, fy in [(0.53, 0.66), (0.47, 0.66), (0.52, 0.42), (0.5, 0.36), (0.5, 0.2), (0.5, 0.5)]:
        page.mouse.move(w * fx, h * fy)
        settle(page, 200)
        if page.evaluate("() => !document.querySelector('.hero3d__label').hidden"):
            break
    shot("hover")
    page.mouse.move(2, 2)
    settle(page, 200)
    for mode in ("skeleton", "xray"):
        page.click(f".hero3d__mode[data-mode='{mode}']")
        page.wait_for_timeout(1200)
        shot(mode)
    page.click(".hero3d__mode[data-mode='glass']")
    page.wait_for_timeout(1200)
    page.evaluate("() => window.dispatchEvent(new CustomEvent('raidiology:slice', { detail: { index: 40, count: 46, level: 'T9' } }))")
    page.wait_for_timeout(1500)
    shot("slice")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shots", help="also save review screenshots to this folder")
    args = ap.parse_args()
    shots = pathlib.Path(args.shots) if args.shots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)

    PAGE.write_text(HTML, encoding="utf-8")
    port = free_port()
    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(SITE)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    try:
        wait_for_server(port)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=CHROME_ARGS)
            try:
                for name, w, h, poster, limit in STAGES:
                    ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=2,
                                              reduced_motion="no-preference")
                    page = ctx.new_page()
                    errors = []
                    page.on("pageerror", lambda e: errors.append(str(e)))
                    page.goto(f"http://127.0.0.1:{port}/{PAGE.name}?w={w}&h={h}")
                    page.wait_for_function("() => window.__heroState", timeout=180_000)
                    state = page.evaluate("() => window.__heroState")
                    if state != "live" or errors:
                        raise SystemExit(f"{name}: the hero did not go live ({state}): {errors[:3]}")
                    stage = page.locator("#stage")
                    page.evaluate("() => document.getElementById('stage').classList.add('hero3d--poster')")
                    settle(page, 400)
                    png = stage.screenshot()
                    if shots:
                        (shots / f"{name}-poster.png").write_bytes(png)
                    q, size = webp(png, poster, limit)
                    print(f"{poster.relative_to(ROOT).as_posix()}: {size} bytes (quality {q})")
                    if shots:
                        page.evaluate("() => document.getElementById('stage').classList.remove('hero3d--poster')")
                        review(page, stage, name, w, h, shots)
                    if errors:
                        raise SystemExit(f"{name}: page errors: {errors[:3]}")
                    ctx.close()
            finally:
                browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
        PAGE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
