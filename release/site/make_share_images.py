"""Redraw site/img/og.png (1200x630) and site/img/screenshot-home.png (1280x800) with headless Playwright."""
import os, subprocess, sys, time, tempfile
from playwright.sync_api import sync_playwright
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = os.path.join(R, "site")
CARD = """<!doctype html><meta charset=utf-8><style>
@font-face{font-family:SS;src:url(fonts/%(serif)s);font-weight:200 900}
@font-face{font-family:IN;src:url(fonts/%(sans)s);font-weight:100 900}
body{margin:0;width:1200px;height:630px;background:#08131f;color:#fff;font-family:IN,sans-serif;position:relative;overflow:hidden}
.l{position:absolute;left:72px;top:64px;display:flex;align-items:center;gap:18px;font:600 34px IN}
.l img{width:60px;height:60px}
h1{position:absolute;left:72px;top:190px;margin:0;font:500 68px/1.08 SS,serif;width:540px}
p{position:absolute;left:72px;top:440px;margin:0;font:400 26px/1.4 IN;color:#c9d5e3;width:500px}
.f{position:absolute;left:72px;bottom:44px;font-size:20px;color:#9fb3c8}
.g{position:absolute;right:56px;top:56px;display:grid;grid-template-columns:repeat(2,230px);gap:16px}
.g img{width:230px;height:240px;object-fit:cover;border-radius:14px;display:block;background:#000}
.g img:nth-child(2){margin-top:60px}.g img:nth-child(3){margin-top:-44px}.g img:nth-child(4){margin-top:16px}
</style><div class=l><img src="img/logo.svg" alt="">rAidiology</div>
<h1>Your scans, on your own computer</h1><p>Open your own imaging on your own computer. Open research models, trained on public data.</p>
<div class=f>Not a medical device. For education and research.</div>
<div class=g><img src="img/scans/ct-organs.webp"><img src="img/scans/mri-spine.webp"><img src="img/scans/xray-chest.webp"><img src="img/scans/xray-wrist.webp"></div>"""
fonts = os.listdir(os.path.join(S, "fonts"))
serif = next(f for f in fonts if f.lower().startswith("sourceserif") and f.endswith(".woff2") and "italic" not in f.lower())
sans = next(f for f in fonts if f.lower().startswith("inter") and f.endswith(".woff2") and "italic" not in f.lower())
tmp = os.path.join(S, "_card.html")
open(tmp, "w", encoding="utf-8").write(CARD % {"serif": serif, "sans": sans})
srv = subprocess.Popen([sys.executable, "-m", "http.server", "8777", "--bind", "127.0.0.1", "-d", S],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.5)
try:
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        pg = b.new_page(viewport={"width": 1200, "height": 630}, service_workers="block")
        pg.goto("http://127.0.0.1:8777/_card.html", wait_until="networkidle")
        pg.evaluate("document.fonts.ready"); pg.wait_for_timeout(300)
        pg.screenshot(path=os.path.join(S, "img/og.png"))
        pg = b.new_page(viewport={"width": 1280, "height": 800}, service_workers="block")
        pg.goto("http://127.0.0.1:8777/index.html", wait_until="networkidle"); pg.wait_for_timeout(800)
        pg.screenshot(path=os.path.join(S, "img/screenshot-home.png"))
        b.close()
finally:
    srv.kill(); os.remove(tmp)
