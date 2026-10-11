"""Redraw site/img/og.png (1200x630) and site/img/screenshot-home.png (1280x800) with headless Playwright."""
import os, subprocess, sys, time, tempfile
from playwright.sync_api import sync_playwright
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = os.path.join(R, "site")
# Site v3's light look: Inter only, the brand's own ink, blue and teal, the scan collage on a dark panel as on the home page.
CARD = """<!doctype html><meta charset=utf-8><style>
@font-face{font-family:IN;src:url(fonts/%(sans)s);font-weight:100 900}
body{margin:0;width:1200px;height:630px;background:#f4f7fa;color:#0f1b2a;font-family:IN,sans-serif;position:relative;overflow:hidden}
.l{position:absolute;left:72px;top:64px;display:flex;align-items:center;gap:16px;font:700 32px IN;letter-spacing:-0.01em}
.l img{width:52px;height:52px}
.c{position:absolute;left:72px;top:170px;width:560px}
.k{margin:0 0 18px;font:700 17px IN;letter-spacing:0.08em;color:#1660c4}
h1{margin:0 0 22px;font:700 56px/1.08 IN;letter-spacing:-0.03em}
p{margin:0;font:400 24px/1.45 IN;color:#2a3a4d}
.f{position:absolute;left:72px;bottom:52px;font:600 19px IN;color:#5b6b7d}
.f b{display:inline-block;width:10px;height:10px;border-radius:50%%;background:#0c8a86;margin-right:10px;vertical-align:1px}
.g{position:absolute;right:56px;top:56px;width:468px;height:518px;box-sizing:border-box;padding:16px;background:#071019;border-radius:22px;
   display:grid;grid-template-columns:repeat(2,1fr);gap:12px;box-shadow:0 18px 40px rgba(15,27,42,.18)}
.g img{width:100%%;height:237px;object-fit:cover;border-radius:12px;display:block;background:#000}
</style><div class=l><img src="img/logo.svg" alt="">rAidiology</div>
<div class=c><div class=k>OPEN SOURCE, FOR EDUCATION AND RESEARCH</div>
<h1>Open your own scans in your browser.</h1><p>Seven open research models for X-ray, CT, MRI and mammogram pictures. Nothing is uploaded.</p></div>
<div class=f><b></b>Not a medical device. For education and research.</div>
<div class=g><img src="img/scans/ct-organs.webp"><img src="img/scans/mri-spine.webp"><img src="img/scans/xray-chest.webp"><img src="img/scans/xray-wrist.webp"></div>"""
fonts = os.listdir(os.path.join(S, "fonts"))
sans = next(f for f in fonts if f.lower().startswith("inter") and f.endswith(".woff2") and "italic" not in f.lower())
tmp = os.path.join(S, "_card.html")
open(tmp, "w", encoding="utf-8").write(CARD % {"sans": sans})
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
