import os, sys, subprocess, time
from playwright.sync_api import sync_playwright
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = os.path.join(R, "work/release-thread/run-2026-10-10c/shots"); os.makedirs(out, exist_ok=True)
srv = subprocess.Popen([sys.executable, "-m", "http.server", "8778", "--bind", "127.0.0.1", "-d", os.path.join(R, "site")], stdout=-3, stderr=-3)
time.sleep(1.5)
try:
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        for w in (1440, 390):
            pg = b.new_page(viewport={"width": w, "height": 900}, service_workers="block")
            for n in ("models", "run"):
                pg.goto(f"http://127.0.0.1:8778/{n}.html", wait_until="networkidle"); pg.wait_for_timeout(400)
                pg.screenshot(path=f"{out}/{n}-{w}.png", full_page=True)
        b.close()
finally:
    srv.kill()
