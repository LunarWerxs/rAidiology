"""Draw the app icons from site/img/logo.svg with headless Chromium (playwright)."""
import pathlib
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMG = ROOT / "site" / "img"
SVG = (IMG / "logo.svg").read_text(encoding="utf-8")
BG = "#15304d"


def page(size, scale, full_bleed):
    # full_bleed: the square is filled (maskable); the logo sits inside the central safe zone.
    px = int(size * scale)
    off = (size - px) // 2
    bg = BG if full_bleed else "transparent"
    svg = SVG.replace("<svg ", "<svg width='100%' height='100%' ", 1)
    return (f"<body style='margin:0;background:{bg};width:{size}px;height:{size}px'>"
            f"<div style='position:absolute;left:{off}px;top:{off}px;width:{px}px;height:{px}px'>"
            f"{svg}</div></body>")


def main():
    jobs = [("icon-192.png", 192, 1.0, False), ("icon-512.png", 512, 1.0, False),
            ("icon-maskable-512.png", 512, 0.7, True)]
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        for name, size, scale, bleed in jobs:
            pg = b.new_page(viewport={"width": size, "height": size})
            pg.set_content(page(size, scale, bleed))
            pg.screenshot(path=str(IMG / name), omit_background=not bleed)
            pg.close()
        b.close()


if __name__ == "__main__":
    main()
