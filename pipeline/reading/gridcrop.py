"""gridcrop.py OUT.png x0 y0 x1 y1 scale SERIES:SLICE[:fluid|:marrow] [...]
Like crop.py, but overlays a millimetre grid (thin line every 1 mm, bright every 5 mm) and
native-pixel coordinates on the edges, so distances can be read straight off the image.
Pixel spacing comes from png/series_meta.json. Slices come from $READER_PNG (render.py's
<reader_dir>/png), else a png/ folder beside this file."""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
PNG = os.environ.get("READER_PNG", os.path.join(ROOT, "png"))
meta = json.load(open(os.path.join(PNG, "series_meta.json")))
out, x0, y0, x1, y1, scale, *items = sys.argv[1:]
x0, y0, x1, y1, scale = int(x0), int(y0), int(x1), int(y1), float(scale)
font = _font(12)
tiles = []

def _font(size):
    """Arial where it is installed (Windows), else Pillow's own font."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()

for it in items:
    s, i, *win = it.split(":")
    row_mm, col_mm = meta[s]["pixel_spacing"]
    im = Image.open(os.path.join(PNG, s, f"{int(i):02d}{'_' + win[0] if win else ''}.png")).crop((x0, y0, x1, y1))
    im = im.resize((int((x1 - x0) * scale), int((y1 - y0) * scale)), Image.BICUBIC).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    mm = 0
    while mm * 1.0 / col_mm <= (x1 - x0):
        X = mm / col_mm * scale
        d.line([(X, 0), (X, im.height)], fill=(0, 255, 255, 110 if mm % 5 == 0 else 35), width=1)
        if mm % 5 == 0:
            d.text((X + 2, im.height - 14), f"{x0 + mm / col_mm:.0f}", fill=(0, 255, 255), font=font)
        mm += 1
    mm = 0
    while mm * 1.0 / row_mm <= (y1 - y0):
        Y = mm / row_mm * scale
        d.line([(0, Y), (im.width, Y)], fill=(0, 255, 255, 110 if mm % 5 == 0 else 35), width=1)
        if mm % 5 == 0:
            d.text((2, Y + 1), f"{y0 + mm / row_mm:.0f}", fill=(0, 255, 255), font=font)
        mm += 1
    d.text((4, 2), f"{it}  grid 1 mm", fill=(255, 255, 0), font=font)
    tiles.append(im)
W = sum(t.width for t in tiles) + 4 * (len(tiles) - 1)
canvas = Image.new("RGB", (W, max(t.height for t in tiles)), (40, 40, 40))
x = 0
for t in tiles:
    canvas.paste(t, (x, 0))
    x += t.width + 4
dest = out if os.path.isabs(out) else os.path.join(os.path.dirname(PNG), "crops", out)
canvas.save(dest)
print(dest, canvas.size)
