"""crop.py OUT.png x0 y0 x1 y1 scale SERIES:SLICE[:fluid|:marrow] [...]
Crops the same box from each listed slice PNG, upsamples, and lays them side by side.
Slices come from $READER_PNG (render.py's <reader_dir>/png), else a png/ folder beside this file."""
import sys, os
from PIL import Image, ImageDraw, ImageFont

PNG = os.environ.get("READER_PNG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "png"))
out, x0, y0, x1, y1, scale, *items = sys.argv[1:]
x0, y0, x1, y1, scale = int(x0), int(y0), int(x1), int(y1), float(scale)
tiles = []

def _font(size):
    """Arial where it is installed (Windows), else Pillow's own font."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()

for it in items:
    s, i, *win = it.split(":")
    im = Image.open(os.path.join(PNG, s, f"{int(i):02d}{'_' + win[0] if win else ''}.png")).crop((x0, y0, x1, y1))
    im = im.resize((int((x1 - x0) * scale), int((y1 - y0) * scale)), Image.BICUBIC).convert("RGB")
    ImageDraw.Draw(im).text((4, 2), it, fill=(255, 255, 0), font=_font(14))
    tiles.append(im)
W = sum(t.width for t in tiles) + 4 * (len(tiles) - 1)
canvas = Image.new("RGB", (W, max(t.height for t in tiles)), (40, 40, 40))
x = 0
for t in tiles:
    canvas.paste(t, (x, 0))
    x += t.width + 4
canvas.save(os.path.join(os.path.dirname(PNG), "crops", out) if not os.path.isabs(out) else out)
print(canvas.size)
