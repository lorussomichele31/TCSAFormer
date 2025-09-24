# pip install tensorboard pillow matplotlib numpy
import argparse, math, re
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.cm as cm

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def wrap_text(text, max_len=28, max_lines=2):
    parts, line = [], ""
    for token in re.split(r"([/_\- ])", text):  # keep separators
        if len(line) + len(token) > max_len and line:
            parts.append(line)
            line = token.strip()
            if len(parts) == max_lines:
                break
        else:
            line += token
    if line and len(parts) < max_lines:
        parts.append(line)
    if len(parts) == max_lines and (len(text) > sum(len(p) for p in parts)):
        parts[-1] = parts[-1][: max(0, len(parts[-1]) - 3)] + "..."
    return parts

def to_color(img: Image.Image, apply_colormap: bool):
    if not apply_colormap:
        return img.convert("RGB")
    arr = np.array(img.convert("L"), dtype=np.float32)
    if arr.max() == arr.min():
        return img.convert("RGB")
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    rgb = (cm.viridis(arr)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")

def load_images_last_step(event_path: str, include_substring: str | None):
    ea = event_accumulator.EventAccumulator(event_path, size_guidance={"images": 0})
    ea.Reload()
    tags = ea.Tags().get("images", [])
    if include_substring:
        tags = [t for t in tags if include_substring in t]
    tags = sorted(tags, key=natural_key)
    if not tags:
        raise SystemExit("No image tags found (check --include).")
    last_step = None
    for t in tags:
        imgs = ea.Images(t)
        if imgs:
            smax = max(e.step for e in imgs)
            last_step = smax if last_step is None else max(last_step, smax)
    if last_step is None:
        raise SystemExit("No images in the event file.")
    pairs = []
    for t in tags:
        imgs = [e for e in ea.Images(t) if e.step == last_step]
        if imgs:
            im = Image.open(BytesIO(imgs[0].encoded_image_string))
            pairs.append((t, im))
    if not pairs:
        raise SystemExit(f"No images at last step {last_step}.")
    return last_step, pairs

def load_font(font_path: str | None, fontsize: int):
    if font_path and Path(font_path).exists():
        return ImageFont.truetype(font_path, fontsize)
    # try some common locations
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # Linux
        "/Library/Fonts/Arial.ttf",                          # macOS
        "C:/Windows/Fonts/arial.ttf",                        # Windows
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, fontsize)
    # fallback (tiny bitmap)
    f = ImageFont.load_default()
    print("[warn] Using PIL default bitmap font (small). Pass --font /path/to/DejaVuSans.ttf for crisp labels.")
    return f

def make_grid(pairs, cols: int, tile: int, pad: int,
              base_fontsize: int | None, font_path: str | None,
              colorize: bool, bg=(250, 250, 250)):
    # auto font size ~6% of tile if not provided
    fontsize = base_fontsize or max(10, int(tile * 0.06))
    font = load_font(font_path, fontsize)

    # preprocess & pad images to tile×tile
    processed = []
    for tag, im in pairs:
        im = to_color(im, colorize)
        w, h = im.size
        scale = tile / max(w, h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile, tile), (255, 255, 255))
        ox, oy = (tile - nw) // 2, (tile - nh) // 2
        canvas.paste(im, (ox, oy))
        canvas = ImageOps.expand(canvas, border=1, fill=(220, 220, 220))
        processed.append((tag, canvas))

    # compute caption height dynamically (two lines)
    draw_tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    line_h = draw_tmp.textbbox((0, 0), "Hg", font=font)[3] + 2
    caption_h = line_h * 2 + 10  # two lines + padding

    TW = tile + 2  # +borders
    TH = tile + 2 + caption_h

    n = len(processed)
    cols = max(1, cols)
    rows = math.ceil(n / cols)
    W = cols * TW + (cols + 1) * pad
    H = rows * TH + (rows + 1) * pad

    grid = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(grid)

    for i, (tag, tile_img) in enumerate(processed):
        r, c = divmod(i, cols)
        x0 = pad + c * (TW + pad)
        y0 = pad + r * (TH + pad)
        grid.paste(tile_img, (x0, y0))
        # caption bar
        bar_y0 = y0 + tile_img.size[1]
        draw.rectangle([x0, bar_y0, x0 + TW, bar_y0 + caption_h], fill=(255, 255, 255))
        # pretty tag = last 2 components
        parts = re.split(r"[\\/]", tag)
        pretty = "/".join(parts[-2:]) if len(parts) >= 2 else tag
        lines = wrap_text(pretty, max_len=int(tile * 0.09), max_lines=2)
        for li, txt in enumerate(lines):
            bbox = draw.textbbox((0, 0), txt, font=font)
            tx = x0 + (TW - (bbox[2] - bbox[0])) // 2
            ty = bar_y0 + 5 + li * line_h
            draw.text((tx, ty), txt, fill=(40, 40, 40), font=font)
    return grid

def auto_cols(n):
    if n <= 4: return n
    if n <= 8: return 4
    if n <= 12: return 4
    if n <= 20: return 5
    if n <= 30: return 6
    return 7

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True, help="path to events.out.tfevents.*")
    ap.add_argument("--out", default="docs/assets/metrics/attn_grid.png")
    ap.add_argument("--include", default="", help="substring filter for tags")
    ap.add_argument("--cols", type=int, default=0, help="columns (0=auto)")
    ap.add_argument("--tile", type=int, default=384, help="tile image size (px)")
    ap.add_argument("--pad", type=int, default=18, help="outer padding between tiles (px)")
    ap.add_argument("--font", type=str, default=None, help="path to a TTF (e.g., DejaVuSans.ttf)")
    ap.add_argument("--fontsize", type=int, default=0, help="override auto font size")
    ap.add_argument("--colorize", action="store_true", help="apply viridis colormap to grayscale")
    ap.add_argument("--max-width", type=int, default=0, help="optional: downscale final image to this width")
    args = ap.parse_args()

    last_step, pairs = load_images_last_step(args.event, args.include or None)
    n = len(pairs)
    cols = args.cols or auto_cols(n)
    print(f"[info] last step {last_step}, images {n}, cols {cols}")

    grid = make_grid(
        pairs=pairs,
        cols=cols,
        tile=args.tile,
        pad=args.pad,
        base_fontsize=(args.fontsize or None),
        font_path=args.font,
        colorize=args.colorize
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.max_width and grid.size[0] > args.max_width:
        scale = args.max_width / grid.size[0]
        grid = grid.resize((int(grid.size[0]*scale), int(grid.size[1]*scale)), Image.Resampling.LANCZOS)
        print(f"[info] downscaled to width {args.max_width}px")

    grid.save(out, format="PNG", optimize=True)
    print(f"[ok] saved → {out.resolve()}  size={grid.size}")

if __name__ == "__main__":
    main()
