"""The demo video: title, the watch-only fencer, the same seconds over the video, the numbers.

    uv run python -m render.cut data/processed/cut_watch_only/frames data/processed/cut_deviation.mp4 riposte.mp4
frames: render/scene.py output for the watch-only take (RGBA PNGs, 1920x1080); deviation: render/deviation.py
output covering the same seconds.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render.deviation import FONT, STOPS

W, H, FPS = 1920, 1080, 30
BG = (11, 11, 13)
MINT = STOPS[0].astype(np.uint8)
FADE = 1.0


def font(size, weight="Light"):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_name(weight)
    return f


def encode(path, frames):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(path)],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.asarray(f.convert("RGB")).tobytes())
    p.stdin.close()
    p.wait()


def card(lines, seconds):
    """lines: (text, size, weight, rgb, dy from centre). Fades in and out over FADE."""
    img = Image.new("RGB", (W, H), BG)
    g = ImageDraw.Draw(img)
    for text, size, weight, rgb, dy in lines:
        g.text((W / 2, H / 2 + dy), text, font=font(size, weight), fill=rgb, anchor="mm")
    base, black = np.asarray(img, dtype=np.float32), np.full((H, W, 3), BG, np.float32)
    n = int(seconds * FPS)
    for i in range(n):
        a = min(1.0, i / (FADE * FPS), (n - 1 - i) / (FADE * FPS))
        yield Image.fromarray((black + (base - black) * a).astype(np.uint8))


def watch_only(frames):
    label = font(26, "Regular")
    for path in frames:
        px = np.asarray(Image.open(path).convert("RGBA")).copy()
        r, g, b = (px[..., i].astype(int) for i in range(3))
        px[(g > r + 30) & (g > b + 20), :3] = MINT  # the skeleton, in the overlay's colour
        px[(r > b + 40) & (r > g + 15), 3] = 0  # the tip trail: it tangles when the watch blade swings
        img = Image.new("RGBA", (W, H), BG + (255,))
        img.alpha_composite(Image.fromarray(px))
        ImageDraw.Draw(img).text((64, H - 64), "Apple Watch only", font=label, fill=(255, 255, 255, 150), anchor="ls")
        yield img


def main():
    frames, deviation, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    tmp = Path(tempfile.mkdtemp())
    grey, mint = (150, 150, 155), tuple(int(v) for v in MINT)
    encode(tmp / "title.mp4", card([("riposte", 120, "Light", (240, 240, 240), -20),
                                    ("one wrist, the whole fencer", 32, "Regular", grey, 80)], 3.0))
    encode(tmp / "watch.mp4", watch_only(sorted(frames.glob("f_*.png"))))
    encode(tmp / "end.mp4", card([("64 mm", 110, "Light", mint, -60),
                                  ("mean joint error on a take the model never saw", 30, "Regular", grey, 40)], 4.0))

    parts = [tmp / "title.mp4", tmp / "watch.mp4", deviation, tmp / "end.mp4"]
    lengths = [float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                               "csv=p=0", str(p)])) for p in parts]
    chain, prev, t = [], "[v0]", 0.0
    for i in range(1, len(parts)):
        t += lengths[i - 1] - FADE
        chain.append(f"{prev}[v{i}]xfade=transition=fade:duration={FADE}:offset={t:.3f}[x{i}]")
        prev = f"[x{i}]"
    norm = ";".join(f"[{i}:v]fps={FPS},scale={W}:{H},setsar=1,format=yuv420p[v{i}]" for i in range(len(parts)))
    subprocess.run(["ffmpeg", "-v", "error", "-y", *[a for p in parts for a in ("-i", str(p))], "-filter_complex",
                    norm + ";" + ";".join(chain), "-map", prev, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", out], check=True)
    print(out)


if __name__ == "__main__":
    main()
