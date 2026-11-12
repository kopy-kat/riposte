"""The watch-only pose beside the video's pose, through the video's camera, coloured by how far each bone is off.

    uv run python -m render.deviation data/processed/demo_video30.sam3d.npz data/processed/demo_video30.watchpose.npz \
        data/processed/deviation.mp4 [start_s end_s]
No video pixels: the video's pose is a white ghost on black. The watch skeleton gets the video's pelvis, since the
watch can't see where the fencer stands. Prints the mean per-joint error over the frames drawn.
"""

import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from model.pairs import JOINTS
from render.prepare import BONES

SS = 2  # supersampling for antialiased lines
GOOD, BAD = 0.04, 0.15  # metres: full green at or under GOOD, full red at or over BAD
STOPS = np.array([[110, 231, 183], [253, 224, 120], [255, 107, 107]], dtype=float)
BG = (11, 11, 13)
ZOOM, LIFT = 0.8, 90
FONT = "/System/Library/Fonts/SFNS.ttf"


def colour(err):
    x = np.clip((err - GOOD) / (BAD - GOOD), 0, 1) * 2
    i = min(int(x), 1)
    return tuple(int(v) for v in STOPS[i] + (STOPS[i + 1] - STOPS[i]) * (x - i))


def main():
    sam, watch, out = sys.argv[1:4]
    d, w = np.load(sam), np.load(watch)
    fps = float(d["fps"])
    K = d["cam_int"].reshape(-1, 3, 3)[0]
    n = len(d["found"])
    start, end = (int(float(s) * fps) for s in sys.argv[4:6]) if len(sys.argv) > 5 else (0, n)
    end = min(end, n)

    true = d["pred_keypoints_3d"][:, JOINTS] + d["pred_cam_t"][:, None]
    tw = np.arange(n) / fps + float(w["offset"])
    flat = w["pose"].reshape(len(w["t"]), -1)
    flat = np.stack([np.interp(tw, w["t"], flat[:, k]) for k in range(flat.shape[1])], 1)
    pred = flat.reshape(n, -1, 3) @ w["R"] + (true[:, 9:10] + true[:, 10:11]) / 2
    found = d["found"].astype(bool) & (tw >= w["t"][0]) & (tw <= w["t"][-1])
    err = np.linalg.norm(pred - true, axis=-1)
    body = np.where(found, err.mean(1), np.nan)

    def project(p):
        # Zoomed out from the video's framing and lifted, so the feet stay in frame in a lunge.
        uv = (p[:, :2] / p[:, 2:]) * [K[0, 0], K[1, 1]]
        return uv * ZOOM + [K[0, 2], K[1, 2] - LIFT]

    W, H = 1920, 1080
    font = ImageFont.truetype(FONT, 30)
    small = ImageFont.truetype(FONT, 18)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out],
                           stdin=subprocess.PIPE)
    for k in range(start, end):
        img = Image.new("RGBA", (W, H), BG + (255,))
        if found[k]:
            layer = Image.new("RGBA", (W * SS, H * SS))
            g = ImageDraw.Draw(layer)
            ghost, watch2d = project(true[k]) * SS, project(pred[k]) * SS
            for a, b in BONES:
                g.line([tuple(ghost[a]), tuple(ghost[b])], fill=(255, 255, 255, 120), width=3 * SS)
            for a, b in BONES:
                c = colour((err[k, a] + err[k, b]) / 2)
                g.line([tuple(watch2d[a]), tuple(watch2d[b])], fill=c + (255,), width=5 * SS)
            r = 2.5 * SS
            for j in {j for bone in BONES for j in bone}:
                x, y = watch2d[j]
                g.ellipse([x - r, y - r, x + r, y + r], fill=colour(err[k, j]) + (255,))
            img.alpha_composite(layer.resize((W, H), Image.LANCZOS))
        recent = body[max(start, k - int(fps)):k + 1]
        if np.isfinite(recent).any():
            t = ImageDraw.Draw(img)
            mm = np.nanmean(recent) * 1000
            t.text((64, H - 64), f"{mm:.0f} mm", font=font, fill=colour(mm / 1000) + (235,), anchor="ls")
            t.text((64, H - 108), "watch vs video pose", font=small, fill=(255, 255, 255, 140), anchor="ls")
        t = ImageDraw.Draw(img)
        for i, (name, rgb) in enumerate([("video pose", (255, 255, 255)), ("Apple Watch", tuple(int(v) for v in STOPS[0]))]):
            y = H - 64 - 36 * (1 - i)
            t.line([(W - 260, y - 7), (W - 228, y - 7)], fill=rgb + (200,), width=3)
            t.text((W - 212, y), name, font=small, fill=(255, 255, 255, 170), anchor="ls")
        enc.stdin.write(np.asarray(img.convert("RGB")).tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"{out}  mean per-joint error {np.nanmean(body[start:end]) * 1000:.0f} mm over {np.isfinite(body[start:end]).sum()} frames")


if __name__ == "__main__":
    main()
