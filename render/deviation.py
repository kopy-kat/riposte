"""The watch-only pose and blade through the video's camera: against the video's own pose, or over the video.

    uv run python -m render.deviation data/processed/demo_video30.sam3d.npz data/processed/demo_video30.watchpose.npz \
        data/raw/demo/watch data/processed/cut_deviation.mp4 71.5 89.5
    uv run python -m render.deviation data/processed/demo_video30.sam3d.npz data/processed/demo_video30.watchpose.npz \
        data/raw/demo/watch data/processed/cut_over.mp4 30 48 --over data/raw/demo/video30.mp4
Without --over there are no video pixels: the video's pose and the blade found in the video (<take>.blade.npz
from imu.blade_video) are a white ghost on black, and the watch's bones and blade are coloured by how far off
they are. With --over the watch's skeleton and blade are drawn on the video itself. Either way the watch
skeleton gets the video's pelvis, since the watch can't see where the fencer stands. Prints the mean per-joint
error over the frames drawn.
"""

import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imu.blade_video import watch_blade
from imu.calib import KNUCKLE_KP
from imu.synth import ELBOW_KP, WRIST_KP
from model.pairs import JOINTS
from render.prepare import BLADE_M, BLADE_MODEL, BONES

SS = 2  # supersampling for antialiased lines
GOOD, BAD = 0.04, 0.15  # metres: full green at or under GOOD, full red at or over BAD
BLADE_GOOD, BLADE_BAD = 4.0, 20.0  # the same for the blade's image angle, degrees
STOPS = np.array([[110, 231, 183], [253, 224, 120], [255, 107, 107]], dtype=float)
BG = (11, 11, 13)
ZOOM, LIFT = 0.8, 90
FADE = 1.0  # render.cut crossfades this clip in and out over a second; keep the text out of the fades
HAND_M = 0.09  # wrist to the knuckles, where the blade leaves the hand
FONT = "/System/Library/Fonts/SFNS.ttf"


def colour(err, good=GOOD, bad=BAD):
    x = np.clip((err - good) / (bad - good), 0, 1) * 2
    i = min(int(x), 1)
    return tuple(int(v) for v in STOPS[i] + (STOPS[i + 1] - STOPS[i]) * (x - i))


def video_frames(video, start, n, w, h):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-vf", f"select=gte(n\\,{start}),scale={w}:{h}",
                          "-fps_mode", "passthrough", "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    for _ in range(n):
        yield Image.frombytes("RGB", (w, h), p.stdout.read(w * h * 3)).convert("RGBA")


def main():
    args = sys.argv[1:]
    over = args.pop(args.index("--over") + 1) if "--over" in args else None
    args = [a for a in args if a != "--over"]
    sam, watch, watch_dir, out = args[:4]
    d, w = np.load(sam), np.load(watch)
    fps = float(d["fps"])
    K = d["cam_int"].reshape(-1, 3, 3)[0]
    n = len(d["found"])
    start, end = (int(float(s) * fps) for s in args[4:6]) if len(args) > 5 else (0, n)
    end = min(end, n)

    true = d["pred_keypoints_3d"][:, JOINTS] + d["pred_cam_t"][:, None]
    tw = np.arange(n) / fps + float(w["offset"])
    at_video = lambda x: np.stack([np.interp(tw, w["t"], x[:, k]) for k in range(x.shape[1])], 1)
    pelvis = (true[:, 9:10] + true[:, 10:11]) / 2
    pred = at_video(w["pose"].reshape(len(w["t"]), -1)).reshape(n, -1, 3) @ w["R"] + pelvis
    found = d["found"].astype(bool) & (tw >= w["t"][0]) & (tw <= w["t"][-1])
    err = np.linalg.norm(pred - true, axis=-1)
    body = np.where(found, err.mean(1), np.nan)

    # The watch's blade leaves the predicted hand along the direction model.predict scores.
    axis = at_video(watch_blade(np.load(BLADE_MODEL), watch_dir, w["t"], np.nan_to_num(w["pose"]))) @ w["R"]
    wr, el = pred[:, JOINTS.index(WRIST_KP["right"])], pred[:, JOINTS.index(ELBOW_KP["right"])]
    guard = wr + HAND_M * (wr - el) / np.linalg.norm(wr - el, axis=1, keepdims=True)
    tip = guard + BLADE_M * axis / np.linalg.norm(axis, axis=1, keepdims=True)
    # The video's blade is an image angle from its own grip; drawn at the length a side-on blade would have.
    seen = np.load(sam.replace(".sam3d.npz", ".blade.npz"))
    grip = d["pred_keypoints_3d"][:, KNUCKLE_KP["right"]] + d["pred_cam_t"]
    seen_ok = found & np.isfinite(seen["obs"]) & (seen["conf"] > np.percentile(seen["conf"][seen["conf"] > 0], 50))

    zoom, lift = (1.0, 0) if over else (ZOOM, LIFT)

    def project(p):
        # Against the video's pose: zoomed out from the video's framing and lifted, so the feet stay in frame in a lunge.
        uv = (p[..., :2] / p[..., 2:]) * [K[0, 0], K[1, 1]]
        return uv * zoom + [K[0, 2], K[1, 2] - lift]

    W, H = 1920, 1080
    font = ImageFont.truetype(FONT, 44)
    small = ImageFont.truetype(FONT, 30)
    mint = tuple(int(v) for v in STOPS[0])
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out],
                           stdin=subprocess.PIPE)
    back = video_frames(over, start, end - start, W, H) if over else None
    bone_w, blade_w = (3, 3) if over else (5, 4)
    for k in range(start, end):
        img = next(back) if over else Image.new("RGBA", (W, H), BG + (255,))
        if found[k]:
            layer = Image.new("RGBA", (W * SS, H * SS))
            g = ImageDraw.Draw(layer)
            ghost, watch2d = project(true[k]) * SS, project(pred[k]) * SS
            a, b = (tuple(project(p[k]) * SS) for p in (guard, tip))
            blade_rgb = (255, 255, 255) if over else mint
            if not over:
                for i, j in BONES:
                    g.line([tuple(ghost[i]), tuple(ghost[j])], fill=(255, 255, 255, 120), width=3 * SS)
                if seen_ok[k]:
                    g0 = project(grip[k]) * SS
                    length = BLADE_M * K[0, 0] / grip[k, 2] * zoom * SS
                    g1 = g0 + length * np.array([np.cos(seen["obs"][k]), np.sin(seen["obs"][k])])
                    g.line([tuple(g0), tuple(g1)], fill=(255, 255, 255, 120), width=2 * SS)
                    off = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(b[1] - a[1], b[0] - a[0]) - seen["obs"][k])))))
                    blade_rgb = colour(off, BLADE_GOOD, BLADE_BAD)
            for i, j in BONES:
                c = mint if over else colour((err[k, i] + err[k, j]) / 2)
                g.line([tuple(watch2d[i]), tuple(watch2d[j])], fill=c + (255,), width=bone_w * SS)
            r = bone_w / 2 * SS
            for j in {j for bone in BONES for j in bone}:
                x, y = watch2d[j]
                g.ellipse([x - r, y - r, x + r, y + r], fill=(mint if over else colour(err[k, j])) + (255,))
            g.line([a, b], fill=blade_rgb + (255,), width=blade_w * SS)
            img.alpha_composite(layer.resize((W, H), Image.LANCZOS))
        a = np.clip(min((k - start) / fps - FADE, (end - 1 - k) / fps - FADE) / 0.3, 0, 1)
        txt = Image.new("RGBA", (W, H))
        t = ImageDraw.Draw(txt)
        if over:
            note = "skeleton and blade: Apple Watch only  ·  position in the room from the video"
            box = t.textbbox((64, H - 64), note, font=small, anchor="ls")
            t.rectangle([box[0] - 16, box[1] - 10, box[2] + 16, box[3] + 12], fill=BG + (int(170 * a),))
            t.text((64, H - 64), note, font=small, fill=(255, 255, 255, int(230 * a)), anchor="ls")
        else:
            recent = body[max(start, k - int(fps)):k + 1]
            if np.isfinite(recent).any():
                mm = np.nanmean(recent) * 1000
                t.text((64, H - 64), f"{mm:.0f} mm", font=font, fill=colour(mm / 1000) + (int(235 * a),), anchor="ls")
                t.text((64, H - 124), "joint error, watch vs video", font=small, fill=(255, 255, 255, int(150 * a)), anchor="ls")
            for i, (name, rgb) in enumerate([("video pose and blade", (255, 255, 255)), ("Apple Watch", mint)]):
                y = H - 64 - 48 * (1 - i)
                t.line([(W - 440, y - 10), (W - 400, y - 10)], fill=rgb + (int(210 * a),), width=4)
                t.text((W - 382, y), name, font=small, fill=(255, 255, 255, int(180 * a)), anchor="ls")
        img.alpha_composite(txt)
        enc.stdin.write(np.asarray(img.convert("RGB")).tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"{out}  mean per-joint error {np.nanmean(body[start:end]) * 1000:.0f} mm over {np.isfinite(body[start:end]).sum()} frames")


if __name__ == "__main__":
    main()
