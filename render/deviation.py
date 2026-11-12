"""The watch-only pose and blade through the video's camera: against the video's own pose, or over the video.

    uv run python -m render.deviation data/processed/demo_video30.sam3d.npz data/processed/demo_video30.watchpose.npz \
        data/raw/demo/watch data/processed/cut_deviation.mp4 71.5 89.5
    uv run python -m render.deviation data/processed/demo_video30.sam3d.npz data/processed/demo_video30.watchpose.npz \
        data/raw/demo/watch data/processed/cut_over.mp4 33.5 49.5 --over data/raw/demo/video30.mp4 \
        [--matte data/processed/cut_fencer/matte.mp4 --back pool]
Without --over there are no video pixels: the video's pose and the blade found in the video (<take>.blade.npz
from imu.blade_video) are a white ghost on black, and the watch's bones and blade are coloured by how far off
they are. With --over the watch's skeleton and blade are drawn on the video itself, with everything but the
fencer and the real blade blurred. --matte (render.matte's alpha for the same seconds) cuts the fencer out
properly, and --back then says what goes behind: blur, dim (the room pushed back into the dark), pool (black,
with the real floor lit under the fencer) or studio (the 3D segment's dark floor and piste). Either way the watch skeleton gets the video's pelvis, since the watch can't
see where the fencer stands, and the watch's blade starts where imu.blade_video.watch_grip puts it on the
predicted wrist. Prints the mean per-joint error over the frames drawn.
"""

import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter, grey_closing, zoom as resize

from imu.blade_video import line_shift, watch_blade, watch_grip
from imu.calib import KNUCKLE_KP
from imu.synth import WRIST_KP
from model.pairs import JOINTS
from render.prepare import BLADE_M, BLADE_MODEL, BONES

SS = 2  # supersampling for antialiased lines
GOOD, BAD = 0.04, 0.15  # metres: full green at or under GOOD, full red at or over BAD
BLADE_GOOD, BLADE_BAD = 4.0, 20.0  # the same for the blade's image angle, degrees
STOPS = np.array([[110, 231, 183], [253, 224, 120], [255, 107, 107]], dtype=float)
BG = (11, 11, 13)
ZOOM, LIFT = 0.8, 90
FADE = 1.0  # render.cut crossfades this clip in and out over a second; keep the text out of the fades
MASK = 4  # the fencer's cut-out is built at 1/MASK resolution
BLUR, DIM = 14, 0.72  # the background behind the fencer: Gaussian radius in pixels, brightness
FONT = "/System/Library/Fonts/SFNS.ttf"
SOLE = 0.035  # the foot keypoints sit this far above the floor
STEEL = np.array([205, 208, 214], np.float32)  # the real blade is dark on the wall; on a dark ground it is redrawn light
FLOOR, PISTE, LINE = (np.array(c, np.float32) for c in ((22, 22, 26), (36, 37, 43), (150, 156, 156)))


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


def cutout(body, blade, guard_r, w, h):
    """Soft mask (h, w), 1 on the fencer and the real blade: body (V, 2) projected mesh vertices, blade the
    two pixel ends of the blade seen in the video, guard_r its bell guard's radius in pixels. The mesh is the
    bare body, so the mask is let out to take in hair, a loose shirt and shoes."""
    m = np.zeros((h // MASK, w // MASK), bool)
    ij = np.round(body / MASK).astype(int)
    ij = ij[(ij[:, 0] >= 0) & (ij[:, 0] < m.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < m.shape[0])]
    m[ij[:, 1], ij[:, 0]] = True
    m = binary_dilation(binary_closing(m, np.ones((5, 5))), np.ones((5, 5)))
    m = gaussian_filter((m | band(blade, guard_r, w, h)).astype(np.float32), 1.2)
    return Image.fromarray((m * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)


def band(blade, guard_r, w, h):
    """Bool mask at 1/MASK resolution around the real blade and its bell guard."""
    line = Image.new("L", (w // MASK, h // MASK))
    g = ImageDraw.Draw(line)
    a, b = (tuple(p / MASK) for p in blade)
    g.line([a, b], fill=255, width=5)
    if guard_r:
        r = guard_r / MASK
        g.ellipse([a[0] - r, a[1] - r, a[0] + r, a[1] + r], fill=255)
    g.ellipse([b[0] - 3, b[1] - 3, b[0] + 3, b[1] + 3], fill=255)
    return np.asarray(line) > 0


def gray_frames(video, n, w, h):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-vf", f"scale={w}:{h}", "-frames:v", str(n),
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    for _ in range(n):
        yield np.frombuffer(p.stdout.read(w * h), np.uint8).reshape(h, w).astype(np.float32) / 255


class Floor:
    """The floor under the fencer as the video's camera sees it: where each pixel's ray meets it, in metres
    along the piste (rear ankle to front) and across it. up is the heading frame's, the plane sits under the
    lowest the feet go, and the camera is static, so this is worked out once at half resolution."""

    def __init__(self, K, up, feet, ankles, w, h):
        self.up = up
        self.h = np.nanpercentile((feet @ up).min(1), 5) - SOLE
        along = np.nanmean(ankles, 0)
        along -= (along @ up) * up
        self.along = along / np.linalg.norm(along)
        self.side = np.cross(up, self.along)
        u, v = np.meshgrid(np.arange(0.5, w, 2), np.arange(0.5, h, 2))
        ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
        t = self.h / np.minimum(ray @ up, -1e-3) if self.h < 0 else self.h / np.maximum(ray @ up, 1e-3)
        self.on = (ray @ up) * np.sign(self.h) > 1e-3  # below the horizon
        q = ray * t[..., None]
        self.s, self.l = q @ self.along, q @ self.side
        # How many metres one (half-resolution) pixel covers, for line edges a pixel soft.
        self.ds = np.hypot(*np.gradient(self.s)) + 1e-6
        self.dl = np.hypot(*np.gradient(self.l)) + 1e-6
        self.size = (w, h)

    def at(self, p):
        """(along, across) of the floor point under the camera-frame point p."""
        return p @ self.along, p @ self.side

    def near(self, p, r_along, r_across):
        """Gaussian falloff on the floor around the point under p; 0 above the horizon."""
        s, l = self.at(p)
        return np.where(self.on, np.exp(-((self.s - s) / r_along) ** 2 - ((self.l - l) / r_across) ** 2), 0)

    def full(self, x):
        return resize(x, (2, 2) + (1,) * (x.ndim - 2), order=1)[:self.size[1], :self.size[0]]

    def piste(self, centre):
        """The studio floor's colour before lighting: a 1.5 m piste through `centre` with edge and cross lines."""
        s0, l0 = self.at(centre)
        l = np.abs(self.l - l0)
        img = np.where((l < 0.75)[..., None], PISTE, FLOOR)
        edge = np.clip(0.5 + (0.012 - np.abs(l - 0.75)) / self.dl, 0, 1)
        marks = np.max([np.clip(0.5 + (0.012 - np.abs(self.s - s0 - x)) / self.ds, 0, 1) for x in (-2, 0, 2)], 0) * (l < 0.75)
        return img + (LINE - img) * np.maximum(edge, marks)[..., None]


def backdrop(back, frame, floor, piste, pelvis, feet):
    """What goes behind the cut-out fencer in one frame (h, w, 3 float): see the module docstring."""
    bg = np.array(BG, np.float32)
    if back == "studio":
        light = floor.near(pelvis, 2.6, 2.6)
        shade = 1 - 0.3 * floor.near(pelvis, 0.6, 0.45)
        for f in feet:  # contact shadows, fading as a foot lifts
            shade *= 1 - 0.6 * floor.near(f, 0.15, 0.12) * np.clip(1 - ((f @ floor.up) - floor.h - SOLE) / 0.12, 0, 1)
        return floor.full(bg + (piste - bg) * (light * shade * floor.on)[..., None])
    pool = floor.full(floor.near(pelvis, 1.5, 0.5))[..., None]
    if back == "pool":
        return bg + (frame * 0.9 - bg) * pool
    h, w = frame.shape[:2]
    lum = gaussian_filter(frame @ np.array([0.3, 0.59, 0.11], np.float32), 3)[..., None] * np.array([0.21, 0.22, 0.25], np.float32)
    return lum + (frame * 0.85 - lum) * pool


def main():
    args = sys.argv[1:]
    over = args.pop(args.index("--over") + 1) if "--over" in args else None
    matte = args.pop(args.index("--matte") + 1) if "--matte" in args else None
    behind = args.pop(args.index("--back") + 1) if "--back" in args else "blur"
    args = [a for a in args if a not in ("--over", "--matte", "--back")]
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
    model, pose = np.load(BLADE_MODEL), np.nan_to_num(w["pose"])
    axis = at_video(watch_blade(model, watch_dir, w["t"], pose)) @ w["R"]
    hand = at_video(watch_grip(model, watch_dir, w["t"], pose)) @ w["R"]
    WR = JOINTS.index(WRIST_KP["right"])
    guard = pred[:, WR] + hand
    tip = guard + BLADE_M * axis / np.linalg.norm(axis, axis=1, keepdims=True)
    # The video's blade is an image angle; it starts where the same grip sits on the video's wrist, and is
    # drawn at the length a side-on blade would have.
    seen = np.load(sam.replace(".sam3d.npz", ".blade.npz"))
    grip = true[:, WR] + hand
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
    if over:
        # What stays sharp: the body's mesh and the blade seen in the video, found from its own knuckle keypoint.
        mesh = d["pred_vertices"][start:end] + d["pred_cam_t"][start:end, None]
        knuckle = d["pred_keypoints_3d"][:, KNUCKLE_KP["right"]] + d["pred_cam_t"]
        ang = np.where(np.isfinite(seen["obs"]) & (seen["conf"] > 0), seen["obs"],
                       np.arctan2(*(project(tip) - project(guard)).T[::-1]))
        shift = line_shift(over, project(knuckle), ang, range(start, end), W, H)
        still = ends = None
        alpha = gray_frames(matte, end - start, W, H) if matte else None
        if behind != "blur":
            ok = found[start:end]
            floor = Floor(K, w["R"][1], true[start:end, 15:21][ok], (true[start:end, 14] - true[start:end, 13])[ok], W, H)
            piste = floor.piste(pelvis[start:end, 0][ok].mean(0))
    bone_w, blade_w = (3, 3) if over else (5, 4)
    for k in range(start, end):
        img = next(back) if over else Image.new("RGBA", (W, H), BG + (255,))
        if over:
            if found[k] or ends is None:
                dirn, px = np.array([np.cos(ang[k]), np.sin(ang[k])]), K[0, 0] / knuckle[k, 2]
                on = project(knuckle[k]) + shift[k - start] * np.array([-dirn[1], dirn[0]])
                ends, guard_r, at = (on + 0.04 * px * dirn, on + (BLADE_M + 0.06) * px * dirn), 0.085 * px, k
                if not matte:
                    still = cutout(project(mesh[k - start]), ends, guard_r, W, H)
            soft = img.filter(ImageFilter.GaussianBlur(BLUR)).point(lambda v: v * DIM)
            if not matte:
                soft.putalpha(255)
                img = Image.composite(img, soft, still)
            else:
                rgb = np.asarray(img.convert("RGB"), np.float32)
                a = np.clip((next(alpha) - 0.1) / 0.8, 0, 1)
                # The matte keeps the guard but mostly drops the blade, a thin dark line: find it by how much
                # darker it is than its surroundings.
                near = gaussian_filter(np.kron(band(ends, 0, W, H), np.ones((MASK, MASK), np.float32)), 2)
                lum = rgb.mean(-1)
                dark = np.clip((grey_closing(lum, size=(11, 11)) - lum - 10) / 35, 0, 1) * near
                if behind == "blur":
                    mix = np.asarray(soft.convert("RGB"), np.float32)
                    mix += (rgb - mix) * np.maximum(a, dark)[..., None]
                else:
                    mix = backdrop(behind, rgb, floor, piste, pelvis[at, 0], [true[at, 15:18].mean(0), true[at, 18:21].mean(0)])
                    mix += (rgb - mix) * a[..., None]
                    mix += (STEEL - mix) * dark[..., None]  # dark steel would vanish on a dark ground
                img = Image.fromarray(np.clip(mix, 0, 255).astype(np.uint8)).convert("RGBA")
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
            if not over:  # the hand, from the wrist to where it holds the blade
                g.line([tuple(watch2d[WR]), a], fill=colour(err[k, WR]) + (255,), width=bone_w * SS)
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
