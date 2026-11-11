"""The blade's image direction found in the video, and the watch-frame blade direction fitted to it.

    uv run python -m imu.blade_video <watch dir> <npz> <video30.mp4>

The blade is a thin dark line against a lighter background: from the grip, score rays at each image
angle by how much darker the ray is than both its sides, and keep the best angle where it stands out.
"""

import argparse
import subprocess

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from imu.calib import KNUCKLE_KP, calibrate, heuristic_axis

SCALE = 0.5  # detect at half resolution
R0, R1 = 30, 180  # ray span in half-res pixels: past the guard, short enough to survive foreshortening
SIDE = 3  # px to each side of the ray
SHIFT = np.arange(-12, 13, 2.0)  # the blade passes the guard, not the knuckle keypoint: let rays shift sideways


def gray_frames(video, w, h):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(w * h)
        if len(b) < w * h:
            return
        yield np.frombuffer(b, np.uint8).reshape(h, w).astype(np.float32)


def project(K, p):
    q = p @ K.T
    return q[..., :2] / q[..., 2:]


def detect(video, grip_px, width, height):
    """Per frame: blade image angle (rad, atan2 of dy, dx from the grip) and a confidence (0 if not found)."""
    w, h = int(width * SCALE), int(height * SCALE)
    ang = np.radians(np.arange(0, 360, 2.0))
    r = np.arange(R0, R1, 2.0)
    d = np.stack([np.cos(ang), np.sin(ang)], 1)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    out_a, out_c = np.full(len(grip_px), np.nan), np.zeros(len(grip_px))
    for i, img in enumerate(gray_frames(video, w, h)):
        if i >= len(grip_px):
            break
        g = grip_px[i] * SCALE
        if not np.isfinite(g).all():
            continue
        # (angles, shifts, radii, xy)
        pts = g + SHIFT[None, :, None, None] * n[:, None, None, :] + r[None, None, :, None] * d[:, None, None, :]

        def sample(p):
            return map_coordinates(img, [p[..., 1].ravel(), p[..., 0].ravel()], order=1, cval=np.nan).reshape(p.shape[:3])

        side = SIDE * n[:, None, None, :]
        resp = np.nanmean(np.minimum(sample(pts + side), sample(pts - side)) - sample(pts), axis=2)
        resp = np.nan_to_num(resp, nan=-np.inf).max(axis=1)
        k = int(np.argmax(resp))
        far = np.abs(((np.arange(len(ang)) - k + len(ang) / 2) % len(ang)) - len(ang) / 2) > 15 / 2
        second = resp[far].max()
        out_a[i] = ang[k]
        out_c[i] = max(resp[k] - max(second, 0), 0)
    return out_a, out_c


def image_angle(K, grip, axis, length=0.9):
    a, b = project(K, grip), project(K, grip + length * axis)
    v = b - a
    return np.arctan2(v[:, 1], v[:, 0])


def fit_b(c, K, grip, obs, w):
    """Watch-frame blade direction b whose projected angle matches the observed blade angle."""
    def axis(x):
        b = np.array([np.cos(x[0]) * np.cos(x[1]), np.sin(x[0]) * np.cos(x[1]), np.sin(x[1])])
        return np.einsum("ij,tjk,k->ti", c["A"], c["R_w"], b), b

    def res(x):
        phi = image_angle(K, grip, axis(x)[0])
        return np.concatenate([w * (np.cos(phi) - np.cos(obs)), w * (np.sin(phi) - np.sin(obs))])

    best = min((least_squares(res, x0, loss="soft_l1", f_scale=0.1) for x0 in
                [(az, el) for az in np.radians(range(0, 360, 45)) for el in np.radians((-45, 0, 45))]),
               key=lambda s: s.cost)
    return axis(best.x)[1]


def fit(watch, npz, video, hand="right", offset=None):
    """Returns b (watch frame), and per-frame observed angle, confidence and the angle errors before/after."""
    d = np.load(npz)
    K = d["cam_int"].reshape(-1, 3, 3)[0]
    grip = d["pred_keypoints_3d"][:, KNUCKLE_KP[hand]].astype(float) + d["pred_cam_t"].astype(float)
    import json

    st = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=width,height", "-of", "json", video], capture_output=True, check=True).stdout)
    width, height = st["streams"][0]["width"], st["streams"][0]["height"]
    obs, conf = detect(video, project(K, grip), width, height)
    c = calibrate(watch, npz, hand, offset)
    ok = c["valid"] & np.isfinite(obs) & np.isfinite(grip).all(1) & (conf > np.percentile(conf[conf > 0], 50))
    b = fit_b({**c, "R_w": c["R_w"][ok]}, K, grip[ok], obs[ok], np.ones(ok.sum()))

    def err(axis):
        e = np.degrees(np.abs((image_angle(K, grip[ok], axis[ok]) - obs[ok] + np.pi) % (2 * np.pi) - np.pi))
        return np.median(e), np.percentile(e, 90)

    watch_axis = np.einsum("ij,tjk,k->ti", c["A"], c["R_w"], b)
    h = heuristic_axis(npz, hand)
    print(f"{ok.sum()} confident frames of {len(ok)}; image-angle error median/p90 deg: "
          f"forearm heuristic {err(h)[0]:.0f}/{err(h)[1]:.0f}, watch fitted to video blade {err(watch_axis)[0]:.0f}/{err(watch_axis)[1]:.0f}")
    return b


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("watch")
    p.add_argument("npz")
    p.add_argument("video")
    p.add_argument("--offset", type=float, default=None)
    a = p.parse_args()
    print("b", np.round(fit(a.watch, a.npz, a.video, offset=a.offset), 3))
