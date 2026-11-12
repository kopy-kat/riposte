"""The blade's image direction found in the video, and the watch-frame blade direction fitted to it.

    uv run python -m imu.blade_video data/processed/blade.npz <watch dir>,<npz>,<video30.mp4>[,offset] ...
fits one blade model over the given takes for the watch-only render (watch_blade), and where the blade
passes the hand (watch_grip).

The blade is a thin dark line against a lighter background: from the grip, score rays at each image
angle by how much darker the ray is than both its sides, and keep the best angle where it stands out.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
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


def fit_b(MA, K, grip, obs, w):
    """Watch-frame blade direction b whose projected angle matches the observed blade angle.

    MA: (T, 3, 3) camera <- watch per frame.
    """
    def axis(x):
        b = np.array([np.cos(x[0]) * np.cos(x[1]), np.sin(x[0]) * np.cos(x[1]), np.sin(x[1])])
        return MA @ b, b

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
    obs, conf = detections(npz, video, project(K, grip))
    c = calibrate(watch, npz, hand, offset)
    ok = c["valid"] & np.isfinite(obs) & np.isfinite(grip).all(1) & (conf > np.percentile(conf[conf > 0], 50))
    b = fit_b(np.einsum("ij,tjk->tik", c["A"], c["R_w"][ok]), K, grip[ok], obs[ok], np.ones(ok.sum()))

    def err(axis):
        e = np.degrees(np.abs((image_angle(K, grip[ok], axis[ok]) - obs[ok] + np.pi) % (2 * np.pi) - np.pi))
        return np.median(e), np.percentile(e, 90)

    watch_axis = np.einsum("ij,tjk,k->ti", c["A"], c["R_w"], b)
    h = heuristic_axis(npz, hand)
    print(f"{ok.sum()} confident frames of {len(ok)}; image-angle error median/p90 deg: "
          f"forearm heuristic {err(h)[0]:.0f}/{err(h)[1]:.0f}, watch fitted to video blade {err(watch_axis)[0]:.0f}/{err(watch_axis)[1]:.0f}")
    return b


def detections(npz, video, grip_px):
    """detect() over the video, cached next to the npz as <take>.blade.npz."""
    cache = Path(npz.replace(".sam3d.npz", ".blade.npz"))
    if not cache.exists():
        st = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                        "stream=width,height", "-of", "json", video], capture_output=True, check=True).stdout)
        obs, conf = detect(video, grip_px, st["streams"][0]["width"], st["streams"][0]["height"])
        np.savez(cache, obs=obs, conf=conf)
    d = np.load(cache)
    return d["obs"], d["conf"]


def line_shift(video, grip_px, angle, frames, width, height):
    """Sideways distance in pixels from the grip keypoint to the blade seen at `angle`, along the normal
    (-sin, cos), for the frame indices `frames`: detect()'s ray score at the one angle, over finer shifts."""
    w, h = int(width * SCALE), int(height * SCALE)
    r, shifts = np.arange(R0, R1, 2.0), np.arange(-24, 25, 1.0)
    want, out = set(int(i) for i in frames), {}
    for i, img in enumerate(gray_frames(video, w, h)):
        if i not in want:
            continue
        d = np.array([np.cos(angle[i]), np.sin(angle[i])])
        n = np.array([-d[1], d[0]])
        pts = grip_px[i] * SCALE + shifts[:, None, None] * n + r[None, :, None] * d

        def sample(q):
            return map_coordinates(img, [q[..., 1].ravel(), q[..., 0].ravel()], order=1, cval=np.nan).reshape(q.shape[:2])

        resp = np.nanmean(np.minimum(sample(pts + SIDE * n), sample(pts - SIDE * n)) - sample(pts), axis=1)
        out[i] = shifts[int(np.argmax(np.nan_to_num(resp, nan=-np.inf)))] / SCALE
    return np.array([out.get(int(i), np.nan) for i in frames])


def fit_grip(specs, per_take=400):
    """Where the blade passes the hand, as one offset from the wrist in the watch frame, metres: the watch is
    strapped beside the hand that holds the grip, and the blade leaves through the bell guard's centre, which
    is neither the wrist nor the knuckles. specs as fit_model's; the point is the knuckle keypoint moved
    sideways in the image onto the blade seen there."""
    from imu.load import load_sensor_logger
    from imu.synth import WRIST_KP
    from model.pairs import attitude, pair

    rng = np.random.default_rng(0)
    c = []
    for spec in specs:
        w, n, v, *off = spec.split(",")
        p = pair(w, n, offset=float(off[0]) if off else None)
        d = np.load(n)
        K = d["cam_int"].reshape(-1, 3, 3)[0]
        kn = d["pred_keypoints_3d"][:, KNUCKLE_KP["right"]].astype(float) + d["pred_cam_t"].astype(float)
        wr = d["pred_keypoints_3d"][:, WRIST_KP["right"]].astype(float) + d["pred_cam_t"].astype(float)
        o, conf = detections(n, v, project(K, kn))
        df = load_sensor_logger(w)
        tv = np.arange(len(o)) / float(d["fps"]) + p["offset"]
        tw = df.index.to_numpy()
        ok = (d["found"].astype(bool) & (tv >= tw[0]) & (tv <= tw[-1]) & np.isfinite(o) & np.isfinite(kn).all(1)
              & (conf > np.percentile(conf[conf > 0], 50)))
        i = np.sort(rng.choice(np.flatnonzero(ok), min(per_take, ok.sum()), replace=False))
        shift = line_shift(v, project(K, kn), o, i, 2 * K[0, 2], 2 * K[1, 2])
        side = np.stack([-np.sin(o[i]), np.cos(o[i]), np.zeros(len(i))], 1) * (shift * kn[i, 2] / K[0, 0])[:, None]
        MA = np.einsum("ij,tjk->tik", p["Q"], attitude(df, tv[i]))
        c.append(np.einsum("tji,tj->ti", MA, kn[i] + side - wr[i])[np.abs(shift) < 24 / SCALE])
    return np.median(np.concatenate(c), 0)


LAGS = (-0.2, -0.1, 0.0, 0.1, 0.2)  # s of watch input either side of a frame for the b(t) correction
RIDGE = 1e-2


def lagged(imu, rate):
    """(T, 9 * len(LAGS)) b(t) inputs from model.pairs.imu_at rows sampled at `rate`."""
    n = len(imu)
    return np.concatenate([imu[np.clip(np.arange(n) + round(l * rate), 0, n - 1)] for l in LAGS], 1)


def tangent(b):
    e1 = np.cross(b, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(b, e1)


def bent(m, x):
    """b(t), (T, 3): the model's b turned by its ridge correction (radians along tangent(b)) for inputs x."""
    d = np.c_[(x - m["mu"]) / m["sd"], np.ones(len(x))] @ m["W"]
    e1, e2 = tangent(m["b"])
    v = m["b"] + d[:, :1] * e1 + d[:, 1:] * e2
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def fit_model(specs, per_take=1500, clip=0.6):
    """One watch-frame blade direction b over all the takes, each in its own watch_to_camera frame, and a
    ridge map from the watch input to b(t)'s turn away from it. Returns the npz fields watch_blade reads.

    specs: "watch,npz,video[,offset]". The turn is fitted to image-angle residuals through their Jacobian,
    clipped at `clip` rad so detection blunders don't set it.
    """
    from imu.load import load_sensor_logger
    from model.pairs import attitude, imu_at, pair

    rng = np.random.default_rng(0)
    MA, grip, obs, x, forearm = [], [], [], [], []
    for spec in specs:
        w, n, v, *off = spec.split(",")
        p = pair(w, n, offset=float(off[0]) if off else None)
        d = np.load(n)
        K = d["cam_int"].reshape(-1, 3, 3)[0]
        g = d["pred_keypoints_3d"][:, KNUCKLE_KP["right"]].astype(float) + d["pred_cam_t"].astype(float)
        o, conf = detections(n, v, project(K, g))
        df = load_sensor_logger(w)
        fps = float(d["fps"])
        tv = np.arange(len(o)) / fps + p["offset"]
        tw = df.index.to_numpy()
        ok = (d["found"].astype(bool) & (tv >= tw[0]) & (tv <= tw[-1]) & np.isfinite(o) & np.isfinite(g).all(1)
              & (conf > np.percentile(conf[conf > 0], 50)))
        i = np.sort(rng.choice(np.flatnonzero(ok), min(per_take, ok.sum()), replace=False))
        MA.append(np.einsum("ij,tjk->tik", p["Q"], attitude(df, tv[i])))
        grip.append(g[i])
        obs.append(o[i])
        x.append(lagged(imu_at(df, tv), fps)[i])
        forearm.append(p["forearm"])
    MA, grip, obs, x = (np.concatenate(a) for a in (MA, grip, obs, x))
    # fx = fy in every take, so the blade's image angle does not depend on K: one identity K serves all.
    K = np.eye(3)
    b = fit_b(MA, K, grip, obs, np.ones(len(obs)))
    ang = lambda bv: image_angle(K, grip, MA @ bv)
    e = np.clip(np.angle(np.exp(1j * (ang(b) - obs))), -clip, clip)
    e1, e2 = tangent(b)
    J = np.stack([np.angle(np.exp(1j * (ang(b + 1e-3 * u) - ang(b - 1e-3 * u)))) / 2e-3 for u in (e1, e2)], 1)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    F = np.c_[(x - mu) / sd, np.ones(len(x))]
    X = np.c_[J[:, :1] * F, J[:, 1:] * F]
    W = np.linalg.solve(X.T @ X + RIDGE * len(e) * np.eye(X.shape[1]), X.T @ -e).reshape(2, -1).T
    f = np.mean(forearm, 0)
    return {"b": b, "forearm": f / np.linalg.norm(f), "W": W, "mu": mu, "sd": sd, "grip": fit_grip(specs)}


def body_from_watch(m, df, t, pose, hand="right"):
    """(T, 3, 3) rotations from the watch frame to the watch-only pose's body frame (right, up, back).

    The body frame's up is gravity (model.pairs.heading_frame), so only its heading is unknown: one angle
    for the take, turning the watch's forearm axis about up onto the predicted forearm.
    """
    from imu.synth import ELBOW_KP, WRIST_KP
    from model.pairs import JOINTS, attitude

    R_w = attitude(df, t)
    C = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # reference (x, y, z up) -> body axes at zero heading
    u = pose[:, JOINTS.index(WRIST_KP[hand])] - pose[:, JOINTS.index(ELBOW_KP[hand])]
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    v = np.einsum("ij,tjk,k->ti", C, R_w, m["forearm"])
    # Heading is only defined where both forearms are far from vertical.
    flat = (np.hypot(u[:, 0], u[:, 2]) > 0.5) & (np.hypot(v[:, 0], v[:, 2]) > 0.5)
    d = np.arctan2(u[flat, 0], u[flat, 2]) - np.arctan2(v[flat, 0], v[flat, 2])
    psi = np.angle(np.exp(1j * d).mean())
    psi += np.median(np.angle(np.exp(1j * (d - psi))))
    c, s = np.cos(psi), np.sin(psi)
    Y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.einsum("ij,jk,tkl->til", Y, C, R_w)


def watch_blade(m, watch, t, pose, hand="right"):
    """Blade direction (T, 3) in the watch-only pose's body frame at watch times t, from the watch alone: m is
    a fit_model() result, pose the predicted (T, J, 3) model.pairs.JOINTS."""
    from imu.load import load_sensor_logger
    from model.pairs import RATE, imu_at

    df = load_sensor_logger(watch)
    b = bent(m, lagged(imu_at(df, t), RATE))
    # The correction jitters frame to frame, zigzagging the tip; smoothing it also scores slightly better.
    a = gaussian_filter1d(np.einsum("til,tl->ti", body_from_watch(m, df, t, pose, hand), b),
                          2 * len(t) / (t[-1] - t[0]) / RATE, axis=0)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def watch_grip(m, watch, t, pose, hand="right"):
    """Where the blade passes the hand, (T, 3) from the predicted wrist in the body frame: m's fit_grip offset
    carried by the watch's attitude."""
    from imu.load import load_sensor_logger

    return body_from_watch(m, load_sensor_logger(watch), t, pose, hand) @ m["grip"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("takes", nargs="+", help="watch,npz,video[,offset]")
    a = p.parse_args()
    np.savez(a.out, **fit_model(a.takes))
    print(a.out)
