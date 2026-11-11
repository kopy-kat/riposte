"""Watch-to-camera calibration and the watch-measured blade.

Model: R_cam<-wrist(t) @ M = A @ R_ref<-watch(t), with M the strap mount (watch frame in the wrist-joint frame)
and A the CoreMotion reference frame in the camera frame.

    uv run python -m imu.calib data/raw/office2/watch data/processed/office2_video.sam3d.npz
"""

import argparse

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from imu.load import load_sensor_logger
from imu.sync import clock_offset
from imu.synth import ELBOW_KP, WRIST_KP, synth

# Sensor Logger's quaternion is CoreMotion attitude, ref<-watch with ref z up: on office2,
# R.T @ (0, 0, -1) matches the logged gravity to 1e-4 degrees.
DOWN_REF = np.array([0.0, 0.0, -1.0])
DOWN_CAM = np.array([0.0, 1.0, 0.0])  # level camera, as imu.synth and model.pairs assume
KNUCKLE_KP = {"right": 32, "left": 53}


def _proj(S):
    """Rotation maximising tr(R.T @ S) (Wahba / chordal mean)."""
    U, _, Vt = np.linalg.svd(S)
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def _angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def solve(R_c, R_w, gravity_weight=0.0, scale_deg=10.0, iters=60):
    """Fit M, A to R_c[t] @ M = A @ R_w[t] over the given frames.

    Cost: sum_t w_t (||R_c M - A R_w||^2 + g ||R_c M R_w^T d_ref - d_cam||^2) + g N ||A d_ref - d_cam||^2,
    alternating closed-form SVD steps for M and A, with Cauchy IRLS weights w_t on the per-frame residual
    angle. The rotations alone fix all of A; the gravity terms (weight g) only add the level-camera
    assumption, so they bias A by the camera's real tilt and default off, leaving the fitted tilt as a check.
    Started from the 24 axis-aligned rotations for M, keeping the lowest robust cost, because the
    alternation has local minima.
    Returns M, A, per-frame residual angles (deg), weights.
    """
    g_w = np.einsum("tji,j->ti", R_w, DOWN_REF)  # gravity in the watch frame
    c = np.radians(scale_deg)
    best = None
    for M0 in Rotation.create_group("O").as_matrix():
        M, w = M0, np.ones(len(R_c))
        for _ in range(iters):
            A = _proj(np.einsum("t,tij,jk,tlk->il", w, R_c, M, R_w)
                      + gravity_weight * w.sum() * np.outer(DOWN_CAM, DOWN_REF))
            d_wrist = np.einsum("tji,j->ti", R_c, DOWN_CAM)  # camera down in the wrist frame
            M = _proj(np.einsum("t,tji,jk,tkl->il", w, R_c, A, R_w)
                      + gravity_weight * np.einsum("t,ti,tj->ij", w, d_wrist, g_w))
            res = np.radians(_angle(np.einsum("tij,jk,tlk,ml->tim", R_c, M, R_w, A)))
            w = 1 / (1 + (res / c) ** 2)
        cost = np.log1p((res / c) ** 2).sum()
        if best is None or cost < best[0]:
            best = cost, M, A, np.degrees(res), w
    return best[1:]


def align(watch, npz, hand="right", offset=None, joint=None):
    """Watch attitude and video wrist rotation per video frame.

    offset: watch time of video t=0; default model.pairs' tap + gyro cross-correlation.
    Returns dict: R_c, R_w (N,3,3), valid (N,), fps, offset, gyro (N,) watch |gyro| rad/s.
    """
    from model.pairs import coarse_offset, fine_offset

    df = load_sensor_logger(watch)
    sy = synth(npz, hand, joint=joint)
    if offset is None:
        clock = clock_offset(watch)
        offset, _ = fine_offset(df, sy, coarse_offset(df, sy) if clock is None else clock)
    tw = df.index.to_numpy()
    t = sy.index.to_numpy() + offset
    inside = (t >= tw[0]) & (t <= tw[-1])
    tc = np.clip(t, tw[0], tw[-1])
    R_w = Slerp(tw, Rotation.from_quat(df[["qw", "qx", "qy", "qz"]].to_numpy(), scalar_first=True))(tc)
    R_c = Rotation.from_quat(sy[["qw", "qx", "qy", "qz"]].to_numpy(), scalar_first=True)
    gyro = np.interp(tc, tw, np.linalg.norm(df[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(), axis=1))
    return dict(R_c=R_c.as_matrix(), R_w=R_w.as_matrix(), valid=sy["valid"].to_numpy() & inside,
                fps=1 / (sy.index[1] - sy.index[0]), offset=offset, gyro=gyro)


def calibrate(watch, npz, hand="right", offset=None, joint=None, gravity_weight=0.0):
    """align + solve, printing residual statistics. Returns the align dict plus M, A, res, w."""
    a = align(watch, npz, hand, offset, joint)
    v = a["valid"]
    M, A, res, _ = solve(a["R_c"][v], a["R_w"][v], gravity_weight)
    full = np.full(len(v), np.nan)
    full[v] = res
    tilt = np.degrees(np.arccos(np.clip(A @ DOWN_REF @ DOWN_CAM, -1, 1)))
    print(f"offset {a['offset']:.3f} s  frames {v.sum()}/{len(v)}  residual median {np.median(res):.1f} deg  "
          f"p90 {np.percentile(res, 90):.1f} deg  inliers(<20 deg) {(res < 20).mean():.0%}  "
          f"camera tilt vs level {tilt:.1f} deg")
    return {**a, "M": M, "A": A, "res": full}


def heuristic_axis(npz, hand="right"):
    """render/prepare.py's blade axis: the mean of forearm and hand directions, NaN where no person."""
    kp = np.load(npz)["pred_keypoints_3d"].astype(float)
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    e, wr, k = kp[:, ELBOW_KP[hand]], kp[:, WRIST_KP[hand]], kp[:, KNUCKLE_KP[hand]]
    return unit(unit(wr - e) + unit(k - wr))


def fit_direction(R_w, A, target):
    """Fixed watch-frame unit vector b minimising sum ||A R_w b - target||^2."""
    b = np.einsum("tji,kj,tk->i", R_w, A, target)
    return b / np.linalg.norm(b)


def stillest(gyro, valid, fps, hold_s=3.0):
    """Frame mask of the hold_s window with the lowest mean watch |gyro| among valid frames (the en garde hold)."""
    n = int(round(hold_s * fps))
    g = np.where(valid, gyro, np.inf)
    i = int(np.argmin(np.convolve(g, np.ones(n) / n, mode="valid")))
    m = np.zeros(len(g), bool)
    m[i:i + n] = True
    return m


def watch_blade_axis(watch, npz, hand="right", blade_in_watch=None, hold=None, grip=None, length=0.9,
                     offset=None, joint=None, gravity_weight=0.0):
    """Per-video-frame blade direction in the OpenCV camera frame from the watch attitude: A @ R_w(t) @ b.

    blade_in_watch: b, the blade's direction in the watch frame; default fits it to the keypoint heuristic
    over `hold` (video seconds (start, end); default the stillest 3 s). grip: (N,3) camera-frame grip
    positions; tip = grip + length * axis when given. Frames outside the watch recording hold the edge
    attitude. Returns the calibrate dict plus axis, b, hold, heuristic, tip.
    """
    c = calibrate(watch, npz, hand, offset, joint, gravity_weight)
    h = heuristic_axis(npz, hand)
    ok = c["valid"] & np.isfinite(h).all(1)
    if hold is None:
        m = stillest(c["gyro"], ok, c["fps"])
    else:
        t = np.arange(len(ok)) / c["fps"]
        m = (t >= hold[0]) & (t < hold[1])
    m &= ok
    b = fit_direction(c["R_w"][m], c["A"], h[m]) if blade_in_watch is None else np.asarray(blade_in_watch, float)
    b = b / np.linalg.norm(b)
    axis = np.einsum("ij,tjk,k->ti", c["A"], c["R_w"], b)
    out = {**c, "axis": axis, "b": b, "hold": m, "heuristic": h}
    if grip is not None:
        out["tip"] = grip + length * axis
    err = np.degrees(np.arccos(np.clip((axis[ok] * h[ok]).sum(1), -1, 1)))
    print(f"blade in watch frame {np.round(b, 3)}  fit over {m.sum()} frames from {np.argmax(m) / c['fps']:.1f} s  "
          f"watch vs heuristic axis: median {np.median(err):.1f} deg  p90 {np.percentile(err, 90):.1f} deg")
    return out


def plot(out, npz, video, path, times, hand="right"):
    """Watch blade (green) vs heuristic blade (red) projected over video frames at the given video seconds."""
    import io
    import subprocess

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz)
    K = d["cam_int"].reshape(-1, 3, 3)[0]
    grip = d["pred_keypoints_3d"][:, KNUCKLE_KP[hand]].astype(float) + d["pred_cam_t"].astype(float)
    fig, axs = plt.subplots(1, len(times), figsize=(5 * len(times), 6))
    for ax, ts in zip(np.atleast_1d(axs), times):
        i = int(round(ts * out["fps"]))
        png = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{i / out['fps']:.4f}", "-i", str(video), "-frames:v", "1",
                              "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
        ax.imshow(plt.imread(io.BytesIO(png)))
        for v, col in ((out["axis"][i], "lime"), (out["heuristic"][i], "red")):
            p = np.stack([grip[i], grip[i] + 0.9 * v]) @ K.T
            ax.plot(*(p[:, :2] / p[:, 2:]).T, color=col, lw=3)
        ax.set_title(f"t={ts:.1f}s  residual {out['res'][i]:.0f} deg")
        ax.axis("off")
    fig.suptitle("green: watch blade, red: keypoint heuristic")
    fig.tight_layout()
    fig.savefig(path, dpi=80)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("watch")
    p.add_argument("npz")
    p.add_argument("--hand", default="right", choices=["right", "left"])
    p.add_argument("--offset", type=float, default=None)
    p.add_argument("--video", default=None, help="draw both blades over these frames")
    p.add_argument("--times", type=float, nargs="*", default=[5.0, 15.0, 25.0, 38.0])
    a = p.parse_args()
    out = watch_blade_axis(a.watch, a.npz, a.hand, offset=a.offset)
    if a.video:
        path = a.npz.replace(".sam3d.npz", ".blade.png")
        plot(out, a.npz, a.video, path, a.times, a.hand)
        print(path)
