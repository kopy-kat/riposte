"""Synthetic wrist-watch IMU from a SAM 3D Body npz (see pose/sam3d_modal.py).

    uv run imu/synth.py data/processed/<stem>.sam3d.npz --hand right --out plot.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

G = 9.80665

# MHR skeleton (mhr_model.pt `character_torch.skeleton.joint_parents`, names from lod*.fbx):
# 40 r_lowarm -> 41 r_wrist_twist -> 42 r_wrist, and 76 -> 77 -> 78 on the left. Twist joints sit at the
# wrist and carry forearm roll but not hand flexion, which a strap on the forearm does not feel.
# sam_3d_body/models/meta_arch/sam3d_body.py:1286-1303 uses the same indices ([76, 40] lowarm, [77, 41] twist,
# [78, 42] wrist).
WRIST_JOINT = {"right": 41, "left": 77}
# MHR70 keypoints (sam_3d_body/metadata/mhr70.py:130,151 and the elbows at 7/8).
WRIST_KP = {"right": 41, "left": 62}
ELBOW_KP = {"right": 8, "left": 7}

# pred_joint_coords / keypoints are flipped to the OpenCV frame (mhr_head.py:340-343), but
# pred_global_rots is not: it stays in the MHR frame (y up, z back).
FLIP = np.diag([1.0, -1.0, -1.0])


def _fill(x, valid, kind):
    """Fill invalid frames: linear for vectors, slerp for Rotation; edges hold the nearest value."""
    t = np.arange(len(valid))
    if kind == "rot":
        slerp = Slerp(t[valid], x[valid])
        return slerp(np.clip(t, t[valid][0], t[valid][-1]))
    return np.stack([np.interp(t, t[valid], x[valid, k]) for k in range(x.shape[1])], axis=1)


def _window(fps, smooth_s, n):
    w = max(int(round(smooth_s * fps)) | 1, 5)
    w = min(w, n if n % 2 else n - 1)
    return w


def _angular_rate(rots):
    """Body-frame angular velocity (rad per frame) from a Rotation sequence by central differences."""
    rel = rots[:-2].inv() * rots[2:]
    w = rel.as_rotvec() / 2
    return np.vstack([w[:1], w, w[-1:]])


def synth(npz_path, hand="right", mount=np.eye(3), gravity=(0.0, 1.0, 0.0), smooth_s=0.2, joint=None):
    """Return a DataFrame indexed by time (s) at the video fps.

    mount: 3x3, sensor axes expressed in the wrist joint frame (fixed, unknown; calibrate later).
    gravity: direction of gravity in the camera frame (default +y, down in the image).
    joint: override the MHR joint index used as the sensor frame (default: the wrist_twist joint).
    """
    d = np.load(npz_path)
    fps = float(d["fps"])
    dt = 1.0 / fps
    j = WRIST_JOINT[hand] if joint is None else joint
    R_cam = FLIP @ d["pred_global_rots"][:, j].astype(float) @ FLIP
    pos = d["pred_joint_coords"][:, j].astype(float) + d["pred_cam_t"].astype(float)
    kp = d["pred_keypoints_3d"].astype(float)
    fore = kp[:, WRIST_KP[hand]] - kp[:, ELBOW_KP[hand]]

    valid = d["found"].astype(bool) & np.isfinite(pos).all(1) & np.isfinite(R_cam).all((1, 2)) & np.isfinite(fore).all(1)
    if valid.sum() < 5:
        raise ValueError("fewer than 5 usable frames")
    n = len(valid)
    w = _window(fps, smooth_s, n)

    pos = _fill(pos, valid, "vec")
    rots = _fill(Rotation.from_matrix(np.where(valid[:, None, None], R_cam, np.eye(3))), valid, "rot")
    fore = _fill(fore, valid, "vec")

    g = np.asarray(gravity, float)
    g = G * g / np.linalg.norm(g)
    acc_cam = savgol_filter(pos, w, 3, deriv=2, delta=dt, axis=0)
    mount = np.asarray(mount, float)
    sensor = rots * Rotation.from_matrix(mount)
    # Accelerometers read a - g: at rest they point up.
    acc = sensor.inv().apply(acc_cam - g)

    gyro = savgol_filter(_angular_rate(sensor) / dt, w, 3, axis=0)

    u = fore / np.linalg.norm(fore, axis=1, keepdims=True)
    u = savgol_filter(u, w, 3, axis=0)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    ang = np.arccos(np.clip((u[:-2] * u[2:]).sum(1), -1, 1)) / (2 * dt)
    forearm_speed = savgol_filter(np.r_[ang[0], ang, ang[-1]], w, 3)

    q = sensor.as_quat(scalar_first=True)
    q *= np.where(np.cumsum(np.r_[0, (q[1:] * q[:-1]).sum(1) < 0]) % 2 == 1, -1, 1)[:, None]

    return pd.DataFrame(
        {"ax": acc[:, 0], "ay": acc[:, 1], "az": acc[:, 2],
         "gx": gyro[:, 0], "gy": gyro[:, 1], "gz": gyro[:, 2],
         "qw": q[:, 0], "qx": q[:, 1], "qy": q[:, 2], "qz": q[:, 3],
         "forearm_speed": forearm_speed},
        index=pd.Index(np.arange(n) * dt, name="t"),
    ).assign(valid=valid)


def plot(df, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 7))
    ax[0].plot(df.index, np.linalg.norm(df[["ax", "ay", "az"]], axis=1))
    ax[0].set_ylabel("|accel| m/s²")
    ax[1].plot(df.index, np.linalg.norm(df[["gx", "gy", "gz"]], axis=1))
    ax[1].set_ylabel("|gyro| rad/s")
    ax[2].plot(df.index, df["forearm_speed"])
    ax[2].set_ylabel("forearm speed rad/s")
    ax[2].set_xlabel("t (s)")
    for a in ax:
        a.fill_between(df.index, 0, 1, where=~df["valid"], transform=a.get_xaxis_transform(), color="red", alpha=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--hand", default="right", choices=["right", "left"])
    p.add_argument("--out", default=None)
    p.add_argument("--smooth", type=float, default=0.2, help="Savitzky-Golay window, seconds")
    a = p.parse_args()
    df = synth(a.npz, a.hand, smooth_s=a.smooth)
    out = Path(a.out or Path(a.npz).with_suffix(".imu.png"))
    plot(df, out)
    print(out)
    print(df.drop(columns="valid").describe().loc[["min", "max", "mean"]].T)
