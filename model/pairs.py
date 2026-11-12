"""Paired (watch IMU, video pose) sequences on one 30 Hz time base, in watch time.

    uv run python -m model.pairs data/raw/office2/watch data/processed/office2_video.sam3d.npz [offset] [--save out.npz]
"""

import sys

import numpy as np
from scipy.signal import butter, savgol_filter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

from imu.load import load_sensor_logger
from imu.sync import clock_offset, find_tap, xcorr_offset
from imu.synth import ELBOW_KP, WRIST_KP, sg_window, synth

RATE = 30.0

# MHR70 body keypoints (COCO-17 order without the wrists, then feet, then the two wrists and the arm/neck
# extras); the 42 finger keypoints are left out because a wrist IMU says nothing about them.
JOINTS = list(range(21)) + [WRIST_KP["right"], WRIST_KP["left"]] + list(range(63, 70))
L_HIP, R_HIP = 9, 10
ARM = [6, ELBOW_KP["right"], WRIST_KP["right"]]  # right shoulder, elbow, wrist: the weapon arm
LEGS = list(range(9, 21))  # hips, knees, ankles, feet
IMU_COLS = ["gyr_x", "gyr_y", "gyr_z", "acc_x", "acc_y", "acc_z", "grav_x", "grav_y", "grav_z"]
# Elbow->wrist in the watch frame: the case's -x axis, within 18 deg on every take.
FOREARM_W = np.array([-1.0, 0.0, 0.0])


def coarse_offset(df, sy, search_s=20.0):
    """Watch time of video t=0: watch tap minus the biggest forearm-speed peak in the video's first search_s."""
    early = sy[sy.index < search_s]
    return find_tap(df) - float(early["forearm_speed"].idxmax())


def fine_offset(df, sy, coarse, max_lag_s=1.5):
    """Refine `coarse` (watch time = video time + offset) by cross-correlating |gyro|.

    |gyro| is the same in any sensor frame, so the unknown strap mount drops out.
    """
    rate = df.attrs["rate_hz"]
    w = np.linalg.norm(df[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(), axis=1)
    v = np.linalg.norm(sy[["gx", "gy", "gz"]].to_numpy(), axis=1)
    tw = df.index.to_numpy()
    tv = sy.index.to_numpy() + coarse
    lo, hi = max(tw[0], tv[0]), min(tw[-1], tv[-1])
    keep = (tw >= lo) & (tw <= hi)
    a = savgol_filter(w, sg_window(rate, 0.2, len(w)), 3)[keep]
    b = np.interp(tw[keep], tv, v)  # video signal laid on the watch grid at the coarse offset
    s = xcorr_offset(a, rate, b, rate, max_lag_s)
    # b(t) = video(t - coarse); an event at watch t sits at video t - true, i.e. b at t + coarse - true.
    off = coarse - s
    peak = float(np.corrcoef(a, np.interp(tw[keep], sy.index.to_numpy() + off, v))[0, 1])
    return off, peak


def watch_to_camera(fore_cam, R_w, scale_deg=10.0, iters=20):
    """Rotation camera <- CoreMotion reference (z up) aligning the video forearm directions with the watch's.

    Robust Wahba over frames: fore_cam (T, 3) elbow->wrist in the camera, R_w (T, 3, 3) ref <- watch.
    Its z column is gravity's up in the camera, whatever the camera's tilt. Also returns the forearm axis
    in the watch frame, starting from FOREARM_W.
    """
    from imu.calib import _proj

    u = fore_cam / np.linalg.norm(fore_cam, axis=1, keepdims=True)
    f, w = FOREARM_W, np.ones(len(u))
    for _ in range(iters):
        v = R_w @ f
        Q = _proj(np.einsum("t,ti,tj->ij", w, u, v))
        r = np.arccos(np.clip((u * (v @ Q.T)).sum(1), -1, 1))
        w = 1 / (1 + (r / np.radians(scale_deg)) ** 2)
        # The strap sits differently each take, so the forearm axis in the watch frame is refitted too.
        f = np.einsum("t,tji,kj,tk->i", w, R_w, Q, u)
        f /= np.linalg.norm(f)
    return Q, f


def heading_frame(kp_cam, valid, up):
    """Rotation into a right-handed (right, up, back) body frame, one heading for the whole take.

    up: gravity's up in the camera frame; lateral is the median left->right hip direction flattened onto
    the floor. One heading per take keeps the pelvis yaw in the targets while removing camera placement.
    """
    hip = np.nanmedian((kp_cam[valid, R_HIP] - kp_cam[valid, L_HIP]), axis=0)
    lat = hip - (hip @ up) * up
    lat /= np.linalg.norm(lat)
    return np.stack([lat, up, np.cross(lat, up)])


def imu_at(df, t):
    """(T, 9) model input at watch times t: IMU_COLS, low-passed for the 30 Hz grid."""
    raw = sosfiltfilt(butter(4, 0.4 * RATE, fs=df.attrs["rate_hz"], output="sos"), df[IMU_COLS].to_numpy(), axis=0)
    tw = df.index.to_numpy()
    return np.stack([np.interp(t, tw, raw[:, k]) for k in range(raw.shape[1])], 1)


def attitude(df, t):
    """(T, 3, 3) CoreMotion attitude, reference <- watch, at watch times t (held at the recording's ends)."""
    tw = df.index.to_numpy()
    q = Rotation.from_quat(df[["qw", "qx", "qy", "qz"]].to_numpy(), scalar_first=True)
    return Slerp(tw, q)(np.clip(t, tw[0], tw[-1])).as_matrix()


def level_frame(R):
    """Rotation levelled <- CoreMotion reference, from the watch alone: a turn about the vertical that puts
    the take's mean forearm (FOREARM_W) heading on +x. R (T, 3, 3) is attitude(). Fencing points down the
    piste, so that heading stands in for the body's, which the watch can't see."""
    f = R @ FOREARM_W
    psi = np.angle(np.mean(np.exp(1j * np.arctan2(f[:, 1], f[:, 0]))))
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def level_attitude(df, t):
    """(T, 9) attitude rows in level_frame. The rows are the watch axes' components and rotate with the
    strap like the IMU vectors."""
    R = attitude(df, t)
    return (level_frame(R) @ R).reshape(len(t), 9).astype(np.float32)


def model_input(df, t):
    """(T, 18) pose-model input at watch times t: imu_at, then level_attitude."""
    return np.concatenate([imu_at(df, t), level_attitude(df, t)], 1)


def pair(watch_path, npz_path, offset=None, hand="right"):
    """Align a watch take with its video poses.

    offset: watch time of video t=0, if known; otherwise tap vs forearm peak. Either way it is then refined.
    Returns a dict: t (T,) watch seconds; imu (T, 18) model_input: gyro rad/s, user accel g, gravity g, all
    in the watch frame, then the levelled attitude rows; pose (T, J, 3) metres, pelvis-relative in the heading
    frame; root (T, 3) pelvis in the heading frame; track (T, 3) pelvis in the watch's level_frame (x along the
    mean forearm heading, z up), the frame a watch-only pelvis prediction can be made in; R heading-frame rotation (camera -> heading); Q camera <- watch reference and forearm the
    watch-frame forearm axis (watch_to_camera); valid (T,) pose present; offset, coarse, peak (correlation
    at the chosen offset); tap (watch seconds).
    """
    df = load_sensor_logger(watch_path)
    sy = synth(npz_path, hand)
    if offset is None:
        offset = clock_offset(watch_path)
    coarse = coarse_offset(df, sy) if offset is None else float(offset)
    off, peak = fine_offset(df, sy, coarse)

    d = np.load(npz_path)
    fps = float(d["fps"])
    kp = d["pred_keypoints_3d"][:, JOINTS].astype(float) + d["pred_cam_t"][:, None].astype(float)
    ok = d["found"].astype(bool) & np.isfinite(kp).all((1, 2))
    tv = np.arange(len(kp)) / fps + off

    tw = df.index.to_numpy()
    t0, t1 = max(tw[0], tv[0]), min(tw[-1], tv[-1])
    t = np.arange(np.ceil(t0 * RATE), np.floor(t1 * RATE) + 1) / RATE

    imu = model_input(df, t)

    # Linear interpolation between video frames; a sample is valid only if both neighbours are.
    x = (t - off) * fps
    i0 = np.clip(np.floor(x).astype(int), 0, len(kp) - 2)
    f = np.clip(x - i0, 0, 1)[:, None, None]
    valid = ok[i0] & ok[i0 + 1]
    kp_t = np.where(valid[:, None, None], (1 - f) * np.nan_to_num(kp[i0]) + f * np.nan_to_num(kp[i0 + 1]), np.nan)

    jl, jr = JOINTS.index(L_HIP), JOINTS.index(R_HIP)
    R_w = attitude(df, t)
    fore = kp_t[:, JOINTS.index(WRIST_KP[hand])] - kp_t[:, JOINTS.index(ELBOW_KP[hand])]
    Q, forearm = watch_to_camera(fore[valid], R_w[valid])
    R = heading_frame(kp_t, valid, Q[:, 2])
    pelvis = (kp_t[:, jl] + kp_t[:, jr]) / 2
    pose = (kp_t - pelvis[:, None]) @ R.T
    root = pelvis @ R.T
    track = pelvis @ Q @ level_frame(R_w).T
    return dict(t=t, imu=imu.astype(np.float32), pose=pose.astype(np.float32), root=root.astype(np.float32),
                track=track.astype(np.float32), R=R, Q=Q, forearm=forearm, valid=valid, offset=off, coarse=coarse,
                peak=peak, tap=find_tap(df))


def load(spec):
    """A take from its spec. `watch_dir,sam3d_npz[,offset]` runs pair(); a lone .npz is a pair() result written
    by this module's --save, which needs neither the video's poses nor the watch recording (sample/takes/)."""
    w, *rest = spec.split(",")
    if rest:
        return pair(w, rest[0], offset=float(rest[1]) if len(rest) > 1 else None)
    d = np.load(w)
    return {k: d[k][()] if d[k].ndim == 0 else d[k] for k in d.files}


if __name__ == "__main__":
    save = sys.argv.pop(sys.argv.index("--save") + 1) if "--save" in sys.argv else None
    args = [v for v in sys.argv[1:] if v != "--save"]
    p = pair(args[0], args[1], offset=float(args[2]) if len(args) > 2 else None)
    if save:
        np.savez_compressed(save, **p)
    print(f"coarse {p['coarse']:.3f} s  fine {p['offset']:.3f} s  corr {p['peak']:.3f}  "
          f"{len(p['t'])} samples {p['t'][0]:.2f}-{p['t'][-1]:.2f} s  valid {p['valid'].mean():.0%}")
