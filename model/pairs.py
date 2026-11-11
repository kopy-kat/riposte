"""Paired (watch IMU, video pose) sequences on one 30 Hz time base, in watch time.

    uv run python -m model.pairs data/raw/office2/watch data/processed/office2_video.sam3d.npz
"""

import sys

import numpy as np
from scipy.signal import butter, savgol_filter, sosfiltfilt

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


def heading_frame(kp_cam, valid):
    """Rotation into a right-handed (right, up, back) body frame, one heading for the whole take.

    Up is camera -y (assumes a roughly level camera, as imu.synth does); lateral is the median
    left->right hip direction flattened onto the floor. One heading per take keeps the pelvis yaw in the
    targets while removing camera placement.
    """
    up = np.array([0.0, -1.0, 0.0])
    hip = np.nanmedian((kp_cam[valid, R_HIP] - kp_cam[valid, L_HIP]), axis=0)
    lat = hip - (hip @ up) * up
    lat /= np.linalg.norm(lat)
    return np.stack([lat, up, np.cross(lat, up)])


def pair(watch_path, npz_path, offset=None, hand="right"):
    """Align a watch take with its video poses.

    offset: watch time of video t=0, if known; otherwise tap vs forearm peak. Either way it is then refined.
    Returns a dict: t (T,) watch seconds; imu (T, 9) gyro rad/s, user accel g, gravity g, all in the watch
    frame; pose (T, J, 3) metres, pelvis-relative in the heading frame; root (T, 3) pelvis in the heading
    frame; R heading-frame rotation (camera -> heading); valid (T,) pose present; offset, coarse, peak (correlation at the chosen offset); tap (watch
    seconds).
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

    rate = df.attrs["rate_hz"]
    raw = df[IMU_COLS].to_numpy()
    raw = sosfiltfilt(butter(4, 0.4 * RATE, fs=rate, output="sos"), raw, axis=0)  # anti-alias before 30 Hz
    imu = np.stack([np.interp(t, tw, raw[:, k]) for k in range(raw.shape[1])], 1)

    # Linear interpolation between video frames; a sample is valid only if both neighbours are.
    x = (t - off) * fps
    i0 = np.clip(np.floor(x).astype(int), 0, len(kp) - 2)
    f = np.clip(x - i0, 0, 1)[:, None, None]
    valid = ok[i0] & ok[i0 + 1]
    kp_t = np.where(valid[:, None, None], (1 - f) * np.nan_to_num(kp[i0]) + f * np.nan_to_num(kp[i0 + 1]), np.nan)

    jl, jr = JOINTS.index(L_HIP), JOINTS.index(R_HIP)
    R = heading_frame(kp_t, valid)
    pelvis = (kp_t[:, jl] + kp_t[:, jr]) / 2
    pose = (kp_t - pelvis[:, None]) @ R.T
    root = pelvis @ R.T
    return dict(t=t, imu=imu.astype(np.float32), pose=pose.astype(np.float32), root=root.astype(np.float32),
                R=R, valid=valid, offset=off, coarse=coarse, peak=peak, tap=find_tap(df))


if __name__ == "__main__":
    p = pair(sys.argv[1], sys.argv[2], offset=float(sys.argv[3]) if len(sys.argv) > 3 else None)
    print(f"coarse {p['coarse']:.3f} s  fine {p['offset']:.3f} s  corr {p['peak']:.3f}  "
          f"{len(p['t'])} samples {p['t'][0]:.2f}-{p['t'][-1]:.2f} s  valid {p['valid'].mean():.0%}")
