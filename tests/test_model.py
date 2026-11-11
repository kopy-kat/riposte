"""uv run python -m tests.test_model : full pairs -> train path on a synthetic fencer with a known offset."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from imu.load import G
from imu.synth import FLIP, WRIST_JOINT
from model.pairs import JOINTS, pair
from model.train import report, run_fold

OFFSET = -6.213  # watch time of video t=0: the watch starts 6.2 s after the camera
FPS, WATCH_HZ, VIDEO_S, WATCH_S = 59.94, 100.0, 40.0, 32.0
TAP_V = 8.5  # forearm flick, video seconds; the watch feels the impact 0.1 s later
MOUNT = Rotation.from_euler("xyz", [0.4, -0.3, 1.1])
rng = np.random.default_rng(0)
THRUSTS = np.sort(rng.uniform(13, 39, 40))

# Body frame: x right, y up, z back (the fencer faces -z), pelvis at the origin.
BASE = {0: (0, .62, -.08), 1: (-.03, .65, -.06), 2: (.03, .65, -.06), 3: (-.07, .63, 0), 4: (.07, .63, 0),
        5: (-.18, .5, 0), 6: (.18, .5, 0), 7: (-.25, .25, 0), 9: (-.1, 0, 0), 10: (.1, 0, 0),
        11: (-.12, -.45, .05), 13: (-.14, -.88, .1), 15: (-.14, -.92, .0), 16: (-.12, -.92, .02), 17: (-.15, -.92, .15),
        62: (-.3, 0, 0), 63: (-.25, .27, .03), 65: (-.25, .25, -.02), 67: (-.2, .52, 0), 68: (.2, .52, 0), 69: (0, .55, 0)}


def motion(t):
    """Body-frame keypoints (N, 70, 3), pelvis trajectory (N, 3) and wrist rotation for times t (video s)."""
    t = np.asarray(t, float)
    on = 1 / (1 + np.exp(-(t - 12) / 0.3))  # smooth: a step would put an accel impulse on the watch
    ext = np.tanh(sum(np.exp(-((t - c) / 0.3) ** 2) for c in THRUSTS)) + 0.1 * np.sin(0.7 * t) * on
    lunge = np.tanh(sum(np.exp(-((t - c - 0.15) / 0.35) ** 2) for c in THRUSTS[::3]))
    a1 = 0.7 + 0.9 * ext
    a2 = a1 + 0.9 * (1 - ext) + 1.2 * np.exp(-((t - TAP_V) / 0.08) ** 2)
    roll = 0.8 * np.sin(1.3 * t) * on + 0.5 * ext

    kp = np.zeros((len(t), 70, 3))
    for j, p in BASE.items():
        kp[:, j] = p
    sh = np.array([.18, .5, 0])
    one = np.ones_like(t)
    d1 = np.stack([0 * one, -np.cos(a1), -np.sin(a1)], 1)
    d2 = np.stack([0.1 * one, -np.cos(a2), -np.sin(a2)], 1)
    d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
    kp[:, 8] = sh + 0.3 * d1
    kp[:, 41] = kp[:, 8] + 0.27 * d2
    kp[:, 64] = kp[:, 66] = kp[:, 8] + [0, 0, .02]
    kp[:, 12] = np.stack([.12 * one, -.45 + .1 * lunge, -.05 - .35 * lunge], 1)
    kp[:, 14] = np.stack([.14 * one, -.88 + .05 * lunge, -.1 - .7 * lunge], 1)
    for j, o in zip((18, 19, 20), ((0, -.04, -.1), (.02, -.04, -.08), (-.01, -.04, .05))):
        kp[:, j] = kp[:, 14] + o
    root = np.stack([0 * one, 1.0 - 0.12 * lunge, -0.4 * lunge + 0.3 * np.sin(0.3 * t)], 1)

    e2 = np.cross(d2, [0, 1.0, 0])
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    e3 = np.cross(d2, e2)
    R0 = np.stack([d2, e2, e3], axis=2)  # columns: forearm axis and two perpendiculars
    wrist_rot = Rotation.from_matrix(R0) * Rotation.from_rotvec(roll[:, None] * [1.0, 0, 0])
    return kp, root, wrist_rot


def cam_from_body(yaw):
    side_on = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], float).T  # body up -> cam -y, forward -> cam +x
    return Rotation.from_euler("y", yaw).as_matrix() @ side_on


def write_npz(path, yaw):
    t = np.arange(int(VIDEO_S * FPS)) / FPS
    kp_b, root_b, wr = motion(t)
    kp_b = kp_b + np.random.default_rng(1).normal(0, 0.005, kp_b.shape)  # pose-estimate jitter
    R = cam_from_body(yaw)
    T = np.array([0.3, 0.2, 4.0])
    kp_c = (kp_b + root_b[:, None]) @ R.T + T
    cam_t = root_b @ R.T + T
    jc = np.zeros((len(t), 127, 3))
    jc[:, WRIST_JOINT["right"]] = kp_c[:, 41] - cam_t
    grots = np.tile(np.eye(3), (len(t), 127, 1, 1))
    grots[:, WRIST_JOINT["right"]] = FLIP @ (R @ wr.as_matrix()) @ FLIP
    found = np.ones(len(t), bool)
    found[1500:1510] = False
    kp = kp_c - cam_t[:, None]
    for a in (kp, jc, grots, cam_t):
        a[~found] = np.nan
    np.savez(path, fps=FPS, found=found, pred_keypoints_3d=kp, pred_cam_t=cam_t, pred_joint_coords=jc,
             pred_global_rots=grots)
    return kp_b - (kp_b[:, 9:10] + kp_b[:, 10:11]) / 2


def write_watch(d, yaw=0.3):
    d.mkdir()
    tw = 0.05 + np.arange(int(WATCH_S * WATCH_HZ)) / WATCH_HZ
    tv = tw - OFFSET
    R = Rotation.from_matrix(cam_from_body(yaw))
    h = 1e-3

    def sensor(tt):
        kp, root, wr = motion(tt)
        return R * wr * MOUNT, R.apply(kp[:, 41] + root)

    s0, p0 = sensor(tv)
    sm, pm = sensor(tv - h)
    sp, pp = sensor(tv + h)
    gyr = (sm.inv() * sp).as_rotvec() / (2 * h)
    a_cam = (pp - 2 * p0 + pm) / h**2
    acc = s0.inv().apply(a_cam) / G
    grav = s0.inv().apply([0, 1.0, 0])  # camera +y is down
    n = np.random.default_rng(2)
    gyr += n.normal(0, 0.02, gyr.shape)
    acc += n.normal(0, 0.01, acc.shape)
    acc[np.argmin(abs(tw - (TAP_V + 0.1 + OFFSET)))] += [0.3, 3.0, 0.5]
    q = s0.as_quat(scalar_first=True)
    cols = {"time": (1.7e9 + tw) * 1e9, "seconds_elapsed": tw}
    for i, a in enumerate("XYZ"):
        cols[f"rotationRate{a}"], cols[f"gravity{a}"], cols[f"acceleration{a}"] = gyr[:, i], grav[:, i], acc[:, i]
    for i, a in enumerate("WXYZ"):
        cols[f"quaternion{a}"] = q[:, i]
    pd.DataFrame(cols).to_csv(d / "WristMotion.csv", index=False)
    pd.DataFrame({"device name": ["fake"], "standardisation": [False]}).to_csv(d / "Metadata.csv", index=False)


def test_all():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_watch(tmp / "watch")
        truth = write_npz(tmp / "a.npz", yaw=0.3)
        write_npz(tmp / "b.npz", yaw=-0.9)
        pa = pair(tmp / "watch", tmp / "a.npz")
        pb = pair(tmp / "watch", tmp / "b.npz", offset=OFFSET + 0.5)
    for p in (pa, pb):
        print(f"coarse {p['coarse'] - OFFSET:+.4f} s  fine {p['offset'] - OFFSET:+.4f} s from truth, corr {p['peak']:.3f}")
        assert abs(p["offset"] - OFFSET) < 0.01

    # Camera placement must not reach the targets.
    v = pa["valid"] & pb["valid"]
    d = np.abs(pa["pose"][v] - pb["pose"][v]).max()
    print(f"targets, two camera placements: max diff {1000 * d:.2f} mm")
    assert d < 0.005  # only the two sub-ms offset estimates differ
    # Targets equal the body-frame skeleton (x right, y up, z back), pelvis-relative.
    i = np.round((pa["t"] - pa["offset"]) * FPS).astype(int)
    err = np.linalg.norm(pa["pose"][pa["valid"]] - truth[i[pa["valid"]]][:, JOINTS], axis=-1)
    print(f"target vs body-frame truth: median {1000 * np.median(err):.1f} mm")
    assert np.median(err) < 0.02
    assert pa["valid"].mean() > 0.95 and not np.isnan(pa["imu"]).any()

    r = run_fold([pa], folds=5, k=4, epochs=60, device="cpu", log=None)
    report([r])
    assert r["model"]["arm"] < r["mean pose"]["arm"]


if __name__ == "__main__":
    test_all()
    print("ok")
