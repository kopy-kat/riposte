"""uv run python -m imu.test_synth : analytic check on a synthetic npz."""

import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from imu.load import G
from imu.synth import ELBOW_KP, FLIP, WRIST_JOINT, WRIST_KP, synth

fps, N = 60.0, 240
t = np.arange(N) / fps
r, w_c = 0.3, 2 * np.pi * 1.5  # circle radius (m), rate (rad/s)
centre = np.array([0.0, 0.0, 2.0])
w_rot, axis = 3.0, np.array([1.0, 2.0, 2.0]) / 3  # wrist spin
mount = Rotation.from_euler("xyz", [0.3, -0.5, 0.8]).as_matrix()
grav = np.array([0.0, 1.0, 0.0])

# Circle in the x-z plane (cam frame).
pos = centre + r * np.stack([np.cos(w_c * t), np.zeros(N), np.sin(w_c * t)], 1)
a_world = -w_c**2 * (pos - centre)
R0 = Rotation.from_euler("zyx", [0.4, 0.2, -0.3])
R = Rotation.from_rotvec(w_rot * t[:, None] * axis) * R0  # body spin about a fixed axis; sensor-frame rate is constant
R_cam = R.as_matrix()

j = WRIST_JOINT["right"]
kp = np.zeros((N, 70, 3))
fdir = np.stack([np.cos(2 * t), np.sin(2 * t), np.zeros(N)], 1)  # forearm direction rotates at 2 rad/s
kp[:, WRIST_KP["right"]] = pos
kp[:, ELBOW_KP["right"]] = pos - 0.27 * fdir
cam_t = np.tile(centre, (N, 1))
jc = np.zeros((N, 127, 3))
jc[:, j] = pos - cam_t  # joints/keypoints are stored relative to cam_t
kp -= cam_t[:, None]
grots = np.tile(np.eye(3), (N, 127, 1, 1))
grots[:, j] = FLIP @ R_cam @ FLIP
found = np.ones(N, bool)
s = slice(100, 104)
found[s] = False
jc[s] = kp[s] = grots[s] = cam_t[s] = np.nan

with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "s.npz"
    np.savez(p, fps=fps, found=found, pred_keypoints_3d=kp, pred_cam_t=cam_t, pred_joint_coords=jc,
             pred_global_rots=grots)
    df = synth(p, "right", mount=mount, gravity=grav, smooth_s=0.1)

assert not df.isna().any().any()
core = np.ones(N, bool)
core[:15] = core[N - 15:] = False
core[92:112] = False  # linear interpolation across the gap is not the true path

# The spin is a left-multiplication (fixed world axis), so sensor-frame omega(t) = mount^T R(t)^T axis * w_rot.
gyro_true = np.stack([(Rotation.from_matrix(R_cam[i]).inv().apply(axis) * w_rot) @ mount for i in range(N)])
g_err = np.abs(df[["gx", "gy", "gz"]].values - gyro_true)[core].max()

f_true = np.stack([(R_cam[i] @ mount).T @ (a_world[i] - G * grav) for i in range(N)])
a_err = np.abs(df[["ax", "ay", "az"]].values - f_true)[core].max()

fs_err = np.abs(df["forearm_speed"].values[core] - 2.0).max()
q = df[["qw", "qx", "qy", "qz"]].values
q_err = np.abs(Rotation.from_quat(q, scalar_first=True).as_matrix() - np.stack([R_cam[i] @ mount for i in range(N)]))[core].max()
print(f"gyro max err {g_err:.4f} rad/s (|w|={w_rot})  accel max err {a_err:.4f} m/s² (|a|={w_c**2*r:.1f})  "
      f"forearm err {fs_err:.4f} rad/s  quat err {q_err:.2e}")
assert g_err < 0.02 and a_err < 0.03 * w_c**2 * r and fs_err < 0.02 and q_err < 1e-9
print("ok")
