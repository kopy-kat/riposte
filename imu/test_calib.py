"""uv run python -m imu.test_calib : recover a known mount and reference frame from noisy synthetic rotations."""

import numpy as np
from scipy.spatial.transform import Rotation

from imu.calib import DOWN_REF, _angle, fit_direction, solve

rng = np.random.default_rng(0)
N, fps = 1500, 30.0
M = Rotation.from_euler("xyz", [20, -110, 35], degrees=True).as_matrix()
# Reference frame: yaw arbitrary, plus a 4 deg camera tilt so the gravity prior is slightly wrong.
A = (Rotation.from_euler("x", 4, degrees=True) * Rotation.from_euler("x", 90, degrees=True)
     * Rotation.from_euler("z", 137, degrees=True)).as_matrix()

w = np.cumsum(rng.normal(0, 0.15, (N, 3)), 0)  # wandering angular velocity, rad/s
w = 3 * w / np.abs(w).max()
R_w = Rotation.identity()
seq = []
for k in range(N):
    R_w = R_w * Rotation.from_rotvec(w[k] / fps)
    seq.append(R_w)
R_w = Rotation.concatenate(seq).as_matrix()

noise = Rotation.from_rotvec(rng.normal(0, np.radians(6) / np.sqrt(3), (N, 3)))
R_c = (noise * Rotation.from_matrix(A @ R_w @ M.T)).as_matrix()
bad = rng.random(N) < 0.15  # SAM3D-style gross failures: arbitrary rotations
R_c[bad] = Rotation.random(bad.sum(), random_state=1).as_matrix()

for g in (0.0, 1.0):
    M_, A_, res, _ = solve(R_c, R_w, gravity_weight=g)
    eM, eA = _angle(M_.T @ M), _angle(A_.T @ A)
    print(f"gravity_weight {g}: M error {eM:.2f} deg  A error {eA:.2f} deg  "
          f"inlier residual median {np.median(res[~bad]):.1f} deg  p90 {np.percentile(res[~bad], 90):.1f} deg")
    assert max(eM, eA) < (1 if g == 0 else 5), (eM, eA)  # the prior pulls A toward the 4 deg tilt

b = np.array([0.2, 0.95, -0.1])
b /= np.linalg.norm(b)
h = np.einsum("ij,tjk,k->ti", A, R_w, b) + rng.normal(0, 0.1, (N, 3))
h /= np.linalg.norm(h, axis=1, keepdims=True)
b_ = fit_direction(R_w[:90], A, h[:90])
eb = np.degrees(np.arccos(b_ @ b))
print(f"blade direction from a 3 s window: {eb:.2f} deg")
assert eb < 5
assert np.allclose(np.einsum("tji,j->ti", R_w, DOWN_REF)[0], R_w[0].T @ DOWN_REF)
print("ok")
