import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/ is not an installed package

from analysis.blade3d import angles, triangulate
from analysis.faults import BLOCKS, compare, separators
from imu.blade_video import image_angle
from model.pairs import FOREARM_W, JOINTS, RATE


def test_two_views_recover_the_blade():
    """A known blade seen by two cameras 45 deg apart, whose body frames differ by an unknown turn."""
    rng = np.random.default_rng(0)
    T = 400
    t = np.arange(T) / RATE
    K = np.array([[1400.0, 0, 960], [0, 1400.0, 540], [0, 0, 1]])
    az, el = rng.uniform(-0.6, 0.6, T) + np.pi, rng.uniform(-0.4, 0.4, T)
    blade = np.stack([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)], 1)  # body frame, y up
    pose = rng.normal(0, 0.3, (T, len(JOINTS), 3))
    grip_body = np.array([-0.5, 0.3, 0.0]) + rng.normal(0, 0.1, (T, 3))

    def view(az, up_deg, frame_yaw, dist):
        """targets for a camera `dist` from the fencer at azimuth az, looking down by up_deg, whose body frame
        is turned by frame_yaw."""
        be = np.radians(up_deg)
        f = -np.array([np.cos(be) * np.sin(az), np.sin(be), np.cos(be) * np.cos(az)])
        r = np.cross(f, [0, 1.0, 0])
        r /= np.linalg.norm(r)
        C = np.stack([r, np.cross(f, r), f])  # camera <- body: x right, y down, z forward
        c, s = np.cos(frame_yaw), np.sin(frame_yaw)
        F = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])  # this view's body frame <- the true one
        grip = (grip_body + f * dist) @ C.T
        return dict(t=t, pose=pose @ F.T, valid=np.ones(T, bool), blade_ok=np.ones(T, bool), K=K, R=F @ C.T, grip=grip,
                    blade_angle=image_angle(K, grip, blade @ C.T) + rng.normal(0, np.radians(1), T))

    # Level cameras at the blade's height cannot do this: both sight planes are the horizontal one. The
    # second camera looks up from the floor.
    a, b = view(0.0, 0.0, 0.0, 3.0), view(np.radians(40), -25.0, np.radians(7), 3.5)
    ia, _, video, info = triangulate(a, b)
    assert abs(info["views_apart_deg"] - np.degrees(np.arccos(np.cos(np.radians(25)) * np.cos(np.radians(40))))) < 1
    assert info["pose_gap_mm"] < 1
    full, _, _ = angles(blade[ia], video)
    assert len(ia) > 0.5 * T and np.median(full) < 4


def test_fault_blocks_from_the_watch_and_the_share_shown():
    rng = np.random.default_rng(0)
    rate, n = 100.0, 6
    t = np.arange(int(40 * n * rate)) / rate
    df = pd.DataFrame({**{f"gyr_{k}": rng.normal(0, 1.0, len(t)) for k in "xyz"},
                       **{f"acc_{k}": rng.normal(0, 0.1, len(t)) for k in "xyz"},
                       "grav_x": 0.0, "grav_y": 0.0, "grav_z": -1.0, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}, index=t)
    df.attrs["rate_hz"] = rate
    starts = 40.0 * np.arange(n) + 2
    for s in starts:  # a 4 s arm-down hold before each block
        m = (t >= s) & (t < s + 4)
        df.loc[m, ["gyr_x", "gyr_y", "gyr_z"]] = 0.02
        df.loc[m, ["grav_x", "grav_y", "grav_z"]] = FOREARM_W
    found = separators(df)
    assert len(found) == n and np.allclose([e for _, e in found], starts + 4, atol=0.05)

    # Eight lunges per block; the short-lunge block is 30% shallower in the video and 15% in the watch pose.
    tt = np.arange(int(40 * n * RATE)) / RATE
    L, R_ = JOINTS.index(13), JOINTS.index(14)

    def poses(short):
        p = np.zeros((len(tt), len(JOINTS), 3), np.float32)
        p[:, JOINTS.index(5), 1] = p[:, JOINTS.index(6), 1] = 0.5
        p[:, JOINTS.index(8)] = [-0.3, 0.3, 0]
        p[:, JOINTS.index(41)] = [-0.6, 0.35, 0]
        width = np.full(len(tt), 0.5)
        for k, name in enumerate(BLOCKS):
            for r in range(8):
                c = starts[k] + 8 + 3.5 * r
                width += 0.6 * (short if name == "short lunge" else 1.0) * np.exp(-((tt - c) / 0.25) ** 2)
        p[:, L, 0], p[:, R_, 0] = width / 2, -width / 2
        return p

    tg = dict(t=tt, pose=poses(0.7), valid=np.ones(len(tt), bool), root=np.zeros((len(tt), 3), np.float32),
              blade_ok=np.ones(len(tt), bool), blade_angle=np.full(len(tt), np.pi), K=np.eye(3), R=np.eye(3),
              grip=np.tile([0.0, 0.0, 3.0], (len(tt), 1)))
    wp = dict(pose=poses(0.85), blade=np.tile([-1.0, 0.0, 0.0], (len(tt), 1)))
    res = compare(df, tg, wp, [e for _, e in found])
    assert all(r["n"] == 8 for r in res.values())
    dv = np.median(res["short lunge"]["video"]["depth x guard"]) - np.median(res["normal"]["video"]["depth x guard"])
    dw = np.median(res["short lunge"]["watch"]["depth x guard"]) - np.median(res["normal"]["watch"]["depth x guard"])
    assert dv < 0 and abs(dw / dv - 0.5) < 0.05
