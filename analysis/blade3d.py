"""The blade's 3D direction from two cameras, against the watch's.

    uv run python -m analysis.blade3d <view A targets.npz> <view B targets.npz> <view A watchpose.npz>
Each camera sees the blade as an image line from the grip: a plane through its lens. Two cameras at an angle
give two planes, and the blade lies along their intersection, including the part pointing toward or away
from either camera that one view cannot score. The planes only differ if the cameras differ in height
relative to the blade: two level cameras at hand height both see a level blade in the horizontal plane.
Both views' targets are model.targets files on the same watch clock; their body frames differ by a turn
about vertical, found from the two video poses.
"""

import sys

import numpy as np

from imu.blade_video import image_angle, project
from imu.synth import ELBOW_KP, WRIST_KP
from model.pairs import JOINTS, RATE

MIN_CROSS_DEG = 25.0  # planes closer to parallel than this don't fix a direction


def sight_plane(tg, i):
    """Unit normals, in the view's body frame, of the planes through the lens and the blade's image line."""
    K, g = tg["K"].astype(float), tg["grip"][i]
    px = project(K, g) + 100 * np.stack([np.cos(tg["blade_angle"][i]), np.sin(tg["blade_angle"][i])], 1)
    ray = np.c_[px, np.ones(len(px))] @ np.linalg.inv(K).T
    n = np.cross(g, ray)
    return (n / np.linalg.norm(n, axis=1, keepdims=True)) @ tg["R"].T


def yaw_between(pa, pb):
    """Rotation about vertical (y) taking view B's body frame onto view A's, from the same poses seen by both."""
    za, zb = pa[..., 0] + 1j * pa[..., 2], pb[..., 0] + 1j * pb[..., 2]
    psi = np.angle(np.nansum(za * np.conj(zb)))
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def triangulate(a, b):
    """Common frames (indices into a and b) with the blade seen by both, and its direction in A's body frame."""
    _, ia, ib = np.intersect1d(np.round(a["t"] * RATE).astype(int), np.round(b["t"] * RATE).astype(int), return_indices=True)
    both = a["valid"][ia] & b["valid"][ib]
    Y = yaw_between(a["pose"][ia][both], b["pose"][ib][both])
    gap = 1000 * np.nanmean(np.linalg.norm(a["pose"][ia][both] - b["pose"][ib][both] @ Y.T, axis=-1))
    m = a["blade_ok"][ia] & b["blade_ok"][ib]
    ia, ib = ia[m], ib[m]
    d = np.cross(sight_plane(a, ia), sight_plane(b, ib) @ Y.T)
    cross = np.linalg.norm(d, axis=1)
    keep = cross > np.sin(np.radians(MIN_CROSS_DEG))
    crossing = np.degrees(np.arcsin(np.clip(np.median(cross), 0, 1))) if len(cross) else np.nan
    ia, ib, d = ia[keep], ib[keep], d[keep] / cross[keep, None]
    # Of the line's two directions, the one view A sees pointing the way the blade does.
    flip = np.cos(image_angle(a["K"], a["grip"][ia], d @ a["R"]) - a["blade_angle"][ia]) < 0
    d[flip] *= -1
    apart = np.degrees(np.arccos(np.clip(a["R"][:, 2] @ (Y @ b["R"][:, 2]), -1, 1)))
    return ia, ib, d, dict(pose_gap_mm=gap, views_apart_deg=apart, seen_by_both=int(m.sum()), crossing_deg=crossing)


def angles(u, v):
    """3D angle between unit rows, and its split into heading (about vertical) and elevation, degrees."""
    full = np.degrees(np.arccos(np.clip((u * v).sum(1), -1, 1)))
    az = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(u[:, 2], u[:, 0]) - np.arctan2(v[:, 2], v[:, 0]))))))
    el = np.abs(np.degrees(np.arcsin(np.clip(u[:, 1], -1, 1)) - np.arcsin(np.clip(v[:, 1], -1, 1))))
    return full, az, el


def main():
    a, b, wp = (np.load(p) for p in sys.argv[1:4])
    ia, _, video, info = triangulate(a, b)
    print(f"cameras {info['views_apart_deg']:.0f} deg apart; their video poses agree to {info['pose_gap_mm']:.0f} mm per joint; "
          f"blade seen by both in {info['seen_by_both']} frames, their sight planes cross at {info['crossing_deg']:.0f} deg "
          f"(median), direction fixed in the {len(ia)} above {MIN_CROSS_DEG:.0f}")
    if not len(ia):
        return
    fore = wp["pose"][:, JOINTS.index(WRIST_KP["right"])] - wp["pose"][:, JOINTS.index(ELBOW_KP["right"])]
    for name, axis in (("watch", wp["blade"]), ("along the forearm", fore / np.linalg.norm(fore, axis=1, keepdims=True))):
        full, az, el = angles(axis[ia], video)
        print(f"blade 3D direction error deg, {name}: {np.median(full):.1f} median, {np.percentile(full, 90):.1f} p90 "
              f"(heading {np.median(az):.1f}, elevation {np.median(el):.1f})")


if __name__ == "__main__":
    main()
