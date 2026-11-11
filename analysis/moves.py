"""Watch-only error per move, with moves labelled from the video pose itself.

    uv run python -m analysis.moves data/raw/home1/watch,data/processed/home1_video30.sam3d.npz
reads the matching <take>.watchpose.npz that model.train --out wrote.
"""

import sys

import numpy as np

from model.pairs import JOINTS, RATE, pair
from model.train import GROUPS, mpjpe

L_ANK, R_ANK = JOINTS.index(13), JOINTS.index(14)
SH, EL, WR = (JOINTS.index(j) for j in (6, 8, 41))


def label(p):
    """lunge > extension > step > guard, per frame, from the video pose."""
    pose = p["pose"]
    stance = np.linalg.norm((pose[:, L_ANK] - pose[:, R_ANK])[:, [0, 2]], axis=1)
    guard = np.nanmedian(stance)
    u, v = pose[:, SH] - pose[:, EL], pose[:, WR] - pose[:, EL]
    elbow = np.degrees(np.arccos(np.clip((u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)), -1, 1)))
    speed = np.linalg.norm(np.gradient(p["root"][:, [0, 2]], axis=0), axis=1) * RATE
    out = np.full(len(pose), "guard", dtype=object)
    out[speed > 0.3] = "step"
    out[elbow > 150] = "extension"
    out[stance > 1.4 * guard] = "lunge"
    return out


def main():
    w, n, *off = sys.argv[1].split(",")
    p = pair(w, n, offset=float(off[0]) if off else None)
    pred = np.load(n.replace(".sam3d.npz", ".watchpose.npz"))["pose"]
    moves = label(p)
    ok = p["valid"] & np.isfinite(pred).all((1, 2))
    print(f"{'move':<10}{'share':>7}" + "".join(f"{g:>8}" for g in GROUPS))
    for m in ["guard", "step", "extension", "lunge"]:
        v = ok & (moves == m)
        if v.sum():
            print(f"{m:<10}{v.mean():7.0%}" + "".join(f"{mpjpe(pred, p['pose'], v, j):8.0f}" for j in GROUPS.values()))


if __name__ == "__main__":
    main()
