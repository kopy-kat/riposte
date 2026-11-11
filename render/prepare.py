"""SAM 3D Body npz -> point caches + scene json for render/scene.py.

    uv run render/prepare.py data/processed/<take>.sam3d.npz [--hand right]
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

# MHR70 keypoint indices (sam_3d_body/metadata/mhr70.py)
KP = {"right_elbow": 8, "left_elbow": 7, "right_wrist": 41, "left_wrist": 62,
      "right_middle_knuckle": 32, "left_middle_knuckle": 53}

BLADE_M = 0.90
GRIP_M = 0.10
TRAIL_FRAMES = 12


def to_blender(p):
    # OpenCV camera (x right, y down, z forward) -> Blender camera at origin (x right, y up, looks down -z)
    return p * np.array([1.0, -1.0, -1.0], dtype=np.float32)


def write_pc2(path, frames, fps):
    frames = np.ascontiguousarray(frames, dtype="<f4")
    n_frames, n_points, _ = frames.shape
    with open(path, "wb") as f:
        f.write(b"POINTCACHE2\0" + struct.pack("<iiffi", 1, n_points, 0.0, 1.0, n_frames))
        f.write(frames.tobytes())


def fill_gaps(x):
    # Forward/backward fill frames where no person was detected.
    x = x.copy()
    bad = np.isnan(x).any(axis=tuple(range(1, x.ndim)))
    idx = np.where(~bad, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    x = x[idx]
    first = np.argmax(~bad)
    x[:first] = x[first]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    args = ap.parse_args()

    d = np.load(args.npz)
    out = Path(args.npz).with_suffix("").with_suffix("")
    out.mkdir(exist_ok=True)

    cam_t = fill_gaps(d["pred_cam_t"].astype(np.float32))
    verts = fill_gaps(d["pred_vertices"].astype(np.float32)) + cam_t[:, None]
    kps = fill_gaps(d["pred_keypoints_3d"].astype(np.float32)) + cam_t[:, None]

    elbow, wrist, knuckle = (kps[:, KP[f"{args.hand}_{n}"]] for n in ("elbow", "wrist", "middle_knuckle"))
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    # Blade axis ~ between forearm and hand direction; replaced by the watch's orientation later.
    axis = unit(unit(wrist - elbow) + unit(knuckle - wrist))
    grip = knuckle
    guard = grip + axis * GRIP_M * 0.3
    tip = guard + axis * BLADE_M

    fps = float(d["fps"])
    write_pc2(out / "body.pc2", to_blender(verts), fps)

    # Blade as a 2-vertex-wide ribbon is invisible edge-on, so use a thin quad pair (cross shape).
    side = unit(np.cross(axis, np.array([0, 0, 1], dtype=np.float32))) * 0.004
    up = unit(np.cross(axis, side)) * 0.004
    blade = np.stack([guard + side, guard - side, tip + side, tip - side,
                      guard + up, guard - up, tip + up, tip - up], axis=1)
    write_pc2(out / "blade.pc2", to_blender(blade), fps)

    # Tip trail: ribbon over the last TRAIL_FRAMES tip positions, oldest first.
    n = len(tip)
    hist = np.stack([tip[np.clip(np.arange(n) - k, 0, None)] for k in range(TRAIL_FRAMES - 1, -1, -1)], axis=1)
    width = np.linspace(0.001, 0.012, TRAIL_FRAMES, dtype=np.float32)[None, :, None]
    trail = np.concatenate([hist + up[:, None] / 0.004 * width, hist - up[:, None] / 0.004 * width], axis=1)
    write_pc2(out / "trail.pc2", to_blender(trail), fps)

    K = d["cam_int"].reshape(-1, 3, 3)[0]
    scene = {
        "fps": fps,
        "frames": n,
        "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "body_faces": d["faces"].astype(int).tolist(),
        "body_rest": to_blender(verts[0]).tolist(),
        "blade_rest": to_blender(blade[0]).tolist(),
        "trail_rest": to_blender(trail[0]).tolist(),
        "trail_frames": TRAIL_FRAMES,
    }
    (out / "scene.json").write_text(json.dumps(scene))
    print(out)


if __name__ == "__main__":
    main()
