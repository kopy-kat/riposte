"""SAM 3D Body npz -> point caches + scene json for render/scene.py.

    uv run python -m render.prepare data/processed/<take>.sam3d.npz [--hand right] [--watch data/raw/<take>/watch [--offset s]]
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
# Bones over model.pairs.JOINTS indices (MHR70 0-20, then right wrist 21, left wrist 22).
BONES = [(0, 5), (0, 6), (5, 6), (5, 7), (7, 22), (6, 8), (8, 21), (5, 9), (6, 10), (9, 10), (9, 11), (11, 13),
         (10, 12), (12, 14), (13, 15), (13, 17), (14, 18), (14, 20)]
BONE_M = 0.012


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
    ap.add_argument("--watch", default=None, help="Sensor Logger dir: blade orientation from the watch attitude")
    ap.add_argument("--watch-pose", default=None, help="model.train --out predictions: draw them as a skeleton")
    ap.add_argument("--offset", type=float, default=None, help="watch time of video t=0")
    args = ap.parse_args()

    d = np.load(args.npz)
    out = Path(args.npz).with_suffix("").with_suffix("")
    out.mkdir(exist_ok=True)

    cam_t = fill_gaps(d["pred_cam_t"].astype(np.float32))
    verts = fill_gaps(d["pred_vertices"].astype(np.float32)) + cam_t[:, None]
    kps = fill_gaps(d["pred_keypoints_3d"].astype(np.float32)) + cam_t[:, None]

    elbow, wrist, knuckle = (kps[:, KP[f"{args.hand}_{n}"]] for n in ("elbow", "wrist", "middle_knuckle"))
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    # Blade axis ~ between forearm and hand direction, unless the watch measures it.
    axis = unit(unit(wrist - elbow) + unit(knuckle - wrist))
    if args.watch:
        from imu.calib import watch_blade_axis

        axis = watch_blade_axis(args.watch, args.npz, args.hand, offset=args.offset)["axis"].astype(np.float32)
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
    width = np.linspace(0.002, 0.03, TRAIL_FRAMES, dtype=np.float32)[None, :, None]
    # Widen across both the path and the line of sight so the ribbon always faces the camera
    # (the camera sits at the origin); a fixed offset direction goes edge-on and vanishes.
    along = np.gradient(hist, axis=1)
    across = np.cross(along, hist)
    across /= np.linalg.norm(across, axis=-1, keepdims=True) + 1e-9
    trail = np.concatenate([hist + across * width, hist - across * width], axis=1)
    write_pc2(out / "trail.pc2", to_blender(trail), fps)

    if args.watch_pose:
        w = np.load(args.watch_pose)
        tw = np.arange(n) / fps + float(w["offset"])
        flat = w["pose"].reshape(len(w["t"]), -1)
        flat = np.stack([np.interp(tw, w["t"], flat[:, k]) for k in range(flat.shape[1])], 1)
        # The watch gives the pose about the pelvis; where the pelvis is comes from the video.
        joints = flat.reshape(n, -1, 3) @ w["R"]
        pelvis = (kps[:, 9] + kps[:, 10]) / 2
        joints = fill_gaps(joints.astype(np.float32)) + pelvis[:, None]
        a, b = joints[:, [i for i, _ in BONES]], joints[:, [j for _, j in BONES]]
        across = np.cross(b - a, (a + b) / 2)
        across = across / (np.linalg.norm(across, axis=-1, keepdims=True) + 1e-9) * BONE_M
        skel = np.stack([a + across, a - across, b - across, b + across], 2).reshape(n, -1, 3)
        write_pc2(out / "skeleton.pc2", to_blender(skel), fps)

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
    if args.watch_pose:
        scene["skeleton_rest"] = to_blender(skel[0]).tolist()
        scene["skeleton_faces"] = [(4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3) for k in range(len(BONES))]
    (out / "scene.json").write_text(json.dumps(scene))
    print(out)


if __name__ == "__main__":
    main()
