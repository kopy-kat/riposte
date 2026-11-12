"""SAM 3D Body npz -> point caches + scene json for render/scene.py.

    uv run python -m render.prepare data/processed/<take>.sam3d.npz [--hand right] [--watch data/raw/<take>/watch [--offset s]]
    uv run python -m render.prepare --watch-pose data/processed/<take>.watchpose.npz --watch-only <out dir> \
        --watch data/raw/<take>/watch --frames 5401 6300        # for render/studio.py
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from model.pairs import RATE

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
BLADE_MODEL = Path(__file__).resolve().parents[1] / "data/processed/blade.npz"
BODY_MODEL = Path(__file__).resolve().parents[1] / "data/processed/body/body.npz"
# Studio blade: distance along the axis from the hand and radius, a bell guard then the blade, drawn thicker
# than steel so it survives a 1080p frame from 5 m.
BLADE_S, BLADE_R = [0.0, 0.02, 0.04, BLADE_M], [0.06, 0.045, 0.007, 0.004]
TRAIL_S, TRAIL_N, TRAIL_R = 0.5, 60, 0.012  # seconds of tip path, points along it, radius at the tip


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


def bone_quads(joints):
    """(N, J, 3) camera-frame joints -> (N, 4 * len(BONES), 3) quads, widened across the line of sight."""
    a, b = joints[:, [i for i, _ in BONES]], joints[:, [j for _, j in BONES]]
    across = np.cross(b - a, (a + b) / 2)
    across = across / (np.linalg.norm(across, axis=-1, keepdims=True) + 1e-9) * BONE_M
    return np.stack([a + across, a - across, b - across, b + across], 2).reshape(len(joints), -1, 3)


def blade_and_trail(guard, axis, out, fps):
    """Writes blade.pc2 and trail.pc2 for a blade from guard along axis (N, 3), camera frame; returns them."""
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    tip = guard + axis * BLADE_M
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
    return blade, trail


def blade_scene(blade, trail):
    return {"blade_rest": to_blender(blade[0]).tolist(), "trail_rest": to_blender(trail[0]).tolist(),
            "trail_frames": TRAIL_FRAMES}


def skeleton_scene(skel):
    return {"skeleton_rest": to_blender(skel[0]).tolist(),
            "skeleton_faces": [(4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3) for k in range(len(BONES))]}


def tube(points, radii, sides=6):
    """(N, K, 3) polylines -> (N, K * sides, 3) rings of the given radii around them, and the quads joining them."""
    t = np.gradient(points, axis=1)
    t /= np.linalg.norm(t, axis=-1, keepdims=True) + 1e-9
    n1 = np.cross(t, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    flat = np.linalg.norm(n1, axis=-1, keepdims=True) < 1e-3
    n1 = np.where(flat, np.cross(t, np.array([1.0, 0.0, 0.0], dtype=np.float32)), n1)
    n1 /= np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-9
    n2 = np.cross(t, n1)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False, dtype=np.float32)
    ring = np.cos(ang)[:, None] * n1[:, :, None] + np.sin(ang)[:, None] * n2[:, :, None]
    verts = points[:, :, None] + np.asarray(radii, np.float32)[None, :, None, None] * ring
    k = points.shape[1]
    faces = [(i * sides + j, i * sides + (j + 1) % sides, (i + 1) * sides + (j + 1) % sides, (i + 1) * sides + j)
             for i in range(k - 1) for j in range(sides)]
    return verts.reshape(len(points), -1, 3).astype(np.float32), faces


def watch_only(watch_pose, out, watch, first, last, hand="right"):
    """The watch-only fencer in a studio scene for render/studio.py: frames first..last (frame N is pose
    sample N - 1), nothing from the video. The body is render.body's surface on the predicted joints, the
    blade comes from the watch's attitude and the BLADE_MODEL fitted by imu.blade_video.

    The pose is pelvis-relative in the body frame (right, up, back), so the pelvis stays over one spot; each
    frame is lowered until the lowest point of the body rests on the floor.
    """
    from scipy.interpolate import CubicSpline
    from scipy.signal import savgol_filter

    from imu.blade_video import watch_blade
    from render.body import dress

    w = np.load(watch_pose)
    lo = max(first - 1 - int(TRAIL_S * RATE) - 8, 0)
    full = fill_gaps(w["pose"].astype(np.float32))
    world = lambda p: np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)  # (right, up, back) -> Blender's z up
    model = np.load(BODY_MODEL)
    verts, knuckle = dress(model, full[lo:last])
    verts, knuckle = world(verts), world(knuckle)
    ground = savgol_filter(verts[..., 2].min(1), 9, 2)
    verts[..., 2] -= ground[:, None]
    knuckle[:, 2] -= ground

    axis = world(watch_blade(np.load(BLADE_MODEL), watch, w["t"], full, hand).astype(np.float32)[lo:last])
    axis = savgol_filter(axis, 5, 2, axis=0)
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    s, r = np.array(BLADE_S, np.float32), BLADE_R
    blade, blade_faces = tube(knuckle[:, None] + s[None, :, None] * axis[:, None], r)

    # The trail follows a smoothed tip, sampled between frames: at 30 Hz a fast point leaves a polygon.
    tip = savgol_filter(knuckle + BLADE_M * axis, 7, 2, axis=0)
    path = CubicSpline(np.arange(len(tip)), tip)
    k = first - 1 - lo
    ages = np.linspace(0, TRAIL_S * RATE, TRAIL_N, dtype=np.float32)
    hist = path(np.clip(np.arange(k, len(tip))[:, None] - ages[None], 0, None)).astype(np.float32)
    trail, trail_faces = tube(hist, np.linspace(TRAIL_R, 0.002, TRAIL_N))

    fps = float(RATE)
    write_pc2(out / "body.pc2", verts[k:], fps)
    write_pc2(out / "blade.pc2", blade[k:], fps)
    write_pc2(out / "trail.pc2", trail, fps)
    from model.pairs import JOINTS

    feet = world(full[first - 1:last])
    reach = (feet[:, JOINTS.index(14)] - feet[:, JOINTS.index(13)])[:, :2].mean(0)  # rear ankle to front: the piste
    scene = {"fps": fps, "first": first, "frames": last - first + 1,
             "facing_deg": float(np.degrees(np.arctan2(reach[1], reach[0]))),
             "body_rest": verts[k].tolist(), "body_faces": model["faces"].astype(int).tolist(),
             "blade_rest": blade[k].tolist(), "blade_faces": blade_faces,
             "trail_rest": trail[0].tolist(), "trail_faces": trail_faces,
             "trail_age": np.repeat(ages / ages[-1], 6).tolist()}
    (out / "scene.json").write_text(json.dumps(scene))
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", help="not needed with --watch-only")
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--watch", default=None, help="Sensor Logger dir: blade orientation from the watch attitude")
    ap.add_argument("--watch-pose", default=None, help="model.train --out predictions: draw them as a skeleton")
    ap.add_argument("--frames", type=int, nargs=2, metavar=("FIRST", "LAST"), help="with --watch-only: the frames to prepare")
    ap.add_argument("--video", default=None, help="with --watch: fit the watch-to-blade direction to the blade seen in this video")
    ap.add_argument("--offset", type=float, default=None, help="watch time of video t=0")
    ap.add_argument("--watch-only", default=None, metavar="OUT_DIR",
                    help="with --watch-pose, --watch and --frames: the watch-only fencer for render/studio.py, into this dir")
    args = ap.parse_args()
    if args.watch_only:
        out = Path(args.watch_only)
        out.mkdir(parents=True, exist_ok=True)
        return watch_only(args.watch_pose, out, args.watch, *args.frames, args.hand)

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

        b = None
        if args.video:
            from imu.blade_video import fit

            b = fit(args.watch, args.npz, args.video, args.hand, offset=args.offset)
        axis = watch_blade_axis(args.watch, args.npz, args.hand, blade_in_watch=b, offset=args.offset)["axis"].astype(np.float32)
    grip = knuckle
    guard = grip + axis * GRIP_M * 0.3

    fps = float(d["fps"])
    write_pc2(out / "body.pc2", to_blender(verts), fps)

    blade, trail = blade_and_trail(guard, axis, out, fps)
    n = len(guard)

    if args.watch_pose:
        w = np.load(args.watch_pose)
        tw = np.arange(n) / fps + float(w["offset"])
        flat = w["pose"].reshape(len(w["t"]), -1)
        flat = np.stack([np.interp(tw, w["t"], flat[:, k]) for k in range(flat.shape[1])], 1)
        # The watch gives the pose about the pelvis; where the pelvis is comes from the video.
        joints = flat.reshape(n, -1, 3) @ w["R"]
        pelvis = (kps[:, 9] + kps[:, 10]) / 2
        joints = fill_gaps(joints.astype(np.float32)) + pelvis[:, None]
        skel = bone_quads(joints)
        write_pc2(out / "skeleton.pc2", to_blender(skel), fps)

    K = d["cam_int"].reshape(-1, 3, 3)[0]
    scene = {
        "fps": fps,
        "frames": n,
        "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "body_faces": d["faces"].astype(int).tolist(),
        "body_rest": to_blender(verts[0]).tolist(),
        **blade_scene(blade, trail),
    }
    if args.watch_pose:
        scene.update(skeleton_scene(skel))
    (out / "scene.json").write_text(json.dumps(scene))
    print(out)


if __name__ == "__main__":
    main()
