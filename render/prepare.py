"""SAM 3D Body npz -> point caches + scene json for render/scene.py.

    uv run python -m render.prepare data/processed/<take>.sam3d.npz [--hand right] [--watch data/raw/<take>/watch [--offset s]]
    uv run python -m render.prepare --watch-pose data/processed/<take>.watchpose.npz --watch-only <out dir> \
        --watch data/raw/<take>/watch --frames 5672 6271        # the fencer for render/studio.py
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
BLADE_MODEL = Path(__file__).resolve().parents[1] / "data/processed/body/blade_grip.npz"
BODY_MODEL = Path(__file__).resolve().parents[1] / "data/processed/body/body.npz"
# Studio épée: distance along the axis from the knuckles and radius. The grip in the hand, the bell guard (a
# 13.5 cm bowl, convex toward the point), then the blade tapering to the point.
BLADE_S = [-0.11, -0.10, 0.0, 0.012, 0.016, 0.026, 0.038, 0.048, 0.054, 0.058, 0.30, BLADE_M, BLADE_M + 0.004]
BLADE_R = [0.0, 0.011, 0.011, 0.012, 0.066, 0.064, 0.052, 0.034, 0.016, 0.006, 0.0038, 0.0024, 0.0]
TRAIL_S, TRAIL_N, TRAIL_R = 0.45, 60, 0.006  # seconds of tip path, points along it, radius at the tip

# Fencing kit on the body mesh, for render/studio.py: linear RGB, roughness, metallic per garment. Where one
# garment ends and the next begins is a field from kit(), so the shader draws each edge as a clean line.
KIT = {"jacket": ((0.78, 0.78, 0.76), 0.9, 0.0), "breeches": ((0.70, 0.70, 0.69), 0.94, 0.0),
       "sock": ((0.80, 0.80, 0.79), 0.97, 0.0), "shoe": ((0.16, 0.16, 0.17), 0.55, 0.0),
       "glove": ((0.33, 0.30, 0.26), 0.75, 0.0), "skin": ((0.50, 0.36, 0.29), 0.65, 0.0),
       "mesh": ((0.035, 0.035, 0.04), 0.42, 0.35)}
HEM_M, CUFF_M, SHOE_M, GLOVE_M = 0.11, 0.07, 0.055, 0.10  # jacket below the hip, breeches below the knee, shoe above the ankle, glove up the forearm
# Pose joints (model.pairs.JOINTS positions): COCO-17 without wrists, feet (big toe, small toe, heel), wrists.
NOSE, _, _, L_EAR, R_EAR, L_SH, R_SH, L_EL, R_EL, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = range(15)
L_TOE, L_HEEL, R_TOE, R_HEEL, R_WR, L_WR = 15, 17, 18, 20, 21, 22


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
    """(N, K, 3) polylines -> (N, K * sides, 3) rings of the given radii around them, and the quads joining them.
    The ring's frame is carried along the line instead of rebuilt per point, so the tube neither twists nor
    pinches where the path doubles back."""
    t = np.gradient(points, axis=1)
    norm = np.linalg.norm(t, axis=-1, keepdims=True)
    for i in range(1, points.shape[1]):  # a point where the path stops takes its neighbour's direction
        t[:, i] = np.where(norm[:, i] < 1e-5, t[:, i - 1], t[:, i])
    t /= np.linalg.norm(t, axis=-1, keepdims=True) + 1e-9
    seed = np.where(np.abs(t[:, 0, 2:]) < 0.9, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    n1 = np.empty_like(points)
    prev = seed
    for i in range(points.shape[1]):
        prev = prev - (prev * t[:, i]).sum(-1, keepdims=True) * t[:, i]
        prev = prev / (np.linalg.norm(prev, axis=-1, keepdims=True) + 1e-9)
        n1[:, i] = prev
    n2 = np.cross(t, n1)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False, dtype=np.float32)
    ring = np.cos(ang)[:, None] * n1[:, :, None] + np.sin(ang)[:, None] * n2[:, :, None]
    verts = points[:, :, None] + np.asarray(radii, np.float32)[None, :, None, None] * ring
    k = points.shape[1]
    faces = [(i * sides + j, i * sides + (j + 1) % sides, (i + 1) * sides + (j + 1) % sides, (i + 1) * sides + j)
             for i in range(k - 1) for j in range(sides)]
    return verts.reshape(len(points), -1, 3).astype(np.float32), faces


def kit(v, j, faces):
    """Fencing kit for the body mesh, from one frame with the limbs apart: v (V, 3) vertices and j (30, 3) joints,
    z up. Returns the garment fields per vertex, in metres and positive inside (hem: below the jacket; cuff:
    below the breeches; shoe; glove: the weapon hand's; hand: the bare back hand), how much each vertex is
    cloth, and the mask: the head's vertices, where they sit on a mask-shaped shell, and the neck's push
    outward into a bib."""
    unit = lambda x: x / np.linalg.norm(x)
    mid = lambda a, b: (j[a] + j[b]) / 2
    sh, hip, ear = mid(L_SH, R_SH), mid(L_HIP, R_HIP), mid(L_EAR, R_EAR)
    hand = lambda wr, el: j[wr] + 0.2 * unit(j[wr] - j[el])
    # (from, to, limb radius): each vertex belongs to its nearest limb.
    segs = [(sh, hip, 0.15), (j[L_SH], j[R_SH], 0.10), (j[L_HIP], j[R_HIP], 0.13),
            (j[L_SH], j[L_EL], 0.06), (j[R_SH], j[R_EL], 0.06), (j[L_EL], j[L_WR], 0.05), (j[R_EL], j[R_WR], 0.05),
            (j[L_WR], hand(L_WR, L_EL), 0.05), (j[R_WR], hand(R_WR, R_EL), 0.05),
            (j[L_HIP], j[L_KNEE], 0.085), (j[R_HIP], j[R_KNEE], 0.085), (j[L_KNEE], j[L_ANK], 0.06), (j[R_KNEE], j[R_ANK], 0.06),
            (j[L_ANK], j[L_TOE], 0.05), (j[R_ANK], j[R_TOE], 0.05), (j[L_ANK], j[L_HEEL], 0.05), (j[R_ANK], j[R_HEEL], 0.05),
            (ear + 0.3 * (sh - ear), ear + 0.6 * (ear - sh), 0.11)]
    SPINE, HIPS, L_FORE, R_FORE, L_HAND, R_HAND, L_THIGH, R_THIGH, L_SHIN, R_SHIN, FEET, HEAD = 0, 2, 5, 6, 7, 8, 9, 10, 11, 12, (13, 14, 15, 16), 17
    best, limb, along = np.full(len(v), np.inf), np.zeros(len(v), int), np.zeros(len(v))
    for i, (a, b, r) in enumerate(segs):
        u = np.clip(((v - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
        d = np.linalg.norm(v - (a + u[:, None] * (b - a)), axis=1) / r
        m = d < best
        best[m], limb[m], along[m] = d[m], i, u[m]
    length = lambda i: np.linalg.norm(segs[i][1] - segs[i][0])

    # Distance down the leg from the hip joint. The trunk measures along the nearer thigh so the hem has no
    # step, and calf vertices that fell to the thigh measure along the shin so the cuff has none.
    leg = np.full(len(v), -1.0)
    for thigh, shin, feet in ((L_THIGH, L_SHIN, (13, 15)), (R_THIGH, R_SHIN, (14, 16))):
        (a, b), (_, c) = segs[thigh][:2], segs[shin][:2]
        side = (v - mid(L_HIP, R_HIP)) @ (a - mid(L_HIP, R_HIP)) > 0
        upper = (np.isin(limb, (SPINE, HIPS)) & side) | (limb == thigh)
        leg[upper] = np.minimum((v[upper] - a) @ unit(b - a), length(thigh))
        past = (v - b) @ unit(c - b)
        calf = (limb == thigh) & (past > 0) & (np.linalg.norm(v - b - past[:, None] * unit(c - b), axis=1) < 0.075)
        leg[calf] = length(thigh) + past[calf]
        leg[limb == shin] = length(thigh) + along[limb == shin] * length(shin)
        leg[np.isin(limb, feet)] = length(thigh) + length(shin) + 0.05
    thigh, shin = (length(L_THIGH) + length(R_THIGH)) / 2, (length(L_SHIN) + length(R_SHIN)) / 2
    glove = np.where(limb == R_HAND, 0.1, np.where(limb == R_FORE, GLOVE_M - (1 - along) * length(R_FORE), -1.0))
    bare = np.where(limb == L_HAND, 0.004 + along * 0.2, np.where(limb == L_FORE, -(1 - along) * length(L_FORE) - 0.004, -1.0))
    fields = {"hem": leg - HEM_M, "cuff": leg - thigh - CUFF_M, "shoe": leg - thigh - shin + SHOE_M, "glove": glove, "hand": bare}
    cloth = np.where(fields["shoe"] > 0, 0.25, 1.0)
    cloth[(glove > 0) | (bare > 0) | (limb == HEAD)] = 0.0

    # Head axes: across the ears, toward the nose, up.
    x = unit(j[L_EAR] - j[R_EAR])
    y = unit((j[NOSE] - ear) - ((j[NOSE] - ear) @ x) * x)
    A = np.stack([x, y, np.cross(y, x)])
    c = ear + A.T @ np.array([0.0, 0.015, 0.02])
    q = (v - c) @ A.T
    rad = np.hypot(q[:, 0], q[:, 1])
    near_neck = np.isin(limb, [SPINE, 1, HEAD]) & (rad < 0.2)  # spine, shoulder line, head: not a raised arm
    head = np.flatnonzero(near_neck & (q[:, 2] >= -0.09))
    # The shell: taller than wide, deeper in front of the face than behind the skull.
    h = q[head]
    semi = np.stack([np.full(len(h), 0.105), np.where(h[:, 1] > 0, 0.15, 0.118), np.full(len(h), 0.155)], 1)
    shell = h / np.linalg.norm(h / semi, axis=1, keepdims=True)
    # Positive on the wire mesh over the face, negative on the padded shell; the shader draws the seam at zero.
    front = np.full(len(v), -1.0)
    front[head] = np.minimum(shell[:, 1] + 0.035, shell[:, 2] + 0.115)
    # The bib: the neck widens into a cone from the mask's chin to the collarbones.
    neck = np.flatnonzero(near_neck & (q[:, 2] < -0.09) & (q[:, 2] > -0.24))
    n = q[neck]
    ease = lambda x: x * x * (3 - 2 * x)
    want = (0.088 + 0.45 * (-0.09 - n[:, 2])) * ease(np.clip((n[:, 2] + 0.24) / 0.09, 0, 1))
    push = np.maximum(want / np.maximum(rad[neck], 1e-6), 1.0)
    bib = (n * np.stack([push, push, np.ones(len(n))], 1) - n) @ A
    cloth[head] = 0.0
    # Feather the bib into the jacket: average the push over the mesh's edges, holding the mask's shell.
    from scipy.sparse import coo_matrix

    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    adj = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), (len(v), len(v))).tocsr()
    adj.data[:] = 1.0
    deg = np.asarray(adj.sum(1))
    move = np.zeros((len(v), 3))
    move[head], move[neck] = (shell - h) @ A, bib
    for _ in range(12):
        move = adj @ move / deg
        move[head] = (shell - h) @ A
    neck = np.setdiff1d(np.flatnonzero(np.linalg.norm(move, axis=1) > 2e-4), head)
    return fields, cloth, front, head, (shell - h) @ A, neck, move[neck]


def masked(verts, ref, head, shell, neck, bib):
    """Moves the head's vertices onto the mask and the neck's into the bib in every frame: the offsets found on
    the frame `ref` by kit(), turned with the head (a rigid fit of its vertices to that frame's)."""
    a = verts[ref, head] - verts[ref, head].mean(0)
    b = verts[:, head] - verts[:, head].mean(1, keepdims=True)
    U, _, Vt = np.linalg.svd(np.einsum("vi,tvj->tij", a, b))
    D = np.eye(3, dtype=np.float32)[None].repeat(len(verts), 0)
    D[:, 2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt  # row vectors: ref offset @ R[t] is the offset in frame t
    verts[:, head] += np.einsum("vi,tij->tvj", shell, R)
    verts[:, neck] += np.einsum("vi,tij->tvj", bib, R)
    return verts


def watch_only(watch_pose, out, watch, first, last, hand="right"):
    """The watch-only fencer in a studio scene for render/studio.py: frames first..last (frame N is pose
    sample N - 1), nothing from the video. The body is render.body's surface on the predicted joints, the
    blade comes from the watch's attitude and the BLADE_MODEL fitted by imu.blade_video.

    The pose is pelvis-relative in the body frame (right, up, back); each frame is lowered until the lowest
    point of the body rests on the floor. With a predicted pelvis track (`root`, same frame) in the watch pose
    the fencer travels along it; without one the pelvis stays over one spot.
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
    # Dress the kit on the frame with the limbs furthest apart, where nearest-limb is unambiguous.
    joints = world(full[lo:last])
    spread = (np.linalg.norm(joints[:, L_WR], axis=1) + np.linalg.norm(joints[:, R_WR], axis=1)
              + np.linalg.norm(joints[:, L_ANK] - joints[:, R_ANK], axis=1))
    ref = int(np.argmax(spread))
    fields, cloth, front, *mask = kit(verts[ref], joints[ref], model["faces"].astype(int))
    verts = masked(verts, ref, *mask)
    ground = savgol_filter(verts[..., 2].min(1), 9, 2)
    verts[..., 2] -= ground[:, None]
    knuckle[:, 2] -= ground

    k = first - 1 - lo
    follow = None
    if "root" in w.files:
        root = world(savgol_filter(fill_gaps(w["root"].astype(np.float32))[lo:last], 9, 2, axis=0))
        root[:, 2] = 0
        root -= root[k:].mean(0)  # centred on the piste over the frames shown
        verts += root[:, None]
        knuckle += root
        follow = savgol_filter(root[k:, :2], min(45, (len(root) - k) // 2 * 2 - 1), 2, axis=0)  # what the camera tracks

    axis = world(watch_blade(np.load(BLADE_MODEL), watch, w["t"], full, hand).astype(np.float32)[lo:last])
    axis = savgol_filter(axis, 5, 2, axis=0)
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    s, r = np.array(BLADE_S, np.float32), BLADE_R
    blade, blade_faces = tube(knuckle[:, None] + s[None, :, None] * axis[:, None], r, 16)

    # The trail follows a smoothed tip, sampled between frames: at 30 Hz a fast point leaves a polygon.
    tip = savgol_filter(knuckle + BLADE_M * axis, 9, 3, axis=0)
    path = CubicSpline(np.arange(len(tip)), tip)
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
    stance = np.linalg.norm((feet[:, JOINTS.index(14)] - feet[:, JOINTS.index(13)])[:, :2], axis=1)
    scene = {"fps": fps, "first": first, "frames": last - first + 1, "widest": first + int(np.argmax(stance)),
             "facing_deg": float(np.degrees(np.arctan2(reach[1], reach[0]))),
             "body_rest": verts[k].tolist(), "body_faces": model["faces"].astype(int).tolist(),
             "kit": {name: [*rgb, rough, metal] for name, (rgb, rough, metal) in KIT.items()},
             "body_fields": {name: f.tolist() for name, f in fields.items()}, "body_cloth": cloth.tolist(),
             "body_mesh": front.tolist(),
             "blade_rest": blade[k].tolist(), "blade_faces": blade_faces,
             "trail_rest": trail[0].tolist(), "trail_faces": trail_faces,
             "trail_age": np.repeat(ages / ages[-1], 6).tolist()}
    if follow is not None:
        scene["follow"] = follow.tolist()
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
