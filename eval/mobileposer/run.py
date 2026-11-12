"""Pretrained MobilePoser, right-watch input only, against the video pose on the held-out demo take.

Needs MobilePoser cloned at /tmp/mobileposer (github.com/SPICExLAB/MobilePoser), its weights at
/tmp/mobileposer/weights/weights.pth, and a Python with torch, numpy<1.24 and chumpy:

    /tmp/mobileposer/.venv/bin/python -W ignore eval/mobileposer/run.py
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
MP = Path("/tmp/mobileposer")
sys.path[:0] = [str(REPO), str(MP)]
inspect.getargspec = inspect.getfullargspec  # chumpy, needed to unpickle the SMPL file, predates Python 3.11

from imu.load import G, load_sensor_logger  # noqa: E402
from imu.synth import ELBOW_KP, WRIST_KP  # noqa: E402
from model.pairs import JOINTS, RATE, attitude, pair  # noqa: E402
from model.train import mpjpe  # noqa: E402
from mobileposer.config import amass, paths  # noqa: E402

paths.smpl_file = MP / "mobileposer/smpl/basicmodel_m.pkl"
from mobileposer.utils.model_utils import load_model, smooth_avg  # noqa: E402

WATCH = REPO / "data/raw/demo/watch"
NPZ = REPO / "data/processed/demo_video30.sam3d.npz"
OURS = REPO / "data/processed/demo_video30.watchpose.npz"

# (SMPL joint, MHR70 keypoint) pairs that name the same anatomical point on both skeletons.
SHARED = {"l_shoulder": (16, 5), "r_shoulder": (17, 6), "l_elbow": (18, ELBOW_KP["left"]),
          "r_elbow": (19, ELBOW_KP["right"]), "l_wrist": (20, WRIST_KP["left"]), "r_wrist": (21, WRIST_KP["right"]),
          "l_hip": (1, 9), "r_hip": (2, 10), "l_knee": (4, 11), "r_knee": (5, 12), "l_ankle": (7, 13),
          "r_ankle": (8, 14)}
SMPL_IDX = [s for s, _ in SHARED.values()]
OUR_IDX = [JOINTS.index(k) for _, k in SHARED.values()]
NAMES = list(SHARED)
GROUPS = {"body": NAMES, "arm": ["r_shoulder", "r_elbow", "r_wrist"],
          "legs": ["l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]}
GROUPS = {g: [NAMES.index(n) for n in ns] for g, ns in GROUPS.items()}
SWAP = [NAMES.index(n.replace("l_", "#").replace("r_", "l_").replace("#", "r_")) for n in NAMES]

# SMPL world (x left, y up, z forward) <- our heading frame (right, up, back).
SMPL_FROM_HEADING = np.diag([-1.0, 1.0, -1.0])
MIRROR = np.diag([-1.0, 1.0, 1.0])
RW, LW = 1, 0  # MobilePoser IMU slots: left wrist, right wrist, left thigh, right thigh, head


def bone_from_watch(forearm, twist_deg=0.0):
    """Right-forearm bone <- watch, assuming a T-pose with the palm down and the screen on the back of the wrist.

    The SMPL right forearm points along -x in the T-pose, so the watch's elbow->wrist axis goes to -x and the
    screen normal (watch +z) to +y. twist_deg turns the strap about the forearm for the sensitivity sweep.
    """
    e1 = -forearm / np.linalg.norm(forearm)
    z = np.array([0.0, 0.0, 1.0])
    e2 = z - (z @ e1) * e1
    e2 /= np.linalg.norm(e2)
    B = np.stack([e1, e2, np.cross(e1, e2)])
    return B @ Rotation.from_rotvec(np.deg2rad(twist_deg) * e1).as_matrix().T


def mobileposer_input(Rw, user_acc_g, W, B, mirror=False):
    """(T, 60) MobilePoser input with only one wrist slot filled, as its loader builds it.

    Rw (T,3,3) CoreMotion reference <- watch; user_acc_g (T,3) CoreMotion userAcceleration in the watch frame;
    W SMPL world <- reference; B bone <- watch.
    """
    ori = W @ Rw @ B.T
    # CoreMotion's userAcceleration is minus the kinematic acceleration (it reads -1 g along z at rest
    # face up); MobilePoser wants the free acceleration in m/s^2, gravity removed.
    acc = np.einsum("ij,tjk,tk->ti", W, Rw, -user_acc_g * G)
    slot = RW
    if mirror:
        ori, acc, slot = MIRROR @ ori @ MIRROR, acc @ MIRROR, LW
    A = torch.zeros(len(acc), 5, 3)
    O = torch.zeros(len(acc), 5, 3, 3)
    A[:, slot] = torch.as_tensor(acc, dtype=torch.float32) / amass.acc_scale
    O[:, slot] = torch.as_tensor(ori, dtype=torch.float32)
    A = smooth_avg(A)
    return torch.cat([A.flatten(1), O.flatten(1)], 1)


@torch.no_grad()
def run(net, x, mirror=False, online=False):
    """(T, 12, 3) shared joints in the SMPL world, pelvis (mid-hip) relative.

    Offline runs the bidirectional nets over the whole take, as MobilePoser's evaluate.py does; online is its
    live mode, 40 past and 5 future frames, fed and trimmed the way evaluate.py's ONLINE path does.
    """
    if online:
        net.reset()
        f = net.num_future_frames
        pose = torch.stack([net.forward_online(xi)[0] for xi in torch.cat([x, x[-1].repeat(f, 1)])])[f:]
    else:
        pose, *_ = net.forward(x[None], [len(x)])
    _, j = net.bodymodel.forward_kinematics(pose.view(-1, 24, 3, 3))
    j = j.numpy()[:, SMPL_IDX]
    j = j - (j[:, [NAMES.index("l_hip")]] + j[:, [NAMES.index("r_hip")]]) / 2
    if mirror:
        j = j[:, SWAP] @ MIRROR
    return j


def best_yaw(pred, true, valid):
    """Rotation about up (heading y) that best maps pred onto true over all valid frames and joints."""
    a, b = pred[valid].reshape(-1, 3), true[valid].reshape(-1, 3)
    th = np.arctan2((a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]).sum(), (a[:, 0] * b[:, 0] + a[:, 2] * b[:, 2]).sum())
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), np.degrees(th)


def procrustes(pred, true):
    """Per-frame rotation-only Procrustes about the pelvis (no scale, no translation)."""
    H = np.einsum("tji,tjk->tik", pred, np.nan_to_num(true))  # invalid frames are NaN; mpjpe drops them
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    U[:, :, -1] *= d[:, None]
    return np.einsum("tji,tik->tjk", pred, U @ Vt)


def errors(pred, true, valid):
    return {g: mpjpe(pred, true, valid, j) for g, j in GROUPS.items()}


def row(name, e):
    print(f"{name:<44}" + "".join(f"{e[g]:7.0f}" for g in GROUPS))


def main():
    p = pair(WATCH, NPZ)
    ours = np.load(OURS)
    assert np.allclose(ours["t"], p["t"]) and np.allclose(ours["R"], p["R"]), "watchpose.npz is from another pairing"
    true, valid = p["pose"][:, OUR_IDX].astype(float), p["valid"]
    mine = ours["pose"][:, OUR_IDX].astype(float)

    Rw = attitude(load_sensor_logger(WATCH), p["t"])
    user_acc = p["imu"][:, 3:6].astype(float)  # model.pairs low-passes it to 12 Hz on the 30 Hz grid
    W = SMPL_FROM_HEADING @ p["R"] @ p["Q"]
    B = bone_from_watch(p["forearm"])

    # Input check: the bone's forearm direction should be the video's elbow->wrist, both in the SMPL world.
    fore_in = (W @ Rw @ B.T)[:, :, 0] * -1
    fv = p["pose"][:, JOINTS.index(WRIST_KP["right"])] - p["pose"][:, JOINTS.index(ELBOW_KP["right"])]
    fv = fv @ SMPL_FROM_HEADING
    fv /= np.linalg.norm(fv, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((fore_in[valid] * fv[valid]).sum(1), -1, 1)))
    print(f"input forearm vs video forearm: median {np.median(ang):.1f} deg, p90 {np.percentile(ang, 90):.1f} deg")

    net = load_model(MP / "weights/weights.pth").eval()
    print(f"{len(p['t'])} frames, {valid.sum()} valid; MPJPE mm over the {len(NAMES)} shared joints")
    print(f"{'':<44}" + "".join(f"{g:>7}" for g in GROUPS))

    def heading(j):
        return j @ SMPL_FROM_HEADING  # SMPL_FROM_HEADING is its own inverse

    results = {}
    for mirror in (False, True):
        pr = heading(run(net, mobileposer_input(Rw, user_acc, W, B, mirror), mirror))
        Y, deg = best_yaw(pr, true, valid)
        tag = "MobilePoser, " + ("left slot, mirrored" if mirror else "right-wrist slot")
        row(f"{tag}: calibrated frame", errors(pr, true, valid))
        row(f"{tag}: + best-fit yaw {deg:+.0f} deg", errors(pr @ Y.T, true, valid))
        row(f"{tag}: per-frame Procrustes", errors(procrustes(pr, true), true, valid))
        results[mirror] = pr
    Y, deg = best_yaw(mine, true, valid)
    row("ours: as predicted", errors(mine, true, valid))
    row(f"ours: + best-fit yaw {deg:+.0f} deg", errors(mine @ Y.T, true, valid))
    row("ours: per-frame Procrustes", errors(procrustes(mine, true), true, valid))
    still = np.broadcast_to(np.nanmean(true[valid], 0), true.shape)
    row("mean pose of this take (oracle, reference)", errors(still, true, valid))

    pr = results[False]
    scale = float((pr[valid] * true[valid]).sum() / (pr[valid] ** 2).sum())
    Y, _ = best_yaw(pr * scale, true, valid)
    row(f"MobilePoser right, yaw + scale x{scale:.2f}", errors(pr * scale @ Y.T, true, valid))

    print("\nsensitivity, right-wrist slot, best-fit yaw (body arm legs):")
    for twist in (-60, -30, 30, 60):
        pr = heading(run(net, mobileposer_input(Rw, user_acc, W, bone_from_watch(p["forearm"], twist))))
        Y, _ = best_yaw(pr, true, valid)
        row(f"  strap twist {twist:+d} deg", errors(pr @ Y.T, true, valid))
    # Control: the video's own wrist acceleration, put back into CoreMotion's userAcceleration convention.
    wrist = p["root"] + p["pose"][:, JOINTS.index(WRIST_KP["right"])]
    wrist = pd.DataFrame(wrist).interpolate(limit_direction="both").to_numpy()
    acc_h = savgol_filter(wrist, 9, 3, deriv=2, delta=1 / RATE, axis=0)
    video_acc = -np.einsum("tji,jk,tk->ti", Rw, (p["R"] @ p["Q"]).T, acc_h) / G
    for name, kw, xin in (("online (live) mode", dict(online=True), mobileposer_input(Rw, user_acc, W, B)),
                          ("acceleration zeroed", {}, mobileposer_input(Rw, 0 * user_acc, W, B)),
                          ("acceleration sign flipped", {}, mobileposer_input(Rw, -user_acc, W, B)),
                          ("acceleration from the video wrist", {}, mobileposer_input(Rw, video_acc, W, B))):
        pr = heading(run(net, xin, **kw))
        Y, _ = best_yaw(pr, true, valid)
        row(f"  {name}", errors(pr @ Y.T, true, valid))
    for yaw in range(-180, 180, 45):
        if yaw == 0:
            continue
        Wy = Rotation.from_euler("y", yaw, degrees=True).as_matrix() @ W
        pr = heading(run(net, mobileposer_input(Rw, user_acc, Wy, B)))
        Y, _ = best_yaw(pr, true, valid)
        row(f"  calibration heading off by {yaw:+d} deg", errors(pr @ Y.T, true, valid))


if __name__ == "__main__":
    main()
