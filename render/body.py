"""A body surface for the watch-only pose: the 30 predicted joints -> the SAM 3D Body mesh's vertices.

    uv run python -m render.body data/processed/body/body.npz $(cat data/takes.txt) \
        --test data/raw/demo/watch,data/processed/demo_video30.sam3d.npz --watch-pose data/processed/demo_video30.watchpose.npz
fits it on the training takes' own video meshes (PCA of the vertices, an MLP from the joints to its
coefficients) and prints the vertex error on the test take, fed the video's joints and the watch's.
The surface only dresses the joints for the render: fingers and face are whatever the joints imply on average.
"""

import argparse

import numpy as np
import torch
from torch import nn

from model.pairs import JOINTS, L_HIP, R_HIP, pair

STRIDE = 3  # video frames between training samples: neighbours are near-duplicates
N_PC = 64
KNUCKLE_KP = 32  # MHR70 right middle knuckle, carried as an extra vertex: where the blade leaves the hand


def load(spec, stride=1):
    """(joints (T, 30, 3), vertices (T, V + 1, 3)) of a take's video frames, pelvis-relative in pair()'s heading frame."""
    w, n, *off = spec.split(",")
    R = pair(w, n, offset=float(off[0]) if off else None)["R"]
    d = np.load(n)
    ok = np.flatnonzero(d["found"].astype(bool))[::stride]
    kp = d["pred_keypoints_3d"][ok].astype(np.float32)
    pelvis = (kp[:, L_HIP] + kp[:, R_HIP])[:, None] / 2
    v = np.concatenate([d["pred_vertices"][ok].astype(np.float32), kp[:, [KNUCKLE_KP]]], 1)
    good = np.isfinite(kp).all((1, 2)) & np.isfinite(v).all((1, 2))
    Rt = R.T.astype(np.float32)
    return ((kp[:, JOINTS] - pelvis) @ Rt)[good], ((v - pelvis) @ Rt)[good], ok[good], d


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, n_out))


def dress(model, joints):
    """joints (T, 30, 3) -> vertices (T, V, 3) and the knuckle (T, 3)."""
    net = mlp(90, len(model["pc"]))
    net.load_state_dict({k[4:]: torch.as_tensor(model[k]) for k in model.files if k.startswith("net.")})
    x = (joints.reshape(len(joints), -1) - model["x_mu"]) / model["x_sd"]
    with torch.no_grad():
        c = net(torch.as_tensor(x, dtype=torch.float32)).numpy() * model["c_sd"]
    v = (c @ model["pc"] + model["mean"]).reshape(len(joints), -1, 3)
    return v[:, :-1], v[:, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("takes", nargs="+")
    ap.add_argument("--test")
    ap.add_argument("--watch-pose")
    a = ap.parse_args()

    X, V = [], []
    for spec in a.takes:
        x, v, _, d = load(spec, STRIDE)
        X.append(x)
        V.append(v)
        print(spec.split(",")[1], len(x))
    X, V = np.concatenate(X).reshape(sum(map(len, X)), -1), np.concatenate(V)
    V = V.reshape(len(V), -1)
    mean = V.mean(0)
    V -= mean
    _, s, pc = torch.pca_lowrank(torch.as_tensor(V), q=N_PC, center=False, niter=4)
    pc = pc.T.numpy()
    C = V @ pc.T
    print(f"PCA {N_PC}: {1000 * np.sqrt(((V - C @ pc) ** 2).reshape(len(V), -1, 3).sum(-1)).mean():.1f} mm left out")
    x_mu, x_sd, c_sd = X.mean(0), X.std(0) + 1e-6, C.std(0)

    torch.manual_seed(0)
    net = mlp(X.shape[1], N_PC)
    xt, ct = torch.as_tensor((X - x_mu) / x_sd, dtype=torch.float32), torch.as_tensor(C / c_sd, dtype=torch.float32)
    wt = torch.as_tensor(c_sd / c_sd.max(), dtype=torch.float32)  # the loss in vertex units, not per-component
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=6000)
    for step in range(6000):
        i = torch.randint(len(xt), (256,))
        # Noise at the watch model's error scale: the surface must stay sane on joints it predicts.
        loss = (((net(xt[i] + 0.1 * torch.randn(256, xt.shape[1])) - ct[i]) * wt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    np.savez(a.out, mean=mean, pc=pc, x_mu=x_mu, x_sd=x_sd, c_sd=c_sd, faces=d["faces"],
             **{f"net.{k}": v.numpy() for k, v in net.state_dict().items()})
    model = np.load(a.out)

    if a.test:
        x, v, frames, d = load(a.test)
        err = lambda p: 1000 * np.linalg.norm(p - v[:, :-1], axis=-1).mean()
        pv = dress(model, x)[0]
        print(f"test take, {len(x)} frames: vertices from the video's joints {err(pv):.0f} mm")
        if a.watch_pose:
            w = np.load(a.watch_pose)
            tv = frames / float(d["fps"]) + float(w["offset"])
            flat = w["pose"].reshape(len(w["t"]), -1)
            wj = np.stack([np.interp(tv, w["t"], flat[:, k]) for k in range(flat.shape[1])], 1).reshape(len(tv), -1, 3)
            keep = (tv >= w["t"][0]) & (tv <= w["t"][-1]) & np.isfinite(wj).all((1, 2))
            pw = dress(model, wj[keep])[0]
            print(f"  from the watch's joints {1000 * np.linalg.norm(pw - v[keep, :-1], axis=-1).mean():.0f} mm "
                  f"(the joints themselves {1000 * np.linalg.norm(wj[keep] - x[keep], axis=-1).mean():.0f} mm)")
            for name, p in (("video joints", pv), ("watch joints", pw), ("video mesh", v[:, :-1])):
                print(f"  frame-to-frame vertex step, {name}: {1000 * np.linalg.norm(np.diff(p, axis=0), axis=-1).mean():.1f} mm")


if __name__ == "__main__":
    main()
