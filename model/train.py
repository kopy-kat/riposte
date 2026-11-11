"""Wrist IMU -> full-body pose sequence, a bidirectional GRU, k-fold over time blocks.

    uv run python -m model.train data/raw/office2/watch,data/processed/office2_video.sam3d.npz[,offset] ...
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from model.pairs import ARM, JOINTS, LEGS, RATE, pair

WIN, STRIDE = int(2 * RATE), int(round(0.25 * RATE))
GROUPS = {"arm": [JOINTS.index(j) for j in ARM], "legs": [JOINTS.index(j) for j in LEGS], "body": list(range(len(JOINTS)))}


def windows(lo, hi, valid, min_valid=0.8):
    return [s for s in range(lo, hi - WIN + 1, STRIDE) if valid[s:s + WIN].mean() >= min_valid]


def split(p, folds, k):
    """Train window starts and the val frame range for fold k. Val is one contiguous time block; train
    windows keep a window's length away from it so no training window shares frames with a val window."""
    T = len(p["t"])
    edges = np.linspace(0, T, folds + 1).astype(int)
    a, b = edges[k], edges[k + 1]
    train = windows(0, max(a - WIN, 0), p["valid"]) + windows(min(b + WIN, T), T, p["valid"])
    return train, (a, b)


class Net(nn.Module):
    def __init__(self, n_in, n_joints, mean_pose, hidden=128, causal=False):
        super().__init__()
        # causal: each frame sees only the past, as a live watch app would.
        self.gru = nn.GRU(n_in, hidden, num_layers=2, batch_first=True, bidirectional=not causal, dropout=0.2)
        self.out = nn.Linear(hidden if causal else 2 * hidden, n_joints * 3)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.copy_(torch.as_tensor(mean_pose).flatten())  # start at the mean-pose baseline

    def forward(self, x):
        h, _ = self.gru(x)
        return self.out(h).unflatten(-1, (-1, 3))


def mpjpe(pred, true, valid, joints):
    """Mean per-joint position error in mm over valid frames."""
    e = np.linalg.norm(pred[valid][:, joints] - true[valid][:, joints], axis=-1)
    return 1000 * float(e.mean())


def predict(net, x, lo, hi, device):
    """Per-frame prediction over [lo, hi), averaging overlapping windows."""
    starts = list(range(lo, hi - WIN + 1, STRIDE))
    if starts[-1] != hi - WIN:
        starts.append(hi - WIN)
    with torch.no_grad():
        xb = torch.as_tensor(np.stack([x[s:s + WIN] for s in starts]), device=device)
        yb = net(xb).cpu().numpy()
    acc = np.zeros((hi - lo, *yb.shape[2:]))
    n = np.zeros(hi - lo)
    for s, y in zip(starts, yb):
        acc[s - lo:s - lo + WIN] += y
        n[s - lo:s - lo + WIN] += 1
    return acc / n[:, None, None]


def knn(xs, items, ys, x, lo, hi, k=5):
    """Per-frame prediction over [lo, hi): the mean pose sequence of the k training windows nearest in
    normalised IMU, averaged over overlapping windows like predict."""
    tx = np.stack([xs[i][s:s + WIN].ravel() for i, s in items])
    ty = np.stack([ys[i][s:s + WIN] for i, s in items])
    starts = list(range(lo, hi - WIN + 1, STRIDE))
    if starts[-1] != hi - WIN:
        starts.append(hi - WIN)
    acc = np.zeros((hi - lo, *ty.shape[2:]))
    n = np.zeros(hi - lo)
    for s in starts:
        near = np.argsort(((tx - x[s:s + WIN].ravel()) ** 2).sum(1))[:k]
        acc[s - lo:s - lo + WIN] += ty[near].mean(0)
        n[s - lo:s - lo + WIN] += 1
    return acc / n[:, None, None]


def en_garde(p, hold_s=3.0):
    """Mean pose over the hold that follows the start tap."""
    m = (p["t"] >= p["tap"] + 0.5) & (p["t"] < p["tap"] + 0.5 + hold_s) & p["valid"]
    return np.nanmean(p["pose"][m], axis=0)


def run_fold(takes, folds, k, epochs, device, seed=0, log=print, held_out=None, rot_deg=15.0, n_test=0, seeds=1,
             hidden=128, causal=False):
    """With n_test, the last n_test takes are the whole validation set and every other take trains in full."""
    if n_test:
        n_train = len(takes) - n_test
        splits = [(windows(0, len(p["t"]), p["valid"]), (0, 0)) if i < n_train else ([], (0, len(p["t"])))
                  for i, p in enumerate(takes)]
    else:
        splits = [split(p, folds, k) for p in takes]
    items = [(i, s) for i, (tr, _) in enumerate(splits) for s in tr]
    if not items:
        raise ValueError("no training windows; take too short for this many folds")
    # Normalisation and the mean pose come from training frames only.
    frames = [(i, np.unique(np.concatenate([np.arange(s, s + WIN) for s in tr]))) for i, (tr, _) in enumerate(splits) if tr]
    tr_imu = np.concatenate([takes[i]["imu"][f] for i, f in frames])
    mu, sd = tr_imu.mean(0), tr_imu.std(0) + 1e-6
    mean_pose = np.nanmean(np.concatenate([takes[i]["pose"][f] for i, f in frames]), axis=0)

    xs = [((p["imu"] - mu) / sd).astype(np.float32) for p in takes]
    mu_t, sd_t = torch.as_tensor(mu, device=device), torch.as_tensor(sd, device=device)
    ys = [np.nan_to_num(p["pose"]).astype(np.float32) for p in takes]

    nets = [train_net(takes, items, ys, mean_pose, mu_t, sd_t, epochs, device, seed + s, log, k, rot_deg, hidden, causal)
            for s in range(seeds)]

    res = {name: {g: [] for g in GROUPS} for name in ("mean pose", "en garde", "knn", "model")}
    for i, (p, x, (_, (a, b))) in enumerate(zip(takes, xs, splits)):
        if b - a < WIN:
            continue
        true, v = p["pose"][a:b], p["valid"][a:b]
        preds = {"mean pose": np.broadcast_to(mean_pose, true.shape),
                 "en garde": np.broadcast_to(en_garde(p), true.shape),
                 "knn": knn(xs, items, ys, x, a, b),
                 "model": np.mean([predict(net, x, a, b, device) for net in nets], axis=0)}
        if held_out is not None:
            held_out[i][a:b] = preds["model"]
        for name, pr in preds.items():
            for g, j in GROUPS.items():
                res[name][g].append((mpjpe(pr, true, v, j), int(v.sum())))
    # Frame-weighted over takes.
    return {name: {g: sum(e * n for e, n in r) / max(sum(n for _, n in r), 1) for g, r in gs.items()}
            for name, gs in res.items()}


def train_net(takes, items, ys, mean_pose, mu_t, sd_t, epochs, device, seed, log, k, rot_deg, hidden, causal):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(takes[0]["imu"].shape[1], len(JOINTS), mean_pose, hidden, causal).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(items))
        for b in range(0, len(order), 32):
            batch = [items[j] for j in order[b:b + 32]]
            x = torch.as_tensor(np.stack([takes[i]["imu"][s:s + WIN] for i, s in batch]), device=device)
            y = torch.as_tensor(np.stack([ys[i][s:s + WIN] for i, s in batch]), device=device)
            m = torch.as_tensor(np.stack([takes[i]["valid"][s:s + WIN] for i, s in batch]), device=device)
            # One random rotation per window on gyro, accel and gravity alike: a different strap placement.
            rot = Rotation.from_rotvec(rng.normal(size=(len(batch), 3)) * np.deg2rad(rot_deg) / np.sqrt(3))
            rot = torch.as_tensor(rot.as_matrix(), dtype=torch.float32, device=device)
            x = (x.unflatten(-1, (3, 3)) @ rot.transpose(1, 2)[:, None]).flatten(-2)
            x = (x - mu_t) / sd_t + 0.05 * torch.randn_like(x)
            err = (net(x) - y).norm(dim=-1).mean(-1)
            loss = (err * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if log and (ep + 1) % max(epochs // 5, 1) == 0:
            log(f"  fold {k} epoch {ep + 1}: train MPJPE {1000 * loss.item():.0f} mm")
    return net.eval()


def report(results):
    print(f"{'MPJPE mm':<10}" + "".join(f"{g:>8}" for g in GROUPS))
    for name in results[0]:
        print(f"{name:<10}" + "".join(f"{np.mean([r[name][g] for r in results]):8.0f}" for g in GROUPS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("takes", nargs="+", help="watch_path,npz_path[,offset_s]")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--out", help="write each take's held-out predictions to <npz stem>.watchpose.npz in this dir")
    ap.add_argument("--test", nargs="+", default=[], help="takes never trained on: train on all of the others, predict these")
    ap.add_argument("--win", type=float, default=2.0, help="window seconds")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--rot", type=float, default=15.0, help="strap-rotation augmentation, degrees")
    ap.add_argument("--seeds", type=int, default=1, help="average this many independently trained models")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    global WIN
    WIN = int(a.win * RATE)
    takes = []
    for spec in a.takes + a.test:
        w, n, *off = spec.split(",")
        p = pair(w, n, offset=float(off[0]) if off else None)
        print(f"{n}: offset {p['offset']:.3f} s (coarse {p['coarse']:.3f}), corr {p['peak']:.2f}, "
              f"{len(p['t']) / RATE:.1f} s overlap, {p['valid'].mean():.0%} valid")
        takes.append(p)
    held_out = [np.full_like(p["pose"], np.nan) for p in takes]
    kw = dict(held_out=held_out, rot_deg=a.rot, seeds=a.seeds, hidden=a.hidden, causal=a.causal)
    if a.test:
        results = [run_fold(takes, a.folds, 0, a.epochs, a.device, n_test=len(a.test), **kw)]
    else:
        results = [run_fold(takes, a.folds, k, a.epochs, a.device, **kw) for k in range(a.folds)]
    report(results)
    if a.out:
        for spec, p, pred in zip(a.takes + a.test, takes, held_out):
            if np.isnan(pred).all():
                continue
            out = Path(a.out) / (Path(spec.split(",")[1]).name.removesuffix(".sam3d.npz") + ".watchpose.npz")
            np.savez(out, t=p["t"], pose=pred, R=p["R"], offset=p["offset"])
            print(out)


if __name__ == "__main__":
    main()
