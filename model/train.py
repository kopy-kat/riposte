"""Wrist IMU -> full-body pose sequence, a bidirectional GRU, k-fold over time blocks.

    uv run python -m model.train data/raw/office2/watch,data/processed/office2.sam3d.npz[,offset] ...
"""

import argparse

import numpy as np
import torch
from torch import nn

from model.pairs import ARM, JOINTS, LEGS, RATE, pair

WIN, STRIDE = int(2 * RATE), int(round(0.25 * RATE))
GROUPS = {"arm": [JOINTS.index(j) for j in ARM], "legs": [JOINTS.index(j) for j in LEGS], "body": list(range(len(JOINTS)))}


def windows(lo, hi, valid, min_valid=0.8):
    return [s for s in range(lo, hi - WIN + 1, STRIDE) if valid[s:s + WIN].mean() >= min_valid]


def split(p, folds, k):
    """Train and val window starts for fold k. Val is one contiguous time block; train windows keep a
    window's length away from it so no training window shares frames with a val window."""
    T = len(p["t"])
    edges = np.linspace(0, T, folds + 1).astype(int)
    a, b = edges[k], edges[k + 1]
    train = windows(0, max(a - WIN, 0), p["valid"]) + windows(min(b + WIN, T), T, p["valid"])
    return train, (a, b)


class Net(nn.Module):
    def __init__(self, n_in, n_joints, mean_pose, hidden=128):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.out = nn.Linear(2 * hidden, n_joints * 3)
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


def en_garde(p, hold_s=3.0):
    """Mean pose over the hold that follows the start tap."""
    m = (p["t"] >= p["tap"] + 0.5) & (p["t"] < p["tap"] + 0.5 + hold_s) & p["valid"]
    return np.nanmean(p["pose"][m], axis=0)


def run_fold(takes, folds, k, epochs, device, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
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
    ys = [np.nan_to_num(p["pose"]).astype(np.float32) for p in takes]

    net = Net(xs[0].shape[1], len(JOINTS), mean_pose).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(items))
        for b in range(0, len(order), 32):
            batch = [items[j] for j in order[b:b + 32]]
            x = torch.as_tensor(np.stack([xs[i][s:s + WIN] for i, s in batch]), device=device)
            y = torch.as_tensor(np.stack([ys[i][s:s + WIN] for i, s in batch]), device=device)
            m = torch.as_tensor(np.stack([takes[i]["valid"][s:s + WIN] for i, s in batch]), device=device)
            x = x + 0.05 * torch.randn_like(x)
            err = (net(x) - y).norm(dim=-1).mean(-1)
            loss = (err * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if log and (ep + 1) % max(epochs // 5, 1) == 0:
            log(f"  fold {k} epoch {ep + 1}: train MPJPE {1000 * loss.item():.0f} mm")

    net.eval()
    res = {name: {g: [] for g in GROUPS} for name in ("mean pose", "en garde", "model")}
    for p, x, (_, (a, b)) in zip(takes, xs, splits):
        if b - a < WIN:
            continue
        true, v = p["pose"][a:b], p["valid"][a:b]
        preds = {"mean pose": np.broadcast_to(mean_pose, true.shape),
                 "en garde": np.broadcast_to(en_garde(p), true.shape),
                 "model": predict(net, x, a, b, device)}
        for name, pr in preds.items():
            for g, j in GROUPS.items():
                res[name][g].append((mpjpe(pr, true, v, j), int(v.sum())))
    # Frame-weighted over takes.
    return {name: {g: sum(e * n for e, n in r) / max(sum(n for _, n in r), 1) for g, r in gs.items()}
            for name, gs in res.items()}


def report(results):
    print(f"{'MPJPE mm':<10}" + "".join(f"{g:>8}" for g in GROUPS))
    for name in results[0]:
        print(f"{name:<10}" + "".join(f"{np.mean([r[name][g] for r in results]):8.0f}" for g in GROUPS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("takes", nargs="+", help="watch_path,npz_path[,offset_s]")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    takes = []
    for spec in a.takes:
        w, n, *off = spec.split(",")
        p = pair(w, n, offset=float(off[0]) if off else None)
        print(f"{n}: offset {p['offset']:.3f} s (coarse {p['coarse']:.3f}), corr {p['peak']:.2f}, "
              f"{len(p['t']) / RATE:.1f} s overlap, {p['valid'].mean():.0%} valid")
        takes.append(p)
    results = [run_fold(takes, a.folds, k, a.epochs, a.device) for k in range(a.folds)]
    report(results)


if __name__ == "__main__":
    main()
