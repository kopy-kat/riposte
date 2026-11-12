"""Wrist IMU -> pelvis velocity in the watch's levelled frame: where the fencer moves, from the watch alone.

    uv run python -m model.piste $(cat data/takes.txt) --test data/raw/demo/watch,data/processed/demo_video30.sam3d.npz \\
        --save data/processed/piste/piste.pt --plot data/processed/piste/demo.png
trains on the first takes and scores the --test take: velocity and integrated position against the video's
pelvis, both in model.pairs.level_frame, which is fixed by the watch (the video's pelvis is turned into it
with the take's camera <- watch alignment, for training and for scoring).
"""

import argparse

import numpy as np
import torch
from scipy.signal import butter, savgol_filter, sosfiltfilt

import model.train as train
from analysis.moves import agreement, events, label
from imu.load import G, load_sensor_logger
from model.pairs import RATE, attitude, level_frame, load

SMOOTH = 9  # frames: the video pelvis jitters in depth frame to frame


def velocity(track, valid):
    """(T, 3) m/s from a pelvis track with gaps, and the frames where it can be trusted."""
    i = np.arange(len(track))
    filled = np.stack([np.interp(i, i[valid], track[valid, k]) for k in range(3)], 1)
    v = savgol_filter(filled, SMOOTH, 3, deriv=1, delta=1 / RATE, axis=0)
    ok = np.convolve(valid, np.ones(SMOOTH), "same") == SMOOTH
    return v.astype(np.float32), ok


def integrated_accel(df, t, highpass_s=None):
    """Baseline: the watch's own user acceleration in level_frame, integrated to a velocity at times t."""
    tw = df.index.to_numpy()
    R = attitude(df, tw)
    # Sensor Logger's user acceleration has the opposite sign to the wrist's acceleration in the video.
    a = -np.einsum("ij,tjk,tk->ti", level_frame(attitude(df, t)), R, df[["acc_x", "acc_y", "acc_z"]].to_numpy()) * G
    v = np.cumsum(a, axis=0) / df.attrs["rate_hz"]
    if highpass_s:
        v = sosfiltfilt(butter(2, 1 / highpass_s, "high", fs=df.attrs["rate_hz"], output="sos"), v, axis=0)
    return np.stack([np.interp(t, tw, v[:, k]) for k in range(3)], 1)


def drift(pred, true, ok, seconds):
    """Median and 90th percentile horizontal position error (m) at the end of windows integrated from their start."""
    n, errs = int(seconds * RATE), []
    for a in range(0, len(pred) - n, n // 2):
        if ok[a:a + n].all():
            errs.append(np.linalg.norm((pred[a:a + n, :2] - true[a:a + n, :2]).sum(0)) / RATE)
    return (np.median(errs), np.percentile(errs, 90), len(errs)) if errs else (np.nan, np.nan, 0)


def leaky(v, ok, tau):
    """Position from velocity with a leak of tau seconds: where the fencer is relative to where they have
    been lately. A plain integral of a predicted velocity drifts without bound."""
    out, k = np.zeros_like(v), np.exp(-1 / (tau * RATE))
    for i in range(1, len(v)):
        out[i] = out[i - 1] * k + (v[i] if ok[i] else 0) / RATE
    return out


def travel(pose, vel, tau=3.0):
    """(T, 3) pelvis track in the pose's frame (right, up, back) from model.predict's pelvis_velocity: leaky
    position on the floor, turned about the vertical so the levelled frame's +x, the take's mean forearm
    heading, lies along the predicted pose's own mean forearm. Nothing from the video."""
    from imu.synth import ELBOW_KP, WRIST_KP
    from model.pairs import JOINTS

    p = leaky(vel, np.isfinite(vel).all(1), tau)
    p = np.stack([p[:, 0], np.zeros(len(p)), -p[:, 1]], 1)  # levelled (x, y, z up) -> y up
    f = np.nanmean(pose[:, JOINTS.index(WRIST_KP["right"])] - pose[:, JOINTS.index(ELBOW_KP["right"])], axis=0)
    c, s = f[0] / np.hypot(f[0], f[2]), f[2] / np.hypot(f[0], f[2])
    return (p @ np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])).astype(np.float32)


def score(name, v, true, ok, axis):
    """One line: velocity along the piste axis (corr, RMS), then drift over 5, 10 and 30 s windows."""
    a, b = v[ok] @ axis, true[ok] @ axis
    d = "  ".join(f"{s:>2d} s: {m:.2f} / {p:.2f} m" for s in (5, 10, 30) for m, p, _ in [drift(v, true, ok, s)])
    print(f"{name:<28} corr {np.corrcoef(a, b)[0, 1]:5.2f}  RMS {np.sqrt(np.mean((a - b) ** 2)):.2f} m/s   drift median / p90  {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("takes", nargs="+", help="watch_path,npz_path[,offset_s], or a pair file from model.pairs --save")
    ap.add_argument("--test", required=True, help="the take never trained on")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--save")
    ap.add_argument("--plot")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    takes = []
    for spec in a.takes + [a.test]:
        p = load(spec)
        p["vel"], p["valid"] = velocity(p["track"], p["valid"])
        takes.append(p)
    tr, te = takes[:-1], takes[-1]
    items = [(i, s) for i, p in enumerate(tr) for s in train.windows(0, len(p["t"]), p["valid"])]
    imu = np.concatenate([p["imu"][p["valid"]] for p in tr])
    mu, sd = imu.mean(0), imu.std(0) + 1e-6
    mu_t, sd_t = torch.as_tensor(mu, device=a.device), torch.as_tensor(sd, device=a.device)
    ys = [p["vel"][:, None] for p in takes]
    zero = np.zeros((1, 3), np.float32)
    nets = [train.train_net(tr, items, ys, zero, mu_t, sd_t, a.epochs, a.device, s, None, 0, 15.0, 128, False)
            for s in range(a.seeds)]
    if a.save:
        torch.save(dict(nets=[dict(net.state_dict()) for net in nets], mu=torch.as_tensor(mu), sd=torch.as_tensor(sd),
                        hidden=128, win=train.WIN), a.save)

    x = ((te["imu"] - mu) / sd).astype(np.float32)
    pred = np.mean([train.predict(net, x, 0, len(x), a.device) for net in nets], axis=0)[:, 0]
    true, ok = te["vel"], te["valid"]
    # The piste: the horizontal line the video's pelvis moves along, in the watch's frame. Scoring only.
    axis = np.r_[np.linalg.svd(true[ok, :2] - true[ok, :2].mean(0), full_matrices=False)[2][0], 0.0]
    axis *= np.sign(axis[0])
    print(f"piste axis {np.degrees(np.arctan2(axis[1], axis[0])):+.0f} deg from the watch's +x; video speed along it "
          f"SD {np.std(true[ok] @ axis):.2f} m/s; {ok.sum() / RATE:.0f} s scored")
    score("watch model", pred, true, ok, axis)
    score("zero velocity", np.zeros_like(pred), true, ok, axis)
    if "," in a.test:  # a pair file carries the model input, not the recording this baseline integrates
        df = load_sensor_logger(a.test.split(",")[0])
        score("integrated watch accel", integrated_accel(df, te["t"]), true, ok, axis)
        score("  high-passed at 2 s", integrated_accel(df, te["t"], 2.0), true, ok, axis)

    pos = lambda v: np.cumsum(np.where(ok[:, None], v, 0), axis=0) / RATE
    e = np.linalg.norm((pos(pred) - pos(true))[:, :2], axis=1)
    print(f"whole take, integrated from the start: end error {e[-1]:.2f} m, max {e.max():.2f} m; "
          f"video range along the piste {np.ptp(pos(true) @ axis):.2f} m")
    for tau in (3, 5, 10):
        lp, lt = leaky(pred, ok, tau) @ axis, leaky(true, ok, tau) @ axis
        print(f"leaky position along the piste, {tau:>2d} s leak: RMS error {np.sqrt(np.mean((lp - lt)[ok] ** 2)):.2f} m, "
              f"corr {np.corrcoef(lp[ok], lt[ok])[0, 1]:.2f}; video's own RMS {np.sqrt(np.mean(lt[ok] ** 2)):.2f} m")
    ev = [e for e in events(te["pose"], label(te["pose"])) if e["kind"] == "lunge" and ok[e["onset"]:e["peak"] + 1].all()]
    travel = lambda v: np.array([(v[e["onset"]:e["peak"] + 1] @ axis).sum() / RATE for e in ev])
    rho, err, sd = agreement(travel(true), travel(pred))
    print(f"pelvis travel per lunge, onset to peak, over {len(ev)} video lunges: video median {np.median(travel(true)):.2f} m, "
          f"watch {np.median(travel(pred)):.2f} m, median error {err:.2f} m, rank correlation {rho:.2f} (video SD {sd:.2f} m)")
    if a.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = te["t"] - te["t"][0]
        fig, ax = plt.subplots(3, 1, figsize=(16, 10))
        ax[0].plot(t, leaky(true, ok, 5) @ axis, "k", lw=1, label="video")
        ax[0].plot(t, leaky(pred, ok, 5) @ axis, "C2", lw=1, label="watch")
        ax[0].set_ylabel("position along the piste, 5 s leak, m")
        m = (t > 60) & (t < 90)
        for k, lab in ((1, "position, re-zeroed at 60 s, m"), (2, "velocity along the piste, m/s")):
            f = (lambda v: (pos(v) - pos(v)[m][0]) @ axis) if k == 1 else (lambda v: v @ axis)
            ax[k].plot(t[m], f(true)[m], "k", lw=1, label="video")
            ax[k].plot(t[m], f(pred)[m], "C2", lw=1, label="watch")
            ax[k].set_ylabel(lab)
        for x_ in ax:
            x_.legend(loc="upper right")
        ax[2].set_xlabel("s")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=110)
        print(a.plot)
    if a.save:
        np.savez(a.save.replace(".pt", ".demo.npz"), t=te["t"], pred=pred, true=true, ok=ok, axis=axis)


if __name__ == "__main__":
    main()
