"""Watch-only pose from a Sensor Logger recording and weights from model.train --save; no video needed.

    uv run python -m model.predict sample/demo_watch.zip sample/weights.pt --targets sample/demo_targets.npz
writes <watch>.watchpose.npz beside the input. With --targets (video poses on the same watch-time grid), it
prints the error and the lunges and thrusts found from the watch against the video's.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import model.train as train
from analysis.moves import WATCH_LUNGE_PEAK, events, label, match
from imu.load import load_sensor_logger
from model.pairs import JOINTS, RATE, model_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("watch")
    ap.add_argument("weights")
    ap.add_argument("--targets")
    a = ap.parse_args()

    ck = torch.load(a.weights, map_location="cpu", weights_only=True)
    train.WIN = ck["win"]
    df = load_sensor_logger(a.watch)
    tg = np.load(a.targets) if a.targets else None
    t = tg["t"] if tg is not None else np.arange(np.ceil(df.index[0] * RATE), np.floor(df.index[-1] * RATE) + 1) / RATE
    x = ((model_input(df, t) - ck["mu"].numpy()) / ck["sd"].numpy()).astype(np.float32)
    nets = []
    for sd in ck["nets"]:
        net = train.Net(x.shape[1], len(JOINTS), ck["mean_pose"], ck["hidden"], ck["causal"])
        net.load_state_dict(sd)
        nets.append(net.eval())
    pred = np.mean([train.predict(net, x, 0, len(t), "cpu") for net in nets], axis=0)
    out = Path(a.watch).with_suffix(".watchpose.npz")
    np.savez(out, t=t, pose=pred)
    print(out)
    if tg is None:
        return

    true, valid = tg["pose"], tg["valid"]
    mean = np.broadcast_to(ck["mean_pose"].numpy(), true.shape)
    for name, pr in (("watch model", pred), ("average pose", mean)):
        print(f"{name:<14}MPJPE mm  " + "".join(f"{g}: {train.mpjpe(pr, true, valid, j):.0f}  " for g, j in train.GROUPS.items()))
    video, watch = events(true, label(true)), events(pred, label(pred), WATCH_LUNGE_PEAK)
    pairs = match(video, watch)
    for kind, name in (("lunge", "lunges"), ("extension", "thrusts")):
        pk = [(e, f) for e, f in pairs if e["kind"] == kind]
        onset = np.median([abs(f["onset"] - e["onset"]) for e, f in pk]) * 1000 / RATE if pk else np.nan
        false = sum(f["kind"] == kind for f in watch) - len(pk)
        print(f"{name}: watch found {len(pk)} of the video's {sum(e['kind'] == kind for e in video)} ({false} false), "
              f"median onset error {onset:.0f} ms")


if __name__ == "__main__":
    main()
