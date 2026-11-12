"""Watch-only pose from a Sensor Logger recording and weights from model.train --save; no video needed.

    uv run python -m model.predict sample/demo_watch.zip sample/weights.pt --targets sample/demo_targets.npz
writes <watch>.watchpose.npz beside the input: the pose, and the blade direction from imu.blade_video's
fitted model (sample/blade.npz, beside the weights). With --targets (video poses on the same watch-time grid), it
prints the error, the lunges and thrusts found from the watch against the video's, and the blade's image-angle
error against the blade detected in the video (frames where it was found), next to a blade along the forearm.
Without --targets too, it prints each lunge and thrust found from the watch: when, onset to peak, and depth or arm speed.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import model.train as train
from analysis.moves import WATCH_LUNGE_PEAK, events, label, match, stance
from imu.blade_video import image_angle, watch_blade
from imu.load import load_sensor_logger
from imu.synth import ELBOW_KP, WRIST_KP
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
    blade = watch_blade(np.load(Path(a.weights).with_name("blade.npz")), a.watch, t, pred)
    np.savez(out, t=t, pose=pred, blade=blade)
    print(out)

    watch = events(pred, label(pred), WATCH_LUNGE_PEAK)
    st = stance(pred)
    print("\nmove     at      onset to peak")
    for e in watch:
        m, s = divmod((t[e["onset"]] - t[0]), 60)
        extra = f"stance {st[e['peak']] / np.nanmedian(st):.1f}x guard" if e["kind"] == "lunge" else f"arm {e['speed']:.1f} m/s"
        print(f"{'lunge' if e['kind'] == 'lunge' else 'thrust':<8} {m:.0f}:{s:04.1f}  {e['duration'] * 1000:4.0f} ms   {extra}")
    for kind, name in (("lunge", "lunges"), ("extension", "thrusts")):
        d = [e["duration"] * 1000 for e in watch if e["kind"] == kind]
        if d:
            print(f"{len(d)} {name}: median {np.median(d):.0f} ms onset to peak")
    print()
    if tg is None:
        return

    true, valid = tg["pose"], tg["valid"]
    mean = np.broadcast_to(ck["mean_pose"].numpy(), true.shape)
    for name, pr in (("watch model", pred), ("average pose", mean)):
        print(f"{name:<14}MPJPE mm  " + "".join(f"{g}: {train.mpjpe(pr, true, valid, j):.0f}  " for g, j in train.GROUPS.items()))
    video = events(true, label(true))
    pairs = match(video, watch)
    for kind, name in (("lunge", "lunges"), ("extension", "thrusts")):
        pk = [(e, f) for e, f in pairs if e["kind"] == kind]
        onset = np.median([abs(f["onset"] - e["onset"]) for e, f in pk]) * 1000 / RATE if pk else np.nan
        false = sum(f["kind"] == kind for f in watch) - len(pk)
        print(f"{name}: watch found {len(pk)} of the video's {sum(e['kind'] == kind for e in video)} ({false} false), "
              f"median onset error {onset:.0f} ms")


    ok, K, grip = tg["blade_ok"], tg["K"], tg["grip"][tg["blade_ok"]]
    fore = pred[:, JOINTS.index(WRIST_KP["right"])] - pred[:, JOINTS.index(ELBOW_KP["right"])]
    errs = []
    for axis in (blade, fore / np.linalg.norm(fore, axis=1, keepdims=True)):
        # body frame -> camera is R.T; R is the video's heading frame, used here only to score.
        e = np.degrees(np.abs(np.angle(np.exp(1j * (image_angle(K, grip, axis[ok] @ tg["R"]) - tg["blade_angle"][ok])))))
        errs.append(f"{np.median(e):.1f} median, {np.percentile(e, 90):.1f} p90")
    print(f"blade image-angle error deg over {ok.sum()} frames: watch {errs[0]}; along the forearm {errs[1]}")


if __name__ == "__main__":
    main()
