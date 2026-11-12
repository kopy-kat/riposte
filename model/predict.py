"""Watch-only pose from a Sensor Logger recording and weights from model.train --save; no video needed.

    uv run python -m model.predict sample/demo_watch.zip sample/weights.pt --targets sample/demo_targets.npz
writes <watch>.watchpose.npz beside the input: the pose, the blade's direction and where it leaves the hand
(grip, an offset from the predicted wrist) from imu.blade_video's fitted model (sample/blade.npz, beside the
weights), and hand, the hand speed of each move analysis.moves.events finds in that pose, in its order. With --targets (video poses on the same watch-time grid), it
prints the error, the lunges and thrusts found from the watch against the video's, how well each per-rep number
agrees with the video's, and the blade's image-angle error against the blade detected in the video (frames where
it was found), next to a blade along the forearm.
Without --targets too, it prints each lunge and thrust found from the watch with its hand speed: the one per-rep
number that orders reps as the video does. Onset to peak and lunge depth do not, so they are given as medians.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import model.train as train
from analysis.moves import WATCH_LUNGE_PEAK, agreement, events, hand_speed, label, match, stance, video_hand_speed
from imu.blade_video import image_angle, watch_blade, watch_grip
from imu.load import load_sensor_logger
from imu.synth import ELBOW_KP, WRIST_KP
from model.pairs import JOINTS, RATE, model_input
from model.piste import velocity


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
    fit = np.load(Path(a.weights).with_name("blade.npz"))
    blade = watch_blade(fit, a.watch, t, pred)
    watch = events(pred, label(pred), WATCH_LUNGE_PEAK)
    hand = hand_speed(df, t, watch)
    # Where the fencer moves: pelvis velocity (m/s) in the watch's levelled frame, if model.piste's weights are there.
    pk = Path(a.weights).with_name("piste.pt")
    vel = None
    if pk.exists():
        pk = torch.load(pk, map_location="cpu", weights_only=True)
        xv = ((model_input(df, t) - pk["mu"].numpy()) / pk["sd"].numpy()).astype(np.float32)
        vel = []
        for sd in pk["nets"]:
            net = train.Net(xv.shape[1], 1, np.zeros((1, 3), np.float32), pk["hidden"])
            net.load_state_dict(sd)
            vel.append(train.predict(net.eval(), xv, 0, len(t), "cpu")[:, 0])
        vel = np.mean(vel, axis=0)
    np.savez(out, t=t, pose=pred, blade=blade, grip=watch_grip(fit, a.watch, t, pred), hand=hand,
             **({} if vel is None else {"pelvis_velocity": vel}))
    print(out)

    st = stance(pred)
    for e, v in zip(watch, hand):
        e["hand"], e["depth"] = v, st[e["peak"]] / np.nanmedian(st)
    print("\nmove     at      hand speed")
    for e in watch:
        m, s = divmod((t[e["onset"]] - t[0]), 60)
        print(f"{'lunge' if e['kind'] == 'lunge' else 'thrust':<8} {m:.0f}:{s:04.1f}  {e['hand']:.1f} m/s")
    for kind, name in (("lunge", "lunges"), ("extension", "thrusts")):
        ev = [e for e in watch if e["kind"] == kind]
        if ev:
            depth = f", stance {np.median([e['depth'] for e in ev]):.1f}x guard" if kind == "lunge" else ""
            print(f"{len(ev)} {name}: median {np.median([e['duration'] for e in ev]) * 1000:.0f} ms onset to peak{depth}, "
                  f"hand {np.median([e['hand'] for e in ev]):.1f} m/s")
    print()
    if tg is None:
        return

    true, valid = tg["pose"], tg["valid"]
    mean = np.broadcast_to(ck["mean_pose"].numpy(), true.shape)
    for name, pr in (("watch model", pred), ("average pose", mean)):
        print(f"{name:<14}MPJPE mm  " + "".join(f"{g}: {train.mpjpe(pr, true, valid, j):.0f}  " for g, j in train.GROUPS.items()))
    video = events(true, label(true))
    pairs = match(video, watch)
    stv = stance(true)
    for kind, name in (("lunge", "lunges"), ("extension", "thrusts")):
        pk = [(e, f) for e, f in pairs if e["kind"] == kind]
        onset = np.median([abs(f["onset"] - e["onset"]) for e, f in pk]) * 1000 / RATE if pk else np.nan
        false = sum(f["kind"] == kind for f in watch) - len(pk)
        print(f"{name}: watch found {len(pk)} of the video's {sum(e['kind'] == kind for e in video)} ({false} false), "
              f"median onset error {onset:.0f} ms")
    for kind, name in (("lunge", "lunge"), ("extension", "thrust")):
        pk = [(e, f) for e, f in pairs if e["kind"] == kind]
        if len(pk) < 3:
            continue
        rho, err, sd = agreement(video_hand_speed(true, tg["root"], [f for _, f in pk]), [f["hand"] for _, f in pk])
        print(f"hand speed per {name}, watch vs video over {len(pk)}: rank correlation {rho:.2f}, median error {err:.2f} m/s "
              f"(video rep-to-rep SD {sd:.2f})")
        rest = [("onset to peak", [e["duration"] for e, _ in pk], [f["duration"] for _, f in pk])]
        if kind == "lunge":
            rest.append(("depth", [stv[e["peak"]] / np.nanmedian(stv) for e, _ in pk], [f["depth"] for _, f in pk]))
        print(f"  medians only, rank correlation per {name}: " + ", ".join(f"{n} {agreement(a, b)[0]:.2f}" for n, a, b in rest))

    if vel is not None:
        # The video's pelvis is in its own heading frame (right, up, back); one turn about the vertical, fitted
        # here, lines the watch's frame up with it. The piste is the line the video's pelvis moves along.
        tv, okv = velocity(tg["root"], valid)
        tv, pv = tv[okv][:, [0, 2]] * [1, -1], vel[okv, :2]
        th = np.arctan2((pv[:, 0] * tv[:, 1] - pv[:, 1] * tv[:, 0]).sum(), (pv * tv).sum())
        pv = pv @ np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        axis = np.linalg.svd(tv - tv.mean(0), full_matrices=False)[2][0]
        print(f"pelvis velocity along the piste, watch vs video over {okv.sum() / RATE:.0f} s: correlation "
              f"{np.corrcoef(pv @ axis, tv @ axis)[0, 1]:.2f}, RMS error {np.sqrt(np.mean(((pv - tv) @ axis) ** 2)):.2f} m/s "
              f"(video SD {np.std(tv @ axis):.2f}); frames aligned by one yaw")

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
