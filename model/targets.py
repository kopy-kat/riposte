"""Video targets for model.predict --targets, on the watch's time grid: what sample/demo_targets.npz holds.

    uv run python -m model.targets data/raw/demo/watch data/processed/demo_video30.sam3d.npz data/raw/demo/video30.mp4 out.npz
t, pose, valid, root, R come from model.pairs.pair; K, grip (the knuckle keypoint in the camera) and the blade's
image angle from imu.blade_video.detections, taken at the video frame nearest each t. blade_ok keeps the frames
whose detection is in the confident half.
"""

import argparse

import numpy as np

from imu.blade_video import detections, project
from imu.calib import KNUCKLE_KP
from imu.load import load_sensor_logger
from imu.sync import xcorr_offset
from imu.synth import synth
from model.pairs import pair


def wide_offset(watch, npz, hand="right"):
    """Watch time of video t=0 from |gyro| alone, searched over the whole take: needs no shared clock and no
    tap, so it serves a camera whose file carries no usable start time. pair() refines it."""
    df, sy = load_sensor_logger(watch), synth(npz, hand)
    w = np.linalg.norm(df[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(), axis=1)
    v = np.linalg.norm(sy[["gx", "gy", "gz"]].to_numpy(), axis=1)
    fps = 1 / (sy.index[1] - sy.index[0])
    span = max(len(w) / df.attrs["rate_hz"], len(v) / fps)
    # An event at video time t sits at watch time t + s, counted from each recording's first sample.
    return xcorr_offset(v, fps, w, df.attrs["rate_hz"], span) + df.index[0] - sy.index[0]


def build(watch, npz, video, offset=None, until=None, hand="right"):
    """until: drop samples after this watch time (s)."""
    p = pair(watch, npz, offset=wide_offset(watch, npz, hand) if offset is None else offset, hand=hand)
    d = np.load(npz)
    K = d["cam_int"].reshape(-1, 3, 3)[0]
    grip = d["pred_keypoints_3d"][:, KNUCKLE_KP[hand]].astype(float) + d["pred_cam_t"].astype(float)
    obs, conf = detections(npz, video, project(K, grip))
    i = np.clip(np.round((p["t"] - p["offset"]) * float(d["fps"])).astype(int), 0, len(grip) - 1)
    ok = (d["found"].astype(bool)[i] & np.isfinite(obs[i]) & np.isfinite(grip[i]).all(1)
          & (conf[i] > np.percentile(conf[conf > 0], 50)) & p["valid"])
    out = dict(t=p["t"], pose=p["pose"], valid=p["valid"], blade_ok=ok, blade_angle=obs[i], grip=grip[i], K=K,
               R=p["R"], root=p["root"])
    if until is not None:
        m = p["t"] <= until
        out = {k: v[m] if k not in ("K", "R") else v for k, v in out.items()}
    print(f"{npz}: offset {p['offset']:.3f} s, sync correlation {p['peak']:.2f}, {len(out['t']) / 30:.0f} s, "
          f"{out['valid'].mean():.0%} with a pose, blade seen in {out['blade_ok'].sum()} frames")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("watch")
    ap.add_argument("npz")
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--offset", type=float, help="watch time of video t=0; default: found from |gyro|")
    ap.add_argument("--until", type=float)
    a = ap.parse_args()
    np.savez(a.out, **build(a.watch, a.npz, a.video, a.offset, a.until))
    print(a.out)
