"""Which deliberate faults the watch-only replay shows, block by block against a block of normal lunges.

    uv run python -m analysis.faults <watch> <targets.npz> <watchpose.npz> [--blocks 151.2,178.0,...]
    uv run python -m analysis.faults <watch>          # only prints the separators found
The take holds BLOCKS in order, each preceded by a separator: the weapon arm hanging straight down, still,
for 4 s. No fencing posture holds the forearm vertical and still that long, so the watch alone finds the
blocks. Reps are the video's stance peaks; each feature is then read at those moments from the video pose
and from the watch-only pose, and a fault counts as shown when the watch moves the fault's own feature the
same way as the video does.
"""

import argparse
import warnings

import numpy as np
from scipy.signal import find_peaks

from analysis.moves import GUARD_WIDTH, elbow, hand_speed, stance, video_hand_speed
from imu.blade_video import image_angle
from imu.load import load_sensor_logger
from model.pairs import FOREARM_W, JOINTS, RATE
from model.train import mpjpe

# Block name -> the feature that defines the fault, and the sign of its change against normal lunges.
BLOCKS = {"normal": None, "late arm": ("arm leads foot ms", -1), "bent arm": ("elbow at finish deg", -1),
          "dropped point": ("blade elevation deg", -1), "short lunge": ("depth x guard", -1),
          "lean forward": ("torso lean deg", 1)}
HOLD, DOWN_DEG, STILL = 2.5, 40.0, 1.0  # separator: this many seconds within DOWN_DEG of vertical under STILL rad/s
L_SH, R_SH = JOINTS.index(5), JOINTS.index(6)


def separators(df):
    """(start, end) watch seconds of each arm-down hold."""
    t = df.index.to_numpy()
    g = df[["grav_x", "grav_y", "grav_z"]].to_numpy()
    m = ((g @ FOREARM_W / np.linalg.norm(g, axis=1) > np.cos(np.radians(DOWN_DEG)))
         & (np.linalg.norm(df[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(), axis=1) < STILL)).astype(int)
    a, b = np.flatnonzero(np.diff(np.r_[0, m]) == 1), np.flatnonzero(np.diff(np.r_[m, 0]) == -1)
    return [(t[i], t[j]) for i, j in zip(a, b) if t[j] - t[i] >= HOLD]


def _cross(c, lo, hi, level):
    """Sub-frame index of the last upward crossing of level in c[lo:hi]."""
    for i in range(hi - 1, lo, -1):
        if c[i - 1] < level <= c[i]:
            return i - 1 + (level - c[i - 1]) / (c[i] - c[i - 1])
    return np.nan


def features(pose, peaks, guard):
    """Per rep (a stance peak, frames at RATE) from one pose sequence. guard: its en-garde stance width."""
    st, el = stance(pose), elbow(pose)
    sh = (pose[:, L_SH] + pose[:, R_SH]) / 2  # pelvis-relative, y up
    lean = np.degrees(np.arctan2(np.hypot(sh[:, 0], sh[:, 2]), sh[:, 1]))
    out = {k: [] for k in ("depth x guard", "elbow at finish deg", "arm leads foot ms", "torso lean deg")}
    for p in peaks:
        lo, hi = max(p - int(RATE), 0), min(p + int(0.6 * RATE), len(st) - 1)
        q = max(p - 6, 0) + int(np.nanargmax(st[max(p - 6, 0):p + 7]))  # this sequence's own peak near the video's
        out["depth x guard"].append(st[q] / guard)
        out["elbow at finish deg"].append(np.nanmean(el[q - 2:q + 3]))
        foot = _cross(st, lo, q + 1, (np.nanmin(st[lo:q + 1]) + st[q]) / 2)
        k = lo + int(np.nanargmin(el[lo:q + 1]))
        arm = _cross(el, k, hi + 1, (el[k] + np.nanmax(el[k:hi + 1])) / 2)
        out["arm leads foot ms"].append((foot - arm) * 1000 / RATE)
        out["torso lean deg"].append(np.nanmean(lean[q - 2:q + 3]))
    return {k: np.array(v) for k, v in out.items()}


def elevation(angle):
    """A blade image angle as degrees above the image's horizontal."""
    return np.degrees(np.arcsin(-np.sin(angle)))


def reps(st, guard, a, b):
    """Stance peaks in frames [a, b): every lunge, short ones included."""
    p, _ = find_peaks(np.nan_to_num(st[a:b], nan=guard), prominence=0.2 * guard, distance=int(RATE))
    return p + a


def compare(df, tg, wp, blocks):
    """blocks: watch seconds where each of BLOCKS begins, in order (the last ends with the take)."""
    t, true, pred = tg["t"], tg["pose"], wp["pose"]
    gv, gw = np.nanmedian(stance(true)), np.nanmedian(stance(pred))
    edges = [int(np.searchsorted(t, s)) for s in blocks] + [len(t)]
    ok = tg["blade_ok"]
    video_el = np.where(ok, elevation(tg["blade_angle"]), np.nan)
    watch_el = elevation(image_angle(tg["K"], tg["grip"], wp["blade"] @ tg["R"]))
    res = {}
    for name, a, b in zip(BLOCKS, edges[:-1], edges[1:]):
        peaks = reps(stance(true), gv, a, b)
        ev = []
        for p in peaks:
            lo = max(p - int(1.5 * RATE), 0)
            narrow = np.flatnonzero(stance(true)[lo:p] <= GUARD_WIDTH * gv)
            ev.append(dict(onset=lo + int(narrow[-1]) if len(narrow) else lo, peak=int(p)))
        v, w = features(true, peaks, gv), features(pred, peaks, gw)
        fin = [slice(max(p - 2, 0), p + 4) for p in peaks]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a rep with no confident blade frame is NaN
            v["blade elevation deg"] = np.array([np.nanmedian(video_el[s]) for s in fin])
            w["blade elevation deg"] = np.array([np.nanmedian(np.where(ok[s], watch_el[s], np.nan)) for s in fin])
        v["hand speed m/s"], w["hand speed m/s"] = video_hand_speed(true, tg["root"], ev), hand_speed(df, t, ev)
        m = np.zeros(len(t), bool)
        m[a:b] = True
        res[name] = dict(n=len(peaks), video=v, watch=w, mpjpe=mpjpe(pred, true, m & tg["valid"], list(range(len(JOINTS)))))
    return res


def report(res):
    base = res["normal"]
    print(f"{'block':<14}{'reps':>5}{'MPJPE mm':>10}   its own feature: video change, watch change (share of the video's), shown?")
    print(f"{'normal':<14}{base['n']:5d}{base['mpjpe']:10.0f}")
    for name, (key, sign) in list(BLOCKS.items())[1:]:
        if name not in res or not res[name]["n"] or not base["n"]:
            print(f"{name:<14}  no reps found")
            continue
        r = res[name]
        dv = np.nanmedian(r["video"][key]) - np.nanmedian(base["video"][key])
        dw = np.nanmedian(r["watch"][key]) - np.nanmedian(base["watch"][key])
        sd = np.nanstd(base["video"][key], ddof=1) if base["n"] > 1 else np.nan
        done = np.sign(dv) == sign and abs(dv) > sd
        shown = "yes" if done and np.sign(dw) == sign and abs(dw) > 0.25 * abs(dv) else "no" if done else "fault not in the video"
        print(f"{name:<14}{r['n']:5d}{r['mpjpe']:10.0f}   {key}: video {dv:+.1f} (normal reps vary by {sd:.1f}), "
              f"watch {dw:+.1f} ({dw / dv:.0%}), {shown}")
    keys = list(base["video"])
    print("\nmedians per block, video / watch")
    print(f"{'':<14}" + "".join(f"{k:>24}" for k in keys))
    for name, r in res.items():
        if r["n"]:
            print(f"{name:<14}" + "".join(f"{np.nanmedian(r['video'][k]):>14.1f} /{np.nanmedian(r['watch'][k]):7.1f}" for k in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("watch")
    ap.add_argument("targets", nargs="?")
    ap.add_argument("watchpose", nargs="?")
    ap.add_argument("--blocks", help="watch seconds where each block starts, comma-separated, if the holds aren't found")
    a = ap.parse_args()
    df = load_sensor_logger(a.watch)
    seps = separators(df)
    print("arm-down holds at " + (", ".join(f"{s:.1f}-{e:.1f} s" for s, e in seps) or "none"))
    if not a.targets:
        return
    blocks = [float(x) for x in a.blocks.split(",")] if a.blocks else [e for _, e in seps]
    if len(blocks) != len(BLOCKS):
        raise SystemExit(f"{len(blocks)} block starts for {len(BLOCKS)} blocks ({', '.join(BLOCKS)}); pass --blocks")
    report(compare(df, np.load(a.targets), np.load(a.watchpose), blocks))


if __name__ == "__main__":
    main()
