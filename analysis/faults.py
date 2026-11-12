"""Whether the watch-only replay shows deliberate faults, rep by rep against the video.

    uv run python -m analysis.faults <watch> <targets.npz> <watchpose.npz>
    uv run python -m analysis.faults <watch>          # only prints the separators found
The take holds stretches of three lunges in the order of KINDS, each stretch preceded by a separator: the
weapon arm hanging straight down, still, for 4 s. No fencing posture holds the forearm vertical and still
that long, so the watch alone finds the stretches. Reps are the video's stance peaks; each kind's own
feature is read at those moments from the video pose and from the watch-only pose.
"""

import argparse

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import spearmanr

from analysis.moves import SH, WR, elbow, stance
from imu.load import load_sensor_logger
from model.pairs import FOREARM_W, RATE

# Rep k of a stretch is KINDS[k]; a fault lowers its own feature against the normal lunge.
KINDS = {"normal": None, "late arm": "elbow at finish deg", "low arm": "wrist above shoulder cm"}
HOLD, DOWN_DEG, STILL = 2.5, 40.0, 1.0  # separator: this many seconds within DOWN_DEG of vertical under STILL rad/s


def separators(df):
    """(start, end) watch seconds of each arm-down hold."""
    t = df.index.to_numpy()
    g = df[["grav_x", "grav_y", "grav_z"]].to_numpy()
    m = ((g @ FOREARM_W / np.linalg.norm(g, axis=1) > np.cos(np.radians(DOWN_DEG)))
         & (np.linalg.norm(df[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(), axis=1) < STILL)).astype(int)
    a, b = np.flatnonzero(np.diff(np.r_[0, m]) == 1), np.flatnonzero(np.diff(np.r_[m, 0]) == -1)
    return [(t[i], t[j]) for i, j in zip(a, b) if t[j] - t[i] >= HOLD]


def reps(st, guard, a, b):
    """Stance peaks in frames [a, b): every lunge, short ones included."""
    p, _ = find_peaks(np.nan_to_num(st[a:b], nan=guard), prominence=0.2 * guard, distance=int(RATE))
    return p + a


def features(pose, peaks):
    """Per rep, around its stance peak."""
    el, up = elbow(pose), 100 * (pose[:, WR, 1] - pose[:, SH, 1])
    return {"elbow at finish deg": np.array([np.nanmean(el[p - 2:p + 3]) for p in peaks]),
            "wrist above shoulder cm": np.array([np.nanmean(up[p - 2:p + 3]) for p in peaks])}


def compare(df, tg, wp):
    """(kind of each rep, its features from the video pose, the same from the watch-only pose)."""
    t, true = tg["t"], tg["pose"]
    seps, guard = separators(df), np.nanmedian(stance(true))
    starts = [int(np.searchsorted(t, e)) for _, e in seps]
    ends = [int(np.searchsorted(t, s)) for s, _ in seps[1:]] + [len(t)]
    per = [reps(stance(true), guard, a, b)[:len(KINDS)] for a, b in zip(starts, ends)]
    peaks = np.concatenate(per).astype(int)
    return np.concatenate([np.arange(len(p)) for p in per]), features(true, peaks), features(wp["pose"], peaks)


def report(kind, video, watch):
    keys = list(video)
    print(f"{len(kind)} reps; medians per kind, video / watch")
    print(f"{'':<10}{'reps':>5}" + "".join(f"{k:>28}" for k in keys))
    for i, name in enumerate(KINDS):
        m = kind == i
        print(f"{name:<10}{m.sum():5d}" + "".join(f"{np.median(video[k][m]):>18.0f} /{np.median(watch[k][m]):7.0f}" for k in keys))
    print("watch vs video rank correlation per rep: " + ", ".join(f"{k} {spearmanr(video[k], watch[k])[0]:.2f}" for k in keys))
    for i, (name, key) in list(enumerate(KINDS.items()))[1:]:
        below = watch[key][kind == i] < watch[key][kind == 0].min()
        print(f"{name}: the watch puts {below.sum()} of {len(below)} reps below every normal rep's {key}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("watch")
    ap.add_argument("targets", nargs="?")
    ap.add_argument("watchpose", nargs="?")
    a = ap.parse_args()
    df = load_sensor_logger(a.watch)
    print("arm-down holds at " + (", ".join(f"{s:.1f}-{e:.1f} s" for s, e in separators(df)) or "none"))
    if a.targets:
        report(*compare(df, np.load(a.targets), np.load(a.watchpose)))


if __name__ == "__main__":
    main()
