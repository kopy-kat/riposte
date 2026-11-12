"""Moves, their timing and arm speed, from a pose sequence: the video's, or the watch-only prediction.

    uv run python -m analysis.moves data/raw/home1/watch,data/processed/home1_video30.sam3d.npz
reads the matching <take>.watchpose.npz that model.train --out wrote, prints the watch-only error per
move (moves labelled from the video) and how well watch-only events match the video's.
"""

import sys

import numpy as np

from model.pairs import JOINTS, RATE, pair
from model.train import GROUPS, mpjpe

L_ANK, R_ANK = JOINTS.index(13), JOINTS.index(14)
SH, EL, WR = (JOINTS.index(j) for j in (6, 8, 41))
LUNGE, STRAIGHT = 1.4, 150  # stance width over the guard median; elbow angle in degrees
# A lunge run must peak at this stance ratio: below ~1.9 the training takes' video shows a wide guard, not a lunge.
LUNGE_PEAK, GUARD_WIDTH = 1.9, 1.1
# The watch-only pose under-reaches lunge depth; this peak balances recall and precision against LUNGE_PEAK
# video lunges, cross-validated on the training takes.
WATCH_LUNGE_PEAK = 1.8
MATCH = int(0.5 * RATE)
GAP = int(0.3 * RATE)  # runs this close are one move: the thresholds flicker at their edges


def stance(pose):
    return np.linalg.norm((pose[:, L_ANK] - pose[:, R_ANK])[:, [0, 2]], axis=1)


def elbow(pose):
    u, v = pose[:, SH] - pose[:, EL], pose[:, WR] - pose[:, EL]
    return np.degrees(np.arccos(np.clip((u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)), -1, 1)))


def arm_speed(pose):
    """Wrist speed relative to the pelvis, m/s: the pose is pelvis-relative, so body travel is left out."""
    return np.linalg.norm(np.gradient(pose[:, WR], axis=0), axis=1) * RATE


def label(pose, root=None):
    """lunge > extension > step > guard, per frame. Steps need the pelvis track (root), which the watch can't see."""
    st, el = stance(pose), elbow(pose)
    out = np.full(len(pose), "guard", dtype=object)
    if root is not None:
        out[np.linalg.norm(np.gradient(root[:, [0, 2]], axis=0), axis=1) * RATE > 0.3] = "step"
    out[el > STRAIGHT] = "extension"
    out[st > LUNGE * np.nanmedian(st)] = "lunge"
    return out


def events(pose, moves, lunge_peak=LUNGE_PEAK):
    """Each run of lunge or extension frames as a dict, in frames at RATE.

    onset: where the movement begins, the last frame at guard width before the peak (lunge) or the most bent
    elbow in the second before the run (thrust); duration: onset to the peak (widest stance, straightest elbow).
    lunge_peak: the stance ratio a lunge run must reach; WATCH_LUNGE_PEAK for a watch-only pose.
    """
    st, el, sp = stance(pose), elbow(pose), arm_speed(pose)
    guard = np.nanmedian(st)
    out = []
    for kind, curve in (("lunge", st), ("extension", el)):
        m = (moves == kind).astype(int)
        on = np.flatnonzero(m)
        for a, b in zip(on[:-1], on[1:]):
            if 1 < b - a <= GAP:
                m[a:b] = 1
        starts = np.flatnonzero(np.diff(np.r_[0, m]) == 1)
        ends = np.flatnonzero(np.diff(np.r_[m, 0]) == -1) + 1
        for a, b in zip(starts, ends):
            peak = a + int(np.nanargmax(curve[a:b]))
            lo = max(a - int(RATE), 0)
            if kind == "lunge":
                if st[peak] < lunge_peak * guard:
                    continue
                lo = max(peak - int(1.5 * RATE), 0)
                narrow = np.flatnonzero(st[lo:peak] <= GUARD_WIDTH * guard)
                onset = lo + int(narrow[-1]) if len(narrow) else lo
            else:
                onset = lo + int(np.nanargmin(curve[lo:a + 1]))
            out.append(dict(kind=kind, start=int(a), end=int(b), onset=onset, peak=peak, duration=(peak - onset) / RATE,
                            speed=float(np.nanmax(sp[onset:peak + 1]))))
    return sorted(out, key=lambda e: e["start"])


def match(truth, pred):
    """(video event, watch event) pairs of the same kind whose starts are within MATCH frames, nearest first."""
    pairs, used = [], set()
    for e in truth:
        c = [(abs(f["start"] - e["start"]), i) for i, f in enumerate(pred)
             if f["kind"] == e["kind"] and i not in used and abs(f["start"] - e["start"]) <= MATCH]
        if c:
            used.add(min(c)[1])
            pairs.append((e, pred[min(c)[1]]))
    return pairs


def main():
    w, n, *off = sys.argv[1].split(",")
    p = pair(w, n, offset=float(off[0]) if off else None)
    pred = np.load(n.replace(".sam3d.npz", ".watchpose.npz"))["pose"]
    moves = label(p["pose"], p["root"])
    ok = p["valid"] & np.isfinite(pred).all((1, 2))
    print(f"{'move':<10}{'share':>7}" + "".join(f"{g:>8}" for g in GROUPS))
    for m in ["guard", "step", "extension", "lunge"]:
        v = ok & (moves == m)
        if v.sum():
            print(f"{m:<10}{v.mean():7.0%}" + "".join(f"{mpjpe(pred, p['pose'], v, j):8.0f}" for j in GROUPS.values()))

    truth = events(p["pose"], label(p["pose"]))
    watch = events(pred, label(pred), WATCH_LUNGE_PEAK)
    pairs = match(truth, watch)
    print(f"\n{'event':<10}{'video':>6}{'watch':>6}{'matched':>8}   median |watch - video|: onset ms, duration ms, "
          "arm speed m/s")
    for kind in ("lunge", "extension"):
        pk = [(e, f) for e, f in pairs if e["kind"] == kind]
        err = lambda k, s=1: np.median([abs(f[k] - e[k]) * s for e, f in pk]) if pk else np.nan
        print(f"{kind:<10}{sum(e['kind'] == kind for e in truth):6d}{sum(e['kind'] == kind for e in watch):6d}"
              f"{len(pk):8d}   {err('onset', 1000 / RATE):.0f}, {err('duration', 1000):.0f}, {err('speed'):.2f}"
              f"   video median duration {1000 * np.median([e['duration'] for e, _ in pk]):.0f} ms")

if __name__ == "__main__":
    main()
