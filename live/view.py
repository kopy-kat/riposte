"""Replay a watch-only take in the browser: play, scrub, slow down, orbit.

    uv run python -m live.view sample/demo_watch.watchpose.npz
writes <take>.html beside the npz (model.predict's output) and opens it: live/index.html with the take embedded,
so the file stands alone. Lunges and thrusts found from the pose are marked on the timeline with their hand
speed, and the blade starts at its offset from the wrist (grip). If the npz has model.predict's pelvis_velocity
the fencer travels along the piste on model.piste.travel's short-memory track.
"""

import base64
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from analysis.moves import L_ANK, R_ANK, WATCH_LUNGE_PEAK, events, label


def packed(x, scale):
    return base64.b64encode(np.round(np.nan_to_num(x) * scale).astype("<i2").tobytes()).decode()


def rate(d):
    return float(1 / np.median(np.diff(d["t"])))


def moves(d):
    """Lunges and thrusts found from the watch pose, each with its hand speed from model.predict."""
    ev = events(d["pose"], label(d["pose"]), WATCH_LUNGE_PEAK)
    for e, v in zip(ev, d["hand"]):
        e["hand"] = round(float(v), 2)
    return ev


def write(path):
    d = np.load(path)
    pose, t = d["pose"], d["t"]
    feet = np.nanmean((pose[:, R_ANK] - pose[:, L_ANK])[:, [0, 2]], axis=0)
    # The pose hangs from the pelvis, so the feet rise and sink with every lunge: the page stands them on the floor.
    ground = savgol_filter(np.nan_to_num(pose[..., 1]).min(axis=1), 9, 2)
    take = dict(name=path.name.removesuffix(".watchpose.npz"), n=len(t), fps=rate(d),
                pose=packed(pose, 1000), blade=packed(d["blade"], 10000),
                piste=(feet / np.linalg.norm(feet)).tolist(), floor=float(np.median(ground)), ground=packed(ground, 1000),
                events=[dict(kind=e["kind"], onset=int(e["onset"]), peak=int(e["peak"]), end=int(e["end"]),
                             hand=e["hand"]) for e in moves(d)], grip=packed(d["grip"], 1000))
    if "pelvis_velocity" in d:
        from model.piste import travel

        take["root"] = packed(travel(pose, d["pelvis_velocity"]), 1000)
    page = Path(__file__).with_name("index.html").read_text()
    out = path.with_name(take["name"] + ".html")
    out.write_text(page.replace("<!--TAKE-->", f"<script>window.TAKE = {json.dumps(take)}</script>"))
    return out


def main():
    out = write(Path(sys.argv[1]))
    print(out)
    webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
