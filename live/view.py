"""Replay a watch-only take in the browser: play, scrub, slow down, orbit.

    uv run python -m live.view sample/demo_watch.watchpose.npz [sample/demo_watch.zip]
writes <take>.html beside the npz (model.predict's output) and opens it: live/index.html with the take embedded,
so the file stands alone. Lunges and thrusts found from the pose are marked on the timeline; with the watch
recording, each carries its hand speed. If the npz has a pelvis track (root) the fencer travels along the piste,
and if it has the blade's offset from the wrist (grip) the blade starts there.
"""

import base64
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np

from analysis.moves import L_ANK, R_ANK, WATCH_LUNGE_PEAK, events, hand_speed, label
from imu.load import load_sensor_logger


def packed(x, scale):
    return base64.b64encode(np.round(np.nan_to_num(x) * scale).astype("<i2").tobytes()).decode()


def main():
    path = Path(sys.argv[1])
    d = np.load(path)
    pose, t = d["pose"], d["t"]
    ev = events(pose, label(pose), WATCH_LUNGE_PEAK)
    if len(sys.argv) > 2:
        for e, v in zip(ev, hand_speed(load_sensor_logger(sys.argv[2]), t, ev)):
            e["hand"] = round(float(v), 2)
    feet = np.nanmean((pose[:, R_ANK] - pose[:, L_ANK])[:, [0, 2]], axis=0)
    take = dict(name=path.name.removesuffix(".watchpose.npz"), n=len(t), fps=float(1 / np.median(np.diff(t))),
                pose=packed(pose, 1000), blade=packed(d["blade"], 10000),
                piste=(feet / np.linalg.norm(feet)).tolist(), floor=float(np.median(pose[..., 1].min(axis=1))),
                events=[dict(kind=e["kind"], onset=int(e["onset"]), peak=int(e["peak"]), end=int(e["end"]),
                             hand=e.get("hand")) for e in ev])
    if "root" in d:
        root = d["root"] - d["root"][0]
        take["root"] = packed(root * [1, 0, 1], 1000)
    if "grip" in d:
        take["grip"] = packed(d["grip"], 1000)
    page = Path(__file__).with_name("index.html").read_text()
    out = path.with_name(take["name"] + ".html")
    out.write_text(page.replace("<!--TAKE-->", f"<script>window.TAKE = {json.dumps(take)}</script>"))
    print(out)
    webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
