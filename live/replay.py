"""Push a Sensor Logger recording to live.server at real time, in Sensor Logger's HTTP push format.

    uv run python -m live.replay sample/demo_watch.zip [--url http://localhost:8000/data] [--seconds 60]
    uv run python -m live.replay sample/demo_watch.zip --score data/processed/live/<session>/stream.npz --targets sample/demo_targets.npz
The second form scores the poses the server streamed for that replay against the video's.
"""

import argparse
import io
import json
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from imu.load import SOURCE

BATCH_S = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--url", default="http://localhost:8000/data")
    ap.add_argument("--seconds", type=float, help="stop after this much of the recording")
    ap.add_argument("--score")
    ap.add_argument("--targets")
    a = ap.parse_args()
    with zipfile.ZipFile(a.zip) as z:
        raw = pd.read_csv(io.BytesIO(z.read("WristMotion.csv")))

    if a.score:
        import model.train as train
        from model.pairs import RATE

        s, tg = np.load(a.score), np.load(a.targets)
        # The stream's frames sit on a 1/RATE grid in epoch time, the targets on one in recording time.
        el = np.interp(s["t"], raw["time"].to_numpy() / 1e9, raw["seconds_elapsed"].to_numpy())
        i = np.clip(np.searchsorted(el, tg["t"]), 1, len(el) - 1)
        f = ((tg["t"] - el[i - 1]) / (el[i] - el[i - 1]))[:, None, None]
        pose = (1 - f) * s["pose"][i - 1] + f * s["pose"][i]
        v = tg["valid"] & (tg["t"] >= el[0] + 2) & (tg["t"] <= el[-1])
        print(f"{v.sum() / RATE:.0f} s scored   MPJPE mm  "
              + "".join(f"{g}: {train.mpjpe(pose, tg['pose'], v, j):.0f}  " for g, j in train.GROUPS.items()))
        return

    cols = list(SOURCE.values())
    t = raw["time"].to_numpy()
    rows = raw[cols].to_numpy()
    edges = np.searchsorted(t, np.arange(t[0], t[-1], BATCH_S * 1e9))
    start, sid = time.monotonic(), f"replay-{int(time.time())}"
    for mid, (i, j) in enumerate(zip(edges, np.r_[edges[1:], len(t)])):
        if a.seconds and (t[i] - t[0]) / 1e9 > a.seconds:
            break
        time.sleep(max(start + (t[j - 1] - t[0]) / 1e9 - time.monotonic(), 0))
        body = dict(messageId=mid, sessionId=sid, deviceId="replay",
                    payload=[dict(name="wrist motion", time=int(t[k]), values=dict(zip(cols, rows[k]))) for k in range(i, j)])
        req = urllib.request.Request(a.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        urllib.request.urlopen(req).read()


if __name__ == "__main__":
    main()
