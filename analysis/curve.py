"""How much video the watch model needs: held-out error against minutes of training video.

    uv run python -m analysis.curve data/processed/curve
trains model.train on stretches of the home takes (recording order, then the office takes) and scores each on
the held-out demo take; writes curve.csv and curve.png there. One seed per run.
"""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from analysis.moves import WATCH_LUNGE_PEAK, events, label, match
from model.pairs import load

TAKES = [f"sample/takes/home{i}.npz" for i in range(6)] + ["sample/takes/office1.npz", "sample/takes/office2.npz"]
TEST = "sample/takes/demo.npz"
# (start minute, minutes): three different stretches per size where the takes allow it.
RUNS = [(s, m) for m in (1, 3, 5, 10) for s in (0, 10, 15) if s + m <= 21] + [(0, 22)]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    procs = []
    for s, m in RUNS:
        d = out / f"start{s}_min{m}"
        d.mkdir(exist_ok=True)
        procs.append((s, m, d, subprocess.Popen(
            [sys.executable, "-m", "model.train", *TAKES, "--test", TEST, "--span", f"{s},{m}", "--device", "cpu",
             "--out", str(d)], stdout=open(d / "train.log", "w"), stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "1"})))
    p = load(TEST)
    video = events(p["pose"], label(p["pose"]))
    rows = []
    for s, m, d, proc in procs:
        proc.wait()
        log = (d / "train.log").read_text()
        err = {k: [int(v) for v in re.search(rf"^{k}\s+(\d+)\s+(\d+)\s+(\d+)", log, re.M).groups()]
               for k in ("mean pose", "knn", "model")}
        pred = np.load(d / "demo.watchpose.npz")["pose"]
        watch = events(pred, label(pred), WATCH_LUNGE_PEAK)
        pairs = match(video, watch)
        row = dict(start_min=s, minutes=min(m, 21.5), arm=err["model"][0], legs=err["model"][1], body=err["model"][2],
                   retrieval_body=err["knn"][2], average_pose_body=err["mean pose"][2])
        for kind, name in (("lunge", "lunges"), ("extension", "thrusts")):
            hit = sum(e["kind"] == kind for e, _ in pairs)
            row[name] = f"{hit}/{sum(e['kind'] == kind for e in video)}"
            row[f"false_{name}"] = sum(f["kind"] == kind for f in watch) - hit
        rows.append(row)
        print(row, flush=True)
    with open(out / "curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, c, lab in (("body", "C2", "watch model"), ("retrieval_body", "C1", "nearest-neighbour retrieval"),
                        ("average_pose_body", "0.5", "average pose")):
        ax.scatter([r["minutes"] for r in rows], [r[key] for r in rows], color=c, s=18, label=lab)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 5, 10, 21.5], ["1", "3", "5", "10", "21.5"])
    ax.set_xlabel("minutes of training video")
    ax.set_ylabel("held-out mean joint error, mm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "curve.png", dpi=130)


if __name__ == "__main__":
    main()
