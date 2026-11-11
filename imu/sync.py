import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags

from imu.load import load_sensor_logger


def find_tap(df, window_s=10.0) -> float:
    d = df[df.index < df.index[0] + window_s]
    mag = np.linalg.norm(d[["acc_x", "acc_y", "acc_z"]].to_numpy(), axis=1)
    k = int(np.argmax(np.abs(np.diff(mag))))
    return float(d.index[k + int(mag[k + 1] > mag[k])])  # the larger side of the biggest step is the impact


def xcorr_offset(sig_a, rate_a, sig_b, rate_b, max_lag_s) -> float:
    """Seconds s such that an event at t in a appears at t + s in b (b lags a for s > 0)."""
    rate = max(rate_a, rate_b)

    def prep(s, r):
        s = np.asarray(s, float)
        t = np.arange(int((len(s) - 1) / r * rate) + 1) / rate
        s = np.interp(t, np.arange(len(s)) / r, s)
        return (s - s.mean()) / s.std()

    a, b = prep(sig_a, rate_a), prep(sig_b, rate_b)
    c = correlate(b, a, mode="full")
    lags = correlation_lags(len(b), len(a))
    m = np.abs(lags) <= max_lag_s * rate
    c, lags = c[m], lags[m]
    i = int(np.argmax(c))
    shift = 0.0
    if 0 < i < len(c) - 1:  # parabolic peak refinement
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return float((lags[i] + shift) / rate)


def plot_take(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    df = load_sensor_logger(path)
    tap = find_tap(df)
    acc = np.linalg.norm(df[["acc_x", "acc_y", "acc_z"]], axis=1)
    gyr = np.linalg.norm(df[["gyr_x", "gyr_y", "gyr_z"]], axis=1)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    ax[0].plot(df.index, acc, lw=0.7)
    ax[0].set_ylabel("|accel| (g)")
    ax[1].plot(df.index, gyr, lw=0.7)
    ax[1].set_ylabel("|gyro| (rad/s)")
    ax[1].set_xlabel("seconds from recording start")
    for a in ax:
        a.axvline(tap, color="r", ls="--")
    ax[0].set_title(f"{path.name}: tap at {tap:.3f} s")
    out = path.parent / f"{path.stem}_sync.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"tap at {tap:.3f} s -> {out}")


def clock_offset(watch_dir):
    """Watch time of video t=0 from the two devices' wall clocks, or None without a sibling video.mov.

    The iPhone's creation_time has one-second resolution, so this only seeds the cross-correlation.
    """
    import json
    import subprocess
    from datetime import datetime

    video = Path(watch_dir).parent / "video.mov"
    if not video.exists():
        return None
    meta = pd.read_csv(Path(watch_dir) / "Metadata.csv")
    watch_start = meta["recording epoch time"].iloc[0] / 1000
    # The iPhone's com.apple.quicktime.creationdate is when capture started; creation_time moves when the
    # clip is edited in Photos.
    tags = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(video)],
                                     capture_output=True, check=True).stdout)["format"]["tags"]
    stamp = tags.get("com.apple.quicktime.creationdate", tags["creation_time"])
    video_start = datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
    return video_start - watch_start


if __name__ == "__main__":
    plot_take(sys.argv[1])
