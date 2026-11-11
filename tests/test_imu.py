import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from imu.load import load_sensor_logger
from imu.sync import find_tap, xcorr_offset

FIXTURE = Path("data/raw/fixture")
TAP_T, RATE = 3.217, 50.0


def make_fixture():
    rng = np.random.default_rng(0)
    n = int(30 * RATE)
    t = np.arange(n) / RATE + rng.uniform(-0.004, 0.004, n)  # jittered stamps
    t[0] = 0
    acc = rng.normal(0, 0.02, (n, 3))
    acc[np.argmin(abs(t - TAP_T))] += [0.2, 3.0, 0.5]
    acc[np.argmin(abs(t - 15)) : np.argmin(abs(t - 20))] += 0.5 * np.sin(2 * np.pi * 2 * t[:, None][np.argmin(abs(t - 15)) : np.argmin(abs(t - 20))])
    gyr = rng.normal(0, 0.05, (n, 3))
    gyr[:, 0] += 3 * np.exp(-((t - 17) ** 2) / 0.5) + 2 * np.exp(-((t - 24) ** 2) / 0.2)
    grav = np.tile([0, 0, -1.0], (n, 1))
    q = np.tile([1.0, 0, 0, 0], (n, 1))
    cols = {"time": (1.7e9 + t) * 1e9, "seconds_elapsed": t}
    for i, a in enumerate("XYZ"):
        cols[f"rotationRate{a}"] = gyr[:, i]
        cols[f"gravity{a}"] = grav[:, i]
        cols[f"acceleration{a}"] = acc[:, i]
    for i, a in enumerate("WXYZ"):
        cols[f"quaternion{a}"] = q[:, i]
    FIXTURE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(FIXTURE / "WristMotion.csv", index=False)
    pd.DataFrame({"device name": ["fixture"], "platform": ["ios"], "standardisation": [False]}).to_csv(
        FIXTURE / "Metadata.csv", index=False
    )
    with zipfile.ZipFile(FIXTURE.parent / "fixture.zip", "w") as z:
        for f in FIXTURE.glob("*.csv"):
            z.write(f, f"take/{f.name}")


def test_all():
    make_fixture()
    for src in (FIXTURE, FIXTURE.parent / "fixture.zip"):
        df = load_sensor_logger(src)
        assert abs(df.attrs["rate_hz"] - RATE) < 0.5
        assert abs(find_tap(df) - TAP_T) <= 1 / RATE, find_tap(df)

    gyr = np.linalg.norm(df[["gyr_x", "gyr_y", "gyr_z"]], axis=1)
    video_rate, true_off = 30.0, 0.347  # video lags watch by 0.347 s
    tv = np.arange(0, 30, 1 / video_rate)
    video = np.interp(tv - true_off, df.index, gyr) + np.random.default_rng(1).normal(0, 0.05, len(tv))
    off = xcorr_offset(gyr, df.attrs["rate_hz"], video, video_rate, max_lag_s=1.0)
    print(f"xcorr offset {off:.4f} s (true {true_off})")
    assert abs(off - true_off) < 0.01
    off_neg = xcorr_offset(video, video_rate, gyr, df.attrs["rate_hz"], max_lag_s=1.0)
    assert abs(off_neg + true_off) < 0.01


if __name__ == "__main__":
    test_all()
    print("ok")
