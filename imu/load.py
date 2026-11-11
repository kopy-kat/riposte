import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

G = 9.80665

# Sensor Logger on watchOS writes one combined CSV (sensor "WristMotion"); column names from
# https://github.com/tszheichoi/awesome-sensor-logger/blob/main/UNITS.md
SOURCE = {
    **{f"acc_{a}": f"acceleration{a.upper()}" for a in "xyz"},
    **{f"gyr_{a}": f"rotationRate{a.upper()}" for a in "xyz"},
    **{f"grav_{a}": f"gravity{a.upper()}" for a in "xyz"},
    **{f"q{a}": f"quaternion{a.upper()}" for a in "wxyz"},
}


def _read_csvs(path):
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            return {Path(n).name: z.read(n) for n in z.namelist() if n.lower().endswith(".csv")}
    return {p.name: p.read_bytes() for p in path.rglob("*.csv")}


def load_sensor_logger(path) -> pd.DataFrame:
    """Uniformly resampled watch IMU, index = seconds_elapsed.

    Columns: acc_x/y/z (user acceleration, g), grav_x/y/z (g), gyr_x/y/z (rad/s),
    qw/qx/qy/qz when present. df.attrs["rate_hz"] holds the detected native rate.
    """
    files = _read_csvs(path)
    name = next((n for n, b in files.items() if b"rotationRateX" in b.split(b"\n", 1)[0]), None)
    if name is None:
        raise ValueError(f"no CSV with watch motion columns (rotationRateX) in {path}; found {sorted(files)}")
    raw = pd.read_csv(io.BytesIO(files[name]))
    raw = raw.drop_duplicates("seconds_elapsed").sort_values("seconds_elapsed")

    meta = pd.read_csv(io.BytesIO(files["Metadata.csv"])) if "Metadata.csv" in files else None
    standardised = meta is not None and str(meta.get("standardisation", pd.Series([False])).iloc[0]).lower() == "true"
    to_g = 1 / G if standardised else 1.0  # m/s^2 -> g when "Standardise Units & Frames" is on

    t = raw["seconds_elapsed"].to_numpy()
    rate = (len(t) - 1) / (t[-1] - t[0])
    grid = t[0] + np.arange(int((t[-1] - t[0]) * rate) + 1) / rate

    out = {}
    for col, src in SOURCE.items():
        if src not in raw:
            continue
        v = raw[src].to_numpy(float)
        if col.startswith(("acc_", "grav_")):
            v = v * to_g
        out[col] = np.interp(grid, t, v)
    df = pd.DataFrame(out, index=pd.Index(grid, name="t"))
    if "qw" in df:
        q = df[["qw", "qx", "qy", "qz"]].to_numpy()
        df[["qw", "qx", "qy", "qz"]] = q / np.linalg.norm(q, axis=1, keepdims=True)
    df.attrs["rate_hz"] = rate
    print(f"{name}: {len(df)} samples, {rate:.2f} Hz native, {grid[-1] - grid[0]:.1f} s, "
          f"accel in {'m/s^2 converted to g' if standardised else 'g'}")
    return df
