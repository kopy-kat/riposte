"""Can the watch alone tell the office2 action blocks apart? 1 s windows, split by time within each block."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imu.load import load_sensor_logger

df = load_sensor_logger("data/raw/office2/watch")
blocks = pd.read_csv("data/raw/office2/labels.csv")

rate = df.attrs["rate_hz"]
win, hop = int(rate), int(rate / 4)
acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy()
gyr = df[["gyr_x", "gyr_y", "gyr_z"]].to_numpy()
grav = df[["grav_x", "grav_y", "grav_z"]].to_numpy()
t = df.index.to_numpy()


def features(s):
    a, g = acc[s], gyr[s]
    am, gm = np.linalg.norm(a, axis=1), np.linalg.norm(g, axis=1)
    return np.concatenate([[am.mean(), am.std(), am.max(), gm.mean(), gm.std(), gm.max()],
                           a.std(0), g.std(0), grav[s].mean(0)])


X, y, split = [], [], []
for _, b in blocks.iterrows():
    starts = [i for i in range(0, len(t) - win, hop) if t[i] >= b.start_s and t[i + win] <= b.end_s]
    # Last 40% of each block is held out; windows overlapping the boundary are dropped.
    cut = starts[int(len(starts) * 0.6)] if starts else 0
    for i in starts:
        if i < cut and i + win > cut:
            continue
        X.append(features(slice(i, i + win)))
        y.append(b.label)
        split.append("test" if i >= cut else "train")

X, y, split = np.array(X), np.array(y), np.array(split)
clf = RandomForestClassifier(300, random_state=0).fit(X[split == "train"], y[split == "train"])
pred = clf.predict(X[split == "test"])
labels = sorted(set(y))
print(classification_report(y[split == "test"], pred, labels=labels, zero_division=0))
print(pd.DataFrame(confusion_matrix(y[split == "test"], pred, labels=labels), index=labels, columns=labels))
