"""Watch samples in, poses out, one frame at a time: the live counterpart of model.pairs.model_input and
model.train.predict for a causal model.

    uv run python -m live.stream sample/demo_watch.zip data/processed/live/causal.pt --targets sample/demo_targets.npz
feeds a recording through in 0.1 s batches, as Sensor Logger pushes them, and prints the error against the
video's pose beside the offline (whole-take) error of the same weights.
"""

import argparse

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

import model.train as train
from imu.synth import ELBOW_KP, WRIST_KP
from model.pairs import FOREARM_W, JOINTS, RATE

KEEP_S = 4.0  # raw samples kept: the 2 s model window plus room for the low-pass to settle
# Seconds behind the newest sample that a frame is taken: the zero-phase low-pass needs samples after it.
MARGIN_S = 0.03
C = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # CoreMotion reference (z up) -> body axes (right, up, back)
# The heading mean forgets: walking to the piste after the recording starts otherwise leaves it 30-50 deg
# off for the first minute of fencing.
HEAD_TAU_S = 15.0
FORGET = np.exp(-1 / (RATE * HEAD_TAU_S))
WR, EL = JOINTS.index(WRIST_KP["right"]), JOINTS.index(ELBOW_KP["right"])


def load(weights):
    ck = torch.load(weights, map_location="cpu", weights_only=True)
    if not ck["causal"]:
        raise ValueError(f"{weights} is not a causal model (model.train --causal)")
    nets = []
    for sd in ck["nets"]:
        net = train.Net(len(ck["mu"]), len(JOINTS), ck["mean_pose"], ck["hidden"], True)
        net.load_state_dict(sd)
        nets.append(net.eval())
    return dict(nets=nets, mu=ck["mu"].numpy(), sd=ck["sd"].numpy(), win=ck["win"])


class Stream:
    """push(t, rows) -> [(t, pose (J, 3), blade (3,) or None)] for each new 1/RATE frame.

    t: sample times in seconds, any epoch; rows: (n, 13) gyro rad/s, user accel g, gravity g (the order of
    model.pairs.IMU_COLS), then the attitude quaternion w, x, y, z.
    The forearm heading that model.pairs.level_attitude averages over a whole take is a forgetting mean here.
    Fixing it from an opening en-garde hold instead costs 20-30 mm: the hold's heading is not the take's.
    blade_b: the fitted watch-frame blade direction (blade.npz "b"); the offline IMU-driven correction reads
    samples after the frame, so the live blade is the fixed direction only.
    """

    def __init__(self, model, blade_b=None, margin_s=MARGIN_S):
        self.m, self.blade_b, self.margin = model, blade_b, margin_s
        self.t, self.rows = np.empty(0), np.empty((0, 13))
        self.k = None  # next frame index: frame times are k / RATE
        self.imu, self.R = [], []  # the last model window of frames
        self.head = 0j  # sum of unit forearm headings in the reference frame
        self.yaw = 0j  # sum of unit (predicted forearm - watch forearm) headings in the body frame

    def push(self, t, rows):
        t, rows = np.asarray(t, float), np.asarray(rows, float)
        self.t, i = np.unique(np.r_[self.t, t], return_index=True)
        self.rows = np.r_[self.rows, rows][i]
        keep = self.t >= self.t[-1] - KEEP_S
        self.t, self.rows = self.t[keep], self.rows[keep]
        if len(self.t) < 50:
            return []
        if self.k is None:
            self.k = int(np.ceil(self.t[0] * RATE))
        last = int(np.floor((self.t[-1] - self.margin) * RATE))
        if last < self.k:
            return []
        tf = np.arange(self.k, last + 1) / RATE
        self.k = last + 1

        # As load_sensor_logger and imu_at: a uniform grid at the native rate, low-passed, read at frame times.
        rate = (len(self.t) - 1) / (self.t[-1] - self.t[0])
        grid = self.t[0] + np.arange(int((self.t[-1] - self.t[0]) * rate) + 1) / rate
        raw = np.stack([np.interp(grid, self.t, self.rows[:, c]) for c in range(9)], 1)
        raw = sosfiltfilt(butter(4, 0.4 * RATE, fs=rate, output="sos"), raw, axis=0)
        imu = np.stack([np.interp(tf, grid, raw[:, c]) for c in range(9)], 1)
        q = self.rows[:, 9:] / np.linalg.norm(self.rows[:, 9:], axis=1, keepdims=True)
        R = Slerp(self.t, Rotation.from_quat(q, scalar_first=True))(np.clip(tf, self.t[0], self.t[-1])).as_matrix()

        out = []
        for ti, x, Ri in zip(tf, imu, R):
            f = Ri @ FOREARM_W
            self.head = self.head * FORGET + np.exp(1j * np.arctan2(f[1], f[0]))
            self.imu = (self.imu + [x])[-self.m["win"]:]
            self.R = (self.R + [Ri])[-self.m["win"]:]
            pose = self._pose()
            out.append((ti, pose, self._blade(pose, Ri)))
        return out

    def _pose(self):
        # model.pairs.level_attitude with the heading known so far, over the whole window.
        c, s = np.cos(np.angle(self.head)), np.sin(np.angle(self.head))
        Rz = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        x = np.concatenate([np.array(self.imu), (Rz @ np.array(self.R)).reshape(len(self.R), 9)], 1)
        x = torch.as_tensor(((x - self.m["mu"]) / self.m["sd"]).astype(np.float32))[None]
        with torch.no_grad():
            return np.mean([net(x)[0, -1].numpy() for net in self.m["nets"]], axis=0)

    def _blade(self, pose, R):
        if self.blade_b is None:
            return None
        # As imu.blade_video.watch_blade: one heading turns the watch's forearm onto the predicted forearm.
        u, v = pose[WR] - pose[EL], C @ R @ FOREARM_W
        if np.hypot(u[0], u[2]) > 0.5 * np.linalg.norm(u) and np.hypot(v[0], v[2]) > 0.5:
            self.yaw += np.exp(1j * (np.arctan2(u[0], u[2]) - np.arctan2(v[0], v[2])))
        c, s = np.cos(np.angle(self.yaw)), np.sin(np.angle(self.yaw))
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ C @ R @ self.blade_b


def rows_of(df):
    from model.pairs import IMU_COLS

    return df[IMU_COLS + ["qw", "qx", "qy", "qz"]].to_numpy()


def main():
    from imu.load import load_sensor_logger
    from model.pairs import model_input

    ap = argparse.ArgumentParser()
    ap.add_argument("watch")
    ap.add_argument("weights")
    ap.add_argument("--targets", required=True)
    a = ap.parse_args()
    m = load(a.weights)
    train.WIN = m["win"]
    df, tg = load_sensor_logger(a.watch), np.load(a.targets)
    t, true, valid = tg["t"], tg["pose"], tg["valid"]

    x = ((model_input(df, t) - m["mu"]) / m["sd"]).astype(np.float32)
    offline = np.mean([train.predict(net, x, 0, len(t), "cpu") for net in m["nets"]], axis=0)

    s = Stream(m)
    tw, rows = df.index.to_numpy(), rows_of(df)
    got = {}
    for i in range(0, len(tw), 10):
        for ti, pose, _ in s.push(tw[i:i + 10], rows[i:i + 10]):
            got[int(round(ti * RATE))] = pose
    k = np.round(t * RATE).astype(int)
    have = np.array([i in got for i in k])
    live = np.stack([got.get(i, np.full_like(true[0], np.nan)) for i in k])
    # Skip the first window, where the live model has less than its 2 s of context.
    v = valid & have & (t >= t[0] + m["win"] / RATE)
    for name, pr in (("offline, whole take", offline), ("live, frame by frame", live)):
        print(f"{name:<22}MPJPE mm  " + "".join(f"{g}: {train.mpjpe(pr, true, v, j):.0f}  " for g, j in train.GROUPS.items()))


if __name__ == "__main__":
    main()
