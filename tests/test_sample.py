"""uv run python -m tests.test_sample"""


import subprocess
import sys

import numpy as np


def test_sample_reproduces_readme():
    out = subprocess.run([sys.executable, "-m", "model.predict", "sample/demo_watch.zip", "sample/weights.pt",
                          "--targets", "sample/demo_targets.npz"], capture_output=True, text=True, check=True).stdout
    assert "watch model   MPJPE mm  arm: 62  legs: 67  body: 61" in out
    assert "average pose  MPJPE mm  arm: 104  legs: 114  body: 106" in out
    assert "blade image-angle error deg over 3585 frames: watch 4.1 median, 11.2 p90" in out
    d = np.load("sample/demo_watch.watchpose.npz")
    assert d["blade"].shape == (7325, 3) and np.allclose(np.linalg.norm(d["blade"], axis=1), 1)
    # The blade leaves the hand a few centimetres from the wrist, the same distance every frame.
    assert d["grip"].shape == (7325, 3) and np.allclose(np.linalg.norm(d["grip"], axis=1), 0.074, atol=0.002)
    assert len(d["hand"]) == 40
    assert "found 13 of the video's 13 (3 false)" in out and "found 20 of the video's 22 (4 false)" in out


if __name__ == "__main__":
    test_sample_reproduces_readme()
    print("ok")
