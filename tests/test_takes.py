"""uv run python -m tests.test_takes : training runs from the shipped pair files alone."""

import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from model.pairs import JOINTS
from model.train import Net


def test_train_from_shipped_pairs():
    with tempfile.TemporaryDirectory() as d:
        w = Path(d) / "weights.pt"
        out = subprocess.run([sys.executable, "-m", "model.train", "sample/takes/office1.npz", "sample/takes/home0.npz",
                              "--test", "sample/takes/office2.npz", "--epochs", "2", "--device", "cpu", "--save", str(w)],
                             capture_output=True, text=True, check=True).stdout
        assert "model" in out and "knn" in out
        ck = torch.load(w, map_location="cpu", weights_only=True)
        Net(18, len(JOINTS), ck["mean_pose"], ck["hidden"], ck["causal"]).load_state_dict(ck["nets"][0])


if __name__ == "__main__":
    test_train_from_shipped_pairs()
    print("ok")
