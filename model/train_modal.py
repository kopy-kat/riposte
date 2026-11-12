"""model.train on Modal, one container per run, all runs in parallel.

    uv run python -m model.train_modal [<test spec>]     # stage data/takes.txt (+ test take, printed) for upload
    modal run --detach model/train_modal.py --runs "base:;demo:--test <printed spec> --seeds 5"
    modal volume get riposte-cache /sweep data/processed/sweep   # once they finish
Runs are ;-separated name:extra args. Each run's log and --out predictions are written to the volume under
/sweep/<name>; detached, the runs survive the client's connection dropping.
"""

import shlex
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
STAGE = Path("/tmp/riposte_stage")
# The keys pair() and synth() read: the full SAM 3D npz is ~0.5 GB per take, these are a few MB.
NPZ_KEYS = ["fps", "found", "pred_keypoints_3d", "pred_cam_t", "pred_joint_coords", "pred_global_rots"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "pandas", "scipy", "scikit-learn")
    .add_local_python_source("model", "imu")
    .add_local_dir(STAGE, "/stage")
)
app = modal.App("riposte-train", image=image)
cache = modal.Volume.from_name("riposte-cache", create_if_missing=True)


@app.function(gpu=["L4", "A10G", "T4", "L40S"], timeout=6 * 3600, volumes={"/cache": cache})
def train(name: str, args: list) -> str:
    out = Path(f"/cache/sweep/{name}")
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["python", "-m", "model.train", *args, "--device", "cuda", "--out", str(out)], cwd="/stage",
                       capture_output=True, text=True, check=False)
    log = r.stdout + r.stderr
    (out / "train.log").write_text(log)
    cache.commit()
    return log


def stage(specs):
    """Slim npz copies plus the watch dirs under STAGE, and each spec with its clock offset filled in.

    clock_offset() reads the sibling video.mov, which is not uploaded; pair() treats a given offset the
    same way, as the seed it refines.
    """
    import shutil

    import numpy as np

    from imu.sync import clock_offset

    out = []
    for spec in specs:
        w, n, *off = spec.split(",")
        if not off:
            o = clock_offset(ROOT / w)
            off = [] if o is None else [f"{o:.3f}"]
        dst = STAGE / n
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            d = np.load(ROOT / n)
            np.savez(dst, **{k: d[k] for k in NPZ_KEYS})
        if not (STAGE / w).exists():
            shutil.copytree(ROOT / w, STAGE / w)
        out.append(",".join([w, n, *off]))
    return out


@app.local_entrypoint()
def main(runs: str):
    takes = (STAGE / "takes.txt").read_text().split()
    jobs = [(r.split(":", 1)[0], [*takes, *shlex.split(r.split(":", 1)[1])]) for r in runs.split(";")]
    for (name, _), log in zip(jobs, train.starmap(jobs)):
        print(name, next((l for l in log.splitlines() if l.startswith("model")), log[-500:]))


if __name__ == "__main__":
    import sys

    (STAGE / "takes.txt").write_text(" ".join(stage((ROOT / "data/takes.txt").read_text().split())))
    print(*stage(sys.argv[1:]))
