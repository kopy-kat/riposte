"""One recorded session, two cameras and the watch, to its numbers with the frozen weights.

    uv run python scripts/session.py <take> --phone phone.mov --mac mac.mov --watch export.zip
copies the files to data/raw/<take>/, runs SAM 3D Body on Modal for both videos (in parallel), and prints,
from sample/weights.pt: the pose error and the moves found before the fault blocks (phone view), the blade's
image-angle error in each view and its 3D error from both, and which deliberate faults the watch-only replay
shows (analysis.faults). Outputs go to data/processed/<take>_session/. Without the file options it reuses
what is already in data/raw/<take>/ and data/processed/, so
    uv run python scripts/session.py demo
runs the phone-view half on last night's held-out take with no Modal call.
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.faults import BLOCKS, separators  # noqa: E402
from imu.load import load_sensor_logger  # noqa: E402

WEIGHTS = "sample/weights.pt"


def run(*cmd):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    ap.add_argument("--phone")
    ap.add_argument("--mac")
    ap.add_argument("--watch", help="Sensor Logger export (.zip), or the folder live.server saved")
    ap.add_argument("--mac-flip", action="store_true", help="the Mac file is mirrored: flip it before the pose run")
    a = ap.parse_args()
    raw, out = ROOT / "data/raw" / a.take, ROOT / "data/processed" / f"{a.take}_session"
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    if a.watch:
        shutil.rmtree(raw / "watch", ignore_errors=True)
        if zipfile.is_zipfile(a.watch):
            zipfile.ZipFile(a.watch).extractall(raw / "watch")
        else:
            shutil.copytree(a.watch, raw / "watch")
    # 30 fps on a constant grid whatever the camera wrote, rotation applied, as SAM 3D reads raw frames.
    views = {}
    for view, src, stem in (("phone", a.phone, "video30"), ("mac", a.mac, "mac30")):
        mp4 = raw / f"{stem}.mp4"
        if src:
            flip = "hflip," if view == "mac" and a.mac_flip else ""
            run("ffmpeg", "-v", "error", "-y", "-i", src, "-vf", f"{flip}fps=30000/1001", "-an", "-c:v", "libx264",
                "-crf", "18", mp4)
        if mp4.exists():
            views[view] = (mp4, ROOT / "data/processed" / f"{a.take}_{stem}.sam3d.npz")
    todo = [mp4 for mp4, npz in views.values() if not npz.exists()]
    jobs = [subprocess.Popen(["modal", "run", "pose/sam3d_modal.py", "--video", str(mp4)], cwd=ROOT) for mp4 in todo]
    if any(j.wait() for j in jobs):
        raise SystemExit("a pose run failed; rerun this command to retry only the missing one")

    watch = raw / "watch"
    seps = separators(load_sensor_logger(watch))
    until = seps[0][0] if seps else None
    print(f"\n{len(seps)} arm-down holds" + (f", the first at {until:.1f} s: scoring normal fencing up to there" if seps else
                                              ": scoring the whole take as normal fencing"))

    for view, (mp4, npz) in views.items():
        for part, cut in (("normal", until), ("full", None)):
            if part == "full" and until is None:
                continue
            tg = out / f"{view}_{part}.targets.npz"
            run("uv", "run", "python", "-m", "model.targets", watch, npz, mp4, tg, *(["--until", f"{cut}"] if cut else []))
            print(f"\n=== {view} view, {part}: frozen weights against this camera ===", flush=True)
            run("uv", "run", "python", "-m", "model.predict", watch, WEIGHTS, "--targets", tg)
            shutil.move(watch.with_suffix(".watchpose.npz"), out / f"{view}_{part}.watchpose.npz")

    if len(views) == 2:
        print("\n=== blade in 3D from both cameras ===", flush=True)
        run("uv", "run", "python", "-m", "analysis.blade3d", out / "phone_normal.targets.npz", out / "mac_normal.targets.npz",
            out / "phone_normal.watchpose.npz")
    if len(seps) == len(BLOCKS):
        print("\n=== deliberate faults: does the watch-only replay show them? ===", flush=True)
        run("uv", "run", "python", "-m", "analysis.faults", watch, out / "phone_full.targets.npz", out / "phone_full.watchpose.npz")
    elif seps:
        print(f"\n{len(seps)} holds found for {len(BLOCKS)} blocks: run analysis.faults with --blocks (see its docstring)")


if __name__ == "__main__":
    main()
