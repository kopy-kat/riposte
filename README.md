# riposte

Watching yourself is the quickest way to spot a fault, but a camera often can't follow you, and a salle rarely has room for one that sees the whole piste. riposte is the replay without the camera: from one Apple Watch on the weapon wrist it reconstructs the whole fencer in 3D, points the épée, and times each lunge and thrust. You film yourself once to teach it; after that, the watch is enough.

Single-wrist golf systems ([Lauer 2025](https://arxiv.org/abs/2506.17505), [Tan et al. 2026](https://arxiv.org/abs/2606.22876)) reconstruct the swing but not the club. [IMUPoser](https://arxiv.org/abs/2304.12518), [MobilePoser](https://arxiv.org/abs/2504.12492) and [ArmPoser](https://arxiv.org/abs/2609.08806) estimate everyday pose. riposte recovers the blade and learns one fencer's own movement.

[riposte.mp4](riposte.mp4): the watch-only pose, then against the pose from video the model never saw.

## How it works

1. Sensor Logger records the watch's IMU at 100 Hz while an iPhone films side-on (`imu/load.py`).
2. A wrist tap syncs the streams (`imu/sync.py`, `model/pairs.py`).
3. SAM 3D Body on Modal GPUs turns each video frame into the 3D training target (`pose/sam3d_modal.py`).
4. A bidirectional GRU maps 2 s windows of wrist IMU and attitude to 30 joints relative to the pelvis. It trains on 21 minutes of my fencing with strap-rotation augmentation (`model/train.py`).
5. The blade's direction and its offset from the wrist come from the watch's attitude, fitted to the training video's blade (`imu/blade_video.py`). Moves and their timing are read off the predicted pose (`analysis/moves.py`).

## How to run

Reproduce the held-out numbers:

```sh
uv sync
uv run python -m model.predict sample/demo_watch.zip sample/weights.pt --targets sample/demo_targets.npz
```

The full pipeline needs video:

```sh
modal run pose/sam3d_modal.py --video data/raw/demo/video30.mp4
uv run python -m model.train $(cat data/takes.txt) --test data/raw/demo/watch,data/processed/demo_video30.sam3d.npz --seeds 5 --out data/processed
uv run python -m analysis.moves data/raw/demo/watch,data/processed/demo_video30.sam3d.npz
uv run python -m render.deviation data/processed/demo_video30.sam3d.npz data/processed/demo_video30.watchpose.npz data/raw/demo/watch out.mp4 71.5 89.5
```

## What works

On a held-out 4-minute take:

| | mean joint error |
|---|---|
| watch model | **61 mm** (arm 62, legs 67) |
| nearest-neighbour retrieval (`model.train`) | 87 mm |
| average pose | 106 mm |
| pretrained MobilePoser, watch only, 12 shared joints (`eval/mobileposer/run.py`) | 159 mm (riposte 58) |

- Cross-validated over time blocks within the 8 training takes: 58 mm.
- 13 of 13 lunges (3 false) and 20 of 22 thrusts (4 false), median onset within 33 and 67 ms of the video's.
- Hand speed per rep, from the accelerometer, ranks reps as the video does (ρ 0.90 over 13 lunges, 0.74 over 20 thrusts); onset-to-peak time and lunge depth don't (0.34, 0.19).
- Without the other free-fencing take in training: 78 mm.
- Blade: 4.1° median image-angle error (p90 11°) against the video's, vs 18° along the forearm.

## What's faked or mocked

Nothing is mocked.

## Limitations

- The watch can't tell where you stand or step: the pelvis is fixed in the watch-only view and taken from the video in the comparison.
- Ground truth is SAM 3D Body on monocular video, not motion capture.
- The model is offline: the GRU sees IMU after each frame.
- One fencer (me), one room, no opponent.
- Thrusts are weakest: 103 mm at full extension.
- The blade is scored as a 2D image angle, projected with the fencer's heading from the video: a side-on camera can't see depth.
