# riposte

From one Apple Watch on the weapon wrist, riposte points the épée, reconstructs the whole fencer in 3D, and times each lunge and thrust. No camera at inference. It's built for fencers drilling alone: the onset-to-peak time of every lunge in a drill set.

Single-wrist golf systems ([Lauer 2025](https://arxiv.org/abs/2506.17505), [Tan et al. 2026](https://arxiv.org/abs/2606.22876)) reconstruct the swing and its events but not the club. [IMUPoser](https://arxiv.org/abs/2304.12518) and [MobilePoser](https://arxiv.org/abs/2504.12492) estimate everyday pose. riposte recovers the blade, works on fencing, and trains on its user's own phone video.

[riposte.mp4](riposte.mp4): the watch-only pose, then the same seconds against the pose from video the model never saw (white), coloured by error.

## How it works

1. Sensor Logger records the watch's IMU at 100 Hz while an iPhone films side-on (`imu/load.py`).
2. A wrist tap, seeded by the clocks of both devices, syncs the two streams to a frame (`imu/sync.py`, `model/pairs.py`).
3. SAM 3D Body on Modal GPUs turns each video frame into the 3D training target (`pose/sam3d_modal.py`).
4. A bidirectional GRU maps 2 s windows of wrist IMU and attitude to 30 joints relative to the pelvis. It trains on about 21 minutes of my fencing, with strap-rotation and noise augmentation (`model/train.py`).
5. The blade comes from the watch's attitude, fitted to the blade seen in the training video (`imu/blade_video.py`; the fit ships in `sample/`). Moves and their timing are read off the predicted pose (`analysis/moves.py`).

## How to run

Reproduce the held-out numbers from the watch alone, in under a minute:

```sh
uv sync
uv run python -m model.predict sample/demo_watch.zip sample/weights.pt --targets sample/demo_targets.npz
```

`sample/` holds the held-out take's watch IMU, weights and pose targets (no video).

The full pipeline needs raw video (not in the repo):

```sh
modal run pose/sam3d_modal.py --video data/raw/demo/video30.mp4
uv run python -m model.train $(cat data/takes.txt) --test data/raw/demo/watch,data/processed/demo_video30.sam3d.npz --seeds 5 --out data/processed
uv run python -m analysis.moves data/raw/demo/watch,data/processed/demo_video30.sam3d.npz
uv run python -m render.deviation data/processed/demo_video30.sam3d.npz data/processed/demo_video30.watchpose.npz out.mp4
```

## What works

On a 4-minute take held out from training:

| | mean joint error |
|---|---|
| watch model | **61 mm** (arm 62, legs 67) |
| nearest-neighbour retrieval | 87 mm |
| average pose | 106 mm |

- The retrieval row comes from `model.train`. Cross-validated over the 8 training takes, the model gets 58 mm.
- Moves found from the watch alone: 13 of 13 lunges (3 false alarms) and 20 of 22 thrusts (4 false alarms). Their median onset is within 33 ms and 67 ms of the video's.
- Blade: 4.1° median image-angle error (p90 11°) against the video's, vs 18° along the forearm.

## What's faked or mocked

Nothing in the pipeline or the numbers is mocked.

## Limitations

- The watch can't tell where you stand: the pelvis is fixed in the watch-only view and taken from the video in the comparison.
- The "ground truth" is SAM 3D Body run on monocular video, not motion capture.
- The model is offline: its bidirectional GRU also uses IMU from after each frame.
- One fencer (me), one room, no opponent.
- Thrusts are the weak spot, at 103 mm during full extension.
- Steps aren't detected.
- The blade is scored as a 2D image angle only: a side-on camera can't see it point toward or away.
