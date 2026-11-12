# riposte

A whole épée fencer in 3D, from an Apple Watch on the weapon wrist. No camera at inference. Built for fencers drilling alone, who have no coach watching their lunges and thrusts.

[IMUPoser](https://arxiv.org/abs/2304.12518) and [MobilePoser](https://arxiv.org/abs/2504.12492) estimate everyday full-body pose from consumer IMUs. riposte is trained on fencing: its targets come from my own iPhone video, it gets the blade from the watch, and it times lunges and thrusts.

[riposte.mp4](riposte.mp4): first the watch-only pose, then the same seconds drawn over the video it never saw. The white ghost is the video's pose. The watch skeleton is coloured by its distance from the ghost.

## How it works

1. Sensor Logger records the watch's IMU at 100 Hz while an iPhone films side-on (`imu/load.py`).
2. A wrist tap, seeded by the clocks of both devices, syncs the two streams to a frame (`imu/sync.py`, `model/pairs.py`).
3. SAM 3D Body turns each video frame into a 3D body on Modal GPUs. That is the training target (`pose/sam3d_modal.py`).
4. A bidirectional GRU maps 2 s windows of wrist IMU to 30 joints relative to the pelvis. It trains on about 21 minutes of my fencing, with synthetic IMU and strap-rotation augmentation (`model/train.py`).
5. The blade comes from the watch's attitude, fitted to the blade seen in the video (`imu/blade_video.py`). Moves and their timing are read off the predicted pose (`analysis/moves.py`).

## How to run

Raw data isn't in the repo: it's video of me and my flat. The pipeline, given a take in `data/raw/<take>/`:

```sh
uv sync
modal run pose/sam3d_modal.py --video data/raw/demo/video30.mp4
uv run python -m model.train $(cat data/takes.txt) --test data/raw/demo/watch,data/processed/demo_video30.sam3d.npz --out data/processed
uv run python -m analysis.moves data/raw/demo/watch,data/processed/demo_video30.sam3d.npz
uv run python -m render.deviation data/processed/demo_video30.sam3d.npz data/processed/demo_video30.watchpose.npz data/raw/demo/video30.mp4 out.mp4
```

## What works

On a 4-minute take held out from training:

| | mean joint error |
|---|---|
| watch model | **64 mm** (arm 68, legs 70) |
| nearest-neighbour retrieval | 92 mm |
| average pose | 106 mm |

- A causal model that sees no future IMU gets 70 mm. Cross-validated over the 8 training takes, the model gets 59 mm.
- Moves found from the watch alone: 11 of 17 lunges and 18 of 22 thrusts. Their median onset is within 33 ms and 50 ms of the video's.
- `render.deviation` recomputes the error frame by frame against the video and gets 65 mm over the take.

## What's faked or mocked

- The watch can't see where you stand. The watch-only view holds the pelvis still, and the overlay places it at the video's pelvis.
- The lunge threshold in `analysis/moves.py` was tuned by watching the held-out take.

## Limitations

- The "ground truth" is SAM 3D Body run on monocular video, not motion capture.
- The model is offline. The GRU is bidirectional over 2 s windows, so each frame also uses IMU from after it.
- There is one fencer (me), one room and no opponent.
- Thrusts are the weak spot, at 134 mm during full extension.
- Steps aren't detected.
