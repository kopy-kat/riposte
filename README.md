# riposte

Watching yourself is the fastest way to fix a fault, but a camera often can't come with you. riposte is the replay without the camera: from one Apple Watch on the weapon wrist it reconstructs the fencer, points the épée, moves them along the piste and shows faults such as a late arm. You film yourself once to teach it; after that, the watch is enough.

Single-wrist golf systems ([Lauer 2025](https://arxiv.org/abs/2506.17505), [Tan et al. 2026](https://arxiv.org/abs/2606.22876)) reconstruct the swing but not the club. [IMUPoser](https://arxiv.org/abs/2304.12518), [MobilePoser](https://arxiv.org/abs/2504.12492) and [ArmPoser](https://arxiv.org/abs/2609.08806) estimate everyday pose. None recovers the held implement. riposte does, and its replay shows a late-arm fault.

[riposte.mp4](riposte.mp4): narrated slides; the watch-only skeleton alone, over footage, against the video's pose, in 3D.

## How it works

1. Sensor Logger records the watch while an iPhone films; a wrist tap syncs them (`imu/sync.py`).
2. SAM 3D Body on Modal turns each frame into a 3D target (`pose/sam3d_modal.py`).
3. A bidirectional GRU maps 2 s of wrist IMU and attitude to 30 joints, trained on 21 minutes of my fencing (`model/train.py`).
4. A second GRU predicts pelvis velocity along the piste (`model/piste.py`).
5. The blade's direction and grip come from the watch's attitude, fitted to the training video (`imu/blade_video.py`).
6. A causal GRU streams the pose to a browser (`live/server.py`).

## How to run

```sh
uv sync
# The held-out take: every number in the first table and its bullets.
uv run python -m model.predict sample/demo_watch.zip sample/weights.pt --targets sample/demo_targets.npz
# Scrub, slow down and orbit the replay in a browser.
uv run python -m live.view sample/demo_watch.watchpose.npz
# The next day's take against the phone (116 mm) and the Mac camera (89), then its deliberate faults.
uv run python -m model.predict sample/nextday_watch.zip sample/weights.pt --targets sample/nextday_mac.npz
uv run python -m model.predict sample/nextday_watch.zip sample/weights.pt --targets sample/nextday_phone.npz
uv run python -m analysis.faults sample/nextday_watch.zip sample/nextday_phone.npz sample/nextday_watch.watchpose.npz
# Retrain from sample/takes/ (paired watch input and video pose): ~3 min on one core.
# Prints average pose 106, retrieval 87, model 63 (the shipped weights are 5 seeds of 150 epochs: 61).
uv run python -m model.train $(cat sample/takes.txt) --test sample/takes/demo.npz --device cpu --epochs 30
# Live: start the server, then push a recording to it at real time as the watch would.
uv run python -m live.server sample/causal.pt &
uv run python -m live.replay sample/demo_watch.zip --seconds 60
```

## What works

Held-out 4-minute take, mean joint error: watch model **61 mm**, nearest-neighbour retrieval 87, average pose 106. Pretrained MobilePoser on 12 shared joints: 159 (riposte 58).

- 13 of 13 lunges (3 false), 20 of 22 thrusts (4 false).
- Hand speed ranks reps as the video does (ρ 0.90 lunges, 0.74 thrusts); timing and depth don't.
- Pelvis-velocity correlation 0.95.
- Blade: 4.1° median image-angle error, vs 18° along the forearm.

A day later, watch re-strapped, same weights:

- 89 mm against a front-on camera, 116 side-on (average pose 125, 141). The cameras disagree by 84 mm.
- Deliberate faults: 6 normal, 6 late-arm, 6 low-arm lunges. At the video's lunge finishes, watch and video rank the 18 alike (ρ 0.93 elbow angle, 0.92 hand height). All 6 late-arm lunges show a more bent elbow than any normal one; 4 of 6 low-arm a lower hand.
- Blade in 3D from both cameras (`analysis/blade3d.py`): 16° median, vs 42° along the forearm.
- Lunges found: 21 of 22 (10 false) front-on, 6 of 6 (27 false) side-on. Thrusts: 13 of 34. Blade image angle: 7.6°, 8.5°.

Five training minutes with free fencing reach 65 mm; all 21.5 reach 62 (`analysis/curve.py`).

## What's faked or mocked

Nothing.

## Limitations

- One fencer, one room, no opponent.
- Ground truth is SAM 3D Body on monocular video, not motion capture.
- Live runs 2 s behind and scores 66 mm offline, 70 streamed, worse in the first minute.
- Blade and pelvis scores use the video's heading, and the overlay clip its room position; the watch has neither.
- The 3D fencer is a mesh fitted to the joints (`render/body.py`).
