# riposte

A 3D replay of fencing from Apple Watch data only.

In many sports it is impractical to film (no space, the camera needs to move, too many people etc), but it is still useful to watch replays of your technique. This is a proof of concept applied to fencing.

- One watch on the weapon wrist, no camera
- Rebuilds the body, the épée and the movement along the piste
- Trained on 21 minutes of video of me
- 61 mm mean joint error on a take it hasn't seen
- Blade direction within 4.1°
- Finds 13 of 13 lunges and 20 of 22 thrusts

https://github.com/user-attachments/assets/76052a86-cf7d-4cf3-8db6-dbf79b1ba246

## How it works

1. Sensor Logger records the watch while an iPhone films; a wrist tap syncs them (`imu/sync.py`).
2. SAM 3D Body on Modal turns each frame into a 3D target (`pose/sam3d_modal.py`).
3. A bidirectional GRU maps 2 s of wrist IMU and attitude to 30 joints, trained on 21 minutes of my fencing (`model/train.py`).
4. A second GRU predicts pelvis velocity along the piste (`model/piste.py`).
5. The blade's direction and grip come from the watch's attitude, fitted to the training video (`imu/blade_video.py`).
6. A causal GRU streams the pose to a browser (`live/server.py`).

## How to run it

```sh
uv sync
# Predict the held-out take and score it against the video.
uv run python -m model.predict sample/demo_watch.zip sample/weights.pt --targets sample/demo_targets.npz
# Scrub, slow down and orbit the replay in a browser.
uv run python -m live.view sample/demo_watch.watchpose.npz
# Retrain from the sample takes, about 3 minutes on one core.
uv run python -m model.train $(cat sample/takes.txt) --test sample/takes/demo.npz --device cpu --epochs 30
# Live: start the server, then push a recording to it at real time as the watch would.
uv run python -m live.server sample/causal.pt &
uv run python -m live.replay sample/demo_watch.zip --seconds 60
```

## Ideas for the future

- Other sports where a camera is hard to bring: swimming, tennis.
- Other fencers, without filming each one first.
- Bouts: two fencers with a watch each, replayed together.
- A report after a session that lists the faults and how often they happened.
- A watch app that records and uploads on its own.
