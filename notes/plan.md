# Épée pose from an Apple Watch

Reconstruct a 3D épée fencer from wrist IMU alone, shown next to synced video and rendered in Blender. Hackathon deadline: midnight 2026-11-12 (London).

## Setup
- Apple Watch Series 8, watchOS 11.6.2, worn on the weapon wrist. Paid Apple developer account. iPhone's Lightning port is broken: deploy via Xcode wireless pairing, else TestFlight.
- M2 Pro Mac, Xcode installed, no local GPU. Cloud: Modal (free $30/month credit, per-second billing).
- Fencing at home, solo: shadow fencing plus lunges into a wall target (with or without a pillow). No opponent.
- A borrowed phone for a second camera, or a hip IMU in a running belt, is available tomorrow.

## Pipeline
1. Watch IMU: Sensor Logger on watchOS (one combined CSV), loaded by `imu/load.py`.
2. Video: the iPhone films side-on. A wrist tap at the start of each take, seeded by the iPhone's capture-start tag, syncs the streams (`imu/sync.py`).
3. Ground truth: SAM 3D Body on Modal, run per frame in parallel chunks (`pose/sam3d_modal.py`).
4. Model: a GRU from wrist IMU to full-body pose (`model/train.py`), with a kNN retrieval baseline, synthetic-IMU augmentation and held-out test takes.
5. Output: a three.js or Blender render of the avatar beside the video, plus an error metric on held-out takes.
6. Watch-only blade: `imu/blade_video.py` fits `data/processed/blade.npz` (a grip direction plus an IMU-driven correction) over the épée takes in `data/takes.txt` (the office takes used a stand-in blade, so leave them out); refit it after adding takes, then `render.prepare --watch-only ... --watch`.

## Optional data
FIE épée bouts from YouTube → GVHMR → synthetic wrist IMU, as a fencing motion prior.
