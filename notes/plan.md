# Épée pose from an Apple Watch

Reconstruct a 3D épée fencer from wrist IMU alone, shown next to synced video and rendered in Blender. Hackathon deadline: midnight 2026-11-12 (London).

## Setup
- Apple Watch Series 8, watchOS 11.6.2, worn on the weapon wrist. Paid Apple developer account. iPhone's Lightning port is broken: deploy via Xcode wireless pairing, else TestFlight.
- M2 Pro Mac, Xcode installed, no local GPU. Cloud: Modal (free $30/month credit, per-second billing).
- Fencing at home, solo: shadow fencing plus lunges into a wall target (with or without a pillow). No opponent.
- A borrowed phone for a second camera, or a hip IMU in a running belt, is available tomorrow.

## Pipeline
1. A watchOS recorder: an HKWorkoutSession plus `CMBatchedSensorManager` (800 Hz accelerometer, 200 Hz device motion; Series 8 support per WWDC23 notes, unconfirmed on the device). The file transfers to the iPhone, then the Mac.
2. Video: the iPhone films side-on at 120–240 fps. A wrist tap at the start of each take syncs the streams. Hold en garde for 3 s per take to calibrate the axes.
3. Ground truth: GVHMR (or WHAM) on Modal produces world-grounded SMPL per frame.
4. Model:
   - Retrieval first: classify and segment the actions from the IMU, then play the recorded 3D clips time-warped to the watch's timing.
   - Generative stretch: fine-tune MobilePoser (wrist-only) on paired data, pretrained on DIP-IMU, with AMASS synthetic IMU as needed.
5. Output: a three.js or Blender render of the avatar beside the video, plus an error metric on held-out takes.

## Optional data
FIE épée bouts from YouTube → GVHMR → synthetic wrist IMU, as a fencing motion prior.

## Today, in the office (no épée)
Build the watch recorder, check the sample rates, put GVHMR on Modal and test it on an office clip, then build the sync and viewer. Use a pen or ruler as a stand-in blade.
