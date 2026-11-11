#!/bin/sh
# render/composite.sh <video> <take dir> -> <take dir>/overlay.mp4
# The render runs at the pose rate (scene.json), which is lower than the video's when frames were strided.
set -e
fps=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['fps'])" "$2/scene.json")
ffmpeg -v error -y -i "$1" -framerate "$fps" -start_number 1 -i "$2/frames/f_%04d.png" \
  -filter_complex "[0:v]fps=$fps[v];[v][1:v]overlay=0:0:shortest=1" -c:v libx264 -pix_fmt yuv420p -crf 18 "$2/overlay.mp4"
echo "$2/overlay.mp4"
