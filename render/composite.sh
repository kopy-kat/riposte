#!/bin/sh
# render/composite.sh <video> <take dir> -> <take dir>/overlay.mp4
set -e
fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$1")
ffmpeg -v error -y -i "$1" -framerate "$fps" -start_number 1 -i "$2/frames/f_%04d.png" \
  -filter_complex "[0:v][1:v]overlay=0:0:shortest=1" -c:v libx264 -pix_fmt yuv420p -crf 18 "$2/overlay.mp4"
echo "$2/overlay.mp4"
