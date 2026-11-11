#!/bin/sh
# One take from the phone and watch exports to a Blender scene, retraining the watch model on every take.
#   scripts/take.sh <take name> <video.mov> <Sensor Logger watch .zip>
# Takes trained on are listed in data/takes.txt (watch dir,npz[,offset] per line); this take is appended.
set -eu
cd "$(dirname "$0")/.."
take=$1
raw=data/raw/$take
mkdir -p "$raw"
cp "$2" "$raw/video.mov"
rm -rf "$raw/watch" && mkdir "$raw/watch" && unzip -q -o "$3" -d "$raw/watch"

# 30 fps with the phone's rotation applied: SAM 3D reads raw frames, and the poses then match the background.
ffmpeg -v error -y -i "$raw/video.mov" -vf fps=30000/1001 -an -c:v libx264 -crf 18 "$raw/video30.mp4"
modal run pose/sam3d_modal.py --video "$raw/video30.mp4"
npz=data/processed/${take}_video30.sam3d.npz

uv run python -m model.pairs "$raw/watch" "$npz"
grep -q "^$raw/watch," data/takes.txt 2>/dev/null || echo "$raw/watch,$npz" >> data/takes.txt
uv run python -m model.train $(cat data/takes.txt) --out data/processed

uv run python -m render.prepare "$npz" --watch "$raw/watch" --watch-pose "data/processed/${take}_video30.watchpose.npz"
wh=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$raw/video30.mp4" | tr , ' ')
/Applications/Blender.app/Contents/MacOS/Blender -b -P render/scene.py -- "data/processed/${take}_video30" $wh \
  --blend "$raw/video30.mp4" >/dev/null
echo "data/processed/${take}_video30/scene.blend"
