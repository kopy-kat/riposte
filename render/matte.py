"""The fencer's alpha matte for a stretch of video, for render.deviation --over --matte.

    uv run --with "rembg[cpu]" python -m render.matte data/raw/demo/video30.mp4 33.5 49.5 data/processed/cut_fencer/matte.mp4
BiRefNet (rembg's birefnet-general-lite) per frame: it keeps hair, fingers and the bell guard where the projected
body mesh leaves a halo. About 10 s a frame on CPU. The blade is too thin for it; render.deviation finds that.
"""

import subprocess
import sys

import numpy as np
from PIL import Image
from rembg import new_session, remove


def main():
    video, out = sys.argv[1], sys.argv[4]
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=width,height,r_frame_rate", "-of", "csv=p=0", video], text=True).strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    fps = eval(probe[2])
    start, end = (int(float(s) * fps) for s in sys.argv[2:4])
    src = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-vf", f"select=gte(n\\,{start})", "-fps_mode",
                            "passthrough", "-frames:v", str(end - start), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r",
                            f"{fps}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", out],
                           stdin=subprocess.PIPE)
    session = new_session("birefnet-general-lite")
    for i in range(end - start):
        frame = Image.frombytes("RGB", (w, h), src.stdout.read(w * h * 3))
        enc.stdin.write(np.asarray(remove(frame, session=session, only_mask=True).convert("L")).tobytes())
        if i % 30 == 0:
            print(f"{i}/{end - start}", flush=True)
    enc.stdin.close()
    enc.wait()
    print(out)


if __name__ == "__main__":
    main()
