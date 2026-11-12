"""The fencer's alpha matte for a stretch of video, for render.deviation --over --matte.

    uv run --with transformers --with timm --with kornia --with einops python -m render.matte \
        data/raw/demo/video30.mp4 33.5 49.5 data/processed/cut5/matte.mp4
BiRefNet (ZhengPeng7/BiRefNet_lite) per frame: it keeps hair, fingers and the bell guard where the projected
body mesh leaves a halo. About half a second a frame on an Apple GPU. The blade is thin enough that it comes
and goes; render.deviation finds it separately.
"""

import subprocess
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation

SIZE = 1024
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


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
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet_lite", trust_remote_code=True)
    net = net.eval().float().to(device)
    mean, std = (torch.tensor(v, device=device).view(1, 3, 1, 1) for v in (MEAN, STD))
    for i in range(end - start):
        frame = Image.frombytes("RGB", (w, h), src.stdout.read(w * h * 3)).resize((SIZE, SIZE), Image.BILINEAR)
        x = torch.as_tensor(np.asarray(frame), device=device).permute(2, 0, 1)[None].float() / 255
        with torch.no_grad():
            alpha = net((x - mean) / std)[-1].sigmoid()[0, 0].cpu().numpy()
        alpha = Image.fromarray((alpha * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        enc.stdin.write(np.asarray(alpha).tobytes())
        if i % 30 == 0:
            print(f"{i}/{end - start}", flush=True)
    enc.stdin.close()
    enc.wait()
    print(out)


if __name__ == "__main__":
    main()
