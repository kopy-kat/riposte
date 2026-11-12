"""The demo video: title, the watch-only fencer, the same seconds over the video, the numbers.

    uv run python -m render.cut data/processed/cut_watch_only/frames data/processed/cut_deviation.mp4 \
        data/processed/demo_video30.watchpose.npz riposte.mp4
frames: render/scene.py output for the watch-only take (RGBA PNGs, 1920x1080, f_N is pose sample N - 1);
deviation: render/deviation.py output covering the same seconds; the watch pose labels the moves.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analysis.moves import WATCH_LUNGE_PEAK, events, label
from render.deviation import BG, FONT, STOPS

W, H, FPS = 1920, 1080, 30
MINT = STOPS[0].astype(np.uint8)
FADE = 1.0
NAME = {"lunge": "Lunge", "extension": "Thrust"}
HOLD = 1.2  # seconds a move's name stays up after its peak
VOICE = "Daniel"
# (start second in the finished cut, line). Every number is one model.predict prints for the sample take.
NARRATION = [
    (0.3, "Riposte. One Apple Watch on the weapon wrist, and no camera."),
    (5.0, "This fencer is reconstructed from the watch's motion sensors alone: thirty joints, and the direction of "
          "the blade. The model is a small recurrent network, trained on twenty-one minutes of my own fencing, "
          "with phone video as the teacher."),
    (19.5, "It reads lunges and thrusts off the predicted pose. On a take it never saw, it found all thirteen "
           "lunges, and twenty of the twenty-two thrusts."),
    (32.0, "Here are the same seconds against the pose from video of that held-out take, in white. Colour shows "
           "the error on each limb. The blade is within four degrees, median, of the blade seen in the video."),
    (46.0, "The limits: one fencer, one room. The reference is a video pose model, not motion capture. And "
           "thrusts at full extension are the weak spot."),
    (57.0, "Sixty-one millimetres mean joint error on the held-out take. The average pose scores one hundred and six."),
]


def font(size, weight="Light"):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_name(weight)
    return f


def encode(path, frames):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(path)],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.asarray(f.convert("RGB")).tobytes())
    p.stdin.close()
    p.wait()


def card(lines, seconds):
    """lines: (text, size, weight, rgb, dy from centre). Fades in and out over FADE."""
    img = Image.new("RGB", (W, H), BG)
    g = ImageDraw.Draw(img)
    for text, size, weight, rgb, dy in lines:
        g.text((W / 2, H / 2 + dy), text, font=font(size, weight), fill=rgb, anchor="mm")
    base, black = np.asarray(img, dtype=np.float32), np.full((H, W, 3), BG, np.float32)
    n = int(seconds * FPS)
    for i in range(n):
        a = min(1.0, i / (FADE * FPS), (n - 1 - i) / (FADE * FPS))
        yield Image.fromarray((black + (base - black) * a).astype(np.uint8))


def move_at(ev, i):
    """(name, seconds onset to peak or None before the peak, opacity) of the latest move showing at pose sample i."""
    on = [e for e in ev if e["onset"] <= i <= e["peak"] + HOLD * FPS]
    if not on:
        return None
    e = max(on, key=lambda e: e["onset"])
    alpha = min(1.0, (i - e["onset"] + 1) / 4, (e["peak"] + HOLD * FPS - i) / 8)
    return NAME[e["kind"]], e["duration"] if i >= e["peak"] else None, alpha


def watch_only(frames, pose):
    ev = events(pose, label(pose), WATCH_LUNGE_PEAK)
    small, big, mid = font(36, "Regular"), font(64, "Light"), font(40, "Light")
    mint = tuple(int(v) for v in MINT)
    for i, path in enumerate(frames):
        px = np.asarray(Image.open(path).convert("RGBA")).copy()
        r, g, b = (px[..., i].astype(int) for i in range(3))
        px[(g > r + 30) & (g > b + 20), :3] = MINT  # the skeleton, in the overlay's colour
        px[(r > b + 40) & (r > g + 15), 3] = 0  # the tip trail: it tangles when the watch blade swings
        img = Image.new("RGBA", (W, H), BG + (255,))
        img.alpha_composite(Image.fromarray(px))
        out = np.clip(((len(frames) - 1 - i) / FPS - FADE) / 0.3, 0, 1)  # text gone before the crossfade
        txt = Image.new("RGBA", (W, H))
        g = ImageDraw.Draw(txt)
        g.text((64, H - 64), "Apple Watch only", font=small, fill=(255, 255, 255, int(150 * out)), anchor="ls")
        m = move_at(ev, int(path.stem[2:]) - 1)
        if m:
            name, dur, a = m
            a *= out
            g.text((64, H - 130), name, font=big, fill=(255, 255, 255, int(235 * a)), anchor="ls")
            if dur is not None:
                x = 64 + g.textlength(name, font=big) + 24
                g.text((x, H - 130), f"{dur:.2f} s", font=mid, fill=mint + (int(235 * a),), anchor="ls")
        img.alpha_composite(txt)
        yield img


def narrate(silent, out, tmp):
    """Mux the spoken NARRATION onto the silent cut."""
    clips = []
    for i, (_, line) in enumerate(NARRATION):
        clips.append(tmp / f"line{i}.aiff")
        subprocess.run(["say", "-v", VOICE, "-o", str(clips[-1]), line], check=True)
    delay = ";".join(f"[{i + 1}:a]adelay={int(t * 1000)}:all=1[a{i}]" for i, (t, _) in enumerate(NARRATION))
    mix = "".join(f"[a{i}]" for i in range(len(clips))) + f"amix=inputs={len(clips)}:normalize=0[a]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), *[a for c in clips for a in ("-i", str(c))],
                    "-filter_complex", delay + ";" + mix, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "128k", "-movflags", "+faststart", out], check=True)


def main():
    frames, deviation, watch, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    tmp = Path(tempfile.mkdtemp())
    grey, mint = (150, 150, 155), tuple(int(v) for v in MINT)
    encode(tmp / "title.mp4", card([("riposte", 120, "Light", (240, 240, 240), -20),
                                    ("one wrist, the whole fencer", 32, "Regular", grey, 80)], 3.0))
    encode(tmp / "watch.mp4", watch_only(sorted(frames.glob("f_*.png")), np.load(watch)["pose"]))
    encode(tmp / "end.mp4", card([("61 mm", 110, "Light", mint, -60),
                                  ("mean joint error on a take the model never saw", 30, "Regular", grey, 40),
                                  ("average pose: 106 mm", 26, "Regular", (95, 95, 100), 90)], 4.0))

    parts = [tmp / "title.mp4", tmp / "watch.mp4", deviation, tmp / "end.mp4"]
    lengths = [float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                               "csv=p=0", str(p)])) for p in parts]
    chain, prev, t = [], "[v0]", 0.0
    for i in range(1, len(parts)):
        t += lengths[i - 1] - FADE
        chain.append(f"{prev}[v{i}]xfade=transition=fade:duration={FADE}:offset={t:.3f}[x{i}]")
        prev = f"[x{i}]"
    norm = ";".join(f"[{i}:v]fps={FPS},scale={W}:{H},setsar=1,format=yuv420p[v{i}]" for i in range(len(parts)))
    subprocess.run(["ffmpeg", "-v", "error", "-y", *[a for p in parts for a in ("-i", str(p))], "-filter_complex",
                    norm + ";" + ";".join(chain), "-map", prev, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    str(tmp / "silent.mp4")], check=True)
    narrate(tmp / "silent.mp4", out, tmp)
    print(out)


if __name__ == "__main__":
    main()
