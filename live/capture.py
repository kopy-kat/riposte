"""Films the replay page being used, for the demo video: headless Chrome, real pointer events, and the page's
animation clock stepped one video frame at a time so every run gives the same frames.

    uv run --with playwright python -m live.capture <take>.watchpose.npz out.mp4 --match 5671 --cam=-10.09,5.3,1.2
writes <take>.html with live.view, then clicks the fastest lunge's mark, lets it play, drags to orbit, clicks a
later mark, and drags and scrolls until the page's camera is render/studio.py's at its first frame.
--match: the pose sample on screen FADE seconds before the end, where the cut crossfades into the Blender fencer
(studio.py's first frame N is sample N - 1); the clip runs on at that camera through the fade.
--cam: studio.py's camera there: angle about the fencer in degrees (its `a`), RADIUS, and height above the floor.
"""

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

from live import view

W, H, FPS = 1920, 1080, 30
SECONDS, FADE = 8.0, 1.0
# The page looks at a point 0.85 m above the lowest joint; the joints of the feet sit about 3 cm above the soles
# that studio.py stands on its floor.
LOOK_M = 0.88
YAW_PX, PITCH_PX, ZOOM = 0.008, 0.006, 0.001  # index.html's drag and scroll gains
# Headless Chrome draws no cursor, and its own clock would tie playback to how long a screenshot takes.
CLOCK = """(() => { let t = 0, queue = [];
  performance.now = () => t;
  window.requestAnimationFrame = f => queue.push(f);
  window.step = ms => { t += ms; const run = queue; queue = []; for (const f of run) f(t); }; })()"""
POINTER = """(() => { const d = document.createElement('div'); d.id = 'pointer';
  d.style.cssText = 'position:fixed;z-index:9;pointer-events:none;left:0;top:0;width:26px;height:26px;margin:-13px 0 0 -13px;' +
    'border-radius:50%;border:2px solid #fff;background:rgba(255,255,255,.18)';
  document.body.appendChild(d); })()"""


def ease(u):
    return u * u * (3 - 2 * u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--match", type=int, required=True)
    ap.add_argument("--cam", required=True)
    a = ap.parse_args()
    angle, radius, height = map(float, a.cam.split(","))
    end = dict(yaw=math.pi / 2 - math.radians(angle), dist=math.hypot(radius, height - LOOK_M))
    end["pitch"] = math.asin((height - LOOK_M) / end["dist"])

    d = np.load(a.npz)
    ev = view.moves(d)
    lead = int(round(view.rate(d) / 2))  # a clicked mark starts the take this many samples before its move
    n_frames, at = int(SECONDS * FPS), int((SECONDS - FADE) * FPS)
    click2 = 100
    second = max((e for e in ev if e["onset"] - lead == a.match - (at - click2)), key=lambda e: e["onset"], default=None)
    if second is None:  # no mark starts exactly there: take the nearest earlier one and click it that much sooner
        second = max((e for e in ev if e["onset"] - lead <= a.match - (at - click2)), key=lambda e: e["onset"])
        click2 = at - (a.match - (second["onset"] - lead))
    first = max((e for e in ev if e["kind"] == "lunge" and e["end"] < second["onset"]), key=lambda e: e["hand"])
    page_path = view.write(Path(a.npz))

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "image2pipe", "-framerate", str(FPS), "-i", "-", "-c:v",
                            "libx264", "-pix_fmt", "yuv420p", "-crf", "14", a.out], stdin=subprocess.PIPE)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        pg = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        pg.add_init_script(CLOCK)
        pg.goto(page_path.resolve().as_uri())
        pg.evaluate(POINTER)
        left, top, width = pg.evaluate("(r => [r.left, r.top, r.width])(document.getElementById('line').getBoundingClientRect())")
        n = len(d["t"])
        mark = lambda e: ((e["onset"] + e["end"]) / 2 / n * width + left, top + 20)
        state = lambda: dict(zip(("yaw", "pitch", "dist"), pg.evaluate("[yaw, pitch, dist]")))
        pos = [1180.0, 640.0]

        def to(x, y):
            pos[:] = x, y
            pg.mouse.move(x, y)
            pg.evaluate(f"document.getElementById('pointer').style.transform = 'translate({x}px, {y}px)'")

        def press(down):
            pg.mouse.down() if down else pg.mouse.up()
            fill = "rgba(110,231,183,.75)" if down else "rgba(255,255,255,.18)"
            pg.evaluate(f"document.getElementById('pointer').style.background = '{fill}'")

        # Pointer moves are (first frame, last frame, from, to), eased; a click lands on the frame it is keyed by,
        # pressed two frames before so the press shows.
        canvas1, canvas2 = (760.0, 600.0), (1260.0, 620.0)
        mid = dict(yaw=state()["yaw"] + 0.8, pitch=0.3)  # three-quarters on, where the next move still reads
        moves = [(0, 9, tuple(pos), mark(first)), (13, 44, mark(first), canvas1), (86, 98, None, mark(second)),
                 (103, 126, mark(second), canvas2)]
        drags = [(50, 84, mid), (132, 204, end)]
        zoom = (150, 200)
        clicks = {10: first, click2: second}
        to(*pos)
        plan = {}
        for f in range(n_frames):
            # Input lands half way through each frame: a jump to a whole sample then stays half a sample clear of
            # the next one, and each later frame shows exactly the next sample.
            pg.evaluate(f"step({500 / FPS})")
            for f0, f1, src, dst in moves:
                if f0 <= f <= f1:
                    if f == f0 and src is None:
                        plan[f0] = tuple(pos)
                    src = src or plan[f0]
                    u = ease((f - f0) / (f1 - f0))
                    to(src[0] + (dst[0] - src[0]) * u, src[1] + (dst[1] - src[1]) * u)
            if f + 2 in clicks or f in clicks:
                press(f + 2 in clicks)
            for f0, f1, target in drags:
                if f == f0:
                    s = state()
                    plan[f0] = (tuple(pos), (target["yaw"] - s["yaw"]) / YAW_PX, (target["pitch"] - s["pitch"]) / PITCH_PX)
                    press(True)
                if f0 < f <= f1:
                    src, dx, dy = plan[f0]
                    u = ease((f - f0) / (f1 - f0))
                    to(src[0] + dx * u, src[1] + dy * u)
                if f == f1:
                    press(False)
            if f == zoom[0]:
                plan["zoom"] = math.log(end["dist"] / state()["dist"]) / ZOOM
            if zoom[0] < f <= zoom[1]:
                step = lambda g: ease((g - zoom[0]) / (zoom[1] - zoom[0]))
                pg.mouse.wheel(0, plan["zoom"] * (step(f) - step(f - 1)))
            pg.evaluate(f"step({500 / FPS})")
            if f == at:
                shown = float(pg.evaluate("document.getElementById('head').style.left").rstrip("%")) / 100 * n
                print(f"frame {at}: sample {int(shown)} (match {a.match}), camera {state()}, wanted {end}")
            enc.stdin.write(pg.screenshot(type="png"))
        browser.close()
    enc.stdin.close()
    enc.wait()
    print(a.out, f"clicked the lunge at sample {first['onset']} and the {second['kind']} at {second['onset']}")


if __name__ == "__main__":
    main()
