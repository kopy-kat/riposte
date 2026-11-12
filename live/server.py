"""Live watch-only pose: Sensor Logger's HTTP push in, a skeleton in the browser out.

    uv run python -m live.server data/processed/live/causal.pt
then open the printed http://<this Mac>:8000/ and set Sensor Logger's HTTP Push URL to .../data. Each recording
(Sensor Logger sessionId) is saved under data/processed/live/<start time>/ as a WristMotion.csv that
imu.load reads, with the poses that were streamed in stream.npz.
"""

import argparse
import csv
import json
import queue
import signal
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from imu.load import G, SOURCE
from live.stream import Stream, load
from model.pairs import IMU_COLS

ROOT = Path(__file__).resolve().parents[1]
COLS = [SOURCE[c] for c in IMU_COLS + ["qw", "qx", "qy", "qz"]]
PAGE = Path(__file__).with_name("index.html")


class Session:
    def __init__(self, model, blade_b, out):
        self.stream = Stream(model, blade_b)
        self.dir = out / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True)
        self.csv = open(self.dir / "WristMotion.csv", "w", newline="")
        self.w = csv.writer(self.csv)
        self.w.writerow(["time", "seconds_elapsed", *COLS])
        self.t0 = self.metres = None
        self.t, self.pose = [], []

    def add(self, readings):
        """Sensor Logger "wrist motion" readings -> [(t, pose, blade)] for the frames they complete."""
        ns = np.array([r["time"] for r in readings], dtype=np.int64)
        rows = np.array([[r["values"][c] for c in COLS] for r in readings], float)
        if self.t0 is None:
            self.t0 = ns.min()
            # "Standardise Units & Frames" sends accelerations in m/s^2 instead of g.
            self.metres = np.linalg.norm(rows[:, 6:9], axis=1).mean() > 5
            (self.dir / "Metadata.csv").write_text(f"standardisation\n{str(self.metres).lower()}\n")
        for n, r in zip(ns, rows):
            self.w.writerow([n, (n - self.t0) / 1e9, *r])
        self.csv.flush()
        if self.metres:
            rows[:, 3:9] /= G
        frames = self.stream.push(ns / 1e9, rows)
        for t, pose, _ in frames:
            self.t.append(t)
            self.pose.append(pose)
        return frames

    def close(self):
        self.csv.close()
        if self.t:
            np.savez(self.dir / "stream.npz", t=np.array(self.t), pose=np.array(self.pose))
        print(f"saved {self.dir} ({len(self.t)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--blade", default=str(ROOT / "sample/blade.npz"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default=str(ROOT / "data/processed/live"))
    a = ap.parse_args()
    model, blade_b = load(a.weights), np.load(a.blade)["b"]
    lock, clients, sessions = threading.Lock(), [], {}
    stat = dict(n=0, at=time.time(), age=0.0, names=set())

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            arrived = time.time()
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            readings = [r for r in body.get("payload", []) if r.get("name") == "wrist motion"]
            with lock:
                stat["n"] += len(readings)
                stat["names"] |= {r.get("name") for r in body.get("payload", [])}
                if readings:
                    stat["age"] = arrived - max(r["time"] for r in readings) / 1e9
                if arrived - stat["at"] > 2:
                    # What the streaming check needs: is watch motion arriving, how fast and how late.
                    print(f"wrist motion {stat['n'] / (arrived - stat['at']):.0f} samples/s, newest {stat['age'] * 1000:.0f} ms old"
                          if stat["n"] else f"no wrist motion in the stream, only {sorted(stat['names'])}")
                    stat.update(n=0, at=arrived, names=set())
            if not readings:
                return
            with lock:
                sid = body.get("sessionId")
                if sid not in sessions:
                    for s in sessions.values():
                        s.close()
                    sessions.clear()
                    sessions[sid] = Session(model, blade_b, Path(a.out))
                    print(f"recording {sid}: {sessions[sid].dir}")
                frames = sessions[sid].add(readings)
                for t, pose, blade in frames[-1:]:  # the page draws only the newest pose
                    msg = json.dumps(dict(t=t, arrived=arrived, pose=np.round(pose.astype(float), 4).tolist(),
                                          blade=np.round(blade, 4).tolist(), compute=time.time() - arrived))
                    for q in clients:
                        q.put(msg)

        def do_GET(self):
            if self.path != "/events":
                page = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            q = queue.Queue()
            clients.append(q)
            try:
                while True:
                    self.wfile.write(f"data: {q.get()}\n\n".encode())
                    self.wfile.flush()
            except OSError:
                clients.remove(q)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))  # no packet is sent; this only picks the LAN interface
        ip = s.getsockname()[0]
    print(f"viewer            http://localhost:{a.port}/\nSensor Logger URL http://{ip}:{a.port}/data")
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # save the take on kill as on Ctrl-C
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for s in sessions.values():
            s.close()


if __name__ == "__main__":
    main()
