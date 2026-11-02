"""Stream a Gazebo transport topic as parsed JSON through the `gz topic` CLI (no gz python bindings
in the image).  Used by the EVALUATOR (ground truth), by the FANET bridge (link geometry only) and by
each vehicle's own power monitor (its own rotor-speed commands).  Never imported by agent safety code
for neighbour state."""
from __future__ import annotations

import json
import subprocess
import threading


class GzStream:
    def __init__(self, topic: str, on_msg):
        self.topic, self.on_msg = topic, on_msg
        self.proc = subprocess.Popen(["gz", "topic", "-e", "-t", topic, "--json-output"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.n = 0
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.n += 1
            try:
                self.on_msg(msg)
            except Exception:
                pass

    def close(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def stamp_s(msg) -> float:
    st = msg.get("header", {}).get("stamp", {})
    return float(st.get("sec", 0)) + 1e-9 * float(st.get("nsec", 0))


def model_poses(msg, prefix="x500_"):
    """{uav_id: (xyz, quat_xyzw)} for top-level models named x500_<id>."""
    out = {}
    for p in msg.get("pose", []):
        name = p.get("name", "")
        if not name.startswith(prefix):
            continue
        try:
            k = int(name[len(prefix):])
        except ValueError:
            continue
        pos = p.get("position", {})
        q = p.get("orientation", {})
        out[k] = ((float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))),
                  (float(q.get("x", 0.0)), float(q.get("y", 0.0)), float(q.get("z", 0.0)), float(q.get("w", 1.0))))
    return out
