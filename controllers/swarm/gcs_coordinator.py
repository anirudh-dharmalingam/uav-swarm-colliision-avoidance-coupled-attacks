#!/usr/bin/env python3
"""Ground-station role for RAL26 runs: start/land/done markers only (no state is relayed).

Waits until every agent reports HOLD (airborne at its spawn hover point), publishes
{"cmd":"start","t0":<monotonic>} on /ral/gcs, publishes {"cmd":"land"} after the mission duration and
{"cmd":"done"} once all agents report DONE (or on timeout).  Exit code 0 = complete, 2 = censored.
The mission duration is measured on Gazebo SIM time (/world/default/stats) when available, so a real-time
factor below 1 does not shorten the flown mission; the start marker also carries t0_sim.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gz_stream import GzStream, stamp_s  # noqa: E402


class GCS(Node):
    def __init__(self, n, duration, run_dir: Path, takeoff_timeout=120.0, land_timeout=40.0):
        super().__init__("ral26_gcs")
        self.n, self.duration = n, duration
        self.takeoff_timeout, self.land_timeout = takeoff_timeout, land_timeout
        self.phase = {}
        self.pub = self.create_publisher(String, "/ral/gcs", 10)
        for i in range(n):
            self.create_subscription(String, f"/ral/status/uav_{i}", self._on_status, 10)
        self.t_boot = time.monotonic()
        self.t0 = self.t_land = None
        self.state = "WAIT_HOLD"
        self.code = None
        self.log = (run_dir / "gcs_events.jsonl").open("w")
        self.t_sim = None
        self.t0_sim = None
        self.gz = GzStream("/world/default/stats", self._on_stats)
        self.timer = self.create_timer(0.25, self.tick)

    def _on_stats(self, msg):
        st = msg.get("simTime") or msg.get("sim_time")
        t = float(st.get("sec", 0)) + 1e-9 * float(st.get("nsec", 0)) if isinstance(st, dict) else stamp_s(msg)
        if t > 0:
            self.t_sim = t

    def _on_status(self, msg):
        d = json.loads(msg.data)
        self.phase[int(d["id"])] = d["phase"]

    def ev(self, name, **kw):
        self.log.write(json.dumps({"wall": time.time(), "mono": time.monotonic(), "sim": self.t_sim, "event": name,
                                   **kw}) + "\n")
        self.log.flush()

    def send(self, **kw):
        self.pub.publish(String(data=json.dumps(kw)))

    def tick(self):
        now = time.monotonic()
        ph = [self.phase.get(i) for i in range(self.n)]
        if self.state == "WAIT_HOLD":
            if all(p in ("HOLD",) for p in ph):
                self.t0 = now
                self.t0_sim = self.t_sim
                self.state = "MISSION"
                self.ev("start", phases=ph)
            elif now - self.t_boot > self.takeoff_timeout:
                self.ev("takeoff_timeout", phases=ph)
                self.state, self.t_land, self.code = "LAND", now, 2
            return
        if self.state == "MISSION":
            self.send(cmd="start", t0=self.t0, t0_sim=self.t0_sim)
            if self.t0_sim is not None and self.t_sim is not None:
                elapsed = self.t_sim - self.t0_sim
            else:
                elapsed = now - self.t0
            if elapsed >= self.duration or all(p in ("DONE", "LAND") for p in ph):
                self.state, self.t_land = "LAND", now
                self.ev("land", phases=ph)
            return
        if self.state == "LAND":
            self.send(cmd="land", t0=self.t0)
            if all(p == "DONE" for p in ph) or now - self.t_land > self.land_timeout:
                self.ev("done", phases=ph, all_done=all(p == "DONE" for p in ph))
                for _ in range(8):
                    self.send(cmd="done", t0=self.t0)
                    time.sleep(0.1)
                if self.code is None:
                    self.code = 0 if all(p == "DONE" for p in ph) else 3
                raise SystemExit(self.code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    rclpy.init()
    g = GCS(a.n, a.duration, Path(a.run_dir))
    code = 0
    try:
        rclpy.spin(g)
    except SystemExit as e:
        code = int(e.code or 0)
    finally:
        g.log.close()
        g.gz.close()
        g.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
