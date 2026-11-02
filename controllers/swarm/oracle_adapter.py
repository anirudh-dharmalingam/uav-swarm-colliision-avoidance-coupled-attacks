#!/usr/bin/env python3
"""ORACLE / EMULATED-SENSOR ADAPTER (coupled-safety campaign E2).  Diagnostic information interventions ONLY.

This is the one process in the campaign that hands Gazebo ground truth to a controller process, so it is a separate, explicitly
labelled process that the experiment manager starts only when `cs.info.mode` is "oracle" or "sensor".  Standard / identity episodes
never start it and agents never subscribe to its topics in those modes.

  /oracle/truth       (mode oracle)  Float64MultiArray [t_mono, (id, x,y,z, vx,vy,vz) * n]   50 Hz, exact Gazebo state of every UAV.
                      Velocities are backward differences of sim-stamped poses over >= 20 ms.  The oracle agent substitutes the
                      TRUE state at the packet send time for the packet content and the TRUE ego state for the ego estimate,
                      keeping the packet transit/update timing, actuator model and W (diagnostic: "geometry known exactly").
  /sensor/uav_<i>     (mode sensor)  Float64MultiArray [t_acq_mono, (j, rx,ry,rz, rvx,rvy,rvz) * k]   relative state r = p_i - p_j,
                      v = v_i - v_j of every neighbour j within range, acquired at `rate_hz`, error uniform in a ball of
                      radius pos_err / vel_err, delivered after `latency_s`, no dropout.  A DECLARED SENSOR ABSTRACTION; not a model of a
                      real radar/vision/UWB sensor.
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gz_stream import GzStream, model_poses, stamp_s  # noqa: E402


def _ball(rng, radius):
    d = rng.normal(size=3)
    d /= max(np.linalg.norm(d), 1e-12)
    return d * radius * rng.random() ** (1 / 3)


class Oracle(Node):
    def __init__(self, n, cfg):
        super().__init__("cs_oracle_adapter")
        self.n = n
        I = cfg.get("cs", {}).get("info", {})
        self.mode = I.get("mode", "oracle")
        S = I.get("sensor", {})
        self.pos_err, self.vel_err = float(S.get("pos_err", 0.10)), float(S.get("vel_err", 0.15))
        self.rate, self.lat, self.rng_m = float(S.get("rate_hz", 20.0)), float(S.get("latency_s", 0.05)), float(S.get("range_m", 20.0))
        self.rng = np.random.default_rng(int(cfg.get("experiment", {}).get("seed", 0)) + 424242)
        self.lock = threading.Lock()
        self.pos = {}
        self.vel = {i: np.zeros(3) for i in range(n)}
        self.t_pose = {}
        self.hist = {}
        self.truth_pub = self.create_publisher(Float64MultiArray, "/oracle/truth", 20)
        self.sens_pub = {i: self.create_publisher(Float64MultiArray, f"/sensor/uav_{i}", 20) for i in range(n)}
        self.q = []
        self.seq = 0
        self.n_pose = {}
        self.t_first = {}
        self.t_last = {}
        self.gz = GzStream("/world/default/dynamic_pose/info", self._on_pose)
        self.timer_truth = self.create_timer(0.02, self._tick_truth)
        self.timer_sens = self.create_timer(1.0 / self.rate, self._tick_sensor)
        self.timer_q = self.create_timer(0.005, self._flush)

    def _on_pose(self, msg):
        P = model_poses(msg)
        ts = stamp_s(msg)
        with self.lock:
            for k, (pos, _) in P.items():
                p = np.array(pos, float)
                self.n_pose[k] = self.n_pose.get(k, 0) + 1
                self.t_first.setdefault(k, ts)
                self.t_last[k] = ts
                h = self.hist.setdefault(k, [])
                h.append((ts, p))
                while len(h) > 2 and h[-1][0] - h[1][0] >= 0.02:     # keep the shortest history spanning >= 20 ms
                    h.pop(0)
                if len(h) >= 2 and h[-1][0] - h[0][0] >= 0.02:
                    self.vel[k] = (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0])
                self.pos[k] = p

    def _snapshot(self):
        with self.lock:
            return {k: (self.pos[k].copy(), self.vel[k].copy()) for k in self.pos}

    def _tick_truth(self):
        if self.mode != "oracle":
            return
        snap = self._snapshot()
        if len(snap) < self.n:
            return
        data = [time.monotonic()]
        for i in range(self.n):
            p, v = snap[i]
            data += [float(i), *map(float, p), *map(float, v)]
        self.truth_pub.publish(Float64MultiArray(data=data))

    def _tick_sensor(self):
        if self.mode != "sensor":
            return
        snap = self._snapshot()
        if len(snap) < self.n:
            return
        t = time.monotonic()
        for i in range(self.n):
            row = [t]
            for j in range(self.n):
                if j == i:
                    continue
                r = snap[i][0] - snap[j][0]
                if np.linalg.norm(r) > self.rng_m:
                    continue
                rm = r + _ball(self.rng, self.pos_err)
                vm = (snap[i][1] - snap[j][1]) + _ball(self.rng, self.vel_err)
                row += [float(j), *map(float, rm), *map(float, vm)]
            self.seq += 1
            heapq.heappush(self.q, (t + self.lat, self.seq, i, row))

    def _flush(self):
        t = time.monotonic()
        while self.q and self.q[0][0] <= t:
            _, _, i, row = heapq.heappop(self.q)
            self.sens_pub[i].publish(Float64MultiArray(data=row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    rclpy.init()
    node = Oracle(a.n, cfg)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            import os
            rd = os.environ.get("RAL_RUN_DIR")
            if rd:
                (Path(rd) / "oracle_adapter_stats.json").write_text(json.dumps({
                    "mode": node.mode, "pose_msgs": node.n_pose, "pose_rate_hz": {k: node.n_pose[k] / max(node.t_last[k] - node.t_first[k], 1e-9) for k in node.n_pose},
                    "velocity": "backward difference of sim-stamped Gazebo poses over >= 20 ms (truth velocity itself is an estimate)"}))
        except Exception:
            pass
        node.gz.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
