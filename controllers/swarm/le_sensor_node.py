#!/usr/bin/env python3
"""Basic proximity-sensor EMULATOR for the leader-election study on PX4/Gazebo.

Reads Gazebo truth poses (like the FANET bridge does for link geometry) and publishes, for every UAV i,
the ANONYMOUS detections its onboard range/bearing sensor would produce at 10 Hz on /le/det/uav_<i>
(flattened [dx,dy,dz, ...], relative to the TRUE vehicle position, i.e. a body-mounted sensor).  Noise,
detection probability, range and false alarms are swarmsec.le.sensing.BasicSensor.scan -- the same model as
Layer 1.  Truth never leaves this process except through these detections.  Landed vehicles (z < 1 m)
are not detected.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gz_stream import GzStream, model_poses, stamp_s  # noqa: E402
from swarmsec.le.sensing import BasicSensor, SensorConfig  # noqa: E402


class SensorEmu(Node):
    def __init__(self, n, seed, rate_hz=10.0, keyed=False, sensor_cfg=None):
        super().__init__("le_sensor_emulator")
        self.n = n
        self.seed, self.keyed = int(seed), bool(keyed)
        self.t_sim = 0.0
        self.sensor = BasicSensor(n, np.random.default_rng(seed), SensorConfig(**(sensor_cfg or {})))
        self.pos = None
        self.lock = threading.Lock()
        self.gz = GzStream("/world/default/dynamic_pose/info", self._on_pose)
        self.pub = {i: self.create_publisher(Float64MultiArray, f"/le/det/uav_{i}", 10) for i in range(n)}
        self.create_timer(1.0 / rate_hz, self._tick)

    def _on_pose(self, msg):
        P = model_poses(msg)
        if len(P) >= self.n:
            with self.lock:
                self.pos = np.array([P[i][0] for i in range(self.n)])
                self.t_sim = stamp_s(msg)

    def _tick(self):
        with self.lock:
            p = None if self.pos is None else self.pos.copy()
        if p is None:
            return
        airborne = p[:, 2] > 1.0
        if self.keyed:
            # exogenous draws keyed by (scenario key, sim-time bin): identical for every defence at a scenario
            self.sensor.rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(round(self.t_sim * 10.0))]))
        dets = self.sensor.scan(p, airborne)
        for i in range(self.n):
            self.pub[i].publish(Float64MultiArray(data=[float(x) for x in np.asarray(dets[i]).ravel()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    rclpy.init()
    L = cfg.get("le", {})
    s = SensorEmu(a.n, int(cfg.get("experiment", {}).get("seed", 0)) + 515, keyed=bool(L.get("res", False)),
                  sensor_cfg=L.get("sensor"))
    try:
        rclpy.spin(s)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        s.gz.close()
        s.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
