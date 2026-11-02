#!/usr/bin/env python3
"""HF1: PX4 authority / time-response identification, one process per vehicle.

After the GCS start marker each vehicle flies a seeded program of held horizontal commands u (ENU,
m/s^2) through the selected offboard interface:
  acceleration : TrajectorySetpoint.acceleration = u (xy), z position hold
  velocity     : the legacy mapping v_sp = V + u * tau (xy), z position hold
Segments: steps of magnitude in MAGS with random heading, immediate reversals (u then -u) and zero
segments; a home-seeking bias keeps vehicles inside their own 20 m cell, and a speed envelope
replaces the command by braking when |v| > vmax.  Logs at 50 Hz: command, setpoint, PX4 estimate
(position, velocity, acceleration).  Gazebo truth is recorded separately by truth_evaluator.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent))
from px4_iface import PX4Vehicle  # noqa: E402

MAGS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)


class HF1(Node):
    def __init__(self, uid, cfg, geo, run_dir: Path, mode: str, seed: int):
        super().__init__(f"hf1_{uid}")
        self.i = uid
        self.alt = float(cfg["swarm"].get("altitude_m", 5.0))
        self.mode = mode
        H = cfg.get("hf1", {})
        self.tau = float(H.get("velocity_tau_s", 0.5))
        self.hold = float(H.get("hold_s", 1.5))
        self.vmax = float(H.get("vmax", 4.0))
        self.cell = float(H.get("cell_m", 8.0))
        self.jerk = float(H.get("jerk", 0.0))           # 0 = pure steps; >0 = slew-limited transitions [m/s^3]
        self.u_applied = np.zeros(3)
        self.rng = np.random.default_rng(seed * 100 + uid)
        self.spawn = np.array(geo["spawn_positions"][str(uid)], float)
        self.veh = PX4Vehicle(self, uid, self.spawn)
        self.st_pub = self.create_publisher(String, f"/ral/status/uav_{uid}", 10)
        self.create_subscription(String, "/ral/gcs", self._on_gcs, 10)
        self.t0 = None
        self.land = False
        self.phase, self.t_phase = "BOOT", time.monotonic()
        self.seg_end, self.u, self.seg_kind, self.seg_id = 0.0, np.zeros(3), "zero", 0
        self.pending = []
        self.hold_since = None
        run_dir.mkdir(parents=True, exist_ok=True)
        self.f = (run_dir / f"hf1_{uid}.csv").open("w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "seg", "kind", "ux", "uy", "spx", "spy", "px", "py", "pz", "vx", "vy", "vz", "ax", "ay", "az", "nav"])
        self.timer = self.create_timer(0.02, self.loop)

    def _on_gcs(self, m):
        d = json.loads(m.data)
        if d.get("cmd") == "start" and self.t0 is None:
            self.t0 = float(d["t0"])
        if d.get("cmd") in ("land", "done"):
            self.land = True

    def next_segment(self, now, p, v):
        if not self.pending:
            mag = float(self.rng.choice(MAGS))
            th = self.rng.uniform(0, 2 * math.pi)
            d = np.array([math.cos(th), math.sin(th), 0.0])
            off = (p - self.spawn)
            off[2] = 0
            if np.linalg.norm(off) > self.cell:                 # home-seeking: always head back
                d = -off / np.linalg.norm(off)
                mag = min(mag, 3.0)
            kind = self.rng.choice(["step", "reversal", "step_zero"])
            if kind == "reversal":
                self.pending = [("step", mag * d), ("reversal", -mag * d), ("zero", np.zeros(3))]
            elif kind == "step_zero":
                self.pending = [("step", mag * d), ("zero", np.zeros(3))]
            else:
                self.pending = [("step", mag * d)]
        k, u = self.pending.pop(0)
        self.seg_kind, self.u = k, u
        self.seg_id += 1
        self.seg_end = now + self.hold

    def loop(self):
        now = time.monotonic()
        self.st_pub.publish(String(data=json.dumps({"id": self.i, "phase": self.phase})))
        veh = self.veh
        if not veh.has_state():
            return
        p, v = veh.pos_enu(), veh.vel_enu()
        if self.phase == "BOOT":
            if now - self.t_phase > 2.0 and veh.bootstrap(now, self.alt):
                self.phase, self.t_phase = "TAKEOFF", now
            return
        if self.phase in ("TAKEOFF", "HOLD"):
            veh.publish_mode("position"); veh.hover_setpoint(self.alt)
            if self.phase == "TAKEOFF" and abs(p[2] - self.alt) < 0.3 and np.linalg.norm(v) < 0.3:
                self.hold_since = self.hold_since or now
                if now - self.hold_since > 1.0:
                    self.phase = "HOLD"
            if self.phase == "HOLD" and self.t0 is not None:
                self.phase, self.t_phase = "MISSION", now
            if self.land:
                self.phase, self.t_phase = "LAND", now
            return
        if self.phase == "MISSION":
            if self.land:
                self.phase, self.t_phase = "LAND", now
                return
            if now >= self.seg_end:
                self.next_segment(now, p, v)
            u = self.u.copy()
            if self.jerk > 0:
                du = u - self.u_applied
                nd = float(np.linalg.norm(du))
                if nd > self.jerk * 0.02:
                    u = self.u_applied + du * (self.jerk * 0.02 / nd)
            kind = self.seg_kind
            sp = float(np.linalg.norm(v[:2]))
            off = p - self.spawn; off[2] = 0.0
            far = np.linalg.norm(off) > 2.0 * self.cell and float(v[:2] @ off[:2]) > 0
            if (sp > self.vmax and float(u[:2] @ v[:2]) > 0) or far:
                ub = -2.0 * v / sp; ub[2] = 0.0; kind = "brake"
                if self.jerk > 0:
                    du = ub - self.u_applied; nd = float(np.linalg.norm(du))
                    ub = self.u_applied + du * min(1.0, self.jerk * 0.02 / max(nd, 1e-9))
                u = ub
                self.seg_end = min(self.seg_end, now + 0.02)
            self.u_applied = u.copy()
            if self.mode == "velocity":
                vsp = v + u * self.tau; vsp[2] = 0.0
                veh.publish_mode("velocity"); veh.velocity_setpoint(vsp, self.alt)
                spx, spy = vsp[0], vsp[1]
            else:
                veh.publish_mode("acceleration"); veh.accel_setpoint(u, self.alt)
                spx, spy = u[0], u[1]
            a = veh.acc_enu()
            self.w.writerow([f"{now - self.t0:.4f}", self.seg_id, kind, f"{u[0]:.4f}", f"{u[1]:.4f}", f"{spx:.4f}", f"{spy:.4f}",
                             *[f"{x:.4f}" for x in (*p, *v, *a)], veh.nav_state()])
            return
        if self.phase == "LAND":
            veh.publish_mode("position")
            if now - veh.last_land > 2.0:
                veh.request_land(); veh.last_land = now
            if not veh.armed() and now - self.t_phase > 2.0:
                self.phase = "DONE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", default="acceleration")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    geo = json.loads(Path(a.geometry).read_text())
    rclpy.init()
    node = HF1(a.id, cfg, geo, Path(a.run_dir) / "agents", a.mode, a.seed)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.phase == "DONE" and time.monotonic() - node.t_phase > 1.0:
                break
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.f.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
