#!/usr/bin/env python3
"""RAL26 evaluator: Gazebo ground truth, isolated from every controller.

Reads model poses straight from Gazebo transport (/world/default/dynamic_pose/info, sim-time stamped),
rotor-speed commands (/x500_<i>/command/motor_speed) and RTF (/world/default/stats).  It subscribes to
the GCS topic only to learn mission start/end markers, publishes nothing, and its outputs are written
after the run -- no controller can read them.

Outputs in <run_dir>/eval/: truth.csv (10 Hz sim-time samples: position, tilt per UAV), summary.json.

Propulsion power is the Gazebo multicopter-motor-model shaft power  P = sum_k c_m c_T w_k^3
(c_T = motorConstant, c_m = momentConstant of the X500 SDF): an ESTIMATED mechanical power, not a
measured electrical energy (no ESC/motor efficiency model).  PX4's simulated battery is not used.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gz_stream import GzStream, model_poses, stamp_s  # noqa: E402

C_T, C_M = 8.54858e-06, 0.016           # X500 SDF motorConstant, momentConstant (PX4 v1.17 models)


def tilt_deg(q):
    x, y, z, w = q
    # angle between body z and world z
    bz = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, bz))))


class Evaluator(Node):
    def __init__(self, n, run_dir: Path, attacker: int | None, d_min=1.5, d_safe=2.5, truth_hz=10.0):
        super().__init__("ral26_truth_evaluator")
        self.n, self.run_dir, self.attacker = n, run_dir, attacker
        self.d_min, self.d_safe = d_min, d_safe
        self.lock = threading.Lock()
        self.samples = []                 # (t_sim, wall, pos(n,3), tilt(n))
        self.last_keep = -1.0
        self.keep_dt = 0.0 if truth_hz <= 0 else 1.0 / truth_hz - 1e-4
        self.power = {i: [] for i in range(n)}      # (t_wall, P)
        self.rtf = []
        self.markers = {}
        self.create_subscription(String, "/ral/gcs", self._on_gcs, 10)
        self.streams = [GzStream("/world/default/dynamic_pose/info", self._on_pose),
                        GzStream("/world/default/stats", self._on_stats)]
        for i in range(n):
            self.streams.append(GzStream(f"/x500_{i}/command/motor_speed", lambda m, i=i: self._on_motor(i, m)))
        self.t_sim_last = 0.0
        self.pose_n = 0
        self.clock = []

    def _on_gcs(self, msg):
        d = json.loads(msg.data)
        with self.lock:
            if d["cmd"] not in self.markers:
                t_sim = self.t_sim_last
                if "t0" in d and d["cmd"] == "start":
                    # back-date to the GCS start instant: latency on the shared monotonic clock x current RTF
                    rtf = self.rtf[-1][1] if self.rtf else 1.0
                    t_sim -= max(0.0, time.monotonic() - float(d["t0"])) * min(max(rtf, 0.05), 1.5)
                self.markers[d["cmd"]] = {"t_sim": t_sim, "wall": time.time(), **d}

    def _on_pose(self, msg):
        t = stamp_s(msg)
        poses = model_poses(msg)
        self.models_seen = max(getattr(self, "models_seen", 0), len(poses))
        if len(poses) < self.n:
            return
        self.pose_n += 1
        self.t_sim_last = t
        if t - self.last_keep < self.keep_dt:
            return
        self.last_keep = t
        P = np.array([poses[i][0] for i in range(self.n)])
        T = np.array([tilt_deg(poses[i][1]) for i in range(self.n)])
        with self.lock:
            self.samples.append((t, time.time(), P, T))
            self.clock.append((t, time.time(), time.monotonic()))     # passive host<->plant clock map (rarf_ablation_v1)

    def _on_stats(self, msg):
        r = msg.get("realTimeFactor", msg.get("real_time_factor"))
        if r is not None:
            with self.lock:
                self.rtf.append((self.t_sim_last, float(r)))

    def _on_motor(self, i, msg):
        w = np.asarray(msg.get("velocity", []), float)
        if w.size:
            with self.lock:
                self.power[i].append((self.t_sim_last, float(np.sum(C_M * C_T * np.abs(w) ** 3))))

    # --------------------------------------------------------------------------------------------
    def finish(self, geometry, mission):
        for s in self.streams:
            s.close()
        out = self.run_dir / "eval"
        out.mkdir(parents=True, exist_ok=True)
        with self.lock:
            S = list(self.samples)
            markers = dict(self.markers)
            power = {i: list(v) for i, v in self.power.items()}
            rtf = list(self.rtf)
        with (out / "truth.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_sim"] + [f"{a}{i}" for i in range(self.n) for a in ("x", "y", "z", "tilt")])
            for t, _, P, T in S:
                w.writerow([f"{t:.3f}"] + [f"{v:.3f}" for i in range(self.n) for v in (*P[i], T[i])])
        with (out / "clock_map.csv").open("w", newline="") as fh:     # sim stamp of the newest kept pose vs host wall/monotonic at receipt
            w = csv.writer(fh)
            w.writerow(["t_sim", "wall", "mono"])
            for t, wl, mo in list(self.clock):
                w.writerow([f"{t:.4f}", f"{wl:.4f}", f"{mo:.4f}"])
        summ = summarize(S, markers, power, rtf, self.n, self.attacker, self.d_min, self.d_safe, geometry, mission)
        summ["pose_msgs"] = self.pose_n
        summ["models_seen"] = getattr(self, "models_seen", 0)       # isolation check: must equal n
        summ["isolation_ok"] = summ["models_seen"] == self.n
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
        return summ


def _pairs(n, attacker):
    hh, ha = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (ha if attacker in (i, j) else hh).append((i, j))
    return hh, ha


def summarize(S, markers, power, rtf, n, attacker, d_min, d_safe, geometry, mission):
    t0 = markers.get("start", {}).get("t_sim")
    t1 = markers.get("land", {}).get("t_sim")
    res = {"mission_started": t0 is not None, "mission_ended": t1 is not None, "n": n, "attacker": attacker,
           "t_mission_start_sim": t0, "t_mission_end_sim": t1}
    if t0 is None or not S:
        res["censored"] = "mission_never_started"
        return res
    t1 = t1 if t1 is not None else S[-1][0]
    W = [(t, P, T) for t, _, P, T in S if t0 <= t <= t1]
    if not W:
        res["censored"] = "no_truth_samples_in_window"
        return res
    hh, ha = _pairs(n, attacker if attacker is not None else -1)
    ts = np.array([t for t, _, _ in W])
    Ps = np.stack([P for _, P, _ in W])

    def pair_min(pairs):
        if not pairs:
            return np.full(len(W), np.inf)
        d = np.stack([np.linalg.norm(Ps[:, i] - Ps[:, j], axis=1) for i, j in pairs], axis=1)
        return d.min(axis=1)

    dhh, dha = pair_min(hh), pair_min(ha)

    def events(d):
        below = d < d_min
        return int(np.sum(below[1:] & ~below[:-1]) + (1 if below[0] else 0))

    res.update({
        "window_s": float(ts[-1] - ts[0]), "samples": len(W),
        "min_sep_hh": float(dhh.min()), "min_sep_ha": float(dha.min()) if ha else None,
        "collision_hh": bool(dhh.min() < d_min), "collision_ha": bool(ha and dha.min() < d_min),
        "collision_events_hh": events(dhh), "collision_events_ha": events(dha) if ha else 0,
        "frac_below_dsafe_hh": float(np.mean(dhh < d_safe)),
        "frac_below_dsafe_ha": float(np.mean(dha < d_safe)) if ha else None,
        "contact_proxy_hh": bool(dhh.min() < 0.7),
        "max_tilt_deg": float(max(T.max() for _, _, T in W)),
        "alt_err_max": float(np.max(np.abs(Ps[:, :, 2] - mission.get("altitude", 5.0)))),
    })
    honest = [i for i in range(n) if i != attacker]
    off = np.array([geometry["desired_offsets"][str(i)] for i in range(n)], float)
    # translation-invariant formation error of the honest sub-swarm (nominal, unsqueezed shape)
    H = Ps[:, honest, :2]
    O = off[honest, :2]
    e = (H - H.mean(axis=1, keepdims=True)) - (O - O.mean(axis=0))[None]
    res["formation_rmse_hh"] = float(np.sqrt(np.mean(np.sum(e ** 2, axis=2))))
    home = np.array([np.mean([geometry["spawn_positions"][str(i)][0] for i in honest]),
                     np.mean([geometry["spawn_positions"][str(i)][1] for i in honest])])
    c_end = Ps[-1, honest, :2].mean(axis=0)
    goal = home + np.array(mission.get("route", [0.0, 0.0]))
    res["route_len_m"] = float(np.linalg.norm(goal - home))
    res["centroid_progress_m"] = float((c_end - home) @ ((goal - home) / max(np.linalg.norm(goal - home), 1e-9))) if res["route_len_m"] > 0 else 0.0
    res["final_goal_err_m"] = float(np.linalg.norm(c_end - goal))
    res["max_goal_progress_m"] = float(max(((Ps[k, honest, :2].mean(axis=0) - home) @ ((goal - home) / max(np.linalg.norm(goal - home), 1e-9)))
                                           for k in range(len(W)))) if res["route_len_m"] > 0 else 0.0
    # energy (Gazebo motor-model shaft power, estimated)
    E = {}
    for i in range(n):
        pw = [(t, p) for t, p in power[i] if t0 <= t <= t1]
        if len(pw) > 2:
            tt = np.array([t for t, _ in pw]); pp = np.array([p for _, p in pw])
            E[i] = float(np.sum(0.5 * (pp[1:] + pp[:-1]) * np.diff(tt)) / 3600.0)
    res["energy_wh_window"] = E
    res["energy_wh_honest_total"] = float(sum(v for k, v in E.items() if k != attacker))
    r = [x for t, x in rtf if t0 <= t <= t1]
    res["rtf_mean"] = float(np.mean(r)) if r else None
    res["rtf_min"] = float(np.min(r)) if r else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--attacker", type=int, default=-1)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    geometry = json.loads(Path(a.geometry).read_text())
    mission = {"altitude": float(cfg.get("swarm", {}).get("altitude_m", 5.0)),
               "route": cfg.get("ral26", {}).get("route", [0.0, 0.0])}
    rclpy.init()
    ev = Evaluator(a.n, Path(a.run_dir), None if a.attacker < 0 else a.attacker,
                   float(cfg.get("safety", {}).get("d_min", 1.5)), float(cfg.get("safety", {}).get("d_safe", 2.5)),
                   float(cfg.get("ral26", {}).get("truth_hz", 10.0)))
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    try:
        while rclpy.ok() and not stop["flag"]:
            rclpy.spin_once(ev, timeout_sec=0.2)
            if "done" in ev.markers:
                break
    finally:
        s = ev.finish(geometry, mission)
        print(json.dumps({k: s.get(k) for k in ("min_sep_hh", "min_sep_ha", "collision_hh", "rtf_mean")}), flush=True)
        ev.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
