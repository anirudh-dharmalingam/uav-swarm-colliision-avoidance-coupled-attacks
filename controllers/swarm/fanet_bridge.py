#!/usr/bin/env python3
"""RAL26 FANET bridge: mobility-driven, receiver-specific inter-UAV packet delivery.

Agents publish their self-reports on /ral/tx (Float64MultiArray [sender, seq, t_send, px,py,pz,vx,vy,vz])
and refresh requests on /ral/req_tx ([requester, t]).  The bridge pushes each offer through
swarmsec.fanet.Fanet (per-link Gilbert-Elliott loss, distance loss from Gazebo geometry, delay/jitter
bounded by T_net_max, outages/partitions, per-sender byte budget) and delivers to /ral/rx/uav_<i> and
/ral/req/uav_<i>.  The Gazebo geometry is used ONLY for link quality; it is never forwarded.

Note: this is an application-level impairment model, not a radio PHY/MAC simulation, and it does not
touch the PX4 <-> uXRCE-DDS autopilot link.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gz_stream import GzStream, model_poses  # noqa: E402
from swarmsec.fanet import Fanet, FanetConfig, outage_schedule  # noqa: E402
from swarmsec.le_v2.fanet_v2 import KeyedFanet  # noqa: E402


class Bridge(Node):
    def __init__(self, n, cfg: FanetConfig, seed, run_dir: Path, outage=None, sim_time=False, key=None):
        super().__init__("ral26_fanet_bridge")
        self.n = n
        self.sim_time = bool(sim_time)
        self._clk = None
        self.fan = KeyedFanet(n, cfg, np.random.default_rng(seed), key) if (sim_time and key is not None) else Fanet(n, cfg, np.random.default_rng(seed))
        self.req = Fanet(n, FanetConfig(**{**cfg.__dict__, "packet_bytes": 16,
                                           "bucket_bytes_per_s": 1e9}), np.random.default_rng(seed + 1))
        self.t0 = time.monotonic()
        self.outage = outage
        self.pos = np.zeros((n, 3))
        self.lock = threading.Lock()
        self.gz = GzStream("/world/default/dynamic_pose/info", self._on_pose)
        self.gz_clk = GzStream("/world/default/stats", self._on_stats) if self.sim_time else None
        self.rx_pub = {i: self.create_publisher(Float64MultiArray, f"/ral/rx/uav_{i}", 50) for i in range(n)}
        self.req_pub = {i: self.create_publisher(Float64MultiArray, f"/ral/req/uav_{i}", 10) for i in range(n)}
        self.create_subscription(Float64MultiArray, "/ral/tx", self._on_tx, 100)
        self.create_subscription(Float64MultiArray, "/ral/req_tx", self._on_req, 50)
        from std_msgs.msg import String as _S
        self.create_subscription(_S, "/le/ctl", self._on_ctl, 10)
        self.bytes_by_sender = np.zeros(n)
        self.t_start = None                              # bridge-clock time of the GCS start marker (mission t = 0)
        self.offered = {}                                # sec -> per-sender offered broadcasts
        self.deliv = {}                                  # sec -> (sender, receiver) delivered counts
        self.req_bytes_by_sender = np.zeros(n)
        self.run_dir = run_dir
        self.timer = self.create_timer(0.005, self._deliver)
        self.stat_timer = self.create_timer(5.0, self._dump)

    def _on_pose(self, msg):
        P = model_poses(msg)
        if len(P) >= self.n:
            with self.lock:
                self.pos = np.array([P[i][0] for i in range(self.n)])

    def _on_stats(self, msg):
        st = msg.get("simTime") or {}
        ts = float(st.get("sec", 0)) + 1e-9 * float(st.get("nsec", 0))
        r = msg.get("realTimeFactor")
        if ts > 0:
            self._clk = (ts, time.monotonic(), float(min(max(float(r), 0.0), 1.5)) if r is not None else 1.0)

    def sim_now(self):
        c = self._clk
        if c is None:
            return 0.0
        return c[0] + min(1.0, max(0.0, time.monotonic() - c[1])) * c[2]

    def t(self):
        return self.sim_now() if self.sim_time else time.monotonic() - self.t0

    def _on_tx(self, m):
        d = list(m.data)
        j = int(d[0])
        targets = None
        if d[1] < 0:                                   # LE: negative seq encodes a receiver bit mask
            mask = int(round(-d[1] - 1.0))
            targets = {k for k in range(self.n) if mask >> k & 1}
        with self.lock:
            pos = self.pos.copy()
        if self.fan.send(self.t(), j, d, pos, targets=targets):
            self.bytes_by_sender[j] += self.fan.cfg.packet_bytes
        if self.t_start is not None:
            sec = int(self.t() - self.t_start)
            self.offered.setdefault(sec, np.zeros(self.n, int))[j] += 1

    def _on_ctl(self, m):
        """LE fault injection: {"radio_mult": {"<id>": f}} scales that sender's radio range."""
        d = json.loads(m.data)
        rb = self.fan.cfg.range_by_sender
        if rb is not None:
            for k, f in d.get("radio_mult", {}).items():
                rb[int(k)] = float(rb[int(k)]) * float(f)

    def _on_req(self, m):
        d = list(m.data)
        j = int(d[0])
        with self.lock:
            pos = self.pos.copy()
        if self.req.send(self.t(), j, d, pos):
            self.req_bytes_by_sender[j] += 16

    def _deliver(self):
        t = self.t()
        for _, i, j_, payload in self.fan.due(t):
            self.rx_pub[i].publish(Float64MultiArray(data=payload))
            if self.t_start is not None:
                self.deliv.setdefault(int(t - self.t_start), np.zeros((self.n, self.n), int))[j_, i] += 1
        for _, i, _, payload in self.req.due(t):
            self.req_pub[i].publish(Float64MultiArray(data=payload))

    def _dump(self):
        (self.run_dir / "fanet_stats.json").write_text(json.dumps({
            "t": self.t(), "data": self.fan.stats, "req": self.req.stats,
            "bytes_by_sender": self.bytes_by_sender.tolist(),
            "req_bytes_by_sender": self.req_bytes_by_sender.tolist(),
            "config": self.fan.cfg.__dict__}, indent=1))
        if self.t_start is not None:
            (self.run_dir / "fanet_delivery.json").write_text(json.dumps({
                "t_start_bridge": self.t_start, "n": self.n, "clock": "sim" if self.sim_time else "wall",
                "outage_first_drop": getattr(self.fan, "outage_first", None), "outage_last_drop": getattr(self.fan, "outage_last", None),
                "offered": {str(k): v.tolist() for k, v in sorted(self.offered.items())},
                "delivered": {str(k): v.tolist() for k, v in sorted(self.deliv.items())}}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    sect = "le" if "le" in cfg else "ral26"
    fc = dict(cfg.get(sect, {}).get("fanet", {}))
    o = cfg.get(sect, {}).get("outage", {})
    if o:
        # outage times are mission-relative in the config; the bridge clock starts before takeoff, so
        # outages are re-anchored by the agents' GCS start time via an offset estimated at runtime.
        fc["outages"] = []
    rclpy.init()
    Lc = cfg.get(sect, {})
    b = Bridge(a.n, FanetConfig.from_dict(fc), int(cfg.get("experiment", {}).get("seed", 0)) + 777, Path(a.run_dir),
               sim_time=bool(Lc.get("sim_time_net", False)), key=Lc.get("res_key"))
    if o:
        from std_msgs.msg import String

        def on_gcs(msg, b=b, o=o):
            d = json.loads(msg.data)
            if d.get("cmd") == "start" and not b.fan.cfg.outages:
                t_start = float(d["t0_sim"]) if (b.sim_time and d.get("t0_sim") is not None) else b.t() - (time.monotonic() - float(d["t0"]))
                sched = outage_schedule(o.get("kind", "none"), a.n, t_start + float(o.get("t0", 10.0)),
                                        float(o.get("dur", 0.0)), o.get("nodes"))
                b.fan.cfg.outages = sched
                b.req.cfg.outages = sched
        b.create_subscription(String, "/ral/gcs", on_gcs, 10)
    from std_msgs.msg import String as _S2

    def on_start(msg, b=b):
        d = json.loads(msg.data)
        if d.get("cmd") == "start" and b.t_start is None:
            b.t_start = float(d["t0_sim"]) if (b.sim_time and d.get("t0_sim") is not None) else b.t() - (time.monotonic() - float(d["t0"]))
    b.create_subscription(_S2, "/ral/gcs", on_start, 10)
    try:
        rclpy.spin(b)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            b._dump()
        except Exception:
            pass
        b.gz.close()
        b.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
