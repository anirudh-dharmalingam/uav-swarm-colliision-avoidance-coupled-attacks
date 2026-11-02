"""Application-level FANET impairment model for the RAL26 PX4 path (NOT a radio PHY/MAC simulator).

Per DIRECTED link j -> i: Gilbert-Elliott burst-loss state, distance-dependent loss in the good
state, delay + bounded jitter, optional scheduled outages / partitions; per SENDER a token bucket
(byte-rate budget).  Link quality uses true inter-UAV geometry, but only inside this channel model;
receivers get the delivered packets only.  Every delivered packet's transit is <= T_net_max by
construction (a packet that would arrive later is dropped and counted), so the defenders'
max-transit premise holds for the channel itself.  ROS/DDS delivery adds latency on top, so the
end-to-end transit can exceed T_net_max.  The bridge does not log it; uav_agent_node records the
end-to-end t_rx - t_send per packet (summary key `transit_s`).  In the v2 campaigns a fraction <= 3e-4
of packets exceeded T_net_max = 0.05 s on the ideal channel.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FanetConfig:
    delay_s: float = 0.0
    jitter_s: float = 0.0
    loss: float = 0.0                 # i.i.d. loss in the good state at short range
    ge_p_gb: float = 0.0              # good -> bad per packet
    ge_p_bg: float = 0.3              # bad -> good per packet
    range_m: float = 60.0             # hard communication range
    range_soft_m: float = 25.0        # distance where loss starts to grow
    range_loss_slope: float = 0.02    # extra loss per metre beyond range_soft_m
    outages: list = field(default_factory=list)   # [{"t0":..,"dur":..,"nodes":[k,..]}] (k's links both ways)
    bucket_bytes_per_s: float = 1e9
    bucket_burst_bytes: float = 4096.0
    packet_bytes: int = 72
    t_net_max: float = 0.25           # declared bound; delivery later than this is dropped
    range_by_sender: list | None = None   # optional per-sender radio range (m), overrides range_m (LE study)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


class Fanet:
    def __init__(self, n: int, cfg: FanetConfig, rng: np.random.Generator):
        self.n, self.cfg, self.rng = n, cfg, rng
        self.bad = np.zeros((n, n), bool)
        self.tokens = np.full(n, cfg.bucket_burst_bytes, float)
        self.t_tok = np.zeros(n)
        self.q: list = []
        self.seq = 0
        self.stats = {"tx": 0, "tx_bytes": 0, "budget_drop": 0, "range_drop": 0, "loss_drop": 0,
                      "outage_drop": 0, "late_drop": 0, "delivered": 0}

    def _outage(self, t, i, j):
        for o in self.cfg.outages:
            if not (o["t0"] <= t < o["t0"] + o["dur"]):
                continue
            if o.get("all"):                                              # full blackout: every link is down
                return True
            if "cut" in o and ((i in o["cut"]) != (j in o["cut"])):      # partition: cross links only
                return True
            if i in o.get("nodes", ()) or j in o.get("nodes", ()):
                return True
        return False

    def link_loss(self, dist, rng_m=None):
        c = self.cfg
        rng_m = c.range_m if rng_m is None else rng_m
        if dist > rng_m:
            return 1.0
        if c.range_by_sender is not None:             # LE: linear decay over the last 30 % of the range
            return min(1.0, c.loss + max(0.0, (dist - 0.7 * rng_m) / (0.3 * rng_m)))
        return min(1.0, c.loss + max(0.0, dist - c.range_soft_m) * c.range_loss_slope)

    def send(self, t: float, sender: int, payload, true_pos: np.ndarray | None, nbytes: int | None = None,
             targets=None) -> bool:
        """Offer one broadcast from `sender` at bridge time t.  Returns False on budget drop.
        `targets` (optional set of receiver ids) restricts delivery (directional suppression by the sender)."""
        c = self.cfg
        nbytes = c.packet_bytes if nbytes is None else nbytes
        self.tokens[sender] = min(c.bucket_burst_bytes,
                                  self.tokens[sender] + (t - self.t_tok[sender]) * c.bucket_bytes_per_s)
        self.t_tok[sender] = t
        if self.tokens[sender] < nbytes:
            self.stats["budget_drop"] += 1
            return False
        self.tokens[sender] -= nbytes
        self.stats["tx"] += 1
        self.stats["tx_bytes"] += nbytes
        rng_m = c.range_m if c.range_by_sender is None else float(c.range_by_sender[sender])
        for i in range(self.n):
            if i == sender or (targets is not None and i not in targets):
                continue
            if self._outage(t, i, sender):
                self.stats["outage_drop"] += 1
                continue
            d = float(np.linalg.norm(true_pos[i] - true_pos[sender])) if true_pos is not None else 0.0
            if d > rng_m:
                self.stats["range_drop"] += 1
                continue
            # Gilbert-Elliott transition, then loss draw
            if self.bad[i, sender]:
                if self.rng.random() < c.ge_p_bg:
                    self.bad[i, sender] = False
            elif self.rng.random() < c.ge_p_gb:
                self.bad[i, sender] = True
            if self.bad[i, sender] or self.rng.random() < self.link_loss(d, rng_m):
                self.stats["loss_drop"] += 1
                continue
            lat = c.delay_s + (self.rng.random() * c.jitter_s if c.jitter_s > 0 else 0.0)
            if lat > c.t_net_max:
                self.stats["late_drop"] += 1
                continue
            self.seq += 1
            heapq.heappush(self.q, (t + lat, self.seq, i, sender, payload))
        return True

    def due(self, t: float):
        out = []
        while self.q and self.q[0][0] <= t:
            ta, _, i, j, payload = heapq.heappop(self.q)
            self.stats["delivered"] += 1
            out.append((ta, i, j, payload))
        return out


def outage_schedule(kind: str, n: int, t0: float, dur: float, nodes=None):
    """Named outage presets: 'none', 'node' (one node's links), 'partition' (split swarm in half)."""
    if kind == "none" or dur <= 0:
        return []
    if kind == "node":
        return [{"t0": t0, "dur": dur, "nodes": list(nodes or [0])}]
    if kind == "partition":
        return [{"t0": t0, "dur": dur, "cut": list(range(n // 2))}]
    if kind == "all":
        return [{"t0": t0, "dur": dur, "all": True}]
    raise ValueError(kind)
