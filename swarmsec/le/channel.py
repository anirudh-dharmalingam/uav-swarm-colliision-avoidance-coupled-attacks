"""Broadcast packet channel for the election study (one packet per sender per protocol tick).

Per DIRECTED link j -> i:  delivered iff  sender alive  AND  receiver alive  AND  not cut by an outage
AND  Bernoulli(p_link) where
   p_link = (1 - loss) * clip((R_j - d_ij) / (0.3 R_j), 0, 1)       (R_j = sender's radio range)
and a Gilbert-Elliott bad state drops everything.  Delivered packets arrive after delay_ticks
(+ jitter in {0..jitter_ticks}).  The receiver keeps its latest delivered packet per sender and the
time it arrived.  A sender may suppress delivery to chosen receivers (send_mask), which is how a
selective-heartbeat adversary is modelled; honest senders broadcast to all.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

NET_PRESETS = {
    "ideal":    dict(loss=0.0,  delay_ticks=0, jitter_ticks=0, burst=False),
    "nominal":  dict(loss=0.05, delay_ticks=1, jitter_ticks=1, burst=False),
    "adverse":  dict(loss=0.15, delay_ticks=1, jitter_ticks=2, burst=True),
}


@dataclass
class NetConfig:
    loss: float = 0.05
    delay_ticks: int = 1
    jitter_ticks: int = 1
    burst: bool = False
    p_gb: float = 0.03
    p_bg: float = 0.3
    outages: list = field(default_factory=list)     # [{"t0","dur","cut":[ids on one side]}]


class PacketChannel:
    def __init__(self, n, width, rng, cfg: NetConfig):
        self.n, self.F, self.rng, self.cfg = n, width, rng, cfg
        self.hist = deque(maxlen=8)
        self.view = np.zeros((n, n, width))
        self.t_rx = np.full((n, n), -np.inf)      # time the latest packet arrived (receiver clock)
        self.t_tx = np.full((n, n), -np.inf)      # send time of that packet
        self.bad = np.zeros((n, n), bool)
        self.tx = 0
        self.delivered = 0
        self.last_expected = 0.0
        self.last_delivered = 0
        np.fill_diagonal(self.t_rx, 0.0)

    def _cut(self, t):
        cut = np.zeros((self.n, self.n), bool)
        for o in self.cfg.outages:
            if o["t0"] <= t < o["t0"] + o["dur"]:
                if o.get("all"):                          # full blackout: every directed link is down
                    cut[:] = True
                    continue
                side = np.zeros(self.n, bool)
                side[list(o["cut"])] = True
                cut |= side[:, None] != side[None, :]
        return cut

    def step(self, t, payload, true_p, radio, alive, send_mask=None):
        """payload (N,F) sent at time t; returns delivered (N,N) bool for this tick's arrivals."""
        c, n = self.cfg, self.n
        self.hist.append((t, payload.copy()))
        self.tx += int(alive.sum())
        d = np.linalg.norm(true_p[:, None, :2] - true_p[None, :, :2], axis=2)        # (i,j)
        R = radio[None, :]
        p = (1.0 - c.loss) * np.clip((R - d) / (0.3 * R), 0.0, 1.0)
        pair = alive[:, None] & alive[None, :] & ~np.eye(n, dtype=bool)
        self.last_expected = float((p * pair).sum())         # expected deliveries before bursts/outages (diagnostics)
        if c.burst:
            u = self.rng.random((n, n))
            self.bad = np.where(self.bad, u > c.p_bg, u < c.p_gb)
            p = np.where(self.bad, 0.0, p)
        ok = self.rng.random((n, n)) < p
        ok &= alive[:, None] & alive[None, :]
        ok &= ~self._cut(t)
        if send_mask is not None:
            ok &= send_mask
        np.fill_diagonal(ok, False)
        lag = c.delay_ticks + (self.rng.integers(0, c.jitter_ticks + 1, size=(n, n)) if c.jitter_ticks
                               else np.zeros((n, n), int))
        lag = np.minimum(lag, len(self.hist) - 1)
        out = np.zeros((n, n), bool)
        for L in np.unique(lag[ok]) if ok.any() else []:
            ts, pl = self.hist[-1 - int(L)]
            m = ok & (lag == L) & (ts > self.t_tx)          # never replace a newer packet with an older one
            if not m.any():
                continue
            ii, jj = np.nonzero(m)
            self.view[ii, jj] = pl[jj]
            self.t_tx[ii, jj] = ts
            self.t_rx[ii, jj] = t
            out |= m
        self.delivered += int(out.sum())
        self.last_delivered = int(out.sum())
        # own row is always fresh
        idx = np.arange(n)
        self.view[idx, idx] = payload
        self.t_rx[idx, idx] = t
        self.t_tx[idx, idx] = t
        return out
