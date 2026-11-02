"""FANET with SIMULATOR-time events and KEYED link opportunities (v2 campaign, audit item C5).

v1 FANET ran on the container's wall clock and drew loss/jitter from one RNG whose draws depended on traffic volume, so (a) blackout times were
wall seconds while the mission uses Gazebo sim time (RTF ~ 0.99) and (b) two methods with different packet counts saw different link realisations.
Here `t` is simulator time, and for each directed link (sender j -> receiver i) the *environmental* link state is a function of (scenario key, i, j,
time bin of 0.1 s) only: an independent loss uniform per bin and a Gilbert-Elliott chain advanced bin by bin with keyed transitions.  Method-induced
load effects (token-bucket budget, queueing) remain real and are NOT matched away.  Radio/range model and clock are explicit: application-level
impairment model (range-dependent loss, G-E bursts, delay + jitter, outages); not a PHY/MAC simulation; sim-time clock.
"""
from __future__ import annotations

import heapq

import numpy as np

from ..fanet import Fanet

BIN = 0.1


class KeyedFanet(Fanet):
    def __init__(self, n, cfg, rng, key):
        super().__init__(n, cfg, rng)
        self.key = int(key)
        self._ge = {}                         # (i, j) -> (bin, bad)
        self.outage_first = None
        self.outage_last = None

    def _u(self, i, j, b, tag):
        return float(np.random.default_rng(np.random.SeedSequence([self.key, i, j, int(b), tag])).random())

    def _bad(self, i, j, b):
        c = self.cfg
        b0, bad = self._ge.get((i, j), (b - 1, False))
        while b0 < b:
            b0 += 1
            u = self._u(i, j, b0, 1)
            bad = (u > c.ge_p_bg) if bad else (u < c.ge_p_gb)
        self._ge[(i, j)] = (b0, bad)
        return bad

    def send(self, t, sender, payload, true_pos, nbytes=None, targets=None):
        c = self.cfg
        nbytes = c.packet_bytes if nbytes is None else nbytes
        self.tokens[sender] = min(c.bucket_burst_bytes, self.tokens[sender] + (t - self.t_tok[sender]) * c.bucket_bytes_per_s)
        self.t_tok[sender] = t
        if self.tokens[sender] < nbytes:
            self.stats["budget_drop"] += 1
            return False
        self.tokens[sender] -= nbytes
        self.stats["tx"] += 1
        self.stats["tx_bytes"] += nbytes
        rng_m = c.range_m if c.range_by_sender is None else float(c.range_by_sender[sender])
        b = int(t / BIN)
        for i in range(self.n):
            if i == sender or (targets is not None and i not in targets):
                continue
            if self._outage(t, i, sender):
                self.stats["outage_drop"] += 1
                self.outage_first = t if self.outage_first is None else self.outage_first
                self.outage_last = t
                continue
            d = float(np.linalg.norm(true_pos[i] - true_pos[sender])) if true_pos is not None else 0.0
            if d > rng_m:
                self.stats["range_drop"] += 1
                continue
            if self._bad(i, sender, b) or self._u(i, sender, b, 2) < self.link_loss(d, rng_m):
                self.stats["loss_drop"] += 1
                continue
            lat = c.delay_s + (self._u(i, sender, b, 3) * c.jitter_s if c.jitter_s > 0 else 0.0)
            if lat > c.t_net_max:
                self.stats["late_drop"] += 1
                continue
            self.seq += 1
            heapq.heappush(self.q, (t + lat, self.seq, i, sender, payload))
        return True
