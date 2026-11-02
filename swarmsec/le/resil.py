"""Scenario-level exogenous data for the resilience campaign.

A scenario is drawn ONCE from a manifest row (scripts/resilience/make_manifest.py): fleet draw, route heading, compromised
identity, attack onset/strength, disturbance and network traces, disruption schedule.  Every defence at that scenario consumes the
same exogenous data.  Same seed alone is not enough (different controller branches consume a shared RNG differently), so
exogenous randomness is drawn from KEYED streams that are re-seeded per tick: the draw at tick k depends only on (scenario key,
stream name, k), never on how many draws earlier ticks consumed.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _h(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


class KeyedStreams:
    def __init__(self, key: int):
        self.key = int(key)

    def stream(self, name: str) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.key, _h(name)]))

    def tick(self, name: str, k: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.key, _h(name), int(k)]))


def build_setup(cfg, RS, seed):
    """EpisodeSetup for the scenario: PX4-matched plan scale / hover power if the manifest asks, rotated route."""
    from .missions import make_plan, rotate_plan
    from .setup import episode_setup
    kw = {}
    if RS.get("plan_scale") is not None:
        kw["plan_scale"] = float(RS["plan_scale"])
    if RS.get("p_hover_w") is not None:
        kw["p_hover_w"] = float(RS["p_hover_w"])
    SU = episode_setup(cfg.n, cfg.mission, seed, spacing=cfg.spacing, cruise=cfg.cruise, t_boot=cfg.t_boot, dt=cfg.dt,
                       battery_scale=cfg.battery_scale, **kw)
    th = float(RS.get("heading_rad", 0.0))
    if th != 0.0:
        SU.plan = rotate_plan(SU.plan, th)
    return SU
