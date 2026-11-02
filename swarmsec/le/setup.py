"""Episode setup shared by Layer 1 (swarmsec.le.sim) and the PX4 specs (experiments/le/make_px4_specs.py).

For a seed this fixes the fleet draw (class per id, initial charge), the mission-scaled nameplates and the
energy-need rate, so that the same seed means the same fleet in both layers.

Battery sizing (public, part of the fleet registry): the class-B nameplate is the energy a LEADER needs for
1.3 x the planned route time + boot + 40 s; class A = 22/17 x B, class C = 11/17 x B (so C cannot finish a
mission, B finishes as a follower but is marginal as leader, A can lead throughout).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..formation import formation_offsets
from .fleet import Fleet, FleetConfig
from .missions import make_plan


@dataclass
class EpisodeSetup:
    offsets: np.ndarray
    footprint: float
    plan: object
    fleet: Fleet
    fcfg: FleetConfig
    need_rate: float          # Wh/s a LEADER needs (hover x (1 + k_u 0.3) + leader overhead)
    designated: int

    def need_fn(self, k, c):
        return self.plan.remaining_time(int(k), np.asarray(c)[:2]) * self.need_rate + self.fcfg.e_land_wh + 0.3


def episode_setup(n, mission, seed, spacing=4.0, cruise=1.5, t_boot=6.0, dt=0.05, battery_scale=1.0,
                  offsets=None, plan_scale=1.0, p_hover_w=None):
    if offsets is None:
        offsets = formation_offsets("triangular_lattice", n, spacing)
    footprint = 2 * float(np.max(np.linalg.norm(offsets[:, :2], axis=1))) + spacing
    plan = make_plan(mission, footprint, cruise, scale=plan_scale)
    fcfg = FleetConfig(battery_scale=battery_scale)
    if p_hover_w is not None:
        fcfg.p_hover_w = float(p_hover_w)
        fcfg.p_min_w = 0.9 * float(p_hover_w)
    fleet = Fleet(n, dt, np.random.default_rng(seed + 11), fcfg)
    need_rate = (fcfg.p_hover_w * (1 + fcfg.k_u * 0.3) + fcfg.p_lead_w) / 3600.0
    route_T = plan.remaining_time(0, np.zeros(2))
    fleet.cap *= (1.3 * route_T + t_boot + 40.0) * need_rate / 17.0
    fleet.E = fleet.cap * np.random.default_rng(seed + 12).uniform(*fcfg.soc0, size=n)
    designated = int(np.argmax(np.where(np.array(fleet.classes) == "A", fleet.E, -1)))
    return EpisodeSetup(offsets, footprint, plan, fleet, fcfg, need_rate, designated)
