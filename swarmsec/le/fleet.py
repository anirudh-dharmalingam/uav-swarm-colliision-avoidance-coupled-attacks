"""Heterogeneous fleet model: what each UAV can actually do (ground truth) and what it can sense about itself.

Three airframe/avionics classes (fleet registry is public: every UAV knows every id's CLASS, i.e. its
nameplate battery capacity and radio class, as in a real fleet manifest; it does NOT know another UAV's
state of charge, navigation error or link quality):

  class  battery (Wh)  nav sigma (m)  radio range (m)   role it suits
  A      22.0          0.08 (RTK)     80                command-grade
  B      17.0          0.25           50                standard
  C      11.0          0.40           30                light / expendable

Energy:  P_i = P_hover (1 + k_u ||u_i|| / u_max) + [i is leader] P_lead ;  E_i(t) = E_i(0) - int P_i dt.
A UAV whose energy falls to E_land departs (lands; announces departure if honest).
Navigation: own position estimate p_hat_i = p_i + b_i(t) + w, with b_i an OU bias of std sigma_nav(class)
(time constant 20 s) and w white (sigma 0.03).  A GNSS fault adds a bias ramp (rate, cap).
Radio: per-sender range by class; link success decays linearly from 0.7*range to range.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLASSES = {
    "A": dict(cap_wh=22.0, nav_sigma=0.08, radio_m=80.0),
    "B": dict(cap_wh=17.0, nav_sigma=0.25, radio_m=50.0),
    "C": dict(cap_wh=11.0, nav_sigma=0.40, radio_m=30.0),
}
MIX = (("A", 0.2), ("B", 0.5), ("C", 0.3))


@dataclass
class FleetConfig:
    p_hover_w: float = 200.0
    k_u: float = 0.10
    p_lead_w: float = 24.0          # extra compute + radio duty of the leader (12 % of hover)
    p_min_w: float = 180.0          # public lower bound on airborne power (envelope for claims)
    e_land_wh: float = 1.0          # energy at which a UAV must land
    soc0: tuple = (0.80, 1.0)       # initial state of charge range (fraction of nameplate)
    bias_tau_s: float = 20.0
    white_sigma: float = 0.03
    battery_scale: float = 1.0      # scales all capacities (missions of different length)


def assign_classes(n: int, rng: np.random.Generator) -> list:
    """Deterministic class counts from MIX, random id assignment.  Guarantees >= 1 class-A UAV."""
    counts = [max(1, int(round(f * n))) if k == "A" else int(round(f * n)) for k, f in MIX]
    counts[1] = n - counts[0] - counts[2]
    cls = sum(([k] * c for (k, _), c in zip(MIX, counts)), [])
    rng.shuffle(cls)
    return cls


@dataclass
class Fleet:
    n: int
    dt: float
    rng: np.random.Generator
    cfg: FleetConfig = field(default_factory=FleetConfig)
    classes: list | None = None

    def __post_init__(self):
        c = self.cfg
        if self.classes is None:
            self.classes = assign_classes(self.n, self.rng)
        self.cap = np.array([CLASSES[k]["cap_wh"] for k in self.classes]) * c.battery_scale
        self.nav_sigma = np.array([CLASSES[k]["nav_sigma"] for k in self.classes])
        self.radio = np.array([CLASSES[k]["radio_m"] for k in self.classes])
        self.E = self.cap * self.rng.uniform(*c.soc0, size=self.n)
        self.bias = self.rng.normal(size=(self.n, 3)) * self.nav_sigma[:, None]
        self.bias[:, 2] *= 0.3
        self.fault_bias = np.zeros((self.n, 3))
        self.gnss_fault = np.zeros(self.n, bool)
        self.fault_dir = np.zeros((self.n, 3))
        self.fault_rate = np.zeros(self.n)
        self.fault_cap = np.zeros(self.n)
        self.drain_mult = np.ones(self.n)
        self.radio_mult = np.ones(self.n)
        self.alive = np.ones(self.n, bool)        # airborne and functioning
        self.crashed = np.zeros(self.n, bool)     # fail-silent (no departure announcement)
        self.departed_energy = np.zeros(self.n, bool)
        self.t_air = 0.0

    # -- ground truth -------------------------------------------------------------------------
    def step(self, u, leader_mask, u_max):
        c = self.cfg
        self.t_air += self.dt
        P = c.p_hover_w * (1.0 + c.k_u * np.linalg.norm(u, axis=1) / u_max) + leader_mask * c.p_lead_w
        P = P * self.drain_mult
        self.E = np.where(self.alive, self.E - P * self.dt / 3600.0, self.E)
        k = self.dt / c.bias_tau_s
        self.bias += -k * self.bias + self.nav_sigma[:, None] * np.sqrt(2 * k) * self.rng.normal(size=(self.n, 3))
        self.bias[:, 2] *= 0.98
        grow = self.gnss_fault & (np.linalg.norm(self.fault_bias, axis=1) < self.fault_cap)
        self.fault_bias[grow] += (self.fault_rate[grow] * self.dt)[:, None] * self.fault_dir[grow]
        newly = self.alive & (self.E <= c.e_land_wh)
        self.departed_energy |= newly
        self.alive &= ~newly
        return newly

    def nav_error(self):
        return self.bias + self.fault_bias

    def estimate(self, p, v):
        w = self.cfg.white_sigma
        ph = p + self.nav_error() + self.rng.normal(scale=w, size=p.shape)
        vh = v + self.rng.normal(scale=w, size=v.shape)
        return ph, vh

    def radio_range(self):
        return self.radio * self.radio_mult

    # -- faults ---------------------------------------------------------------------------------
    def crash(self, i):
        self.alive[i] = False
        self.crashed[i] = True

    def start_gnss_fault(self, i, rate=0.4, cap=8.0):
        a = self.rng.uniform(0, 2 * np.pi)
        self.gnss_fault[i] = True
        self.fault_dir[i] = (np.cos(a), np.sin(a), 0.0)
        self.fault_rate[i] = rate
        self.fault_cap[i] = cap

    def battery_anomaly(self, i, mult=5.0):
        self.drain_mult[i] = mult

    def radio_degrade(self, i, mult=0.3):
        self.radio_mult[i] = mult

    # -- public envelope used by verifiers --------------------------------------------------------
    def energy_envelope(self, t_air):
        """Upper bound on any UAV's remaining energy from public information only."""
        return self.cap - self.cfg.p_min_w * t_air / 3600.0
