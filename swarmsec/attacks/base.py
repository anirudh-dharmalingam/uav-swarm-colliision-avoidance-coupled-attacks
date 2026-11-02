"""Attack base class, I/O contract, and the attacker's local observation.

The simulator calls `attack.act(...)` once per tick for the compromised agent
and receives an `AttackOutput`:

  delta_u           bounded malicious acceleration added to the attacker's own
                    nominal command (||delta_u|| <= epsilon)
  report_pos_bias   additive bias on the *reported* position the attacker
                    broadcasts (false-data injection); honest = zeros
  report_vel_bias   additive bias on the reported velocity
  bypass_self_filter whether the attacker ignores its own CBF safety filter
  triggered         whether the attack policy considered itself active this tick

Honest behaviour (A0) sets delta_u = 0, zero biases, bypass_self_filter = False.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AttackOutput:
    delta_u: np.ndarray = field(default_factory=lambda: np.zeros(3))
    report_pos_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    report_vel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bypass_self_filter: bool = False
    triggered: bool = False


def local_observation(
    attacker: int,
    positions: np.ndarray,
    velocities: np.ndarray,
    offsets: np.ndarray,
    radius: float,
) -> dict:
    """Everything the attacker can realistically compute from local messages."""
    from ..graph import local_connectivity

    p_rel = positions[attacker] - positions
    v_rel = velocities[attacker] - velocities
    dist = np.linalg.norm(p_rel, axis=1)
    dist[attacker] = np.inf
    nbr = np.nonzero(dist <= radius)[0]

    if len(nbr) == 0:
        return {
            "neighbor_count": 0,
            "density": 0.0,
            "min_distance": float("inf"),
            "mean_closing_rate": 0.0,
            "compression": 0.0,
            "local_lambda2": 0.0,
            "neighbor_centroid": positions[attacker].copy(),
            "neighbors": nbr,
        }

    pr, vr, dd = p_rel[nbr], v_rel[nbr], dist[nbr]
    # closing rate < 0 means approaching
    closing = (pr * vr).sum(axis=1) / np.maximum(dd, 1e-9)
    desired = np.linalg.norm(offsets[attacker] - offsets[nbr], axis=1)
    compression = float(np.mean(desired - dd))        # >0 => tighter than desired
    density = float(np.mean(1.0 / np.maximum(dd, 1e-6)))
    centroid = positions[nbr].mean(axis=0)

    return {
        "neighbor_count": int(len(nbr)),
        "density": density,
        "min_distance": float(dd.min()),
        "mean_closing_rate": float(closing.mean()),
        "compression": compression,
        "local_lambda2": float(local_connectivity(attacker, positions, radius)),
        "neighbor_centroid": centroid,
        "neighbors": nbr,
    }


class Attack:
    """Base attacker. Subclasses override `act`."""

    def __init__(self, attacker: int = 0, epsilon: float = 2.0,
                 neighbor_radius: float = 10.0, rng=None, **kw):
        self.attacker = attacker
        self.epsilon = float(epsilon)
        self.neighbor_radius = float(neighbor_radius)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.offsets = kw.get("offsets")

    def _clip(self, v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n > self.epsilon:
            return v * (self.epsilon / n)
        return v

    def reset(self):
        pass

    def act(self, t, positions, velocities, nominal_self) -> AttackOutput:
        raise NotImplementedError


class NoAttack(Attack):
    """A0 reference: the designated agent stays honest."""

    def act(self, t, positions, velocities, nominal_self, **kw) -> AttackOutput:
        return AttackOutput()
