"""The proposed collective safety-feasibility-erosion attack (A8) and the
trigger-parameterised CompressionAttack used for the T1..T7 ablation.

Mechanism (no direct attack on any victim's CBF, weights, or network):

    locally-bounded malicious action  ->  neighbour-geometry perturbation
    ->  victim safety-margin erosion  ->  victim CBF-QP saturation
    ->  topology change               ->  swarm-level degradation

When its trigger says the neighbourhood is collectively vulnerable, the attacker
applies a small acceleration that compresses its local geometry; otherwise it
flies nominally, which keeps the time-averaged deviation (stealth cost) low.
"""

from __future__ import annotations

import numpy as np

from .base import Attack, AttackOutput, local_observation
from .triggers import make_trigger


class CompressionAttack(Attack):
    """Compression action gated by a pluggable trigger (used for the ablation).

    target: 'centroid' compresses the whole neighbourhood (collective);
            'nearest' compresses a single edge (local, non-adaptive).
    """

    def __init__(self, attacker=0, epsilon=2.0, neighbor_radius=10.0, rng=None,
                 trigger="T7", target="centroid", trigger_ctx=None, **kw):
        super().__init__(attacker, epsilon, neighbor_radius, rng, **kw)
        self.trigger_name = trigger.upper()
        self.trigger = make_trigger(self.trigger_name)
        self.target = target
        self.ctx = dict(trigger_ctx or {})
        self.ctx.setdefault("rng", self.rng)

    def act(self, t, positions, velocities, nominal_self, dt=0.05, global_lambda2=0.0):
        obs = local_observation(self.attacker, positions, velocities,
                                self.offsets, self.neighbor_radius)
        self.ctx["global_lambda2"] = global_lambda2
        fired = bool(obs["neighbor_count"] > 0 and self.trigger(obs, t, dt, self.ctx))
        if not fired:
            return AttackOutput(triggered=False)

        if self.target == "nearest" and len(obs["neighbors"]):
            nbr = obs["neighbors"]
            d = np.linalg.norm(positions[nbr] - positions[self.attacker], axis=1)
            target_pt = positions[nbr[int(np.argmin(d))]]
        else:
            target_pt = obs["neighbor_centroid"]

        direction = target_pt - positions[self.attacker]
        nrm = float(np.linalg.norm(direction))
        if nrm < 1e-6:
            return AttackOutput(triggered=True)
        delta_u = self._clip(self.epsilon * direction / nrm)
        return AttackOutput(delta_u=delta_u, bypass_self_filter=True, triggered=True)


class CollectiveMarginAttack(CompressionAttack):
    """A8: CompressionAttack locked to the proposed local collective trigger."""

    def __init__(self, attacker=0, epsilon=2.0, neighbor_radius=10.0, rng=None,
                 trigger_ctx=None, **kw):
        super().__init__(attacker, epsilon, neighbor_radius, rng,
                         trigger="T7", target="centroid",
                         trigger_ctx=trigger_ctx, **kw)


class StealthRampAttack(CompressionAttack):
    """A9: defense-aware low-and-slow adaptive attacker (resilience stress test).

    Knows the defense monitors QP-stress/blame, so it ramps its action slowly
    after the trigger first fires, aiming to keep per-tick stress under the
    detection baseline while still eroding margins over time.
    """

    def __init__(self, attacker=0, epsilon=2.0, neighbor_radius=10.0, rng=None,
                 ramp_s=10.0, trigger_ctx=None, **kw):
        super().__init__(attacker, epsilon, neighbor_radius, rng, trigger="T7",
                         target="centroid", trigger_ctx=trigger_ctx, **kw)
        self.ramp_s = float(ramp_s)
        self._first = None

    def reset(self):
        self._first = None

    def act(self, t, positions, velocities, nominal_self, dt=0.05, global_lambda2=0.0):
        out = super().act(t, positions, velocities, nominal_self, dt=dt,
                          global_lambda2=global_lambda2)
        if not out.triggered:
            return out
        if self._first is None:
            self._first = t
        frac = min(1.0, (t - self._first) * dt / max(self.ramp_s, 1e-6))
        out.delta_u = out.delta_u * frac
        return out
