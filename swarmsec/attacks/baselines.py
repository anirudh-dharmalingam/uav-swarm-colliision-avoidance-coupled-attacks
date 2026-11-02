"""Baseline attackers A1..A6 (orthogonal to the proposed A8).

All respect the threat model: bounded self-actuation and/or falsified self
reports only. They differ in *what information they use* and *how adaptive*
they are, which is exactly what the attack ablation isolates.
"""

from __future__ import annotations

import numpy as np

from .base import Attack, AttackOutput, local_observation


class RandomPerturbation(Attack):
    """A1: naive attacker, bounded random acceleration every tick."""

    def act(self, t, positions, velocities, nominal_self, **kw):
        d = self.rng.normal(size=3)
        d = self._clip(self.epsilon * d / max(np.linalg.norm(d), 1e-9))
        return AttackOutput(delta_u=d, bypass_self_filter=True, triggered=True)


class ConstantBias(Attack):
    """A2: classical constant actuator bias along a fixed direction."""

    def __init__(self, *a, direction=(1.0, 0.0, 0.0), **kw):
        super().__init__(*a, **kw)
        v = np.asarray(direction, float)
        self.dir = v / max(np.linalg.norm(v), 1e-9)

    def act(self, t, positions, velocities, nominal_self, **kw):
        return AttackOutput(delta_u=self._clip(self.epsilon * self.dir),
                            bypass_self_filter=True, triggered=True)


class PeriodicTrigger(Attack):
    """A3: scheduling-only trigger; pushes toward neighbour centroid on a duty cycle."""

    def __init__(self, *a, period=4.0, duty=0.5, dt=0.05, **kw):
        super().__init__(*a, **kw)
        self.period = period
        self.duty = duty
        self.dt = dt

    def act(self, t, positions, velocities, nominal_self, dt=0.05, **kw):
        phase = (t * dt) % self.period
        if phase > self.duty * self.period:
            return AttackOutput(triggered=False)
        obs = local_observation(self.attacker, positions, velocities,
                                self.offsets, self.neighbor_radius)
        if obs["neighbor_count"] == 0:
            return AttackOutput(triggered=True)
        direction = obs["neighbor_centroid"] - positions[self.attacker]
        n = max(np.linalg.norm(direction), 1e-9)
        return AttackOutput(delta_u=self._clip(self.epsilon * direction / n),
                            bypass_self_filter=True, triggered=True)


class FalseDataInjection(Attack):
    """A4: cyber-only. Reports a position biased outward from the swarm so victims
    under-estimate proximity and relax their CBF, while the attacker flies on."""

    def __init__(self, *a, spoof_gain=1.5, **kw):
        super().__init__(*a, **kw)
        self.spoof_gain = spoof_gain

    def act(self, t, positions, velocities, nominal_self, **kw):
        obs = local_observation(self.attacker, positions, velocities,
                                self.offsets, self.neighbor_radius)
        if obs["neighbor_count"] == 0:
            return AttackOutput(triggered=False)
        # Report being farther from the local centroid than it truly is.
        outward = positions[self.attacker] - obs["neighbor_centroid"]
        n = max(np.linalg.norm(outward), 1e-9)
        bias = self.spoof_gain * self.epsilon * outward / n
        return AttackOutput(report_pos_bias=bias, bypass_self_filter=False, triggered=True)


class ReplayDelay(Attack):
    """A5: network/state inconsistency. Broadcasts a stale (delayed) self state."""

    def __init__(self, *a, delay_ticks=20, **kw):
        super().__init__(*a, **kw)
        self.delay = int(delay_ticks)
        self._buf = []

    def reset(self):
        self._buf = []

    def act(self, t, positions, velocities, nominal_self, **kw):
        self._buf.append((positions[self.attacker].copy(), velocities[self.attacker].copy()))
        if len(self._buf) <= self.delay:
            return AttackOutput(triggered=False)
        stale_p, stale_v = self._buf[-self.delay - 1]
        return AttackOutput(
            report_pos_bias=stale_p - positions[self.attacker],
            report_vel_bias=stale_v - velocities[self.attacker],
            bypass_self_filter=False,
            triggered=True,
        )


class FixedStateTrigger(Attack):
    """A6: state-triggered but NON-adaptive. When the nearest neighbour crosses a
    fixed distance threshold, compress that single edge. Trigger without the
    collective adaptation of A8."""

    def __init__(self, *a, dist_thresh=4.0, **kw):
        super().__init__(*a, **kw)
        self.dist_thresh = dist_thresh

    def act(self, t, positions, velocities, nominal_self, **kw):
        obs = local_observation(self.attacker, positions, velocities,
                                self.offsets, self.neighbor_radius)
        if obs["neighbor_count"] == 0 or obs["min_distance"] > self.dist_thresh:
            return AttackOutput(triggered=False)
        nbr = obs["neighbors"]
        d = np.linalg.norm(positions[nbr] - positions[self.attacker], axis=1)
        target = positions[nbr[int(np.argmin(d))]]
        direction = target - positions[self.attacker]
        n = max(np.linalg.norm(direction), 1e-9)
        return AttackOutput(delta_u=self._clip(self.epsilon * direction / n),
                            bypass_self_filter=True, triggered=True)
