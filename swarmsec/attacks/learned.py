"""A7: learned adaptive attacker (MLP policy over locally observable features).

The policy sees only what a compromised endpoint can compute from the messages it
receives (neighbour count/density/closest distance/closing rate/compression/local
lambda_2/direction to the neighbour centroid/own speed). It outputs a bounded
acceleration (gate x direction x epsilon). Weights are trained offline by
evolution strategies (scripts/sim/train_a7.py) against the episode reward

    J = unsafe_time_fraction + 2 * QP_infeasibility_rate - lambda * effort / epsilon

and deployed as a tiny NumPy forward pass. The policy is warm-started with a
skip connection that points at the neighbour centroid, so ES refines *when* and
*how hard* to act rather than rediscovering the geometry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Attack, AttackOutput, local_observation

OBS_DIM = 12
HIDDEN = 16
WEIGHTS = Path(__file__).with_name("a7_weights.npy")


def init_params(rng=None, scale=0.05):
    rng = rng if rng is not None else np.random.default_rng(0)
    w1 = scale * rng.normal(size=(HIDDEN, OBS_DIM))
    b1 = np.zeros(HIDDEN)
    w2 = scale * rng.normal(size=(4, HIDDEN))
    b2 = np.array([0.0, 0.0, 0.0, 0.0])
    skip = np.zeros((4, OBS_DIM))
    skip[0:3, 6:9] = 2.0 * np.eye(3)          # direction follows unit vector to centroid
    skip[3, 0] = 0.0
    b2[3] = 0.0                                # gate starts at 0.5
    return pack(w1, b1, w2, b2, skip)


def pack(w1, b1, w2, b2, skip):
    return np.concatenate([w1.ravel(), b1, w2.ravel(), b2, skip.ravel()])


def unpack(theta):
    i = 0
    w1 = theta[i:i + HIDDEN * OBS_DIM].reshape(HIDDEN, OBS_DIM); i += HIDDEN * OBS_DIM
    b1 = theta[i:i + HIDDEN]; i += HIDDEN
    w2 = theta[i:i + 4 * HIDDEN].reshape(4, HIDDEN); i += 4 * HIDDEN
    b2 = theta[i:i + 4]; i += 4
    skip = theta[i:i + 4 * OBS_DIM].reshape(4, OBS_DIM)
    return w1, b1, w2, b2, skip


N_PARAMS = HIDDEN * OBS_DIM + HIDDEN + 4 * HIDDEN + 4 + 4 * OBS_DIM


def features(obs, own_pos, own_vel):
    d = obs["neighbor_centroid"] - own_pos
    n = float(np.linalg.norm(d))
    u = d / n if n > 1e-6 else np.zeros(3)
    return np.array([
        obs["neighbor_count"] / 8.0, obs["density"] / 0.3, obs["min_distance"] / 6.0
        if np.isfinite(obs["min_distance"]) else 1.0,
        obs["mean_closing_rate"], obs["compression"] / 1.5, obs["local_lambda2"] / 5.0,
        u[0], u[1], u[2], float(np.linalg.norm(own_vel)) / 2.0, 1.0, 0.0,
    ])


def policy(theta, x):
    w1, b1, w2, b2, skip = unpack(theta)
    h = np.tanh(w1 @ x + b1)
    o = w2 @ h + b2 + skip @ x
    gate = 1.0 / (1.0 + np.exp(-o[3]))
    d = np.tanh(o[:3])
    return gate, d


class LearnedAttack(Attack):
    def __init__(self, attacker=0, epsilon=2.0, neighbor_radius=10.0, rng=None,
                 theta=None, gate_thresh=0.15, **kw):
        super().__init__(attacker, epsilon, neighbor_radius, rng, **kw)
        if theta is None:
            theta = np.load(WEIGHTS) if WEIGHTS.exists() else init_params()
        self.theta = np.asarray(theta, float)
        self.gate_thresh = gate_thresh

    def act(self, t, positions, velocities, nominal_self, dt=0.05, global_lambda2=0.0):
        obs = local_observation(self.attacker, positions, velocities, self.offsets, self.neighbor_radius)
        if obs["neighbor_count"] == 0:
            return AttackOutput()
        x = features(obs, positions[self.attacker], velocities[self.attacker])
        gate, d = policy(self.theta, x)
        if gate < self.gate_thresh:
            return AttackOutput()
        nrm = max(float(np.linalg.norm(d)), 1.0)
        return AttackOutput(delta_u=self._clip(self.epsilon * gate * d / nrm),
                            bypass_self_filter=True, triggered=True)
