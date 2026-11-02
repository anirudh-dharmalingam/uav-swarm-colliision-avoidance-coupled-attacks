"""A13: randomised attack family and the informed-attacker search space.

One compromised UAV draws a full attack *configuration* (mode, amplitude, onset,
direction, trigger thresholds, duty cycle, optional evidence-evasion level) from a
fixed distribution.  Evaluating a defense against many draws measures distributional
robustness rather than robustness to a few hand-picked attacks.

An *informed* attacker (A14 in the experiment runner) knows the defense family and the
swarm size, and has black-box query access to its own episodes: it runs a random search
over this same configuration space against the target defense on its own training seeds
and deploys the best configuration (scripts/sim/informed_search.py).  The family is
deliberately the same for both, so "informed" differs from "randomised" only by
optimisation against the defense, not by extra actuation authority: every configuration
respects ||delta_u|| <= epsilon and the self-report-only falsification of the threat model.

Modes
  bias      constant push along a fixed direction                (A2-like)
  compress  collective compression gated by the local trigger    (A8-like)
  periodic  compression on a duty cycle                          (A3-like)
  ramp      compression with a slow ramp-up                      (A9-like)
  fdi       outward-biased position report, honest actuation     (A4-like)
  hybrid    compression plus falsified position report
"""

from __future__ import annotations

import numpy as np

from .base import Attack, AttackOutput, local_observation
from .triggers import make_trigger

MODES = ("bias", "compress", "periodic", "ramp", "fdi", "hybrid")


def sample_params(rng, modes=MODES) -> dict:
    d = rng.normal(size=3)
    d[2] *= 0.3
    d = d / max(np.linalg.norm(d), 1e-9)
    return {
        "mode": str(rng.choice(list(modes))),
        "amp": float(rng.uniform(0.4, 1.0)),            # fraction of the budget epsilon
        "onset_s": float(rng.uniform(0.0, 8.0)),        # delay after the attack window opens
        "dir": [float(x) for x in d],
        "closing_thresh": float(rng.uniform(-0.4, 0.0)),
        "compression_thresh": float(rng.uniform(0.0, 1.0)),
        "density_thresh": float(rng.uniform(0.15, 0.35)),
        "period_s": float(rng.uniform(2.0, 8.0)),
        "duty": float(rng.uniform(0.2, 0.8)),
        "ramp_s": float(rng.uniform(3.0, 15.0)),
        "spoof": float(rng.uniform(0.5, 2.0)),
        "rho": float(rng.choice([0.0, 0.0, 0.5, 0.8])),   # evidence-evasion level (0 = not informed of alarms)
    }


def perturb_params(p: dict, rng, scale=0.3) -> dict:
    """Local move for the informed search (keeps the configuration inside the family)."""
    q = dict(p)
    r = lambda lo, hi, v: float(np.clip(v + scale * (hi - lo) * rng.normal(), lo, hi))
    q["amp"] = r(0.2, 1.0, p["amp"])
    q["onset_s"] = r(0.0, 12.0, p["onset_s"])
    q["closing_thresh"] = r(-0.6, 0.1, p["closing_thresh"])
    q["compression_thresh"] = r(-0.2, 1.5, p["compression_thresh"])
    q["density_thresh"] = r(0.1, 0.4, p["density_thresh"])
    q["period_s"] = r(1.0, 10.0, p["period_s"])
    q["duty"] = r(0.1, 1.0, p["duty"])
    q["ramp_s"] = r(1.0, 20.0, p["ramp_s"])
    q["spoof"] = r(0.2, 2.5, p["spoof"])
    d = np.asarray(p["dir"]) + scale * rng.normal(size=3)
    q["dir"] = [float(x) for x in d / max(np.linalg.norm(d), 1e-9)]
    if rng.random() < 0.15:
        q["mode"] = str(rng.choice(MODES))
    if rng.random() < 0.15:
        q["rho"] = float(rng.choice([0.0, 0.5, 0.8]))
    return q


class RandomizedAttack(Attack):
    whitebox = True        # simulator feeds the defense's alarm evidence; used only if rho > 0
    channel = None

    def __init__(self, attacker=0, epsilon=2.0, neighbor_radius=10.0, rng=None,
                 params=None, **kw):
        super().__init__(attacker, epsilon, neighbor_radius, rng, **kw)
        self.p = dict(params) if params is not None else sample_params(self.rng)
        self.trigger = make_trigger("T7")
        self.ctx = {"rng": self.rng, "closing_thresh": self.p["closing_thresh"],
                    "compression_thresh": self.p["compression_thresh"],
                    "density_thresh": self.p["density_thresh"]}
        self.ctx.update({k: self.p[k] for k in ("k0", "lambda_thresh") if k in self.p})   # RAL: always-on gating
        self.evidence = 0.0
        self._first = None

    def reset(self):
        self._first = None

    def _compress_dir(self, obs, positions):
        direction = obs["neighbor_centroid"] - positions[self.attacker]
        n = float(np.linalg.norm(direction))
        return direction / n if n > 1e-6 else None

    def act(self, t, positions, velocities, nominal_self, dt=0.05, global_lambda2=0.0):
        p, now = self.p, t * dt
        if now < 5.0 + p["onset_s"]:
            return AttackOutput(triggered=False)
        amp = p["amp"] * self.epsilon
        if p["rho"] > 0.0:                        # evade the defense's alarm statistic
            amp *= float(np.clip((p["rho"] - self.evidence) / 0.3, 0.0, 1.0))
            if amp <= 0.0:
                return AttackOutput(triggered=False)
        mode = p["mode"]
        beta_p = p.get("beta_pos_m")          # RAL: independent, explicit falsification budgets
        beta_v = p.get("beta_vel_mps", 0.0)
        if mode == "bias":
            return AttackOutput(delta_u=self._clip(amp * np.asarray(p["dir"])),
                                bypass_self_filter=True, triggered=True)
        obs = local_observation(self.attacker, positions, velocities,
                                self.offsets, self.neighbor_radius)
        if obs["neighbor_count"] == 0:
            return AttackOutput(triggered=False)
        if mode == "fdi":
            out = positions[self.attacker] - obs["neighbor_centroid"]
            n = max(float(np.linalg.norm(out)), 1e-9)
            if beta_p is not None:
                return AttackOutput(report_pos_bias=beta_p * out / n, report_vel_bias=beta_v * out / n,
                                    triggered=True)
            return AttackOutput(report_pos_bias=p["spoof"] * self.epsilon * out / n, triggered=True)
        if mode == "periodic":
            if ((now - 5.0 - p["onset_s"]) % p["period_s"]) > p["duty"] * p["period_s"]:
                return AttackOutput(triggered=False)
        else:                                      # compress / ramp / hybrid are trigger-gated
            if not self.trigger(obs, t, dt, self.ctx):
                return AttackOutput(triggered=False)
        u = self._compress_dir(obs, positions)
        if u is None:
            return AttackOutput(triggered=True)
        if mode == "ramp":
            if self._first is None:
                self._first = now
            amp *= min(1.0, (now - self._first) / max(p["ramp_s"], 1e-6))
        out = AttackOutput(delta_u=self._clip(amp * u), bypass_self_filter=True, triggered=True)
        if mode == "hybrid":
            o = positions[self.attacker] - obs["neighbor_centroid"]
            on = max(float(np.linalg.norm(o)), 1e-9)
            if beta_p is not None:
                out.report_pos_bias = beta_p * o / on
                out.report_vel_bias = beta_v * o / on
            else:
                out.report_pos_bias = p["spoof"] * self.epsilon * o / on
        return out
