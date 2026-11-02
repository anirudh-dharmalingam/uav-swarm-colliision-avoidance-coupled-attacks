"""Worst-case WHITE-BOX adaptive attackers (defense-aware evaluation).

Standard practice for security claims is to evaluate against an adaptive adversary
that knows the defense. These attackers read the defense's normalised alarm
evidence for the compromised UAV (`Defense.evidence`) -- information a real
attacker would have to estimate -- and modulate the amplitude of the collective
compression action so the evidence hovers just below the alarm level:

    alpha = clip((rho - e) / band, 0, 1),   delta_u = alpha * eps * direction

  A10  evades whatever the deployed defense measures (margin CUSUM and/or
       innovation residual): the strongest attacker in this repository.
  A12  evades only the model-based innovation channel (D1/D2/D5); it ignores the
       margin layer, to show that layer's complementary coverage.

Because the evidence is the defense's own alarm statistic, the damage these
attackers inflict while staying below `rho` is an empirical estimate of the
residual risk of that defense (a bound on what an undetected agent can do).
"""

from __future__ import annotations

import numpy as np

from .base import AttackOutput
from .collective import CompressionAttack


class WhiteBoxEvader(CompressionAttack):
    whitebox = True

    def __init__(self, attacker=0, epsilon=2.0, neighbor_radius=10.0, rng=None,
                 rho=0.8, band=0.3, trigger_ctx=None, **kw):
        super().__init__(attacker, epsilon, neighbor_radius, rng, trigger="T7",
                         target="centroid", trigger_ctx=trigger_ctx, **kw)
        self.rho, self.band = float(rho), float(band)
        self.evidence = 0.0

    def act(self, t, positions, velocities, nominal_self, dt=0.05, global_lambda2=0.0):
        out = super().act(t, positions, velocities, nominal_self, dt=dt,
                          global_lambda2=global_lambda2)
        if not out.triggered:
            return out
        alpha = float(np.clip((self.rho - self.evidence) / max(self.band, 1e-6), 0.0, 1.0))
        out.delta_u = out.delta_u * alpha
        out.triggered = alpha > 0.0
        return out


class InnovationEvader(WhiteBoxEvader):
    """A12: same policy; the simulator feeds it only the model-based evidence."""
    whitebox = True
    channel = "innovation"
