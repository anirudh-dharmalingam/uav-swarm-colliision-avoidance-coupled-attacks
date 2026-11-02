"""B-A1 defence-aware adaptive attacker for the Layer-1 coupled-attack study (programme B-A1).

A parametric replacement for `uav_agent_node.Agent.attack_terms` on the compromised UAV only.  It keeps every declared budget of the
coupled threat model (experiments/coupled_safety_v1/bounds_declared.json), which are the same budgets the fixed attacks A8/A14/A15 use:
  physical   ||u|| <= U + epsilon                                  (A15 full-authority pursuit attains this bound)
  position   ||rep_p - p|| <= beta_attack                           (self-report only; no other node's packets are touched)
  velocity   ||rep_v - v|| <= beta_v_attack
It sees what the compromised node sees (its received packets, its own state) plus the public method description: it is white-box with
respect to the defence's DESIGN but has no access to other UAVs' internal state.  The parameter vector theta is searched per defence
(scripts/rarf_l1/adaptive_search.py), so the attack adapts to each defence offline; the online part is target choice and lead.

theta (all dimensionless or SI, ranges in BOUNDS):
  tgt      < 0.5: nearest received neighbour;  >= 0.5: neighbour with the smallest predicted time to closest approach
  lead     prediction horizon [s] for the aim point  p_j + lead (v_j - v)
  w_full   blend between nominal + epsilon push (0, = A14) and full-authority pursuit (1, = A15)
  r_trig   pursuit trigger radius [m]
  r_lie    report falsification starts only when the target is within r_lie [m] (late lying: less time for consistency monitors);
           r_lie = 14 (its upper bound) means always, as in A14
  phi_p    direction of the position lie, angle from "away from the target" (0 = A14) [rad]
  a_p      position-lie magnitude as a fraction of beta_attack
  phi_v    direction of the velocity lie, angle from "hide closing speed" (0 = A14) [rad]
  a_v      velocity-lie magnitude as a fraction of beta_v_attack
theta = A14_THETA reproduces A14 except for the A14 rule rep_v = v - min(v_c, beta_v) to (closing speed only), which is the a_v = 1,
phi_v = 0 member of this family only when the closing speed exceeds beta_v.
"""
from __future__ import annotations

import math

import numpy as np

BOUNDS = {"tgt": (0.0, 1.0), "lead": (0.0, 2.5), "w_full": (0.0, 1.0), "r_trig": (3.0, 14.0), "r_lie": (0.0, 14.0),
          "phi_p": (-math.pi, math.pi), "a_p": (0.0, 1.0), "phi_v": (-math.pi, math.pi), "a_v": (0.0, 1.0)}
KEYS = tuple(BOUNDS)
A14_THETA = {"tgt": 0.0, "lead": 0.0, "w_full": 0.0, "r_trig": 9.0, "r_lie": 14.0, "phi_p": 0.0, "a_p": 1.0, "phi_v": 0.0, "a_v": 1.0}


def clip_theta(th):
    return {k: float(min(max(float(th.get(k, A14_THETA[k])), lo), hi)) for k, (lo, hi) in BOUNDS.items()}


def _rot(x, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * x[0] - s * x[1], s * x[0] + c * x[1], 0.0])


def make_hook(theta, exclude=()):
    """Returns attacker_hook(agent) for swarmsec.rarf_l1.harness.run_episode (l1['attacker_hook']).  `exclude`: identities the attacker
    never targets (its co-attackers in multi-attacker episodes, B-N1)."""
    th = clip_theta(theta)
    excl = {int(x) for x in exclude}

    def hook(ag):
        import uav_agent_node as base

        def attack_terms(tm, p, v, u_safe, u_nom):
            rep_p, rep_v = p.copy(), v.copy()
            if ag.attack == "A0" or ag.t0 is None or tm < ag.att_start:
                return u_safe, rep_p, rep_v, False
            now = base.time.monotonic()
            best = None
            for j, q in ag.pk.items():
                if int(j) in excl:
                    continue
                age = max(0.0, now - q.t_rx)
                pj = q.p_msg + age * q.v_msg
                d = float(np.linalg.norm(pj - p))            # 3-D, as attack_terms
                r = pj - p
                r[2] = 0.0
                if th["tgt"] < 0.5:
                    key = d
                else:                                        # smallest predicted time to closest approach (closing targets first)
                    w = (q.v_msg - v)
                    w[2] = 0.0
                    vc = -float(r @ w) / max(float(np.linalg.norm(r)), 1e-9)        # closing speed > 0 when approaching
                    key = d / max(vc + ag.B.U, 1e-3)
                if best is None or key < best[0]:
                    best = (key, j, pj, q.v_msg, d)
            if best is None:
                return u_safe, rep_p, rep_v, False
            _, j, pj, vj, d = best
            aim = pj + th["lead"] * (vj - v)
            to = aim - p
            to[2] = 0.0
            to = to / max(float(np.linalg.norm(to)), 1e-9)
            u, active = u_safe, False
            U, eps = float(ag.B.U), float(ag.eps)
            if d <= th["r_trig"] and eps > 0:
                un = u_nom * min(1.0, U / max(float(np.linalg.norm(u_nom)), 1e-9))
                u = (1.0 - th["w_full"]) * (un + eps * to) + th["w_full"] * (U + eps) * to
                nu = float(np.linalg.norm(u))
                if nu > U + eps:                              # convex blend already satisfies this; guard rounding
                    u = u * (U + eps) / nu
                active = True
            if (d <= th["r_lie"] or th["r_lie"] >= BOUNDS["r_lie"][1]) and (ag.beta_att > 0 or ag.beta_v_att > 0):
                rep_p = p + th["a_p"] * float(ag.beta_att) * _rot(-to, th["phi_p"])
                vc = max(0.0, float(v @ to))
                mag_v = th["a_v"] * float(ag.beta_v_att)
                if th["phi_v"] == 0.0:
                    mag_v = min(vc, mag_v)                    # A14 member: hide only the closing speed actually present
                rep_v = v + mag_v * _rot(-to, th["phi_v"])
                active = True
            return u, rep_p, rep_v, active

        ag.attack_terms = attack_terms
        ag.adaptive_theta = th
    return hook
