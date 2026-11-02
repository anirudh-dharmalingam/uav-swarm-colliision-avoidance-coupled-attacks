"""Coupled-safety campaign extensions to the RAL26 per-agent policy (information interventions; no change to historical behaviour).

`CsPolicy` = `AgentPolicy` whose envelopes honour the E2 information modes while keeping the SAME hard rows, solver, actuator ball,
slew, bounds and fallback:

  standard   report envelopes only (build_envelope_sound, every neighbour treated as possibly compromised)
  identity   the compromised id is known from t=0: honest senders' packets are built with beta = 0 and A_j = A_honest (justified
             tightening; identity does NOT reveal the attacker's geometry, its packet keeps beta and A_any)
  oracle     DIAGNOSTIC: packet content replaced by the true state at send time, ego replaced by the true ego state, beta = sender
             error = ego error = 0; packet timing, A_j, W, solver unchanged
  sensor     EMULATED trusted relative sensor, fused with the report envelope by VALID enclosing-set intersection (the smaller of two
             enclosing balls encloses their intersection); a contradiction (disjoint balls) flags INVALID_ASSUMPTIONS and uses the
             larger ball -- never an artificially tiny one
"""
from __future__ import annotations

import math

import numpy as np

from . import ral26_core as rc
from .ral26_policy import AgentPolicy


def sensor_envelope(sender, r_meas, v_meas, s, lat_min, lat_max, eps_p, eps_v, D_rel, A_j):
    """Trusted RELATIVE measurement (r = p_i - p_j, v) acquired a in [s + lat_min, s + lat_max] seconds ago.
    |r(t) - (r_meas + a_c v_meas)| <= eps_p + a_+ eps_v + ((lat_max-lat_min)/2)||v_meas|| + D_rel a_+^2/2,   a_c midpoint age
    |v(t) - v_meas| <= eps_v + D_rel a_+,    D_rel = A_j + U + W bounds the relative acceleration (neighbour + ego)."""
    a_c = s + 0.5 * (lat_min + lat_max)
    a_p = s + lat_max
    Rp = eps_p + a_p * eps_v + 0.5 * (lat_max - lat_min) * float(np.linalg.norm(v_meas)) + 0.5 * D_rel * a_p * a_p
    Rv = eps_v + D_rel * a_p
    return rc.Envelope(sender, np.asarray(r_meas, float) + a_c * np.asarray(v_meas, float), np.asarray(v_meas, float), Rp, Rv, A_j, a_p, True)


def intersect_envelopes(e1: rc.Envelope, e2: rc.Envelope, detail=False):
    """Valid enclosing set of the intersection of two enclosures of the same (r, v).  Returns (envelope, contradiction)
    (and, with detail=True, a third value: (position_disjoint, velocity_disjoint))."""
    bad = False
    bad_p = bad_v = False
    cp, Rp = (e1.r_hat, e1.B_p) if e1.B_p <= e2.B_p else (e2.r_hat, e2.B_p)
    if float(np.linalg.norm(e1.r_hat - e2.r_hat)) > e1.B_p + e2.B_p + 1e-9:
        bad = bad_p = True
        cp, Rp = (e1.r_hat, e1.B_p) if e1.B_p >= e2.B_p else (e2.r_hat, e2.B_p)
    cv, Rv = (e1.v_hat, e1.B_v) if e1.B_v <= e2.B_v else (e2.v_hat, e2.B_v)
    if float(np.linalg.norm(e1.v_hat - e2.v_hat)) > e1.B_v + e2.B_v + 1e-9:
        bad = bad_v = True
        cv, Rv = (e1.v_hat, e1.B_v) if e1.B_v >= e2.B_v else (e2.v_hat, e2.B_v)
    env = rc.Envelope(e1.sender, cp, cv, Rp, Rv, e1.A_j, max(e1.age, e2.age), True)
    return (env, bad, (bad_p, bad_v)) if detail else (env, bad)


class CsPolicy(AgentPolicy):
    """info: object with attributes mode, sensor (dict j -> (r_meas, v_meas, t_rx)), sensor_cfg dict, fusion_events list."""

    def __init__(self, *a, info=None, **k):
        super().__init__(*a, **k)
        self.info = info
        self.contradictions = 0
        self.contra_p = 0
        self.contra_v = 0
        self.fused = 0
        self.sensor_used = 0

    def envelopes(self, t, p_own, v_own, packets):
        B, mode = self.B, (self.info.mode if self.info is not None else "standard")
        if B.envelope != "sound_age":
            return super().envelopes(t, p_own, v_own, packets)
        envs, valid = [], True
        ego_p, ego_v, sp, sv = B.ego_err_p, B.ego_err_v, B.sender_err_p, B.sender_err_v
        if mode == "oracle":
            ego_p = ego_v = sp = sv = 0.0
        for j, pkt in packets.items():
            e = rc.build_envelope_sound(pkt, t, p_own, v_own, ego_p, ego_v, sp, sv)
            if not e.valid:
                valid = False
            if mode == "sensor" and e.valid:
                m = self.info.sensor.get(j)
                S = self.info.sensor_cfg
                if m is not None:
                    r_meas, v_meas, t_rx = m
                    s = max(0.0, t - t_rx)
                    if s <= S.get("max_age_s", 1.0):
                        es = sensor_envelope(j, r_meas, v_meas, s, S["lat_min"], S["lat_max"], S["pos_err"], S["vel_err"],
                                             pkt.A_j + B.U + B.W, pkt.A_j)
                        e2, bad, (bp_, bv_) = intersect_envelopes(e, es, detail=True)
                        self.contra_p += int(bp_)
                        self.contra_v += int(bv_)
                        self.fused += 1
                        self.sensor_used += int(e2.B_p < e.B_p - 1e-12)
                        if bad:
                            self.contradictions += 1
                            valid = False
                        e = e2
            envs.append(e)
        return envs, valid
