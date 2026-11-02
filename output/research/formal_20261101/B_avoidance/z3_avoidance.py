#!/usr/bin/env python3
"""Z3 checks for manuscript B (run: PYTHONPATH=<z3 dir> python3 z3_avoidance.py).  `PROVED` = Z3 found no counterexample over all
real values satisfying the hypotheses; `WITNESS FOUND` = a concrete state exhibiting the stated failure.

Planar relative state of honest i and neighbour j: r = p_i - p_j, v = v_i - v_j, accelerations a_i (commanded, ||a_i|| <= U) and
a_j (neighbour, unknown).  Barrier h = ||r||^2 - D^2, psi1 = hdot + k1 h, psi2 = psi1dot + k2 psi1 (exponential HOCBF, Xiao & Belta).
    psi2 = 2||v||^2 + 2 r.(a_i - a_j) + (k1 + k2) 2 r.v + k1 k2 h.
Robust row (RHOCBF, swarmsec/safety.py accel_bound): 2 r.a_i + 2||v||^2 + (k1 + k2) 2 r.v + k1 k2 h - 2||r|| abar >= 0.

B1  soundness with exact information: robust row  and  ||a_j|| <= abar  ==>  psi2 >= 0.
B2  falsification breaks B1: the row evaluated on a REPORTED state (r + e_p, v + e_v), ||e_p|| <= beta_p, ||e_v|| <= beta_v, can hold
    while the true psi2 < 0 (witness at the declared beta_p = 1.25 m, beta_v = 0.5 m/s).
B3a ISSf algebra: soft row (+ xi) and ||a_j|| <= abar  ==>  psi2 >= -xi.
B3b feasibility limit: on the boundary (h = 0, psi1 = 0) with r.v = 0 the row needs U >= abar - ||v||^2/||r||; with abar > U and
    ||v|| small there is NO admissible a_i (||a_i|| <= U): slack is forced (witness).
B4  blocking: at a cross-section of half-width w occupied at its centre by the attacker, every passing point is within w of it;
    with w < d_safe, passage and separation >= d_safe are incompatible.
"""
import json
import sys
from pathlib import Path

from z3 import And, Not, Real, Solver, sat, unsat

RES = {}


def prove(name, claim, *hyp):
    s = Solver()
    s.set("timeout", 600000)
    s.add(*hyp)
    s.add(Not(claim))
    r = s.check()
    RES[name] = {"result": "PROVED" if r == unsat else ("COUNTEREXAMPLE" if r == sat else "UNKNOWN"),
                 "model": str(s.model()) if r == sat else None}
    print(name, RES[name]["result"], RES[name]["model"] or "", flush=True)


def witness(name, *cons):
    s = Solver()
    s.set("timeout", 600000)
    s.add(*cons)
    r = s.check()
    RES[name] = {"result": "WITNESS FOUND" if r == sat else ("NO WITNESS" if r == unsat else "UNKNOWN"),
                 "model": str(s.model()) if r == sat else None}
    print(name, RES[name]["result"], RES[name]["model"] or "", flush=True)


rx, ry, vx, vy, aix, aiy, ajx, ajy = (Real(n) for n in ("rx", "ry", "vx", "vy", "aix", "aiy", "ajx", "ajy"))
k1, k2, D, abar, U, xi, nr = (Real(n) for n in ("k1", "k2", "D", "abar", "U", "xi", "nr"))
gains = [k1 > 0, k2 > 0, D > 0, abar >= 0, nr >= 0, nr * nr == rx * rx + ry * ry]


def dot(ax, ay, bx, by):
    return ax * bx + ay * by


def psi2(rx_, ry_, vx_, vy_, ajx_, ajy_):
    h = dot(rx_, ry_, rx_, ry_) - D * D
    return 2 * dot(vx_, vy_, vx_, vy_) + 2 * dot(rx_, ry_, aix - ajx_, aiy - ajy_) + (k1 + k2) * 2 * dot(rx_, ry_, vx_, vy_) + k1 * k2 * h


def row(rx_, ry_, vx_, vy_, nr_):
    h = dot(rx_, ry_, rx_, ry_) - D * D
    return 2 * dot(rx_, ry_, aix, aiy) + 2 * dot(vx_, vy_, vx_, vy_) + (k1 + k2) * 2 * dot(rx_, ry_, vx_, vy_) + k1 * k2 * h - 2 * nr_ * abar


aj_bounded = dot(ajx, ajy, ajx, ajy) <= abar * abar
# B1
prove("B1_robust_row_sound_exact_info", psi2(rx, ry, vx, vy, ajx, ajy) >= 0, *gains, aj_bounded, row(rx, ry, vx, vy, nr) >= 0)
# B3a
prove("B3a_soft_row_ISSf_algebra", psi2(rx, ry, vx, vy, ajx, ajy) >= -xi, *gains, xi >= 0, aj_bounded, row(rx, ry, vx, vy, nr) + xi >= 0)
# B2: reported state = true + error; row holds on the report with the report's own norm; true psi2 < 0
epx, epy, evx, evy, nh = (Real(n) for n in ("epx", "epy", "evx", "evy", "nh"))
hx, hy, hvx, hvy = rx + epx, ry + epy, vx + evx, vy + evy
witness("B2_falsification_breaks_robust_row",
        k1 == 1, k2 == 1, D == 1.5, abar == 2.5, dot(epx, epy, epx, epy) <= 1.25 ** 2, dot(evx, evy, evx, evy) <= 0.5 ** 2,
        nh >= 0, nh * nh == dot(hx, hy, hx, hy), dot(aix, aiy, aix, aiy) <= 9, dot(ajx, ajy, ajx, ajy) <= 2.5 ** 2,
        dot(rx, ry, rx, ry) >= 1.5 ** 2,                       # truly still outside the safe radius
        row(hx, hy, hvx, hvy, nh) >= 0, psi2(rx, ry, vx, vy, ajx, ajy) < 0)
# B3b: boundary state, abar > U, no a_i with ||a_i|| <= U satisfies the row
witness("B3b_row_infeasible_when_abar_exceeds_U",
        k1 == 1, k2 == 1, D == 1.5, U == 3, abar == 5.5, rx == 1.5, ry == 0, vx == 0, vy == 0, nr == 1.5,
        dot(aix, aiy, aix, aiy) <= U * U, row(rx, ry, vx, vy, nr) >= 0)          # expected: NO WITNESS (= infeasible)
# B4: cross-section {x = 0, |y| <= w}, attacker at the centre (0, 0); every point of the section is within w
w, py, ds = Real("w"), Real("py"), Real("ds")
prove("B4_blocking_every_passing_point_within_w", py * py <= w * w, w > 0, py >= -w, py <= w)
prove("B4_blocking_incompatible_if_w_lt_dsafe", Not(py * py >= ds * ds), w > 0, ds > w, py >= -w, py <= w)

Path(__file__).with_name("z3_avoidance_results.json").write_text(json.dumps(RES, indent=1))
ok = (RES["B1_robust_row_sound_exact_info"]["result"] == "PROVED" and RES["B3a_soft_row_ISSf_algebra"]["result"] == "PROVED"
      and RES["B2_falsification_breaks_robust_row"]["result"] == "WITNESS FOUND"
      and RES["B3b_row_infeasible_when_abar_exceeds_U"]["result"] == "NO WITNESS")
sys.exit(0 if ok else 1)
