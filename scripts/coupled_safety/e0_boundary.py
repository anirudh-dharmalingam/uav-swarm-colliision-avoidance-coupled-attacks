#!/usr/bin/env python3
"""E0: exact 1-D boundary validation (coupled-safety campaign).  No PX4; deterministic cases, not statistical trials.

Model (MATH_AND_ASSUMPTIONS.md, Prop. 1-2): clearance q = x_a - x_i - d, closing speed c = -q', unresponsive interval tau with
inward relative acceleration A_d, then honest maximum escape with net authority B = U - A > 0 (q'' = +B).  Worst corner of the
box uncertainty: q = q_hat - R_q, c = c_bar = max(0, c_hat + R_c).  M = q_hat - R_q - c_bar tau - A_d tau^2/2 - (c_bar + A_d tau)^2/(2B)
equals the minimum clearance margin of that corner trajectory.

    python3 scripts/coupled_safety/e0_boundary.py --out results/coupled_safety_v1/e0
Writes e0_cases.csv (1701 grid cases), e0_slew_cases.csv (200 paired boundary cases with a slew-limited actuator), e0_summary.json.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

B_GRID = (0.5, 1.0, 2.0)
C_GRID = (0.5, 1.0, 2.0)
TAU_GRID = (0.0, 0.25, 0.5)
RQ_GRID = (0.0, 0.5, 1.25)
RC_GRID = (0.0, 0.25, 0.5)
OFFSETS = (-0.5, -0.1, -0.02, 0.0, 0.02, 0.1, 0.5)
A_D = 1.0
TOL_ANALYTIC = 1e-9        # equality tolerance on the analytic margin (offset == 0 is reported as 'boundary')
TOL_NUMERIC = 2e-6         # equality tolerance on the integrated margin (RK4, dt = 1 ms, error ~ B dt^2 / 8)


def c_bar(c_hat, R_c):
    return max(0.0, c_hat + R_c)


def q_required(B, c_hat, tau, R_q, R_c, A_d=A_D):
    cb = c_bar(c_hat, R_c)
    return R_q + cb * tau + 0.5 * A_d * tau ** 2 + (cb + A_d * tau) ** 2 / (2 * B)


def margin_M(q_hat, B, c_hat, tau, R_q, R_c, A_d=A_D):
    cb = c_bar(c_hat, R_c)
    return q_hat - R_q - cb * tau - 0.5 * A_d * tau ** 2 - (cb + A_d * tau) ** 2 / (2 * B)


def accel_ideal(t, tau, B, A_d):
    return -A_d if t < tau else B                 # q'' (negative = closing)


def integrate_min(q0, c0, accel, t_end, dt, method="rk4"):
    """min clearance (q) of q' = -c, c' = -q'' ; accel(t) = q''(t).  State (q, w=q').  RK4 on a fixed grid."""
    n = int(round(t_end / dt))
    q, w = q0, -c0
    qmin = q
    for k in range(n):
        t = k * dt
        if method == "rk4":
            a1 = a2 = a3 = a4 = accel(t + 0.5 * dt)     # piecewise-constant input: one value per step (switches lie on the grid)
            k1q, k1w = w, a1
            k2q, k2w = w + 0.5 * dt * k1w, a2
            k3q, k3w = w + 0.5 * dt * k2w, a3
            k4q, k4w = w + dt * k3w, a4
            q += dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
            w += dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
        else:                                       # semi-implicit Euler (independent, first order)
            w += accel(t) * dt
            q += w * dt
        qmin = min(qmin, q)
    return qmin


def slew_accel(tau, B, A_d, J):
    """Relative acceleration with a slew-limited honest actuator: unresponsive until tau at -A_d, then the honest acceleration
    ramps linearly at jerk J from the delay value to +B and holds."""
    T_r = (A_d + B) / J
    def f(t):
        if t < tau:
            return -A_d
        s = t - tau
        return -A_d + J * s if s < T_r else B
    return f


def slew_min_clearance_analytic(q0, c0, tau, B, A_d, J):
    """Piecewise-polynomial minimum clearance with the ramp (reachable-tube extension of Prop. 1).  Returns min q."""
    q1 = q0 - c0 * tau - 0.5 * A_d * tau ** 2
    c1 = c0 + A_d * tau
    T_r = (A_d + B) / J
    qs = [q0, q1]
    # ramp phase: q'' = -A_d + J s ; q(s) = q1 - c1 s - A_d s^2/2 + J s^3/6 ; c(s) = c1 + A_d s - J s^2/2
    def c_ramp(s): return c1 + A_d * s - 0.5 * J * s * s
    def q_ramp(s): return q1 - c1 * s - 0.5 * A_d * s * s + J * s ** 3 / 6.0
    # candidate minima: s where c=0 inside [0, T_r], endpoints
    roots = []
    disc = A_d * A_d + 2 * J * c1
    if disc >= 0:
        for sgn in (1, -1):
            s = (A_d + sgn * math.sqrt(disc)) / J
            if 0 <= s <= T_r:
                roots.append(s)
    for s in [0.0, T_r] + roots:
        qs.append(q_ramp(s))
    qr, cr = q_ramp(T_r), c_ramp(T_r)
    if cr > 0:                                        # still closing after the ramp: constant B deceleration
        qs.append(qr - cr * cr / (2 * B))
    return min(qs)


def classify(m, tol):
    return "boundary" if abs(m) <= tol else ("safe" if m > 0 else "unsafe")


def run(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for B, c, tau, Rq, Rc, off in itertools.product(B_GRID, C_GRID, TAU_GRID, RQ_GRID, RC_GRID, OFFSETS):
        q_req = q_required(B, c, tau, Rq, Rc)
        q_hat = q_req + off
        M = margin_M(q_hat, B, c, tau, Rq, Rc)
        cb = c_bar(c, Rc)
        q_true, c_true = q_hat - Rq, cb
        t_end = tau + (cb + A_D * tau) / B + 1.0
        acc = lambda t, tau=tau, B=B: accel_ideal(t, tau, B, A_D)
        mins = {}
        for dt in (2e-3, 1e-3, 5e-4):
            mins[dt] = integrate_min(q_true, c_true, acc, t_end, dt)
        mins_euler = integrate_min(q_true, c_true, acc, t_end, 5e-4, "euler")
        # honest alternatives (necessity sample): any bounded honest accel history against the max inward adversary
        rng = np.random.default_rng(abs(hash((B, c, tau, Rq, Rc, off))) % (2 ** 32) if False else int(1e6 * (B + c + tau + Rq + Rc + off)) % (2 ** 31))
        best_alt = -math.inf
        for _ in range(5):
            levels = rng.uniform(-B, B, size=8)               # honest relative accel after tau in [-B, B] (net of attacker)
            switch = np.sort(rng.uniform(tau, t_end, size=7))
            def alt(t, tau=tau, levels=levels, switch=switch):
                if t < tau:
                    return -A_D
                return float(levels[np.searchsorted(switch, t)])
            best_alt = max(best_alt, integrate_min(q_true, c_true, alt, t_end, 1e-3))
        rows.append({"B": B, "c_hat": c, "tau": tau, "R_q": Rq, "R_c": Rc, "offset": off, "A_d": A_D, "c_bar": cb,
                     "q_required": q_req, "q_hat": q_hat, "M": M, "class_analytic": classify(M, TOL_ANALYTIC),
                     "minq_closed_form": M, "minq_rk4_2ms": mins[2e-3], "minq_rk4_1ms": mins[1e-3], "minq_rk4_05ms": mins[5e-4],
                     "minq_euler_05ms": mins_euler,
                     "class_numeric": classify(mins[1e-3], TOL_NUMERIC),
                     "err_rk4_1ms": abs(mins[1e-3] - M), "err_euler_05ms": abs(mins_euler - M),
                     "best_alt_policy_minq": best_alt, "alt_policy_never_better": bool(best_alt <= M + TOL_NUMERIC)})
    with (out / "e0_cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    agree = sum(r["class_analytic"] == r["class_numeric"] for r in rows)
    conv = [abs(r["minq_rk4_2ms"] - r["M"]) for r in rows], [abs(r["minq_rk4_1ms"] - r["M"]) for r in rows], [abs(r["minq_rk4_05ms"] - r["M"]) for r in rows]
    cls = {k: sum(r["class_analytic"] == k for r in rows) for k in ("safe", "boundary", "unsafe")}
    # ---- 200 paired boundary examples with a slew-limited honest actuator -------------------------------------------------------
    rng = np.random.default_rng(20261025)
    srows = []
    for i in range(200):
        B = float(rng.choice([0.5, 1.0, 2.0])); c = float(rng.uniform(0.5, 2.0)); tau = float(rng.choice([0.0, 0.25, 0.5]))
        Rq = float(rng.choice([0.0, 0.5, 1.25])); Rc = float(rng.choice([0.0, 0.25, 0.5])); J = float(rng.choice([2.0, 5.0, 8.0, 20.0]))
        off = float(rng.uniform(-0.1, 0.1))
        q_hat = q_required(B, c, tau, Rq, Rc) + off
        M = margin_M(q_hat, B, c, tau, Rq, Rc)
        cb = c_bar(c, Rc); q_true = q_hat - Rq
        m_slew = slew_min_clearance_analytic(q_true, cb, tau, B, A_D, J)
        acc = slew_accel(tau, B, A_D, J)
        t_end = tau + (A_D + B) / J + (cb + A_D * tau + A_D * 0 + 5.0) / B + 2.0
        m_num = integrate_min(q_true, cb, acc, t_end, 5e-4)
        srows.append({"case": i, "B": B, "c_hat": c, "tau": tau, "R_q": Rq, "R_c": Rc, "jerk": J, "offset": off, "M_ideal": M,
                      "M_slew_analytic": m_slew, "M_slew_rk4": m_num, "analytic_vs_rk4_err": abs(m_slew - m_num),
                      "ideal_class": classify(M, TOL_ANALYTIC), "slew_class": classify(m_slew, TOL_ANALYTIC),
                      "ideal_model_mismatch": bool(M >= 0 > m_slew), "ideal_conservative": bool(M < 0 <= m_slew)})
    with (out / "e0_slew_cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0]))
        w.writeheader()
        w.writerows(srows)
    summary = {
        "n_grid_cases": len(rows), "grid": {"B": B_GRID, "c_hat": C_GRID, "tau": TAU_GRID, "R_q": RQ_GRID, "R_c": RC_GRID, "offset": OFFSETS, "A_d": A_D},
        "analytic_class_counts": cls, "tolerances": {"analytic": TOL_ANALYTIC, "numeric": TOL_NUMERIC},
        "analytic_vs_rk4_class_agreement": f"{agree}/{len(rows)}",
        "max_abs_err_rk4": {"dt=2ms": max(conv[0]), "dt=1ms": max(conv[1]), "dt=0.5ms": max(conv[2])},
        "max_abs_err_euler_0.5ms": max(r["err_euler_05ms"] for r in rows),
        "alt_honest_policies_never_better_than_formula": f"{sum(r['alt_policy_never_better'] for r in rows)}/{len(rows)}",
        "slew": {"n": len(srows), "ideal_model_mismatch (ideal safe, slew-limited unsafe)": sum(r["ideal_model_mismatch"] for r in srows),
                 "ideal_conservative": sum(r["ideal_conservative"] for r in srows),
                 "max_analytic_vs_rk4_err": max(r["analytic_vs_rk4_err"] for r in srows),
                 "median_M_ideal_minus_M_slew": float(np.median([r["M_ideal"] - r["M_slew_analytic"] for r in srows]))},
        "note": "deterministic validation cases, not independent statistical trials; the slew-limited rows label the mismatch with the ideal "
                "formula (Prop. 1 has no actuator state) and do not modify the theorem"}
    (out / "e0_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/coupled_safety_v1/e0")
    run(Path(ap.parse_args().out))
