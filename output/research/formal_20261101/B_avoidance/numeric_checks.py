#!/usr/bin/env python3
"""Numerical checks for manuscript B theorems B3c and B5 (inside the image: python3 numeric_checks.py).

B3c exact penalty: for the soft QP  min 1/2||u - u_nom||^2 + rho ||xi||_1  s.t.  A u + xi >= b, xi >= 0, ||u|| <= U,
    the violation V(u) = sum (b - A u)^+ is polyhedral, so it has weak sharp minima and the L1 penalty is exact: above a finite
    threshold rho_bar the solution no longer changes (it is the least-violation command closest to u_nom).  rho_bar is measured on
    random row sets drawn like the RARF rows (rows 2 r_k with ||r_k|| in [1.5, 6] m, offsets placing u_nom 0-6 m/s^2 inside the
    violated region, U = 3).  The B-ISS sweep saw saturation by rho = 1e2; the check reports the distribution of rho_bar.
B5  pursuit regimes (planar double integrators, dt = 0.01 s, horizon 120 s, capture radius 0.77 m = rotor contact):
    attacker: max-acceleration pursuit of the defender's position with velocity matching (u_A = a_A unit(k_p r + k_d v_rel));
    defender strategies: flee (max acceleration along the line of sight away), flee lateral, and tube-confined flee (|y| <= 3 m
    around a route along x at 1 m/s, the mission constraint).  Regimes: (i) V_A > V_D; (ii) V_A = V_D, a_A > a_D, unconstrained;
    (iii) V_A = V_D, a_A > a_D, tube-confined, attacker behind; (iv) a_A > a_D, no speed caps; (v) equal authority, free;
    (vi)/(vii) tube-confined defender that may not run backwards along the route, attacker AHEAD (the blocking geometry), with
    a_A > a_D and with equal authority.  First run (2026-11-01): capture in (i), (iv) and every lateral-flee case; NO capture in
    (ii) and (iii): a defender fleeing forward at equal top speed outruns a pursuer from behind, so tube confinement alone does not
    give the impossibility.  (vi)/(vii) were added after that run to test the blocking geometry (theorem B4).
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("numeric_checks_results.json")


def b3c(n=400, seed=20261121):
    import cvxpy as cp
    rng = np.random.default_rng(seed)
    U = 3.0
    rhos = [float(x) for x in np.logspace(-1, 4, 26)]
    rhos = sorted(set(rhos) | {1e1, 1e2, 1e3})
    thr, dev = [], []
    for _ in range(n):
        K = rng.integers(1, 5)
        ang = rng.uniform(0, 2 * np.pi, K)
        dist = rng.uniform(1.5, 6.0, K)
        A = 2 * dist[:, None] * np.c_[np.cos(ang), np.sin(ang)]
        u_nom = rng.uniform(-3, 3, 2)
        b = A @ u_nom + rng.uniform(0.0, 6.0, K) * 2 * dist        # u_nom violates every row by 0-6 m/s^2 along the row
        u, xi = cp.Variable(2), cp.Variable(K)
        rho = cp.Parameter(nonneg=True)
        P = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(u - u_nom) + rho * cp.sum(xi)), [A @ u + xi >= b, xi >= 0, cp.norm(u) <= U])
        sols = []
        for r in rhos:
            rho.value = r
            P.solve(solver=cp.CLARABEL)
            sols.append(u.value.copy())
        sols = np.array(sols)
        fin = sols[-1]
        stable = [i for i in range(len(rhos)) if np.all(np.linalg.norm(sols[i:] - fin, axis=1) < 1e-4)]
        thr.append(float(rhos[stable[0]]) if stable else float("inf"))
        dev.append([float(np.linalg.norm(sols[list(rhos).index(r)] - fin)) for r in (1e1, 1e2, 1e3)])
    thr, dev = np.array(thr), np.array(dev)
    return {"n": n, "exact_threshold_quantiles": {q: float(np.quantile(thr, q)) for q in (0.5, 0.9)},
            "share_exact_by_1e3": float(np.mean(thr <= 1000.0)),
            "dist_to_rho1e4_solution_m_s2": {f"rho{r:g}": {"median": float(np.median(dev[:, k])), "q90": float(np.quantile(dev[:, k], 0.9)),
                                                          "max": float(dev[:, k].max())} for k, r in enumerate((1e1, 1e2, 1e3))}}


def chase(VA, VD, aA, aD, strategy, tube=None, T=120.0, dt=0.01, r0=None, seed=0):
    rng = np.random.default_rng(seed)
    pD, vD = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    pA = np.array(r0 if r0 is not None else [-15.0 + rng.uniform(-2, 2), rng.uniform(-6, 6)])
    vA = np.zeros(2)
    t = 0.0
    dmin = np.inf
    while t < T:
        r = pD - pA
        d = float(np.linalg.norm(r))
        dmin = min(dmin, d)
        if d < 0.77:
            return True, t, dmin
        los = r / max(d, 1e-9)
        cmdA = 2.0 * r + 3.0 * (vD - vA)
        uA = aA * cmdA / max(np.linalg.norm(cmdA), 1e-9)
        if strategy == "flee":
            uD = aD * los
        elif strategy == "lateral":
            perp = np.array([-los[1], los[0]])
            uD = aD * perp * np.sign(perp @ vD + 1e-9)
        else:                                                       # tube: flee, but y confined and x progress at >= 1 m/s on average
            want = los.copy()
            if tube is not None:
                if abs(pD[1]) > tube - 0.5:
                    want[1] = -np.sign(pD[1])
                want[0] = max(want[0], 0.3)                         # mission: may not run backwards along the route
            uD = aD * want / max(np.linalg.norm(want), 1e-9)
        vA = vA + dt * uA
        vD = vD + dt * uD
        if VA is not None and np.linalg.norm(vA) > VA:
            vA *= VA / np.linalg.norm(vA)
        if VD is not None and np.linalg.norm(vD) > VD:
            vD *= VD / np.linalg.norm(vD)
        if tube is not None:
            vD[0] = min(max(vD[0], 0.0), VD if VD else np.inf)
        pA = pA + dt * vA
        pD = pD + dt * vD
        if tube is not None and abs(pD[1]) > tube:
            pD[1] = np.sign(pD[1]) * tube
            vD[1] = 0.0
        t += dt
    return False, T, dmin


def b5(trials=30):
    regimes = {
        "i_VA_gt_VD": dict(VA=5.0, VD=4.0, aA=3.0, aD=3.0, tube=None),
        "ii_equal_speed_aA_gt_aD_free": dict(VA=4.0, VD=4.0, aA=5.5, aD=3.0, tube=None),
        "iii_equal_speed_aA_gt_aD_tube3m": dict(VA=4.0, VD=4.0, aA=5.5, aD=3.0, tube=3.0),
        "iv_no_speed_caps_aA_gt_aD": dict(VA=None, VD=None, aA=5.5, aD=3.0, tube=None),
        "v_equal_authority_free": dict(VA=4.0, VD=4.0, aA=3.0, aD=3.0, tube=None),
        "vi_equal_speed_attacker_ahead_tube3m": dict(VA=4.0, VD=4.0, aA=5.5, aD=3.0, tube=3.0, r0=[20.0, 0.0]),
        "vii_equal_authority_attacker_ahead_tube3m": dict(VA=4.0, VD=4.0, aA=3.0, aD=3.0, tube=3.0, r0=[20.0, 0.0]),
    }
    out = {}
    for name, p in regimes.items():
        for strat in (("tube",) if p["tube"] else ("flee", "lateral")):
            res = [chase(p["VA"], p["VD"], p["aA"], p["aD"], strat, tube=p["tube"], seed=s,
                         r0=None if p.get("r0") is None else [p["r0"][0] + s % 5, p["r0"][1] + (s % 7 - 3) * 0.8]) for s in range(trials)]
            cap = [r for r in res if r[0]]
            out[f"{name}|{strat}"] = {"capture_share": len(cap) / trials,
                                      "t_capture_max": max((r[1] for r in cap), default=None),
                                      "min_dist_if_escaped": min((r[2] for r in res if not r[0]), default=None)}
    return out


if __name__ == "__main__":
    R = {"B5_pursuit_regimes": b5()}
    print(json.dumps(R["B5_pursuit_regimes"], indent=1), flush=True)
    R["B3c_exact_penalty"] = b3c()
    print(json.dumps(R["B3c_exact_penalty"], indent=1))
    OUT.write_text(json.dumps(R, indent=1))
