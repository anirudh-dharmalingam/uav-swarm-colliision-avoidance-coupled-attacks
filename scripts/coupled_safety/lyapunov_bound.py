#!/usr/bin/env python3
"""Supporting Lyapunov tracking bound for the ACTUAL nominal controller of the agent (uav_agent_node.Agent.nominal), as in
MATH_AND_ASSUMPTIONS.md section 5.  Per-vehicle tracking error e = (e_p, e_v) of the slot reference obeys
    e_p'' = -ka e_p - kva e_v + nu,   nu = -kp * mean_j(report-derived slot error) - kv * mean_j(report velocity diff) + (u_safe - u_nom) + w
so A_c = [[0, I], [-ka I, -kva I]] (Hurwitz), V = e^T P e with A_c^T P + P A_c = -I, K = ||P B_e||, and
    V' <= -V/(2 lam_max(P)) + 2 K^2 |nu|^2 ,  limsup ||e|| <= sqrt( 2 K^2 nu_bar^2 / lam / lam_min(P) ),  lam = 1/(2 lam_max(P)).
nu_bar is the sum of (declared, never observed) bounds: filter correction <= 2U (u_safe and u_nom both in the U ball), tracking residual W,
consensus coupling kp*(spacing slot error bound) + kv*(relative velocity bound) with neighbour report error beta (+ sender error).
Prints the bound and compares it with the lane clearance; this is a supporting, probably vacuous, result and is reported as such.
    python3 scripts/coupled_safety/lyapunov_bound.py --U 3 --W 3.2 --beta-p 1.25 --beta-v 0.5 --spacing 6 [--out file.json]
"""
import argparse
import json

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

ap = argparse.ArgumentParser()
ap.add_argument("--ka", type=float, default=0.5)
ap.add_argument("--kva", type=float, default=1.0)
ap.add_argument("--kp", type=float, default=0.6)
ap.add_argument("--kv", type=float, default=1.2)
ap.add_argument("--U", type=float, default=3.0)
ap.add_argument("--W", type=float, default=3.2)
ap.add_argument("--beta-p", type=float, default=1.25)
ap.add_argument("--beta-v", type=float, default=0.5)
ap.add_argument("--sender-p", type=float, default=0.6)
ap.add_argument("--sender-v", type=float, default=0.6)
ap.add_argument("--vrel", type=float, default=2.0, help="bound on honest-neighbour relative velocity entering the consensus term [m/s]")
ap.add_argument("--lane-clearance", type=float, default=2.0, help="lane half-margin around the formation [m] (lane width = span + 4)")
ap.add_argument("--out", default=None)
a = ap.parse_args()
A = np.array([[0, 1.0], [-a.ka, -a.kva]])
P = solve_continuous_lyapunov(A.T, -np.eye(2))
Be = np.array([[0.0], [1.0]])
K = float(np.linalg.norm(P @ Be))
ev = np.linalg.eigvalsh(P)
lam = 1.0 / (2 * ev.max())
nu_bar = 2 * a.U + a.W + a.kp * (a.beta_p + a.sender_p) + a.kv * (a.beta_v + a.sender_v + a.vrel)
V_inf = 2 * K ** 2 * nu_bar ** 2 / lam
bound = float(np.sqrt(V_inf / ev.min()))
out = {"A_c_eigs": np.linalg.eigvals(A).real.tolist(), "P": P.tolist(), "K": K, "lambda": lam, "nu_bar": nu_bar, "ultimate_tracking_bound_m": bound,
       "lane_half_margin_m": a.lane_clearance, "useful_relative_to_lane": bool(bound < a.lane_clearance),
       "note": "bound is a sufficient conditional ultimate bound for the per-vehicle slot tracking error (planar, one axis norm-combined); "
               "it does not imply separation, formation convergence or mission completion"}
print(json.dumps(out, indent=1))
if a.out:
    open(a.out, "w").write(json.dumps(out, indent=1))
