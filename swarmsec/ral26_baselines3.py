"""Published multi-robot avoidance baselines for manuscript B (programme B-PB; new implementations, nothing existing is modified).

SBC  Safety barrier certificates: L. Wang, A. D. Ames and M. Egerstedt, "Safety barrier certificates for collisions-free multirobot
     systems", IEEE T-RO 33(3):661-674, 2017, double-integrator form with acceleration limit alpha (Sec. IV):
         h_ij = sqrt(2 (alpha_i + alpha_j) (||dp|| - Ds)) + dp.dv / ||dp||,      dp = p_i - p_j, dv = v_i - v_j
         -dp.da <= gamma h^3 ||dp|| - (dv.dp)^2/||dp||^2 + ||dv||^2 + (alpha_i + alpha_j) dv.dp / sqrt(2 (alpha_i + alpha_j)(||dp|| - Ds))
     decentralised as in the paper: robot i enforces  -dp.a_i <= (alpha_i / (alpha_i + alpha_j)) b_ij  (its share of the pairwise
     constraint; `responsibility` 0.5 = the paper's equal split, 1.0 = i takes the whole constraint, a development candidate because
     the pursuing attacker does not do its share).  QP: min ||u - u_nom||^2 subject to the rows (squared-slack-free exact 2-D active
     set; an infeasible row set takes the least-max-violation candidate and reports it), then norm U and the common slew limiter.
     Adaptation: alpha_i = alpha_j = U (the harness's actuator ball); inside ||dp|| <= Ds + 1e-3 the square-root terms are dropped
     (the state is already outside the safe set; the paper assumes h >= 0 initially).
BVC  Buffered Voronoi cells: D. Zhou, Z. Wang, S. Bandyopadhyay and M. Schwager, "Fast, on-line collision avoidance for dynamic
     vehicles using buffered Voronoi cells", IEEE RA-L 2(2):1047-1054, 2017:
         BVC_i = { x : (x - (p_i + p_j)/2) . n_ij + r_s <= 0  for every neighbour j },   n_ij = (p_j - p_i) / ||p_j - p_i||
     and the go-to-goal law u = -k (p_i - g*), g* = argmin_{x in BVC_i} ||x - g_i|| (Sec. III, single integrator).
     Adaptation (as for ORCA): the goal is g_i = p_i + tau_goal v_pref with v_pref = v + tau_pref u_nom (the nominal law's
     preferred velocity, capped at v_max); v_des = k (g* - p_i) capped at v_max; the harness commands acceleration
     u = (v_des - v) / tau_acc, then norm U and the common slew limiter.  The paper's double-integrator extension (MPC within the
     cell) is not reproduced; the single-integrator law under a velocity-tracking loop is the common deployment of BVC on
     multirotors.
Both use the neighbour state every comparator uses: age-propagated received packets within neighbor_radius.  Neither models report
falsification, report age or the attacker's pursuit: they are the published methods as their papers state them.
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np

from swarmsec.ral26_cs import CsPolicy
from swarmsec.ral26_policy import Decision

B3_VARIANTS = ("SBC", "BVC")


def qp2(x0, A, b):
    """Exact min ||x - x0||^2 s.t. A x <= b in 2-D (at most two active rows).  Returns (x, max violation >= 0, n_active)."""
    x0 = np.asarray(x0, float)
    if not len(A):
        return x0.copy(), 0.0, 0
    A, b = np.asarray(A, float), np.asarray(b, float)

    def viol(x):
        return float(np.max(A @ x - b))
    if viol(x0) <= 1e-9:
        return x0.copy(), 0.0, 0
    best, bestv, bestd, best_na = None, math.inf, math.inf, 0
    cands = []
    for k in range(len(A)):
        a = A[k]
        aa = float(a @ a)
        if aa < 1e-12:
            continue
        cands.append((x0 - (float(a @ x0) - b[k]) / aa * a, 1))
    for k, m in itertools.combinations(range(len(A)), 2):
        M = A[[k, m]]
        if abs(float(np.linalg.det(M))) < 1e-10:
            continue
        cands.append((np.linalg.solve(M, b[[k, m]]), 2))
    for x, na in cands:
        v = max(viol(x), 0.0)
        d = float(np.sum((x - x0) ** 2))
        if v <= 1e-7:
            if bestv > 1e-7 or d < bestd:
                best, bestv, bestd, best_na = x, v, d, na
        elif bestv > 1e-7 and v < bestv:
            best, bestv, bestd, best_na = x, v, d, na
    return (x0.copy(), viol(x0), 0) if best is None else (best, bestv, best_na)


def sbc_rows(p, v, nb, alpha, Ds, gamma, resp):
    A, b = [], []
    for pj, vj in nb:
        dp, dv = p - pj, v - vj
        n = float(np.linalg.norm(dp))
        if n < 1e-6:
            continue
        s = 2.0 * (2.0 * alpha) * max(n - Ds, 0.0)
        sq = math.sqrt(s) if s > 1e-9 else 0.0
        h = sq + float(dp @ dv) / n
        bij = gamma * h ** 3 * n - float(dv @ dp) ** 2 / n ** 2 + float(dv @ dv)
        if sq > 1e-3:
            bij += (2.0 * alpha) * float(dv @ dp) / sq
        A.append(-dp)
        b.append(resp * bij)
    return A, b


def bvc_rows(p, nb, r_s):
    A, b = [], []
    for pj, _ in nb:
        d = pj - p
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            continue
        nij = d / n
        A.append(nij)
        b.append(float(nij @ (0.5 * (p + pj))) - r_s)
    return A, b


class Baseline3Policy(CsPolicy):
    def __init__(self, kind: str, bounds, params: dict, vmax=4.0, **k):
        super().__init__("D0R", bounds, **k)
        if kind not in B3_VARIANTS:
            raise ValueError(kind)
        self.kind, self.P, self.vmax = kind, dict(params or {}), float(vmax)
        self.variant_label = kind
        self.diag, self.solver_fail = {}, 0

    def step(self, t, p_own, v_own, packets: dict, u_nom, energy=None, tracking_valid=True) -> Decision:
        t0 = time.perf_counter()
        B, P = self.B, self.P
        self._t_last = t
        u_nom = np.array(u_nom, float)
        u_nom[2] = 0.0
        nn = float(np.linalg.norm(u_nom))
        if nn > B.U:
            u_nom *= B.U / nn
        p, v = np.asarray(p_own, float)[:2], np.asarray(v_own, float)[:2]
        nb = []
        for q in packets.values():
            s = max(0.0, t - q.t_rx)
            pj = (q.p_msg + s * q.v_msg)[:2].astype(float)
            if np.linalg.norm(pj - p) <= float(P.get("neighbor_radius", 12.0)):
                nb.append((pj, q.v_msg[:2].astype(float)))
        t_s = time.perf_counter()
        if self.kind == "SBC":
            A, b = sbc_rows(p, v, nb, float(B.U), float(P.get("Ds", 2.5)), float(P.get("gamma", 1.0)), float(P.get("responsibility", 0.5)))
            u2, vi, na = qp2(u_nom[:2], A, b)
        else:
            v_pref = v + float(P.get("tau_pref", 1.0)) * u_nom[:2]
            sp = float(np.linalg.norm(v_pref))
            if sp > self.vmax:
                v_pref *= self.vmax / sp
            g = p + float(P.get("tau_goal", 1.0)) * v_pref
            A, b = bvc_rows(p, nb, float(P.get("r_s", 1.0)))
            gs, vi, na = qp2(g, A, b)
            v_des = float(P.get("k", 1.0)) * (gs - p)
            sp = float(np.linalg.norm(v_des))
            if sp > self.vmax:
                v_des *= self.vmax / sp
            u2 = (v_des - v) / float(P.get("tau_acc", 0.5))
        u_f = np.r_[u2, 0.0]
        nn = float(np.linalg.norm(u_f))
        if nn > B.U:
            u_f *= B.U / nn
        qp_ms = 1e3 * (time.perf_counter() - t_s)
        u_out = self.slew_limit(u_f)
        u_out[2] = 0.0
        self.diag = {"b2_rows": len(A), "b2_slack": round(vi, 4), "b2_status": f"active{na}", "b2_viol": round(vi, 4) if len(A) else "",
                     "b2_ic_ok": "", "b2_infeasible": int(vi > 1e-7), "b2_vnew": "", "b2_ms": round(qp_ms, 3),
                     "b2_slew_mod": int(float(np.linalg.norm(u_out - u_f)) > 1e-9), "b2_variant": self.kind}
        self.u_prev = u_out.copy()
        return Decision(u=u_out, status=self.kind, mode=self.mode, gamma=math.nan, nominal_unsafe=False, certified_horizon=0.0,
                        min_row_residual=math.nan, worst_age=max((max(0.0, t - q.t_rx) for q in packets.values()), default=0.0),
                        n_rows_agents=len(A), n_excluded=0, failure_reason="", relax=0.0, refresh=False,
                        solve_ms=1e3 * (time.perf_counter() - t0), assumptions_valid=True, extra={})
