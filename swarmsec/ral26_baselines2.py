"""Two further comparator arms for rarf_hocbf_v1 (new implementations; nothing existing is modified).

ORCA   Optimal Reciprocal Collision Avoidance (van den Berg, Guy, Lin, Manocha, "Reciprocal n-body collision avoidance", 2011): a
       Python port of the RVO2 library's velocity-obstacle half-planes and incremental 2-D linear programs (linearProgram1/2/3), without
       static obstacles.  RVO2 itself is not installed in the image, so this is a re-implementation; tests/test_baselines2.py checks
       the half-plane construction and the LP against brute force.
       Harness interface (ORCA selects VELOCITIES; the harness commands ACCELERATION):
           v_pref = v + tau_pref * u_nom (speed capped at v_max);  v_new = ORCA(v_pref);  u = (v_new - v) / tau_acc,
       then norm U, the common slew limiter, planar zeroing, as for the HOCBF arms.  Neighbour state = age-propagated received packets.
       ORCA's reciprocity premise (each agent takes `responsibility` of the avoidance) is violated by the pursuing attacker; the
       responsibility parameter (0.5 reciprocal, 1.0 full) is a development-tuned candidate.

MRCBF  Measurement-robust exponential HOCBF over the receiver-age enclosure (coupled_safety_v1 MATH_AND_ASSUMPTIONS.md §5; derived there,
       IMPLEMENTED HERE for the first time and verified by tests/test_baselines2.py::test_mrcbf_lower_bound).  For each neighbour envelope
       (r_hat, v_hat, R_p = B_p, R_v = B_v from rc.build_envelope_sound: sender falsification beta, sender and ego estimation errors,
       receiver-local age, transit bound, neighbour acceleration A_j) and D = A_j + W:
         F(u) = 2 l_v^2 + 2 r_hat.u - 2 R_p ||u|| + 2a r_hat.v_hat - 2a E_rv + b (l_r^2 - d^2) - 2D(||r_hat|| + R_p)  >=  eta
       a = k1 + k2, b = k1 k2, l_r = max(||r_hat|| - R_p, 0), l_v = max(||v_hat|| - R_v, 0), E_rv = ||r_hat|| R_v + ||v_hat|| R_p + R_p R_v.
       F is a lower bound of psi1dot + k2 psi1 for every true relative state in the enclosure balls and every ||d_ij|| <= D (the test samples
       it).  The row is a second-order cone in u.  QP: min 1/2||u - u_nom||^2 + rho sum xi  s.t.  F_j(u) + xi_j >= eta, xi >= 0, ||u|| <= U,
       ||u - u_prev|| <= slew (Clarabel).  With xi > 0 the command is uncertified (slack reported), exactly like the HOCBF arms' slack.
       eta: optional zero-order-hold margin; it is a tuning candidate (0 or the derived eta_sd) and is reported.  Initial-condition premises
       (h >= 0, psi1 >= 0 over the balls) are not enforced; they are logged (mr_ic_ok).
"""
from __future__ import annotations

import math
import time

import numpy as np

from swarmsec import ral26_core as rc
from swarmsec.ral26_cs import CsPolicy
from swarmsec.ral26_policy import Decision

EPS = 1e-5
B2_VARIANTS = ("ORCA", "MRCBF")


# ----------------------------------------------------------------------------------------------- ORCA (RVO2 port)
def _det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def orca_lines(p, v, neighbours, radius, tau, dt, responsibility=0.5):
    """Half-planes (point, direction) on the ego velocity; neighbours = [(p_j, v_j)] (planar)."""
    lines = []
    inv_tau = 1.0 / tau
    for pj, vj in neighbours:
        rel_p = pj - p
        rel_v = v - vj
        dist_sq = float(rel_p @ rel_p)
        r = radius
        r_sq = r * r
        if dist_sq > r_sq:
            w = rel_v - inv_tau * rel_p
            w_sq = float(w @ w)
            dot1 = float(w @ rel_p)
            if dot1 < 0.0 and dot1 * dot1 > r_sq * w_sq:
                wl = math.sqrt(w_sq)
                uw = w / wl
                direction = np.array([uw[1], -uw[0]])
                u = (r * inv_tau - wl) * uw
            else:
                leg = math.sqrt(dist_sq - r_sq)
                if _det(rel_p, w) > 0.0:
                    direction = np.array([rel_p[0] * leg - rel_p[1] * r, rel_p[0] * r + rel_p[1] * leg]) / dist_sq
                else:
                    direction = -np.array([rel_p[0] * leg + rel_p[1] * r, -rel_p[0] * r + rel_p[1] * leg]) / dist_sq
                dot2 = float(rel_v @ direction)
                u = dot2 * direction - rel_v
        else:
            inv_dt = 1.0 / dt
            w = rel_v - inv_dt * rel_p
            wl = math.sqrt(float(w @ w))
            uw = w / max(wl, 1e-12)
            direction = np.array([uw[1], -uw[0]])
            u = (r * inv_dt - wl) * uw
        lines.append((v + responsibility * u, direction))
    return lines


def _lp1(lines, k, radius, opt, dir_opt):
    pt, d = lines[k]
    dot = float(pt @ d)
    disc = dot * dot + radius * radius - float(pt @ pt)
    if disc < 0.0:
        return None
    sd = math.sqrt(disc)
    t_l, t_r = -dot - sd, -dot + sd
    for i in range(k):
        pi, di = lines[i]
        den = _det(d, di)
        num = _det(di, pt - pi)
        if abs(den) <= EPS:
            if num < 0.0:
                return None
            continue
        t = num / den
        if den >= 0.0:
            t_r = min(t_r, t)
        else:
            t_l = max(t_l, t)
        if t_l > t_r:
            return None
    if dir_opt:
        return pt + (t_r if float(opt @ d) > 0.0 else t_l) * d
    t = float(d @ (opt - pt))
    return pt + (t_l if t < t_l else t_r if t > t_r else t) * d


def _lp2(lines, radius, opt, dir_opt):
    if dir_opt:
        res = opt * radius
    elif float(opt @ opt) > radius * radius:
        res = opt / np.linalg.norm(opt) * radius
    else:
        res = opt.copy()
    for i, (pi, di) in enumerate(lines):
        if _det(di, pi - res) > 0.0:
            r = _lp1(lines, i, radius, opt, dir_opt)
            if r is None:
                return i, res
            res = r
    return len(lines), res


def _lp3(lines, begin, radius, res):
    dist = 0.0
    for i in range(begin, len(lines)):
        pi, di = lines[i]
        if _det(di, pi - res) > dist:
            proj = []
            for j in range(i):
                pj, dj = lines[j]
                det = _det(di, dj)
                if abs(det) <= EPS:
                    if float(di @ dj) > 0.0:
                        continue
                    point = 0.5 * (pi + pj)
                else:
                    point = pi + (_det(dj, pi - pj) / det) * di
                dd = dj - di
                proj.append((point, dd / np.linalg.norm(dd)))
            k, r = _lp2(proj, radius, np.array([-di[1], di[0]]), True)
            if k >= len(proj):
                res = r
            dist = _det(di, pi - res)
    return res


def orca_velocity(lines, v_pref, vmax):
    k, res = _lp2(lines, vmax, v_pref, False)
    infeasible = k < len(lines)
    if infeasible:
        res = _lp3(lines, k, vmax, res)
    return res, infeasible


# ----------------------------------------------------------------------------------------------- MRCBF rows
def mrcbf_terms(env: rc.Envelope, D, k1, k2, d):
    """F(u) = g0 + 2 r_hat.u - 2 R_p ||u||; returns (r_hat_planar, R_p, g0, ic_ok)."""
    r, v = np.asarray(env.r_hat, float).copy(), np.asarray(env.v_hat, float).copy()
    r[2] = v[2] = 0.0
    Rp, Rv = float(env.B_p), float(env.B_v)
    a, b = k1 + k2, k1 * k2
    nr, nv = float(np.linalg.norm(r)), float(np.linalg.norm(v))
    lr, lv = max(nr - Rp, 0.0), max(nv - Rv, 0.0)
    Erv = nr * Rv + nv * Rp + Rp * Rv
    g0 = 2 * lv * lv + 2 * a * float(r @ v) - 2 * a * Erv + b * (lr * lr - d * d) - 2 * D * (nr + Rp)
    h_lo = lr * lr - d * d
    psi1_lo = 2 * (float(r @ v) - Erv) + k1 * h_lo
    return r, Rp, g0, bool(h_lo >= 0 and psi1_lo >= 0)


class _MRQP:
    def __init__(self, K):
        import cvxpy as cp
        self.cp, self.K = cp, K
        self.r = cp.Parameter((K, 3))
        self.Rp = cp.Parameter(K, nonneg=True)
        self.g = cp.Parameter(K)
        self.un, self.up = cp.Parameter(3), cp.Parameter(3)
        self.U, self.dm, self.rho = cp.Parameter(nonneg=True), cp.Parameter(nonneg=True), cp.Parameter(nonneg=True)
        u, xi = cp.Variable(3), cp.Variable(K, nonneg=True)
        self.u, self.xi = u, xi
        cons = [2 * self.Rp[k] * cp.norm(u, 2) <= 2 * self.r[k] @ u + self.g[k] + xi[k] for k in range(K)]
        cons += [cp.norm(u, 2) <= self.U, cp.norm(u - self.up, 2) <= self.dm, u[2] == 0]
        self.prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(u - self.un) + self.rho * cp.sum(xi)), cons)

    def solve(self, R, Rp, g, un, up, U, dm, rho):
        K = self.K
        Rr, Rpp, gg = np.zeros((K, 3)), np.zeros(K), np.full(K, 1e3)       # padding rows: 0 <= 1e3
        n = len(Rp)
        Rr[:n], Rpp[:n], gg[:n] = R, Rp, g
        self.r.value, self.Rp.value, self.g.value = Rr, Rpp, gg
        self.un.value, self.up.value = un, up
        self.U.value, self.dm.value, self.rho.value = U, dm, rho
        try:
            self.prob.solve(solver="CLARABEL")
        except Exception as exc:
            return None, None, f"solver_error:{type(exc).__name__}"
        if self.u.value is None:
            return None, None, str(self.prob.status)
        return np.asarray(self.u.value, float), np.asarray(self.xi.value, float)[:n], str(self.prob.status)


_MR: dict[int, _MRQP] = {}


def _mrqp(n):
    K = 4
    while K < n:
        K *= 2
    if K not in _MR:
        _MR[K] = _MRQP(K)
    return _MR[K]


# ----------------------------------------------------------------------------------------------- policies
class Baseline2Policy(CsPolicy):
    def __init__(self, kind: str, bounds, params: dict, vmax=4.0, **k):
        super().__init__("D24R" if kind == "MRCBF" else "D0R", bounds, **k)   # D24R gives the sound-age envelope path for MRCBF
        if kind not in B2_VARIANTS:
            raise ValueError(kind)
        self.kind, self.P, self.vmax = kind, dict(params or {}), float(vmax)
        self.variant_label = kind
        self.diag, self.solver_fail = {}, 0
        if kind == "MRCBF":
            for K in (4, 8):
                _mrqp(K)

    def _prep(self, u_nom):
        u = np.array(u_nom, float)
        u[2] = 0.0
        nn = float(np.linalg.norm(u))
        return u * (self.B.U / nn) if nn > self.B.U else u

    def step(self, t, p_own, v_own, packets: dict, u_nom, energy=None, tracking_valid=True) -> Decision:
        t0 = time.perf_counter()
        B = self.B
        self._t_last = t
        u_nom = self._prep(u_nom)
        p, v = np.asarray(p_own, float), np.asarray(v_own, float)
        t_s = time.perf_counter()
        if self.kind == "ORCA":
            P = self.P
            nb = []
            for q in packets.values():
                s = max(0.0, t - q.t_rx)
                pj = q.p_msg + s * q.v_msg
                if np.linalg.norm((pj - p)[:2]) <= float(P.get("neighbor_radius", 12.0)):
                    nb.append((pj[:2].astype(float), q.v_msg[:2].astype(float)))
            v_pref = v[:2] + float(P.get("tau_pref", 1.0)) * u_nom[:2]
            sp = float(np.linalg.norm(v_pref))
            if sp > self.vmax:
                v_pref *= self.vmax / sp
            lines = orca_lines(p[:2], v[:2], nb, float(P.get("radius", 2.0)), float(P.get("tau", 3.0)), 0.05,
                               float(P.get("responsibility", 0.5)))
            v_new, infeas = orca_velocity(lines, v_pref, self.vmax)
            u_f = np.r_[(v_new - v[:2]) / float(P.get("tau_acc", 0.5)), 0.0]
            nn = float(np.linalg.norm(u_f))
            if nn > B.U:
                u_f *= B.U / nn
            viol = max((_det(d, pt - v_new) for pt, d in lines), default=-math.inf)
            self.diag = {"b2_rows": len(lines), "b2_infeasible": int(infeas), "b2_viol": round(float(viol), 4) if lines else "",
                         "b2_vnew": f"{v_new[0]:.3f};{v_new[1]:.3f}", "b2_slack": "", "b2_status": "lp3" if infeas else "lp2", "b2_ic_ok": ""}
            fail = False
        else:
            P = self.P
            envs, _ = self.envelopes(t, p, v, packets)
            k1, k2, d = float(P.get("k1", 1.0)), float(P.get("k2", 1.0)), float(P.get("d", B.d_safe))
            rows, ic = [], []
            for e in envs:
                if not e.valid or float(np.linalg.norm(e.r_hat[:2])) > float(P.get("neighbor_radius", 12.0)):
                    continue
                r, Rp, g0, ok = mrcbf_terms(e, float(e.A_j) + B.W, k1, k2, d)
                rows.append((r, Rp, g0 - float(P.get("eta", 0.0))))
                ic.append(ok)
            if rows:
                R = np.array([x[0] for x in rows])
                Rp = np.array([x[1] for x in rows])
                g = np.array([x[2] for x in rows])
                u_f, xi, st = _mrqp(len(rows)).solve(R, Rp, g, u_nom, self.u_prev, B.U, B.slew, float(P.get("rho", 1e3)))
                fail = u_f is None
                if fail:
                    u_f, xi = self.slew_limit(u_nom), np.zeros(len(rows))
                Fv = [2 * float(R[k] @ u_f) - 2 * Rp[k] * float(np.linalg.norm(u_f)) + g[k] for k in range(len(rows))]
            else:
                u_f, xi, st, fail, Fv = u_nom.copy(), np.zeros(0), "none", False, []
            self.solver_fail += int(fail)
            self.diag = {"b2_rows": len(rows), "b2_slack": round(float(np.max(xi)), 4) if len(xi) else 0.0, "b2_status": st,
                         "b2_viol": round(-min(Fv), 4) if Fv else "", "b2_ic_ok": int(all(ic)) if ic else "", "b2_infeasible": int(fail),
                         "b2_vnew": ""}
        qp_ms = 1e3 * (time.perf_counter() - t_s)
        u_out = self.slew_limit(u_f)
        u_out[2] = 0.0
        self.diag.update(b2_ms=round(qp_ms, 3), b2_slew_mod=int(float(np.linalg.norm(u_out - u_f)) > 1e-9), b2_variant=self.kind)
        self.u_prev = u_out.copy()
        return Decision(u=u_out, status=self.kind, mode=self.mode, gamma=math.nan, nominal_unsafe=False, certified_horizon=0.0,
                        min_row_residual=math.nan, worst_age=max((max(0.0, t - q.t_rx) for q in packets.values()), default=0.0),
                        n_rows_agents=int(self.diag.get("b2_rows", 0)), n_excluded=0, failure_reason="", relax=0.0, refresh=False,
                        solve_ms=1e3 * (time.perf_counter() - t0), assumptions_valid=True, extra={})
