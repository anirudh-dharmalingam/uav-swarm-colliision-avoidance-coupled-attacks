"""Existing HOCBF filters (swarmsec/safety.py, unchanged) as comparator arms in the RARF per-agent harness (rarf_hocbf_v1).

Two arms, both calling the unmodified `swarmsec.safety.SafetyFilter` row builder (`_constraints`) and cached OSQP QP (`_CachedQP`,
`_CachedQPTwoTier`) for the ego agent only (`filter_one` below reproduces `SafetyFilter.filter`'s per-agent body for index 0; a unit
test checks it against `SafetyFilter.filter` bit-for-bit):
  HOCBF   plain second-order exponential CBF on h = ||p_rel||^2 - Ds^2 with the decentralised convention a_j = 0, squared-slack QP
          (rho = slack_penalty), box |u_d| <= u_max, then norm clip.
  RHOCBF  the same with the existing acceleration-bound robustification: each neighbour row is tightened by ||A_k|| a_bar
          (worst case of the neighbour-acceleration term), optionally as the existing two-tier soft robust layer (`robust_soft`).
          It pads ONLY neighbour acceleration.  It does NOT account for falsified reports, report-age propagation error, transit
          delay, ego-estimate error, or zero-order hold: those are not in the implementation.

Harness adaptations (identical for every arm of rarf_hocbf_v1, documented):
  * information: the ego state is the PX4 estimate; each neighbour's state is the received packet propagated by receiver-local age
    (p_msg + s v_msg, v_msg), exactly what nominal consensus and the speed rule use.  Rows are built on the horizontal relative state
    (z differences set to 0) because the harness commands planar acceleration (PX4 holds altitude); RARF's rows are planar too.
  * actuator: u_max = U = 3 m/s^2 (the filter's box, then its own norm clip), then the harness's common per-tick slew limit
    (0.4 m/s^2, AgentPolicy.slew_limit, the same limiter as the nominal arm) and planar zeroing; then the node's v1 speed-envelope
    projection, as for every other arm.  The downstream slew limit can alter the QP command; the final policy output is re-evaluated
    against the HOCBF rows each tick (`row_res_final`) and that is reported, never hidden.
  * the previous-command reference is the policy output (v1 bookkeeping, identical to NOM / RARF / FF).
"""
from __future__ import annotations

import math
import time

import numpy as np

from swarmsec.ral26_cs import CsPolicy
from swarmsec.ral26_policy import Decision
from swarmsec.safety import SafetyConfig, SafetyFilter, _CachedQPTwoTier

HOCBF_VARIANTS = ("HOCBF", "RHOCBF")


def filter_one(F: SafetyFilter, positions, velocities, u_nom_i, d_safe_row, accel_bound=None, robust_soft=None):
    """SafetyFilter.filter's loop body for agent i = 0 only (no change to the computation).  Returns u, slack, margin, qp_corr, A, c, valid, status."""
    cfg = F.cfg
    A, c, nbr_ids = F._constraints(0, positions, velocities, d_safe_row)
    row_norms = np.linalg.norm(A, axis=1)
    valid = nbr_ids >= 0
    margin = math.inf
    if np.any(valid):
        signed = np.where(valid, (A @ u_nom_i - c) / np.maximum(row_norms, 1e-9), np.inf)
        margin = float(np.min(signed))
    c_qp, ab = c, None
    if accel_bound is not None and np.any(valid):
        ab = np.asarray(accel_bound, float)[np.where(valid, nbr_ids, 0)]
        c_qp = np.where(valid, c + row_norms * ab, c)
    status = "none"
    if robust_soft is not None and ab is not None and F._qp is not None:
        if F._qp2 is None:
            F._qp2 = _CachedQPTwoTier(F.n_rows, cfg.u_max, cfg.slack_penalty)
        rn = np.maximum(row_norms, 1e-9)
        act = c > -1.0e5
        A_hat = np.where(act[:, None], A / rn[:, None], 0.0)
        c_hat = np.where(act, c / rn, c)
        c_rob = np.where(valid, c_hat + ab, c_hat)
        u_i, s_i = F._qp2.solve(A_hat, c_hat, c_rob, valid, float(robust_soft), u_nom_i)
        status = str(F._qp2.prob.status)
    elif F._qp is not None and (np.any(valid) or np.any(c > -1.0e5)):
        u_i, s_i = F._qp.solve(A, c_qp, u_nom_i)
        status = str(F._qp.prob.status)
    else:
        u_i, s_i = u_nom_i, np.zeros(F.n_rows)
    nrm = float(np.linalg.norm(u_i))
    if nrm > cfg.u_max:
        u_i = u_i * (cfg.u_max / nrm)
    return (np.asarray(u_i, float), float(np.max(s_i)) if len(s_i) else 0.0, margin, float(np.linalg.norm(u_i - u_nom_i)),
            A, c_qp, valid, status, c)


class HocbfPolicy(CsPolicy):
    """Comparator arm: D0R command chain with the existing HOCBF filter inserted between the nominal command and the slew limit."""

    def __init__(self, kind: str, bounds, hocbf: dict, **k):
        super().__init__("D0R", bounds, **k)                  # D0R chain: norm clip -> [filter] -> slew limit -> u_prev
        if kind not in HOCBF_VARIANTS:
            raise ValueError(kind)
        self.variant_label = kind
        H = dict(hocbf or {})
        self.cfg = SafetyConfig(d_min=float(H.get("d_min", 1.5)), d_safe=float(H.get("d_safe", 2.5)), k1=float(H.get("k1", 2.0)),
                                k2=float(H.get("k2", 2.0)), u_max=float(bounds.U), slack_penalty=float(H.get("slack_penalty", 1.0e4)),
                                neighbor_radius=float(H.get("neighbor_radius", 10.0)), k_max=int(H.get("k_max", 8)))
        self.F = SafetyFilter(self.cfg)
        self.a_bar = float(H.get("a_bar", 0.0)) if kind == "RHOCBF" else None
        self.robust_soft = H.get("robust_soft") if kind == "RHOCBF" else None
        self.diag = {}
        self.solver_fail = 0

    def step(self, t, p_own, v_own, packets: dict, u_nom, energy=None, tracking_valid=True) -> Decision:
        t0 = time.perf_counter()
        B = self.B
        self._t_last = t
        u_nom = np.array(u_nom, float)
        u_nom[2] = 0.0
        nn = float(np.linalg.norm(u_nom))
        if nn > B.U:
            u_nom = u_nom * (B.U / nn)
        js = list(packets)
        P = np.zeros((1 + len(js), 3))
        V = np.zeros((1 + len(js), 3))
        P[0, :2], V[0, :2] = np.asarray(p_own, float)[:2], np.asarray(v_own, float)[:2]
        for k, j in enumerate(js):
            q = packets[j]
            s = max(0.0, t - q.t_rx)
            P[1 + k, :2] = (q.p_msg + s * q.v_msg)[:2]
            V[1 + k, :2] = q.v_msg[:2]
        ab = None
        if self.a_bar is not None:
            ab = np.full(1 + len(js), self.a_bar)
        t_s = time.perf_counter()
        u_f, slack, margin, corr, A, c, valid, status, c_plain_all = filter_one(self.F, P, V, u_nom, np.full(1 + len(js), self.cfg.d_safe), ab,
                                                                    None if self.robust_soft is None else float(self.robust_soft))
        qp_ms = 1e3 * (time.perf_counter() - t_s)
        fail = status not in ("optimal", "none")
        self.solver_fail += int(fail)
        u_f[2] = 0.0
        u_out = self.slew_limit(u_f)                          # harness's common slew limiter (same as NOM)
        u_out[2] = 0.0
        self.u_prev = u_out.copy()
        A0 = A[valid] if np.any(valid) else np.zeros((0, 3))
        c_plain = c_plain_all[valid] if np.any(valid) else np.zeros(0)          # plain HOCBF rows (no acceleration padding)
        c_rob = c[valid] if np.any(valid) else np.zeros(0)                      # rows the QP was given (padded for RHOCBF)
        res_qp = float(np.min(A0 @ u_f - c_plain)) if len(A0) else math.inf
        res_fin = float(np.min(A0 @ u_out - c_plain)) if len(A0) else math.inf
        res_rob = float(np.min(A0 @ u_out - c_rob)) if len(A0) else math.inf
        self.diag = {"hc_slack": round(slack, 5), "hc_margin": round(margin, 4) if math.isfinite(margin) else "", "hc_corr": round(corr, 4),
                     "hc_rows": int(np.sum(valid)), "hc_status": status, "hc_fail": int(fail), "hc_ms": round(qp_ms, 3),
                     "hc_res_qp": round(res_qp, 4) if math.isfinite(res_qp) else "",
                     "hc_res_final": round(res_fin, 4) if math.isfinite(res_fin) else "",
                     "hc_res_rob": round(res_rob, 4) if math.isfinite(res_rob) else "",
                     "hc_slew_mod": int(float(np.linalg.norm(u_out - u_f)) > 1e-9)}
        return Decision(u=u_out, status=self.variant_label, mode=self.mode, gamma=math.nan, nominal_unsafe=bool(margin < 0),
                        certified_horizon=0.0, min_row_residual=res_fin, worst_age=max((max(0.0, t - q.t_rx) for q in packets.values()), default=0.0),
                        n_rows_agents=int(np.sum(valid)), n_excluded=0, failure_reason=status if fail else "", relax=0.0, refresh=False,
                        solve_ms=1e3 * (time.perf_counter() - t0), assumptions_valid=True, extra={})
