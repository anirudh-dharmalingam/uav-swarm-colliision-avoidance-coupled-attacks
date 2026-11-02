"""Exec chain v2 policy (swarmsec/ral26_exec.py) for the rarf_ablation arms; the v1 classes in ral26_policy / ral26_cs are unchanged.

Same command laws as v1 (D0R, D0R_SPD, D24R_SH, D24R_FF):
  * nominal / speed rule: the historical norm clip + slew limit, then (only if that result leaves the execution set X) the Euclidean
    projection onto X;
  * D24R_SH: hard QP if certified, else best-estimate soft L1 QP, else max-reserve rescue (soft and rescue solved every tick);
  * D24R_FF: the same three solves, soft (else rescue) always selected.
Differences from v1, all in the execution chain:
  1. every command source is solved over X = norm ball ∩ slew ball around the PREVIOUSLY PUBLISHED command ∩ slew-compatible speed
     half-space, so the node's post-policy speed projection is no longer needed (cs_agent_node_x2 does not apply it);
  2. the previous-command reference (u_prev, slew centre of every solve) is the command actually published, committed by `finalize`;
  3. `finalize` revalidates the final command against X and, for the hard branch, every hard row, and sets the label
     (ral26_exec.validate_final).  Decision.status from `step` is provisional; the node logs the final label.
"""
from __future__ import annotations

import math
import time

import numpy as np

from swarmsec import ral26_core as rc
from swarmsec import ral26_exec as rx
from swarmsec.ral26_cs import CsPolicy
from swarmsec.ral26_policy import Decision

X2_VARIANTS = ("D0R", "D0R_SPD", "D24R_SH", "D24R_FF")


class ExecV2Policy(CsPolicy):
    def __init__(self, *a, vmax: float = 4.0, **k):
        super().__init__(*a, **k)
        if self.variant not in X2_VARIANTS:
            raise ValueError(f"exec chain v2 is defined for {X2_VARIANTS}, not {self.variant}")
        self.vmax = float(vmax)
        self.u_prev = np.zeros(3)            # = last PUBLISHED command (committed in finalize)
        self._ctx = None
        self.exec_replaced = 0

    def step(self, t, p_own, v_own, packets: dict, u_nom, energy: dict | None = None, tracking_valid: bool = True) -> Decision:
        t0 = time.perf_counter()
        B = self.B
        self._t_last = t
        u_nom = np.array(u_nom, float)
        u_nom[2] = 0.0
        nn = float(np.linalg.norm(u_nom))
        if nn > B.U:
            u_nom = u_nom * (B.U / nn)
        X = rx.exec_set(B.U, self.u_prev, B.slew, v_own, self.vmax)
        x_info = {"x_active": int(X.active), "x_c": round(X.c, 4) if X.active else ""}
        if self.variant in ("D0R", "D0R_SPD"):
            extra = {}
            if self.speed_rule is not None:
                u_nom, extra = self.speed_restrict(t, p_own, v_own, packets, u_nom)
            u_hist = self.slew_limit(u_nom)                    # v1 law, slew centre = last published command
            u, how = rx.project_exec(u_hist, X)
            extra["br"] = {"slew_clip": int(float(np.linalg.norm(u_hist - u_nom)) > 1e-9), "applied": "nominal", "x_proj": how, **x_info}
            self._ctx = (np.zeros((0, 3)), np.zeros(0), X, "nominal", True)
            return Decision(u=u, status="NO_FILTER", mode=self.mode, gamma=math.nan, nominal_unsafe=False, certified_horizon=0.0,
                            min_row_residual=math.nan, worst_age=0.0, n_rows_agents=0, n_excluded=0, failure_reason="", relax=0.0,
                            refresh=False, solve_ms=1e3 * (time.perf_counter() - t0), assumptions_valid=True, extra=extra)
        envs, assumptions_ok = self.envelopes(t, p_own, v_own, packets)
        assumptions_ok = assumptions_ok and tracking_valid and B.tracking_valid
        worst_age = max((e.age for e in envs), default=0.0)
        A, b, used, n_excl = self._rows(envs, B)
        ok_nom, _, _ = rc.recheck(u_nom, A, b, B.U)
        t_h = time.perf_counter()
        u_hard, hard_cert, res_h, why = rx.solve_hard_x(u_nom, A, b, X)
        hard_ms = 1e3 * (time.perf_counter() - t_h)
        gamma, u_star, g_valid = rx.reserve_x(A, b, X)
        k0 = bool(A.shape[0]) and bool(np.any((np.abs(A).sum(axis=1) < 1e-15) & (b > rc.ROW_TOL)))
        cert = hard_cert and not self.force_fallback
        t_s = time.perf_counter()
        est = {j: rc.NeighborPacket(q.sender, q.p_msg, q.v_msg, q.t_rx, q.max_transit, 0.0, 0.0, 0.0) for j, q in packets.items()}
        envs_e, _ = self.envelopes(t, p_own, v_own, est)
        Ae, be, _, _ = self._rows(envs_e, B)
        u_lv = rx.soft_x(u_nom, Ae, be, X)
        soft_ms = 1e3 * (time.perf_counter() - t_s)
        u_fb = u_star if u_lv is None else u_lv
        selected = "hard" if cert else ("soft" if u_lv is not None else "rescue")
        u = np.asarray(u_hard if cert else u_fb, float)
        res = res_h if cert else (float(np.min(A @ u - b)) if A.shape[0] else math.inf)
        br = {"hard_ok": int(hard_cert), "hard_why": why, "k0_fail": int(k0), "assump_ok": int(assumptions_ok), "hard_ms": round(hard_ms, 3),
              "soft_ok": int(u_lv is not None), "rescue": int(u_lv is None), "soft_ms": round(soft_ms, 3), "applied": selected,
              "gamma_valid": int(g_valid), "planar_chg": int(abs(u[2]) > 1e-12),
              "dev_hard_fb": round(float(np.linalg.norm(u_fb - u_hard)), 4) if u_hard is not None else "", **x_info}
        u[2] = 0.0
        self._ctx = (A, b, X, selected, bool(assumptions_ok))
        if not cert and self.mode == "MISSION":
            self._set_mode(t, "EMERGENCY", why or "rows infeasible")
        elif cert and self.mode == "EMERGENCY":
            self._set_mode(t, "MISSION", "certificate regained")
        status = rc.CERTIFIED if cert else (rx.FALLBACK_SOFT if selected == "soft" else rx.FALLBACK_RESCUE)
        return Decision(u=u, status=status, mode=self.mode, gamma=float(gamma), nominal_unsafe=not ok_nom, certified_horizon=0.0,
                        min_row_residual=float(res), worst_age=float(worst_age), n_rows_agents=len(used), n_excluded=n_excl,
                        failure_reason=why if not cert else "", relax=0.0, refresh=False,
                        solve_ms=1e3 * (time.perf_counter() - t0), assumptions_valid=bool(assumptions_ok),
                        extra={"gamma_validated": bool(g_valid), "energy_margin_wh": math.nan, "rows_with": used, "br": br})

    def finalize(self, u_pub):
        """Revalidate the command the node is about to publish; commit it as the next slew centre.  Returns (u, label, why, replaced)."""
        A, b, X, sel, aok = self._ctx
        u, label, why, replaced = rx.validate_final(u_pub, A, b, X, sel, aok)
        u = np.asarray(u, float)
        self.exec_replaced += int(replaced)
        self.u_prev = u.copy()
        self._ctx = None
        return u, label, why, replaced
