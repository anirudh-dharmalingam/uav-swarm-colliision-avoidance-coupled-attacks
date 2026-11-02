"""Per-agent RAL26 safety policies: corrected baseline D24R and the D26 RAVS supervisor + ablations.

One `AgentPolicy` instance lives inside ONE UAV's own controller process.  Its inputs are only what
that UAV has onboard: its own PX4 estimate, the packets its own FANET receiver delivered (with local
receive times), its own energy estimate and the shared mission plan.  It never sees evaluator truth.

D24R and every D26 variant share the SAME hard rows (ral26_core.certificate_rows), the same actuator
ball, the same bounds object and the same emergency policy (the max-reserve command u* when no hard
certificate exists).  They differ only in the supervisor:

  variant        packet-age model           mission relax   link refresh   energy return
  D24R_STATIC    fixed worst-case age       no              no             no
  D24R_AOI       measured AoI (same as D26) no              no             no
  D26_NOAOI      fixed nominal age          yes             yes            yes
  D26_NOBATT     measured AoI               yes             yes            no
  D26_NOLINK     measured AoI               yes             no             yes
  D26 (full)     measured AoI               yes             yes            yes
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import ral26_core as rc

VARIANTS = {
    #               aoi    relax  link   battery
    "D0R":         (True,  False, False, False),     # reference: NO safety layer (nominal, actuator-clipped)
    "D24R_STATIC": (False, False, False, False),
    "D24R_AOI":    (True,  False, False, False),
    "D24R":        (True,  False, False, False),
    "D26_NOAOI":   (False, True,  True,  True),
    "D26_NOBATT":  (True,  True,  True,  False),
    "D26_NOLINK":  (True,  True,  False, True),
    "D26":         (True,  True,  True,  True),
    # v2 baselines: D24R rows + emergency, honest agents IGNORE the commanded squeeze (static spacing,
    # full route speed); _BATT adds the same energy-return supervisor D26 uses (fair energy baseline)
    "D24R_NOSQ":      (True,  False, False, False),
    "D24R_NOSQ_BATT": (True,  False, False, True),
    "D26_FULL":    (True,  True,  True,  True),
    # RARF ablation campaign (rarf_ablation_v1; historical variants above are unchanged).  All three add per-decision branch
    # accounting in Decision.extra["br"]:
    #   D24R_SH   complete RARF (= D24R command law) + a SHADOW best-estimate fallback solve every tick (never applied), so the
    #             compute load matches D24R_FF
    #   D24R_FF   forced best-estimate fallback: the hard solve still runs and is logged (hypothetical) but the fallback
    #             (soft L1-slack QP, max-reserve rescue) is executed at every decision
    #   D0R_SPD   nominal control + proximity speed restriction (no rows, no solver); parameters from the `speed_rule` dict
    "D24R_SH":     (True,  False, False, False),
    "D24R_FF":     (True,  False, False, False),
    "D0R_SPD":     (True,  False, False, False),
}
SPEED_RULE = {"d_slow": 6.0, "d_stop": 3.0, "v_slow": 1.0, "v_near": 0.3, "k": 2.0}


@dataclass
class Decision:
    u: np.ndarray
    status: str
    mode: str
    gamma: float
    nominal_unsafe: bool
    certified_horizon: float
    min_row_residual: float
    worst_age: float
    n_rows_agents: int
    n_excluded: int
    failure_reason: str
    relax: float
    refresh: bool
    solve_ms: float
    assumptions_valid: bool
    extra: dict = field(default_factory=dict)


class AgentPolicy:
    def __init__(self, variant: str, bounds: rc.Bounds, static_age_s: float = 1.0, nominal_age_s: float = 0.45,
                 gamma_warn: float = 0.5, gamma_refresh: float = 0.75, relax_up: float = 1.0, relax_down: float = 0.25,
                 horizons=(0.2, 0.4, 0.6, 1.0), planar: bool = True, speed_rule: dict | None = None):
        v = variant.upper()
        if v not in VARIANTS:
            raise ValueError(f"unknown RAL26 variant {variant!r}")
        self.variant = v
        self.aoi, self.relax_on, self.link_on, self.batt_on = VARIANTS[v]
        self.B = bounds
        self.static_age = static_age_s if v.startswith("D24R") else nominal_age_s
        self.gamma_warn, self.gamma_refresh = gamma_warn, gamma_refresh
        self.relax_up, self.relax_down = relax_up, relax_down
        self.horizons = horizons
        self.planar = planar                 # u_z := 0 (PX4 holds altitude); vertical error lives in W
        self.force_fallback = v == "D24R_FF"
        self.shadow = v in ("D24R_SH", "D24R_FF")
        self.speed_rule = {**SPEED_RULE, **(speed_rule or {})} if v == "D0R_SPD" else None
        self.relax = 0.0
        self.u_prev = np.zeros(3)
        self.mode = "MISSION"
        self.transitions: list = []
        self._t_last = None

    # --------------------------------------------------------------------------------------------
    def _set_mode(self, t, mode, why):
        if mode != self.mode:
            self.transitions.append({"t": t, "from": self.mode, "to": mode, "why": why})
            self.mode = mode

    def envelopes(self, t, p_own, v_own, packets):
        B = self.B
        envs, valid = [], True
        ep, ev = B.ego_err_p + B.sender_err_p, B.ego_err_v + B.sender_err_v
        for pkt in packets.values():
            if self.B.envelope == "sound_age":
                if not self.aoi:
                    raise NotImplementedError("sound_age envelope is defined for measured-age variants only")
                e = rc.build_envelope_sound(pkt, t, p_own, v_own, B.ego_err_p, B.ego_err_v, B.sender_err_p, B.sender_err_v)
            elif self.aoi:
                e = rc.build_envelope(pkt, t, p_own, v_own, ep, ev)
            else:
                e = rc.build_envelope(pkt, t, p_own, v_own, ep, ev, fixed_age=self.static_age)
                true_age = max(0.0, t - pkt.t_rx) + pkt.max_transit
                if true_age > self.static_age + 1e-9:          # fixed-age premise violated
                    valid = False
            if not e.valid:
                valid = False
            envs.append(e)
        return envs, valid

    def certified_horizon(self, envs, u):
        """Largest tested horizon whose rows the command satisfies (diagnostic only)."""
        best = 0.0
        for h in self.horizons:
            Bh = rc.Bounds(**{**self.B.__dict__, "h": h})
            A, b, _, _ = self._rows(envs, Bh)
            ok, _, _ = rc.recheck(u, A, b, self.B.U, self._up(), self._dmax())
            if not ok:
                break
            best = h
        return best

    def _up(self):
        return self.u_prev if math.isfinite(self.B.slew) else None

    def _dmax(self):
        return self.B.slew if math.isfinite(self.B.slew) else None

    def slew_limit(self, u):
        if not math.isfinite(self.B.slew):
            return u
        du = u - self.u_prev
        nd = float(np.linalg.norm(du))
        return u if nd <= self.B.slew else self.u_prev + du * (self.B.slew / nd)

    def speed_restrict(self, t, p_own, v_own, packets, u_nom):
        """D0R_SPD comparator: cap own planar speed by the distance to the nearest received neighbour, using exactly the
        information the nominal consensus uses (packet centre propagated by receiver-local age; no bounds, no rows).
        v_cap(d) = no cap for d >= d_slow, v_near for d <= d_stop, linear from v_near to v_slow on (d_stop, d_slow); above the cap a braking term
        -k (|v| - v_cap) v/|v| is added to the nominal command (norm/slew limits follow as for D0R)."""
        R = self.speed_rule
        d_min = math.inf
        for q in packets.values():
            pj = q.p_msg + max(0.0, t - q.t_rx) * q.v_msg
            d_min = min(d_min, float(np.linalg.norm((pj - p_own)[:2])))
        if d_min >= R["d_slow"]:
            cap = math.inf
        elif d_min <= R["d_stop"]:
            cap = R["v_near"]
        else:
            w = (d_min - R["d_stop"]) / max(R["d_slow"] - R["d_stop"], 1e-9)
            cap = R["v_near"] + w * (R["v_slow"] - R["v_near"])
        u = np.array(u_nom, float)
        sp = float(np.linalg.norm(np.asarray(v_own, float)[:2]))
        brake = 0
        if math.isfinite(cap) and sp > cap and sp > 1e-9:
            u[:2] += -R["k"] * (sp - cap) * np.asarray(v_own, float)[:2] / sp
            brake = 1
            nn = float(np.linalg.norm(u))
            if nn > self.B.U:
                u *= self.B.U / nn
        return u, {"spd": {"d_min": d_min, "cap": cap, "brake": brake}}

    def _rows(self, envs, B):
        A, b, used, n_excl = rc.rows_for(envs, B)
        if self.planar and A.shape[0]:
            A = A.copy()
            A[:, 2] = 0.0
        return A, b, used, n_excl

    # --------------------------------------------------------------------------------------------
    def step(self, t, p_own, v_own, packets: dict, u_nom, energy: dict | None = None,
             tracking_valid: bool = True) -> Decision:
        t0 = time.perf_counter()
        B = self.B
        dt = 0.05 if self._t_last is None else max(1e-3, t - self._t_last)
        self._t_last = t
        u_nom = np.array(u_nom, float)
        if self.planar:
            u_nom[2] = 0.0
        nn = float(np.linalg.norm(u_nom))
        if nn > B.U:
            u_nom = u_nom * (B.U / nn)

        if self.variant in ("D0R", "D0R_SPD"):
            extra = {}
            if self.speed_rule is not None:
                u_nom, extra = self.speed_restrict(t, p_own, v_own, packets, u_nom)
            u_pre = u_nom.copy()
            u_nom = self.slew_limit(u_nom)
            self.u_prev = u_nom.copy()
            if self.speed_rule is not None:
                extra["br"] = {"slew_clip": int(float(np.linalg.norm(u_nom - u_pre)) > 1e-9)}
            return Decision(u=u_nom, status="NO_FILTER", mode=self.mode, gamma=math.nan, nominal_unsafe=False,
                            certified_horizon=0.0, min_row_residual=math.nan, worst_age=0.0, n_rows_agents=0,
                            n_excluded=0, failure_reason="", relax=0.0, refresh=False,
                            solve_ms=1e3 * (time.perf_counter() - t0), assumptions_valid=True, extra=extra)
        envs, assumptions_ok = self.envelopes(t, p_own, v_own, packets)
        assumptions_ok = assumptions_ok and tracking_valid and B.tracking_valid
        worst_age = max((e.age for e in envs), default=0.0)
        A, b, used, n_excl = self._rows(envs, B)

        up, dmax = self._up(), self._dmax()
        ok_nom, _, _ = rc.recheck(u_nom, A, b, B.U)
        t_h = time.perf_counter()
        u, cert, res, why = rc.solve_hard_safety(u_nom, A, b, B.U, up, dmax)
        hard_ms = 1e3 * (time.perf_counter() - t_h)
        gamma, u_star, g_valid = rc.authority_reserve_m(A, b, B.U, up, dmax)
        hard_cert, u_hard = cert, (None if u is None else np.array(u, float))
        br = None
        if self.shadow:                      # branch accounting (overlapping flags; no precedence imposed here)
            k0 = bool(A.shape[0]) and bool(np.any((np.abs(A).sum(axis=1) < 1e-15) & (b > rc.ROW_TOL)))
            br = {"hard_ok": int(hard_cert), "hard_why": why, "k0_fail": int(k0), "assump_ok": int(assumptions_ok),
                  "hard_ms": round(hard_ms, 3), "soft_ok": "", "rescue": 0, "soft_ms": "", "applied": "",
                  "gamma_valid": int(g_valid), "planar_chg": 0, "dev_hard_fb": ""}
            if self.force_fallback:
                cert = False                 # forced branch: the hard result is hypothetical only (logged, never applied)
        if not cert or self.shadow:
            # documented emergency policy (identical for D24R and D26): per-row L1-slack soft QP;
            # u* (max reserve) only if that solve fails its validation
            # rows rebuilt on the BEST ESTIMATE (beta = 0, A_j = 0; age-propagated centres, ego errors and W
            # kept), so near neighbours dominate instead of stale worst-case envelopes
            t_s = time.perf_counter()
            est = {}
            for j, q in packets.items():
                est[j] = rc.NeighborPacket(q.sender, q.p_msg, q.v_msg, q.t_rx, q.max_transit, 0.0, 0.0, 0.0)
            envs_e, _ = self.envelopes(t, p_own, v_own, est)
            Ae, be, _, _ = self._rows(envs_e, B)
            u_lv = rc.least_violation_command(u_nom, Ae, be, B.U, None, up, dmax)
            u_fb = u_star if u_lv is None else u_lv
            if br is not None:
                br.update(soft_ok=int(u_lv is not None), rescue=int(u_lv is None), soft_ms=round(1e3 * (time.perf_counter() - t_s), 3))
                if u_hard is not None:
                    br["dev_hard_fb"] = round(float(np.linalg.norm(np.asarray(u_fb, float) - u_hard)), 4)
            if not cert:
                u = u_fb
                res = float(np.min(A @ u - b)) if A.shape[0] else math.inf
        if self.planar:
            u = np.array(u, float)
            if br is not None:
                br["planar_chg"] = int(abs(u[2]) > 1e-12)
            u[2] = 0.0
        self.u_prev = np.array(u, float)
        status = rc.CERTIFIED if cert else rc.UNCERTIFIED
        if self.force_fallback:
            status = "FALLBACK_FORCED"
        if not assumptions_ok:
            status = rc.INVALID
        if br is not None:
            br["applied"] = "hard" if cert else ("soft" if br["soft_ok"] == 1 else "rescue")
            ok_x, _, _ = rc.recheck(u, A, b, B.U, up, dmax)          # do the EXECUTED (pre-node) command's rows hold?
            br["exec_rows_ok"] = int(ok_x)

        # ---- D26 supervisor (lexicographic: safety rows above, then mission/communication/energy)
        refresh = False
        if self.relax_on:
            if gamma < self.gamma_warn or not cert:
                self.relax = min(1.0, self.relax + self.relax_up * dt)
            else:
                self.relax = max(0.0, self.relax - self.relax_down * dt)
        if self.link_on and gamma < self.gamma_refresh and used:
            refresh = True
        e_margin = math.nan
        if energy is not None and self.batt_on:
            e_margin = energy.get("margin_wh", math.nan)
            if energy.get("mode") == rc.RETURN_REQUIRED and self.mode in ("MISSION", "EMERGENCY"):
                self._set_mode(t, "RETURN_REQUIRED", f"energy margin {e_margin:.3f} Wh < 0")
        if not cert and self.mode == "MISSION":
            self._set_mode(t, "EMERGENCY", why or "rows infeasible")
        elif cert and self.mode == "EMERGENCY":
            self._set_mode(t, "MISSION", "certificate regained")

        ch = self.certified_horizon(envs, u) if cert else 0.0
        return Decision(u=np.asarray(u, float), status=status, mode=self.mode, gamma=float(gamma),
                        nominal_unsafe=not ok_nom, certified_horizon=ch,
                        min_row_residual=float(res), worst_age=float(worst_age), n_rows_agents=len(used),
                        n_excluded=n_excl, failure_reason=why if not cert else "", relax=self.relax,
                        refresh=refresh, solve_ms=1e3 * (time.perf_counter() - t0),
                        assumptions_valid=bool(assumptions_ok),
                        extra={"gamma_validated": bool(g_valid), "energy_margin_wh": e_margin,
                               "rows_with": used, **({"br": br} if br is not None else {})})
