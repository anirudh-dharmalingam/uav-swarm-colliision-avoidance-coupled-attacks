"""Leader election engines: CADLE (capability-aware dynamic leader election with basic sensing) and baselines.

Every scheme sees the SAME packets (swarmsec.le.channel), the same public plan and fleet registry, and
runs at the protocol rate (10 Hz).  Per receiver i the engine keeps only what i could know: packets i
received (with their arrival times), i's own battery / estimate / sensing evidence.

Packet layout (one broadcast per node per protocol tick), see `Fld`.

Quorum family (one engine, feature flags; Raft-style terms, one vote per term, leader needs q = floor(M/2)+1
votes, M = members not known to have departed):
  cap      rank candidates by capability (else capability-blind, Raft)
  verify   capability = verifiable sufficiency: energy claim capped by the public envelope and saturated at
           mission sufficiency; nav quality from basic-sensing consistency (median over >=3 reporters);
           link quality from RECEIVER reports (sender's self-claim ignored) + radio geometry
  sense    basic-sensing claim consistency -> per-pair trust; f+1 accusations -> distrust
  impeach  followers withdraw from a leader that is distrusted, ineligible, plan-inconsistent (CUSUM) or
           that leaves >theta of the members without heartbeats (coverage no-confidence)
  gate     bounded leader authority: followers use plan(k, c) + clip(leader deviation, delta_max)
  hyst     handover only when a challenger beats the incumbent by Delta for tau_dwell (else immediately)
  sbcbf    sensing-backed safety: detections no claim explains become extra CBF obstacles (needs sense)
  bftq     Byzantine quorum q = floor((M+f)/2)+1 (with one equivocating signer two majority quorums can
           coexist in one term when M is odd; RAFT_H/CAPRAFT keep the crash-fault majority floor(M/2)+1)
  tctl     trust-aware control: a receiver drops the claims of nodes it distrusts from formation control and its
           CBF (needs sense; CADLE family on, TRUST baseline off)
  prevote  leases + bounded term jump + quorum-certificate (QC) check on leader heartbeats: a receiver accepts
           leader L at term t only after it has itself seen q distinct signed votes for (t, L) in packets;
           with impeach, a sender seen voting for two candidates in one term is distrusted (equivocation proof)
Strong baselines added in the addendum (PREREG_LE_ADDENDUM.md):
  CAPRAFT_M  CAPRAFT + the plan-consistency CUSUM monitor and impeachment (a standard leader monitor on a
             capability-aware Raft; no claim verification, sensing, bounded authority or Byzantine quorum)
  TRUST      trust-filtered capability election (representative of trust-based cluster-head selection):
             self-reported capability, nodes distrusted by sensing-consistency Beta trust are ineligible;
             no verification, no impeachment, no bounded authority, no trust-aware control
Local-rule family:  STATIC (designated leader, no failover), BULLY (highest alive id),
                    CAPMAX (highest self-claimed capability, no verification, no quorum).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ROLE_F, ROLE_C, ROLE_L = 0, 1, 2


class Fld:
    TERM, ROLE, LEADER, VOTE, VTERM, ECL, NAVCL, REACHCL = range(8)
    PX, PY, PZ, VX, VY, VZ = range(8, 14)
    WP, CMDX, CMDY, SCALE, CRX, CRY, XFER, LEAVE, QCN = range(14, 23)
    BASE = 23

    @staticmethod
    def width(n):
        return Fld.BASE + 3 * n          # + heard bits, accusation bits, nav-quality reports

    @staticmethod
    def H(n):
        return slice(Fld.BASE, Fld.BASE + n)

    @staticmethod
    def ACC(n):
        return slice(Fld.BASE + n, Fld.BASE + 2 * n)

    @staticmethod
    def NAV(n):
        return slice(Fld.BASE + 2 * n, Fld.BASE + 3 * n)


QUORUM_FLAGS = ("cap", "verify", "sense", "impeach", "gate", "hyst", "prevote", "sbcbf", "bftq", "tctl")
_ALL = {k: True for k in QUORUM_FLAGS}
_NONE = {k: False for k in QUORUM_FLAGS}
SCHEMES = {
    "STATIC": dict(kind="static"),
    "BULLY": dict(kind="bully"),
    "CAPMAX": dict(kind="capmax"),
    "RAFT": dict(kind="quorum", **_NONE),
    "RAFT_H": dict(kind="quorum", **{**_NONE, "prevote": True}),
    "CAPRAFT": dict(kind="quorum", **{**_NONE, "prevote": True, "cap": True, "hyst": True}),
    "CAPRAFT_M": dict(kind="quorum", **{**_NONE, "prevote": True, "cap": True, "hyst": True, "impeach": True}),
    "TRUST": dict(kind="quorum", **{**_NONE, "prevote": True, "cap": True, "hyst": True, "sense": True}),
    "CADLE": dict(kind="quorum", **_ALL),
    "CADLE_NS": dict(kind="quorum", **{**_ALL, "sense": False, "sbcbf": False}),
    "CADLE_NV": dict(kind="quorum", **{**_ALL, "verify": False}),
    "CADLE_NI": dict(kind="quorum", **{**_ALL, "impeach": False}),
    "CADLE_NG": dict(kind="quorum", **{**_ALL, "gate": False}),
    "CADLE_NH": dict(kind="quorum", **{**_ALL, "hyst": False}),
    "CADLE_NPV": dict(kind="quorum", **{**_ALL, "prevote": False}),
    "CADLE_NSB": dict(kind="quorum", **{**_ALL, "sbcbf": False}),
    "CADLE_NIG": dict(kind="quorum", **{**_ALL, "impeach": False, "gate": False}),   # neither impeachment nor gate
    # ---- resilience campaign (ral_resilience_*): additive method labels; legacy entries above are unchanged ----
    # C (conventional baseline): CAPRAFT_M with the SAME sensing/occupancy information as CADLE (basic-sensing trust, trust-aware
    #    control, sensing-backed barrier ghosts), the public-registry energy envelope applied to raw claims (`envclip`: the registry is
    #    public to every scheme) and a conventional plan/command SHIELD (`shield`): every leader packet is accepted verbatim iff each
    #    channel (command, reference, cross-track, scale, index) is inside the monitor's benign-calibrated tolerance AND the leader's
    #    body lies within `tol_c` of the plan-consistent slot; otherwise this UAV executes its local public plan for that packet and
    #    advances its waypoint on its own physical arrival (same guard radius as CADLE's gate).  No continuous authority budget, no
    #    verified capability, crash-fault quorum.
    "CAPRAFT_MS": dict(kind="quorum", **{**_NONE, "prevote": True, "cap": True, "hyst": True, "impeach": True,
                                         "sense": True, "sbcbf": True, "tctl": True}, shield=True, envclip=True),
    # C-tight: the same shield with the reference tolerance and the body-slot radius set to CADLE's authority budgets (r_anchor =
    #    r_auth = 0.45 m).  Disclosed as borrowing CADLE's derived budgets; it is the strongest conventional variant (benign cost reported).
    "CAPRAFT_MS_T": dict(kind="quorum", **{**_NONE, "prevote": True, "cap": True, "hyst": True, "impeach": True,
                                           "sense": True, "sbcbf": True, "tctl": True}, shield=True, envclip=True,
                         shield_tight=True),
    # weaker C without the registry envelope on claims (secondary, not in the rosters)
    "CAPRAFT_MS_RAW": dict(kind="quorum", **{**_NONE, "prevote": True, "cap": True, "hyst": True, "impeach": True,
                                             "sense": True, "sbcbf": True, "tctl": True}, shield=True),
    # CADLE with the waypoint advanced on this UAV's own physical arrival (isolates the waypoint-authority rule of the gate)
    "CADLE_LA": dict(kind="quorum", **_ALL, local_wp=True),
    # D: independent local public-plan execution (no leader, no election): every UAV follows the public plan from its
    #    own median-of-claims centre estimate with the same sensing/CBF platform as CADLE
    # ---- v2 campaign (ral_resilience_v2_*): separately versioned repaired designs; original CADLE above is unchanged and stays a comparator ----
    # CADLE_V2I: CADLE + a follower interface with NO leader-body channel (slot = anchor reference + offset), NO neighbour-consensus term, reduced anchor
    #            position gain (v2_kp/v2_kv) and a tighter command budget (v2_delta); waypoint progression still by the leader's announcement
    # CADLE_V2 : CADLE_V2I + local-arrival waypoint progression (physical successor guard, no announcement needed)
    "CADLE_V2I": dict(kind="quorum", **_ALL, iface2=True),
    "CADLE_V2": dict(kind="quorum", **_ALL, iface2=True, local_wp=True),
    "CADLE_V2_NG": dict(kind="quorum", **{**_ALL, "gate": False}, iface2=True, local_wp=True),      # ablation: V2 interface, no authority gate
    "LOCAL": dict(kind="local", **{**_NONE, "sense": True, "sbcbf": True, "tctl": True}),
}


# Per-controller plan-consistency tolerances (tol_v m/s, tol_c m, tol_xt m), FROZEN from the TRAIN-seed calibration
# cal_le v2 by scripts/le/freeze_tolerances.py --apply (experiments/le/frozen_le.json, DEVIATIONS_LE D4).
FROZEN_TOL = {"CONS": (0.2, 1.5, 4.0), "FLOCK": (0.2, 4.5, 8.0), "LF": (0.2, 1.5, 4.0), "VS": (0.2, 2.5, 6.5)}


def frozen_params(controller, **overrides):
    """ElectionParams with the frozen per-controller tolerances applied (overrides win)."""
    p = ElectionParams(**overrides)
    if controller in FROZEN_TOL:
        tv, tc, tx = FROZEN_TOL[controller]
        p.tol_v = overrides.get("tol_v", tv)
        p.tol_c = overrides.get("tol_c", tc)
        p.tol_xt = overrides.get("tol_xt", tx)
    return p


@dataclass
class ElectionParams:
    t_hb: float = 1.0            # follower: leader silent this long -> lost (plus U(0, t_hb_jit))
    t_hb_jit: float = 0.5
    raft_to: tuple = (1.0, 2.0)  # Raft election timeout range
    t_elect: float = 1.5         # candidate gives up after this
    rank_step: float = 0.4       # CADLE: candidacy delay per rank position
    t_alive: float = 1.5         # local rules: a node counts as alive if heard within this
    w: tuple = (0.35, 0.35, 0.30)  # energy, navigation, link
    delta_vote: float = 0.15     # vote for a candidate within this of the best known
    delta_h: float = 0.15        # handover hysteresis
    tau_dwell: float = 5.0
    trust_min: float = 0.6
    f: int = 1                   # tolerated Byzantine nodes (accusation threshold f+1)
    tol_v: float = 0.35          # plan-consistency tolerances
    tol_c: float = 4.0
    tol_xt: float = 6.0
    cusum_h: float = 6.0
    delta_max: float = 0.3       # bounded authority (m/s)
    r_anchor: float = 0.45       # gate: slot-loop POSITION anchor clip (m); (k_v/k_p) delta_max, see LE_METHOD Prop. 1
    theta_cov: float = 0.3       # coverage no-confidence fraction
    t_cov: float = 3.0
    term_jump: int = 2
    e_headroom: float = 0.25     # energy sufficiency saturates at need + 25 % of mission need
    nav_scale: float = 1.0       # nav error (m) mapped to quality 0
    unverified_cap: float = 0.8
    leave_margin_wh: float = 0.3
    v2_kp: float = 0.3               # CADLE_V2 interface: anchor-position gain
    v2_kv: float = 1.8               # CADLE_V2 interface: velocity gain
    v2_delta: float = 0.2            # CADLE_V2 interface: command-velocity authority budget (m/s)
    local_membership: bool = False   # resilience campaign: departure knowledge is per RECEIVER (False = legacy global reduction)
    shield_tight_r: float = 0.45     # CAPRAFT_MS_T: body-slot radius AND reference tolerance (= r_auth = r_anchor of the CADLE gate)


class Election:
    def __init__(self, n, scheme, plan, fleet, offsets, rng, params: ElectionParams | None = None,
                 designated=0, f_need=None):
        self.n, self.plan, self.fleet, self.offsets, self.rng = n, plan, fleet, offsets, rng
        self.name = scheme
        self.S = dict(SCHEMES[scheme])
        self.kind = self.S["kind"]
        self.p = params or ElectionParams()
        self.designated = designated
        self.f_need = f_need             # callable(k, c) -> energy (Wh) to finish + land, per node
        n_ = n
        self.term = np.zeros(n_, int)
        self.cterm = np.zeros(n_, int)        # committed term: highest term with a QC-backed leader seen
        self.role = np.zeros(n_, int)
        self.leader = np.full(n_, -1)
        self.vote = np.full(n_, -1)
        self.vterm = np.full(n_, -1)
        self.t_last_hb = np.zeros(n_)
        self.to = self._new_timeouts(np.arange(n_))
        self.t_unled = np.zeros(n_)          # time the node became unled
        self.t_cand = np.full(n_, -np.inf)
        self.wp = np.zeros(n_, int)          # progress each node believes (leader's is authoritative)
        self.cref = np.zeros((n_, 2))
        self.cusum = np.zeros(n_)
        self.cov_since = np.full(n_, np.inf)
        self.better_since = np.full(n_, np.inf)
        self.xfer = np.full(n_, -1)
        # per-RECEIVER evidence of signed votes: obs[i][(term, cand)] = voters whose vote i has itself seen
        # (votes persist in every packet until the voter's next term, so late receivers catch up)
        self.obs = [dict() for _ in range(n_)]
        self.vote_seen = {}                  # (i, j, term) -> candidate first seen (equivocation evidence)
        self.equiv = np.zeros((n_, n_), bool)
        self._sc_t = -np.inf
        self.alpha = np.full((n_, n_), 5.0)  # Beta trust, optimistic prior (benign fleet at take-off)
        self.beta = np.ones((n_, n_))
        self.t_boot = 0.0
        self._last_pkt = {}
        self.graw = []                       # raw plan-consistency statistics (calibration / audit)
        self.acc_cmd = np.zeros((n_, n_), bool)
        self.rejected = np.full((n_, n_), -1)   # term at which i withdrew from leader j
        self.departed_known = np.zeros(n_, bool)     # legacy (global) departure knowledge
        self.dep = np.zeros((n_, n_), bool)          # [receiver, member]: what receiver i has itself learned
        self.stats = dict(elections=0, terms_max=0, qc_reject=0, term_ignored=0, impeach=0, xfer=0,
                          stepdown_quorum=0, stepdown_inelig=0)
        self.v_used = np.zeros((n_, 3))
        self.scale_used = np.ones(n_)
        self.cref_used = np.full((n_, 2), np.nan)
        self.shield_reject = np.zeros(n_, int)   # packets the conventional shield replaced by the local plan
        self.cobs = np.full((n_, 2), np.nan)     # local route-anchor estimate (median of used claims), logged for the premise audit
        self.led_by = np.full(n_, -1)        # valid leader each node acts on (-1 = leaderless)
        self.score_cache = None

    # ------------------------------------------------------------------ helpers
    def _new_timeouts(self, idx):
        p = self.p
        if self.kind == "quorum" and not self.S["cap"] and not self.S["prevote"]:
            return self.rng.uniform(*p.raft_to, size=len(idx))
        if self.kind == "quorum" and not self.S["cap"]:
            return self.rng.uniform(*p.raft_to, size=len(idx))
        return p.t_hb + self.rng.uniform(0, p.t_hb_jit, size=len(idx))

    def members(self, i=None):
        """Members not known to have departed.  i=None -> (N,N) [receiver, member]; i -> (N,) row of receiver i.
        Legacy mode: every receiver shares one global vector; local_membership: receiver i uses only the departure
        flags in packets it received itself."""
        m = ~self.dep
        return m if i is None else m[int(i)]

    def quorum(self, i=0):
        M = int(self.members(i).sum())
        if self.S.get("bftq"):
            return (M + self.p.f) // 2 + 1
        return M // 2 + 1

    def trust(self):
        return self.alpha / (self.alpha + self.beta)

    def add_evidence(self, cons, incons):
        lam = 0.98
        self.alpha = 1.0 + lam * (self.alpha - 1.0) + cons
        self.beta = 1.0 + lam * (self.beta - 1.0) + 2.0 * incons

    def own_distrust(self):
        T = self.trust()
        d = ((self.beta - 1.0) >= 8.0) & (T < self.p.trust_min)
        if self.S.get("prevote") and self.S.get("impeach"):
            d = d | self.equiv                # a signed double vote is proof of misbehaviour
        return d

    # ------------------------------------------------------------------ packets
    def compose(self, own_p, own_v, E, nav_claim, heard, navrep, t):
        """Honest packets for every node (adversary edits its own row afterwards)."""
        n, F = self.n, Fld
        P = np.zeros((n, F.width(n)))
        P[:, F.TERM] = self.term
        P[:, F.ROLE] = self.role
        P[:, F.LEADER] = self.leader
        P[:, F.VOTE] = self.vote
        P[:, F.VTERM] = self.vterm
        P[:, F.ECL] = E
        P[:, F.NAVCL] = nav_claim
        P[:, F.REACHCL] = heard.mean(axis=1)
        P[:, F.PX:F.PZ + 1] = own_p
        P[:, F.VX:F.VZ + 1] = own_v
        P[:, F.WP] = self.wp
        P[:, F.CMDX:F.CMDY + 1] = self.v_used[:, :2]
        P[:, F.SCALE] = self.scale_used
        P[:, F.CRX:F.CRY + 1] = self.cref
        P[:, F.XFER] = self.xfer
        P[:, F.LEAVE] = 0.0
        P[:, F.QCN] = 0
        P[:, F.H(n)] = heard
        acc = self.own_distrust() | self.acc_cmd
        np.fill_diagonal(acc, False)
        P[:, F.ACC(n)] = acc
        P[:, F.NAV(n)] = navrep
        return P

    # ------------------------------------------------------------------ capability scores
    def scores(self, V, fresh, t_air, sense_nav=None, sensed=None):
        """(N,N) score[i, j] = i's assessment of candidate j, and eligibility (N,N) bool."""
        n, p, F = self.n, self.p, Fld
        fl = self.fleet
        S = self.S
        c_i = np.where(np.isnan(self.cref_used[:, 0]), self.cref[:, 0], self.cref_used[:, 0])
        c_i = np.stack([c_i, np.where(np.isnan(self.cref_used[:, 1]), self.cref[:, 1], self.cref_used[:, 1])], 1)
        need = np.array([self.f_need(int(self.wp[i]), c_i[i]) for i in range(n)])   # (N,) leader-role need
        need_tot = max(self.f_need(0, np.zeros(2)), 1e-6)
        Ecl = V[:, :, F.ECL]
        if S.get("verify"):
            Ecap = np.minimum(Ecl, fl.energy_envelope(t_air)[None, :])
            e_term = np.clip((Ecap - need[:, None]) / (p.e_headroom * need_tot), 0.0, 1.0)
            elig = Ecap >= need[:, None]
            # navigation: verified by sensing (median over >= 3 reporters), else capped claim
            nav = np.minimum(V[:, :, F.NAVCL], p.unverified_cap)
            if S.get("sense") and sense_nav is not None:
                R = V[:, :, F.NAV(n)]                    # (i, k, j) k's report about j as received by i
                own = np.where(sensed, 1.0 - np.clip(sense_nav / p.nav_scale, 0, 1), np.nan)
                rep = np.where(fresh[:, :, None] & (R >= 0), R, np.nan)          # (i, k, j)
                ii = np.arange(n)
                rep[ii, ii, :] = own                                            # i's own view replaces its packet
                rep[:, ii, ii] = np.nan                                         # nobody vouches for itself
                cnt = np.sum(~np.isnan(rep), axis=1)                            # (i, j)
                srt = np.sort(np.where(np.isnan(rep), np.inf, rep), axis=1)     # NaN-free median over k
                lo = np.clip((cnt - 1) // 2, 0, n - 1)
                hi = np.clip(cnt // 2, 0, n - 1)
                med = 0.5 * (np.take_along_axis(srt, lo[:, None, :], 1)[:, 0] + np.take_along_axis(srt, hi[:, None, :], 1)[:, 0])
                nav = np.where(cnt >= 3, med, np.where(sensed, own, nav))
            # link: receiver reports (exclude the candidate's own), + radio geometry
            H = V[:, :, F.H(n)]                                   # (i, k, j)
            rep = fresh & self.members()                          # (i, k) usable reporters (own membership view)
            w = rep[:, :, None] & ~np.eye(n, dtype=bool)[None, :, :]   # exclude k == j (self report)
            heard_frac = (H * w).sum(1) / np.maximum(w.sum(1), 1)
            Pv = V[:, :, F.PX:F.PY + 1]
            dd = np.sqrt(((Pv[:, :, None, :] - Pv[:, None, :, :]) ** 2).sum(-1))   # (i, j, k)
            m = rep                                                # (i, k)
            geo = ((dd <= 0.7 * fl.radio[None, :, None]) & m[:, None, :]).sum(2) / np.maximum(m.sum(1), 1)[:, None]
            link = 0.5 * heard_frac + 0.5 * geo
        else:
            capmax = fl.cap.max()
            if S.get("envclip"):
                # public registry (nameplate - minimum airborne power x time) bounds any honest claim; applied to the raw
                # claim only -- no sufficiency saturation, no sensing-verified navigation/link quality
                Ecl = np.minimum(Ecl, fl.energy_envelope(t_air)[None, :])
            e_term = np.clip(Ecl / capmax, 0, 1)
            elig = Ecl >= need[:, None] if S.get("hyst") or S.get("cap") else np.ones((n, n), bool)
            nav = np.clip(V[:, :, F.NAVCL], 0, 1)
            link = np.clip(V[:, :, F.REACHCL], 0, 1)
        w = p.w
        score = w[0] * e_term + w[1] * nav + w[2] * link
        if S.get("sense") or S.get("impeach"):
            elig &= ~self.distrusted(V, fresh)
        if S.get("verify"):
            # graceful degradation: a receiver that knows NO energy-sufficient member ranks by remaining
            # (envelope-capped) energy instead of declaring nobody eligible (prevents a hover deadlock)
            known = fresh & self.members() & (V[:, :, F.LEAVE] < 0.5)
            if S.get("sense") or S.get("impeach"):
                known &= ~self.distrusted(V, fresh)
            none = ~(elig & known).any(axis=1)
            if none.any():
                frac = np.clip(Ecap / np.maximum(need[:, None], 1e-6), 0, 1)
                score[none] = (w[0] * frac + w[1] * nav + w[2] * link)[none]
                elig[none] = known[none]
        elig = elig & (V[:, :, F.LEAVE] < 0.5)
        if S.get("sense") or S.get("impeach"):
            elig &= ~self.distrusted(V, fresh)          # fallback never re-admits a distrusted node
        return score, elig

    def distrusted(self, V, fresh):
        n = self.n
        own = self.own_distrust() | self.acc_cmd
        if not self.S.get("sense"):
            own = self.acc_cmd.copy()
        ACC = V[:, :, Fld.ACC(n)] * fresh[:, :, None]               # (i, k, j)
        ii = np.arange(n)
        ACC[ii, ii, :] = 0                                         # i's own packet: use its live state below
        ACC[:, ii, ii] = 0                                         # self-accusation ignored
        cnt = ACC.sum(1).astype(int) + own.astype(int)
        return own | (cnt >= self.p.f + 1)

    # ------------------------------------------------------------------ main step
    def step(self, t, V, t_rx, delivered, own_p, own_v, E, t_air, alive_self, sense_nav=None, sensed=None,
             leaving=None, adversary=None):
        """One protocol tick.  V (N,N,F) latest received packets, t_rx their arrival time."""
        n, p = self.n, self.p
        fresh = (t - t_rx) <= p.t_alive
        np.fill_diagonal(fresh, True)
        if leaving is not None:
            self.departed_known |= leaving
        if p.local_membership:
            # a receiver learns of a departure only from a packet IT received (no cross-receiver reduction)
            self.dep |= (V[:, :, Fld.LEAVE] > 0.5) & ((t - t_rx) < 5.0)
            if leaving is not None:
                self.dep |= leaving[None, :]
        else:
            # legacy: departures announced in ANY receiver's packets become known to every receiver
            self.departed_known |= (V[:, :, Fld.LEAVE] > 0.5).any(axis=0) & ((t - t_rx) < 5.0).any(axis=0)
            self.dep[:] = self.departed_known[None, :]
        if self.kind == "static":
            self._local_rule(t, V, fresh, alive_self, rule="static")
        elif self.kind == "bully":
            self._local_rule(t, V, fresh, alive_self, rule="bully")
        elif self.kind == "capmax":
            self._local_rule(t, V, fresh, alive_self, rule="capmax")
        elif self.kind == "local":
            self._local_rule(t, V, fresh, alive_self, rule="self")
        else:
            self._quorum(t, V, t_rx, fresh, own_p, E, t_air, alive_self, sense_nav, sensed, adversary)
        self.stats["terms_max"] = int(max(self.stats["terms_max"], self.term.max()))
        return fresh

    # ------------------------------------------------------------------ local rules
    def _local_rule(self, t, V, fresh, alive_self, rule):
        n, F = self.n, Fld
        for i in range(n):
            if not alive_self[i]:
                self.leader[i] = -1
                continue
            if rule == "self":                       # LOCAL: every UAV executes the public plan on its own authority
                self.leader[i] = i
                self.role[i] = ROLE_F                # nobody holds the leader role: no UAV has authority over another
                self.t_last_hb[i] = t
                continue
            heard = fresh[i] & self.members(i)
            if rule == "static":
                L = self.designated if heard[self.designated] else -1
            elif rule == "bully":
                ids = np.nonzero(heard)[0]
                L = int(ids.max()) if len(ids) else -1
            else:
                capmax = self.fleet.cap.max()
                s = (self.p.w[0] * np.clip(V[i, :, F.ECL] / capmax, 0, 1) + self.p.w[1] * np.clip(V[i, :, F.NAVCL], 0, 1)
                     + self.p.w[2] * np.clip(V[i, :, F.REACHCL], 0, 1))
                s = np.round(s / 0.05) * 0.05            # claims are quantised (5 % steps) as in deployed systems
                s = np.where(heard & (V[i, :, F.LEAVE] < 0.5), s, -np.inf)
                L = int(np.argmax(s)) if np.isfinite(s).any() else -1   # ties -> lowest id (argmax order)
            self.leader[i] = L
            self.role[i] = ROLE_L if L == i else ROLE_F
            if L >= 0:
                self.t_last_hb[i] = t
        self.led_by = self.leader.copy()

    # ------------------------------------------------------------------ quorum family
    def cast(self, term, voter, cand):
        """voter signs a vote; it is evidence only for the voter itself until others receive it."""
        self.obs[int(voter)].setdefault((int(term), int(cand)), set()).add(int(voter))

    def observe_votes(self, i, V, fresh):
        """Receiver i records the signed votes carried by the packets it holds; two different votes from one
        sender in one term are a (signed) proof of equivocation."""
        F = Fld
        js = np.nonzero(fresh[i] & (V[i, :, F.VOTE] >= 0) & (np.arange(self.n) != i))[0]
        for j in js:
            tm, c = int(V[i, j, F.VTERM]), int(V[i, j, F.VOTE])
            self.obs[i].setdefault((tm, c), set()).add(int(j))
            prev = self.vote_seen.setdefault((i, int(j), tm), c)
            if prev != c:
                self.equiv[i, j] = True

    def _qc_valid(self, i, term, cand):
        return len(self.obs[i].get((int(term), int(cand)), ())) >= self.quorum(i)

    def _quorum(self, t, V, t_rx, fresh, own_p, E, t_air, alive_self, sense_nav, sensed, adversary):
        n, p, F, S = self.n, self.p, Fld, self.S
        att = adversary.id if adversary is not None else -1
        if S["cap"]:
            if t - self._sc_t >= 0.5 - 1e-9 or self.score_cache is None:
                self._sc_t = t
                score, elig = self.scores(V, fresh, t_air, sense_nav, sensed)
            else:
                score, elig = self.score_cache
        else:
            score = np.zeros((n, n))
            elig = np.ones((n, n), bool)
            if S["sense"] or S["impeach"]:
                elig &= ~self.distrusted(V, fresh)
        self.score_cache = (score, elig)
        Vterm = V[:, :, F.TERM].astype(int)
        Vrole = V[:, :, F.ROLE].astype(int)
        hb_age = t - t_rx
        for i in range(n):
            if not alive_self[i] or i == att and adversary.controls_protocol:
                continue
            self.observe_votes(i, V, fresh)
            # ---- 1. leader heartbeats / higher terms
            leaders = np.nonzero(fresh[i] & (Vrole[i] == ROLE_L) & (np.arange(n) != i) & (hb_age[i] <= p.t_hb))[0]
            for j in leaders:
                tj = Vterm[i, j]
                floor_t = self.cterm[i] if S["prevote"] else self.term[i]
                if tj < floor_t or self.rejected[i, j] >= tj:
                    continue
                if S["impeach"] and not elig[i, j]:
                    continue
                if S["prevote"] and not self._qc_valid(i, tj, j):
                    self.stats["qc_reject"] += 1
                    continue
                if tj != self.term[i] or self.leader[i] != j:
                    if self.role[i] == ROLE_L and tj <= self.term[i]:
                        continue            # two leaders in one term would need two quorums: impossible w/ QC
                    # PreVote semantics: a node's own failed candidacies are tentative; a QC-backed leader at a
                    # term >= the committed term is accepted even if the tentative term is higher
                    self.term[i] = tj
                    self.cterm[i] = max(self.cterm[i], tj)
                    self.role[i] = ROLE_F
                    self.leader[i] = j
                    self.t_last_hb[i] = t
                    self.cusum[i] = 0.0
                    self.vote[i] = -1 if self.vterm[i] < tj else self.vote[i]
                if self.leader[i] == j:
                    self.t_last_hb[i] = max(self.t_last_hb[i], t_rx[i, j])
            # Raft: any higher term seen -> adopt it and step down (no prevote / lease)
            if not S["prevote"]:
                hi = int(Vterm[i][fresh[i]].max(initial=0))
                if hi > self.term[i]:
                    self.term[i] = hi
                    self.role[i] = ROLE_F
                    self.leader[i] = -1
                    self.t_unled[i] = t
            # ---- 2. leader validity (impeachment) for followers
            L = self.leader[i]
            if self.role[i] == ROLE_F and L >= 0:
                # lost: lease expired, or L's latest packet says it no longer leads (stepped down / handed over)
                lost = (t - self.t_last_hb[i]) > self.to[i] or (fresh[i, L] and Vrole[i, L] != ROLE_L
                                                                 and V[i, L, F.XFER] < 0)
                bad = False
                if S["impeach"] and not lost:
                    cmd_bad = self.cusum[i] > p.cusum_h
                    bad = (not elig[i, L]) or cmd_bad or self._coverage_alarm(i, L, V, fresh, t)
                if lost or bad:
                    if bad:
                        self.stats["impeach"] += 1
                        self.rejected[i, L] = self.term[i]
                        if cmd_bad:
                            self.acc_cmd[i, L] = True
                    self.leader[i] = -1
                    self.t_unled[i] = t
                    self.to[i] = self._new_timeouts([i])[0]
            if self.role[i] == ROLE_F and self.leader[i] < 0 and self.t_unled[i] <= 0:
                self.t_unled[i] = t
            # ---- 3. leader duties: CheckQuorum, handover / step-down
            if self.role[i] == ROLE_L:
                if S["prevote"]:
                    fol = fresh[i] & (V[i, :, F.LEADER] == i) & (Vterm[i] == self.term[i])
                    fol[i] = True
                    if fol.sum() < self.quorum(i):
                        if self.cov_since[i] == np.inf:
                            self.cov_since[i] = t
                        if t - self.cov_since[i] > 2.0:
                            self._step_down(i, t)
                            self.stats["stepdown_quorum"] += 1
                            continue
                    else:
                        self.cov_since[i] = np.inf
                if S["cap"]:
                    self._handover(i, t, score, elig, fresh)
            # ---- 4. candidacy
            if self.role[i] == ROLE_C and t - self.t_cand[i] > p.t_elect:
                self.role[i] = ROLE_F
                self.t_unled[i] = t
                self.to[i] = self._new_timeouts([i])[0]
            xfer_to_me = (S["cap"] and self.leader[i] >= 0 and V[i, self.leader[i], F.XFER] == i
                          and self.role[i] == ROLE_F)
            if self.role[i] == ROLE_F and (self.leader[i] < 0 or xfer_to_me):
                wait = self.to[i]
                if S["cap"] and not xfer_to_me:
                    cand = elig[i] & fresh[i] & self.members(i)
                    if not cand[i]:
                        wait = np.inf
                    else:
                        s = np.where(cand, score[i], -np.inf)
                        rank = int((s > s[i] + 1e-9).sum())
                        wait = self.to[i] + p.rank_step * rank
                if xfer_to_me or t - self.t_unled[i] >= wait:
                    self._become_candidate(i, t)
            # ---- 5. voting
            if self.role[i] == ROLE_F:
                cands = np.nonzero(fresh[i] & (Vrole[i] == ROLE_C) & (np.arange(n) != i))[0]
                if len(cands):
                    self._vote(i, t, cands, Vterm, V, score, elig, fresh)
            # ---- 6. candidate tally
            if self.role[i] == ROLE_C:
                voters = fresh[i] & (V[i, :, F.VOTE] == i) & (V[i, :, F.VTERM] == self.term[i])
                voters[i] = True
                if voters.sum() >= self.quorum(i):
                    self.role[i] = ROLE_L
                    self.leader[i] = i
                    self.cterm[i] = self.term[i]
                    self.xfer[i] = -1
                    self.t_last_hb[i] = t
                    self.better_since[i] = np.inf
                    self.cov_since[i] = np.inf
        # acted-on leader
        self.led_by = np.where(self.role == ROLE_L, np.arange(n), self.leader)
        self.led_by[~alive_self] = -1

    def _become_candidate(self, i, t):
        # never re-enter a term this node already voted in: under PreVote the term can move back to a
        # committed QC term below a tentative vote, and term+1 would then be a second signed vote (DEVIATIONS D6)
        self.term[i] = max(self.term[i], self.vterm[i]) + 1
        self.role[i] = ROLE_C
        self.vote[i] = i
        self.vterm[i] = self.term[i]
        self.cast(self.term[i], i, i)
        self.t_cand[i] = t
        self.leader[i] = -1
        self.stats["elections"] += 1

    def _step_down(self, i, t):
        self.role[i] = ROLE_F
        self.leader[i] = -1
        self.t_unled[i] = t
        self.xfer[i] = -1
        self.to[i] = self._new_timeouts([i])[0]

    def _vote(self, i, t, cands, Vterm, V, score, elig, fresh):
        p, S = self.p, self.S
        F = Fld
        led = self.leader[i] >= 0 and (t - self.t_last_hb[i]) <= p.t_hb
        best = None
        for c in cands:
            tc = Vterm[i, c]
            if tc <= self.vterm[i] and not (tc == self.vterm[i] and self.vote[i] == c):
                continue
            endorsed = self.leader[i] >= 0 and V[i, self.leader[i], F.XFER] == c
            if S["prevote"]:
                if led and not endorsed:
                    self.stats["term_ignored"] += 1
                    continue
                if tc > self.term[i] + p.term_jump and not endorsed:
                    self.stats["term_ignored"] += 1
                    continue
            if S["cap"]:
                if not elig[i, c]:
                    continue
                cand = elig[i] & fresh[i] & self.members(i)
                bestk = np.max(np.where(cand, score[i], -np.inf))
                if score[i, c] < bestk - p.delta_vote and not endorsed:
                    continue
            elif S["sense"] or S["impeach"]:
                if not elig[i, c]:
                    continue
            key = (score[i, c] if S["cap"] else 0.0, -tc)
            if best is None or key > best[0]:
                best = (key, c, tc)
        if best is None:
            return
        _, c, tc = best
        if not S["prevote"] and tc > self.term[i]:
            self.term[i] = tc
        self.term[i] = max(self.term[i], tc)
        self.vote[i] = c
        self.vterm[i] = tc
        self.cast(tc, i, c)
        if S["prevote"] and self.leader[i] >= 0 and V[i, self.leader[i], F.XFER] == c:
            self.leader[i] = -1

    def _handover(self, i, t, score, elig, fresh):
        p, S = self.p, self.S
        cand = elig[i] & fresh[i] & self.members(i)
        cand[i] = False
        if not cand.any():
            return
        b = int(np.argmax(np.where(cand, score[i], -np.inf)))
        self_ok = elig[i, i]
        margin = p.delta_h if S["hyst"] else 0.0
        dwell = p.tau_dwell if S["hyst"] else 0.0
        if not self_ok:
            self.xfer[i] = b
            self.stats["stepdown_inelig"] += 1
            return
        if score[i, b] > score[i, i] + margin:
            if self.better_since[i] == np.inf:
                self.better_since[i] = t
            if t - self.better_since[i] >= dwell and self.xfer[i] < 0:
                self.xfer[i] = b
                self.stats["xfer"] += 1
        else:
            self.better_since[i] = np.inf

    # ------------------------------------------------------------------ plan consistency
    def _expected(self, i, L, V, t):
        """What follower i expects leader L to command given the public plan and L's own reference."""
        F = Fld
        k = int(V[i, L, F.WP])
        cr = V[i, L, F.CRX:F.CRY + 1]
        v_plan, s_plan = self.plan.command(k, cr)
        cmd = V[i, L, F.CMDX:F.CMDY + 1]
        if t < self.t_boot + 0.05 or (t < self.t_boot + 0.25 and not np.any(cmd)):
            v_plan = np.zeros(3)            # packets composed before the leader's first mission step say "hold"
        return k, cr, v_plan, s_plan

    def _coverage_alarm(self, i, L, V, fresh, t):
        """Too many members report not hearing the leader although they are inside its radio range."""
        n, p, F = self.n, self.p, Fld
        H = V[i, :, F.H(n)]
        rep = fresh[i] & self.members(i)
        rep[L] = False
        if rep.sum() < 3:
            return False
        Pv = V[i, :, F.PX:F.PY + 1]
        inr = np.linalg.norm(Pv - Pv[L], axis=1) <= 0.6 * self.fleet.radio[L]
        cnt = rep & inr
        if cnt.sum() < 3:
            return False
        miss = float(((H[:, L] < 0.5) & cnt).sum()) / cnt.sum()
        if miss > p.theta_cov:
            if self.cov_since[i] == np.inf:
                self.cov_since[i] = t
            return t - self.cov_since[i] > p.t_cov
        self.cov_since[i] = np.inf
        return False

    def update_cusum(self, i, L, V, c_obs, t_pkt):
        """Plan-consistency CUSUM for follower i against leader L, evaluated ONCE per new packet from L
        against the plan as it stood at the packet's send time t_pkt.  Per-packet increment <= 3, so at
        least two grossly inconsistent packets are needed to reach cusum_h = 6."""
        p, F = self.p, Fld
        if self._last_pkt.get((i, L)) == t_pkt:
            return None
        self._last_pkt[(i, L)] = t_pkt
        k, cr, v_plan, s_plan = self._expected(i, L, V, t_pkt)
        cmd = V[i, L, F.CMDX:F.CMDY + 1]
        raw = (float(np.linalg.norm(cmd - v_plan[:2])), float(np.linalg.norm(cr - c_obs)),
               float(self.plan.cross_track(np.r_[cr, 0.0], k)), float(abs(V[i, L, F.SCALE] - s_plan)))
        self.graw.append(raw)
        g = max(raw[0] / p.tol_v, raw[1] / p.tol_c, raw[2] / p.tol_xt, raw[3] / 0.05,
                0.0 if k <= self.wp[i] + 1 else 10.0)
        self.cusum[i] = max(0.0, self.cusum[i] + min(g, 4.0) - 1.0)
        return g
