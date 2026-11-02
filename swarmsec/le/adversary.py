"""Election-layer adversary (one compromised UAV, authenticated identity, bounded actuation).

The compromised node runs the normal protocol process but may (within the threat model of
swarmsec.attacks: no impersonation, no forging other nodes' signed votes, no access to other UAVs):
  * put anything in ITS OWN packet fields (claims, term, role, votes, accusations, link/nav reports),
  * choose which receivers get its packets (directional suppression),
  * command its own vehicle within epsilon of its nominal,
  * when it is the accepted leader, issue any mission command / reference.

Election attacks (ids used in the campaign):
  L0        none (honest)
  LINF      capability inflation (energy = fleet max, nav = 1, reach = 1) + aggressive candidacy
  LTERM     term flooding: candidacy with term = max seen + 5 every 0.5 s
  LFORGE    forged leadership: claims ROLE=leader with term = max seen + 1 without a quorum
  LDEF      defamation: accuses the leader and the 3 best rivals, reports not hearing them, nav 0
  LEQUIV    equivocation: signs a vote for EVERY candidate it hears in a term (rotating the broadcast vote so
            each candidate counts it) -- tries to make two leaders in one term
  LSPOOF    reports its position 3 m toward the swarm centre (to look central) + LINF claims
  LSPOOFS   adaptive spoof: offset = 0.8 x the receivers' class-aware consistency gate (public), toward the
            centre; below the sensing verifier's threshold by construction (residual-harm probe)
  LSTEER    insider leader steers the swarm toward the nearest keep-out (cmd +0.8 m/s, reference
            drifting 0.5 m/s, flies its own vehicle along) -- overt
  LSTEALTH  insider leader steers with every plan-consistency statistic kept at 0.8 x tolerance
  LSEL      insider leader suppresses heartbeats to the far half of the swarm
  LCOMBO    LINF + LDEF + LTERM + LFORGE + LEQUIV until it leads, then LSTEALTH
  LBURST    timed insider (addendum): leads honestly until the formation reaches the route's closest approach
            to a keep-out, then commands a burst of 3 m/s straight at it and shifts its reference 10 m that way
            (overt; harm is decided by the detection delay -- the extremal case for detection-based defences)
Physical attacks (swarmsec.attacks A2/A4/A8/A9, and A14 = A8 push + A4 report spoof) can be layered
on top via `phys`; they act whenever the compromised node is not leading.
"""
from __future__ import annotations

import numpy as np

from .protocol import Fld, ROLE_C, ROLE_L

ELECTION_ATTACKS = ("L0", "LINF", "LTERM", "LFORGE", "LDEF", "LEQUIV", "LSPOOF", "LSPOOFS", "LSTEER", "LSTEALTH", "LSEL",
                    "LCOMBO", "LBURST", "LDECEP", "LPLAUS", "LSTALL", "LJUMP", "LAUTH", "LROLL")
INSIDER = ("LSTEER", "LSTEALTH", "LSEL", "LBURST")          # legacy: these re-target to the current leader in the old sim
# resilience campaign (additive threat labels; the attacker identity is FIXED by the scenario manifest, never by the defence):
#   LDECEP  deception then steering: LSPOOF-style 3 m position spoof + inflated claims until accepted; steers (LSTEALTH) while leading
#   LPLAUS  plausible/threshold-aware capability deception: energy claim = the public registry envelope, navigation claim =
#           the verifier's unverified cap, truthful position; steers (LSTEALTH) while leading
#   LSTALL  accepted leader never announces progress past waypoint 0 (service denial by index stalling; every statistic stays plan-consistent)
#   LJUMP   accepted leader announces a +2 waypoint-index jump once (corner cutting by index spoofing)
#   LAUTH   (v2) held/accepted insider with PRESCRIBED authority amplitudes on each leader-to-control channel, all pointed at the nearest keep-out:
#           command velocity +auth_cmd [m/s], reference shift auth_ref [m], own body speed auth_body [m/s] beyond its command, claimed-position bias
#           auth_claim [m]; used to probe the predicted admissible authority boundary (separately and jointly)
#   LROLL   (v2) accepted leader announces the PREVIOUS waypoint index once progress has started (rollback)
INSIDER_RES = INSIDER + ("LSTALL", "LJUMP", "LAUTH", "LROLL")
BURST_SPEED, BURST_REF = 3.0, 10.0


def keepout_distance(plan, c):
    """Distance from xy point c to the nearest keep-out boundary (negative inside)."""
    best = np.inf
    for cx, cy, r in plan.keepout_circles:
        best = min(best, float(np.hypot(c[0] - cx, c[1] - cy) - r))
    for x0, x1, y0, y1 in plan.keepout_boxes:
        dx = max(x0 - c[0], 0.0, c[0] - x1)
        dy = max(y0 - c[1], 0.0, c[1] - y1)
        best = min(best, float(np.hypot(dx, dy)) if (dx > 0 or dy > 0) else -1.0)
    return best


def route_min_keepout_distance(plan, step=0.5):
    pts = np.vstack([[0.0, 0.0], plan.waypoints[:, :2]])
    best = np.inf
    for a, b in zip(pts[:-1], pts[1:]):
        m = max(2, int(np.linalg.norm(b - a) / step))
        for u in np.linspace(0, 1, m):
            best = min(best, keepout_distance(plan, a + u * (b - a)))
    return best


class Adversary:
    controls_protocol = False

    def __init__(self, kind, att_id, plan, fleet, offsets, start_s, rng, phys=None, params=None):
        self.kind = kind.upper()
        self.id = int(att_id)
        self.plan, self.fleet, self.offsets = plan, fleet, offsets
        self.start = float(start_s)
        self.rng = rng
        self.phys = phys
        self.p = params
        self.next_flood = 0.0
        self.report_bias = np.zeros(3)
        self.steer_off = np.zeros(2)
        self.active = False
        self.burst = False
        self.burst_t = np.nan
        self.trig_d = route_min_keepout_distance(plan) + 2.0      # burst trigger (used by LBURST and post-capture bursts)
        self.burst_speed = BURST_SPEED               # scenario manifests may override (attacker-strength variation)
        self.stall_k = 0
        self.auth = {}                                  # LAUTH amplitudes (scenario manifest)
        self.end = np.inf                              # finite attacks (recovery population): the attack stops at `end`
        self.post = "LSTEALTH"                         # behaviour once accepted: LSTEALTH | LSTEER | LBURST (scenario manifest)
        self.jumped = False

    def on(self, t):
        return self.kind != "L0" and self.start <= t < self.end

    def mode(self, t, el):
        """Which behaviour applies now."""
        if not self.on(t):
            return "honest"
        leading = el.role[self.id] == ROLE_L
        k = self.kind
        if k == "LCOMBO":
            return "LSTEALTH" if leading else "LCOMBO"
        if k in INSIDER_RES:
            return k if leading else "honest"
        if k in ("LDECEP", "LPLAUS"):
            return self.post if leading else k
        return k

    # ---------------------------------------------------------------- protocol-state manipulation
    def pre_step(self, t, el, V):
        m = self.mode(t, el)
        a = self.id
        self.active = m != "honest"
        if m in ("LEQUIV", "LCOMBO"):
            cands = [int(c) for c in np.nonzero(V[a, :, Fld.ROLE] == ROLE_C)[0] if c != a]
            for c in cands:
                el.cast(int(V[a, c, Fld.TERM]), a, c)          # signed votes for every candidate
            self._equiv = cands
        if m in ("LTERM", "LCOMBO") and t >= self.next_flood and el.role[a] != ROLE_L:
            hi = int(V[a, :, Fld.TERM].max())
            el.term[a] = hi + 5
            el.role[a] = ROLE_C
            el.vote[a] = a
            el.vterm[a] = el.term[a]
            el.cast(el.term[a], a, a)
            el.t_cand[a] = t
            self.next_flood = t + 0.5
        elif m in ("LINF", "LSPOOF", "LSPOOFS", "LDEF", "LDECEP", "LPLAUS") and el.role[a] == 0 and el.leader[a] < 0:
            el._become_candidate(a, t)            # campaign immediately, no rank delay

    # ---------------------------------------------------------------- packet editing
    def edit_packet(self, P, t, el, V, true_p):
        m = self.mode(t, el)
        a, n = self.id, el.n
        if m == "honest":
            # a physical attack runs on its own clock, with or without an election threat (DEVIATIONS_LE D8)
            if self.phys is not None and t >= self.start:
                P[a, Fld.PX:Fld.PZ + 1] += self.report_bias
            else:
                self.report_bias[:] = 0.0
            return P
        if m == "LPLAUS":
            P[a, Fld.ECL] = float(self.fleet.energy_envelope(t)[a])    # the most any claim can be under the public registry
            P[a, Fld.NAVCL] = 0.8                                      # the verifier's unverified-navigation cap
        if m in ("LINF", "LSPOOF", "LSPOOFS", "LCOMBO", "LDECEP"):
            P[a, Fld.ECL] = float(self.fleet.cap.max())
            P[a, Fld.NAVCL] = 1.0
            P[a, Fld.REACHCL] = 1.0
            P[a, Fld.H(n)] = 0.0                       # "I hear nobody else" lowers rivals' reach
        if m in ("LDEF", "LCOMBO"):
            rivals = self._rivals(el, V)
            acc = P[a, Fld.ACC(n)]
            acc[rivals] = 1.0
            P[a, Fld.H(n)][rivals] = 0.0
            nav = P[a, Fld.NAV(n)]
            nav[rivals] = 0.0
            P[a, Fld.VOTE] = a
        if m in ("LEQUIV", "LCOMBO") and getattr(self, "_equiv", None) and el.role[a] != ROLE_C:
            c = self._equiv[int(t * 10) % len(self._equiv)]   # rotate the broadcast vote over the candidates
            P[a, Fld.VOTE] = c
            P[a, Fld.VTERM] = V[a, c, Fld.TERM]
        if m in ("LFORGE", "LCOMBO") and el.role[a] != ROLE_L:
            hi = int(V[a, :, Fld.TERM].max())
            P[a, Fld.TERM] = hi + 1
            P[a, Fld.ROLE] = ROLE_L
            P[a, Fld.LEADER] = a
            P[a, Fld.QCN] = el.quorum(a)                # claims a quorum it does not have
        if m in ("LSPOOF", "LSPOOFS", "LDECEP"):
            alive = self.fleet.alive
            c = np.median(V[a, alive, Fld.PX:Fld.PZ + 1], axis=0)
            d = c - P[a, Fld.PX:Fld.PZ + 1]
            nd = np.linalg.norm(d[:2])
            mag = 3.0
            if m == "LSPOOFS":
                # smallest class-aware gate any receiver applies to this node (fleet registry is public)
                sg = self.fleet.nav_sigma
                gate = np.maximum(1.0, 3.0 * np.sqrt(sg ** 2 + sg[a] ** 2) * 1.25 + 0.3)
                mag = 0.8 * float(np.min(np.delete(gate, a)))
            self.report_bias = np.zeros(3)
            if nd > 1e-6:
                self.report_bias[:2] = d[:2] / nd * min(mag, nd)
            P[a, Fld.PX:Fld.PZ + 1] += self.report_bias
        if m == "LAUTH" and float(self.auth.get("claim", 0.0)) > 0.0:
            dd = self.plan.nearest_keepout_dir(P[a, Fld.PX:Fld.PZ + 1])
            P[a, Fld.PX:Fld.PZ + 1] += float(self.auth["claim"]) * dd
        if self.phys is not None and el.role[a] != ROLE_L:
            P[a, Fld.PX:Fld.PZ + 1] += self.report_bias
        return P

    def _rivals(self, el, V):
        a = self.id
        sc = el.score_cache
        L = el.leader[a]
        r = set([int(L)] if L >= 0 and L != a else [])
        if sc is not None:
            s = sc[0][a].copy()
            s[a] = -np.inf
            r.update(int(x) for x in np.argsort(-s)[:3])
        r.discard(a)
        return np.array(sorted(r), int)

    def send_mask(self, t, el, true_p):
        if self.mode(t, el) != "LSEL":
            return None
        n, a = el.n, self.id
        mask = np.ones((n, n), bool)                 # [receiver, sender]
        d = np.linalg.norm(true_p[:, :2] - true_p[a, :2], axis=1)
        far = d > np.median(d[self.fleet.alive])
        mask[far, a] = False
        return mask

    # ---------------------------------------------------------------- leader behaviour
    def leader_override(self, t, el, cref, v_cmd, k, scale, c_obs):
        """Insider leader: returns (cref, v_cmd, k, scale, v_fly) -- what it broadcasts and how it flies."""
        m = self.mode(t, el)
        if m == "LSTALL":
            k = min(int(k), self.stall_k)              # never announce progress; command/reference stay plan-consistent
            v, _ = self.plan.command(k, cref)
            return cref, v, k, scale, v
        if m == "LROLL":
            k = max(int(k) - 1, 0) if int(k) >= 1 else int(k)
            v, _ = self.plan.command(k, cref)
            return cref, v, k, scale, v
        if m == "LAUTH":
            a = self.auth
            d = self.plan.nearest_keepout_dir(np.r_[c_obs, 0.0])
            v = np.asarray(v_cmd, float) + float(a.get("cmd", 0.0)) * d
            cr = np.asarray(cref, float) + float(a.get("ref", 0.0)) * d[:2]
            vf = v + float(a.get("body", 0.0)) * d
            return cr, v, k, scale, vf
        if m == "LJUMP":
            if not self.jumped:
                self.jumped = True
                k = int(k) + 2
            return cref, v_cmd, k, scale, v_cmd
        if m not in ("LSTEER", "LSTEALTH", "LBURST"):
            return cref, v_cmd, k, scale, v_cmd
        p = self.p
        d = self.plan.nearest_keepout_dir(np.r_[c_obs, 0.0])
        if m == "LBURST":
            if not self.burst and keepout_distance(self.plan, c_obs) <= self.trig_d:
                self.burst, self.burst_t = True, t
            if not self.burst:
                return cref, v_cmd, k, scale, v_cmd
            v = np.asarray(v_cmd, float) + self.burst_speed * d
            return c_obs + BURST_REF * d[:2], v, k, scale, v
        if m == "LSTEER":
            # overt hijack: fly the formation straight at the nearest keep-out, report everything truthfully
            v = self.plan.cruise * d
            return c_obs.copy(), v, k, scale, v
        # adaptive white-box insider: knows every follower check and its tolerance; picks the reference
        # (within 0.8 tol_c of the centre it observes, cross-track <= 0.8 tol_xt) that lies furthest toward
        # the keep-out, then commands plan(k, ref) + 0.8 tol_v toward the keep-out.  Every statistic stays at
        # <= 0.8 of its tolerance in the attacker's own view, so the CUSUM never grows there.
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        rad = np.array([0.25, 0.5, 0.75, 1.0]) * 0.8 * p.tol_c
        cand = [c_obs.copy()] + [c_obs + r * np.array([np.cos(a), np.sin(a)]) for r in rad for a in ang]
        best, bv = c_obs.copy(), -np.inf
        for cr in cand:
            if self.plan.cross_track(np.r_[cr, 0.0], k) > 0.8 * p.tol_xt:
                continue
            val = float(np.dot(cr - c_obs, d[:2]))
            if val > bv:
                bv, best = val, cr
        v = self.plan.command(k, best)[0] + 0.8 * p.tol_v * d
        return best, v, k, scale, v

    # ---------------------------------------------------------------- physical layer
    def physical(self, t, el, true_p, true_v, dt):
        # a physical attack layered on an election threat yields to it while the attacker leads; on its own
        # (L0) it acts whether or not the compromised UAV happens to lead (DEVIATIONS_LE D8)
        if self.phys is None or t < self.start or (self.kind != "L0" and el.role[self.id] == ROLE_L):
            return None
        out = self.phys.act(int(round(t / dt)), true_p, true_v, np.zeros(3), dt=dt, global_lambda2=0.0)
        self.report_bias = np.asarray(out.report_pos_bias, float).copy()
        return out
