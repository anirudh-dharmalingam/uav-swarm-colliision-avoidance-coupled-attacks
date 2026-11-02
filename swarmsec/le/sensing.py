"""Basic onboard proximity sensing and the claim-consistency evidence derived from it.

Sensor (per UAV, 10 Hz): ANONYMOUS detections of airborne bodies within range r_s (360 deg horizontal,
e.g. UWB-ranging + wide-FOV camera blob detector).  Each detection is the true relative position with
range noise sigma_r (+1 % of range) and bearing noise sigma_b; detection probability p_d; Poisson false
alarms (rate lambda_fa per scan, uniform in the sensing disc).  No identities: the sensor cannot read
who is who.  Ground truth enters ONLY here (the emulator), never in the verifier.

Verifier (per receiver i, CADLE): every neighbour j whose CLAIMED position (relative to i's own
estimate) lies inside the sensing disc should be explained by a detection.  Greedy global-nearest
association with gate g.  Common-mode removal: the median residual over all associated pairs estimates
i's own navigation error (b_i) and is subtracted, so a receiver with a bad GNSS does not accuse everyone.
Per pair evidence per scan:
   consistent      associated, |resid - median| <= g_c
   inconsistent    associated with |resid - median| > g_c, or claimed in range (margin m_in) but no detection
                   left within the gate ("phantom claim")
Unexplained detections (no claim) are counted as ghosts (an object that is not where anyone says).
The per-pair residual EWMA is also the verified NAVIGATION QUALITY of j as seen by i.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorConfig:
    range_m: float = 12.0
    sigma_r: float = 0.10
    sigma_b_deg: float = 1.5
    p_d: float = 0.95
    lambda_fa: float = 0.05
    gate: float = 2.5          # association gate (m)
    gate_c: float = 1.0        # consistency gate on the common-mode-corrected residual (m)
    margin_in: float = 1.5     # claim must be this far inside range before a miss counts
    ewma: float = 0.1


class BasicSensor:
    def __init__(self, n, rng, cfg: SensorConfig | None = None, nav_sigma=None):
        self.n, self.rng = n, rng
        self.cfg = cfg or SensorConfig()
        # class-aware consistency gate from the PUBLIC fleet registry: 3 sigma of the 2-D difference of two
        # independent nav errors + sensor noise; never below cfg.gate_c
        sg = np.zeros(n) if nav_sigma is None else np.asarray(nav_sigma, float)
        self.gate_c = np.maximum(self.cfg.gate_c, 3.0 * np.sqrt(sg[:, None] ** 2 + sg[None, :] ** 2) * 1.25 + 0.3)
        self.gate_a = np.maximum(self.cfg.gate, self.gate_c + 0.5)
        self.resid = np.zeros((n, n))           # EWMA of |corrected residual| (m)
        self.seen = np.zeros((n, n), bool)       # ever associated
        self.self_bias = np.zeros((n, 3))        # each receiver's estimate of its own nav error
        self.n_scans = 0
        self.ghost_rel = [np.zeros((0, 3)) for _ in range(n)]   # unexplained detections (relative), last scan
        self.det_rel = [np.zeros((0, 3)) for _ in range(n)]     # last scan's detections and the claim each
        self.det_owner = [np.zeros(0, int) for _ in range(n)]   # was associated with (-1 = none)

    def scan(self, true_p, alive):
        """Anonymous detections for every receiver: list of (M_i,3) relative vectors (detection - own)."""
        c = self.cfg
        out = []
        for i in range(self.n):
            if not alive[i]:
                out.append(np.zeros((0, 3)))
                continue
            rel = true_p - true_p[i]
            d = np.linalg.norm(rel, axis=1)
            vis = alive & (d <= c.range_m) & (np.arange(self.n) != i)
            vis &= self.rng.random(self.n) < c.p_d
            r = rel[vis]
            if len(r):
                rng_ = np.linalg.norm(r, axis=1, keepdims=True)
                u = r / np.maximum(rng_, 1e-9)
                rng_ = rng_ + self.rng.normal(size=rng_.shape) * (c.sigma_r + 0.01 * rng_)
                sb = np.deg2rad(c.sigma_b_deg)
                ang = self.rng.normal(size=(len(r),)) * sb
                ca, sa = np.cos(ang), np.sin(ang)
                ux = ca * u[:, 0] - sa * u[:, 1]
                uy = sa * u[:, 0] + ca * u[:, 1]
                u = np.stack([ux, uy, u[:, 2]], axis=1)
                r = u * rng_
            k = self.rng.poisson(c.lambda_fa)
            if k:
                a = self.rng.uniform(0, 2 * np.pi, k)
                rr = c.range_m * np.sqrt(self.rng.random(k))
                fa = np.stack([rr * np.cos(a), rr * np.sin(a), np.zeros(k)], axis=1)
                r = np.vstack([r, fa]) if len(r) else fa
            out.append(r)
        return out

    def verify(self, dets, own_hat, claimed_p, heard_fresh, alive_view, distrust=None):
        """Consistency evidence from one scan.

        dets       : list of (M_i,3) detections (relative, true geometry + noise)
        own_hat    : (N,3) each receiver's own position estimate
        claimed_p  : (N,N,3) receiver views of claimed positions (row i = what i has heard)
        heard_fresh: (N,N) bool, receiver has a recent claim from j
        Returns (cons, incons) (N,N) int arrays of evidence counts this scan, and ghosts (N,) int."""
        c = self.cfg
        n = self.n
        cons = np.zeros((n, n), int)
        incons = np.zeros((n, n), int)
        ghosts = np.zeros(n, int)
        self.n_scans += 1
        for i in range(n):
            if not alive_view[i]:
                continue
            D = dets[i]
            js = np.nonzero(heard_fresh[i] & (np.arange(n) != i))[0]
            if len(js) == 0:
                self.ghost_rel[i] = D.copy()
                self.det_rel[i], self.det_owner[i] = D.copy(), -np.ones(len(D), int)
                continue
            rel_claim = claimed_p[i, js] - own_hat[i]                 # (J,3)
            dc = np.linalg.norm(rel_claim[:, :2], axis=1)
            cand = dc <= c.range_m + c.gate
            js, rel_claim, dc = js[cand], rel_claim[cand], dc[cand]
            used = np.zeros(len(D), bool)
            res = np.full((len(js), 3), np.nan)
            amb = np.zeros(len(js), bool)
            any_near = np.zeros(len(js), bool)
            if len(D) and len(js):
                M = np.sqrt(((rel_claim[:, None, :2] - D[None, :, :2]) ** 2).sum(-1))   # (J,M)
                ga = self.gate_a[i, js]
                within = M <= ga[:, None]
                # a claim with no detection is excused only if it is CO-LOCATED (within the tight consistency
                # gate) with a real body that another claim explains -- otherwise it is a phantom claim
                any_near = (M <= self.gate_c[i, js][:, None]).any(axis=1)
                # claims i already distrusts are associated LAST (cannot steal an honest node's detection)
                pen = np.zeros(len(js)) if distrust is None else np.where(distrust[i, js], 1e3, 0.0)
                aa, bb = np.nonzero(within)                                  # only gated pairs
                order = np.argsort(M[aa, bb] + pen[aa], kind="stable")
                done = np.zeros(len(js), bool)
                pick = np.full(len(js), -1)
                for q in order:
                    x, y = aa[q], bb[q]
                    if done[x] or used[y]:
                        continue
                    done[x] = used[y] = True
                    pick[x] = y
                got = pick >= 0
                res[got] = rel_claim[got] - D[pick[got]]                      # claim - truth (+ i's own error)
                # ambiguity: the detection also fits another (trusted) claim -> no evidence either way
                fit = M <= self.gate_c[i, js][:, None]                        # (J,M)
                if distrust is not None:
                    fit &= ~distrust[i, js][:, None]
                nfit = fit.sum(0)                                             # trusted claims fitting each det
                ga_idx = np.nonzero(got)[0]
                self_fit = fit[ga_idx, pick[ga_idx]]
                amb[ga_idx] = (nfit[pick[ga_idx]] - self_fit.astype(int)) > 0
            ok = ~np.isnan(res[:, 0])
            med = np.median(res[ok], axis=0) if ok.sum() >= 3 else np.zeros(3)
            self.self_bias[i] = 0.9 * self.self_bias[i] + 0.1 * (-med)
            e_all = np.sqrt(((res[:, :2] - med[:2]) ** 2).sum(1))
            gc = self.gate_c[i, js]
            m_ok = ok & ~amb
            jo = js[m_ok]
            e = e_all[m_ok]
            first = ~self.seen[i, jo]
            self.resid[i, jo] = np.where(first, e, (1 - c.ewma) * self.resid[i, jo] + c.ewma * e)
            self.seen[i, jo] = True
            cons[i, jo] += (e <= gc[m_ok]).astype(int)
            incons[i, jo] += (e > gc[m_ok]).astype(int)
            m_ph = ~ok & ~amb & (dc <= c.range_m - c.margin_in) & ~any_near    # phantom claim: nothing there
            jp = js[m_ph]
            g = self.gate_a[i, jp]
            incons[i, jp] += 1
            self.resid[i, jp] = np.where(self.seen[i, jp], (1 - c.ewma) * self.resid[i, jp] + c.ewma * g, g)
            self.seen[i, jp] = True
            ghosts[i] = int((~used).sum())
            self.ghost_rel[i] = D[~used] if len(D) else np.zeros((0, 3))
            owner = -np.ones(len(D), int)
            if len(D) and len(js):
                owner[pick[pick >= 0]] = js[pick >= 0]
            self.det_rel[i], self.det_owner[i] = D, owner
        return cons, incons, ghosts

    def ghosts_excluding(self, excl):
        """Recompute the safety ghosts against the claims each receiver actually USES: a detection explained
        only by a claim the receiver excludes (distrusted) is still a body and must stay a barrier obstacle."""
        for i in range(self.n):
            o = self.det_owner[i]
            if len(o):
                self.ghost_rel[i] = self.det_rel[i][(o < 0) | excl[i, np.maximum(o, 0)]]

