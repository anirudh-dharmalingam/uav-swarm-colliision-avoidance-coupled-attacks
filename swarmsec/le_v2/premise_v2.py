"""Theorem-1 applicability audit, v2 (replaces the v1 audit that (i) ignored P4, (ii) checked the physical anchor only, (iii) reported zero violations
under an infinite tube).

Model (formulation sect. 7-8, cruise chart): for honest i on its own segment k_i (unit tangent t, normal n, lookahead ell, cruise v_c)
    z_i = p_i - s d_i                       physical route anchor [m]
    zt_i = c_obs,i                          anchor the controller actually uses (median of claims) [m];   e_i = zt_i - z_i,  eps_i = |e_i|
    f_k(z) = v_c (ell t - y(z) n)/sqrt(ell^2 + y(z)^2)   untruncated constant-speed pure-pursuit field
    zdot_i = f_k(zt_i) + eta_i             eta_i = b_i + w_i  (aggregate leader authority + velocity residual) [m/s]; |eta_i| =: Delta_i
Premises (units): P1 eps <= B_eps [m]; P2 Delta + e_diff <= B_Delta [m/s] (e_diff: allowance for a numerically differentiated velocity; 0 for exact
velocity); P3 domain: the field's EVALUATION point zt AND the physical anchor z are both inside the untruncated chart of segment k_i
(0 <= xi <= L - ell - margin; speed region full), scale constant, UAV airborne, acting under a leader or its own plan (not HOLD), >= t_trans after
the last segment/leader change; P4 non-vacuity: B_Delta + e_diff < v_c, kappa(R*) >= kappa_min > 0 and the tube fits the route clearance
(R* + slot + margin <= clearance).  Applicability = P1 and P2 and P3 and P4; when P4 fails the certificate is UNESTABLISHED and no interval counts
as a successful theorem test.  Conclusions checked on each chart run (maximal block of consecutive applicable samples, >= min_run_s):
    (23) max |y| <= max(|y(t0)|, R*)       (26) xi(t1) - xi(t0) >= kappa (t1 - t0) - tol_xi
"""
from __future__ import annotations

import math

import numpy as np

from ..le.rmetrics import command_vec


def kappa_R(vc, ell, eps, delta, R):
    return vc * ell / math.sqrt(ell * ell + (R + eps) ** 2) - delta


def tube(vc, ell, eps, delta):
    if delta >= vc:
        return math.inf
    return eps + ell * delta / math.sqrt(vc * vc - delta * delta)


def admissible_delta(vc, ell, eps, R, kappa_min):
    """Largest aggregate disturbance for which BOTH sufficient inequalities hold for a desired tube R and progress kappa_min:
       Delta < v_c (R-eps)/sqrt(ell^2+(R-eps)^2)   and   Delta <= v_c ell/sqrt(ell^2+(R+eps)^2) - kappa_min   (negative => infeasible)."""
    if R <= eps:
        return -math.inf
    d1 = vc * (R - eps) / math.sqrt(ell * ell + (R - eps) ** 2)
    d2 = vc * ell / math.sqrt(ell * ell + (R + eps) ** 2) - kappa_min
    return min(d1, d2)


def series(t, P, V, honest, offsets, wp, cobs, scale, led, plan, t_boot=6.0, t_trans=5.0, margin=1.0, h=0.25, att=-1):
    """Per-sample, per-UAV arrays.  V = exact xy velocity (T,n,2) or None (then a central difference of z with half-width h is used and
    `diff_allowance` must be added to Delta)."""
    T, n = len(t), P.shape[1]
    s = np.where(np.isfinite(scale), scale, 1.0)
    z = P[:, :, :2] - s[:, :, None] * offsets[None, :, :2]
    dtm = float(np.median(np.diff(t)))
    if V is not None:
        zdot = V[:, :, :2].astype(float)
    else:
        sh = max(1, int(round(h / dtm)))
        zdot = np.full_like(z, np.nan)
        zdot[sh:-sh] = (z[2 * sh:] - z[:-2 * sh]) / (2 * sh * dtm)
    pts = np.vstack([np.zeros(2), plan.waypoints[:, :2]])
    K = len(plan.waypoints)
    kk = np.clip(wp.astype(int), 0, K - 1)
    a_, b_ = pts[kk], pts[kk + 1]
    sg = b_ - a_
    L = np.linalg.norm(sg, axis=2)
    u = sg / np.maximum(L, 1e-9)[..., None]
    nrm = np.stack([-u[..., 1], u[..., 0]], axis=-1)

    def chart(pos):
        rel = pos - a_
        return (rel * u).sum(-1), (rel * nrm).sum(-1)
    xi, y = chart(z)
    xi_t, y_t = chart(cobs)
    eps = np.linalg.norm(cobs - z, axis=-1)
    f = np.stack([command_vec(plan, wp[:, i].astype(int), cobs[:, i]) for i in range(n)], axis=1)
    eta = np.linalg.norm(zdot - f, axis=-1)
    dom = lambda xx: (xx >= 0.0) & (xx <= L - plan.lookahead - margin)
    domain = dom(xi) & dom(xi_t) & (wp < K)
    last = np.full(wp.shape, float(t_boot))
    cur = np.full(n, float(t_boot))
    for ti in range(1, T):
        ch = (wp[ti] != wp[ti - 1]) | (led[ti] != led[ti - 1])
        cur = np.where(ch, t[ti], cur)
        last[ti] = cur
    domain &= ((t[:, None] - last) >= t_trans) & (led >= 0)
    ds = np.abs(np.diff(s, axis=0, prepend=s[:1]))
    domain &= ds < 1e-3
    hm = np.asarray(honest, bool)
    domain &= hm[None, :]
    for arr in (eps, eta, xi, y):
        arr[:, ~hm] = np.nan
    return dict(z=z, zdot=zdot, eps=eps, eta=eta, xi=xi, y=y, xi_t=xi_t, y_t=y_t, domain=domain, k=wp, L=L)


def audit(t, ps, alive, t_boot, bounds, vc=1.5, ell=8.0, clearance=16.0, slot=7.2, margin_clear=0.5, kappa_min=0.1, diff_allowance=0.0,
          min_run_s=2.0, window_end=None):
    """bounds = {"B_eps","B_delta"}.  Returns a dict with applicability accounting and per-run checks."""
    Bd = bounds["B_delta"] + diff_allowance
    Be = bounds["B_eps"]
    R = tube(vc, ell, Be, Bd)
    kap = kappa_R(vc, ell, Be, Bd, R) if math.isfinite(R) else -math.inf
    fits = math.isfinite(R) and (R + slot + margin_clear <= clearance)
    p4 = bool(Bd < vc and kap >= kappa_min and fits)
    T = len(t)
    win = (t >= t_boot) & (True if window_end is None else (t <= window_end))
    base = win[:, None] & alive & np.isfinite(ps["eps"]) & np.isfinite(ps["eta"])
    P1 = ps["eps"] <= Be
    P2 = ps["eta"] <= Bd
    P3 = ps["domain"]
    tot = int(base.sum())
    cov = {"frac_P1": float((base & P1).sum()) / max(tot, 1), "frac_P2": float((base & P2).sum()) / max(tot, 1), "frac_P3": float((base & P3).sum()) / max(tot, 1),
           "frac_P123": float((base & P1 & P2 & P3).sum()) / max(tot, 1)}
    appl = base & P1 & P2 & P3 & p4
    cov["frac_applicable_P1234"] = float(appl.sum()) / max(tot, 1)
    runs = []
    dt = float(np.median(np.diff(t)))
    k = ps["k"]
    for i in range(appl.shape[1]):
        a = (base & P1 & P2 & P3)[:, i]            # chart runs are formed on P1-P3; P4 decides whether they count as theorem tests
        j = 0
        while j < T:
            if not a[j]:
                j += 1
                continue
            j1 = j
            while j1 + 1 < T and a[j1 + 1] and k[j1 + 1, i] == k[j, i]:
                j1 += 1
            if (j1 - j + 1) * dt >= min_run_s:
                y = np.abs(ps["y"][j:j1 + 1, i])
                bound = max(float(y[0]), R)
                dxi = float(ps["xi"][j1, i] - ps["xi"][j, i])
                dur = float(t[j1] - t[j])
                runs.append(dict(uav=int(i), t0=float(t[j]), t1=float(t[j1]), y0=float(y[0]), ymax=float(y.max()), bound=bound,
                                 tube_excess=float(y.max() - bound) if math.isfinite(bound) else -math.inf, dxi=dxi, dur=dur,
                                 progress_rate=dxi / max(dur, 1e-9), counts_as_test=p4,
                                 violated_tube=int(p4 and y.max() > bound + 1e-6),
                                 violated_progress=int(p4 and dxi < kap * dur - 0.1)))
            j = j1 + 1
    return dict(R_star=R, kappa=kap, vacuous=not math.isfinite(R), tube_fits=bool(fits), P4=p4, coverage=cov, runs=runs, n_samples=tot,
                B_eps=Be, B_delta_eff=Bd)
