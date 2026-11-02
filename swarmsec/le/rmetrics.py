"""Truth-side mission-service, safety, recovery and premise metrics for the resilience campaign.

ONE implementation for both layers: Layer 1 feeds its internal truth series, the PX4 evaluator feeds Gazebo truth
(eval/truth.csv) and the agents' logs.  Nothing here reads a controller's claimed state: completion, progress and safety
are computed from true honest positions only (a leader's waypoint index or completion message is never an input).

Conventions
  t        (T,)       mission time [s], 0 = mission start marker (agents' t = 0)
  P        (T,n,3)    true positions in the mission frame [m]
  airborne (T,n)      bool, True while the UAV is in active flight
  honest   (n,)       bool; the compromised UAV (if any) is excluded from honest outcomes and never averaged into them
Scoring window: [t_boot, t_end_score] with t_end_score = min(t_complete + 10 s, last sample): the same rule the legacy Layer-1
simulator used (post-mission hover is not scored); ground proximity / landing is excluded by `airborne`.
"""
from __future__ import annotations

import math

import numpy as np

W_SERVICE = 5.0          # service window W [s] for q_W
TAU_HOLD = 10.0          # sustaining dwell tau_hold [s]
POST_COMPLETE_S = 10.0


def polyline(plan):
    pts = np.vstack([np.zeros(2), plan.waypoints[:, :2]])
    seg = np.diff(pts, axis=0)
    ls = np.linalg.norm(seg, axis=1)
    cum = np.r_[0.0, np.cumsum(ls)]
    return pts, seg, ls, cum


def arclength(plan, c, k):
    """Route arclength of centroid c for TRUE progress index k (0..K): cumulative length of finished segments plus the
    clamped projection on the current one."""
    pts, seg, ls, cum = polyline(plan)
    K = len(ls)
    if k >= K:
        return float(cum[-1])
    u = seg[k] / max(ls[k], 1e-9)
    return float(cum[k] + np.clip(np.dot(c[:2] - pts[k], u), 0.0, ls[k]))


def command_vec(plan, k, c):
    """Vectorised twin of Plan.command (pure pursuit, never beyond waypoint k, slows within 6 m); k (M,), c (M,2) -> (M,2).
    Checked against Plan.command in tests/test_resilience.py."""
    pts = np.vstack([np.zeros(2), plan.waypoints[:, :2]])
    K = len(plan.waypoints)
    kc = np.clip(k, 0, K - 1)
    prev, wpk = pts[kc], pts[kc + 1]
    seg = wpk - prev
    Ls = np.linalg.norm(seg, axis=1)
    u = seg / np.maximum(Ls, 1e-9)[:, None]
    s_proj = np.clip(((c - prev) * u).sum(1), 0.0, Ls)
    aim = prev + (np.minimum(Ls, s_proj + plan.lookahead))[:, None] * u
    aim = np.where((Ls > 1e-6)[:, None], aim, wpk)
    d = aim - c
    nd = np.linalg.norm(d, axis=1)
    dw = np.linalg.norm(wpk - c, axis=1)
    sp = plan.cruise * np.minimum(1.0, dw / 6.0)
    out = d / np.maximum(nd, 1e-9)[:, None] * sp[:, None]
    out[nd < 1e-6] = 0.0
    out[k >= K] = 0.0
    return out


def _edges(flag):
    f = np.asarray(flag, bool)
    return int(np.sum(f[1:] & ~f[:-1]) + (1 if f.size and f[0] else 0))


def episode_truth_metrics(t, P, airborne, honest, plan, att=-1, t_boot=6.0, tilt=None, d_contact=0.8, d_coll=1.5, d_safe=2.5,
                          t_att=None, t_g=None, q_rec=None, q_ref=None, keepout_fn=None, wp_radius=None):
    """Service / safety / recovery from truth.  Returns a flat dict (JSON-able; np.inf for 'never')."""
    from .adversary import keepout_distance
    n = P.shape[1]
    honest = np.asarray(honest, bool)
    out = {}
    pts, seg, ls, cum = polyline(plan)
    Ltot = float(cum[-1])
    K = len(plan.waypoints)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    hon = honest[None, :] & airborne
    # ---- true route progress (index by honest-centroid ARRIVAL, as in the legacy truth metric)
    k_true, t_done = 0, math.inf
    r_wp = plan.tol + 1.0 if wp_radius is None else float(wp_radius)       # waypoint service radius (frozen from benign development data)
    wp_closest = np.full(K, math.inf)
    s_prof = np.full(len(t), np.nan)
    hc_all = np.full((len(t), 2), np.nan)
    for i in range(len(t)):
        h = hon[i]
        if not h.any():
            continue
        c = P[i, h, :2].mean(0)
        hc_all[i] = c
        wp_closest = np.minimum(wp_closest, np.linalg.norm(plan.waypoints - c, axis=1))
        if k_true < K and np.linalg.norm(plan.waypoints[k_true] - c) < r_wp:
            k_true += 1
            if k_true == K and not math.isfinite(t_done):
                t_done = float(t[i])
        s_prof[i] = arclength(plan, c, k_true)
    # progress never decreases for service purposes (an arclength drop is a retreat, not negative service)
    s_mono = np.fmax.accumulate(np.where(np.isnan(s_prof), -np.inf, s_prof))
    s_mono = np.where(np.isfinite(s_mono), s_mono, 0.0)
    t_end = float(t[-1]) if len(t) else 0.0
    t_score_end = min(t_done + POST_COMPLETE_S, t_end) if math.isfinite(t_done) else t_end
    win = (t >= t_boot) & (t <= t_score_end)
    out.update(t_complete=t_done, complete=int(math.isfinite(t_done)), progress=float(k_true) / K, route_len=Ltot,
               wp_closest=[float(x) for x in wp_closest], wp_radius=r_wp,
               progress_arc=float(s_mono[win].max() / Ltot) if win.any() else 0.0, t_score_end=t_score_end, t_end=t_end)
    # ---- service rate q_W on a 1 s grid (m/s), complete = target met
    grid = np.arange(0.0, t_score_end + 1e-9, 1.0)
    sg = np.interp(grid, t, s_mono)
    qW = np.full(len(grid), np.nan)
    for gi, tg in enumerate(grid):
        if tg >= W_SERVICE:
            qW[gi] = (sg[gi] - np.interp(tg - W_SERVICE, t, s_mono)) / W_SERVICE
    if q_ref is not None and math.isfinite(t_done):
        qW = np.where(grid >= t_done, q_ref, qW)
    # ---- safety (honest outcomes only; windowed to active flight)
    ko = 0.0
    ko_min = math.inf
    for i in np.nonzero(win)[0]:
        h = hon[i]
        if not h.any():
            continue
        ko += float(plan.in_keepout(P[i, h, :2]).sum()) * dt
        if i % 2 == 0:
            ko_min = min(ko_min, min(keepout_distance(plan, q) for q in P[i, h, :2]))
    out.update(keepout_s=ko, ko_min_dist=ko_min)
    D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=3)
    both = airborne[:, :, None] & airborne[:, None, :] & win[:, None, None]
    D = np.where(both, D, np.inf)
    ii = np.arange(n)
    D[:, ii, ii] = np.inf
    hh = honest[:, None] & honest[None, :]
    ha = honest[:, None] ^ honest[None, :]
    d_hh = np.min(np.where(hh[None], D, np.inf), axis=(1, 2))
    d_ha = np.min(np.where(ha[None], D, np.inf), axis=(1, 2)) if att >= 0 else np.full(len(t), np.inf)
    out.update(min_sep_hh=float(d_hh.min()), min_sep_ha=float(d_ha.min()) if att >= 0 else math.nan,
               contact_hh_events=_edges(d_hh < d_contact), contact_ha_events=_edges(d_ha < d_contact) if att >= 0 else 0,
               coll_hh_events=_edges(d_hh < d_coll), coll_ha_events=_edges(d_ha < d_coll) if att >= 0 else 0,
               near_hh_frac=float(np.mean(d_hh[win] < d_safe)) if win.any() else math.nan)
    crash = 0
    t_crash = math.inf
    if tilt is not None:
        bad = (tilt > 60.0) & honest[None, :] & win[:, None]
        crash = int(bad.any())
        if crash:
            t_crash = float(t[np.argmax(bad.any(axis=1))])
    out.update(crash_honest=crash)
    # first unsafe instant (honest keep-out entry, any honest contact, honest crash)
    unsafe_t = math.inf
    inko = np.array([plan.in_keepout(P[i, hon[i], :2]).any() if (hon[i].any() and win[i]) else False for i in range(len(t))])
    cand = []
    for flag in (inko, d_hh < d_contact, d_ha < d_contact):
        if flag.any():
            cand.append(float(t[np.argmax(flag)]))
    if math.isfinite(t_crash):
        cand.append(t_crash)
    if cand:
        unsafe_t = min(cand)
    safe = int(ko <= 0.0 and out["contact_hh_events"] == 0 and out["contact_ha_events"] == 0 and crash == 0)
    out.update(t_unsafe=unsafe_t, safe=safe, safe_complete=int(safe and math.isfinite(t_done)))
    # ---- recovery (13): first r >= 0 with q_W >= q_rec on [t_g + r, t_g + r + tau_hold]
    if t_g is not None and q_rec is not None:
        t_rec = math.inf
        ok = np.nan_to_num(qW, nan=-1.0) >= q_rec
        for r in np.arange(0.0, max(0.0, t_score_end - t_g) + 1e-9, 1.0):
            a, b = t_g + r, t_g + r + TAU_HOLD
            idx = (grid >= a - 1e-9) & (grid <= b + 1e-9)
            if b > grid[-1] + 1e-9:
                # a dwell that cannot complete before the observation ends counts only if the target was already met by completion
                if math.isfinite(t_done) and t_done <= b and idx.any() and ok[idx].all():
                    t_rec = a
                break
            if idx.any() and ok[idx].all():
                t_rec = a
                break
        out.update(t_rec=t_rec - t_g if math.isfinite(t_rec) else math.inf, t_g=t_g)
        if q_ref is not None and t_att is not None:
            m = (grid >= t_att)
            out["service_deficit"] = float(np.sum(np.clip(q_ref - np.nan_to_num(qW, nan=0.0), 0.0, None)[m]) * 1.0)
    out["_grid"] = grid
    out["_qW"] = qW
    out["_sg"] = sg
    return out


def authority_seconds(t, led, honest, att, att_flag):
    """Seconds in which at least half of the honest, airborne UAVs act on the compromised UAV as leader WHILE it is actively
    attacking: the check that a containment trial actually exercised harmful authority."""
    if att < 0:
        return 0.0
    hon = np.asarray(honest, bool)
    m = (led[:, hon] == att).sum(1) >= max(1, hon.sum() // 2 + 1)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    return float(np.sum(m & np.asarray(att_flag, bool)) * dt)


# ------------------------------------------------------------------------------------------------ premise audit (Theorem 1)
def premise_series(t, P, V, honest, offsets, wp, cobs, vused, scale, plan, att=-1, h=0.25, t_boot=6.0, t_trans=5.0, led=None):
    """Per-sample, per-honest-UAV quantities of the Theorem-1 premises (units in comments).

    z_i = p_i - s d_i                      true route anchor [m]
    zt_i = cobs_i                          local anchor estimate (median of used claims) [m]
    eps_i = |zt_i - z_i|                   anchor-estimation error [m]
    zdot_i                                 d/dt z_i by central difference over +-h [m/s]
    eta_i = zdot_i - plan(k_i, zt_i)       total deviation of the anchor velocity from the pure plan velocity [m/s]
    xi_i, y_i                              tangential / signed lateral coordinates of z_i on the follower's OWN segment k_i [m]
    interior_i                             z_i inside the untruncated pure-pursuit chart: 0 <= xi <= L_seg - lookahead - 1
    Returns dict of (T,n) arrays (NaN for non-honest)."""
    T, n = len(t), P.shape[1]
    s = np.where(np.isfinite(scale), scale, 1.0)
    z = P[:, :, :2] - s[:, :, None] * offsets[None, :, :2]
    dtm = np.median(np.diff(t))
    sh = max(1, int(round(h / dtm)))
    zdot = np.full_like(z, np.nan)
    zdot[sh:-sh] = (z[2 * sh:] - z[:-2 * sh]) / (2 * sh * dtm)
    pts = np.vstack([np.zeros(2), plan.waypoints[:, :2]])
    K = len(plan.waypoints)
    kk = np.clip(wp.astype(int), 0, K - 1)
    a, b = pts[kk], pts[kk + 1]
    sg = b - a
    L = np.linalg.norm(sg, axis=2)
    u = sg / np.maximum(L, 1e-9)[..., None]
    nrm = np.stack([-u[..., 1], u[..., 0]], axis=-1)
    rel = z - a
    xi = (rel * u).sum(-1)
    y = (rel * nrm).sum(-1)
    eps = np.linalg.norm(cobs - z, axis=-1)
    pl = np.stack([command_vec(plan, wp[:, i].astype(int), cobs[:, i]) for i in range(n)], axis=1)
    eta = np.linalg.norm(zdot - pl, axis=-1)
    interior = (xi >= 0.0) & (xi <= L - plan.lookahead - 1.0) & (wp < K)
    # settled cruise: at least t_trans s after the mission starts moving or after the last segment switch (velocity-loop transients are
    # outside the constant-speed chart; they are reported as non-interior time, not hidden)
    last = np.full(wp.shape, float(t_boot))
    cur = np.full(wp.shape[1], float(t_boot))
    for ti in range(1, len(t)):
        ch = wp[ti] != wp[ti - 1]
        if led is not None:
            ch = ch | (led[ti] != led[ti - 1])            # a change of the acted-on leader (incl. leaderless <-> led) restarts the dwell
        cur = np.where(ch, t[ti], cur)
        last[ti] = cur
    interior = interior & ((t[:, None] - last) >= t_trans)
    if led is not None:
        interior = interior & (led >= 0)                   # HOLD mode (leaderless: zero mission velocity) is outside the cruise chart
    ds = np.abs(np.diff(s, axis=0, prepend=s[:1]))
    const_scale = ds < 1e-3
    hm = np.asarray(honest, bool)
    for arr in (eps, eta, xi, y):
        arr[:, ~hm] = np.nan
    interior = interior & hm[None, :] & const_scale
    return dict(z=z, zdot=zdot, eps=eps, eta=eta, xi=xi, y=y, L=L, interior=interior, k=wp)


def theorem1_bounds(vc, ell, eps, delta, wv):
    """(22)-(26): R*, kappa(R*) for aggregate authority delta [m/s], velocity residual wv [m/s], anchor error eps [m].
    Returns dict; R_star = inf when Delta >= v_c (vacuous)."""
    D = delta + wv
    if D >= vc:
        return dict(Delta=D, R_star=math.inf, kappa=-math.inf, vacuous=True)
    R = eps + ell * D / math.sqrt(vc * vc - D * D)
    kappa = vc * ell / math.sqrt(ell * ell + (R + eps) ** 2) - D
    return dict(Delta=D, R_star=R, kappa=kappa, vacuous=False)
