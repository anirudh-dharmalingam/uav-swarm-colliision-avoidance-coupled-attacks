"""Attrition-free service, fixed-horizon safety and dwell-safe recovery metrics (v2 campaign).

Fixes demonstrated defects of swarmsec/le/rmetrics.py (frozen v1): the honest-centroid of *whichever UAVs are airborne* changes when a member lands
(a stationary swarm gains progress when a trailing UAV departs), waypoint-disk arrival and arclength resets create progress jumps, the scoring window
ends at completion + 10 s (outcome dependent), and `recovered-safe` checked safety only until recovery STARTED.

Conventions (all outcome quantities are computed from TRUE positions; nothing a leader announces is an input):
  roster        fixed set R0 of honest UAVs at mission start (attacker excluded). A departure/crash/landing never leaves the roster.
  s_i(t)        individual route progress = arclength of the nearest point of the route polyline to UAV i (continuous at corners: no waypoint-disk
                jump, no index reset), held at its last value once i is no longer airborne; `progress` below is the running maximum (monotone).
  cohort S(t)   mean over the FULL roster of the monotone individual progress: a departure freezes that UAV's term, it cannot create progress.
  capacity      initial |R0|, active n_active(t), lost = |R0| - n_active(t_end) are reported separately from completed work.
  tasks         ordered service regions at the route waypoints, each needing >= m_quorum distinct airborne honest UAVs within r_task for a continuous
                dwell tau_task; priority weights; deadline per task. Utility U = sum of weights of tasks completed by their deadline. Physical presence only:
                this is NOT delivered information (no data-delivery process is modelled).
  horizon       one fixed scoring window [t_boot, horizon] for every method (an earlier completion does not shorten the safety observation).
  recovery      q_W(t) = (1/|R0|) sum_i max(0, s_i(t) - s_i(t-W))/W (mean positive individual progress rate; departed UAVs contribute 0).
                Recovered at the COMPLETION of the first dwell [t_g + r, t_g + r + tau_hold] on which q_W >= q_rec throughout AND no unsafe event occurs on
                [t_g, t_g + r + tau_hold]. Recovery time = r + tau_hold (deadline applies to completion). Terminal convention: if every task is complete at
                time C <= the dwell end and no unsafe event occurred up to C, further service is irrelevant and recovery is credited at max(C, t_g).
                Non-recovery stays in the denominator; unsafe/crash outcomes are competing failures.
"""
from __future__ import annotations

import math

import numpy as np

W_SERVICE = 5.0
TAU_HOLD = 10.0


def polyline(plan):
    pts = np.vstack([np.zeros(2), plan.waypoints[:, :2]])
    seg = np.diff(pts, axis=0)
    ls = np.linalg.norm(seg, axis=1)
    return pts, seg, ls, np.r_[0.0, np.cumsum(ls)]


def project_arclength(plan, xy):
    """Arclength of the nearest polyline point for each row of xy (M,2); also the cross-track distance."""
    pts, seg, ls, cum = polyline(plan)
    best = np.full(len(xy), np.inf)
    arc = np.zeros(len(xy))
    for k in range(len(ls)):
        u = seg[k] / max(ls[k], 1e-9)
        tt = np.clip(((xy - pts[k]) @ u), 0.0, ls[k])
        d = np.linalg.norm(xy - (pts[k] + tt[:, None] * u), axis=1)
        m = d < best
        best = np.where(m, d, best)
        arc = np.where(m, cum[k] + tt, arc)
    return arc, best


def individual_progress(t, P, airborne, plan):
    """s_i(t) monotone, held after departure.  Returns (T,n) progress and (T,n) cross-track."""
    T, n = P.shape[0], P.shape[1]
    S = np.zeros((T, n))
    X = np.zeros((T, n))
    for i in range(n):
        arc, ct = project_arclength(plan, P[:, i, :2])
        X[:, i] = ct
        ok = airborne[:, i]
        # a UAV that is not airborne keeps its last airborne value; before it is first airborne the value is 0
        arc_air = np.where(ok, arc, np.nan)
        run = np.fmax.accumulate(np.where(np.isnan(arc_air), -np.inf, arc_air))
        S[:, i] = np.where(np.isfinite(run), run, 0.0)
    return S, X


def tasks_from_plan(plan, deadlines, weights=None, r_task=6.0, m_quorum=4, tau_task=3.0):
    K = len(plan.waypoints)
    w = weights if weights is not None else [1.0] * (K - 1) + [2.0]
    return [dict(k=k, center=plan.waypoints[k, :2], r=r_task, m=m_quorum, tau=tau_task, deadline=float(deadlines[k] if hasattr(deadlines, "__len__") else deadlines),
                 weight=float(w[k]), release=0.0) for k in range(K)]


def task_completion(t, P, airborne, roster, tasks):
    """Ordered tasks: task k can only start after task k-1 is complete (the route is ordered). Returns completion times (inf if never)."""
    C = []
    prev = -math.inf
    dt = float(np.median(np.diff(t)))
    for tk in tasks:
        d = np.linalg.norm(P[:, :, :2] - tk["center"][None, None, :], axis=2)
        inside = (d <= tk["r"]) & airborne & roster[None, :]
        cnt = inside.sum(1)
        ok = (cnt >= tk["m"]) & (t >= max(prev, tk["release"]))
        run = 0.0
        done = math.inf
        for i in range(len(t)):
            run = run + dt if ok[i] else 0.0
            if run >= tk["tau"]:
                done = float(t[i])
                break
        C.append(done)
        prev = done if math.isfinite(done) else math.inf
        if not math.isfinite(done):
            prev = math.inf
    return C


def episode_metrics_v2(t, P, airborne, honest, plan, att=-1, t_boot=6.0, horizon=None, tilt=None, d_contact=0.8, d_coll=1.5, t_g=None,
                       q_rec=None, tasks=None, W=W_SERVICE, tau_hold=TAU_HOLD, keepout_fn=None):
    from swarmsec.le.adversary import keepout_distance  # noqa: F401  (geometry helper reused unchanged)
    n = P.shape[1]
    honest = np.asarray(honest, bool)
    roster = honest.copy()
    hz = float(t[-1]) if horizon is None else float(horizon)
    win = (t >= t_boot) & (t <= hz)
    dt = float(np.median(np.diff(t)))
    out = {"horizon": hz, "roster_n": int(roster.sum())}
    S, X = individual_progress(t, P, airborne, plan)
    Ltot = float(polyline(plan)[3][-1])
    cohort = S[:, roster].mean(1)
    out["cohort_progress_end"] = float(cohort[win][-1] / Ltot) if win.any() else 0.0
    out["cohort_progress_max"] = float(cohort[win].max() / Ltot) if win.any() else 0.0
    act = (airborne & roster[None, :]).sum(1)
    out["active_start"] = int(act[win][0]) if win.any() else 0
    out["active_end"] = int(act[win][-1]) if win.any() else 0
    out["lost_capacity"] = out["roster_n"] - out["active_end"]
    # departures are not crashes: first time a roster UAV is no longer airborne after being airborne
    # --- tasks
    if tasks is not None:
        C = task_completion(t, P, airborne, roster, tasks)
        out["task_completion"] = C
        out["tasks_done"] = int(sum(math.isfinite(c) for c in C))
        out["utility"] = float(sum(tk["weight"] for tk, c in zip(tasks, C) if c <= tk["deadline"]))
        out["utility_max"] = float(sum(tk["weight"] for tk in tasks))
        out["all_tasks_by_deadline"] = int(all(c <= tk["deadline"] for tk, c in zip(tasks, C)))
        out["t_all_tasks"] = float(max(C)) if all(math.isfinite(c) for c in C) else math.inf
    else:
        out["t_all_tasks"] = math.inf
    # --- safety over the FIXED horizon, roster UAVs only, airborne only
    hon = roster[None, :] & airborne
    inko = np.array([plan.in_keepout(P[i, hon[i], :2]).any() if (hon[i].any() and win[i]) else False for i in range(len(t))])
    ko = float(np.sum([plan.in_keepout(P[i, hon[i], :2]).sum() for i in np.nonzero(win)[0] if hon[i].any()]) * dt)
    D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=3)
    both = airborne[:, :, None] & airborne[:, None, :] & win[:, None, None]
    D = np.where(both, D, np.inf)
    ii = np.arange(n)
    D[:, ii, ii] = np.inf
    hh = roster[:, None] & roster[None, :]
    ha = roster[:, None] ^ roster[None, :]
    d_hh = np.min(np.where(hh[None], D, np.inf), axis=(1, 2))
    d_ha = np.min(np.where(ha[None], D, np.inf), axis=(1, 2)) if att >= 0 else np.full(len(t), np.inf)
    crash_t = math.inf
    if tilt is not None:
        bad = (tilt > 60.0) & roster[None, :] & win[:, None]
        if bad.any():
            crash_t = float(t[np.argmax(bad.any(axis=1))])
    cand = [float(t[np.argmax(f)]) for f in (inko, d_hh < d_contact, d_ha < d_contact) if f.any()]
    if math.isfinite(crash_t):
        cand.append(crash_t)
    t_unsafe = min(cand) if cand else math.inf
    out.update(keepout_s=ko, min_sep_hh=float(d_hh.min()), min_sep_ha=float(d_ha.min()) if att >= 0 else math.nan, t_unsafe=t_unsafe,
               safe_full=int(not math.isfinite(t_unsafe)), crash=int(math.isfinite(crash_t)))
    # --- recovery (sustained, dwell-safe)
    if t_g is not None and q_rec is not None:
        grid = np.arange(0.0, hz + 1e-9, 1.0)
        Sg = np.stack([np.interp(grid, t, S[:, i]) for i in range(n)], 1)[:, roster]
        qW = np.full(len(grid), np.nan)
        for gi, tg in enumerate(grid):
            if tg >= W:
                prev = np.stack([np.interp(tg - W, t, S[:, i]) for i in np.nonzero(roster)[0]])
                qW[gi] = float(np.mean(np.clip(Sg[gi] - prev, 0.0, None)) / W)
        ok = np.nan_to_num(qW, nan=-1.0) >= q_rec
        t_rec_complete = math.inf
        reason = "not_recovered"
        t_all = out["t_all_tasks"]
        for r in np.arange(0.0, max(0.0, hz - t_g - tau_hold) + 1e-9, 1.0):
            a, b = t_g + r, t_g + r + tau_hold
            idx = (grid >= a - 1e-9) & (grid <= b + 1e-9)
            if idx.any() and ok[idx].all():
                if t_unsafe <= b:
                    reason = "unsafe_in_dwell" if t_unsafe >= t_g else reason
                    t_rec_complete = math.inf
                else:
                    t_rec_complete = r + tau_hold
                    reason = "recovered"
                break
        if not math.isfinite(t_rec_complete) and reason == "not_recovered":
            # terminal convention
            if math.isfinite(t_all) and t_all >= t_g and t_unsafe > t_all:
                t_rec_complete = max(t_all - t_g, 0.0)
                reason = "recovered_terminal"
            elif math.isfinite(t_unsafe) and t_unsafe >= t_g:
                reason = "unsafe_before_recovery"
        out.update(t_rec_complete=t_rec_complete, rec_reason=reason, q_rec=q_rec, t_g=t_g)
        out["_qW"] = qW
        out["_grid"] = grid
    out["_cohort"] = cohort
    return out
