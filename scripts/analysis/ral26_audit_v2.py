#!/usr/bin/env python3
"""RAL26 v2 certificate + premise audit against Gazebo truth (evaluation only; never fed back).

Differences from ral26_audit.py (kept unchanged for the frozen HF2/HF3/HF6 numbers):
  * per-agent clock alignment over +-4 s (v1 used one agent and +-2 s; some HF6 lags hit the edge);
  * Savitzky-Golay derivatives (0.2 s window) instead of 5-point box smoothing, at the truth rate;
  * EVERY certified tick is checked against EVERY premise, not only ticks whose outcome failed:
      ego   ||p_est - p_true|| <= e_p,  ||v_est - v_true|| <= e_v
      track ||a_true - [u_x, u_y, 0]|| <= W              over [t, t+h]
      rep   ||report_j - p_true_j|| <= beta_p + sender_err_p (honest j: its own estimate; attacker: its
            broadcast rpx/rpy, v2 logs only), velocity likewise with beta_v + sender_err_v
      acc   ||a_true_j|| <= A_any                        over [t - 0.5, t + h]
      net   per-receiver measured transit <= T_net_max   (agent summary, v2 runs only)
    for every neighbour within 12 m truth distance;
  * outcome: truth separation of agent i to every vehicle over [t, t+h] >= d_safe.
A certified tick with a failed outcome while all premises hold would falsify the conditional claim.

    python3 scripts/analysis/ral26_audit_v2.py results/ral26/v2_main [...]   -> <root>/audit_v2.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NEIGH_R = 12.0
REP_WIN = 0.5           # a received packet is at most this old in the audited regimes (5 Hz + T_net_max)


def sg_coeffs(win: int, order: int, deriv: int, dt: float):
    half = win // 2
    x = np.arange(-half, half + 1) * dt
    V = np.vander(x, order + 1, increasing=True)
    pinv = np.linalg.pinv(V)
    import math
    return pinv[deriv] * math.factorial(deriv)


def sg(z, win, order, deriv, dt):
    c = sg_coeffs(win, order, deriv, dt)[::-1]
    half = win // 2
    zp = np.pad(z, half, mode="edge")
    return np.convolve(zp, c, "valid")


def win_max(x, k0, k1):
    """max of x over rows [k0, k1] (inclusive), vectorised for arrays of window bounds."""
    out = np.empty(len(k0))
    for m, (a, b) in enumerate(zip(k0, k1)):
        out[m] = x[a:b + 1].max() if b >= a else np.nan
    return out


def win_min(x, k0, k1):
    out = np.empty(len(k0))
    for m, (a, b) in enumerate(zip(k0, k1)):
        out[m] = x[a:b + 1].min() if b >= a else np.nan
    return out


def align(d, grid, P, i):
    """lag L such that agent time t_a corresponds to truth mission time t_a - L."""
    g = grid[(grid > d.t.min() + 2) & (grid < d.t.max() - 6)]
    if len(g) < 50:
        return 0.0, np.nan
    g = g[:: max(1, len(g) // 600)]
    xi, yi = np.interp(g, grid, P[:, i, 0]), np.interp(g, grid, P[:, i, 1])
    best = (np.inf, 0.0)
    for cand in np.arange(-4.0, 4.0, 0.01):
        e = np.mean(np.hypot(np.interp(g + cand, d.t, d.px) - xi, np.interp(g + cand, d.t, d.py) - yi))
        if e < best[0]:
            best = (e, float(cand))
    return best[1], best[0]


def audit_episode(rd: Path):
    cfg = json.loads((rd / "resolved_config.json").read_text())
    if cfg.get("controller", {}).get("mode") != "ral26":
        return None
    R = cfg["ral26"]
    B = R["bounds"]
    n = int(cfg["swarm"]["num_uavs"])
    att = int(R.get("attacker_id", -1))
    ev = json.loads((rd / "eval" / "summary.json").read_text())
    t0 = ev.get("t_mission_start_sim")
    if t0 is None:
        return None
    T = pd.read_csv(rd / "eval" / "truth.csv")
    ts = T.t_sim.to_numpy() - t0
    dt = float(np.median(np.diff(ts)))
    dt = 0.02 if dt < 0.04 else 0.1
    grid = np.arange(max(0.0, ts[0]), ts[-1], dt)
    win = max(5, int(round(0.2 / dt)) | 1)
    P = np.stack([np.stack([np.interp(grid, ts, T[f"{c}{i}"]) for c in "xyz"], 1) for i in range(n)], 1)
    V = np.stack([np.stack([sg(P[:, i, k], win, 2, 1, dt) for k in range(3)], 1) for i in range(n)], 1)
    Acc = np.stack([np.stack([sg(P[:, i, k], win, 2, 2, dt) for k in range(3)], 1) for i in range(n)], 1)
    amag = np.linalg.norm(Acc, axis=2)
    D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=3) + np.eye(n)[None] * 1e9
    h = float(B["h"])
    ep, evv = float(B["ego_err_p"]), float(B["ego_err_v"])
    bp = float(B["beta_p"]) + float(B.get("sender_err_p", 0.0))
    bv = float(B["beta_v"]) + float(B.get("sender_err_v", 0.0))
    W, A_any, dsafe = float(B["W"]), float(B["A_any"]), float(B["d_safe"])
    agents, lags = {}, {}
    for i in range(n):
        f = rd / "agents" / f"agent_{i}.csv.gz"
        if f.exists():
            d = pd.read_csv(f)
            d = d[d.phase == "MISSION"] if "phase" in d else d
            if len(d) > 100:
                agents[i] = d
                lags[i], _ = align(d, grid, P, i)
    # all agents share one container clock and one GCS t0 -> one lag; the median is robust to agents
    # that barely move (e.g. immediate energy return), whose own alignment is ill-conditioned
    if lags:
        Lm = float(np.median(list(lags.values())))
        lags = {i: Lm for i in lags}
    # physical crash: first truth sample (mission window) with any vehicle tilt > 60 deg
    tilt = np.stack([np.interp(grid, ts, T[f"tilt{i}"]) for i in range(n)], 1) if "tilt0" in T else np.zeros((len(grid), n))
    crash_k = np.flatnonzero((tilt.max(axis=1) > 60.0) & (grid >= 0.0))
    t_crash = float(grid[crash_k[0]]) if len(crash_k) else None
    # body contact (centre distance < 0.8 m ~ two X500 rotor-tip envelopes): impact accelerations start
    # before the tilt exceeds 60 deg, so premise maxima stop 0.3 s before the first contact
    cont_k = np.flatnonzero((D.min(axis=(1, 2)) < 0.8) & (grid >= 0.0))
    t_contact = float(grid[cont_k[0]]) if len(cont_k) else None
    if t_contact is not None:
        t_crash = t_contact - 0.3 if t_crash is None else min(t_crash, t_contact - 0.3)
    t_mis_end = max((float(d.t.max()) - lags[i] for i, d in agents.items()), default=float(grid[-1]))
    t_ok_end = min(t_mis_end, t_crash if t_crash is not None else np.inf)
    okg = (grid >= 0.0) & (grid <= t_ok_end)
    # per-vehicle report error vs truth on the truth grid (sender side)
    rep_err_p, rep_err_v = {}, {}
    for j, d in agents.items():
        tt = d.t.to_numpy() - lags[j]
        has_rep = "rpx" in d and d.rpx.notna().any()
        rx = d.rpx.to_numpy() if has_rep else d.px.to_numpy()
        ry = d.rpy.to_numpy() if has_rep else d.py.to_numpy()
        rvx = d.rvx.to_numpy() if has_rep else d.vx.to_numpy()
        rvy = d.rvy.to_numpy() if has_rep else d.vy.to_numpy()
        if j == att and not has_rep:          # v1 logs: attacker broadcast not recorded -> unknown
            rep_err_p[j] = rep_err_v[j] = None
            continue
        ptx, pty = np.interp(tt, grid, P[:, j, 0]), np.interp(tt, grid, P[:, j, 1])
        vtx, vty = np.interp(tt, grid, V[:, j, 0]), np.interp(tt, grid, V[:, j, 1])
        rep_err_p[j] = (tt, np.hypot(rx - ptx, ry - pty))
        rep_err_v[j] = (tt, np.hypot(rvx - vtx, rvy - vty))
    transit_ok = {}
    for i in range(n):
        f = rd / "agents" / f"agent_{i}_summary.json"
        if f.exists():
            s = json.loads(f.read_text())
            tr = s.get("transit_s")
            transit_ok[i] = None if tr is None else tr["frac_over_T_net_max"] == 0.0
    out = {"name": rd.name, "attacker": att, "dt_truth": dt, "lags": lags, "cert_ticks": 0,
           "prem_ego": 0, "prem_track": 0, "prem_rep": 0, "prem_acc": 0, "prem_all": 0, "prem_rep_unknown": 0,
           "viol": 0, "viol_prem_ok": 0, "transit_ok": transit_ok,
           "ego_err_p_max": 0.0, "ego_err_v_max": 0.0, "track_res_max": 0.0,
           "rep_err_p_max_honest": 0.0, "rep_err_v_max_honest": 0.0,
           "rep_err_p_max_attacker": None, "acc_max_attacker": float(amag[okg, att].max()) if att >= 0 and okg.any() else None,
           "acc_max_honest": float(np.delete(amag[okg], att, axis=1).max()) if att >= 0 and okg.any() else float(amag[okg].max()),
           "t_crash": t_crash, "t_contact_0p8": t_contact, "crash_vehicles": [int(k) for k in np.flatnonzero(tilt[grid >= 0].max(axis=0) > 60.0)]}
    for j, v in rep_err_p.items():
        if v is None:
            continue
        sel = v[0] <= t_ok_end
        if not sel.any():
            continue
        m = float(np.nanquantile(v[1][sel], 0.999))
        mv = float(np.nanquantile(rep_err_v[j][1][sel], 0.999))
        if j == att:
            out["rep_err_p_max_attacker"] = m
            out["rep_err_v_max_attacker"] = mv
        else:
            out["rep_err_p_max_honest"] = max(out["rep_err_p_max_honest"], m)
            out["rep_err_v_max_honest"] = max(out["rep_err_v_max_honest"], mv)
    for i, d in agents.items():
        if i == att:
            continue
        L = lags[i]
        tt = d.t.to_numpy() - L
        ok_t = (tt > grid[0]) & (tt + h < grid[-1]) & (tt <= t_ok_end)
        pt = np.stack([np.interp(tt, grid, P[:, i, k]) for k in range(3)], 1)
        vt = np.stack([np.interp(tt, grid, V[:, i, k]) for k in range(3)], 1)
        e_p = np.linalg.norm(d[["px", "py", "pz"]].to_numpy() - pt, axis=1)
        e_v = np.linalg.norm(d[["vx", "vy", "vz"]].to_numpy() - vt, axis=1)
        out["ego_err_p_max"] = max(out["ego_err_p_max"], float(np.quantile(e_p[ok_t], 0.999)))
        out["ego_err_v_max"] = max(out["ego_err_v_max"], float(np.quantile(e_v[ok_t], 0.999)))
        # tracking residual on the truth grid vs the command in force (zero-order hold of agent ticks)
        k_cmd = np.clip(np.searchsorted(tt, grid, side="right") - 1, 0, len(tt) - 1)
        ucmd = np.stack([d.ux.to_numpy()[k_cmd], d.uy.to_numpy()[k_cmd], np.zeros(len(grid))], 1)
        inwin = (grid >= tt[0]) & (grid <= min(tt[-1], t_ok_end))
        res = np.where(inwin, np.linalg.norm(Acc[:, i, :] - ucmd, axis=1), 0.0)
        out["track_res_max"] = max(out["track_res_max"], float(np.quantile(res[inwin], 0.999)) if inwin.any() else 0.0)
        cert = (d.status.to_numpy() == "CERTIFIED_INTERVAL") & ok_t
        if not cert.any():
            continue
        tc = tt[cert]
        k0 = np.searchsorted(grid, tc)
        k1 = np.searchsorted(grid, tc + h)
        kb = np.searchsorted(grid, tc - REP_WIN)
        c_ego = (e_p[cert] <= ep) & (e_v[cert] <= evv)
        c_track = win_max(res, k0, k1) <= W
        dmin_i = D[:, i, :].min(axis=1)
        outcome_ok = win_min(dmin_i, k0, k1) >= dsafe
        c_rep = np.ones(len(tc), bool)
        c_acc = np.ones(len(tc), bool)
        unknown = np.zeros(len(tc), bool)
        for j in range(n):
            if j == i:
                continue
            near = D[k0, i, j] <= NEIGH_R
            c_acc &= ~near | (win_max(amag[:, j], kb, k1) <= A_any)
            if j not in rep_err_p or rep_err_p[j] is None:
                unknown |= near
                continue
            tj, ejp = rep_err_p[j]
            _, ejv = rep_err_v[j]
            a0 = np.searchsorted(tj, tc - REP_WIN)
            a1 = np.clip(np.searchsorted(tj, tc, side="right") - 1, 0, len(tj) - 1)
            ok = (win_max(ejp, a0, a1) <= bp) & (win_max(ejv, a0, a1) <= bv)
            c_rep &= ~near | ok
        allp = c_ego & c_track & c_rep & c_acc & ~unknown
        if transit_ok.get(i) is False:
            allp &= False
        out["cert_ticks"] += int(cert.sum())
        out["prem_ego"] += int(c_ego.sum())
        out["prem_track"] += int(c_track.sum())
        out["prem_rep"] += int((c_rep & ~unknown).sum())
        out["prem_rep_unknown"] += int(unknown.sum())
        out["prem_acc"] += int(c_acc.sum())
        out["prem_all"] += int(allp.sum())
        out["viol"] += int((~outcome_ok).sum())
        out["viol_prem_ok"] += int((~outcome_ok & allp).sum())
    # separations at the truth rate
    hon = [k for k in range(n) if k != att]
    dhh = min(float(D[:, a, b].min()) for a in hon for b in hon if a < b)
    out["min_sep_hh"] = dhh
    out["min_sep_ha"] = float(D[:, att, :].min()) if att >= 0 else None
    return out


def main():
    rows = []
    for root in sys.argv[1:]:
        rr = []
        for rd in sorted(p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith("_")):
            try:
                r = audit_episode(rd)
            except Exception as exc:
                r = {"name": rd.name, "error": repr(exc)}
            if r:
                rr.append(r)
        Path(root, "audit_v2.json").write_text(json.dumps(rr, indent=1, default=float))
        rows += rr
    keys = ("cert_ticks", "prem_ego", "prem_track", "prem_rep", "prem_rep_unknown", "prem_acc", "prem_all",
            "viol", "viol_prem_ok")
    tot = {k: sum(r.get(k, 0) for r in rows) for k in keys}
    print("episodes", len(rows), "errors", sum(1 for r in rows if "error" in r))
    print(json.dumps(tot))
    if tot["cert_ticks"]:
        print("fraction of certified ticks whose premises all held: %.3f" % (tot["prem_all"] / tot["cert_ticks"]))
    for k in ("ego_err_p_max", "ego_err_v_max", "track_res_max", "rep_err_p_max_honest", "rep_err_v_max_honest",
              "rep_err_p_max_attacker", "acc_max_attacker", "acc_max_honest"):
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            print(f"{k}: max {max(v):.3f}  median {float(np.median(v)):.3f}")
    for r in rows:
        if "error" in r:
            print(r["name"], r["error"])


if __name__ == "__main__":
    main()
