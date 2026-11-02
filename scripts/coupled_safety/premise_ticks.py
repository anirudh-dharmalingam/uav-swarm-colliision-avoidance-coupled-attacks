#!/usr/bin/env python3
"""Independent per-tick premise audit of one episode against Gazebo truth (evaluation/observer only; never fed back to a controller).

    python3 scripts/coupled_safety/premise_ticks.py <episode_dir> --out ticks.csv          (inside the image)

For every honest-agent MISSION tick t: ego (EKF vs truth <= declared e_p, e_v), track (max over [t, t+h] of ||a_truth - [u_x,u_y,0]|| <= W),
rep (every neighbour within 12 m: its report error over [t-0.5, t] <= beta + sender error, position and velocity), acc (neighbour truth
acceleration over [t-0.5, t+h] <= A_any), net (per-receiver measured transit never exceeded T_net_max).  premise_ok = all five.  Truth derivatives:
Savitzky-Golay 0.2 s (see calibration notes on differentiation sensitivity).  Same definitions as scripts/analysis/ral26_audit_v2.py, but tick-resolved
so the timeline figure and the video overlay can show the AUDITED premise state beside the controller's own label.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import ral26_audit_v2 as AU  # noqa: E402


def ticks(rd: Path):
    cfg = json.loads((rd / "resolved_config.json").read_text())
    R = cfg["ral26"]
    B = R["bounds"]
    n, att = int(cfg["swarm"]["num_uavs"]), int(R["attacker_id"])
    ev = json.loads((rd / "eval" / "summary.json").read_text())
    T = pd.read_csv(rd / "eval" / "truth.csv")
    ts = T.t_sim.to_numpy() - ev["t_mission_start_sim"]
    dt = 0.02
    grid = np.arange(max(0.0, ts[0]), ts[-1], dt)
    P = np.stack([np.stack([np.interp(grid, ts, T[f"{c}{i}"]) for c in "xyz"], 1) for i in range(n)], 1)
    V = np.stack([np.stack([AU.sg(P[:, i, k], 11, 2, 1, dt) for k in range(3)], 1) for i in range(n)], 1)
    Ac = np.stack([np.stack([AU.sg(P[:, i, k], 11, 2, 2, dt) for k in range(3)], 1) for i in range(n)], 1)
    amag = np.linalg.norm(Ac, axis=2)
    D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=3) + np.eye(n)[None] * 1e9
    ag, lag = {}, {}
    for i in range(n):
        d = pd.read_csv(rd / "agents" / f"agent_{i}.csv.gz")
        d = d[d.phase == "MISSION"].reset_index(drop=True)
        ag[i] = d
        lag[i], _ = AU.align(d, grid, P, i)
    L = float(np.median(list(lag.values())))
    h, W, A_any = float(B["h"]), float(B["W"]), float(B["A_any"])
    ep, evv = float(B["ego_err_p"]), float(B["ego_err_v"])
    bp, bv = float(B["beta_p"]) + float(B.get("sender_err_p", 0)), float(B["beta_v"]) + float(B.get("sender_err_v", 0))
    # report errors of every sender on the truth grid
    rep = {}
    for j in range(n):
        d = ag[j]
        tt = d.t.to_numpy(float) - L
        rx, ry, rvx, rvy = (d[c].to_numpy(float) for c in ("rpx", "rpy", "rvx", "rvy"))
        ex = np.hypot(rx - np.interp(tt, grid, P[:, j, 0]), ry - np.interp(tt, grid, P[:, j, 1]))
        ev_ = np.hypot(rvx - np.interp(tt, grid, V[:, j, 0]), rvy - np.interp(tt, grid, V[:, j, 1]))
        rep[j] = (tt, ex, ev_)
    net_ok = {}
    for i in range(n):
        f = rd / "agents" / f"agent_{i}_summary.json"
        s = json.loads(f.read_text()) if f.exists() else {}
        tr = s.get("transit_s")
        net_ok[i] = True if tr is None else tr["frac_over_T_net_max"] == 0.0
    rows = []
    for i in range(n):
        if i == att:
            continue
        d = ag[i]
        tt = d.t.to_numpy(float) - L
        pt = np.stack([np.interp(tt, grid, P[:, i, k]) for k in range(3)], 1)
        vt = np.stack([np.interp(tt, grid, V[:, i, k]) for k in range(3)], 1)
        e_p = np.linalg.norm(d[["px", "py", "pz"]].to_numpy(float) - pt, axis=1)
        e_v = np.linalg.norm(d[["vx", "vy", "vz"]].to_numpy(float) - vt, axis=1)
        k_cmd = np.clip(np.searchsorted(tt, grid, side="right") - 1, 0, len(tt) - 1)
        ucmd = np.stack([d.ux.to_numpy(float)[k_cmd], d.uy.to_numpy(float)[k_cmd], np.zeros(len(grid))], 1)
        res = np.linalg.norm(Ac[:, i, :] - ucmd, axis=1)
        k0, k1, kb = np.searchsorted(grid, tt), np.searchsorted(grid, tt + h), np.searchsorted(grid, tt - 0.5)
        ok_t = (tt > grid[0]) & (tt + h < grid[-1])
        c_track = AU.win_max(res, np.clip(k0, 0, len(grid) - 1), np.clip(k1, 0, len(grid) - 1)) <= W
        c_rep = np.ones(len(tt), bool)
        c_acc = np.ones(len(tt), bool)
        for j in range(n):
            if j == i:
                continue
            near = D[np.clip(k0, 0, len(grid) - 1), i, j] <= 12.0
            c_acc &= ~near | (AU.win_max(amag[:, j], np.clip(kb, 0, len(grid) - 1), np.clip(k1, 0, len(grid) - 1)) <= A_any)
            tj, ejp, ejv = rep[j]
            a0 = np.searchsorted(tj, tt - 0.5)
            a1 = np.clip(np.searchsorted(tj, tt, side="right") - 1, 0, len(tj) - 1)
            c_rep &= ~near | ((AU.win_max(ejp, a0, a1) <= bp) & (AU.win_max(ejv, a0, a1) <= bv))
        c_ego = (e_p <= ep) & (e_v <= evv)
        ok = c_ego & c_track & c_rep & c_acc & net_ok[i] & ok_t
        dmin = np.min(D[np.clip(k0, 0, len(grid) - 1), i, :], axis=1)
        rows.append(pd.DataFrame({"agent": i, "t": d.t.to_numpy(float), "t_truth": tt, "status": d.status.to_numpy(), "ego_ok": c_ego, "track_ok": c_track, "rep_ok": c_rep,
                                  "acc_ok": c_acc, "net_ok": net_ok[i], "premise_ok": ok, "min_dist_any": dmin}))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = ticks(Path(a.episode))
    df.to_csv(a.out, index=False)
    print(len(df), "ticks; premise_ok", float(df.premise_ok.mean()), "certified", float((df.status == "CERTIFIED_INTERVAL").mean()))
