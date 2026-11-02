#!/usr/bin/env python3
"""Gate C: operating-domain residual/estimation bounds from the 24 HF1 calibration flights (train seeds 11000-11011, holdout 11100-11111).

Per vehicle-tick (offboard, 50 Hz truth, clock aligned per flight by the audit's +-4 s search):
  ego_p  = || EKF position - truth position ||,   ego_v = || EKF velocity - truth velocity ||                      (ego estimation error)
  W_xy3  = || a_truth - [u_x, u_y, 0] ||  (3-D, so the altitude-hold deviation is included)                       (tracking residual W)
  a_mag  = || a_truth_xy ||                                                                                        (physical acceleration)
truth derivatives: Savitzky-Golay (2nd order) over `--win` seconds (default 0.2 s); sensitivity to the window (0.1/0.2/0.3 s) and to PX4's own
acceleration estimate is reported -- numerical differentiation of 50 Hz positions amplifies noise, so W/a_mag from truth are window-dependent.
Declared bounds = ceil_0.05( margin x TRAIN max ), computed BEFORE the holdout is examined; every holdout exceedance is then listed.

    python3 scripts/coupled_safety/calib_bounds.py --root results/coupled_safety_v1/calib --margin 1.15 --out output/research/coupled_safety_v1/calibration
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import ral26_audit_v2 as AU  # noqa: E402

OFFBOARD = 14


def flight(rd: Path, win_s):
    cfg = json.loads((rd / "resolved_config.json").read_text())
    n = int(cfg["swarm"]["num_uavs"])
    ev = json.loads((rd / "eval" / "summary.json").read_text())
    t0 = ev.get("t_mission_start_sim")
    if t0 is None:
        return None
    T = pd.read_csv(rd / "eval" / "truth.csv")
    ts = T.t_sim.to_numpy() - t0
    dt = 0.02
    grid = np.arange(max(0.0, ts[0]), ts[-1], dt)
    P = np.stack([np.stack([np.interp(grid, ts, T[f"{c}{i}"]) for c in "xyz"], 1) for i in range(n)], 1)
    out = {w: {} for w in win_s}
    rows = []
    for i in range(n):
        f = next(iter(sorted((rd / "agents").glob(f"hf1_{i}.csv*"))), None)
        if f is None:
            continue
        d = pd.read_csv(f)
        d = d[d.nav == OFFBOARD].reset_index(drop=True)
        if len(d) < 200:
            continue
        lag, _ = AU.align(d.rename(columns={}), grid, P, i)
        tt = d.t.to_numpy() - lag
        ok = (tt > grid[0] + 1) & (tt < grid[-1] - 1)
        for w in win_s:
            win = max(5, int(round(w / dt)) | 1)
            V = np.stack([AU.sg(P[:, i, k], win, 2, 1, dt) for k in range(3)], 1)
            Ac = np.stack([AU.sg(P[:, i, k], win, 2, 2, dt) for k in range(3)], 1)
            pt = np.stack([np.interp(tt, grid, P[:, i, k]) for k in range(3)], 1)
            vt = np.stack([np.interp(tt, grid, V[:, k]) for k in range(3)], 1)
            at = np.stack([np.interp(tt, grid, Ac[:, k]) for k in range(3)], 1)
            ego_p = np.linalg.norm(d[["px", "py", "pz"]].to_numpy() - pt, axis=1)
            ego_v = np.linalg.norm(d[["vx", "vy", "vz"]].to_numpy() - vt, axis=1)
            u = np.stack([d.ux.to_numpy(), d.uy.to_numpy(), np.zeros(len(d))], 1)
            W3 = np.linalg.norm(at - u, axis=1)
            Wxy = np.linalg.norm(at[:, :2] - u[:, :2], axis=1)
            amag = np.linalg.norm(at[:, :2], axis=1)
            ekf_acc_err = np.linalg.norm(np.stack([d.ax, d.ay], 1) - at[:, :2], axis=1)
            out[w][i] = dict(ego_p=ego_p[ok], ego_v=ego_v[ok], W3=W3[ok], Wxy=Wxy[ok], amag=amag[ok], az=np.abs(at[ok, 2]),
                             ekf_acc_err=ekf_acc_err[ok], unorm=np.hypot(d.ux, d.uy).to_numpy()[ok])
    return out


def agg(per_flight, split_of, w, key, umax=None, hold="holdout"):
    """TRAIN ticks and ticks of the split(s) named by `hold` ('holdout' = every non-train split); umax restricts to ticks whose command
    satisfies ||u|| <= umax (the operating domain the filter admits)."""
    tr, ho = [], []
    for name, o in per_flight.items():
        for i, v in o[w].items():
            x = v[key] if umax is None else v[key][v["unorm"] <= umax + 1e-9]
            sp = split_of[name]
            if sp == "train":
                tr.append(x)
            elif hold == "all" or sp == hold:
                ho.append(x)
    return (np.concatenate(tr) if tr else np.zeros(0)), (np.concatenate(ho) if ho else np.zeros(0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--margin", type=float, default=1.15)
    ap.add_argument("--win", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule", choices=["original", "domain", "pooled"], default="original")
    ap.add_argument("--U", type=float, default=3.0)
    ap.add_argument("--eps", type=float, default=2.5)
    a = ap.parse_args()
    wins = sorted({0.1, 0.2, 0.3, a.win})
    pf, split_of = {}, {}
    for rd in sorted(Path(a.root).iterdir()):
        if not (rd / "eval" / "summary.json").exists():
            continue
        seed = int(json.loads((rd / "resolved_config.json").read_text())["experiment"]["seed"])
        r = flight(rd, wins)
        if r:
            pf[rd.name] = r
            split_of[rd.name] = "train" if seed < 11100 else ("holdout" if seed < 11200 else "holdout2")
    ceil = lambda x: math.ceil(x * a.margin / 0.05 - 1e-9) * 0.05
    res = {"flights": {"train": sum(v == "train" for v in split_of.values()), "holdout": sum(v == "holdout" for v in split_of.values())},
           "window_s_used": a.win, "margin": a.margin, "by_window": {}}
    decl = {}
    for w in wins:
        e = {}
        for key in ("ego_p", "ego_v", "W3", "Wxy", "amag", "az", "ekf_acc_err"):
            tr, ho = agg(pf, split_of, w, key)
            e[key] = {"train_max": float(tr.max()), "train_p999": float(np.quantile(tr, 0.999)), "holdout_max": float(ho.max()),
                      "n_train": int(tr.size), "n_holdout": int(ho.size)}
        res["by_window"][str(w)] = e
    e = res["by_window"][str(a.win)]
    decl = {"ego_err_p": ceil(e["ego_p"]["train_max"]), "ego_err_v": ceil(e["ego_v"]["train_max"]), "W": ceil(e["W3"]["train_max"]),
            "A_honest_total_xy": ceil(e["amag"]["train_max"]), "rule": f"ceil_0.05({a.margin} x train max), window {a.win} s; holdout opened afterwards"}
    res["declared"] = decl
    viol = {}
    for key, bname in (("ego_p", "ego_err_p"), ("ego_v", "ego_err_v"), ("W3", "W"), ("amag", "A_honest_total_xy")):
        tr, ho = agg(pf, split_of, a.win, key)
        viol[bname] = {"declared": decl[bname], "holdout_ticks": int(ho.size), "holdout_exceed": int((ho > decl[bname]).sum()),
                       "holdout_max": float(ho.max()), "train_exceed": int((tr > decl[bname]).sum())}
    res["holdout_violations"] = viol
    if a.rule == "pooled":
        # SECOND REVISION (deviation D1b): the domain rule's A_any still failed the first holdout (13 / 322,091 ticks, max 7.65 m/s^2), so that
        # holdout is consumed into the DEVELOPMENT pool; bounds = ceil_0.05(margin x max over train + holdout1 ticks in the admitted domain),
        # validated ONLY on the fresh holdout2 (11200-11211, flown after this rule was written).
        pool = lambda key, um: np.concatenate(list(agg(pf, split_of, a.win, key, um, "holdout")))   # train + holdout1 only
        d3 = {"ego_err_p": ceil(pool("ego_p", None).max()), "ego_err_v": ceil(pool("ego_v", None).max()), "W": ceil(pool("W3", a.U).max()),
              "A_honest": ceil(pool("amag", a.U).max()), "A_any": ceil(pool("amag", a.U + a.eps).max()),
              "rule": f"ceil_0.05({a.margin} x max over TRAIN+HOLDOUT1 ticks, domain |u|<=U (W, A_honest) / <=U+eps (A_any)); window {a.win} s"}
        res["declared_pooled_rule"] = d3
        tr_, ho2 = None, None
        v3 = {}
        for key, bname, um in (("ego_p", "ego_err_p", None), ("ego_v", "ego_err_v", None), ("W3", "W", a.U), ("amag", "A_honest", a.U), ("amag", "A_any", a.U + a.eps)):
            _, ho = agg(pf, split_of, a.win, key, um, "holdout2")
            v3[bname] = {"declared": d3[bname], "ticks": int(ho.size), "exceed": int((ho > d3[bname]).sum()), "max": float(ho.max()) if ho.size else None}
        res["pooled_rule_holdout2_violations"] = v3
        print(json.dumps({"declared_pooled_rule": d3, "holdout2": v3}, indent=1))
    if a.rule == "domain":
        # REVISED RULE (deviation D1, written after the first holdout failed the original rule): bounds describe the operating domain the filter
        # admits -- honest commands ||u|| <= U, attacker commands ||u|| <= U + eps -- instead of every excitation magnitude HF1 draws (up to 5 m/s^2).
        # Declared from TRAIN ticks only; the first holdout (11100-11111) is consumed (reported, not reused); the fresh holdout is 11200-11211.
        d2 = {"ego_err_p": ceil(agg(pf, split_of, a.win, "ego_p")[0].max()), "ego_err_v": ceil(agg(pf, split_of, a.win, "ego_v")[0].max()),
              "W": ceil(agg(pf, split_of, a.win, "W3", a.U)[0].max()), "A_honest": ceil(agg(pf, split_of, a.win, "amag", a.U)[0].max()),
              "A_any": ceil(agg(pf, split_of, a.win, "amag", a.U + a.eps)[0].max()),
              "rule": f"ceil_0.05({a.margin} x TRAIN max over ticks with |u|<=U (W, A_honest) or <=U+eps (A_any)); window {a.win} s"}
        res["declared_domain_rule"] = d2
        v2 = {}
        for hold in ("holdout", "holdout2"):
            v2[hold] = {}
            for key, bname, um in (("ego_p", "ego_err_p", None), ("ego_v", "ego_err_v", None), ("W3", "W", a.U), ("amag", "A_honest", a.U), ("amag", "A_any", a.U + a.eps)):
                tr, ho = agg(pf, split_of, a.win, key, um, hold)
                v2[hold][bname] = {"declared": d2[bname], "ticks": int(ho.size), "exceed": int((ho > d2[bname]).sum()), "max": float(ho.max()) if ho.size else None}
        res["domain_rule_holdout_violations"] = v2
        print(json.dumps({"declared_domain_rule": d2, "violations": v2}, indent=1))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / {"original": "calibration_bounds.json", "domain": "calibration_bounds_domain_rule.json", "pooled": "calibration_bounds_pooled_rule.json"}[a.rule]).write_text(json.dumps(res, indent=1))
    print(json.dumps({"declared": decl, "holdout_violations": viol}, indent=1))


if __name__ == "__main__":
    main()
