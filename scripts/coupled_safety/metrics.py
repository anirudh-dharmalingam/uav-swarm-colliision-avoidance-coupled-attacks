#!/usr/bin/env python3
"""Per-episode outcome extraction for the coupled-safety campaign (independent truth only; nothing here is ever fed back to a controller).

    python3 scripts/coupled_safety/metrics.py <results_root>/<tag> [more tag dirs] --out episodes.csv     (inside the image: needs pandas)

Definitions (frozen in PREREG.md):
  mission time  t = t_sim - eval.summary.t_mission_start_sim (GCS start marker); attack onset t_on = ral26.attack.start_s.
  encounter window  W = [t_on, t_on + 20 s] (fixed length, method independent).
  primary  min_sep_ha = min over W and over honest j of the 3-D centre distance |p_att - p_j|   (the designated UAV exists in every cell incl. A0)
  harm  L = 1.5 m - min_sep_ha;  violation = min_sep_ha < 1.5 m (operational separation, not a collision);  prox08 = min < 0.8 m (rotor-tip envelope
        0.39 m x 2); crash = any vehicle tilt > 60 deg after mission start.  Duration below 1.5 m (s) and depth (m) are logged.
  route progress  s(t) = running max of (honest centroid(t) - honest centroid(0)) . route_dir (ORDERED along-route progress); goal-plane crossing
        = first t with s >= |route| - 1.0 m and centroid lateral error <= lane half-width; route_complete = crossing before 100 s, no crash.
  lane  per honest UAV lateral offset from the route centre line vs half-width (outer/2 outside the middle third, mid/2 inside; widths from the
        ACTUAL lattice: span+4, 0.8*span+4); formation width = lateral span of honest UAVs, reported at the middle-third minimum.
  safe_task = route_complete and min_sep_ha >= 1.5 and min_sep_hh >= 1.5 and no crash.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENC = 20.0
D_OP, D_PROX = 1.5, 0.8


def lattice_offsets(n, spacing):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts" / "swarm"))
    from experiment_manager import generate_lattice
    return np.array(generate_lattice(n, spacing, 1.0)[0], float)[:, :2]


def episode(rd: Path):
    cfg = json.loads((rd / "resolved_config.json").read_text())
    R, n = cfg["ral26"], int(cfg["swarm"]["num_uavs"])
    ev = json.loads((rd / "eval" / "summary.json").read_text())
    t0 = ev.get("t_mission_start_sim")
    if t0 is None:
        return {"name": rd.name, "valid": False, "why": "no mission start"}
    T = pd.read_csv(rd / "eval" / "truth.csv")
    t = T.t_sim.to_numpy() - t0
    P = np.stack([T[[f"x{i}", f"y{i}", f"z{i}"]].to_numpy() for i in range(n)], 1)
    tilt = np.stack([T[f"tilt{i}"].to_numpy() for i in range(n)], 1) if "tilt0" in T else np.zeros((len(t), n))
    att = int(R.get("attacker_id", -1))
    hon = [i for i in range(n) if i != att]
    onset = float(R.get("attack", {}).get("start_s", 10.0))
    m = (t >= onset) & (t <= onset + ENC)
    D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=3)
    d_ha = D[:, att, :][:, hon]
    d_hh = np.min([D[:, a, b] for a in hon for b in hon if a < b], axis=0)
    mission = t >= 0.0
    min_ha = float(d_ha[m].min()) if m.any() else math.nan
    sep_ha_t = d_ha.min(axis=1)
    below = (sep_ha_t < D_OP) & m
    dt = np.gradient(t) if len(t) > 1 else np.zeros_like(t)
    dur_below = float((dt * below).sum())
    crash = bool(((tilt.max(axis=1) > 60.0) & mission).any())
    # route / lane
    route = np.array(R["route"], float)
    L = float(np.linalg.norm(route))
    u = route / L
    nrm = np.array([-u[1], u[0]])
    spacing = float(cfg["formation"]["spacing_m"])
    off = lattice_offsets(n, spacing)
    span = float((off @ nrm).max() - (off @ nrm).min())
    w_out, w_mid = span + 4.0, 0.8 * span + 4.0
    cen = P[:, hon, :2].mean(axis=1)
    c0 = cen[np.searchsorted(t, 0.0)] if mission.any() else cen[0]
    s_t = (cen - c0) @ u
    s_run = np.maximum.accumulate(np.where(mission, s_t, -np.inf))
    lat = ((P[:, hon, :2] - c0[None, None, :]) @ nrm)             # lateral offset of each honest UAV (centroid starts at c0)
    s_each = ((P[:, hon, :2] - c0[None, None, :]) @ u)
    mid = (s_t >= L / 3) & (s_t <= 2 * L / 3)
    half = np.where(mid, w_mid / 2, w_out / 2)[:, None]
    # absolute lateral deviation of each UAV from its commanded slot would need the schedule; here: UAV lateral offset about the centroid line
    lat_c = lat - np.mean(lat, axis=1, keepdims=True) * 0.0
    in_lane = (np.abs(lat_c) <= half) | ~mission[:, None]
    lane_frac = float(in_lane[mission].mean()) if mission.any() else math.nan
    width = lat.max(axis=1) - lat.min(axis=1)
    width_mid_min = float(width[mid & mission].min()) if (mid & mission).any() else math.nan
    cross_idx = np.flatnonzero((s_run >= L - 1.0) & mission)
    t_cross = float(t[cross_idx[0]]) if cross_idx.size else math.nan
    route_complete = bool(cross_idx.size and not crash)
    min_hh = float(d_hh[mission].min())
    all_pair = float(D[mission][:, np.triu_indices(n, 1)[0], np.triu_indices(n, 1)[1]].min())
    ag = {}
    for i in range(n):
        f = rd / "agents" / f"agent_{i}_summary.json"
        if f.exists():
            ag[i] = json.loads(f.read_text())
    tr = [s["transit_s"] for s in ag.values() if "transit_s" in s]
    ticks = sum(s.get("ticks", 0) for s in ag.values())
    miss = sum(s.get("deadline_miss", 0) for s in ag.values())
    return {"name": rd.name, "valid": True, "n": n, "attacker": att, "onset": onset, "min_sep_ha": min_ha, "harm": D_OP - min_ha,
            "viol15": int(min_ha < D_OP), "prox08": int(min_ha < D_PROX), "dur_below15_s": dur_below, "depth": max(0.0, D_OP - min_ha),
            "min_sep_hh": min_hh, "min_sep_all": all_pair, "crash": int(crash), "route_complete": int(route_complete), "t_cross": t_cross,
            "s_end": float(s_run[-1]) if np.isfinite(s_run[-1]) else math.nan, "lane_frac": lane_frac, "span_nominal": span,
            "width_mid_min": width_mid_min, "safe_task": int(route_complete and min_ha >= D_OP and min_hh >= D_OP and not crash),
            "rtf_mean": ev.get("rtf_mean"), "transit_p99_max": max([x["p99"] for x in tr], default=math.nan),
            "transit_max": max([x["max"] for x in tr], default=math.nan),
            "transit_frac_over": float(np.mean([x["frac_over_T_net_max"] for x in tr])) if tr else math.nan,
            "ticks": ticks, "deadline_miss": miss, "tx_bytes": sum(s.get("tx_bytes", 0) for s in ag.values()),
            "final_phases": ",".join(sorted({s.get("final_phase", "?") for s in ag.values()})), "t_last": float(t[-1])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for d in a.dirs:
        for rd in sorted(p for p in Path(d).iterdir() if p.is_dir() and not p.name.startswith("_")):
            if (rd / "eval" / "summary.json").exists():
                try:
                    r = episode(rd)
                except Exception as exc:
                    r = {"name": rd.name, "valid": False, "why": repr(exc)}
                r["tag"] = Path(d).name
                rows.append(r)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(len(rows), "episodes ->", a.out)
