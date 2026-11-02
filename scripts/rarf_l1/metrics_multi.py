#!/usr/bin/env python3
"""Multi-attacker outcome columns for Layer-1 episodes (B-N1).  The primary row is scripts/coupled_safety/metrics.py unchanged (designated
attacker vs everyone else); this adds the same definitions with the attacker SET A = {designated} + extra attackers and honest set H = rest:
  min_sep_ha_any  min over the encounter window W and over a in A, h in H of |p_a - p_h|;  viol15_any, prox08_any from it
  min_sep_hh_H    min honest-honest distance after mission start, honest = H only
  route_H         honest-centroid goal-plane crossing with honest = H (same rule as metrics.py, lane check omitted)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coupled_safety"))
import metrics  # noqa: E402


def episode(rd: Path, extra):
    row = metrics.episode(rd)
    if not row.get("valid"):
        return row
    cfg = json.loads((rd / "resolved_config.json").read_text())
    R, n = cfg["ral26"], int(cfg["swarm"]["num_uavs"])
    t0 = json.loads((rd / "eval" / "summary.json").read_text())["t_mission_start_sim"]
    T = pd.read_csv(rd / "eval" / "truth.csv")
    t = T.t_sim.to_numpy() - t0
    P = np.stack([T[[f"x{i}", f"y{i}", f"z{i}"]].to_numpy() for i in range(n)], 1)
    A = sorted({int(R["attacker_id"])} | {int(x) for x in extra})
    H = [i for i in range(n) if i not in A]
    onset = float(R.get("attack", {}).get("start_s", 10.0))
    m = (t >= onset) & (t <= onset + metrics.ENC)
    D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=3)
    ha = D[:, A, :][:, :, H].reshape(len(t), -1)
    hh = D[:, H, :][:, :, H]
    iu = np.triu_indices(len(H), 1)
    mission = t >= 0
    mn = float(ha[m].min()) if m.any() else math.nan
    route = np.array(R["route"], float)
    L = float(np.linalg.norm(route))
    cen = P[:, H, :2].mean(1)
    c0 = cen[np.searchsorted(t, 0.0)]
    s = np.maximum.accumulate(np.where(mission, (cen - c0) @ (route / L), -np.inf))
    row.update(attackers="+".join(map(str, A)), n_attackers=len(A), min_sep_ha_any=mn, viol15_any=int(mn < metrics.D_OP),
               prox08_any=int(mn < metrics.D_PROX), min_sep_hh_H=float(hh[mission][:, iu[0], iu[1]].min()) if len(H) > 1 else math.nan,
               route_H=int(bool(((s >= L - 1.0) & mission & (t < 100.0)).any()) and not row["crash"]))
    return row
