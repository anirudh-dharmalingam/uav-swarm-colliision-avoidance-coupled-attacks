#!/usr/bin/env python3
"""Resilience-curve columns for Layer-1 episodes (B-RZ, experiments/rarf_l1_resilience/PLAN.md).  The primary row is
scripts/rarf_l1/metrics_multi.py unchanged; this adds time-resolved quantities from eval/truth.csv (true positions only):

  sep_deficit_ha   integral over [onset, onset + ENC] of max(0, D_SAFE - d_ha(t)) dt   [m s]   (resilience-loss area, attacker side)
  sep_deficit_hh   integral over [0, end] of max(0, D_OP - d_hh(t)) dt                  [m s]   (honest-honest loss area)
  t_below_safe     time with d_ha(t) < D_SAFE inside the encounter window              [s]
  t_rec_ha         transient attacks: first r >= 0 after the attack end with d_ha >= D_SAFE held for HOLD s  [s] (inf if never)
  d_ha_at_end      d_ha at the attack end (transient) or at onset + ENC (persistent)
  prog_rate_post   honest-centroid route progress rate over [onset, onset + ENC] / the same over [0, onset]   (absorbed service)
with d_ha(t) = min over attackers a, honest h of |p_a - p_h| and d_hh(t) the honest-honest minimum.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coupled_safety"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402
import metrics_multi  # noqa: E402

D_SAFE = 2.5          # declared safe distance (RAL26 v2: d_safe)
HOLD = 2.0


def episode(rd: Path, extra, t_end=None):
    row = metrics_multi.episode(rd, extra)
    if not row.get("valid"):
        return row
    cfg = json.loads((rd / "resolved_config.json").read_text())
    R, n = cfg["ral26"], int(cfg["swarm"]["num_uavs"])
    t0 = json.loads((rd / "eval" / "summary.json").read_text())["t_mission_start_sim"]
    T = pd.read_csv(rd / "eval" / "truth.csv")
    t = T.t_sim.to_numpy() - t0
    P = np.stack([T[[f"x{i}", f"y{i}", f"z{i}"]].to_numpy() for i in range(n)], 1)
    A = sorted({int(R["attacker_id"])} | {int(x) for x in extra}) if int(R.get("attacker_id", -1)) >= 0 else []
    H = [i for i in range(n) if i not in A]
    onset = float(R.get("attack", {}).get("start_s", 10.0))
    D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=3)
    dt = np.r_[np.diff(t), 0.0]
    mission = t >= 0
    hh = D[:, H, :][:, :, H]
    iu = np.triu_indices(len(H), 1)
    d_hh = hh[:, iu[0], iu[1]].min(1) if len(H) > 1 else np.full(len(t), np.inf)
    row["sep_deficit_hh"] = float((np.clip(metrics.D_OP - d_hh, 0, None) * dt)[mission].sum())
    route = np.array(R["route"], float)
    L = float(np.linalg.norm(route))
    cen = P[:, H, :2].mean(1)
    s = (cen - cen[np.searchsorted(t, 0.0)]) @ (route / L)
    enc = (t >= onset) & (t <= onset + metrics.ENC)
    pre = (t >= 0) & (t < onset)

    def rate(m):
        return float((s[m][-1] - s[m][0]) / max(t[m][-1] - t[m][0], 1e-6)) if m.sum() > 1 else math.nan

    rp, re_ = rate(pre), rate(enc)
    row["prog_rate_post"] = re_ / rp if rp and math.isfinite(rp) and abs(rp) > 1e-6 else math.nan
    if not A:
        row.update(sep_deficit_ha=math.nan, t_below_safe=math.nan, t_rec_ha=math.nan, d_ha_at_end=math.nan)
        return row
    d_ha = D[:, A, :][:, :, H].reshape(len(t), -1).min(1)
    row["sep_deficit_ha"] = float((np.clip(D_SAFE - d_ha, 0, None) * dt)[enc].sum())
    row["t_below_safe"] = float(dt[enc & (d_ha < D_SAFE)].sum())
    te = float(t_end) if t_end is not None else onset + metrics.ENC
    row["d_ha_at_end"] = float(np.interp(te, t, d_ha))
    t_rec = math.nan if t_end is None else math.inf
    if t_end is not None:
        ok = d_ha >= D_SAFE
        idx = np.nonzero(t >= te)[0]
        for k in idx:
            w = (t >= t[k]) & (t <= t[k] + HOLD)
            if t[k] + HOLD > t[-1]:
                break
            if ok[w].all():
                t_rec = float(t[k] - te)
                break
    row["t_rec_ha"] = t_rec
    return row
