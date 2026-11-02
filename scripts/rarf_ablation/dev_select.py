#!/usr/bin/env python3
"""Apply the rules written in experiments/rarf_ablation_v1/DEV_PLAN.md to development outcomes (inside the image: pandas).

    python3 scripts/rarf_ablation/dev_select.py --episodes results/rarf_ablation_v1/analysis/dev_episodes.csv --out experiments/rarf_ablation_v1/FROZEN_PLAN.json

Writes speed_rule (verbatim candidate), n_scenarios and the full calculation; refuses to overwrite an existing FROZEN_PLAN.json.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "rarf_ablation_v1"))
import design  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f"{a.out} exists: the plan is frozen")
    df = pd.read_csv(a.episodes)
    df = df[df.valid.astype(str).str.lower() == "true"]
    cand = {}
    for k in design.SPD_CANDIDATES:
        d = df[df.method == k]
        cand[k] = {"n": int(len(d)), "route_complete": int(d.route_complete.sum()), "crash": int(d.crash.sum()),
                   "mean_min_sep_ha": float(d.min_sep_ha.mean()), "min_min_sep_ha": float(d.min_sep_ha.min()),
                   "viol15": int(d.viol15.sum())}
        cand[k]["eligible"] = cand[k]["n"] == 12 and cand[k]["route_complete"] >= 11 and cand[k]["crash"] == 0
    el = [k for k in cand if cand[k]["eligible"]]
    if el:
        pick, why = max(el, key=lambda k: cand[k]["mean_min_sep_ha"]), "largest mean min_sep_ha among eligible"
    else:
        pick, why = max(cand, key=lambda k: (cand[k]["route_complete"], cand[k]["mean_min_sep_ha"])), "none eligible: most completions, tie by min_sep"
    P = df.pivot_table(index=["seed", "geometry"], columns="method", values="min_sep_ha")
    blk = P.groupby(level=0).mean()                   # per-scenario mean over geometries
    d1 = (blk["RARF"] - blk["FF"]).dropna()
    dn = (blk["RARF"] - blk["NOM"]).dropna()
    s1 = float(d1.std(ddof=1)) if len(d1) > 1 else math.nan
    sn = float(dn.std(ddof=1)) if len(dn) > 1 else math.nan
    s_star = max(x for x in (s1, 0.10, sn) if np.isfinite(x))
    n1 = math.ceil((1.96 * s_star / 0.10) ** 2)
    n2 = math.ceil(((1.96 + 0.84) * s_star / 0.25) ** 2)
    N = min(30, max(12, n1, n2))
    plan = {"frozen_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "speed_rule_label": pick,
            "speed_rule": design.SPD_CANDIDATES[pick], "speed_rule_why": why, "speed_candidates_dev": cand,
            "sample_size": {"dev_D1_per_scenario": d1.round(4).to_dict(), "sd_D1": s1, "sd_RARF_minus_NOM": sn, "s_star": s_star,
                            "n_halfwidth_0.10": n1, "n_power80_delta0.25": n2, "clamp": [12, 30], "N": N,
                            "below_precision_target": max(n1, n2) > 30},
            "n_scenarios": N, "dev_episodes_valid": int(len(df))}
    Path(a.out).write_text(json.dumps(plan, indent=1, default=str))
    print(json.dumps(plan, indent=1, default=str))


if __name__ == "__main__":
    main()
