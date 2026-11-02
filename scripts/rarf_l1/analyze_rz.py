#!/usr/bin/env python3
"""B-RZ analysis (experiments/rarf_l1_resilience/PLAN.md §2).  Inside the image:

    python3 scripts/rarf_l1/analyze_rz.py

Seed-block bootstrap 10,000 resamples, numpy seed 20261114.  Q1 is the one confirmatory test; Q2, Q3 are pre-specified criteria.  Writes output/research/rarf_l1_resilience/.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NB, SEED = 10000, 20261114
ARMS = ("NOM", "SPD", "RARF", "FF", "HOCBF", "RHOCBF", "ORCA", "MRCBF")
NETS = ("nominal", "degraded", "severe", "blackout")
V = ["min_sep_ha", "sep_deficit_ha", "t_below_safe", "t_rec_ha", "crash", "route_complete", "min_sep_hh", "sep_deficit_hh", "prog_rate_post"]
OUT = ROOT / "output/research/rarf_l1_resilience"


def bboot(df, val):
    g = df.groupby("seed")[val].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    if len(s) < 2:
        return [float(s.sum() / max(c.sum(), 1)), np.nan, np.nan, np.nan]
    idx = np.random.default_rng(SEED).integers(0, len(s), (NB, len(s)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return [float(s.sum() / c.sum()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)),
            float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))]


def kendall(a, b):
    s = sum(np.sign((a[i] - a[j]) * (b[i] - b[j])) for i, j in combinations(range(len(a)), 2))
    return float(s / (len(a) * (len(a) - 1) / 2))


def binom_lb(k, n, a=0.05):
    """one-sided Clopper-Pearson lower bound."""
    from scipy.stats import beta
    return float(beta.ppf(a, k, n - k + 1)) if k > 0 else 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (ROOT / "results/rarf_l1/rz/episodes.jsonl").read_text().splitlines() if l.strip()]
    E = pd.DataFrame(rows)
    m = pd.json_normalize(E.meta)
    for c in ("method", "threat", "net", "geometry", "seed"):
        E[c] = m[c].to_numpy()
    R = {"valid": f"{int(E.valid.sum())}/{len(E)}"}
    R["invalid"] = E[E.valid != True].name.tolist()  # noqa: E712
    E = E[E.valid == True].copy()  # noqa: E712
    for v in V:
        E[v] = pd.to_numeric(E[v], errors="coerce")
    E["rec_finite"] = np.where(E.threat.str.endswith("_T"), np.isfinite(E.t_rec_ha).astype(float), np.nan)
    C = E.groupby(["threat", "net", "method"])[V + ["rec_finite"]].agg(lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else np.nan).reset_index()
    C["n"] = E.groupby(["threat", "net", "method"]).size().to_numpy()
    C.to_csv(OUT / "cells.csv", index=False)
    key = ["seed", "geometry", "net"]
    # Q1
    a = E[(E.threat == "A14") & (E.method == "RHOCBF")].set_index(key).sep_deficit_ha
    b = E[(E.threat == "A14") & (E.method == "RARF")].set_index(key).sep_deficit_ha
    q1 = bboot((a - b).dropna().rename("d").reset_index(), "d")
    # Q2
    q2 = {}
    for nt in NETS:
        x = E[(E.threat == "A14_T") & (E.method == "RHOCBF") & (E.net == nt)]
        k, n = int(np.isfinite(x.t_rec_ha).sum()), len(x)
        q2[nt] = {"finite": k, "n": n, "share": k / max(n, 1), "lb95": binom_lb(k, n)}
    q2_pass = all(v["share"] >= 0.95 for v in q2.values())
    # Q3
    x = E[(E.threat == "A0") & (E.net == "degraded")]
    q3_hh = x[x.min_sep_hh < 0.77][["name", "method", "seed", "geometry", "min_sep_hh"]].to_dict("records")
    contacts_all = E[E.crash == 1][["name", "method", "threat", "net", "seed", "geometry", "min_sep_ha", "min_sep_hh"]].to_dict("records")
    R["Q1_RHOCBF-RARF_deficit_A14"] = q1
    R["Q2_RHOCBF_recovery_A14_T"] = q2
    R["Q2_pass"] = q2_pass
    R["Q3_hh_contacts_A0_degraded"] = q3_hh
    R["Q3_pass"] = not q3_hh
    R["contacts_listed"] = contacts_all
    # Q4
    w = E[E.threat == "WORST"].groupby("method").crash.mean()
    R["Q4_WORST_contact_share"] = w.to_dict()
    R["Q4_pass"] = bool(all(w[a_] >= 0.5 for a_ in ARMS if a_ != "MRCBF" and a_ in w))
    # Q5
    wt = E[E.threat == "WORST_T"]
    rec = wt.groupby("method").rec_finite.mean().reindex(list(ARMS))
    ab = wt.groupby("method").min_sep_ha.mean().reindex(list(ARMS))
    R["Q5_recovery_share_WORST_T"] = rec.to_dict()
    R["Q5_tau_absorb_vs_recover"] = kendall(ab.to_numpy(), rec.to_numpy())
    # Q6
    q6 = {}
    for arm in ("RHOCBF", "RARF"):
        s = [float(E[(E.threat == "A14") & (E.method == arm) & (E.net == nt)].sep_deficit_ha.mean()) for nt in ("nominal", "degraded", "severe")]
        q6[arm] = {"deficit": s, "monotone": bool(s[0] <= s[1] <= s[2]), "max_step": float(max(s[1] - s[0], s[2] - s[1])),
                   "graceful": bool(max(s[1] - s[0], s[2] - s[1]) <= s[0] + 2.0)}
    R["Q6"] = q6
    # every pair of interest, every (threat, net) cell
    con = {}
    for (th, nt), x in E.groupby(["threat", "net"]):
        for val in ("sep_deficit_ha", "min_sep_ha"):
            pv = x.pivot_table(index=["seed", "geometry"], columns="method", values=val)
            for a_, b_ in (("RHOCBF", "RARF"), ("RARF", "FF"), ("RARF", "NOM"), ("RHOCBF", "MRCBF"), ("RARF", "HOCBF"), ("RARF", "ORCA")):
                if a_ in pv and b_ in pv:
                    con[f"{th}|{nt}|{a_}-{b_}|{val}"] = bboot((pv[a_] - pv[b_]).dropna().rename("d").reset_index(), "d")[:3]
    R["contrasts"] = con
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=str))
    pd.set_option("display.width", 250)
    for v in ("sep_deficit_ha", "min_sep_ha", "crash", "rec_finite", "route_complete"):
        print(v)
        print(C.pivot_table(index=["threat", "net"], columns="method", values=v).reindex(columns=list(ARMS)).round(2).to_string())
    print(json.dumps({k: v for k, v in R.items() if k not in ("contrasts", "contacts_listed")}, indent=1, default=str))


if __name__ == "__main__":
    main()
