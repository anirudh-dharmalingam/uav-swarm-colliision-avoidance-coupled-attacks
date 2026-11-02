#!/usr/bin/env python3
"""B-ISS / B-DLY analysis (experiments/rarf_l1_theory/PLAN.md §3).  Inside the image:

    python3 scripts/rarf_l1/analyze_theory.py

Seed-block bootstrap 10,000 resamples, numpy seed 20261119.  Writes output/research/rarf_l1_theory/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NB, SEED = 10000, 20261119
OUT = ROOT / "output/research/rarf_l1_theory"
ARMS = ("NOM", "SPD", "RARF", "FF", "HOCBF", "RHOCBF", "ORCA", "MRCBF")


def rows(tag):
    E = pd.DataFrame([json.loads(l) for l in (ROOT / "results/rarf_l1" / tag / "episodes.jsonl").read_text().splitlines() if l.strip()])
    m = pd.json_normalize(E.meta)
    for c in ("method", "epsilon", "geometry", "seed", "rho", "threat", "delay"):
        if c in m:
            E[c] = m[c].to_numpy()
    return E


def bboot(df, val):
    g = df.groupby("seed")[val].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    if len(s) < 2:
        return [float(s.sum() / max(c.sum(), 1)), np.nan, np.nan]
    idx = np.random.default_rng(SEED).integers(0, len(s), (NB, len(s)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return [float(s.sum() / c.sum()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def iss(R):
    E = rows("iss")
    R["iss_valid"] = f"{int(E.valid.sum())}/{len(E)}"
    E = E[E.valid == True].copy()  # noqa: E712
    key = ["seed", "geometry"]
    h0 = E[E.epsilon == 0.0].set_index(["method", "rho"] + key).min_sep_ha
    E["loss"] = [max(0.0, h0.get((m, r, s, g), np.nan) - h) for m, r, s, g, h in zip(E.method, E.rho, E.seed, E.geometry, E.min_sep_ha)]
    C = E.groupby(["method", "rho", "epsilon"]).agg(h_min=("min_sep_ha", "mean"), h_worst=("min_sep_ha", "min"), loss_mean=("loss", "mean"),
                                                   gamma_hat=("loss", "max"), route=("route_complete", "mean"), crash=("crash", "mean"),
                                                   viol15=("viol15", "mean")).reset_index()
    C.to_csv(OUT / "iss_cells.csv", index=False)
    i1 = {}
    for (m, r), x in C.groupby(["method", "rho"]):
        g = x.sort_values("epsilon").gamma_hat.to_numpy()
        i1[f"{m}|{r:g}"] = {"gamma_hat": [round(float(v), 3) for v in g], "nondecreasing_tol0.05": bool(np.all(np.diff(g) >= -0.05))}
    R["I1_class_K_shape"] = i1
    i2 = {}
    for m in ("FF", "RARF"):
        for e_ in (2.5, 3.5, 5.0):
            x = C[(C.method == m) & (C.epsilon == e_)].sort_values("rho")
            lm = x.loss_mean.to_numpy()
            i2[f"{m}|eps{e_:g}"] = {"loss_mean_by_rho": [round(float(v), 3) for v in lm], "nonincreasing_tol0.05": bool(np.all(np.diff(lm) <= 0.05)),
                                    "saturation_1e5-1e3": round(float(lm[-1] - lm[2]), 3)}
    R["I2_gain_vs_rho"] = i2
    con = {}
    for m in ("FF", "RARF"):
        for e_ in (2.5, 5.0):
            pv = E[(E.method == m) & (E.epsilon == e_)].pivot_table(index=key, columns="rho", values="min_sep_ha")
            for r in (1e1, 1e2, 1e4, 1e5):
                if r in pv and 1e3 in pv:
                    con[f"{m}|eps{e_:g}|rho{r:g}-rho1e3"] = bboot((pv[r] - pv[1e3]).dropna().rename("d").reset_index(), "d")
    R["iss_contrasts_min_sep_ha"] = con
    # I3 bit identity with B-S2 (rho = 1e3, epsilon > 0)
    B = rows("breadth_budget")
    ref = {n: r for n, r in zip(B.name, B.min_sep_ha)}
    x = E[(E.rho == 1e3) & (E.epsilon > 0)]
    diffs = [abs(h - ref[n.replace("BISS", "BS2").rsplit("-rho", 1)[0]]) for n, h in zip(x.name, x.min_sep_ha)
             if n.replace("BISS", "BS2").rsplit("-rho", 1)[0] in ref]
    R["I3_bit_identity"] = {"matched": len(diffs), "max_abs_diff": float(max(diffs)) if diffs else None}
    return C


def dly(R):
    E = rows("dly")
    R["dly_valid"] = f"{int(E.valid.sum())}/{len(E)}"
    E = E[E.valid == True].copy()  # noqa: E712
    C = E.groupby(["threat", "delay", "method"])[["min_sep_ha", "viol15", "crash", "route_complete", "min_sep_hh"]].mean().reset_index()
    C.to_csv(OUT / "dly_cells.csv", index=False)
    a = C[C.threat == "A14"]
    d1 = {}
    for m, x in a.groupby("method"):
        x = x.set_index("delay")
        d1[m] = {"viol15_rise_at_0.10": round(float(x.viol15.get(0.10) - x.viol15.get(0.0)), 3),
                 "min_sep_ha_drop_0_to_0.40": round(float(x.min_sep_ha.get(0.0) - x.min_sep_ha.get(0.40)), 3)}
    R["D1_within_declared"] = d1
    R["D1_pass"] = bool(all(v["viol15_rise_at_0.10"] <= 0.10 for v in d1.values()))
    d3 = {}
    for dl, x in E[E.threat == "A14"].groupby("delay"):
        pv = x.pivot_table(index=["seed", "geometry"], columns="method", values="min_sep_ha")
        d3[f"{dl:g}"] = bboot((pv.RHOCBF - pv.RARF).dropna().rename("d").reset_index(), "d")
    R["D3_RHOCBF-RARF_by_delay"] = d3
    b = E[(E.threat == "A0")]
    R["D2_benign_hh_contacts_by_delay"] = b.groupby(["delay", "method"]).apply(lambda x: int((x.min_sep_hh < 0.77).sum())).unstack().to_dict()
    return C


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = {}
    C1 = iss(R)
    C2 = dly(R)
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(C1.pivot_table(index=["method", "epsilon"], columns="rho", values="h_min").round(3).to_string())
    print(C2.pivot_table(index=["threat", "delay"], columns="method", values="min_sep_ha").reindex(columns=list(ARMS)).round(2).to_string())
    print(json.dumps({k: v for k, v in R.items() if k != "iss_contrasts_min_sep_ha"}, indent=1, default=str))


if __name__ == "__main__":
    main()
