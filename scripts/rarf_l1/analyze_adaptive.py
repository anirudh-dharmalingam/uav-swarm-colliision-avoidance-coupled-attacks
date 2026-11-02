#!/usr/bin/env python3
"""B-A1 / B-N1 analysis (experiments/rarf_l1_adaptive/PLAN.md §1-§2).  Inside the image:

    python3 scripts/rarf_l1/analyze_adaptive.py ba1     # adaptive vs A14 per arm, adaptive / worst-case rankings, transfer matrix
    python3 scripts/rarf_l1/analyze_adaptive.py bn1     # scale and coalition size

Seed-block bootstrap (blocks = seeds; geometries inside a block), 10,000 resamples, numpy seed 20261109.
Writes output/research/rarf_l1_adaptive/{ba1,bn1}/.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NB, SEED = 10000, 20261109
ARMS = ("NOM", "SPD", "RARF", "FF", "HOCBF", "RHOCBF", "ORCA", "MRCBF")
OUT = ROOT / "output/research/rarf_l1_adaptive"


def load(tag):
    rows = [json.loads(l) for l in (ROOT / "results/rarf_l1" / tag / "episodes.jsonl").read_text().splitlines() if l.strip()]
    E = pd.DataFrame(rows)
    m = pd.json_normalize(E.meta)
    for c in ("method", "geometry", "seed"):
        E[c] = m[c].to_numpy()
    if "k" in m:
        E["k"] = m["k"].to_numpy()
        E["N"] = m["N"].to_numpy()
    return E


def bboot(df, val):
    """mean of val with a seed-block bootstrap interval (val per row; blocks = seed)."""
    g = df.groupby("seed")[val].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    if len(s) < 2:
        return [float(s.sum() / max(c.sum(), 1)), np.nan, np.nan]
    idx = np.random.default_rng(SEED).integers(0, len(s), (NB, len(s)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return [float(s.sum() / c.sum()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def kendall(a, b):
    s = sum(np.sign((a[i] - a[j]) * (b[i] - b[j])) for i, j in combinations(range(len(a)), 2))
    return float(s / (len(a) * (len(a) - 1) / 2))


def ba1():
    out = OUT / "ba1"
    out.mkdir(parents=True, exist_ok=True)
    E = load("adaptive_test")
    R = {"valid": f"{int(E.valid.sum())}/{len(E)}"}
    E = E[E.valid == True].copy()  # noqa: E712
    E["src"] = E.name.str.extract(r"-(A14|th[A-Z]+)$")[0].str.replace("th", "", regex=False)
    star = json.loads((OUT / "theta_star.json").read_text())
    R["theta_star"] = star
    rows = []
    for arm in ARMS:
        x = E[E.method == arm]
        a14 = x[x.src == "A14"].set_index(["seed", "geometry"])
        own = x[x.src == arm].set_index(["seed", "geometry"])
        r = {"arm": arm}
        for v in ("min_sep_ha", "viol15", "prox08", "crash", "route_complete", "min_sep_hh"):
            r[f"A14|{v}"] = float(a14[v].mean())
            r[f"own|{v}"] = float(own[v].mean())
        d = (own.min_sep_ha - a14.min_sep_ha).dropna().reset_index()
        r["own-A14"] = bboot(d.rename(columns={"min_sep_ha": "d"}), "d")
        r["own-A14_by_geom"] = {f"g{g:g}": float(s.min_sep_ha.mean()) for g, s in d.groupby("geometry")}
        # worst case over every theta* flown against this arm (transfer cells: g2, g3, seeds 19400-19409)
        T = x[x.geometry.isin([2.0, 3.0]) & (x.seed < 19410)]
        per = T.groupby("src").min_sep_ha.mean()
        r["worst_src"] = str(per.idxmin())
        r["worst_mean_g23"] = float(per.min())
        r["A14_mean_g23"] = float(per.get("A14", np.nan))
        rows.append(r)
    A = pd.DataFrame(rows)
    A.to_csv(out / "arms.csv", index=False)
    M = E[E.geometry.isin([2.0, 3.0]) & (E.seed < 19410)].pivot_table(index="method", columns="src", values="min_sep_ha")
    M.loc[list(ARMS), ["A14"] + list(ARMS)].to_csv(out / "transfer_matrix.csv")
    a14 = A.set_index("arm")["A14|min_sep_ha"]
    own = A.set_index("arm")["own|min_sep_ha"]
    wc = A.set_index("arm")["worst_mean_g23"]
    R["rank_A14"] = list(a14.sort_values(ascending=False).index)
    R["rank_adaptive"] = list(own.sort_values(ascending=False).index)
    R["rank_worstcase_g23"] = list(wc.sort_values(ascending=False).index)
    R["tau_A14_adaptive"] = kendall(a14.loc[list(ARMS)].to_numpy(), own.loc[list(ARMS)].to_numpy())
    R["tau_A14_worst"] = kendall(a14.loc[list(ARMS)].to_numpy(), wc.loc[list(ARMS)].to_numpy())
    # paired adaptive contrasts between arms (each arm under its own theta*)
    con = {}
    for a, b in (("RARF", "RHOCBF"), ("RARF", "FF"), ("RARF", "NOM"), ("RHOCBF", "MRCBF"), ("RARF", "MRCBF"), ("RARF", "HOCBF"),
                 ("RARF", "ORCA"), ("RHOCBF", "HOCBF")):
        pa = E[(E.method == a) & (E.src == a)].set_index(["seed", "geometry"]).min_sep_ha
        pb = E[(E.method == b) & (E.src == b)].set_index(["seed", "geometry"]).min_sep_ha
        d = (pa - pb).dropna().rename("d").reset_index()
        con[f"{a}-{b}|own"] = bboot(d, "d")
    R["contrasts_adaptive"] = con
    (out / "results.json").write_text(json.dumps(R, indent=1, default=str))
    print(A[["arm", "A14|min_sep_ha", "own|min_sep_ha", "own-A14", "A14|route_complete", "own|route_complete", "worst_src",
             "worst_mean_g23"]].to_string())
    print(json.dumps({k: v for k, v in R.items() if k != "theta_star"}, indent=1, default=str))


def bn1():
    out = OUT / "bn1"
    out.mkdir(parents=True, exist_ok=True)
    E = load("scale")
    R = {"valid": f"{int(E.valid.sum())}/{len(E)}"}
    E = E[E.valid == True].copy()  # noqa: E712
    V = ["min_sep_ha_any", "viol15_any", "prox08_any", "min_sep_hh_H", "route_H", "crash", "contacts"]
    C = E.groupby(["N", "k", "method"])[V].mean().reset_index()
    ci = [bboot(x, "min_sep_ha_any") for _, x in E.groupby(["N", "k", "method"])]
    C["ha_lo"] = [c[1] for c in ci]
    C["ha_hi"] = [c[2] for c in ci]
    C.to_csv(out / "cells.csv", index=False)
    sl = {}
    for arm, x in C.groupby("method"):
        k1 = x[x.k == 1]
        sl[arm] = {"dN_k1": float(np.polyfit(k1.N, k1.min_sep_ha_any, 1)[0]),
                   "dk_N>=9": float(np.polyfit(x[x.N >= 9].k, x[x.N >= 9].min_sep_ha_any, 1)[0]),
                   "dN_hh": float(np.polyfit(k1.N, k1.min_sep_hh_H, 1)[0])}
    R["slopes"] = sl
    con = {}
    for (n, k), x in E.groupby(["N", "k"]):
        pv = x.pivot_table(index=["seed", "geometry"], columns="method", values="min_sep_ha_any")
        for a, b in (("RARF", "RHOCBF"), ("RARF", "FF"), ("RHOCBF", "MRCBF"), ("RARF", "NOM")):
            d = (pv[a] - pv[b]).dropna().rename("d").reset_index()
            con[f"N{n}|k{k}|{a}-{b}"] = bboot(d, "d")
    R["contrasts"] = con
    (out / "results.json").write_text(json.dumps(R, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(C.pivot_table(index=["N", "k"], columns="method", values="min_sep_ha_any").round(2).to_string())
    print(C.pivot_table(index=["N", "k"], columns="method", values="route_H").round(2).to_string())
    print(json.dumps(R["slopes"], indent=1))


if __name__ == "__main__":
    {"ba1": ba1, "bn1": bn1}[sys.argv[1]]()
