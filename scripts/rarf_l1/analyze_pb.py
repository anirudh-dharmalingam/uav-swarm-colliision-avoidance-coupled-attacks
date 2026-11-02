#!/usr/bin/env python3
"""B-PB analysis (experiments/rarf_l1_published/PLAN.md §3).  Inside the image:

    python3 scripts/rarf_l1/analyze_pb.py

Combines results/rarf_l1/pb_* with the existing B studies (breadth_spectrum, breadth_budget, adaptive_test, rz, scale).
Seed-block bootstrap 10,000 resamples, numpy seed 20261118; Holm over P1.  Writes output/research/rarf_l1_published/analysis/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NB, SEED = 10000, 20261118
OUT = ROOT / "output/research/rarf_l1_published/analysis"
ARMS = ("NOM", "SPD", "RARF", "FF", "HOCBF", "RHOCBF", "ORCA", "MRCBF", "SBC", "BVC")


def rows(tag):
    p = ROOT / "results/rarf_l1" / tag / "episodes.jsonl"
    if not p.exists():
        return pd.DataFrame()
    E = pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l.strip()])
    m = pd.json_normalize(E.meta)
    for c in m.columns:
        if c in ("method", "attack", "epsilon", "beta_p", "geometry", "seed", "threat", "net", "k", "N"):
            E[c] = m[c].to_numpy()
    return E[E.valid == True].copy()  # noqa: E712


def bboot(df, val):
    g = df.groupby("seed")[val].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    if len(s) < 2:
        return [float(s.sum() / max(c.sum(), 1)), np.nan, np.nan, np.nan]
    idx = np.random.default_rng(SEED).integers(0, len(s), (NB, len(s)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return [float(s.sum() / c.sum()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)),
            float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))]


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def pair(E, a, b, val, key):
    pv = E.pivot_table(index=key, columns="method", values=val)
    if a not in pv or b not in pv:
        return None
    return (pv[a] - pv[b]).dropna().rename("d").reset_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = {"frozen_b3": json.loads((ROOT / "output/research/rarf_l1_published/FROZEN_B3.json").read_text())}
    # spectrum
    S = pd.concat([rows("breadth_spectrum"), rows("pb_spectrum")], ignore_index=True)
    S["route_complete"] = pd.to_numeric(S.route_complete, errors="coerce")
    sc = S.groupby(["attack", "method"])[["min_sep_ha", "route_complete", "crash", "viol15"]].mean().reset_index()
    sc.to_csv(OUT / "spectrum_cells.csv", index=False)
    A14 = S[S.attack == "A14"]
    p1 = {}
    for a, b in (("RHOCBF", "SBC"), ("RHOCBF", "BVC"), ("RARF", "SBC"), ("RARF", "BVC")):
        d = pair(A14, a, b, "min_sep_ha", ["seed", "geometry"])
        p1[f"{a}-{b}"] = bboot(d, "d") if d is not None else None
    adj = holm([v[3] for v in p1.values() if v])
    for (k, v), x in zip([(k, v) for k, v in p1.items() if v], adj):
        v.append(float(x))
    R["P1_A14_min_sep_ha"] = p1
    con = {}
    for att, x in S.groupby("attack"):
        for a in ("RHOCBF", "RARF", "HOCBF", "ORCA"):
            for b in ("SBC", "BVC"):
                d = pair(x, a, b, "min_sep_ha", ["seed", "geometry"])
                if d is not None and len(d):
                    con[f"{att}|{a}-{b}"] = bboot(d, "d")[:3]
    R["spectrum_contrasts"] = con
    # budget frontier
    B = pd.concat([rows("breadth_budget"), rows("pb_budget")], ignore_index=True)
    bc = B.groupby(["method", "epsilon", "beta_p"]).agg(worst=("min_sep_ha", "min"), mean=("min_sep_ha", "mean"),
                                                       route=("route_complete", "mean")).reset_index()
    bc.to_csv(OUT / "budget_cells.csv", index=False)
    R["budget_safe_cells"] = {m: int((x.worst >= 1.5).sum()) for m, x in bc.groupby("method")}
    R["budget_safe_inside_declared"] = {m: int(((x.epsilon <= 2.5) & (x.beta_p <= 1.25) & (x.worst >= 1.5)).sum()) for m, x in bc.groupby("method")}
    # adaptive
    T = pd.concat([rows("adaptive_test"), rows("pb_adaptive_test")], ignore_index=True)
    T["src"] = T.name.str.extract(r"-(A14|th[A-Z]+)$")[0].str.replace("th", "", regex=False)
    own = T[T.src == T.method].groupby("method").agg(min_sep_ha=("min_sep_ha", "mean"), route=("route_complete", "mean"), crash=("crash", "mean"))
    R["adaptive_own"] = own.round(3).to_dict()
    tr = T[T.geometry.isin([2.0, 3.0]) & (T.seed < 19410)]
    M = tr.pivot_table(index="method", columns="src", values="min_sep_ha")
    M.to_csv(OUT / "transfer_matrix.csv")
    R["worst_case_g23"] = {m: {"min_sep_ha": float(M.loc[m].min()), "src": str(M.loc[m].idxmin())} for m in M.index}
    # resilience
    Z = pd.concat([rows("rz"), rows("pb_rz")], ignore_index=True)
    for v in ("sep_deficit_ha", "t_rec_ha", "min_sep_hh"):
        Z[v] = pd.to_numeric(Z[v], errors="coerce")
    Z["rec_finite"] = np.where(Z.threat.str.endswith("_T"), np.isfinite(Z.t_rec_ha).astype(float), np.nan)
    zc = Z.groupby(["threat", "net", "method"])[["sep_deficit_ha", "min_sep_ha", "crash", "rec_finite", "route_complete"]].mean().reset_index()
    zc.to_csv(OUT / "rz_cells.csv", index=False)
    rz = {}
    for th in ("A14", "A14_T", "WORST_T"):
        x = Z[Z.threat == th]
        for a in ("RHOCBF", "RARF"):
            for b in ("SBC", "BVC"):
                d = pair(x, a, b, "sep_deficit_ha", ["seed", "geometry", "net"])
                if d is not None and len(d):
                    rz[f"{th}|{a}-{b}|sep_deficit_ha"] = bboot(d, "d")[:3]
    R["rz_contrasts"] = rz
    # scale
    N = pd.concat([rows("scale"), rows("pb_scale")], ignore_index=True)
    if len(N):
        nc = N.groupby(["N", "k", "method"])[["min_sep_ha_any", "route_H", "crash"]].mean().reset_index()
        nc.to_csv(OUT / "scale_cells.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(sc.pivot_table(index="attack", columns="method", values="min_sep_ha").reindex(columns=[a for a in ARMS if a in sc.method.unique()]).round(2).to_string())
    print(sc.pivot_table(index="attack", columns="method", values="route_complete").reindex(columns=[a for a in ARMS if a in sc.method.unique()]).round(2).to_string())
    print(json.dumps({k: v for k, v in R.items() if k not in ("spectrum_contrasts", "rz_contrasts", "frozen_b3")}, indent=1, default=str))


if __name__ == "__main__":
    main()
