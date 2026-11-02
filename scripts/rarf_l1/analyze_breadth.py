#!/usr/bin/env python3
"""B-S1 / B-S2 Layer-1 breadth analysis (experiments/rarf_l1_breadth/PLAN.md §3).  Inside the image:

    python3 scripts/rarf_l1/analyze_breadth.py

Seed-bootstrap 95 % intervals (10,000 resamples, numpy seed 20261106); paired contrasts of every arm against NOM and FF per
(attack, geometry); budget sweep worst case over scenarios per (epsilon, beta_p) and the safe-budget frontier (no scenario < 1.5 m);
Kendall tau of the arm ranking (mean min_sep_ha) across attacks and budgets.  The claim rule of PLAN.md §1 (fidelity gate) is applied
in the report, not here: every row carries the B-F0 status of the pair where one exists.
Writes output/research/rarf_l1/breadth/{spectrum_arms.csv, spectrum_contrasts.csv, budget_cells.csv, frontier.csv, results.json}.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NB, SEED = 10000, 20261106
ARMS = ("NOM", "SPD", "RARF", "FF", "HOCBF", "RHOCBF", "ORCA", "MRCBF")
VARS = ("min_sep_ha", "viol15", "prox08", "crash", "route_complete", "min_sep_hh", "t_cross")


def boot(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [float(np.mean(x)) if len(x) else np.nan, np.nan, np.nan]
    bs = np.random.default_rng(SEED).choice(x, size=(NB, len(x))).mean(1)
    return [float(x.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def kendall(a, b):
    a, b = list(a), list(b)
    s = 0
    for i, j in combinations(range(len(a)), 2):
        s += np.sign((a[i] - a[j]) * (b[i] - b[j]))
    return float(s / (len(a) * (len(a) - 1) / 2))


def load(tag):
    rows = [json.loads(l) for l in (ROOT / "results/rarf_l1" / tag / "episodes.jsonl").read_text().splitlines() if l.strip()]
    E = pd.DataFrame(rows)
    m = pd.json_normalize(E.meta)
    for c in ("method", "attack", "epsilon", "beta_p", "geometry", "seed"):
        E[c] = m[c].to_numpy()
    return E


def gate_status():
    st = {}
    for f in ("fidelity_v3", "fidelity_hocbf"):
        p = ROOT / "output/research/rarf_l1" / f / "results.json"
        if p.exists():
            R = json.loads(p.read_text())
            for pair, d in R.get("contrasts", {}).items():
                st[pair] = d
    return st


def main():
    out = ROOT / "output/research/rarf_l1/breadth"
    out.mkdir(parents=True, exist_ok=True)
    R = {}
    S = load("breadth_spectrum")
    R["spectrum_valid"] = f"{int(S.valid.sum())}/{len(S)}"
    S = S[S.valid == True]  # noqa: E712
    rows = []
    for (att, g, arm), x in S.groupby(["attack", "geometry", "method"]):
        r = dict(attack=att, geometry=g, arm=arm, n=len(x))
        for v in VARS:
            r[v] = float(x[v].mean())
        r["min_sep_ha_ci"] = boot(x.min_sep_ha)[1:]
        r["min_sep_ha_worst"] = float(x.min_sep_ha.min())
        rows.append(r)
    A = pd.DataFrame(rows)
    A.to_csv(out / "spectrum_arms.csv", index=False)
    con = []
    for (att, g), x in S.groupby(["attack", "geometry"]):
        pv = x.pivot_table(index="seed", columns="method", values="min_sep_ha")
        for ref in ("NOM", "FF"):
            for arm in ARMS:
                if arm != ref and arm in pv and ref in pv:
                    ci = boot((pv[arm] - pv[ref]).dropna())
                    con.append(dict(attack=att, geometry=g, contrast=f"{arm}-{ref}", est=ci[0], lo=ci[1], hi=ci[2],
                                    state="pos" if ci[1] > 0 else "neg" if ci[2] < 0 else "unres"))
        for a, b in (("RARF", "RHOCBF"), ("RARF", "MRCBF"), ("RHOCBF", "MRCBF"), ("RARF", "HOCBF"), ("RARF", "ORCA")):
            if a in pv and b in pv:
                ci = boot((pv[a] - pv[b]).dropna())
                con.append(dict(attack=att, geometry=g, contrast=f"{a}-{b}", est=ci[0], lo=ci[1], hi=ci[2],
                                state="pos" if ci[1] > 0 else "neg" if ci[2] < 0 else "unres"))
    C = pd.DataFrame(con)
    C.to_csv(out / "spectrum_contrasts.csv", index=False)
    # ranking stability across attacks (pooled over geometries)
    rk = S.groupby(["attack", "method"]).min_sep_ha.mean().unstack()[list(ARMS)]
    R["spectrum_rank"] = {a: list(rk.loc[a].sort_values(ascending=False).index) for a in rk.index}
    R["spectrum_tau"] = {f"{a}|{b}": kendall(rk.loc[a], rk.loc[b]) for a, b in combinations(rk.index, 2)}
    # budget sweep
    B = load("breadth_budget")
    R["budget_valid"] = f"{int(B.valid.sum())}/{len(B)}"
    B = B[B.valid == True]  # noqa: E712
    bc = B.groupby(["method", "epsilon", "beta_p"]).agg(mean=("min_sep_ha", "mean"), worst=("min_sep_ha", "min"),
                                                       viol15=("viol15", "mean"), prox08=("prox08", "mean"),
                                                       crash=("crash", "mean"), route=("route_complete", "mean")).reset_index()
    bc.to_csv(out / "budget_cells.csv", index=False)
    fr = []
    for arm, x in bc.groupby("method"):
        safe = x[x.worst >= 1.5]
        fr.append(dict(arm=arm, safe_cells=len(safe), of=len(x),
                       safe_set=json.dumps(sorted([[float(e), float(b)] for e, b in zip(safe.epsilon, safe.beta_p)])),
                       inside_declared_safe=int(((x.epsilon <= 2.5) & (x.beta_p <= 1.25) & (x.worst >= 1.5)).sum()),
                       inside_declared=int(((x.epsilon <= 2.5) & (x.beta_p <= 1.25)).sum()),
                       mean_over_budgets=float(x["mean"].mean()), worst_over_budgets=float(x.worst.min())))
    F = pd.DataFrame(fr)
    F.to_csv(out / "frontier.csv", index=False)
    rb = B.groupby(["epsilon", "beta_p", "method"]).min_sep_ha.mean().unstack()[list(ARMS)]
    ref = rb.loc[(2.5, 1.25)]
    R["budget_tau_vs_declared"] = {f"{e:g}|{b:g}": kendall(rb.loc[(e, b)], ref) for e, b in rb.index}
    # degradation slope in beta_p per arm (falsification-awareness prediction)
    sl = {}
    for arm, x in B.groupby("method"):
        m = x.groupby("beta_p").min_sep_ha.mean()
        sl[arm] = float(np.polyfit(m.index.to_numpy(float), m.to_numpy(), 1)[0])
    R["slope_beta_p"] = sl
    sle = {}
    for arm, x in B.groupby("method"):
        m = x.groupby("epsilon").min_sep_ha.mean()
        sle[arm] = float(np.polyfit(m.index.to_numpy(float), m.to_numpy(), 1)[0])
    R["slope_epsilon"] = sle
    R["fidelity_gate"] = gate_status()
    (out / "results.json").write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps({k: v for k, v in R.items() if k != "fidelity_gate"}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
