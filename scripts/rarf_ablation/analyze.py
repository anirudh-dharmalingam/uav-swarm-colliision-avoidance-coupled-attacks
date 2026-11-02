#!/usr/bin/env python3
"""rarf_ablation_v1 confirmatory analysis (frozen with ANALYSIS_PLAN.md before confirmatory flights).  Inside the image (pandas).

    python3 scripts/rarf_ablation/analyze.py --episodes results/rarf_ablation_v1/analysis/episodes.csv --out results/rarf_ablation_v1/analysis

Unit = scenario (seed).  Each scenario contributes one complete block: 3 geometries x 4 arms (Exp I) and 3 extra cells x 3 arms (Exp II).
Seed reuse across geometries/cells is handled by resampling SCENARIOS (cluster bootstrap): every geometry/cell of a scenario moves together.
Bootstrap: 10,000 resamples, numpy seed 20261029, percentile intervals.  No null-hypothesis p-values are computed.

Primary (Exp I):   D1 = mean over geometries {0,2,3} of [min_sep_ha(RARF) - min_sep_ha(FF)], 95 % CI.
Secondary family (Exp I, Bonferroni 97.5 % CIs over two):  D2 = FF - NOM, D3 = FF - SPD (geometry-averaged min_sep_ha).
Descriptive (95 %): every contrast per geometry; RARF - NOM; 1.5 m violation / 0.8 m proximity / crash / route completion / dwell completion
  risk differences (paired, per geometry and pooled, cluster bootstrap); progress, completion time, effort, sent slew, compute.
Exp II: per arm a in {NOM, RARF, FF}: I_a = L_COUPLED - L_PHYS - L_REPORT + L_A0 with L = 1.5 - min_sep_ha (geometry 0; the COUPLED cell is
  the Exp I g=0 episode of the same scenario); contrast I_RARF - I_FF (95 %) and cellwise benefits arm - NOM per cell.
Missing / non-analysable episodes: complete-block primary analysis; sensitivity with non-analysable min_sep_ha := 0 (all arms);
  timing sensitivity restricted to blocks whose episodes all have rtf_mean >= 0.95.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

NB, SEED = 10000, 20261029
GEOMS = (0.0, 2.0, 3.0)
ARMS = ("NOM", "RARF", "FF", "SPD")


def boot(M, fn, level=0.95):
    """M: (n_scenarios, ...) array; fn maps a resampled M to a scalar.  Returns estimate, lo, hi."""
    rng = np.random.default_rng(SEED)
    n = M.shape[0]
    est = fn(M)
    bs = np.array([fn(M[rng.integers(0, n, n)]) for _ in range(NB)])
    a = (1 - level) / 2
    return float(est), float(np.nanquantile(bs, a)), float(np.nanquantile(bs, 1 - a))


def cube(df, var, cells=("COUPLED",), geoms=GEOMS, arms=ARMS):
    """(scenario, cell, geometry, arm) array of `var`; scenarios with any missing value are dropped (complete blocks)."""
    seeds = sorted(df.seed.dropna().unique())
    X = np.full((len(seeds), len(cells), len(geoms), len(arms)), np.nan)
    for (s, c, g, a), v in df.set_index(["seed", "cell", "geometry", "method"])[var].items():
        if s in seeds and c in cells and g in geoms and a in arms:
            X[seeds.index(s), cells.index(c), geoms.index(g), arms.index(a)] = v
    keep = ~np.isnan(X).reshape(len(seeds), -1).any(axis=1)
    return X[keep], [s for s, k in zip(seeds, keep) if k], len(seeds)


def exp1(df, out, tag):
    X, seeds, n_all = cube(df, "min_sep_ha")
    X = X[:, 0]                                         # (s, g, arm)
    ai = {a: k for k, a in enumerate(ARMS)}
    R = {"n_blocks": len(seeds), "n_scenarios_attempted": n_all, "seeds": seeds}

    def con(a, b, g=None):
        if g is None:
            return lambda M: float(np.mean(M[:, :, ai[a]] - M[:, :, ai[b]]))
        return lambda M: float(np.mean(M[:, g, ai[a]] - M[:, g, ai[b]]))
    R["D1_RARF_minus_FF"] = boot(X, con("RARF", "FF"))
    R["D2_FF_minus_NOM_97.5"] = boot(X, con("FF", "NOM"), 0.975)
    R["D3_FF_minus_SPD_97.5"] = boot(X, con("FF", "SPD"), 0.975)
    R["RARF_minus_NOM"] = boot(X, con("RARF", "NOM"))
    per = {}
    for k, g in enumerate(GEOMS):
        for a, b in (("RARF", "FF"), ("FF", "NOM"), ("FF", "SPD"), ("RARF", "NOM"), ("SPD", "NOM")):
            per[f"g{g:g}|{a}-{b}"] = boot(X, con(a, b, k))
        for a in ARMS:
            per[f"g{g:g}|{a}|mean"] = float(np.mean(X[:, k, ai[a]]))
            per[f"g{g:g}|{a}|min"] = float(np.min(X[:, k, ai[a]]))
    R["per_geometry_min_sep_ha"] = per
    bins = {}
    for var in ("viol15", "prox08", "crash", "route_complete", "dwell_complete"):
        Y, _, _ = cube(df, var)
        Y = Y[:, 0]
        if not len(Y):
            continue
        d = {}
        for k, g in enumerate(GEOMS):
            for a in ARMS:
                d[f"g{g:g}|{a}"] = f"{int(np.nansum(Y[:, k, ai[a]]))}/{Y.shape[0]}"
            for a, b in (("RARF", "FF"), ("FF", "NOM"), ("FF", "SPD")):
                d[f"g{g:g}|{a}-{b}|risk_diff"] = boot(Y, lambda M, k=k, a=a, b=b: float(np.mean(M[:, k, ai[a]] - M[:, k, ai[b]])))
                disc = (int(((Y[:, k, ai[a]] == 1) & (Y[:, k, ai[b]] == 0)).sum()), int(((Y[:, k, ai[a]] == 0) & (Y[:, k, ai[b]] == 1)).sum()))
                d[f"g{g:g}|{a}-{b}|discordant(a_only,b_only)"] = disc
        bins[var] = d
    R["binary"] = bins
    cont = {}
    for var in ("min_sep_hh", "s_end", "t_cross", "effort_rms", "slew_sent_max", "n_slew_sent_402", "dec_ms_p50", "dec_ms_p99",
                "tx_bytes", "truth_gap_max", "rtf_mean"):
        if var not in df:
            continue
        Y, _, _ = cube(df, var)
        if not len(Y):
            cont[var] = "incomplete blocks (e.g. t_cross undefined when the goal is not reached)"
            sub = df[df.cell == "COUPLED"].groupby(["geometry", "method"])[var].agg(["mean", "min", "max", "count"])
            cont[var + "|raw"] = {f"{g:g}|{a}": list(map(float, v)) for (g, a), v in sub.iterrows()}
            continue
        Y = Y[:, 0]
        cont[var] = {f"g{g:g}|{a}": [float(np.mean(Y[:, k, ai[a]])), float(np.min(Y[:, k, ai[a]])), float(np.max(Y[:, k, ai[a]]))]
                     for k, g in enumerate(GEOMS) for a in ARMS}
    R["continuous_mean_min_max"] = cont
    br = {}
    for c in [c for c in df if c.startswith("sh_")]:
        sub = df[df.cell == "COUPLED"].groupby(["geometry", "method"])[[c, "n_win_ticks"]].sum()
        br[c] = {f"g{g:g}|{a}": f"{int(v[c])}/{int(v['n_win_ticks'])}" for (g, a), v in sub.iterrows()}
    R["branch_tick_counts"] = br
    np.save(out / f"exp1_minsep_cube_{tag}.npy", X)
    return R


def exp2(df, out):
    cells = ("A0", "PHYS", "REPORT", "COUPLED")
    arms = ("NOM", "RARF", "FF")
    d = df.copy()
    X, seeds, n_all = cube(d, "min_sep_ha", cells=cells, geoms=(0.0,), arms=arms)
    if not len(X):
        return {"status": "no complete Exp II blocks"}
    X = X[:, :, 0, :]                                   # (s, cell, arm)
    L = 1.5 - X
    R = {"n_blocks": len(seeds), "n_scenarios_attempted": n_all}
    I = lambda M, a: float(np.mean(M[:, 3, a] - M[:, 1, a] - M[:, 2, a] + M[:, 0, a]))  # noqa: E731
    for k, a in enumerate(arms):
        R[f"interaction|{a}"] = boot(L, lambda M, k=k: I(M, k))
    R["interaction|RARF-FF"] = boot(L, lambda M: I(M, 1) - I(M, 2))
    R["interaction|RARF-NOM(excess)"] = boot(L, lambda M: I(M, 1) - I(M, 0))
    R["interaction|FF-NOM(excess)"] = boot(L, lambda M: I(M, 2) - I(M, 0))
    for ci, c in enumerate(cells):
        for k, a in enumerate(arms):
            R[f"cell_mean|{c}|{a}"] = float(np.mean(X[:, ci, k]))
        for k, a in ((1, "RARF"), (2, "FF")):
            R[f"benefit|{c}|{a}-NOM"] = boot(X, lambda M, ci=ci, k=k: float(np.mean(M[:, ci, k] - M[:, ci, 0])))
        R[f"benefit|{c}|RARF-FF"] = boot(X, lambda M, ci=ci: float(np.mean(M[:, ci, 1] - M[:, ci, 2])))
    np.save(out / "exp2_minsep_cube.npy", X)
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.episodes)
    df["valid"] = df.valid.astype(str).str.lower() == "true"
    inv = df[~df.valid]
    res = {"inventory": {"rows": int(len(df)), "valid": int(df.valid.sum()), "invalid": inv[["name", "why"]].to_dict("records")}}
    ok = df[df.valid].copy()
    res["exp1_complete_case"] = exp1(ok, out, "cc")
    worst = df.copy()
    worst.loc[~worst.valid, "min_sep_ha"] = 0.0
    res["exp1_sensitivity_nonanalysable_as_zero"] = {k: v for k, v in exp1(worst[worst.method.notna()], out, "worst").items()
                                                     if k.startswith("D") or k.startswith("n_")}
    slow = set(ok[ok.rtf_mean < 0.95].seed)
    res["exp1_timing_qualified_blocks"] = {k: v for k, v in exp1(ok[~ok.seed.isin(slow)], out, "rtf").items()
                                           if k.startswith("D") or k.startswith("n_")}
    res["exp1_timing_qualified_blocks"]["excluded_seeds_rtf_below_0.95"] = sorted(map(int, slow))
    res["exp2_complete_case"] = exp2(ok, out)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else str(x)))
    print(json.dumps({k: res["exp1_complete_case"].get(k) for k in ("n_blocks", "D1_RARF_minus_FF", "D2_FF_minus_NOM_97.5", "D3_FF_minus_SPD_97.5")}, indent=1))


if __name__ == "__main__":
    main()
