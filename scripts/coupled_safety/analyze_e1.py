#!/usr/bin/env python3
"""E1 primary analysis (frozen in PREREG.md).  Input: episodes.csv from metrics.py (+ meta from the spec).  Scenario = unit; paired.

    python3 scripts/coupled_safety/analyze_e1.py --episodes results/.../episodes.csv --spec experiments/coupled_safety_v1/specs/e1.json --out results/.../e1

Primary family (Holm over 3 two-sided bootstrap tests, 10,000 resamples of COMPLETE scenario blocks, seed 20261025):
  H1  interaction I = L_COUPLED - L_PHYS - L_REPORT + L_A0 under method D24R  (L = 1.5 - min_sep_ha;  I = 0 null)
  H2  COUPLED - PHYS   in min_sep_ha under D24R
  H3  COUPLED - REPORT in min_sep_ha under D24R
Secondary: paired binary risk differences (violation min_sep_ha < 1.5, prox08, crash, safe_task, route_complete) with Newcombe paired intervals; the 4x4 cell table.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED, NBOOT = 20261025, 10000
ATT = ["A0", "PHYS", "REPORT", "COUPLED"]
METH = ["D0R", "D24R", "D26", "D24R_NOSQ"]


def wilson(k, n, z=1.959964):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h




def paired_binary(x1, x2, z=1.959964):
    x1, x2 = np.asarray(x1, int), np.asarray(x2, int)
    n = len(x1)
    A, Bc, C, Dd = (int(np.sum((x1 == 1) & (x2 == 1))), int(np.sum((x1 == 1) & (x2 == 0))), int(np.sum((x1 == 0) & (x2 == 1))), int(np.sum((x1 == 0) & (x2 == 0))))
    p1, p2 = (A + Bc) / n, (A + C) / n
    l1, u1 = wilson(A + Bc, n, z)
    l2, u2 = wilson(A + C, n, z)
    den = (A + Bc) * (C + Dd) * (A + C) * (Bc + Dd)
    phi = 0.0 if den == 0 else (A * Dd - Bc * C) / math.sqrt(den)
    phi = max(0.0, phi)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2))
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2))
    return {"n": n, "both": A, "only_first": Bc, "only_second": C, "neither": Dd, "rate_first": p1, "rate_second": p2, "diff": d, "lo95": lo, "hi95": hi}


def boot(stat, M, rng):
    """M: (n_scn, k) matrix of per-scenario quantities; stat maps column means -> scalar. Returns est, ci95, two-sided p."""
    n = M.shape[0]
    est = stat(M.mean(axis=0))
    idx = rng.integers(0, n, size=(NBOOT, n))
    bs = np.array([stat(M[i].mean(axis=0)) for i in idx])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    p = 2 * min(np.mean(bs <= 0), np.mean(bs >= 0))
    return float(est), (float(lo), float(hi)), float(min(1.0, max(p, 1.0 / NBOOT)))


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj, run = np.zeros(m), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="E1")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    meta = pd.DataFrame([{"name": e["name"], **e["meta"]} for e in spec["episodes"]])
    df = pd.read_csv(a.episodes).merge(meta, on="name", how="inner")
    df = df[(df.set == a.set) & (df.valid == True)]  # noqa: E712
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # inventory: planned vs valid
    planned = meta[meta.set == a.set]
    inv = {"planned": int(len(planned)), "with_outcome": int(len(df)), "missing": sorted(set(planned.name) - set(df.name))[:50]}
    # cell table
    rows = []
    for m in METH:
        for c in ATT:
            g = df[(df.method == m) & (df.cell == c)]
            if not len(g):
                continue
            k = int(g.viol15.sum())
            lo, hi = wilson(k, len(g))
            rows.append({"method": m, "cell": c, "n": len(g), "min_sep_ha_mean": g.min_sep_ha.mean(), "min_sep_ha_median": g.min_sep_ha.median(),
                         "min_sep_ha_min": g.min_sep_ha.min(), "viol15": k, "viol15_rate": k / len(g), "viol15_lo": lo, "viol15_hi": hi,
                         "prox08": int(g.prox08.sum()), "crash": int(g.crash.sum()), "route_complete": int(g.route_complete.sum()),
                         "safe_task": int(g.safe_task.sum()), "min_sep_hh_min": g.min_sep_hh.min(), "lane_frac_mean": g.lane_frac.mean(),
                         "width_mid_min_mean": g.width_mid_min.mean()})
    pd.DataFrame(rows).to_csv(out / "e1_cells.csv", index=False)
    # primary family under D24R
    res = {"inventory": inv}
    g = df[df.method == "D24R"].pivot_table(index="scenario", columns="cell", values="min_sep_ha")
    g = g.dropna()
    res["n_complete_scenarios"] = int(len(g))
    if len(g) >= 5:
        rng = np.random.default_rng(SEED)
        M = g[ATT].to_numpy()                                 # columns A0, PHYS, REPORT, COUPLED (min separation)
        # L = 1.5 - min_sep ->  I = L_C - L_P - L_R + L_0 = -(s_C - s_P - s_R + s_0)
        ests = {}
        ests["H1_interaction_I"] = boot(lambda m: -(m[3] - m[1] - m[2] + m[0]), M, rng)
        ests["H2_COUPLED_minus_PHYS_minsep"] = boot(lambda m: m[3] - m[1], M, rng)
        ests["H3_COUPLED_minus_REPORT_minsep"] = boot(lambda m: m[3] - m[2], M, rng)
        ps = [v[2] for v in ests.values()]
        adj = holm(ps)
        res["primary"] = {k: {"estimate_m": v[0], "ci95": v[1], "p_two_sided_boot": v[2], "p_holm": adj[i], "reject_0_at_0.05_holm": bool(adj[i] < 0.05),
                              "direction": ("positive" if v[0] > 0 else "negative") if k.startswith("H1") else ("COUPLED lower sep" if v[0] < 0 else "COUPLED higher sep")}
                          for i, (k, v) in enumerate(ests.items())}
        res["cell_means_min_sep_D24R"] = {c: float(g[c].mean()) for c in ATT}
        res["bootstrap"] = {"resamples": NBOOT, "seed": SEED, "unit": "scenario (complete paired blocks)", "p_resolution": 1.0 / NBOOT}
    # secondary paired binary risk differences within each method
    sec = []
    for m in METH:
        gm = df[df.method == m]
        for ep in ("viol15", "prox08", "crash", "safe_task", "route_complete"):
            piv = gm.pivot_table(index="scenario", columns="cell", values=ep).dropna()
            for c1, c2 in (("COUPLED", "PHYS"), ("COUPLED", "REPORT"), ("COUPLED", "A0"), ("PHYS", "A0"), ("REPORT", "A0")):
                if c1 in piv and c2 in piv and len(piv) >= 5:
                    sec.append({"method": m, "endpoint": ep, "first": c1, "second": c2, **paired_binary(piv[c1], piv[c2])})
    pd.DataFrame(sec).to_csv(out / "e1_secondary_paired_binary.csv", index=False)
    # method comparisons within COUPLED (secondary performance trade-offs, no inferential claim of superiority)
    mc = []
    for c in ATT:
        gc = df[df.cell == c]
        for m1, m2 in (("D24R", "D0R"), ("D26", "D24R"), ("D24R_NOSQ", "D24R")):
            piv = gc.pivot_table(index="scenario", columns="method", values="viol15").dropna()
            if m1 in piv and m2 in piv and len(piv) >= 5:
                mc.append({"cell": c, "first": m1, "second": m2, **paired_binary(piv[m1], piv[m2])})
    pd.DataFrame(mc).to_csv(out / "e1_method_contrasts_viol15.csv", index=False)
    (out / "e1_primary.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
