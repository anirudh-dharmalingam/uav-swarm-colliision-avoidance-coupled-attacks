#!/usr/bin/env python3
"""rarf_ablation_v1 figures from saved outcomes only (inside the image: matplotlib).  Vector PDF + SVG, 300-dpi PNG previews.

    python3 scripts/rarf_ablation/figures.py --episodes <episodes.csv> --results <results.json> --budget <certified_budget.csv> --out <dir>

F1 paired endpoints: min_sep_ha per scenario, one panel per geometry, arms on x, grey lines join the same scenario, 1.5 m threshold.
F2 paired contrasts: per-scenario differences (RARF-FF, FF-NOM, FF-SPD) by geometry with the bootstrap mean and interval.
F3 factorial: Exp II cell means per arm (scenario lines) + the per-arm four-cell interaction with interval.
F4 certification budget: minimum estimated clearance for any feasible certificate vs closing speed, per information regime, against the
   reference spacings and the observed encounter minima of the RARF arm.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ARMS = ("NOM", "SPD", "FF", "RARF")
NAMES = {"NOM": "Nominal", "SPD": "Speed limit", "FF": "Forced fallback", "RARF": "Complete RARF"}
COL = {"NOM": "#6b6b6b", "SPD": "#c27c0e", "FF": "#1f77b4", "RARF": "#2ca02c"}
GEOMS = (0.0, 2.0, 3.0)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})


def save(fig, out, name):
    for ext in ("pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def f1(df, out):
    d = df[(df.cell == "COUPLED")]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for k, g in enumerate(GEOMS):
        P = d[d.geometry == g].pivot_table(index="seed", columns="method", values="min_sep_ha").reindex(columns=ARMS).dropna()
        for _, r in P.iterrows():
            ax[k].plot(range(4), r.values, color="0.8", lw=0.6, zorder=1)
        for j, a in enumerate(ARMS):
            ax[k].scatter(np.full(len(P), j), P[a], s=7, color=COL[a], zorder=2)
            ax[k].plot([j - .25, j + .25], [P[a].mean()] * 2, color="k", lw=1.2, zorder=3)
        ax[k].axhline(1.5, color="#d62728", lw=0.8, ls="--")
        ax[k].set_xticks(range(4), [NAMES[a].replace(" ", "\n") for a in ARMS])
        ax[k].set_title(f"{g:g} m inward shift (n = {len(P)} scenarios)")
    ax[0].set_ylabel("min. honest–attacker separation (m)")
    ax[0].text(3.45, 1.52, "1.5 m", color="#d62728", fontsize=7, ha="right", va="bottom")
    save(fig, out, "F1_paired_endpoints")


def f2(df, R, out):
    d = df[df.cell == "COUPLED"]
    pairs = (("RARF", "FF", "Complete RARF − forced fallback"), ("FF", "NOM", "Forced fallback − nominal"),
             ("FF", "SPD", "Forced fallback − speed limit"))
    per = R["exp1_complete_case"]["per_geometry_min_sep_ha"]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    for k, (a, b, title) in enumerate(pairs):
        for j, g in enumerate(GEOMS):
            P = d[d.geometry == g].pivot_table(index="seed", columns="method", values="min_sep_ha").dropna()
            x = P[a] - P[b]
            ax[k].scatter(j + np.linspace(-.18, .18, len(x)), x, s=6, color="0.55", zorder=1)
            e, lo, hi = per[f"g{g:g}|{a}-{b}"]
            ax[k].errorbar(j + .3, e, yerr=[[e - lo], [hi - e]], fmt="o", ms=3.5, color="k", capsize=2, lw=1)
        ax[k].axhline(0, color="k", lw=0.6)
        ax[k].set_xticks(range(3), [f"{g:g} m" for g in GEOMS])
        ax[k].set_title(title)
        ax[k].set_xlabel("inward attacker-slot shift")
    n = R["exp1_complete_case"]["n_blocks"]
    ax[0].set_ylabel(f"paired difference in min. separation (m)\n(dots: scenarios, n = {n}; bar: mean, 95 % CI)")
    save(fig, out, "F2_paired_contrasts")


def f3(df, R, out):
    E = R.get("exp2_complete_case", {})
    if "interaction|RARF" not in E:
        return
    cells = ("A0", "PHYS", "REPORT", "COUPLED")
    lab = {"A0": "benign", "PHYS": "pursuit", "REPORT": "false data", "COUPLED": "coupled"}
    d = df[(df.geometry == 0.0) & df.method.isin(("NOM", "RARF", "FF"))]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [2.2, 1]})
    for j, a in enumerate(("NOM", "FF", "RARF")):
        P = d[d.method == a].pivot_table(index="seed", columns="cell", values="min_sep_ha").reindex(columns=cells).dropna()
        x = np.arange(4) + (j - 1) * 0.22
        for _, r in P.iterrows():
            ax[0].plot(x, r.values, color=COL[a], lw=0.3, alpha=0.35)
        ax[0].plot(x, P.mean().values, "o-", color=COL[a], ms=3.5, lw=1.2, label=f"{NAMES[a]} (n = {len(P)})")
    ax[0].axhline(1.5, color="#d62728", lw=0.8, ls="--")
    ax[0].set_xticks(range(4), [lab[c] for c in cells])
    ax[0].set_ylabel("min. honest–attacker separation (m)")
    ax[0].legend(frameon=False, fontsize=7)
    keys = [("NOM", "interaction|NOM"), ("FF", "interaction|FF"), ("RARF", "interaction|RARF"), ("RARF−FF", "interaction|RARF-FF")]
    for j, (nm, k) in enumerate(keys):
        e, lo, hi = E[k]
        ax[1].errorbar(j, e, yerr=[[e - lo], [hi - e]], fmt="o", ms=3.5, color=COL.get(nm, "k"), capsize=2)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xticks(range(4), ["Nominal", "Forced\nfallback", "Complete\nRARF", "RARF −\nfallback"])
    ax[1].set_ylabel("four-cell harm interaction (m)\nmean, 95 % CI")
    save(fig, out, "F3_factorial")


def f4(budget, df, out):
    B = pd.read_csv(budget)
    B = B[B.packet_age_s == 0.0]
    lab = {"standard": "standard (campaign)", "identity_honest_nb": "known-honest neighbour", "sensor_like": "trusted relative sensor",
           "oracle_info": "perfect information, 0.15 s transit", "oracle_info_no_delay": "perfect information, no delay"}
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for r, g in B.groupby("regime", sort=False):
        ax.plot(g.closing_speed_mps, g.r_min_all_rows_m, "o-", ms=3, lw=1, label=lab.get(r, r))
    for y, t in ((6.0, "lattice 6 m"), (4.8, "squeezed reference 4.8 m")):
        ax.axhline(y, color="0.6", lw=0.6, ls=":")
        ax.text(2.0, y + 0.05, t, fontsize=6.5, ha="right", color="0.4")
    if df is not None:
        m = df[(df.cell == "COUPLED") & (df.method == "RARF")].min_sep_ha.dropna()
        if len(m):
            ax.axhspan(m.min(), m.max(), color="#2ca02c", alpha=0.12, lw=0)
            ax.text(0.02, m.min() + 0.05, f"observed RARF encounter minima (n = {len(m)})", fontsize=6.5, color="#2ca02c")
    ax.set_xlabel("estimated closing speed (m/s)")
    ax.set_ylabel("min. estimated clearance for a\nfeasible certificate (m)")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    save(fig, out, "F4_certification_budget")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes")
    ap.add_argument("--results")
    ap.add_argument("--budget", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = None
    if a.episodes:
        df = pd.read_csv(a.episodes)
        df = df[df.valid.astype(str).str.lower() == "true"]
    if df is not None and a.results:
        R = json.loads(Path(a.results).read_text())
        f1(df, out)
        f2(df, R, out)
        f3(df, R, out)
    f4(a.budget, df, out)
    print("figures ->", out)


if __name__ == "__main__":
    main()
