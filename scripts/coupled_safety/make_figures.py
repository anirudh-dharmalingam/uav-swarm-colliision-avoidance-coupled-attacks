#!/usr/bin/env python3
"""Paper/supplement figures (inside the image: matplotlib + pandas).

    python3 scripts/coupled_safety/make_figures.py --e1 <e1 dir with e1_cells.csv, e1_primary.json> --sec <secondary dir> --e0 results/.../e0 \
        --episodes episodes.csv --ep-dir results/coupled_safety_v1/e1 --video-episode <name> --out output/research/coupled_safety_v1/figures
Fig1 boundary map (restricted-model margin M map + E3 outcomes + E0 slew mismatch)  Fig2 E1 factorial effects with CIs
Fig3 E2 information intervention                                                      Fig4 PX4 trajectories / separation / certification + audited premise timeline
Supplement: runtime latency (figS1), E4 transfer (figS2).  Missing inputs are skipped with a note, never fabricated.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COL = {"D0R": "#777777", "D24R": "#1f77b4", "D26": "#2ca02c", "D24R_NOSQ": "#ff7f0e"}
ATT = ["A0", "PHYS", "REPORT", "COUPLED"]


def fig1(sec, e0, out, U=3.0):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    d = pd.read_csv(Path(sec) / "e3_diagnostic.csv") if (Path(sec) / "e3_diagnostic.csv").exists() else None
    if d is not None and len(d):
        qs = np.linspace(0, 8, 200)
        cs = np.linspace(0, 3, 200)
        Q, C = np.meshgrid(qs, cs)
        A_att = float(d.A_att_p95.median())
        B = U - A_att
        Rq, Rc, tau = float(d.R_q.median()), float(d.R_c.median()), 0.1
        cbar = np.maximum(0, C + Rc)
        M = Q - Rq - cbar * tau - 0.5 * A_att * tau ** 2 - (cbar + A_att * tau) ** 2 / (2 * B)
        ax[0].contourf(Q, C, M, levels=[-50, 0, 50], colors=["#f4c7c3", "#d7ecd9"], alpha=0.8)
        ax[0].contour(Q, C, M, levels=[0], colors="k", linewidths=1.2)
        ax[0].plot(qs, np.sqrt(2 * B * qs), "k--", lw=0.8, label="ideal c²=2Bq (no uncertainty/delay)")
        for m, mk in (("D24R", "o"), ("D0R", "s")):
            g = d[d.method == m]
            ax[0].scatter(g.clearance_onset - 1.5, g.c_hat, c=np.where(g.min_sep_ha < 1.5, "red", "tab:blue"), marker=mk, s=28, edgecolor="k", linewidth=0.4, label=f"E3 {m} (red: <1.5 m)")
        ax[0].set_xlabel("clearance q = distance − 1.5 m at onset [m]")
        ax[0].set_ylabel("projected closing speed ĉ [m/s]")
        ax[0].set_title(f"1-D margin M=0 (B={B:.2f}, R_q={Rq:.1f} m) vs PX4 outcomes\n(diagnostic only; not an X500 threshold)", fontsize=9)
        ax[0].legend(fontsize=7)
    else:
        ax[0].text(0.5, 0.5, "E3 diagnostic not available", ha="center")
    s = pd.read_csv(Path(e0) / "e0_slew_cases.csv")
    g = s.groupby("jerk").ideal_model_mismatch.mean()
    ax[1].bar([str(k) for k in g.index], g.values * 100, color="#9467bd")
    ax[1].set_xlabel("honest jerk limit [m/s³]")
    ax[1].set_ylabel("% of near-boundary cases\nideal M≥0 but slew-limited unsafe")
    ax[1].set_title("E0 extension: ideal 1-D formula is not conservative\nwith a slew-limited actuator (200 cases)", fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(out) / "fig1_boundary_map.png", dpi=200)
    fig.savefig(Path(out) / "fig1_boundary_map.pdf")


def fig2(e1dir, out):
    cells = pd.read_csv(Path(e1dir) / "e1_cells.csv")
    prim = json.loads((Path(e1dir) / "e1_primary.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    w = 0.2
    for k, m in enumerate(["D0R", "D24R", "D26", "D24R_NOSQ"]):
        g = cells[cells.method == m].set_index("cell").reindex(ATT)
        ax[0].bar(np.arange(4) + (k - 1.5) * w, g.min_sep_ha_mean, w, color=COL[m], label=m)
        ax[0].scatter(np.arange(4) + (k - 1.5) * w, g.min_sep_ha_min, c="k", s=8, zorder=3)
    ax[0].axhline(1.5, color="r", ls="--", lw=1)
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels(ATT)
    ax[0].set_ylabel("min honest–attacker separation in window [m]\n(bar: mean, dot: worst scenario)")
    ax[0].legend(fontsize=7)
    ax[0].set_title("E1: 30 paired scenarios per cell", fontsize=9)
    P = prim.get("primary", {})
    names = list(P)
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names):
        v = P[n]
        ax[1].errorbar(v["estimate_m"], yi, xerr=[[v["estimate_m"] - v["ci95"][0]], [v["ci95"][1] - v["estimate_m"]]], fmt="o", color="k", capsize=3)
        ax[1].text(v["ci95"][1] + 0.02, yi, f"p_Holm={v['p_holm']:.3g}", va="center", fontsize=7)
    ax[1].axvline(0, color="gray", lw=0.8)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([n.replace("_", " ") for n in names], fontsize=7)
    ax[1].set_xlabel("effect [m]  (bootstrap 95% CI, scenario blocks)")
    ax[1].set_title("Primary family under D24R (Holm)", fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(out) / "fig2_e1_factorial.png", dpi=200)
    fig.savefig(Path(out) / "fig2_e1_factorial.pdf")


def fig3(sec, out):
    f = Path(sec) / "e2_information.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(d))
    ax[0].bar(x, d.paired_diff_vs_standard.fillna(0), yerr=[(d.paired_diff_vs_standard - d.lo95).fillna(0), (d.hi95 - d.paired_diff_vs_standard).fillna(0)], capsize=3, color="#17becf")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(d["info"])
    ax[0].set_ylabel("Δ min separation vs standard [m]\n(paired, bootstrap 95% CI)")
    ax[0].set_title("E2: information changes feasibility, not (necessarily) outcome", fontsize=9)
    ax[0].axhline(0, color="gray", lw=0.8)
    ax[1].bar(x - 0.2, d.cert_frac, 0.4, label="certified (actual)")
    ax[1].bar(x + 0.2, d.cert_std_frac.fillna(d.cert_frac.iloc[0]), 0.4, label="would be certified with standard info")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(d["info"])
    ax[1].set_ylabel("share of honest ticks in window")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out) / "fig3_e2_information.png", dpi=200)
    fig.savefig(Path(out) / "fig3_e2_information.pdf")


def fig4(epdir, name, ticks_csv, out):
    rd = Path(epdir) / name
    cfg = json.loads((rd / "resolved_config.json").read_text())
    R = cfg["ral26"]
    n, att = int(cfg["swarm"]["num_uavs"]), int(R["attacker_id"])
    onset = float(R["attack"]["start_s"])
    ev = json.loads((rd / "eval" / "summary.json").read_text())
    T = pd.read_csv(rd / "eval" / "truth.csv")
    t = T.t_sim.to_numpy() - ev["t_mission_start_sim"]
    P = np.stack([T[[f"x{i}", f"y{i}"]].to_numpy() for i in range(n)], 1)
    fig = plt.figure(figsize=(11, 6.2))
    a0 = fig.add_subplot(2, 2, 1)
    m = t >= 0
    for i in range(n):
        a0.plot(P[m, i, 0], P[m, i, 1], color="red" if i == att else "tab:blue", lw=1.2 if i == att else 0.8)
        a0.plot(*P[m, i][0], "o", color="red" if i == att else "tab:blue", ms=3)
    a0.set_aspect("equal")
    a0.set_title(f"top-down tracks (red = designated UAV)\n{name}", fontsize=7)
    a1 = fig.add_subplot(2, 2, 2)
    hon = [i for i in range(n) if i != att]
    d_ha = np.min([np.linalg.norm(P[:, att] - P[:, j], axis=1) for j in hon], axis=0)
    a1.plot(t[m], d_ha[m], "r", label="min honest–attacker")
    d_hh = np.min([np.linalg.norm(P[:, a] - P[:, b], axis=1) for a in hon for b in hon if a < b], axis=0)
    a1.plot(t[m], d_hh[m], "tab:blue", label="min honest–honest")
    a1.axhline(1.5, color="k", ls="--", lw=0.8)
    a1.axvline(onset, color="gray", ls=":")
    a1.axvspan(onset, onset + 20, color="orange", alpha=0.08)
    a1.set_ylabel("3-D separation [m]")
    a1.legend(fontsize=7)
    a1.set_xlim(0, 45)
    tk = pd.read_csv(ticks_csv)
    a2 = fig.add_subplot(2, 1, 2)
    tg = np.arange(0, 45, 0.5)
    cert = np.array([np.mean(tk[(tk.t >= x) & (tk.t < x + 0.5)].status == "CERTIFIED_INTERVAL") if ((tk.t >= x) & (tk.t < x + 0.5)).any() else np.nan for x in tg])
    prem = np.array([np.mean(tk[(tk.t >= x) & (tk.t < x + 0.5)].premise_ok) if ((tk.t >= x) & (tk.t < x + 0.5)).any() else np.nan for x in tg])
    a2.step(tg, cert, where="post", label="controller label: CERTIFIED_INTERVAL (share of honest UAVs)", color="tab:green")
    a2.step(tg, prem, where="post", label="independently audited premises hold (share)", color="tab:purple", ls="--")
    a2.axvline(onset, color="gray", ls=":")
    a2.set_ylim(-0.05, 1.05)
    a2.set_xlabel("mission time [s]")
    a2.legend(fontsize=7, loc="center right")
    a2.set_title("Certification timeline: a certified label counts only where the audited premises hold", fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out) / "fig4_px4_timeline.png", dpi=200)
    fig.savefig(Path(out) / "fig4_px4_timeline.pdf")


def figS(sec, out):
    f = Path(sec) / "runtime_table.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[d.set == "E1"]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        x = np.arange(len(d))
        ax.bar(x - 0.2, d.dec_ms_p99, 0.4, label="decision p99 [ms]")
        ax.bar(x + 0.2, d.e2e_ms_p99, 0.4, label="state→command p99 [ms]")
        ax.axhline(50, color="r", ls="--", lw=0.8, label="50 ms loop period")
        ax.set_xticks(x)
        ax.set_xticklabels(d.method)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(Path(out) / "figS1_runtime.png", dpi=200)
    f = Path(sec) / "e4_transfer.csv"
    if f.exists():
        d = pd.read_csv(f)
        fig, ax = plt.subplots(figsize=(8, 3.6))
        d["lab"] = d.apply(lambda r: f"N{int(r.N)}/{r.regime[:3]}/{r.cell}/{r.method.replace('D24R_','')}", axis=1)
        ax.bar(range(len(d)), d.min_sep_ha, color=[COL[m] for m in d.method])
        ax.axhline(1.5, color="r", ls="--", lw=0.8)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d.lab, rotation=75, fontsize=6)
        ax.set_ylabel("mean min separation [m] (n=6/cell)")
        fig.tight_layout()
        fig.savefig(Path(out) / "figS2_e4_transfer.png", dpi=200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1")
    ap.add_argument("--sec")
    ap.add_argument("--e0", default="results/coupled_safety_v1/e0")
    ap.add_argument("--ep-dir")
    ap.add_argument("--video-episode")
    ap.add_argument("--ticks")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.sec:
        fig1(a.sec, a.e0, out)
        fig3(a.sec, out)
        figS(a.sec, out)
    if a.e1:
        fig2(a.e1, out)
    if a.ep_dir and a.video_episode and a.ticks:
        fig4(a.ep_dir, a.video_episode, a.ticks, out)
    print("figures ->", out)
