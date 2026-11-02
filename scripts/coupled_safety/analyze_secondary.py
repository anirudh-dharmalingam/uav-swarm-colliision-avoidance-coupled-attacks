#!/usr/bin/env python3
"""Secondary / exploratory analyses E2-E5 and the runtime table (all labelled secondary; no multiplicity claim).

    python3 scripts/coupled_safety/analyze_secondary.py --episodes episodes.csv --agent agent_stats.csv --specs specs/e1.json,specs/e2.json,... --out dir

E2  information intervention: paired (scenario) differences vs the standard-information D24R/COUPLED episode of the same scenario (from E1):
    min_sep_ha, certified share, standard-information counterfactual certified share, info_gain/info_loss, gamma, contradictions.
E3  boundary translation: per method x slot offset: initial clearance, min_sep_ha, violation share, certified share, restricted-model diagnostic M
    (Statement 2 if B_eff = U - A_att,eff > 0 else the finite-window Statement 3) -- a DIAGNOSTIC, not an X500 threshold.
E4  transfer: cells with n = 6 scenarios each (small-n; descriptive).   E5  assumption violations (offline premise audit is separate).
Runtime (Table 8): decision/tick/e2e latency quantiles, deadline misses, bytes, slew activity per method.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED, NBOOT = 20261026, 10000


def boot_mean(x, rng):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(np.mean(x)) if len(x) else math.nan, math.nan, math.nan)
    bs = rng.choice(x, size=(NBOOT, len(x))).mean(axis=1)
    return float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--U", type=float, default=3.0)
    ap.add_argument("--H", type=float, default=20.0)
    a = ap.parse_args()
    meta = pd.DataFrame([{"name": e["name"], **e["meta"]} for s in a.specs.split(",") for e in json.loads(Path(s).read_text())["episodes"]])
    ep = pd.read_csv(a.episodes).merge(pd.read_csv(a.agent).drop(columns=["tag", "ticks", "deadline_miss", "ok"], errors="ignore"), on="name", how="left").merge(meta, on="name", how="inner")
    ep = ep[ep.valid == True]  # noqa: E712
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = {}
    # ---- E2
    std = ep[(ep.set == "E1") & (ep.cell == "COUPLED") & (ep.method == "D24R")].set_index("scenario")
    rows = []
    for info in ("standard", "identity", "oracle", "sensor"):
        g = std if info == "standard" else ep[(ep.set == "E2") & (ep["info"] == info)].set_index("scenario")
        j = g.join(std[["min_sep_ha", "cert_frac"]], rsuffix="_std", how="inner")
        d = (j.min_sep_ha - j.min_sep_ha_std)
        m, lo, hi = boot_mean(d, rng)
        rows.append({"info": info, "n": len(g), "min_sep_ha_mean": g.min_sep_ha.mean(), "paired_diff_vs_standard": m, "lo95": lo, "hi95": hi,
                     "cert_frac": g.cert_frac.mean(), "cert_std_frac": g.get("cert_std_frac", pd.Series(dtype=float)).mean(),
                     "info_gain": g.get("info_gain", pd.Series(dtype=float)).mean(), "info_loss": g.get("info_loss", pd.Series(dtype=float)).mean(),
                     "gamma_mean": g.gamma_mean.mean(), "gamma_std_mean": g.get("gamma_std_mean", pd.Series(dtype=float)).mean(),
                     "viol15": int(g.viol15.sum()), "contradictions": float(g.contradictions.sum()), "fused": float(g.fused.sum())})
    pd.DataFrame(rows).to_csv(out / "e2_information.csv", index=False)
    # ---- E3
    e3 = ep[ep.set == "E3"].copy()
    if len(e3):
        e3["offset"] = e3.slot_shift.apply(lambda s: float(s["shift_toward_m"]) if isinstance(s, dict) else math.nan)
        e3["clearance0"] = 6.0 - e3.offset
        t = e3.groupby(["method", "offset"]).agg(n=("min_sep_ha", "size"), clearance0=("clearance0", "mean"), min_sep_ha=("min_sep_ha", "mean"),
                                                  min_sep_ha_min=("min_sep_ha", "min"), viol15=("viol15", "sum"), cert_frac=("cert_frac", "mean"),
                                                  gamma_mean=("gamma_mean", "mean"), safe_task=("safe_task", "sum"), route_complete=("route_complete", "sum")).reset_index()
        t.to_csv(out / "e3_boundary.csv", index=False)
        e3.to_csv(out / "e3_episodes.csv", index=False)
    # ---- E4 / E5 cells
    e4 = ep[ep.set == "E4"]
    if len(e4):
        e4.groupby(["N", "regime", "cell", "method"]).agg(n=("min_sep_ha", "size"), min_sep_ha=("min_sep_ha", "mean"), min_sep_ha_min=("min_sep_ha", "min"),
                                                          viol15=("viol15", "sum"), crash=("crash", "sum"), safe_task=("safe_task", "sum"), route_complete=("route_complete", "sum"),
                                                          cert_frac=("cert_frac", "mean"), transit_max=("transit_max", "max"), transit_frac_over=("transit_frac_over", "mean"),
                                                          deadline_miss=("deadline_miss", "sum")).reset_index().to_csv(out / "e4_transfer.csv", index=False)
    e5 = ep[ep.set == "E5"]
    if len(e5):
        e5.groupby(["violation"]).agg(n=("min_sep_ha", "size"), min_sep_ha=("min_sep_ha", "mean"), min_sep_ha_min=("min_sep_ha", "min"), viol15=("viol15", "sum"),
                                      crash=("crash", "sum"), cert_frac=("cert_frac", "mean"), invalid_frac=("invalid_frac", "mean")).reset_index().to_csv(out / "e5_violations.csv", index=False)
    # ---- runtime table
    rt = ep.groupby(["set", "method"]).agg(n=("name", "size"), dec_ms_p50=("dec_ms_q50", "mean"), dec_ms_p95=("dec_ms_q95", "mean"), dec_ms_p99=("dec_ms_q99", "mean"),
                                           dec_ms_max=("dec_ms_q100", "max"), tick_ms_p99=("solve_ms_q99", "mean"), tick_ms_max=("solve_ms_q100", "max"),
                                           e2e_ms_p50=("e2e_ms_q50", "mean"), e2e_ms_p99=("e2e_ms_q99", "mean"), e2e_ms_max=("e2e_ms_q100", "max"),
                                           loop_dt_p99=("loop_dt_q99", "mean"), loop_dt_max=("loop_dt_q100", "max"), deadline_miss=("deadline_miss", "sum"),
                                           ticks=("ticks", "sum"), tx_bytes=("tx_bytes", "sum"),
                                           slew_limit_frac=("slew_limit_frac", "mean"), mean_du=("mean_du", "mean")).reset_index()
    rt.to_csv(out / "runtime_table.csv", index=False)
    print("written", [p.name for p in out.glob("*.csv")])


if __name__ == "__main__":
    main()
