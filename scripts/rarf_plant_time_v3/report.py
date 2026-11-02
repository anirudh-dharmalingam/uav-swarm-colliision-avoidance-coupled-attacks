#!/usr/bin/env python3
"""Manuscript-ready tables for rarf_plant_time_v3 from the frozen analysis output (no new computation of contrasts).

    python3 scripts/rarf_plant_time_v3/report.py output/research/rarf_plant_time_v3/confirm > output/research/rarf_plant_time_v3/confirm/RESULTS_TABLES.md
"""
import json
import sys
from pathlib import Path

import pandas as pd


def f(x, d=3):
    return "–" if x is None or x != x else f"{x:+.{d}f}" if isinstance(x, float) else str(x)


def ci(v, d=3):
    return f"{v[0]:+.{d}f} [{v[1]:+.{d}f}, {v[2]:+.{d}f}]"


def main():
    d = Path(sys.argv[1])
    R = json.loads((d / "results.json").read_text())
    E = pd.read_csv(d / "episodes.csv")
    P = R["primary_and_secondary"]
    print(f"## Plant-time confirmation: {P['n_blocks']} complete scenario blocks of {P['n_scenarios']} (seeds {P['seeds'][0]}–{P['seeds'][-1]})\n")
    print("| Contrast (geometry-averaged min honest–designated separation, m) | Role | Level | Estimate [CI] |\n|---|---|---|---|")
    for k in ("D2_FF-NOM", "D3_FF-SPD", "D1_RARF-FF"):
        c = P[k]
        print(f"| {k.replace('_', ' ')} | {c['role']} | {c['level']:.3g} | {ci(c['est_lo_hi'])} |")
    print(f"| RARF − NOM | descriptive | 0.95 | {ci(P['RARF-NOM'])} |")
    print("\nPer-scenario D2, D3, D1:", {k: P[k]["per_scenario"] for k in ("D2_FF-NOM", "D3_FF-SPD", "D1_RARF-FF")}, "\n")
    print("| Shift | FF − NOM | FF − SPD | RARF − FF | RARF − NOM |\n|---|---|---|---|---|")
    for g in ("0", "2", "3"):
        print(f"| {g} m | {ci(P[f'g{g}|FF-NOM'])} | {ci(P[f'g{g}|FF-SPD'])} | {ci(P[f'g{g}|RARF-FF'])} | {ci(P[f'g{g}|RARF-NOM'])} |")
    B = R["binary"]
    print("\n| Shift | Arm | Mean / min sep [m] | < 1.5 m | < 0.8 m | tilt crash | route | dwell |\n|---|---|---|---|---|---|---|---|")
    for g in ("0", "2", "3"):
        for a in ("NOM", "RARF", "FF", "SPD"):
            mm = P[f"g{g}|{a}|mean_min"]
            print(f"| {g} m | {a} | {mm[0]:.3f} / {mm[1]:.3f} | {B['viol15'][f'g{g}|{a}']} | {B['prox08'][f'g{g}|{a}']} | "
                  f"{B['crash'][f'g{g}|{a}']} | {B['route_complete'][f'g{g}|{a}']} | {B['dwell_complete'][f'g{g}|{a}']} |")
    print("\nRisk differences (95 %), FF − NOM / FF − SPD / RARF − FF, 1.5 m threshold events:")
    for g in ("0", "2", "3"):
        print(f"* {g} m:", "; ".join(f"{p}: {ci(B['viol15'][f'g{g}|{p}|risk_diff'], 2)} (discordant {B['viol15'][f'g{g}|{p}|discordant']})"
                                     for p in ("FF-NOM", "FF-SPD", "RARF-FF")))
    print("\n### Execution (instrument) by arm: min / median / max over episodes")
    I = R["instrument_by_arm"]
    keys = [k for k in ("rtf_mean_eval", "miss_frac", "skip_frac", "dec_per_agent_plant_s", "ref_speed", "onset_err", "pub_slack_p99", "pub_slack_max",
                        "hold_p99", "hold_max", "clock_offset_median", "clock_jitter_p99") if k in I["NOM"]]
    print("| Arm | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 1))
    for a in ("NOM", "RARF", "FF", "SPD"):
        print(f"| {a} | " + " | ".join("/".join(f"{x:.4g}" for x in I[a][k]) for k in keys) + " |")
    print(f"\nInstrument pass by arm: {R['instrument_pass']}; failed criteria counts: {R['instrument_fail_criteria']}")
    fb = R["flagged_block_sensitivity"]
    if fb.get("n_blocks"):
        print(f"\nFlagged-block sensitivity ({fb['n_blocks']} blocks with every episode passing): "
              + "; ".join(f"{k}: {ci(fb[k]['est_lo_hi'])}" for k in ("D2_FF-NOM", "D3_FF-SPD", "D1_RARF-FF")))
    else:
        print("\nFlagged-block sensitivity: no block has every episode passing the instrument criteria.")
    z = R["nonanalysable_as_zero"]
    print("Non-analysable := 0 m sensitivity:", {k: [round(x, 3) for x in v["est_lo_hi"]] for k, v in z.items() if k.startswith("D")})
    print("\n### Branch use and labels (honest decisions, agent-clock window)")
    for k, v in R["branch_counts"].items():
        print(f"* {k}: {v}")
    print("* final labels:", R["labels"])
    print("\n### Inventory\n", json.dumps(R["inventory"]))


if __name__ == "__main__":
    main()
