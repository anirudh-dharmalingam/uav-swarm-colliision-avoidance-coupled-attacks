#!/usr/bin/env python3
"""Layer-1 breadth campaigns for the coupled-attack avoidance study (PLAN.md here; programme B-S1, B-S2).

Episodes are built with the unchanged rarf_hocbf_v1 builder (itself the unchanged rarf_plant_time_v3 / rarf_ablation_v1 scenario
generator) and its FROZEN comparator gains, plus the frozen SPDb rule, so every arm is configured exactly as in the PX4
confirmations.  Only the attack identity and budgets are varied, and only through the existing config keys.

    python3 experiments/rarf_l1_breadth/design.py spectrum budget      # writes specs/<tag>.json
"""
import copy
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_sp = importlib.util.spec_from_file_location("hc_design", HERE.parent / "rarf_hocbf_v1" / "design.py")
HC = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(HC)
F = json.loads((HERE.parent / "rarf_hocbf_v1" / "FROZEN.json").read_text())

ARMS = [("NOM", "D0R", None, None), ("SPD", "D0R_SPD", HC.V3.SPD, None), ("RARF", "D24R_SH", None, None), ("FF", "D24R_FF", None, None),
        ("HOCBF", "HOCBF", None, HC.DEV[F["selected"]["HOCBF"]][1]), ("RHOCBF", "RHOCBF", None, HC.DEV[F["selected"]["RHOCBF"]][1]),
        ("ORCA", "ORCA", None, HC.DEV2[F["selected"]["ORCA"]][1]), ("MRCBF", "MRCBF", None, HC.DEV2[F["selected"]["MRCBF"]][1])]
# attack spectrum: id -> (label, epsilon, beta_p, beta_v); budgets as in rarf_ablation_v1 unless the attack does not use them
SPECTRUM = {"A0": ("BENIGN", 2.5, 1.25, 0.5), "A8": ("PURSUIT", 2.5, 0.0, 0.0), "A4": ("REPORT", 0.0, 1.25, 0.5),
            "A14": ("COUPLED", 2.5, 1.25, 0.5), "A15": ("FULLPURSUIT", 2.5, 1.25, 0.5), "A5": ("REPLAY", 0.0, 0.0, 0.0)}
SPECTRUM_SEEDS = list(range(19000, 19020))
EPS = (0.5, 1.5, 2.5, 3.5, 5.0)
BETA = (0.0, 0.6, 1.25, 2.0, 3.0)          # beta_v = 0.4 beta_p (the frozen 0.5 / 1.25 ratio)
BUDGET_SEEDS = list(range(19200, 19210))


def ep(seed, arm, geom, attack, eps, bp, bv, tag):
    lab, var, rule, hc = arm
    e = copy.deepcopy(HC.episode(seed, lab, var, geom, "c", rule, hc))   # the builders share nested template dicts
    o = e["overrides"]
    o["attack"]["id"] = attack
    A = o["ral26"]["attack"]
    A["epsilon"], A["beta_attack"], A["beta_v_attack"] = float(eps), float(bp), float(bv)
    e["name"] = f"{tag}-{attack}-e{eps:g}-b{bp:g}-{lab}-g{geom:g}-s{seed}".replace(".", "p")
    e["meta"].update(set=tag, method=lab, attack=attack, epsilon=eps, beta_p=bp, beta_v=bv, geometry=geom, seed=seed)
    return e


def build(tag):
    eps = []
    if tag == "spectrum":
        for s in SPECTRUM_SEEDS:
            for g in (0.0, 2.0, 3.0):
                for att, (_, e_, bp, bv) in SPECTRUM.items():
                    for arm in ARMS:
                        eps.append(ep(s, arm, g, att, e_, bp, bv, "BS1"))
    elif tag == "budget":
        for s in BUDGET_SEEDS:
            for g in (2.0, 3.0):
                for e_ in EPS:
                    for bp in BETA:
                        for arm in ARMS:
                            eps.append(ep(s, arm, g, "A14", e_, bp, round(0.4 * bp, 3), "BS2"))
    else:
        raise ValueError(tag)
    return eps


def write(tag):
    eps = build(tag)
    assert len({e["name"] for e in eps}) == len(eps)
    for e in eps:                                   # every episode carries its own attack, budgets and arm
        o, m = e["overrides"], e["meta"]
        assert o["attack"]["id"] == m["attack"] and o["ral26"]["attack"]["epsilon"] == m["epsilon"], e["name"]
        assert o["ral26"]["attack"]["beta_attack"] == m["beta_p"] and o["experiment"]["seed"] == m["seed"], e["name"]
    out = HERE / "specs" / f"{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"tag": tag, "base": "config/swarm/coupled_safety_base.toml", "results_root": "results/rarf_l1",
                               "rtf_rerun_below": None, "episodes": eps}, indent=1))
    print(tag, len(eps), "->", out)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        write(t)
