#!/usr/bin/env python3
"""rarf_hocbf_v1 roster (PLAN.md).  Episodes come from experiments/rarf_plant_time_v3/design.py (unchanged), with node v3 and two HOCBF arms.

    python3 experiments/rarf_hocbf_v1/design.py dev | confirm      # writes specs/<tag>.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import importlib.util  # noqa: E402
_sp = importlib.util.spec_from_file_location("rarf_plant_time_v3_design", HERE.parent / "rarf_plant_time_v3" / "design.py")
V3 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(V3)                                  # unchanged v3 builder (it imports rarf_ablation_v1/design.py as D)

DEV = {"H1": ("HOCBF", {"d_safe": 2.0, "k1": 1.0, "k2": 1.0}), "H2": ("HOCBF", {"d_safe": 2.5, "k1": 2.0, "k2": 2.0}),
       "H3": ("HOCBF", {"d_safe": 3.0, "k1": 2.0, "k2": 2.0}),
       "R1": ("RHOCBF", {"d_safe": 2.5, "k1": 2.0, "k2": 2.0, "a_bar": 2.5, "robust_soft": 100.0}),
       "R2": ("RHOCBF", {"d_safe": 2.5, "k1": 2.0, "k2": 2.0, "a_bar": 6.2, "robust_soft": 100.0}),
       "R3": ("RHOCBF", {"d_safe": 2.5, "k1": 2.0, "k2": 2.0, "a_bar": 8.8, "robust_soft": 100.0})}
DEV2 = {"O1": ("ORCA", {"radius": 2.0, "tau": 3.0, "responsibility": 0.5}), "O2": ("ORCA", {"radius": 2.5, "tau": 3.0, "responsibility": 1.0}),
        "O3": ("ORCA", {"radius": 3.0, "tau": 5.0, "responsibility": 1.0}),
        "M1": ("MRCBF", {"k1": 1.0, "k2": 1.0}), "M2": ("MRCBF", {"k1": 2.0, "k2": 2.0}), "M3": ("MRCBF", {"k1": 0.5, "k2": 0.5})}
COMMON2 = {"ORCA": {"tau_pref": 1.0, "tau_acc": 0.5, "neighbor_radius": 12.0},
           "MRCBF": {"d": 1.5, "rho": 1.0e3, "eta": 0.0, "neighbor_radius": 12.0}}
COMMON = {"slack_penalty": 1.0e4, "neighbor_radius": 12.0, "k_max": 8, "d_min": 1.5}
DEV_SEEDS = list(range(17400, 17404))
CONF_SEED0 = 17500
RESULTS_ROOT = "results/rarf_hocbf_v1"


def episode(seed, label, variant, geom, stage, rule=None, hocbf=None):
    e = V3.episode(seed, label, variant, geom, stage, rule)
    e["name"] = e["name"].replace("PT", "HC", 1)
    e["overrides"]["controller"]["script"] = "cs_agent_node_v3.py" if stage == "dv" else "cs_agent_node_v4.py"   # A2
    if hocbf is not None and variant in ("ORCA", "MRCBF"):
        e["overrides"]["cs"]["b2"] = {**COMMON2[variant], **hocbf}
    elif hocbf is not None:
        e["overrides"]["cs"]["hocbf"] = {**COMMON, **hocbf}
    e["meta"].update(set=f"HC-{stage}", hocbf=hocbf)
    return e


def ordered(cells, seeds, stage):
    import numpy as np
    eps = []
    for s in seeds:
        cc = [(g, *c) for g in V3.D.GEOMETRIES for c in cells]
        for k in np.random.default_rng(s + 900).permutation(len(cc)):
            g, lab, var, rule, hc = cc[k]
            e = episode(s, lab, var, g, stage, rule, hc)
            e["meta"]["order"] = len(eps)
            eps.append(e)
    return eps


def build(tag):
    if tag == "dev":
        return ordered([(k, v[0], None, v[1]) for k, v in DEV.items()], DEV_SEEDS, "dv")
    if tag == "dev2":
        return ordered([(k, v[0], None, v[1]) for k, v in DEV2.items()], DEV_SEEDS, "d2")
    if tag == "confirm":
        F = json.loads((HERE / "FROZEN.json").read_text())
        cells = [("NOM", "D0R", None, None), ("RARF", "D24R_SH", None, None), ("FF", "D24R_FF", None, None),
                 ("HOCBF", "HOCBF", None, DEV[F["selected"]["HOCBF"]][1]), ("RHOCBF", "RHOCBF", None, DEV[F["selected"]["RHOCBF"]][1]),
                 ("ORCA", "ORCA", None, DEV2[F["selected"]["ORCA"]][1]), ("MRCBF", "MRCBF", None, DEV2[F["selected"]["MRCBF"]][1])]
        return ordered(cells, range(CONF_SEED0, CONF_SEED0 + int(F["N"])), "c")
    raise ValueError(tag)


def write(tag):
    eps = build(tag)
    assert len({e["name"] for e in eps}) == len(eps)
    out = HERE / "specs" / f"{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"tag": tag, "base": "config/swarm/coupled_safety_base.toml", "results_root": RESULTS_ROOT,
                               "rtf_rerun_below": None, "episodes": eps}, indent=1))
    print(tag, len(eps), "->", out)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        write(t)
