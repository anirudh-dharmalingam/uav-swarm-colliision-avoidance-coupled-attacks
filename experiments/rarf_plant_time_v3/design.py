#!/usr/bin/env python3
"""rarf_plant_time_v3 roster (FROZEN_PLAN.md).  Episodes come from the rarf_ablation_v1 builder unchanged (scenario generator, mission,
world physics, bounds, network, attack A14 eps 2.5 / beta 1.25 / 0.5, trigger 9 m, attacker policy D0R refusing refresh, slot shifts,
arms NOM / RARF / FF / SPD with the frozen SPDb rule).  The ONLY changes are execution settings, identical in every arm:
  controller.script  cs_agent_node_v2.py    (versioned node; exec_chain v1 = the historical command chain)
  cs.clock           plant                  (every schedule on Gazebo plant time; deadline 25 ms plant; see the node docstring)
  wind.sdf           yard_physics_rtf030.sdf (yard_physics.sdf with real_time_factor 0.3 / update rate 75; physics step unchanged;
                     amendment A2 — the first pilot ran rtf050 and failed the deadline-miss criterion)
  ral26.wall_timeout_s 700                  (host-time GCS wait, sized for RTF 0.5)

    python3 experiments/rarf_plant_time_v3/design.py smoke|pilot|pilot2|confirm      # writes specs/<tag>.json
"""
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rarf_ablation_v1"))
import design as D  # noqa: E402  (rarf_ablation_v1/design.py, unchanged)

V1 = json.loads((HERE.parent / "rarf_ablation_v1" / "FROZEN_PLAN.json").read_text())
SPD = V1["speed_rule"]                                  # SPDb, frozen in rarf_ablation_v1 from development outcomes only
ARMS = [("NOM", "D0R", None), ("RARF", "D24R_SH", None), ("FF", "D24R_FF", None), ("SPD", "D0R_SPD", SPD)]
EXEC = {"controller": {"script": "cs_agent_node_v2.py"},
        "cs": {"clock": "plant", "exec_chain": "v1", "deadline_s": 0.025, "rtf_nominal": 0.3},
        "wind": {"speed": 0.0, "sdf": "worlds/coupled_safety/yard_physics_rtf030.sdf"},   # amendment A2 (was rtf050)
        "ral26": {"wall_timeout_s": 700.0}}
SMOKE_SEEDS = [17280]
PILOT_SEEDS = list(range(17290, 17294))
PILOT2_SEEDS = list(range(17294, 17298))                # amendment A2: repeated pilot after the instrument gate failed
CONF_SEED0 = 17300
RESULTS_ROOT = "results/rarf_plant_time_v3"


def merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            merge(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a


def episode(seed, label, variant, geom, stage, rule):
    e = D.ep(D.scn(seed), "COUPLED", label, variant, geom, stage, rule)
    e["name"] = e["name"].replace(f"RA{stage}-", f"PT{stage}-")
    merge(e["overrides"], EXEC)
    e["meta"].update(clock="plant", exec_chain="v1", rtf_target=0.3, deadline_s=0.025, set=f"PT-{stage}")
    return e


def order(seeds, stage):
    """Complete blocks, seed by seed.  Within a block the 12 (geometry, arm) episodes are run in a seed-keyed random permutation, in
    adjacent pairs, so arm order is randomised and both worker slots see every arm."""
    import numpy as np
    eps = []
    for s in seeds:
        cells = [(g, lab, var, r) for g in D.GEOMETRIES for lab, var, r in ARMS]
        perm = np.random.default_rng(s + 700).permutation(len(cells))
        for k in perm:
            g, lab, var, r = cells[k]
            e = episode(s, lab, var, g, stage, r)
            e["meta"]["order"] = len(eps)
            eps.append(e)
    return eps


def build(tag):
    if tag == "smoke":
        return [episode(SMOKE_SEEDS[0], lab, var, 3.0, "sm", r) for lab, var, r in ARMS[1:3]]   # RARF + FF, tightest geometry
    if tag == "pilot":
        return order(PILOT_SEEDS, "pi")
    if tag == "pilot2":
        return order(PILOT2_SEEDS, "p2")
    if tag == "confirm":
        P = json.loads((HERE / "FROZEN_PLAN.json").read_text())
        return order(range(CONF_SEED0, CONF_SEED0 + int(P["N"])), "c")
    raise ValueError(tag)


def write(tag):
    eps = build(tag)
    names = [e["name"] for e in eps]
    assert len(set(names)) == len(names)
    out = HERE / "specs" / f"{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"tag": tag, "base": "config/swarm/coupled_safety_base.toml", "results_root": RESULTS_ROOT,
                               "rtf_rerun_below": None, "episodes": eps}, indent=1))
    print(tag, len(eps), "->", out)


if __name__ == "__main__":
    for t in sys.argv[1:] or ["smoke"]:
        write(t)
