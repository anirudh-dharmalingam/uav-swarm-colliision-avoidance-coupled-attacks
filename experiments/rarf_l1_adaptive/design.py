#!/usr/bin/env python3
"""B-A1 / B-N1 Layer-1 episode builders (PLAN.md here).  Episodes come from experiments/rarf_l1_breadth/design.py `ep` (itself the
unchanged rarf_hocbf_v1 / rarf_plant_time_v3 / rarf_ablation_v1 builders with the FROZEN comparator gains); the only additions are the
harness options in "l1x" (scripts/rarf_l1/run_l1x.py) and, for B-N1, the swarm size N through the unchanged scenario generator.

    python3 experiments/rarf_l1_adaptive/design.py scale            # writes specs/scale.json (B-N1)
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_sp = importlib.util.spec_from_file_location("breadth_design", HERE.parent / "rarf_l1_breadth" / "design.py")
BD = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(BD)
D = BD.HC.V3.D                                   # rarf_ablation_v1/design.py (scenario generator wrapper, slot shifts)
ARMS = BD.ARMS
ARM = {a[0]: a for a in ARMS}
EPS, BP, BV = 2.5, 1.25, 0.5                      # declared coupled-attack budgets (rarf_ablation_v1 / v3 / hocbf_v1)
SCALE_N = {4: (1, 2), 6: (1, 2), 9: (1, 2, 3), 12: (1, 2, 3), 16: (1, 2, 3)}
SCALE_SEEDS = list(range(19600, 19610))
SCALE_GEOMS = (0.0, 3.0)


def ep(seed, arm, geom, tag, theta=None, n=None, extra=None, suffix=""):
    """One episode; n != None rebuilds the scenario at swarm size n (scenarios.scenario('RA', seed, n), validated)."""
    orig = D.scn
    if n is not None:
        def scn(s, n=n):
            sc = D.scenarios.scenario("RA", s, n)
            D.scenarios.validate([sc])
            return sc
        D.scn = scn
    try:
        e = BD.ep(seed, ARM[arm] if isinstance(arm, str) else arm, geom, "A14", EPS, BP, BV, tag)
    finally:
        D.scn = orig
    e["name"] += suffix
    e["l1x"] = {"theta": theta, "extra_attackers": list(extra or [])}
    e["meta"].update(theta=theta, extra_attackers=list(extra or []), N=e["overrides"]["swarm"]["num_uavs"])
    return e


def extra_ids(seed, n, primary, k):
    """k - 1 co-attackers drawn uniformly from the other identities (fixed per scenario, identical for every arm)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, k, 4243]))
    pool = [i for i in range(n) if i != primary]
    return sorted(int(x) for x in rng.choice(pool, size=k - 1, replace=False)) if k > 1 else []


def build_scale():
    eps = []
    for n, ks in SCALE_N.items():
        for s in SCALE_SEEDS:
            prim = D.scenarios.scenario("RA", s, n)["attacker_id"]
            for k in ks:
                ex = extra_ids(s, n, prim, k)
                for g in SCALE_GEOMS:
                    for a in ARMS:
                        e = ep(s, a[0], g, "BN1", n=n, extra=ex, suffix=f"-N{n}-k{k}")
                        e["meta"].update(k=k)
                        eps.append(e)
    return eps


def write(tag):
    eps = {"scale": build_scale}[tag]()
    assert len({e["name"] for e in eps}) == len(eps)
    for e in eps:
        assert e["overrides"]["swarm"]["num_uavs"] == e["meta"]["N"]
    out = HERE / "specs" / f"{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"tag": tag, "base": "config/swarm/coupled_safety_base.toml", "results_root": "results/rarf_l1",
                               "episodes": eps}, indent=1))
    print(tag, len(eps), "->", out)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        write(t)
