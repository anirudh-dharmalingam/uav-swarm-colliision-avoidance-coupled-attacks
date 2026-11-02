#!/usr/bin/env python3
"""Pre-generated scenario table (SCENARIOS.jsonl) for the coupled-safety campaign.  All randomness is drawn HERE, per scenario, from an
independent stream keyed by (seed, N): attacker identity, route heading in [-20, 20] deg, attack onset in [8, 12] s, per-UAV initial-position
jitter (uniform in a disk of radius 0.25 m), and a per-scenario FANET key (per-link traces are a keyed function of (key, link, 0.1 s bin), so
every method sees identical environmental link realisations; method-induced load effects are real and are not matched away).

    python3 experiments/coupled_safety_v1/scenarios.py --out experiments/coupled_safety_v1/SCENARIOS.jsonl
Sets: dev 12000-12007 (N=6) | E1/E2 13000-13029 (N=6) | E3 14000-14009 (N=6) | E4 15000-15005 for N in {4,8} | E5 16000-16011 (N=6).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

SETS = {"dev": (range(12000, 12008), (6,)), "E1": (range(13000, 13030), (6,)), "E3": (range(14000, 14010), (6,)),
        "E4": (range(15000, 15006), (4, 8)), "E5": (range(16000, 16012), (6,))}
JITTER, SPACING = 0.25, 6.0


def scenario(setname, seed, n):
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, 7919]))
    att = int(rng.integers(0, n))
    heading = float(rng.uniform(-20.0, 20.0))
    onset = float(rng.uniform(8.0, 12.0))
    jit = {}
    for i in range(n):
        r = JITTER * math.sqrt(rng.random())
        th = rng.uniform(0, 2 * math.pi)
        jit[str(i)] = [round(r * math.cos(th), 4), round(r * math.sin(th), 4)]
    key = int(np.random.SeedSequence([seed, n, 31337]).generate_state(1)[0]) % (2 ** 31)
    return {"id": f"{setname}-N{n}-{seed}", "set": setname, "seed": seed, "N": n, "attacker_id": att, "heading_deg": round(heading, 4),
            "onset_s": round(onset, 4), "jitter": jit, "net_key": key}


def validate(rows):
    """Prevalidated safe geometry: with spacing >= 4 m (smallest declared) and jitter <= 0.25 m the minimum initial separation is >= spacing - 0.5."""
    from itertools import combinations
    sys_path = None
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "swarm"))
    from experiment_manager import generate_lattice
    for r in rows:
        sp = generate_lattice(r["N"], SPACING, 1.0)[1]
        P = [(sp[i][0] + r["jitter"][str(i)][0], sp[i][1] + r["jitter"][str(i)][1]) for i in range(r["N"])]
        d = min(math.dist(a, b) for a, b in combinations(P, 2))
        r["min_initial_separation_m"] = round(d, 3)
        assert d >= SPACING - 2 * JITTER - 1e-6, (r["id"], d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="experiments/coupled_safety_v1/SCENARIOS.jsonl")
    a = ap.parse_args()
    rows = [scenario(name, s, n) for name, (seeds, ns) in SETS.items() for n in ns for s in seeds]
    rows += [scenario("dev", s, n) for s in (12000, 12001) for n in (4, 8)]          # dev smoke scenarios for N != 6 (E4 transfer smoke)
    validate(rows)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(len(rows), "scenarios ->", a.out)
