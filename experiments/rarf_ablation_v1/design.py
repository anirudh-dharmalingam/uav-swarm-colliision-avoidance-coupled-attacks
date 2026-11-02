#!/usr/bin/env python3
"""rarf_ablation_v1 -- episode roster for the RARF branch/speed ablation (Experiments I and II) and its development stage.

Every episode is built by the coupled_safety_v1 episode builder (experiments/coupled_safety_v1/runs.py, unchanged), so the mission,
world, bounds, network model, attacker policy (fixed D0R, refuses refresh), onset/trigger rules and evaluator are the historical ones.
Only the honest controller variant (and, for D0R_SPD, its speed rule) differs between arms.

Arms (label -> controller variant):
  NOM   D0R       nominal formation control (route tracker + received-neighbour consensus), norm/slew limited, no avoidance layer
  RARF  D24R_SH   complete RARF (historical D24R command law, supervision off) + shadow fallback solve each tick (compute matched to FF)
  FF    D24R_FF   forced best-estimate fallback every tick; the hard solve runs and is logged but never applied
  SPD   D0R_SPD   nominal control + proximity speed restriction; rule chosen on DEV episodes only (FROZEN_PLAN.json "speed_rule")

Geometries: initial attacker slot shifted toward its nearest honest neighbour by 0, 2, 3 m (the E3 construction; 0 m = original geometry).
Scenario seeds are fresh (never flown in coupled_safety_v1): dev 17000-17003, confirmatory 17100-(17100+N-1).

    python3 experiments/rarf_ablation_v1/design.py dev|exp1|exp2      # writes specs/<tag>.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CS = HERE.parent / "coupled_safety_v1"
sys.path.insert(0, str(CS))
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "swarm"))
import runs  # noqa: E402
import scenarios  # noqa: E402
from experiment_manager import generate_lattice  # noqa: E402

B = json.loads((CS / "bounds_declared.json").read_text())["bounds"]
SETTING = "S0"
GEOMETRIES = (0.0, 2.0, 3.0)
ARMS = {"NOM": "D0R", "RARF": "D24R_SH", "FF": "D24R_FF", "SPD": "D0R_SPD"}
# speed-rule candidates, fixed BEFORE any development flight (k = braking gain 1/s; speeds m/s; distances m, planar, received info)
SPD_CANDIDATES = {
    "SPDa": {"d_slow": 4.0, "d_stop": 2.0, "v_slow": 1.0, "v_near": 0.2, "k": 2.0},
    "SPDb": {"d_slow": 5.0, "d_stop": 2.5, "v_slow": 0.8, "v_near": 0.0, "k": 2.0},
    "SPDc": {"d_slow": 6.5, "d_stop": 3.0, "v_slow": 0.6, "v_near": 0.0, "k": 2.0},
}
DEV_SEEDS = range(17000, 17004)
CONF_SEED0 = 17100
RESULTS_ROOT = "results/rarf_ablation_v1"


def plan():
    p = HERE / "FROZEN_PLAN.json"
    return json.loads(p.read_text()) if p.exists() else {}


def scn(seed):
    s = scenarios.scenario("RA", seed, 6)
    scenarios.validate([s])
    return s


def slot(sc, toward):
    off = generate_lattice(sc["N"], runs.SETTINGS[SETTING]["spacing_m"], 1.0)[0]
    a = sc["attacker_id"]
    d, j = min((((off[a][0] - off[k][0]) ** 2 + (off[a][1] - off[k][1]) ** 2) ** 0.5, k) for k in range(sc["N"]) if k != a)
    ux, uy = (off[j][0] - off[a][0]) / d, (off[j][1] - off[a][1]) / d
    return {"id": a, "dx": round(toward * ux, 4), "dy": round(toward * uy, 4), "toward": j, "shift_toward_m": toward}


def ep(sc, cell, label, variant, geom, stage, speed_rule=None):
    extra = {"ral26": {"speed_rule": speed_rule}} if speed_rule else None
    name = f"RA{stage}-{cell}-{label}-g{int(round(geom * 10)):02d}-s{sc['seed']}"
    e = runs.episode(sc, cell, variant, "nominal", SETTING, B, slot_shift=slot(sc, geom), name=name, extra=extra)
    e["meta"].update(set=f"RA-{stage}", method=label, variant=variant, geometry=geom, speed_rule=speed_rule, stage=stage)
    return e


def build(tag):
    P = plan()
    if tag == "dev":
        arms = [("NOM", "D0R", None), ("RARF", "D24R_SH", None), ("FF", "D24R_FF", None)] + \
               [(k, "D0R_SPD", v) for k, v in SPD_CANDIDATES.items()]
        return [ep(scn(s), "COUPLED", lab, var, g, "dev", rule) for s in DEV_SEEDS for g in GEOMETRIES for lab, var, rule in arms]
    n = int(P.get("n_scenarios", 2))
    seeds = range(CONF_SEED0, CONF_SEED0 + n)
    rule = P.get("speed_rule", SPD_CANDIDATES["SPDa"])
    if tag == "exp1":
        arms = [("NOM", "D0R", None), ("RARF", "D24R_SH", None), ("FF", "D24R_FF", None), ("SPD", "D0R_SPD", rule)]
        return [ep(scn(s), "COUPLED", lab, var, g, "c1", r) for s in seeds for g in GEOMETRIES for lab, var, r in arms]
    if tag == "exp2":
        arms = [("NOM", "D0R", None), ("RARF", "D24R_SH", None), ("FF", "D24R_FF", None)]
        return [ep(scn(s), c, lab, var, 0.0, "c2", r) for s in seeds for c in ("A0", "PHYS", "REPORT") for lab, var, r in arms]
    raise ValueError(tag)


def write(tag):
    eps = build(tag)
    names = [e["name"] for e in eps]
    assert len(set(names)) == len(names)
    out = HERE / "specs" / f"{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"tag": tag, "base": "config/swarm/coupled_safety_base.toml", "results_root": RESULTS_ROOT,
                               "rtf_rerun_below": 0.95 if tag == "dev" else None, "episodes": eps}, indent=1))
    print(tag, len(eps), "->", out)


if __name__ == "__main__":
    for t in sys.argv[1:] or ["dev"]:
        write(t)
