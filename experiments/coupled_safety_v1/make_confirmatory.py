#!/usr/bin/env python3
"""Build the locked confirmatory specs e1..e5 (770 episodes) from selection.json (frozen setting, E3 offsets, bounds) and SCENARIOS.jsonl.

    python3 experiments/coupled_safety_v1/make_confirmatory.py          # writes specs/e1.json .. e5.json
Run order (cs_batch.py): e1 first, analyse, then e2..e5.  Episode order is scenario-major so a partial run stays paired.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "swarm"))
import runs  # noqa: E402
from experiment_manager import generate_lattice  # noqa: E402

SEL = json.loads((HERE / "selection.json").read_text())
B = json.loads((HERE / "bounds_declared.json").read_text())["bounds"]
SCN = runs.load_scenarios()
SETTING = SEL["setting"]
ATT = ("A0", "PHYS", "REPORT", "COUPLED")
METH = ("D0R", "D24R", "D26", "D24R_NOSQ")


def write(tag, eps):
    names = [e["name"] for e in eps]
    assert len(set(names)) == len(names), tag
    (HERE / "specs" / f"{tag}.json").write_text(json.dumps({"tag": tag, "base": "config/swarm/coupled_safety_base.toml", "results_root": "results/coupled_safety_v1",
                                                            "rtf_rerun_below": 0.95, "episodes": eps}, indent=1))
    print(tag, len(eps))


def slot(sc, toward):
    off = generate_lattice(sc["N"], runs.SETTINGS[SETTING]["spacing_m"], 1.0)[0]
    a = sc["attacker_id"]
    d, j = min((((off[a][0] - off[k][0]) ** 2 + (off[a][1] - off[k][1]) ** 2) ** 0.5, k) for k in range(sc["N"]) if k != a)
    ux, uy = (off[j][0] - off[a][0]) / d, (off[j][1] - off[a][1]) / d
    return {"id": a, "dx": round(toward * ux, 4), "dy": round(toward * uy, 4), "toward": j, "shift_toward_m": toward}


e1 = [runs.episode(SCN[f"E1-N6-{sd}"], c, m, "nominal", SETTING, B) for sd in range(13000, 13030) for c in ATT for m in METH]
write("e1", e1)
e2 = [runs.episode(SCN[f"E1-N6-{sd}"], "COUPLED", "D24R", "nominal", SETTING, B, info=i, name=f"E2-{SETTING}-COUPLED-D24R-nominal-{i}-N6-s{sd}")
      for sd in range(13000, 13030) for i in ("identity", "oracle", "sensor")]
for e in e2:
    e["meta"]["set"] = "E2"
write("e2", e2)
e3 = []
for sd in range(14000, 14010):
    sc = SCN[f"E3-N6-{sd}"]
    for k, away in enumerate(SEL["e3_offsets_toward_m"]):
        for m in ("D24R", "D0R"):
            e = runs.episode(sc, "COUPLED", m, "nominal", SETTING, B, slot_shift=slot(sc, away), name=f"E3-{SETTING}-{m}-t{int(round(away * 10)):02d}-s{sd}")
            e["meta"]["set"] = "E3"
            e3.append(e)
write("e3", e3)
e4 = []
for sd in range(15000, 15006):
    for n in (4, 8):
        for reg in ("nominal", "adverse"):
            for c in ("A0", "COUPLED"):
                for m in ("D24R", "D24R_NOSQ"):
                    e4.append(runs.episode(SCN[f"E4-N{n}-{sd}"], c, m, reg, SETTING, B))
write("e4", e4)
e5 = []
for sd in range(16000, 16012):
    sc = SCN[f"E5-N6-{sd}"]
    for v, ov in (("report", {"beta_attack": SEL["e5_report_beta_p"], "beta_v_attack": SEL["e5_report_beta_v"]}), ("accel", {"epsilon": SEL["e5_epsilon"]})):
        e5.append(runs.episode(sc, "COUPLED", "D24R", "nominal", SETTING, B, violation=v, attack_override=ov, name=f"E5-{SETTING}-{v}-s{sd}"))
write("e5", e5)
print("total", len(e1) + len(e2) + len(e3) + len(e4) + len(e5))
