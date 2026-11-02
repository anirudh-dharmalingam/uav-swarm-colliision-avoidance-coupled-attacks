"""Episode builders shared by the development and confirmatory specs (the single place where a scenario row + condition become an episode).

An episode = scenario row (SCENARIOS.jsonl) x attack cell x method x network regime x information mode x setting.  Methods are descriptive
labels mapped to implementation flags (defense id + envelope mode); every method uses envelope="sound_age", the same hard rows, solver,
actuator ball/slew, fallback and bounds object:

  D0R         no safety filter (nominal control, actuator ball/slew clip only)                      defense id D0R
  D24R        interval filter, fixed (commanded) geometry, matched fallback                         defense id D24R
  D26         same rows/solver/fallback as D24R; supervisor (mission relaxation/link refresh/energy) defense id D26
  D24R_NOSQ   interval filter, honest UAVs keep full spacing (declines the squeeze)                 defense id D24R_NOSQ

Attack cells (labels, not the old A-numbers): A0 truthful nominal | PHYS = A8 (physical deviation, truthful report) | REPORT = A4 (report
bias only) | COUPLED = A14 (PHYS component + REPORT component, identical parameters).  In every cell the designated UAV runs the fixed independent
policy D0R and refuses refresh requests.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
ATTACK = {
    "A0": ({"id": "A0"}, {"epsilon": 0.0, "beta_attack": 0.0, "beta_v_attack": 0.0}),
    "PHYS": ({"id": "A8"}, {"epsilon": 2.5, "beta_attack": 0.0, "beta_v_attack": 0.0}),
    "REPORT": ({"id": "A4"}, {"epsilon": 0.0, "beta_attack": 1.25, "beta_v_attack": 0.5}),
    "COUPLED": ({"id": "A14"}, {"epsilon": 2.5, "beta_attack": 1.25, "beta_v_attack": 0.5}),
}
REGIME = {
    "nominal": {"fanet": {"delay_s": 0.0, "jitter_s": 0.0, "loss": 0.0, "ge_p_gb": 0.0, "ge_p_bg": 0.3, "t_net_max": 0.05,
                          "bucket_bytes_per_s": 2000.0, "bucket_burst_bytes": 576.0, "packet_bytes": 72}},
    "adverse": {"fanet": {"delay_s": 0.10, "jitter_s": 0.02, "loss": 0.10, "ge_p_gb": 0.0, "ge_p_bg": 0.3, "t_net_max": 0.12 + 1e-6,
                          "bucket_bytes_per_s": 2000.0, "bucket_burst_bytes": 576.0, "packet_bytes": 72}},
}
SETTINGS = {"S0": {"spacing_m": 6.0, "v_mission": 1.0}, "S1": {"spacing_m": 8.0, "v_mission": 1.0}, "S2": {"spacing_m": 8.0, "v_mission": 0.75}}
WORLD = "worlds/coupled_safety/yard_physics.sdf"
ROUTE_M = 24.0
TRIGGER_RADIUS = 9.0                     # existing hybrid-attack convention (config/swarm/ral26_base.toml)


def load_scenarios(path=None):
    p = Path(path or HERE / "SCENARIOS.jsonl")
    return {json.loads(l)["id"]: json.loads(l) for l in p.read_text().splitlines() if l.strip()}


def episode(scn, cell, method, regime="nominal", setting="S0", bounds=None, info="standard", slot_shift=None, violation=None,
            name=None, extra=None, attack_override=None):
    atk, par = ATTACK[cell]
    par = dict(par, **(attack_override or {}))
    S = SETTINGS[setting]
    th = math.radians(scn["heading_deg"])
    ral = {"route": [round(ROUTE_M * math.cos(th), 5), round(ROUTE_M * math.sin(th), 5)], "v_mission": S["v_mission"],
           "attacker_id": scn["attacker_id"], "truth_hz": 50.0, "sim_time_net": True, "res_key": scn["net_key"],
           "attack": {**par, "start_s": scn["onset_s"], "trigger_radius_m": TRIGGER_RADIUS, "attacker_policy": "D0R",
                      "attacker_refuse_refresh": True},
           "bounds": dict(bounds or {}), **REGIME[regime]}
    if regime == "adverse":
        # end-to-end bound = emulated channel bound (delay + jitter) + the frozen middleware allowance measured in development
        ral["bounds"]["T_net_max"] = max(ral["bounds"].get("T_net_max", 0.0), ral["bounds"].get("T_net_max_adverse", 0.0), REGIME["adverse"]["fanet"]["t_net_max"])
    cs = {"spawn_jitter": scn["jitter"], "info": {"mode": info}}
    if slot_shift:
        cs["slot_shift"] = slot_shift
    ov = {"experiment": {"seed": scn["seed"], "duration_s": 100.0}, "swarm": {"num_uavs": scn["N"]}, "formation": {"spacing_m": S["spacing_m"]},
          "attack": atk, "defense": {"id": method}, "wind": {"speed": 0.0, "sdf": WORLD}, "ral26": ral, "cs": cs}
    if extra:
        ov = _merge(ov, extra)
    nm = name or f"{scn['set']}-{setting}-{cell}-{method}-{regime}-{info}-N{scn['N']}-s{scn['seed']}"
    return {"name": nm, "overrides": ov,
            "meta": {"set": scn["set"], "cell": cell, "method": method, "regime": regime, "info": info, "setting": setting, "N": scn["N"],
                     "seed": scn["seed"], "scenario": scn["id"], "attacker_id": scn["attacker_id"], "violation": violation,
                     "slot_shift": slot_shift}}


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
