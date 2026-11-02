#!/usr/bin/env python3
"""Development-setting selection (PREREG rule, applied in order S0 -> S1 -> S2): the FIRST setting whose 8 benign episodes (seeds 12000-12007) all
  (a) complete the route with no separation violation (min separation over all pairs >= 1.5 m, route_complete, no crash),
  (b) have valid-premise interval coverage >= 80 %  (coverage = honest-UAV MISSION ticks that are CERTIFIED_INTERVAL with every audited
      premise holding / all honest-UAV MISSION ticks; premises audited against Gazebo truth by scripts/analysis/ral26_audit_v2.py),
  (c) show no take-off/landing harness fault (GCS exit 0, all agents DONE).
A setting is never chosen because it produces an attack effect.  If none passes the intended certified regime is NOT established.

    python3 scripts/coupled_safety/dev_select.py results/coupled_safety_v1/dev_benign --metrics m.csv --audit audit_v2.json --out sel.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("root")
ap.add_argument("--metrics", required=True)
ap.add_argument("--audit", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
M = pd.read_csv(a.metrics)
AU = {r["name"]: r for r in json.loads(Path(a.audit).read_text()) if "error" not in r}
rows = []
for _, m in M.iterrows():
    rd = Path(a.root) / m["name"]
    setting = m["name"].split("-")[1]
    tot = 0
    for f in sorted((rd / "agents").glob("agent_*.csv.gz")):
        d = pd.read_csv(f, usecols=["phase"])
        tot += int((d.phase == "MISSION").sum())
    au = AU.get(m["name"], {})
    att = int(m["attacker"])
    # honest agents only: the audit skips the designated UAV
    cov = au.get("prem_all", 0) / max(1, tot * 5 / 6)
    st = json.loads((rd / "episode_status.json").read_text()).get("gcs_code") if (rd / "episode_status.json").exists() else None
    rows.append({"name": m["name"], "setting": setting, "min_sep_all": m["min_sep_all"], "route_complete": m["route_complete"], "crash": m["crash"],
                 "cert_ticks": au.get("cert_ticks", 0), "prem_all": au.get("prem_all", 0), "honest_mission_ticks_est": tot * 5 / 6, "coverage_valid_premise": cov,
                 "gcs_code": st, "final_phases": m["final_phases"]})
df = pd.DataFrame(rows)
sel = {}
for s, g in df.groupby("setting"):
    ok_a = bool(((g.min_sep_all >= 1.5) & (g.route_complete == 1) & (g.crash == 0)).all()) and len(g) == 8
    ok_b = bool((g.coverage_valid_premise >= 0.8).all())
    ok_c = bool(((g.gcs_code == 0) & (g.final_phases == "DONE")).all())
    sel[s] = {"n": int(len(g)), "a_no_violation_complete": ok_a, "b_coverage_all_ge_80pct": ok_b, "coverage_mean": float(g.coverage_valid_premise.mean()),
              "coverage_min": float(g.coverage_valid_premise.min()), "c_no_harness_fault": ok_c, "passes": ok_a and ok_b and ok_c}
chosen = next((s for s in ("S0", "S1", "S2") if sel.get(s, {}).get("passes")), None)
out = {"settings": sel, "selected": chosen, "intended_certified_regime_established": chosen is not None}
Path(a.out).write_text(json.dumps(out, indent=1))
df.to_csv(Path(a.out).with_suffix(".csv"), index=False)
print(json.dumps(out, indent=1))
