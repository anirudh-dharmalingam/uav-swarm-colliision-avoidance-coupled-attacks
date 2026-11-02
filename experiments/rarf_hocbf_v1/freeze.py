#!/usr/bin/env python3
"""Freeze / verify rarf_hocbf_v1 (PLAN.md §3-§4): selected gains + N from the dev analysis, sha256 of every runtime / analysis file.

    python3 experiments/rarf_hocbf_v1/freeze.py freeze output/research/rarf_hocbf_v1/dev/results.json output/research/rarf_hocbf_v1/dev2/results.json
    python3 experiments/rarf_hocbf_v1/freeze.py verify
"""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = ["experiments/rarf_hocbf_v1/PLAN.md", "experiments/rarf_hocbf_v1/design.py", "experiments/rarf_hocbf_v1/run.sh",
         "controllers/swarm/cs_agent_node_v3.py", "controllers/swarm/cs_agent_node_v2.py", "swarmsec/ral26_hocbf.py", "swarmsec/safety.py",
         "scripts/rarf_hocbf_v1/analyze.py", "scripts/rarf_plant_time_v3/analyze.py", "scripts/rarf_ablation/outcomes.py",
         "scripts/coupled_safety/metrics.py", "experiments/rarf_plant_time_v3/design.py", "experiments/rarf_ablation_v1/design.py",
         "worlds/coupled_safety/yard_physics_rtf030.sdf", "experiments/coupled_safety_v1/bounds_declared.json",
         "controllers/swarm/cs_agent_node_v4.py", "swarmsec/ral26_baselines2.py", "experiments/rarf_hocbf_v1/freeze.py"]


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def main():
    out = HERE / "FROZEN.json"
    if sys.argv[1] == "freeze":
        if out.exists():
            sys.exit("FROZEN.json exists; refusing to overwrite")
        dev = json.loads(Path(sys.argv[2]).read_text())
        dev2 = json.loads(Path(sys.argv[3]).read_text())                     # amendment A2: ORCA / MRCBF stage
        sel = {**dev["selection"], **dev2["selection"]}
        rec = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "selected": {k: v["selected"] for k, v in sel.items()},
               "selection": sel, "N": 12, "sample_size_rule": dev["sample_size_rule"],
               "dev_instrument_pass": dev.get("instrument_pass"), "dev2_instrument_pass": dev2.get("instrument_pass"), "sha256": {p: sha(p) for p in FILES},
               "bootstrap": {"resamples": 10000, "numpy_seed": 20261101}, "workers": 3}
        out.write_text(json.dumps(rec, indent=1))
        print("frozen", rec["selected"], "N =", rec["N"])
    else:
        rec = json.loads(out.read_text())
        bad = [p for p, h in rec["sha256"].items() if sha(p) != h]
        print("verify:", "OK" if not bad else f"MISMATCH {bad}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
