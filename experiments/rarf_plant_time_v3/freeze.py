#!/usr/bin/env python3
"""Freeze / verify rarf_plant_time_v3 (FROZEN_PLAN.md §7): N from the pilot rule plus sha256 of every runtime and analysis file.

    python3 experiments/rarf_plant_time_v3/freeze.py freeze <pilot results.json>     # writes FROZEN_PLAN.json (refuses to overwrite)
    python3 experiments/rarf_plant_time_v3/freeze.py verify                          # exit 1 on any mismatch
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = ["experiments/rarf_plant_time_v3/FROZEN_PLAN.md", "experiments/rarf_plant_time_v3/design.py",
         "experiments/rarf_plant_time_v3/run_chain.sh", "controllers/swarm/cs_agent_node_v2.py",
         "worlds/coupled_safety/yard_physics_rtf030.sdf", "scripts/rarf_plant_time_v3/analyze.py",
         "scripts/rarf_ablation/outcomes.py", "scripts/coupled_safety/metrics.py", "experiments/rarf_ablation_v1/design.py",
         "experiments/rarf_ablation_v1/FROZEN_PLAN.json", "experiments/coupled_safety_v1/bounds_declared.json",
         "swarmsec/ral26_exec.py", "swarmsec/ral26_policy_x2.py"]


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def runtime_v1():
    """Every file of the rarf_ablation_v1 frozen bundle must still hash identically (runtime chain v1 is reused unchanged)."""
    fz = json.loads((ROOT / "experiments/rarf_ablation_v1/FROZEN_CODE.json").read_text())["files"]
    return {p: h for p, h in fz.items()}


def main():
    cmd = sys.argv[1]
    out = HERE / "FROZEN_PLAN.json"
    if cmd == "freeze":
        if out.exists():
            sys.exit("FROZEN_PLAN.json exists; refusing to overwrite")
        pilot = json.loads(Path(sys.argv[2]).read_text())
        ss = pilot["sample_size_rule"]
        img = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", "dronelab:research-v1.17"], capture_output=True, text=True).stdout.strip()
        rec = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "N": ss["N"], "sample_size_rule": ss,
               "pilot_instrument_pass": pilot.get("instrument_pass"), "pilot_instrument_fail_criteria": pilot.get("instrument_fail_criteria"),
               "image_id": img, "sha256": {p: sha(p) for p in FILES}, "runtime_v1_bundle": runtime_v1(),
               "bootstrap": {"resamples": 10000, "numpy_seed": 20261031}, "workers": 3, "confirm_seeds": [17300, 17300 + ss["N"] - 1]}
        bad = [p for p, h in rec["runtime_v1_bundle"].items() if (ROOT / p).exists() and sha(p) != h]
        if bad:
            sys.exit(f"rarf_ablation_v1 runtime bundle changed: {bad}")
        out.write_text(json.dumps(rec, indent=1))
        print("frozen", out, "N =", rec["N"])
    elif cmd == "verify":
        rec = json.loads(out.read_text())
        bad = [p for p, h in {**rec["sha256"], **rec["runtime_v1_bundle"]}.items() if (ROOT / p).exists() and sha(p) != h]
        missing = [p for p in {**rec["sha256"], **rec["runtime_v1_bundle"]} if not (ROOT / p).exists()]
        if bad or missing:
            print("verify: MISMATCH", bad, "missing", missing)
            sys.exit(1)
        print("verify: OK")


if __name__ == "__main__":
    main()
