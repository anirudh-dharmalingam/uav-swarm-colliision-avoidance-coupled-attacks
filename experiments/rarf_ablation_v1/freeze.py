#!/usr/bin/env python3
"""Freeze / verify the rarf_ablation_v1 confirmatory inputs (sha256 of every runtime, design and analysis file).

    python3 experiments/rarf_ablation_v1/freeze.py freeze      # writes FROZEN_CODE.json (refuses to overwrite)
    python3 experiments/rarf_ablation_v1/freeze.py verify      # exit 1 on any mismatch
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FILES = ["swarmsec/ral26_core.py", "swarmsec/ral26_policy.py", "swarmsec/ral26_cs.py", "swarmsec/fanet.py", "swarmsec/le_v2/fanet_v2.py",
         "controllers/swarm/cs_agent_node.py", "controllers/swarm/uav_agent_node.py", "controllers/swarm/px4_iface.py",
         "controllers/swarm/fanet_bridge.py", "controllers/swarm/truth_evaluator.py", "controllers/swarm/gcs_coordinator.py",
         "controllers/swarm/oracle_adapter.py", "controllers/swarm/gz_stream.py", "scripts/swarm/experiment_manager.py",
         "scripts/swarm/ral26_batch.py", "scripts/coupled_safety/cs_batch.py", "scripts/coupled_safety/metrics.py",
         "config/swarm/coupled_safety_base.toml", "worlds/coupled_safety/yard_physics.sdf", "experiments/coupled_safety_v1/runs.py",
         "experiments/coupled_safety_v1/scenarios.py", "experiments/coupled_safety_v1/bounds_declared.json",
         "experiments/rarf_ablation_v1/design.py", "experiments/rarf_ablation_v1/FROZEN_PLAN.json", "experiments/rarf_ablation_v1/ANALYSIS_PLAN.md",
         "experiments/rarf_ablation_v1/DEV_PLAN.md", "experiments/rarf_ablation_v1/specs/exp1.json", "experiments/rarf_ablation_v1/specs/exp2.json",
         "scripts/rarf_ablation/outcomes.py", "scripts/rarf_ablation/analyze.py", "scripts/rarf_ablation/timing_audit.py",
         "scripts/rarf_ablation/figures.py", "scripts/rarf_ablation/certified_budget.py", "tests/test_rarf_ablation.py"]


def digest():
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() if (ROOT / f).exists() else None for f in FILES}


def main():
    out = HERE / "FROZEN_CODE.json"
    if sys.argv[1:] == ["freeze"]:
        if out.exists():
            sys.exit("FROZEN_CODE.json exists")
        h = digest()
        miss = [f for f, v in h.items() if v is None]
        if miss:
            sys.exit(f"missing: {miss}")
        img = subprocess.run(["docker", "image", "inspect", "dronelab:research-v1.17", "--format", "{{.Id}}"], capture_output=True, text=True).stdout.strip()
        bundle = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(h.items())).encode()).hexdigest()
        out.write_text(json.dumps({"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(), "image_id": img,
            "bundle_sha256": bundle, "files": h, "amendments": []}, indent=1))
        print("frozen", len(h), bundle)
    elif sys.argv[1:] == ["verify"]:
        F = json.loads(out.read_text())
        now = digest()
        amended = {a["file"]: a["amended_sha256"] for a in F.get("amendments", [])}
        bad = [f for f, v in F["files"].items() if now.get(f) not in (v, amended.get(f))]
        print("verify:", "OK" if not bad else f"MISMATCH {bad}")
        sys.exit(1 if bad else 0)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
