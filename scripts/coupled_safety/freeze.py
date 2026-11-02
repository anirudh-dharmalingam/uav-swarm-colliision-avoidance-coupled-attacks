#!/usr/bin/env python3
"""Freeze / verify the confirmatory code, configuration, scenario table and specs.

    python3 scripts/coupled_safety/freeze.py freeze    -> experiments/coupled_safety_v1/FROZEN_CODE.json (+ source snapshot tar.gz)
    python3 scripts/coupled_safety/freeze.py verify -> exit 0 iff every frozen file still has its recorded sha256

Frozen: runtime controller/bridge/evaluator/policy code, config, worlds, scenario table, exact selected settings (inside the specs and
PREREG.md), metric/analysis scripts.  A later bug fix requires a versioned new campaign or re-running all affected matched cells (DEVIATIONS.md).
"""
import hashlib
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "coupled_safety_v1"
FILES = [
    "swarmsec/ral26_core.py", "swarmsec/ral26_policy.py", "swarmsec/ral26_cs.py", "swarmsec/fanet.py", "swarmsec/le_v2/fanet_v2.py",
    "controllers/swarm/cs_agent_node.py", "controllers/swarm/uav_agent_node.py", "controllers/swarm/px4_iface.py",
    "controllers/swarm/fanet_bridge.py", "controllers/swarm/oracle_adapter.py", "controllers/swarm/truth_evaluator.py",
    "controllers/swarm/gcs_coordinator.py", "controllers/swarm/energy_supervisor.py", "controllers/swarm/gz_stream.py",
    "scripts/swarm/experiment_manager.py", "scripts/swarm/ral26_batch.py", "config/swarm/coupled_safety_base.toml",
    "scripts/demo/make_coupled_scene.py", "scripts/analysis/ral26_audit_v2.py",
    "experiments/coupled_safety_v1/runs.py", "experiments/coupled_safety_v1/scenarios.py", "experiments/coupled_safety_v1/make_specs.py",
    "experiments/coupled_safety_v1/make_confirmatory.py", "experiments/coupled_safety_v1/bounds_declared.json",
    "experiments/coupled_safety_v1/SCENARIOS.jsonl", "experiments/coupled_safety_v1/PREREG.md", "tests/test_coupled_safety.py",
]
GLOBS = ["scripts/coupled_safety/*.py", "worlds/coupled_safety/*.sdf", "experiments/coupled_safety_v1/specs/e[0-9]*.json"]
EXCLUDE = {"scripts/coupled_safety/freeze.py"}


def listing():
    out = [ROOT / f for f in FILES]
    for g in GLOBS:
        out += sorted(ROOT.glob(g))
    return [p for p in dict.fromkeys(out) if str(p.relative_to(ROOT)) not in EXCLUDE]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    out = EXP / "FROZEN_CODE.json"
    if cmd == "freeze":
        files = {str(p.relative_to(ROOT)): sha(p) for p in listing() if p.exists()}
        missing = [f for f in FILES if not (ROOT / f).exists()]
        bundle = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        try:
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except Exception:
            head = ""
        out.write_text(json.dumps({"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "git_head": head, "missing_at_freeze": missing,
                                   "bundle_sha256": bundle, "files": files}, indent=1))
        with tarfile.open(EXP / "code_snapshot_frozen.tar.gz", "w:gz") as t:
            for f in files:
                t.add(ROOT / f, arcname=f)
        print("frozen", len(files), "files; bundle", bundle[:16], "missing:", missing)
    elif cmd == "amend":        # amend <file> "<reason>": record a post-freeze change of a NON-RUNTIME-PATH file (original hash kept, never overwritten)
        rec = json.loads(out.read_text())
        f, reason = sys.argv[2], sys.argv[3]
        rec.setdefault("amendments", []).append({"file": f, "frozen_sha256": rec["files"][f], "amended_sha256": sha(ROOT / f), "reason": reason,
                                                 "at": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
        out.write_text(json.dumps(rec, indent=1))
        print("amended", f)
    else:
        rec = json.loads(out.read_text())
        ok = {a["file"]: a["amended_sha256"] for a in rec.get("amendments", [])}      # latest amendment per file
        bad = [f for f, h in rec["files"].items() if not (ROOT / f).exists() or sha(ROOT / f) not in (h, ok.get(f))]
        print("frozen set verified" if not bad else f"MISMATCH: {bad}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
