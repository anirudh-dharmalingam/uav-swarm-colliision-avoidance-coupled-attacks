#!/usr/bin/env python3
"""Coupled-safety PX4 batch runner: scripts/swarm/ral26_batch.py (one Docker container per episode, K parallel) plus
  * results root from the spec ("results_root", default results/coupled_safety_v1)
  * optional custom Gazebo world per episode: overrides {"wind": {"sdf": "worlds/coupled_safety/yard_physics.sdf"}} (speed 0 => still air)
  * a hash/manifest resume rule: an episode whose eval/summary.json exists is skipped (never re-run, never overwritten)

    python3 scripts/coupled_safety/cs_batch.py --spec experiments/coupled_safety_v1/specs/<tag>.json --workers 2 [--only a,b]
Everything else (retry-once-after-infrastructure-failure, RTF re-run rule via spec "rtf_rerun_below", isolation by ROS_DOMAIN_ID/GZ_PARTITION)
is inherited unchanged from ral26_batch; original failed/slow attempts are kept in _failed_attempts/ and _rtf_attempts/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "swarm"))
import ral26_batch as B  # noqa: E402

_orig_world_setup = B.world_setup


def world_setup(wind, rd):
    if wind and wind.get("sdf"):
        sdf = wind["sdf"]
        return f"WORLD=/workspace/{sdf}\n" + (_orig_world_setup({k: v for k, v in wind.items() if k != "sdf"}, rd) if float(wind.get("speed", 0.0)) > 0 else "")
    return _orig_world_setup(wind, rd)


B.world_setup = world_setup
B.RESULTS_ROOT = "results/coupled_safety_v1"

if __name__ == "__main__":
    B.main()
