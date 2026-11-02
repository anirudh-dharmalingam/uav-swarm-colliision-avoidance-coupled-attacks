#!/usr/bin/env python3
"""Video-rerun batch: cs_batch.py + Gazebo state recording (`gz sim --record`) for later replay rendering.

Episodes whose overrides contain {"wind": {"sdf": ..., "record": true}} start the Gazebo server with `--record --record-path <run>/gzlog` (state log
only; cameras are always_on=0, so no rendering cost during the flight).  The flight is a NEW experiment (same seed/scenario as the analysed episode),
run with the frozen controller code; nothing else differs.

    python3 scripts/coupled_safety/cs_batch_video.py --spec output/video/coupled_safety_v1/specs/video_flights.json --workers 2
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "coupled_safety"))
import cs_batch as CB  # noqa: E402

B = CB.B
_ws = B.world_setup


def world_setup(wind, rd):
    base = _ws(wind, rd)
    if wind and wind.get("record"):
        fn = ('gz() { if [ "$1" = sim ]; then shift; command gz sim --record --record-path ' + f'{rd}/gzlog' + ' --log-overwrite "$@"; else command gz "$@"; fi; }\n')
        return base + fn
    return base


B.world_setup = world_setup
B.RESULTS_ROOT = "results/coupled_safety_v1"

if __name__ == "__main__":
    B.main()
