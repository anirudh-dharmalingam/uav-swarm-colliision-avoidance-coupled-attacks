#!/usr/bin/env bash
# Sequential confirmatory chain (never concurrent: each campaign's timing validity depends on an otherwise idle host).
#   1. rarf_plant_time_v3 confirm (3 workers, RTF-locked 0.5)   -> experiments/rarf_plant_time_v3/run_chain.sh confirm (verifies FROZEN_PLAN.json)
#   2. P7 verification-bundle ablation confirm (2 workers, RTF >= 0.95 rule) -> scripts/le/run_px4_p7.sh confirm N (verifies P7_FROZEN.json)
#   3. P7 frozen analysis
set -u
cd "$(dirname "$0")/.."
L=results/confirm_chain_20261029.log
echo "start $(date -Is)" >> $L
bash experiments/rarf_plant_time_v3/run_chain.sh confirm 3 >> $L 2>&1; echo "rarf_v3 rc=$? $(date -Is)" >> $L
N=$(python3 -c "import json;print(json.load(open('experiments/le/P7_FROZEN.json'))['N'])")
bash scripts/le/run_px4_p7.sh confirm "$N" >> $L 2>&1; echo "p7 rc=$? $(date -Is)" >> $L
docker run --rm -u "$(id -u):$(id -g)" -v /home/anirudh/Documents:/w -w /w/Disaster-Swarm-dev dronelab:research-v1.17 \
    python3 scripts/le/p7_analysis.py results/le_px4/px4_p7 --out output/research/le_p7/confirm >> $L 2>&1
echo "CHAIN_DONE $(date -Is)" >> $L
