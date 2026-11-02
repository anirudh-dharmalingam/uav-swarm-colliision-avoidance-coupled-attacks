#!/usr/bin/env bash
# 1. P8 factorial confirm (idle host; RTF >= 0.95 validity)  2. P8 frozen analysis
# 3. rarf_hocbf_v1 confirm (plant-time locked) with the la_robust_v1 simulator discovery screen resumed alongside (3 niced cores)
set -u
cd "$(dirname "$0")/.."
L=results/chain_20261030.log
echo "start $(date -Is)" >> $L
N=$(python3 -c "import json;print(json.load(open('experiments/le/P8_FROZEN.json'))['N'])")
bash scripts/le/run_px4_p8.sh confirm "$N" >> $L 2>&1; echo "p8 rc=$? $(date -Is)" >> $L
docker run --rm -u "$(id -u):$(id -g)" -v /home/anirudh/Documents:/w -w /w/Disaster-Swarm-dev dronelab:research-v1.17 \
    python3 scripts/le/p8_analysis.py results/le_px4/px4_p8 --out output/research/le_p8/confirm >> $L 2>&1
echo "p8 analysis done $(date -Is)" >> $L
docker run -d --rm --name la_robust_screen --cpus 3 -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace -e PYTHONPATH=/workspace \
    dronelab:research-v1.17 nice -n 19 python3 scripts/la_robust/run_screen.py discovery --workers 3 >> $L 2>&1
bash experiments/rarf_hocbf_v1/run.sh confirm 3 >> $L 2>&1; echo "hocbf rc=$? $(date -Is)" >> $L
echo "CHAIN_DONE $(date -Is)" >> $L
