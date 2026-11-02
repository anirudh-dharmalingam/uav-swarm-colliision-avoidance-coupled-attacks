#!/usr/bin/env bash
# rarf_plant_time_v3 runner (FROZEN_PLAN.md).  Resume-safe: cs_batch skips episodes that already have eval/summary.json.
#   bash experiments/rarf_plant_time_v3/run_chain.sh smoke|pilot|confirm [workers]
set -u
cd "$(dirname "$0")/../.."
STAGE=$1
W=${2:-3}
L=results/rarf_plant_time_v3/chain.log
mkdir -p results/rarf_plant_time_v3
dpy() { docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace -e PYTHONPATH=/workspace dronelab:research-v1.17 python3 "$@"; }
echo "$STAGE start $(date) workers=$W" >> $L
if [ "$STAGE" = confirm ]; then
  python3 experiments/rarf_plant_time_v3/freeze.py verify >> $L 2>&1 || { echo "VERIFY FAILED" >> $L; exit 1; }
fi
python3 experiments/rarf_plant_time_v3/design.py "$STAGE" >> $L 2>&1
python3 scripts/coupled_safety/cs_batch.py --spec experiments/rarf_plant_time_v3/specs/$STAGE.json --workers "$W" --timeout 900 \
    >> results/rarf_plant_time_v3/${STAGE}_launch.log 2>&1
echo "$STAGE batch done $(date)" >> $L
PIL=""; case "$STAGE" in pilot*) PIL="--pilot";; esac
[ "$STAGE" = confirm ] && { python3 experiments/rarf_plant_time_v3/freeze.py verify >> $L 2>&1 || echo "POST-RUN VERIFY FAILED" >> $L; }
dpy scripts/rarf_plant_time_v3/analyze.py results/rarf_plant_time_v3/$STAGE --spec experiments/rarf_plant_time_v3/specs/$STAGE.json \
    --out output/research/rarf_plant_time_v3/$STAGE $PIL >> $L 2>&1
echo "${STAGE}_DONE $(date)" >> $L
