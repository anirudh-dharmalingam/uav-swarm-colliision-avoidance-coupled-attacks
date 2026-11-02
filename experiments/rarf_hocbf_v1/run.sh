#!/usr/bin/env bash
# rarf_hocbf_v1 runner (PLAN.md).  bash experiments/rarf_hocbf_v1/run.sh dev|confirm [workers]   (resume-safe)
set -u
cd "$(dirname "$0")/../.."
STAGE=$1; W=${2:-3}
L=results/rarf_hocbf_v1/chain.log; mkdir -p results/rarf_hocbf_v1
echo "$STAGE start $(date) workers=$W" >> $L
if [ "$STAGE" = confirm ]; then python3 experiments/rarf_hocbf_v1/freeze.py verify >> $L 2>&1 || { echo "VERIFY FAILED" >> $L; exit 1; }; fi
python3 experiments/rarf_hocbf_v1/design.py "$STAGE" >> $L 2>&1
python3 scripts/coupled_safety/cs_batch.py --spec experiments/rarf_hocbf_v1/specs/$STAGE.json --workers "$W" --timeout 900 \
    >> results/rarf_hocbf_v1/${STAGE}_launch.log 2>&1
[ "$STAGE" = confirm ] && { python3 experiments/rarf_hocbf_v1/freeze.py verify >> $L 2>&1 || echo "POST-RUN VERIFY FAILED" >> $L; }
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace -e PYTHONPATH=/workspace dronelab:research-v1.17 \
    python3 scripts/rarf_hocbf_v1/analyze.py "$STAGE" >> $L 2>&1
echo "${STAGE}_DONE $(date)" >> $L
