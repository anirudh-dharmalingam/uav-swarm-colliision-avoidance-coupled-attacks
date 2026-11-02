#!/usr/bin/env bash
# Chain 6 (simulator only): A-PB published baselines (experiments/le_x_published/PLAN.md), after chain 4 and before chain 5.
# Runs the A-RZ calibration first (frozen command; chain 5 re-runs it as a no-op resume) because A-PB rz needs q_ref.
set -u
cd "$(dirname "$0")/.."
L=results/chain_20261031f.log
IMG=dronelab:research-v1.17
RUN() { docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace -e PYTHONPATH=/workspace -e OMP_NUM_THREADS=1 \
        -e OPENBLAS_NUM_THREADS=1 --cpus 14 "$@"; }
echo "start $(date -Is)" >> $L
until grep -q "CHAIN_DONE" results/chain_20261031d.log 2>/dev/null; do sleep 60; done
for f in experiments/le_x_published/FROZEN.sha256 experiments/le_x_resilience/FROZEN.sha256; do
  sha256sum -c $f >> $L 2>&1 || { echo "FROZEN MISMATCH $f" >> $L; exit 1; }
done
RUN --name arz_calib $IMG nice -n 5 python3 scripts/le_x/run_rz.py calib --workers 14 >> results/arz_calib.log 2>&1
echo "A-RZ calib rc=$? $(date -Is)" >> $L
RUN --name apb_search $IMG nice -n 5 python3 scripts/le_x/run_pb.py search --workers 14 >> results/apb_search.log 2>&1
echo "A-PB search rc=$? $(date -Is)" >> $L
RUN --name apb_test $IMG nice -n 5 python3 scripts/le_x/run_pb.py test --workers 14 >> results/apb_test.log 2>&1
echo "A-PB test rc=$? $(date -Is)" >> $L
RUN --name apb_rz $IMG nice -n 5 python3 scripts/le_x/run_pb.py rz --workers 14 >> results/apb_rz.log 2>&1
echo "A-PB rz rc=$? $(date -Is)" >> $L
echo "CHAIN_DONE $(date -Is)" >> $L
