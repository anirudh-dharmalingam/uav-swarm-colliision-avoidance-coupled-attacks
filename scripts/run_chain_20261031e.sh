#!/usr/bin/env bash
# Chain 5 (simulator only): resilience matrices A-RZ (experiments/le_x_resilience/PLAN.md) and B-RZ
# (experiments/rarf_l1_resilience/PLAN.md), after chain 8 (A-OO, B-ISS, B-DLY; re-queued 2026-11-01 00:12) and before chain 3.
# Also runs the A-PB analysis, which needs the A-RZ rows.
set -u
cd "$(dirname "$0")/.."
L=results/chain_20261031e.log
IMG=dronelab:research-v1.17
RUN() { docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace -e PYTHONPATH=/workspace -e OMP_NUM_THREADS=1 \
        -e OPENBLAS_NUM_THREADS=1 --cpus 14 "$@"; }
echo "start $(date -Is)" >> $L
until grep -q "CHAIN_DONE" results/chain_20261101h.log 2>/dev/null; do sleep 60; done
for f in experiments/le_x_resilience/FROZEN.sha256 experiments/rarf_l1_resilience/FROZEN.sha256 experiments/le_x_published/FROZEN.sha256; do
  sha256sum -c $f >> $L 2>&1 || { echo "FROZEN MISMATCH $f" >> $L; exit 1; }
done
RUN --name arz_calib $IMG nice -n 5 python3 scripts/le_x/run_rz.py calib --workers 14 >> results/arz_calib.log 2>&1
echo "A-RZ calib rc=$? $(date -Is)" >> $L
RUN --name arz_main $IMG nice -n 5 python3 scripts/le_x/run_rz.py main --workers 14 >> results/arz_main.log 2>&1
echo "A-RZ main rc=$? $(date -Is)" >> $L
RUN $IMG python3 scripts/le_x/analyze_rz.py >> $L 2>&1
RUN $IMG python3 scripts/le_x/analyze_pb.py >> $L 2>&1
RUN --name brz $IMG nice -n 5 python3 scripts/rarf_l1/run_rz.py --workers 14 >> results/brz.log 2>&1
echo "B-RZ rc=$? $(date -Is)" >> $L
RUN $IMG python3 scripts/rarf_l1/analyze_rz.py >> $L 2>&1
echo "CHAIN_DONE $(date -Is)" >> $L
