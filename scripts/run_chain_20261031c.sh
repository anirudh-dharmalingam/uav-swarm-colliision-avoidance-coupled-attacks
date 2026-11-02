#!/usr/bin/env bash
# Chain 3 (simulator only): after chain 5 (re-queued 2026-10-31 20:27), the two holdout validations named in the discovery reports.
#   la_robust_v1 validation of the selected points (PROTOCOL.md §6) and A-C1 validation (experiments/le_x_collusion/VALIDATION.md)
set -u
cd "$(dirname "$0")/.."
L=results/chain_20261031c.log
IMG=dronelab:research-v1.17
RUN() { docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace -e PYTHONPATH=/workspace -e OMP_NUM_THREADS=1 \
        -e OPENBLAS_NUM_THREADS=1 --cpus 14 "$@"; }
echo "start $(date -Is)" >> $L
until grep -q "CHAIN_DONE" results/chain_20261031e.log; do sleep 60; done
RUN --name la_robust_val $IMG nice -n 5 python3 scripts/la_robust/run_screen.py validation --points nominal,X1,P15 --workers 14 \
    >> results/la_robust_validation.log 2>&1
echo "la_robust validation rc=$? $(date -Is)" >> $L
RUN $IMG python3 scripts/la_robust/analyze_screen.py validation >> $L 2>&1
sha256sum -c experiments/le_x_collusion/VALIDATION.sha256 >> $L 2>&1 || { echo "VALIDATION MISMATCH" >> $L; exit 1; }
RUN --name ac1_val $IMG nice -n 5 python3 scripts/le_x/run_collusion.py validation --cells experiments/le_x_collusion/validation_cells.json \
    --workers 14 >> results/le_x_collusion_validation.log 2>&1
echo "A-C1 validation rc=$? $(date -Is)" >> $L
RUN $IMG python3 scripts/le_x/analyze_collusion.py validation >> $L 2>&1
echo "CHAIN_DONE $(date -Is)" >> $L
