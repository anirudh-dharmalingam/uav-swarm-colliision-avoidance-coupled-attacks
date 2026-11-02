#!/usr/bin/env bash
# Chain 4 (simulator only): A-R2 claim verification (experiments/le_x_claimfix/PLAN.md), after chain 2 and before chain 3.
set -u
cd "$(dirname "$0")/.."
L=results/chain_20261031d.log
IMG=dronelab:research-v1.17
RUN() { docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace -e PYTHONPATH=/workspace -e OMP_NUM_THREADS=1 \
        -e OPENBLAS_NUM_THREADS=1 --cpus 14 "$@"; }
echo "start $(date -Is)" >> $L
until grep -q "CHAIN_DONE" results/chain_20261031b.log; do sleep 60; done
sha256sum -c experiments/le_x_claimfix/FROZEN.sha256 >> $L 2>&1 || { echo "FROZEN MISMATCH" >> $L; exit 1; }
RUN --name ar2_search $IMG nice -n 5 python3 scripts/le_x/cv_eval.py search --workers 14 >> results/ar2_search.log 2>&1
echo "A-R2 search rc=$? $(date -Is)" >> $L
RUN --name ar2_test $IMG nice -n 5 python3 scripts/le_x/cv_eval.py test --workers 14 >> results/ar2_test.log 2>&1
echo "A-R2 test rc=$? $(date -Is)" >> $L
RUN $IMG python3 scripts/le_x/analyze_cv.py test >> $L 2>&1
RUN --name ar2_scale $IMG nice -n 5 python3 scripts/le_x/cv_eval.py scale --workers 14 >> results/ar2_scale.log 2>&1
echo "A-R2 scale rc=$? $(date -Is)" >> $L
RUN $IMG python3 scripts/le_x/analyze_cv.py scale >> $L 2>&1
echo "CHAIN_DONE $(date -Is)" >> $L
