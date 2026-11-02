#!/usr/bin/env bash
# Chain 8 (simulator only): A-OO on-off attacks, A-R2 scale re-run with the corrected ledger (DEVIATIONS D1), B-ISS slack-weight
# sweep, B-DLY delay ladder; after chain 7, before chain 5.
set -u
cd "$(dirname "$0")/.."
L=results/chain_20261101h.log
IMG=dronelab:research-v1.17
RUN() { docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace -e PYTHONPATH=/workspace -e OMP_NUM_THREADS=1 \
        -e OPENBLAS_NUM_THREADS=1 --cpus 14 "$@"; }
echo "start $(date -Is)" >> $L
until grep -q "CHAIN_DONE" results/chain_20261031g.log 2>/dev/null; do sleep 60; done
for f in experiments/le_x_onoff/FROZEN.sha256 experiments/rarf_l1_theory/FROZEN.sha256; do
  sha256sum -c $f >> $L 2>&1 || { echo "FROZEN MISMATCH $f" >> $L; exit 1; }
done
RUN --name aoo $IMG nice -n 5 python3 scripts/le_x/run_onoff.py --workers 14 >> results/aoo.log 2>&1
echo "A-OO rc=$? $(date -Is)" >> $L
RUN $IMG python3 scripts/le_x/analyze_onoff.py >> $L 2>&1
RUN --name ar2_scale_v2 $IMG nice -n 5 python3 scripts/le_x/cv_scale_v2.py run --workers 14 >> results/ar2_scale_v2.log 2>&1
echo "A-R2 scale_v2 rc=$? $(date -Is)" >> $L
RUN $IMG python3 scripts/le_x/cv_scale_v2.py analyze >> $L 2>&1
RUN --name biss $IMG nice -n 5 python3 scripts/rarf_l1/run_theory.py iss --workers 14 >> results/biss.log 2>&1
echo "B-ISS rc=$? $(date -Is)" >> $L
RUN --name bdly $IMG nice -n 5 python3 scripts/rarf_l1/run_theory.py dly --workers 14 >> results/bdly.log 2>&1
echo "B-DLY rc=$? $(date -Is)" >> $L
RUN $IMG python3 scripts/rarf_l1/analyze_theory.py >> $L 2>&1
echo "CHAIN_DONE $(date -Is)" >> $L
