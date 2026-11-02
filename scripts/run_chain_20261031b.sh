#!/usr/bin/env bash
# Transactions programme chain, Layer-1 block 2 (simulator only; no PX4 campaign may run beside it):
#   B-A1 search -> A-R1 search -> B-A1 test -> A-R1 test -> B-N1 scale -> A-G1 scale (fixed, then adaptive)
set -u
cd "$(dirname "$0")/.."
L=results/chain_20261031b.log
IMG=dronelab:research-v1.17
RUN() { docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace -e PYTHONPATH=/workspace -e OMP_NUM_THREADS=1 \
        -e OPENBLAS_NUM_THREADS=1 --cpus 14 "$@"; }
echo "start $(date -Is) profile=$(powerprofilesctl get)" >> $L
for f in experiments/rarf_l1_adaptive/FROZEN.sha256 experiments/le_x_adaptive/FROZEN.sha256; do
  sha256sum -c $f >> $L 2>&1 || { echo "FROZEN MISMATCH $f" >> $L; exit 1; }
done
RUN --name ba1_search $IMG nice -n 5 python3 scripts/rarf_l1/adaptive_search.py search --workers 14 >> results/ba1_search.log 2>&1
echo "B-A1 search rc=$? $(date -Is)" >> $L
RUN --name ar1_search $IMG nice -n 5 python3 scripts/le_x/adaptive_search.py search --workers 14 >> results/ar1_search.log 2>&1
echo "A-R1 search rc=$? $(date -Is)" >> $L
RUN --name ba1_test $IMG nice -n 5 python3 scripts/rarf_l1/adaptive_search.py test --workers 14 >> results/ba1_test.log 2>&1
echo "B-A1 test rc=$? $(date -Is)" >> $L
RUN --name ar1_test $IMG nice -n 5 python3 scripts/le_x/adaptive_search.py test --workers 14 >> results/ar1_test.log 2>&1
echo "A-R1 test rc=$? $(date -Is)" >> $L
RUN --name bn1_scale $IMG nice -n 5 python3 scripts/rarf_l1/run_l1x.py --spec experiments/rarf_l1_adaptive/specs/scale.json --tag scale \
    --workers 14 >> results/bn1_scale.log 2>&1
echo "B-N1 scale rc=$? $(date -Is)" >> $L
for st in fixed adaptive; do
  RUN --name ag1_$st $IMG nice -n 5 python3 scripts/le_x/run_scale.py $st --workers 14 >> results/ag1_scale_$st.log 2>&1
  echo "A-G1 $st rc=$? $(date -Is)" >> $L
done
echo "CHAIN_DONE $(date -Is)" >> $L
