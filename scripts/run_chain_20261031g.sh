#!/usr/bin/env bash
# Chain 7 (simulator only): B-PB published avoidance baselines (experiments/rarf_l1_published/PLAN.md), after chain 6 and before chain 5.
set -u
cd "$(dirname "$0")/.."
L=results/chain_20261031g.log
IMG=dronelab:research-v1.17
RUN() { docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace -e PYTHONPATH=/workspace -e OMP_NUM_THREADS=1 \
        -e OPENBLAS_NUM_THREADS=1 --cpus 14 "$@"; }
echo "start $(date -Is)" >> $L
until grep -q "CHAIN_DONE" results/chain_20261031f.log 2>/dev/null; do sleep 60; done
sha256sum -c experiments/rarf_l1_published/FROZEN.sha256 >> $L 2>&1 || { echo "FROZEN MISMATCH" >> $L; exit 1; }
for st in dev spectrum budget search test rz scale; do
  RUN --name bpb_$st $IMG nice -n 5 python3 scripts/rarf_l1/run_pb.py $st --workers 14 >> results/bpb_$st.log 2>&1
  rc=$?
  echo "B-PB $st rc=$rc $(date -Is)" >> $L
  [ "$st" = dev ] && { [ $rc -eq 0 ] && [ -f output/research/rarf_l1_published/FROZEN_B3.json ] || { echo "B-PB dev failed, stopping" >> $L; exit 1; }; }
done
RUN $IMG python3 scripts/rarf_l1/analyze_pb.py >> $L 2>&1
echo "CHAIN_DONE $(date -Is)" >> $L
