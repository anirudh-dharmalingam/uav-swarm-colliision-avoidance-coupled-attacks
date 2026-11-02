#!/usr/bin/env bash
# Transactions programme chain, Layer-1 block 1 (output/research/transactions_program_20261030/PROGRAM.md §5).
# Waits for the rarf_hocbf_v1 PX4 confirmation to finish (host must stay clear while it flies), then runs simulator work only:
#   1. hocbf primary-population analysis (DEVIATIONS.md correction)   2. finish the la_robust_v1 screen + its analysis
#   3. Layer-1 fidelity replays (v3 confirm, hocbf confirm) + gate      4. A-C1 collusion discovery
#   5. B-S1 spectrum, B-S2 budget sweeps
set -u
cd "$(dirname "$0")/.."
L=results/chain_20261031.log
IMG=dronelab:research-v1.17
RUN() { docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace -e PYTHONPATH=/workspace -e OMP_NUM_THREADS=1 \
        -e OPENBLAS_NUM_THREADS=1 "$@"; }
echo "start $(date -Is)" >> $L
until grep -q "confirm_DONE" results/rarf_hocbf_v1/chain.log; do sleep 60; done
echo "hocbf confirm done $(date -Is)" >> $L
RUN $IMG python3 scripts/rarf_hocbf_v1/analyze_population.py >> $L 2>&1; echo "hocbf population analysis rc=$? $(date -Is)" >> $L
# la_robust screen: wait for the container to exit, then its frozen analysis
while docker ps --format '{{.Names}}' | grep -qx la_robust_screen; do sleep 30; done
RUN $IMG python3 scripts/la_robust/analyze_screen.py discovery >> $L 2>&1; echo "la_robust discovery analysis rc=$? $(date -Is)" >> $L
# Layer-1 fidelity: resume the paused v3 replay, then the hocbf replay, then the gate
docker unpause rarf_l1_v3 >> $L 2>&1
while docker ps --format '{{.Names}}' | grep -qx rarf_l1_v3; do sleep 30; done
RUN --cpus 14 $IMG nice -n 5 python3 scripts/rarf_l1/run_l1.py --spec experiments/rarf_hocbf_v1/specs/confirm.json --tag hocbf_confirm_replay \
    --workers 14 >> results/rarf_l1_hocbf_replay.log 2>&1
echo "l1 replays done $(date -Is)" >> $L
RUN $IMG python3 scripts/rarf_l1/fidelity.py --l1 results/rarf_l1/v3_confirm_replay/episodes.jsonl \
    --px4 output/research/rarf_plant_time_v3/confirm/episodes.csv --contrasts FF:NOM,FF:SPD,RARF:FF,RARF:NOM,RARF:SPD \
    --out output/research/rarf_l1/fidelity_v3 >> $L 2>&1
RUN $IMG python3 scripts/rarf_l1/fidelity.py --l1 results/rarf_l1/hocbf_confirm_replay/episodes.jsonl \
    --px4 output/research/rarf_hocbf_v1/confirm_primary_population/episodes.csv \
    --contrasts RARF:RHOCBF,RARF:HOCBF,FF:RHOCBF,RHOCBF:HOCBF,RARF:ORCA,RARF:MRCBF,FF:MRCBF,RARF:NOM,HOCBF:NOM,ORCA:NOM,MRCBF:NOM \
    --out output/research/rarf_l1/fidelity_hocbf >> $L 2>&1
echo "fidelity done $(date -Is)" >> $L
# A-C1 collusion discovery (hash check first)
sha256sum -c experiments/le_x_collusion/FROZEN.sha256 >> $L 2>&1 || { echo "A-C1 FROZEN MISMATCH" >> $L; exit 1; }
RUN --cpus 14 $IMG nice -n 5 python3 scripts/le_x/run_collusion.py discovery --workers 14 >> results/le_x_collusion_discovery.log 2>&1
echo "A-C1 discovery done $(date -Is)" >> $L
# B-S1 / B-S2 breadth sweeps (hash check first)
sha256sum -c experiments/rarf_l1_breadth/FROZEN.sha256 >> $L 2>&1 || { echo "breadth FROZEN MISMATCH" >> $L; exit 1; }
for T in spectrum budget; do
  RUN --cpus 14 $IMG nice -n 5 python3 scripts/rarf_l1/run_l1.py --spec experiments/rarf_l1_breadth/specs/$T.json --tag breadth_$T \
      --workers 14 >> results/rarf_l1_breadth_$T.log 2>&1
  echo "breadth $T done $(date -Is)" >> $L
done
echo "CHAIN_DONE $(date -Is)" >> $L
