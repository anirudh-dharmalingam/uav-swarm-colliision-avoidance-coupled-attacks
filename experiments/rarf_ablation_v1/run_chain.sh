#!/usr/bin/env bash
# rarf_ablation_v1 confirmatory chain: verify frozen hashes -> Exp I -> verify -> Exp II -> outcomes/timing/analysis/figures.
# Resume-safe (cs_batch skips episodes with eval/summary.json).  Usage: bash experiments/rarf_ablation_v1/run_chain.sh <workers>
set -u
cd "$(dirname "$0")/../.."
W=${1:-1}
L=results/rarf_ablation_v1/chain.log
dpy() { docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace -e PYTHONPATH=/workspace dronelab:research-v1.17 python3 "$@"; }
echo "chain start $(date) workers=$W" >> $L
for t in exp1 exp2; do
  python3 experiments/rarf_ablation_v1/freeze.py verify >> $L 2>&1 || { echo "VERIFY FAILED before $t" >> $L; exit 1; }
  python3 scripts/coupled_safety/cs_batch.py --spec experiments/rarf_ablation_v1/specs/$t.json --workers "$W" >> results/rarf_ablation_v1/${t}_launch.log 2>&1
  echo "$t done $(date)" >> $L
done
A=results/rarf_ablation_v1/analysis
mkdir -p $A
dpy scripts/rarf_ablation/outcomes.py results/rarf_ablation_v1/exp1 results/rarf_ablation_v1/exp2 \
    --spec experiments/rarf_ablation_v1/specs/exp1.json --spec experiments/rarf_ablation_v1/specs/exp2.json --out $A/episodes.csv >> $L 2>&1
dpy scripts/rarf_ablation/timing_audit.py results/rarf_ablation_v1/exp1 results/rarf_ablation_v1/exp2 --out $A/timing.csv >> $L 2>&1
dpy scripts/rarf_ablation/analyze.py --episodes $A/episodes.csv --out $A >> $L 2>&1
dpy scripts/rarf_ablation/figures.py --episodes $A/episodes.csv --results $A/results.json \
    --budget output/research/rarf_ablation_v1/certified_budget.csv --out output/research/rarf_ablation_v1/figures >> $L 2>&1
echo "CHAIN_DONE $(date)" >> $L
