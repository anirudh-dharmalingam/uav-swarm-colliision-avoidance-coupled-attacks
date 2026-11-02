# B-A1 / B-N1: defence-aware adaptive attacker and swarm scale, Layer 1 (frozen before any run)

Written 2026-10-31 about 13:15 IST. Programme: `output/research/transactions_program_20261030/PROGRAM.md` §3 (B-A1, B-N1).
No B-A1 or B-N1 episode has run. The PX4 result already known when this was written is the rarf_hocbf_v1 primary-population
analysis: RARF − RHOCBF = −1.41 m [−1.53, −1.28] and RARF − MRCBF = −3.42 m [−4.05, −2.69] (min_sep_ha). The reason for B-A1 is
therefore no longer "does RARF survive adaptation" alone. It is whether the ranking of all eight arms holds when the attacker adapts
to each arm separately. Fixed-attack rankings are known to overstate defences.

Harness: `swarmsec/rarf_l1/` (B-F0 fidelity gate: `output/research/rarf_l1/fidelity_v3`, `fidelity_hocbf`). Runner:
`scripts/rarf_l1/run_l1x.py` (`run_l1.py` plus per-episode `l1x` options). Builders: `design.py` here, on top of the frozen breadth
builder. Episodes use attack A14 at the declared budgets ε 2.5 m/s², β_p 1.25 m, β_v 0.5 m/s. The arms are NOM, SPD, RARF, FF,
HOCBF, RHOCBF, ORCA and MRCBF, with the frozen gains.

## 1. B-A1 adaptive attacker
**Policy family.** `swarmsec/rarf_l1/adaptive.py` replaces `attack_terms` on the compromised UAV only. It has 9 parameters:
* target rule (nearest, or smallest predicted time to closest approach);
* lead horizon;
* blend between the A14 push and A15 full-authority pursuit;
* pursuit trigger radius;
* lie onset radius;
* position-lie direction and magnitude;
* velocity-lie direction and magnitude.

Every member respects ‖u‖ ≤ U + ε, ‖rep_p − p‖ ≤ β_p and ‖rep_v − v‖ ≤ β_v. The attacker is white-box with respect to the method
design and black-box with respect to other UAVs' state. `A14_THETA` reproduces A14 bit-identically in Layer 1. This was checked
before freezing on 3 spectrum episodes: NOM g2 s19000, RHOCBF g2 s19000 and FF g3 s19001.

**Search.** `scripts/rarf_l1/adaptive_search.py search` runs a cross-entropy method per arm:
* 14 candidates, 6 iterations, elite 4, smoothing 0.7, initial SD 0.3 in the unit cube;
* candidate 0 is the current mean and candidate 1 is A14, every iteration;
* each candidate flies 3 seeds × geometries {0, 3} m, with seeds rotating per iteration (19300–19317);
* the attacker minimises mean min_sep_ha.

**Selection.** The 3 best candidates per arm are re-flown on seeds 19330–19333 × {0, 2, 3} m. θ*(arm) is the lowest mean there.
The RNG seed is 20261108.

**Test** (held out; no test seed is used in the search):
* every arm under A14 and under its own θ*, on seeds 19400–19419 × {0, 2, 3} m;
* transfer: θ*(src) against every other arm, on 19400–19409 × {2, 3} m.

**Endpoints.** min_sep_ha, viol15, prox08, contact, route_complete and min_sep_hh (metrics.py definitions).

**Analysis.** Per arm, the paired difference adaptive − A14 (seed-block bootstrap, 10,000 resamples, numpy seed 20261109).
* **Adaptive ranking:** the arm order by mean min_sep_ha under each arm's own θ*.
* **Worst-case ranking:** for each arm, the minimum over all θ* flown against it (own and transfer), as a per-episode mean.
* Kendall τ between the A14 ranking and the adaptive ranking.

**Predictions** (exploratory; Layer-1 claim rule of `experiments/rarf_l1_breadth/PLAN.md` §1 applies):
* **P-A1.** Every arm loses clearance under its own θ*. The adaptive − A14 interval excludes 0 for at least the four arms with avoidance
  (RARF, FF, HOCBF, RHOCBF).
* **P-A2.** Lateral or late lies (φ_p ≠ 0 or r_lie < r_trig) hurt the arms that trust reports (NOM, HOCBF, ORCA, MRCBF) more than the
  arms that discount them (RARF, FF, RHOCBF).
* **P-A3.** The adaptive ranking keeps RHOCBF above RARF. This is the PX4-confirmed fixed-attack ordering.

A failed prediction is reported as a finding. Layer-1 orderings for pairs that B-F0 rates DISAGREE are exploratory only:
* RARF − HOCBF and RARF − NOM at g0;
* every MRCBF contrast at g0 and g2 (MAGNITUDE), whose absolute clearance Layer 1 overstates by about 1 m.

## 2. B-N1 scale and multiple attackers
**Design.** `design.py scale`:
* N ∈ {4, 6, 9, 12, 16}, using the unchanged scenario generator `scenarios.scenario("RA", seed, N)`;
* coalition size k ∈ {1, 2} for N ≤ 6 and {1, 2, 3} for N ≥ 9;
* co-attackers are drawn per (seed, N, k), identical across arms, and all fly the A14 law; co-attackers never target each other;
* geometries {0, 3} m (the primary attacker's slot shift), seeds 19600–19609, all 8 arms;
* 2,080 episodes in total.

**Endpoints.**
* min_sep_ha_any: any attacker to any honest UAV in the encounter window (`scripts/rarf_l1/metrics_multi.py`);
* viol15_any, prox08_any;
* min_sep_hh_H: honest pairs only;
* route_H, contact and decision effort.

**Analysis.**
* Per (N, k, arm): means with seed-bootstrap intervals.
* Slopes of min_sep_ha_any in N and in k per arm (OLS over cell means, with bootstrap intervals).
* Paired arm contrasts RARF − RHOCBF and RARF − FF at each (N, k).

**Predictions** (exploratory):
* **P-N1.** For N = 4–6, k = 1, clearance changes by less than 0.3 m relative to the PX4-matched N = 6.
* **P-N2.** With k ≥ 2, the arms that keep a single-attacker assumption in their filter (RHOCBF, MRCBF, which are robust to one
  disturbance per neighbour) lose more than FF/RARF.
* **P-N3.** honest–honest separation, min_sep_hh_H, falls with N for every arm because of denser lattices. No B-N1 result transfers to
  PX4 without the PX4 B-N1 stage (programme §5).

## 3. Compute and host rule
* B-A1: about 6,500 episodes.
* B-N1: 2,080 episodes.
* Both run at 14 workers and only while no PX4 campaign runs. Chain: `scripts/run_chain_20261031b.sh`.
