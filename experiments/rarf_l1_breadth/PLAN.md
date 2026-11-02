# B-S1 / B-S2: Layer-1 attack-spectrum and budget sweeps (frozen before any breadth run)

Written 2026-10-30 about 23:30 IST. Programme: `output/research/transactions_program_20261030/PROGRAM.md` §3. No breadth episode has run.
Harness: `swarmsec/rarf_l1/` runs the unmodified PX4 agent classes; its plant comes from `plant_fit.json`, fitted on
rarf_plant_time_v3 logs. Specs: `design.py`, built with the unchanged PX4 builders and the FROZEN comparator gains of rarf_hocbf_v1.

## 1. Claim rule (binding)
A Layer-1 result may state an ordering of arms X and Y only if the B-F0 fidelity gate (`scripts/rarf_l1/fidelity.py`, rule in its
docstring) rates X − Y as AGREE or MAGNITUDE at every geometry where PX4 resolves it. Both PX4 sets count: rarf_plant_time_v3 (180)
and rarf_hocbf_v1 (252) when complete. Otherwise every statement about that pair is exploratory. Absolute clearances are never
transferred to PX4. The PX4 confirmation of selected corners (programme B-S2 PX4) is designed after this sweep, on fresh seeds.

## 2. Designs
* **spectrum (B-S1)**: 8 arms (NOM, SPD, RARF, FF, HOCBF, RHOCBF, ORCA, MRCBF) × 6 attacks × 3 geometries {0, 2, 3 m} × seeds
  19000–19019 = 2,880 episodes. The attacks use the existing attack_terms implementations: A0 benign, A8 pursuit only (ε 2.5),
  A4 report falsification only (β_p 1.25, β_v 0.5), A14 coupled, A15 full-authority pursuit + falsification, A5 replay (2 s).
* **budget (B-S2)**: A14 with pursuit ε ∈ {0.5, 1.5, 2.5, 3.5, 5.0} m/s² × falsification β_p ∈ {0, 0.6, 1.25, 2.0, 3.0} m
  (β_v = 0.4 β_p) × 8 arms × geometries {2, 3 m} × seeds 19200–19209 = 4,000 episodes. β_p > 1.25 and ε > 2.5 lie outside the
  declared bounds (β_p = 1.25, A_any = 8.8 requires U + ε ≤ 8.8). They test behaviour outside the assumptions.

## 3. Endpoints and analysis
Endpoints are the `scripts/coupled_safety/metrics.py` definitions (identical to PX4): min_sep_ha in the 20 s encounter window,
viol15, prox08, crash proxy (contact in Layer 1), route_complete, t_cross and min_sep_hh. Effort (RMS command) is computed when agent
logs are kept.
* Per attack and geometry: arm means with seed-bootstrap 95 % intervals (10,000 resamples, numpy seed 20261106), and paired contrasts
  of every arm against NOM and against FF.
* Budget sweep: for each arm, the worst-case clearance over scenarios at each (ε, β_p), and the **safe-budget frontier**, the set of
  (ε, β_p) where no scenario falls below 1.5 m. Reported against the declared-bound box.
* Ranking stability: Kendall τ of the arm ranking across attacks and budgets.
Predictions fixed now: the falsification-aware arms (RARF, FF, MRCBF) degrade less than NOM, HOCBF and ORCA as β_p grows. Under
A8 (pursuit only), RHOCBF ≥ HOCBF. ORCA fails under pursuit because the reciprocity premise is broken. These are exploratory
predictions, not confirmatory hypotheses.

## 4. Compute and host rule
About 6,900 episodes at roughly 15 s each, with up to 14 workers, which is about 2 h. They run only when no PX4 campaign is running.
