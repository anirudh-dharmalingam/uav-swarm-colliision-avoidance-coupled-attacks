# B-PB: coupled-attack avoidance against published multi-robot avoidance baselines (frozen before any confirmatory run)

Written 2026-10-31 about 23:45 IST. Programme: `output/research/transactions_program_20261030/PROGRAM.md`.
No B-PB run exists. Eight smoke episodes (seed 19400, g3) checked the integration and are not used.

## 0. Why
Manuscript B compares RARF/FF with HOCBF and ORCA (published families; ORCA is an RVO2 port) and two project robustifications
(RHOCBF, MRCBF). Two canonical multi-robot collision-avoidance families a T-ASE / TCST reviewer will expect are missing.

## 1. Baselines (`swarmsec/ral26_baselines3.py`, agent `controllers/swarm/cs_agent_node_v5.py`; source, equations and adaptation
documented there)
* **SBC** — safety barrier certificates, Wang, Ames & Egerstedt, IEEE T-RO 33(3), 2017: double-integrator pairwise barrier with
  acceleration limits, decentralised share of the pairwise constraint, QP on the nominal command.
* **BVC** — buffered Voronoi cells, Zhou, Wang, Bandyopadhyay & Schwager, IEEE RA-L 2(2), 2017: projection of the goal onto the
  buffered Voronoi cell and the go-to-goal law, under the same velocity-to-acceleration interface used for ORCA.
Both use the neighbour information every comparator uses (age-propagated received packets within 12 m) and neither models report
falsification or pursuit, as published. The 2-D QPs are solved exactly by active-set enumeration (checked against a brute-force grid
on 300 random instances: zero excess). Not adopted: W-MSR / Cavorsi et al. (RSS 2022) resilient consensus — it trims reported values
in a consensus update and cannot remove a pursuer as an obstacle; it is discussed, not run.

## 2. Design (`scripts/rarf_l1/run_pb.py`)
Each stage reuses the builder, seeds, budgets and base config (`coupled_safety_base.toml`) of the study it extends, so every new-arm
episode pairs with the existing arms by seed / geometry / scenario.
* **dev:** the rarf_hocbf_v1 tuning rule on its dev seeds (17400–17403 × g {0, 2, 3}, A14, declared budgets): eligible if routes ≥ 11/12
  and no crash; max mean min_sep_ha among eligible; else most routes, ties by min_sep_ha. Candidates fixed now:
  SBC S1 (Ds 2.5, γ 1, responsibility 0.5 = the paper's split), S2 (Ds 2.5, γ 1, 1.0), S3 (Ds 3.0, γ 1, 1.0);
  BVC V1 (r_s 0.75), V2 (1.0), V3 (1.5), with τ_pref = τ_goal = 1 s, k = 1, τ_acc = 0.5 s. 72 episodes; selection written to
  `output/research/rarf_l1_published/FROZEN_B3.json` by rule.
* **spectrum** (B-S1: 6 attacks × 3 geometries × seeds 19000–19019): 720. **budget** (B-S2: 5 ε × 5 β_p × g {2, 3} × seeds
  19200–19209): 1,000.
* **search / test** (B-A1: CEM, same family, budget, seeds 19300–19317 / select 19330–19333, objective; RNG seed 20261117; test on
  19400–19419 × g {0, 2, 3} with A14 and own θ*, plus every other θ* on the transfer cells, and the two new θ* against the eight
  existing arms): about 1,100 + 880.
* **rz** (B-RZ: 5 threats × 4 networks × g {0, 3} × seeds 20000–20009): 800. **scale** (B-N1: N 4–16, k 1–3, g {0, 3}, seeds
  19600–19609): 520.

## 3. Analysis (`scripts/rarf_l1/analyze_pb.py`)
Seed-block bootstrap, 10,000 resamples, numpy seed 20261118.
**Confirmatory (Holm over four):** P1 — under A14 at the declared budgets (spectrum stage, pooled over geometry), RHOCBF − SBC,
RHOCBF − BVC, RARF − SBC and RARF − BVC in min_sep_ha. No direction is predicted; each is reported with its sign.
**Pre-specified secondary:** route completion and crash per arm and attack; budget safe-cell frontier; adaptive own-θ* clearance and
worst case over every θ* (ten arms); resilience-loss contrasts under A14, A14_T and WORST_T; scale cells.
Fidelity: SBC and BVC have no PX4 runs yet; every B-PB claim is a Layer-1 claim until PX4 confirms it (the B-F0 rule).
A published baseline that beats RHOCBF or RARF on any endpoint is reported prominently.
