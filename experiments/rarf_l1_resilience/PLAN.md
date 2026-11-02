# B-RZ: resilience of coupled-attack avoidance under network degradation and transient attacks (frozen before any confirmatory run)

Written 2026-10-31 about 20:25 IST. Programme: `output/research/transactions_program_20261030/PROGRAM.md` (resilience focus).
Smoke episodes (seed 20000, 20001) checked the runner and are not used. An in-mission wind factor was built and dropped before
freezing: the Layer-1 OU disturbance has no PX4 calibration, and with it every arm drifted more than 100 m and crashed. Wind is left
to PX4.

## 0. Resilience definition used by manuscript B
Performance is honest safety margin over time: d_ha(t), the minimum true distance between the attacker and any honest UAV, and
d_hh(t), the honest–honest minimum (`scripts/rarf_l1/metrics_rz.py`, true positions only). Encounter window W = [onset, onset + 20 s].
* **absorb:** min_sep_ha over W; t_below_safe = time in W with d_ha < 2.5 m;
* **resilience loss:** sep_deficit_ha = ∫_W max(0, 2.5 − d_ha) dt [m·s], and sep_deficit_hh = ∫ max(0, 1.5 − d_hh) dt;
* **recover** (transient attacks only): t_rec_ha, the first time after the attack ends with d_ha ≥ 2.5 m held 2 s (∞ if never);
* **outcome:** contact (crash), route_complete, prog_rate_post (honest route-progress rate in W relative to before onset).

## 1. Design (`scripts/rarf_l1/run_rz.py`)
Episodes come from the frozen rarf_hocbf_v1 / v3 builders (8 arms with the frozen comparator gains, declared budgets ε = 2.5,
β_p = 1.25, β_v = 0.5). Factors:
* **Threat (5):** A0; A14 (persistent); A14_T (A14 switched off 8 s after onset); WORST (θ*(HOCBF) from B-A1, the attack that drove
  every non-MRCBF arm to contact in the B-A1 transfer matrix; persistent); WORST_T (stops 8 s after onset). Every attack runs
  through `swarmsec/rarf_l1/adaptive.make_hook` (A14_THETA is bit-identical to A14).
* **Network (4):** nominal; degraded (10 % loss, Gilbert–Elliott bursts, 20 ± 10 ms delay); severe (25 % loss, stronger bursts,
  40 ± 10 ms delay against T_net_max = 50 ms); blackout (nominal plus every link down for 4 s from 2 s after onset).
* **Arms (8):** NOM, SPD, RARF, FF, HOCBF, RHOCBF, ORCA, MRCBF. Geometries {0, 3}. Seeds 20000–20009. 3,200 episodes.

## 2. Analysis (`scripts/rarf_l1/analyze_rz.py`)
Seed-block bootstrap (blocks = seeds, geometries inside a block), 10,000 resamples, numpy seed 20261114. The B-F0 fidelity rule applies in the report (not in the script):
claims about pairs B-F0 rated DISAGREE (RARF:HOCBF and RARF:NOM at g0) and about MRCBF's absolute
clearance (MAGNITUDE, overstated by about 1 m) are exploratory.
**Confirmatory test Q1 (one test, no multiplicity correction); Q2 and Q3 are pre-specified pass/fail criteria:**
* **Q1.** Under A14 pooled over the four networks, RHOCBF − RARF in sep_deficit_ha < 0 (the PX4-confirmed ordering holds as a
  resilience-loss ordering under network degradation).
* **Q2.** Under A14_T, RHOCBF's t_rec_ha is finite in ≥ 95 % of episodes in every network level (one-sided binomial lower bound
  reported) — recovery after the attack stops.
* **Q3.** Under A0, no arm has an honest–honest contact in the degraded network level (benign degradation is absorbed); every
  contact in any level is listed individually.
**Further predictions (reported either way):**
* **Q4.** Under WORST (persistent), every arm except MRCBF reaches contact in ≥ 50 % of episodes (B-A1 replicated under a fresh
  seed range: no tested avoidance law withstands the worst-case adaptive pursuer within the declared budget).
* **Q5.** Under WORST_T, the recovery ranking (share of finite t_rec_ha) differs from the absorb ranking (min_sep_ha): reported as
  Kendall τ between the two.
* **Q6.** Degradation is graceful: for RHOCBF and RARF, the A14 sep_deficit_ha rises monotonically nominal → degraded → severe,
  with no step larger than the nominal value + 2 m·s.
A failed prediction is reported as a finding. Exploratory: prog_rate_post as the task cost of resilience; blackout against severe.

## Pre-run correction (2026-10-31 23:43 IST, before any B-RZ episode ran)
`scripts/rarf_l1/run_rz.py` resolved episodes against `config/swarm/ral26_base.toml` (bottleneck scenario, 45 s, uav_agent script).
Every other B study (B-H1 builders, B-S1/S2, B-A1, B-N1) uses `config/swarm/coupled_safety_base.toml` (transit scenario, 100 s,
cs agent, sound-age envelope). The runner now uses the coupled-safety base; the smoke episodes above ran on the wrong base and are
not used. Design, endpoints and analysis are unchanged. Found while building the B-PB runner.
