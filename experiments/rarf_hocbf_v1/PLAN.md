# rarf_hocbf_v1 — existing HOCBF comparators in the plant-time RARF harness (plan written before any HOCBF flight)

Written 2026-10-30 02:30 IST, before any rarf_hocbf_v1 flight. The tuning grid, the selection rule and the confirmatory sample-size
rule are fixed now. The gains chosen on development seeds and N are written to `FROZEN.json`, with source hashes, before the first
confirmatory flight. Historical campaigns and rarf_plant_time_v3 are not modified or relabelled.

## 1. What exists (audit, `output/research/rarf_hocbf_v1/HOCBF_AUDIT.md`)
1. **Plain HOCBF.** `swarmsec/safety.py:SafetyFilter.filter` (and the vectorised `swarmsec/le/fastqp.py` with the same rows). A
   second-order exponential CBF on h = ‖p_rel‖² − Ds² with a_j = 0, a squared-slack QP (OSQP), a box limit, then a norm clip.
2. **Acceleration-bound robust HOCBF.** The same filter with `accel_bound`: each row is tightened by ‖A_k‖·ā (hard or two-tier
   soft). It was used by D13 (constant ā) and D20 (residual-adaptive). It pads neighbour acceleration only. It does nothing about
   falsified reports, report age, transit delay, ego-estimate error or zero-order hold.
3. **Measurement-robust sampled-data HOCBF.** `output/research/coupled_safety_v1/MATH_AND_ASSUMPTIONS.md` §5: an SOCP over observation
   balls with a ZOH margin. It is **derived only, not implemented**, and is not part of this study.
No previous run used an HOCBF in this per-agent PX4 harness. The earlier HOCBF runs (Layer-1 simulator E9–E18; centralised
`swarmsec_node` SITL) differ in attack, information path, controller and geometry, so they are not comparable.

## 2. Arms (fresh matched blocks: same scenario seed, geometry, attack, plant schedule)
NOM (D0R), RARF (D24R_SH), FF (D24R_FF), **HOCBF** (plain), **RHOCBF** (acceleration-bound robust). Common to every arm: the nominal
law, received information (age-propagated packets), the plant-time schedule (cs_agent_node_v3 = v2 + HOCBF variants; RTF 0.3, 25 ms
deadline), attack A14 at the rarf_ablation_v1 budgets, geometry {0, 2, 3 m}, U = 3, slew 0.4 per tick, exec chain v1 with the node
speed projection. Adaptations for the HOCBF arms are in `swarmsec/ral26_hocbf.py` (planar rows, U as the box and norm limit, the
harness's common slew limiter after the QP, neighbour radius 12 m as in nominal consensus). The original implementations are
unchanged, and `tests/test_hocbf_comparator.py` checks that the adapter equals `SafetyFilter.filter`.

## 3. Development tuning (seeds 17400–17403 × {0, 2, 3 m}; never pooled)
Candidates (fixed now):
* plain: H1 Ds 2.0, k1 = k2 = 1.0; **H2 Ds 2.5, k1 = k2 = 2.0** (repo default); H3 Ds 3.0, k1 = k2 = 2.0. Slack penalty 1e4 (repo
  default).
* robust (repo default Ds 2.5, k1 = k2 = 2.0, existing two-tier soft robust tier robust_soft = 100): R1 ā = 2.5 m/s² (the attack's
  ε budget); R2 ā = 6.2 (declared honest-neighbour bound); R3 ā = 8.8 (declared any-neighbour bound).
Selection, per family, on its 12 development episodes (the rarf_ablation_v1 rule): eligible if route completion is ≥ 11/12 and no tilt
crash proxy occurs. Among eligible candidates, take the largest mean min_sep_ha. If none is eligible, take the most completions, with
ties broken by mean min_sep_ha. Gains are frozen after selection.

## 4. Confirmatory design
Fresh seeds 17500, …; complete blocks of 5 arms × 3 geometries in a seed-keyed random order, 3 workers.
* **Primary: RARF − RHOCBF** (geometry-averaged min_sep_ha, 95 %): the complete method against the strongest existing robust HOCBF
  comparator.
* Secondary family (Bonferroni, three contrasts, 98.33 %): RARF − HOCBF, FF − RHOCBF, RHOCBF − HOCBF.
* Descriptive: per-geometry contrasts, threshold events, crash proxy, route and dwell completion, effort, decision time, and the HOCBF
  audit (slack > 0 share, solver non-optimal, final-command row residual < 0 share, slew-modified share, deadline misses).
* N = clamp(⌈(1.96 s* / 0.10)²⌉, 12, 15), with s* = max(SD of the per-scenario primary contrast over the 4 development seeds using the
  selected gains, 0.10). **Compute cap 15 blocks** (225 episodes ≈ 4.7 h). If the rule exceeds the cap, the achieved precision is
  reported as below target.
* Instrument criteria, retries and missing-data rules are those of rarf_plant_time_v3 FROZEN_PLAN.md §5 and §8. The cluster bootstrap
  uses 10,000 resamples, numpy seed 20261101.

## 5. Interpretation limits fixed in advance
RHOCBF results say nothing about robustness to falsified reports or delay. They test only whether padding the neighbour-acceleration
term changes outcomes under this attack. Neither HOCBF arm is a published-method reproduction.

## Amendment A1 (2026-10-30 02:40 IST, during the development batch, before any development outcome was inspected)
The dev roster contains only the HOCBF candidates, so §4's planning SD (the SD of per-scenario RARF − RHOCBF on dev seeds) cannot be
computed. Replacement, using the bound SD(X − Y) ≤ SD(X) + SD(Y):
s* = max(SD of per-scenario geometry-mean min_sep_ha of the selected RHOCBF on the 4 dev seeds + SD of per-scenario geometry-mean
min_sep_ha of RARF in rarf_plant_time_v3 confirm (15 blocks), 0.10). N = clamp(⌈(1.96 s*/0.10)²⌉, 12, 15) is unchanged.

## Amendment A2 (2026-10-30 ~03:20 IST, user request to add ORCA and a robust CBF; before any dev2 or confirmatory flight)
Two further arms, both **new implementations** (`swarmsec/ral26_baselines2.py`, verified by `tests/test_baselines2.py`), flown by
`cs_agent_node_v4.py` (= v3 + these variants; the HOCBF and RARF paths are unchanged):
* **ORCA**: a Python port of RVO2's ORCA half-planes and LPs (no library available in the image). Interface:
  v_pref = v + 1.0 s · u_nom, then ORCA, then u = (v_new − v)/0.5 s, norm U, slew limit. Neighbour state comes from age-propagated
  packets. Its reciprocity premise does not hold against the pursuing attacker.
* **MRCBF**: the measurement-robust exponential HOCBF of coupled_safety_v1 MATH §5 (previously only derived), on the SAME
  receiver-age enclosure as RARF (β, estimation errors, age, transit, A_j; D = A_j + W). It is an SOCP with L1 slack (ρ), slew and
  norm. This is the only comparator that accounts for falsified reports and packet age. It is a CBF condition per tick, not a
  held-interval certificate.
Development stage 2 (seeds 17400–17403 × {0, 2, 3 m}; candidates fixed now; the selection rule is §3's):
* ORCA: O1 radius 2.0, τ 3 s, responsibility 0.5; O2 radius 2.5, τ 3 s, responsibility 1.0; O3 radius 3.0, τ 5 s, responsibility 1.0.
* MRCBF: M1 k1 = k2 = 1.0; M2 k1 = k2 = 2.0; M3 k1 = k2 = 0.5; d = 1.5 m, ρ = 1e3, η = 0.
Confirmatory arms: NOM, RARF, FF, HOCBF, RHOCBF, ORCA, MRCBF. Primary unchanged: **RARF − RHOCBF**. Secondary family (Bonferroni over
six, 99.17 %): RARF − HOCBF, FF − RHOCBF, RHOCBF − HOCBF, RARF − ORCA, RARF − MRCBF, FF − MRCBF. **Compute cap lowered to 12 blocks**
(7 arms × 3 geometries × 12 = 252 episodes ≈ 5.3 h). N = clamp(rule, 12, 12) = 12, and the achieved precision is reported. The
HOCBF-family dev stage ran on cs_agent_node_v3; the confirmatory run uses v4, whose HOCBF path is identical (subclass).
Diagnostics fix before freeze (logging only): `hc_res_final` for RHOCBF was computed on the robust-tightened rows. From the
confirmatory run on, it is reported on the plain rows, with `hc_res_rob` on the robust rows. Dev logs keep the old definition.

## Execution note, dev stage 1 (2026-10-30 03:55 IST)
Selected by the §3 rule: HOCBF = H3 (Ds 3.0, k 2/2), RHOCBF = R3 (ā 8.8, two-tier soft). Nine dev episodes (orders 4–14,
02:24–02:45) exceeded 0.1 % deadline misses (max 0.9 %) and ran at RTF 0.285–0.298. They overlapped operator test containers. Later
episodes had misses ≈ 0. Dev outcomes are used only for selection. The plain-row and robust-row residual split (A2) was applied after
this stage.

## Execution note, dev stage 2 (2026-10-30 05:26 IST)
Selected by the §3 rule: ORCA = O3 (radius 3.0 m, τ 5 s, responsibility 1.0; eligible: 12/12 routes, 0 crash, 3/12 sub-1.5 m);
MRCBF = M1 (k1 = k2 = 1.0). For MRCBF no candidate was eligible (routes 8/12, 8/12, 5/12; crash proxies 2, 4, 4), so the fallback rule
applied. MRCBF rows carried slack on about 48 % of window decisions, and the initial-condition premise held on ≤ 15 %. Dev2 ran
alongside the la_robust_v1 simulator screen (3 niced cores). Three dev2 episodes exceeded 0.1 % deadline misses. Selection uses
outcomes only.
