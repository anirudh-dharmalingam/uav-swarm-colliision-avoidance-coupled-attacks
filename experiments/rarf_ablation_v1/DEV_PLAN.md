# rarf_ablation_v1 — development-stage plan (written before any rarf_ablation_v1 flight)

Written 2026-10-28, before the first development episode was launched. Development outcomes may tune the speed comparator and the
sample size only, by the rules below. They are never pooled with confirmatory outcomes.

## Purpose of the development stage
1. **Instrumentation check.** Every new arm runs end-to-end on PX4/Gazebo. Branch columns are populated. Shadow-solve overhead is measured (dec_ms p50/p99, deadline misses), and RTF ≥ 0.95 holds with 2 workers.
2. **Speed-rule selection** among the three candidates fixed in `design.py:SPD_CANDIDATES` (no other candidates may be added after outcomes are seen).
3. **Variance estimate** of the primary paired contrast, used for sample size.

Roster: scenarios RA-N6-17000…17003 (fresh seeds) × geometries {0, 2, 3 m} × arms {NOM, RARF, FF, SPDa, SPDb, SPDc}, coupled attack, i.e. 72 episodes (`specs/dev.json`).

## Speed-rule selection rule (fixed now)
For each candidate, over its 12 development episodes:
* eligible if registered route completion holds in ≥ 11/12 episodes and there is no crash proxy;
* among eligible candidates choose the largest mean min_sep_ha (mean over the 12 episodes);
* if none is eligible, choose the one with the most route completions, with ties broken by the largest mean min_sep_ha.

The chosen rule's parameters are copied verbatim to `FROZEN_PLAN.json:speed_rule`. Confirmatory outcomes are never used to retune it.

## Sample-size rule (fixed now)
Primary contrast Δ₁ = scenario mean over the three geometries of min_sep_ha(RARF) − min_sep_ha(FF). Let s be the development SD of the four per-scenario Δ₁ values. That SD rests on only 4 scenarios, so the planning SD is s* = max(s, 0.10 m, SD of the per-scenario mean over the three geometries of RARF − NOM).

Using s*, the target is a 95 % CI half-width ≤ 0.10 m, i.e. n₁ = ⌈(1.96 s*/0.10)²⌉, and 80 % power at |δ| = 0.25 m (two-sided α = 0.05), i.e. n₂ = ⌈((1.96+0.84) s*/0.25)²⌉. N = clamp(max(n₁, n₂), 12, 30).

The upper clamp 30 is the compute budget: 21·30 = 630 episodes ≈ 8 h at the measured ~80 episodes/h with 2 workers. If N hits the clamp, the achieved precision is reported as below target.

## Provenance captured before launch
`PROVENANCE_PRE.json`: git HEAD, dirty-file list, sha256 of every runtime/analysis file used, docker image id, design/spec hashes.
