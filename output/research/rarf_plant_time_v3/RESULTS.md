# Timing-controlled confirmation of RARF arm contrasts (rarf_plant_time_v3): results

Confirmatory run 2026-10-29 18:38–22:19 IST, frozen plan `experiments/rarf_plant_time_v3/FROZEN_PLAN.md` (amendments A1 and A2 were made
before any confirmatory flight) and `FROZEN_PLAN.json` (N = 15, hashes verified before and after the batch). 15 fresh scenario blocks
(seeds 17300–17314) × {0, 2, 3 m} × {NOM, RARF, FF, SPD} = 180 episodes. **All 180 attempted, 180 analysable, 0 retries.** Full tables:
`confirm/RESULTS_TABLES.md`; per-episode data: `confirm/episodes.csv`; attempt ledger: `confirm/attempt_ledger.csv`.

## Execution instrument (validated on pilot 2 before outcomes; checked per episode)
Every arm ran on the same plant-time schedule:

| | NOM | RARF | FF | SPD |
|---|---|---|---|---|
| Achieved RTF, median (min–max) | 0.300 (0.299–0.301) | 0.300 (0.299–0.301) | 0.300 (0.299–0.301) | 0.300 (0.299–0.302) |
| Decisions per agent per plant second | 20.02 | 20.02 | 20.02 | 20.02 |
| Plant-time reference speed [m/s] | 1.000 | 1.000 | 1.000 | 1.000 |
| Attack onset − start_s [s], max | 0.048 | 0.049 | 0.050 | 0.050 |
| Deadline misses (held ticks) | 0 | 6 (max 0.22 % of one episode) | 1 | 0 |
| Publication slack p99 [ms] (deadline 25) | 1.5 | 7.0 | 6.9 | 1.6 |
| Plant command hold, max [s] | 0.064 | 0.081 | 0.069 | 0.069 |
| Decision time p99 [ms host] | 0.4 | 20.2–21.1 | 20.4–21.0 | 1.3 |

179/180 episodes pass every instrument criterion. One RARF episode (0.22 % misses) is flagged, and excluding its block changes no
conclusion (below). For comparison, Exp I's host-scheduled execution had reference speed 1.31–1.80 m/s and a 2.0–3.9 s attack lead,
both arm-dependent. **Execution placebo:** at the 0 m shift RARF selected the hard branch on 0 of 28,176 decisions and so executed the
FF command law. Its paired difference is now −0.009 m [−0.042, 0.024], against +0.147 [0.060, 0.253] in Exp I.

## Pre-specified contrasts (geometry-averaged minimum honest–designated separation; cluster bootstrap over 15 blocks)
| Contrast | Role | Estimate [CI] | Exp I (host schedule) |
|---|---|---|---|
| **D2 forced fallback − nominal** | primary, 95 % | **+0.840 m [0.788, 0.892]** | +0.840 [0.732, 0.973] (97.5 %) |
| D3 forced fallback − speed rule | secondary, Bonferroni 97.5 % | +0.533 [0.403, 0.686] | +0.363 [−0.093, 0.788] |
| D1 complete RARF − forced fallback | secondary, Bonferroni 97.5 % | −0.026 [−0.090, 0.033] (unresolved; not equivalence) | +0.008 [−0.097, 0.108] |
| RARF − nominal | descriptive, 95 % | +0.814 [0.760, 0.867] | +0.848 [0.704, 1.009] |

Sensitivities: flagged-block exclusion (14 blocks) gives D2 +0.839 [0.783, 0.896], D3 +0.546 [0.411, 0.703], D1 −0.022 [−0.089, 0.042].
The non-analysable := 0 m sensitivity is identical to the primary (no non-analysable episode). The D2 half-width (0.052 m) meets the
0.10 m target, and D3's 97.5 % half-width is 0.14 m. Per-scenario D2 values range from +0.649 to +1.015 m and are positive in 15/15
blocks. D3 values range from +0.18 to +1.17 m and are positive in 15/15. D1 is positive in 6/15.

## Per geometry (95 %)
| Shift | FF − NOM | FF − SPD | RARF − FF | RARF − NOM |
|---|---|---|---|---|
| 0 m | +0.022 [−0.032, 0.075] | +0.019 [−0.103, 0.142] | −0.009 [−0.042, 0.024] | +0.013 [−0.032, 0.052] |
| 2 m | +1.015 [0.962, 1.069] | +0.602 [0.416, 0.855] | −0.014 [−0.074, 0.042] | +1.001 [0.947, 1.056] |
| 3 m | +1.484 [1.341, 1.624] | +0.978 [0.842, 1.122] | −0.055 [−0.179, 0.054] | +1.429 [1.304, 1.551] |

| Shift | Arm | Mean / min sep [m] | < 1.5 m | < 0.8 m | Tilt crash | Route | Dwell |
|---|---|---|---|---|---|---|---|
| 0 m | NOM / RARF / FF / SPD | 2.78 / 2.79 / 2.80 / 2.79 (min ≥ 2.35) | 0 / 0 / 0 / 0 of 15 | 0 | 0 | 15 each | 15 each |
| 2 m | NOM | 1.215 / 1.048 | 15/15 | 0 | 0 | 15 | 15 |
| | RARF | 2.216 / 1.906 | 0/15 | 0 | 0 | 15 | 15 |
| | FF | 2.230 / 1.914 | 0/15 | 0 | 0 | 15 | 15 |
| | SPD | 1.628 / 0.323 | 2/15 | 2 | 2 | 13 | 13 |
| 3 m | NOM | 0.431 / 0.255 | 15/15 | 15 | 2 | 13 | 13 |
| | RARF | 1.860 / 1.380 | 1/15 | 0 | 0 | 13 | 12 |
| | FF | 1.915 / 1.345 | 1/15 | 0 | 0 | 13 | 11 |
| | SPD | 0.937 / 0.213 | 15/15 | 2 | 2 | 13 | 13 |

Paired 1.5 m risk differences: FF − NOM −1.00 (2 m; discordant 0/15) and −0.93 [−1.00, −0.80] (3 m; 0/14). FF − SPD −0.13 [−0.33, 0]
(2 m) and −0.93 [−1.00, −0.80] (3 m). RARF − FF 0 at every shift, with no discordant pair.

**Task and effort.** Goal-plane crossing time medians are 23.4–24.6 s in every arm, so no arm protected by stopping. Route completion at
3 m is 13/15 in every arm. Dwell-qualified completion at 3 m is 12/15 (RARF), 11/15 (FF), 13/15 (NOM, SPD). At 2–3 m the avoidance arms
use 3–4× the RMS acceleration of NOM (0.96–1.20 vs 0.26–0.40 m/s²) and reduce honest–honest clearance (median 3.9–4.1 vs 4.8 m; no
violation).

**Branch use (honest decisions, scored window).** RARF selected the hard branch on 0 / 19 / 45 decisions of about 28,160 per shift, all
labelled CERTIFIED_INTERVAL. Only 4 of those 64 selections changed the executed command by more than 10⁻³ m/s² relative to the shadow
fallback. Soft was used on every other decision, and rescue on 0 in every arm. FF's hard solve would have been feasible on 0 / 20 / 46
decisions. Exec chain v1 was flown unchanged, so post-projection slew jumps > 0.402 m/s² still occurred: RARF 3, FF 2, SPD 2, NOM 0
(see `rarf_exec_fix/EXECUTION_FIX_REPORT.md`).

## Manuscript-ready statements
* "Under an identical plant-time decision schedule (RTF 0.300 in every arm, 20 decisions per plant second, reference speed 1.000 m/s,
  deadline misses ≤ 0.22 % of any episode), forced best-estimate soft avoidance increased geometry-averaged minimum honest–designated
  separation relative to nominal formation control by 0.840 m (95 % CI 0.788–0.892; 15 scenario blocks). The gain was 1.015 m at the
  2 m shift and 1.484 m at the 3 m shift, and indistinguishable from zero in the original geometry (+0.022 m [−0.032, 0.075])."
* "Forced soft avoidance exceeded the frozen proximity speed rule by 0.533 m (Bonferroni 97.5 % CI 0.403–0.686). The speed rule crossed
  the 1.5 m threshold in 2/15 and 15/15 tight-geometry scenarios and produced four tilt-crash proxies; soft avoidance had 0/15 and 1/15
  and none."
* "Complete RARF and forced soft avoidance did not differ detectably (−0.026 m, 97.5 % CI −0.090 to 0.033). The hard branch was selected
  on 64 of about 84,500 encounter decisions and changed the executed command on 4. This is unresolved evidence of a small or absent
  hard-branch contribution in this regime, not a demonstration of equivalence."
* "At the 3 m shift both avoidance arms crossed the 1.5 m threshold in 1 of 15 scenarios. The historical 0/10 observation therefore does
  not generalise to every tight scenario."

## Limits
SITL only. Continuous-time minima are unresolved between 50 Hz truth samples. Proximity and tilt are proxies. Exec chain v1 was used
(the paper's controller), including its post-projection slew violations. PX4 application is unobserved. The plant clock is the Gazebo
stats stream extrapolated by host time: its offset to the evaluator map is 1.2–1.3 ms median, with jitter p99 ≤ 6.3 ms. The scope is
one route, one attack configuration (A14) and N = 6.
