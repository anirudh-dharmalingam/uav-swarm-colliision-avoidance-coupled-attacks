# Evidence map — RARF branch/speed ablation (rarf_ablation_v1)

**Single contribution these runs test.** In this PX4/Gazebo SITL swarm, which executable part of RARF produces its clearance improvement under coupled pursuit and authenticated false-data injection, and at what task cost? The candidates are the certified hard branch, the best-estimate fallback avoidance layer, and plain slowing. Every proposed run either answers that question or is dropped together with the claim it would support.

## 1. Reusable existing evidence (coupled_safety_v1, 770 audited SITL episodes, unchanged)
| Evidence | Use | Limit |
|---|---|---|
| E1 four-attack × four-method factorial (480) | Historical interaction 0.745 m [0.697, 0.790]; RARF − nominal 0.698 m (pursuit only), 0.002 m [−0.026, 0.027] (coupled) | Cannot separate the hard branch from the fallback |
| E3 shifts 0–3 m (80) | RARF 0/10 violations at 2 m and 3 m vs nominal 10/10 | Complete controller only; n = 10 |
| E2 information interventions (90) | Coverage rises without a detectable clearance gain | The oracle also changes attacker and nominal information |
| E4 (96), E5 (24) | Scope limits (N = 8 timing; beyond-budget stress) | Not re-run here; transfer stays restricted to the measured scope |
| Leader-election / resilience campaigns | Not used: different task, controllers and threat activation | Never pooled |

## 2. Recoverable from existing logs (done; no new flights)
| Analysis | Result | File |
|---|---|---|
| Source identity of the flown code | 46 of the 49 frozen files hash-identical. Of the other three, two are recorded amendments (`make_coupled_scene.py`, D8; `analyze_secondary.py`, D10). `experiment_manager.py` differs only by a later demo-only PX4 preflight watchdog in the `DS_STAGE=px4` path, which campaign episodes never take (diff vs the frozen snapshot: 4 hunks, all inside `hold_px4_stage`/`PreflightWatch`). The historical behaviour is therefore attributable to the frozen snapshot. | `experiments/rarf_ablation_v1/PROVENANCE_PRE.json` |
| Label partition, historical honest encounter ticks | E3 RARF: 55 certified / 77,445; 21,902 zero-time (k = 0) failures; 55,488 other hard-solve/recheck failures; 0 assumption-invalid; 0 post-solve relabels. E1 RARF: 0 / 229,975 certified. Soft fallback vs maximum-reserve rescue is **not logged historically**, so it stays unresolved. | `output/research/rarf_ablation_v1/historical/branch_accounting_summary.json` |
| The 13 primary slew witnesses | **13/13 explained by the node's post-policy speed-envelope projection.** The policy output changed by ≤ 0.401 m/s² (within the 0.4 slew ball) while the projection switched on at that tick (honest speed > 4 m/s). Across E1/E3/E4/E5, 287 sent-command jumps > 0.402 m/s²: all have policy-output change ≤ 0.402 and coincide with projection onset (119), release (75) or both ticks projected (93); 0 come from the fallback/rescue output. Cause: the projection is not slew-limited, and `u_prev` stores the pre-projection command. | `historical/command_jumps_all.json` |
| Certified operating budget (analytic, Exp IV) | Minimum estimated clearance for any feasible row set (no slew; necessary): standard information 6.45 m static (7.02 m at 1 m/s closing); known-honest neighbour 4.48 m; trusted-sensor-like 3.09 m; perfect information with 0.15 s transit 3.01 m; perfect information without delay 2.35 m. All exceed the observed encounter minima. | `certified_budget.csv` |

## 3. Essential new runs (this campaign)
| Run | Question | Design |
|---|---|---|
| Development (72) | Instrumentation, speed-rule selection, variance | 4 fresh scenarios × 3 geometries × {NOM, RARF, FF, SPDa/b/c} |
| Experiment I (12N) | RARF − FF (hard-branch contribution; primary); FF − NOM; FF − SPD | N fresh scenarios × {0, 2, 3 m} × 4 arms, coupled attack |
| Experiment III (no extra runs) | Branch flags, sent-versus-policy slew, host/plant timing | Logged in every Exp I/II episode |

## 4. Claim-dependent extensions
| Extension | Needed only if | Decision |
|---|---|---|
| Experiment II (9N): the factorial for NOM/RARF/FF | The interaction stays a main contribution | Run after Exp I; reuses Exp I's coupled g = 0 blocks |
| Experiment IV certified-regime flights | A practically useful certified filter is claimed | **Not run.** The analytic budget shows no premise-preserving regime at these spacings and authorities; the claim is dropped |
| Experiment V published comparator | Superiority over the literature is claimed | **Not run; claim dropped** |
| N = 8 rerun | A real-time scaling claim is kept | **Not run; scaling claim dropped** (E4 stays a scoped SITL observation) |
| Hardware | Hardware or deployment claims | Not run; no such claim |
