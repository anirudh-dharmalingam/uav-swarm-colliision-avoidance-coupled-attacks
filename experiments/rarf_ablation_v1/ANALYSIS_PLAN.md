# rarf_ablation_v1 — confirmatory analysis plan (frozen before any confirmatory flight)

Status: frozen together with `FROZEN_PLAN.json` (speed rule, N) and `FROZEN_CODE.json` (hashes). Development outcomes (seeds 17000–17003) informed only the speed rule and N, by the rules in `DEV_PLAN.md`, and are never pooled with confirmatory outcomes. The 770 coupled_safety_v1 episodes are untouched and are not re-analysed as confirmation.

## Question
Which executable part of RARF produces its clearance improvement under coupled malicious motion and authenticated false-data injection, and at what task cost?

## Arms (one honest-controller switch each; everything else identical)
| Label | Variant | What changes |
|---|---|---|
| Nominal formation control (NOM) | D0R | no avoidance layer; norm 3 m/s² + slew 0.4 m/s²/tick limits |
| Complete RARF (RARF) | D24R_SH | historical D24R command law (hard branch when certified, else best-estimate fallback, else max-reserve rescue); supervision off; shadow fallback solved every tick and never applied |
| Forced best-estimate fallback (FF) | D24R_FF | fallback (and rescue) selected at every decision; hard solve still run and logged, never applied |
| Nominal + speed restriction (SPD) | D0R_SPD | nominal plus proximity speed cap from received packets; rule frozen in FROZEN_PLAN.json |

**State updates.** In every arm the policy's previous-command state `u_prev` is the command the policy selected (after planar zeroing, before the node's speed-envelope projection). This is the historical bookkeeping, identical across arms, so the RARF and FF arms differ only in the selection switch. The node's 4 m/s speed-envelope projection is applied after the policy in all arms (historical behaviour). It is logged (`clip`) but not fed back into `u_prev`. Mode/supervisor state: relax, link refresh and energy return are off in all arms. The `mode` field is diagnostic only.

**What is matched.** Same scenario seeds and pre-drawn exogenous realisations in all arms: attacker identity, onset, route heading, spawn jitter, network key (link traces keyed on simulator time), attack budgets (ε = 2.5 m/s², β = 1.25 m / 0.5 m/s), trigger radius, world, bounds, formation contraction, actuator limits and the attacker's own D0R policy. The attacker is reactive, so its trajectory depends on the honest arm. Common seeds therefore do not make trajectories identical, and PX4 SITL is not bit-deterministic. Compute is matched between RARF and FF (both run the hard solve, the reserve solve and the fallback solve every tick). NOM and SPD run no solver; their decision time is reported.

## Design
* Experiment I: N scenarios (seeds 17100…) × geometries {0, 2, 3 m inward attacker-slot shift} × 4 arms, coupled attack (12N episodes).
* Experiment II: the same N scenarios × {benign A0, pursuit only, false data only} × {NOM, RARF, FF} at geometry 0 (9N episodes). The coupled cell is reused from Experiment I.
* The sampling unit is the scenario (complete matched block). Seeds are reused across geometries and attack cells, so every inference resamples whole scenarios (cluster bootstrap): 10,000 resamples, numpy seed 20261029, percentile intervals. Ticks and agents are never resampled.

## Endpoints (truth from the independent evaluator; definitions of coupled_safety_v1 PREREG §2 reused verbatim)
Primary per episode: min_sep_ha, the minimum 3-D centre distance between the designated UAV and any honest UAV over [onset, onset + 20 s].

Secondary per episode:
* min_sep_hh; 1.5 m violation; 0.8 m proximity; tilt > 60° crash proxy;
* registered route completion (ordered progress ≥ 23 m before 100 s, no crash); dwell-qualified completion (registered completion and ≥ 5 s contiguous in the goal region; separately named);
* ordered progress s_end, crossing time;
* RMS sent acceleration, maximum sent-command slew and count > 0.402 m/s²;
* decision time p50/p99, bytes, truth sample gap;
* branch tick counts.

The truth stream is sampled at about 50 Hz, so continuous-time minima are not bounded between samples, and the maximum gap is reported. Proximity proxies are not physical contacts.

## Contrasts and inference
* **Primary (single):** D1 = mean over the three geometries of [min_sep_ha(RARF) − min_sep_ha(FF)], with a 95 % CI. Interpretation: a CI including 0 means the hard-branch contribution is unresolved at this precision. It is not equivalence, because no equivalence margin was designed. A CI excluding 0 gives the sign and size of the branch contribution.
* **Secondary family** (Bonferroni, two contrasts, 97.5 % CIs), geometry-averaged min_sep_ha: D2 = FF − NOM (avoidance layer) and D3 = FF − SPD (simpler slowing policy).
* **Descriptive (95 % CIs, no multiplicity claim):** every contrast per geometry; RARF − NOM; paired risk differences and discordant counts for 1.5 m violations, 0.8 m proximity, crash, route and dwell completion; continuous task, effort and compute outcomes.
* **Experiment II:** per arm I = L_COUPLED − L_PHYS − L_REPORT + L_A0, with L = 1.5 − min_sep_ha, plus the paired difference I_RARF − I_FF and the cellwise benefits (arm − NOM) with 95 % CIs. A positive interaction quantifies non-additivity of harm. It does not identify an internal mechanism.
* **Task cost:** route and dwell completion are reported next to separation for every arm. Stopping is never credited as protection: an arm with lower completion is reported as trading task for clearance.
* No null-centred p-values are computed.

## Validity, retries, exclusions (outcome-independent; coupled_safety_v1 PREREG §5 rules)
* An episode without eval/summary.json is re-run once with the same seed, and the attempt is kept.
* **Deviation from coupled_safety_v1 §5, fixed before any confirmatory flight:** confirmatory episodes run with 2 workers and **no RTF re-run** (`rtf_rerun_below` = null). In development with 2 workers the filter arms ran at RTF 0.88–0.95 and the nominal arms at ≈ 0.98, so the re-run rule would have roughly doubled filter-arm cost without reaching 0.95. Every episode is flown once. Those with mean RTF < 0.95 are flagged and handled only through the timing-qualified sensitivity below, and the RTF difference between arms is reported as an execution difference (filter arms receive slightly more decisions per plant second).
* Crashes, failed landings, controller timeouts and fallback use are outcomes, never exclusions. An episode is analysable iff the truth stream covers the encounter window.
* Primary analysis uses complete blocks.
* Sensitivity analyses: non-analysable min_sep_ha set to 0, and timing-qualified blocks (all episodes in the block with RTF ≥ 0.95). Excluding slow runs may change the target population.
* All attempts are listed in the attempt ledger (index.jsonl, _failed_attempts/, _rtf_attempts/).

## Execution audit (Experiment III, new runs)
Per decision, overlapping flags are kept as logged:
* k0_fail (zero-time necessary condition);
* hard_ok / hard_why (solver or independent recheck);
* assumption validity;
* soft_ok (soft fallback validated);
* rescue (maximum-reserve command used);
* planar_chg;
* node speed-envelope clip;
* exec_rows_ok (the executed policy command satisfies the hard rows);
* final label.

Slew is computed both on the policy output and on the command actually sent. Timing comes from timing_audit.py: host-monotonic decision, state receipt and publication times; plant time from the evaluator clock map; plant command hold; and host-versus-plant packet age. Controller acceptance and physical application inside PX4 are not instrumentable without modifying PX4, and this is stated as such.

## Experiments IV and V
Not run. No practically useful certified-filter claim and no superiority-over-literature claim is retained. The analytic budget (`scripts/rarf_ablation/certified_budget.py`) shows why: under the declared bounds the implemented rows need ≥ 6.45 m static estimated clearance with standard information, ≥ 3.0 m even with perfect information against an 8.8 m/s² attacker authority.
