# rarf_plant_time_v3 — timing-controlled confirmation (plan frozen before the pilot)

Written 2026-10-29 (IST). This plan was written before any v3 smoke, pilot or confirmatory flight, and before any v3 outcome was
seen. It supersedes the prospective draft `experiments/rarf_plant_time_v2/PROTOCOL.md`, which is unexecuted and kept. The rules below are
fixed. Only N is set after the pilot, by §7, and is recorded with the source hashes in `FROZEN_PLAN.json` before the first confirmatory
flight. Historical (coupled_safety_v1) and rarf_ablation_v1 code paths and results are unchanged.

## 1. Why this experiment is needed (from `output/research/paper_audit_v1/AUDIT_REPORT.md` §B3–B4, not from v3 outcomes)
Confirmatory Exp I contrasts mix the avoidance law with arm-dependent execution. Every block is timing-unqualified (RTF 0.47–0.95). The
plant-time route reference runs at 1.31 m/s in NOM and 1.59–1.80 m/s in FF, against a nominal 1.0. The attack is already active for a
median 2.0 s (NOM) versus 3.9 s (FF) when the scored window opens. In the 0 m execution placebo, identical command laws differ by
+0.147 m [0.060, 0.253], and the D3 conclusion depends on the window definition. The manuscript's attribution of protection to soft
avoidance is therefore unresolved. The hard-branch contrast D1 is resolved at the command level by existing evidence (the hard selection
never changed the executed command) and is only secondary here.

## 2. Arms (rarf_ablation_v1 command laws, unchanged; exec chain v1)
A **NOM** nominal formation control (D0R). B **RARF** complete RARF (D24R_SH, shadow fallback solved every tick). C **FF** forced
best-estimate soft fallback (D24R_FF; hard solve run and logged, never applied). D **SPD** nominal + frozen proximity speed rule SPDb
(d_slow 5.0, d_stop 2.5, v_slow 0.8, v_near 0.0, k 2.0; from `rarf_ablation_v1/FROZEN_PLAN.json`).

## 3. Held identical across arms
World geometry and physics step (`yard_physics_rtf050.sdf` = `yard_physics.sdf` with only real_time_factor 0.5 and
real_time_update_rate 125). Also identical: route (scenario-drawn heading, 24 m, 1 m/s), squeeze schedule (on route progress), attack
A14 (ε 2.5 m/s², β 1.25 m / 0.5 m/s, trigger 9 m), attacker policy D0R refusing refresh, scenario-drawn attacker identity / onset /
spawn jitter / network key, information mode `standard`, declared bounds (`bounds_declared.json`), U = 3 m/s², slew 0.4 m/s² per tick,
v_max 4 m/s. Computational workload: RARF and FF run the same three solves every tick. NOM and SPD run none. Under a fixed plant
schedule this difference can only show up as deadline misses, which are logged per arm.

## 4. Plant-time schedule and clock definitions (controllers/swarm/cs_agent_node_v2.py, `cs.clock = "plant"`)
* **Plant clock.** The most recent `/world/default/stats` simTime sample, plus host-monotonic elapsed time × the RTF estimated from
  consecutive samples (EWMA 0.2), extrapolated ≤ 0.5 s and kept monotone. It is the same source the GCS uses for mission duration and
  its `t0_sim` start marker.
* **Mission zero** = GCS `t0_sim`. Agent mission time, the attacker's onset (`start_s`), route progress and the evaluator's scored window
  [start_s, start_s + 20 s] (truth stamped in plant time from the same marker) share one origin.
* **Decisions** are due at plant times t_k = k · 0.05 s, polled every 2 ms of host time. Route progress advances by the plant time
  between executed ticks.
* **Deadline.** The command for t_k must be published before plant time t_k + 0.025 s. Otherwise the tick is a MISS: the late result is
  discarded, the previous published set-point is re-sent (held), and the policy's previous-command state is restored. A whole tick
  passing without a decision is a `sched_skip`. There is no catch-up and no retry within an episode.
* **Receiver-local age** = plant time at decision − plant time stamped at packet receipt by the receiving process. Host-monotonic
  receipt is also kept, and the host age of the worst packet is logged (`worst_age_host`).
* **Clock mapping.** Each agent logs host and plant time at decision and at publication. The evaluator logs its own map of pose sim
  stamps to host monotonic time (`eval/clock_map.csv`). PX4-internal acceptance and application remain unobservable.
* Gazebo paces all arms at target RTF 0.5, so each 0.05 s plant tick has about 0.10 s of host time.

## 5. Instrument validity (checked on the pilot BEFORE any outcome is inspected; flagged per confirmatory episode)
Per episode (honest agents, MISSION phase, scored window):
1. achieved mean RTF (evaluator) in [0.45, 0.55];
2. deadline misses < 0.1 % and sched skips < 0.1 % of decisions;
3. decisions per agent per plant second in [19.8, 20.2];
4. plant-time route reference speed within ±1 % of 1.0 m/s;
5. logged attacker onset within one tick (0.05 s) of start_s on plant time;
6. agent plant-clock estimate versus the evaluator clock map, p99 |difference| reported. The map lags by the unmeasured transport, so
   this is descriptive only.

**Pilot gate.** If any of 1–5 fails in the pilot, only execution parameters may change: target RTF (0.5 → 0.4), deadline handling, or
worker count. The pilot is then repeated on new pilot seeds 17294–17297 before any confirmatory flight. Pilot outcomes are never pooled.
In the confirmatory run, episodes failing 1–5 are kept, flagged and analysed in a flagged-block sensitivity analysis. They are never
dropped from the primary analysis.

## 6. Endpoints (frozen coupled_safety_v1 / rarf_ablation_v1 definitions, `scripts/rarf_ablation/outcomes.py`, unchanged)
Primary per episode: min_sep_ha = minimum 3-D honest–designated distance over the scored window. Secondary: 1.5 m threshold events, 0.8 m
proximity, tilt > 60° crash proxy, registered route completion, dwell completion, RMS sent acceleration, decision time, branch use
(applied hard / soft / rescue, final labels, hard-changed-command), deadline misses, plant hold, reference speed, attack lead. The extended
window back to the logged plant onset is reported as a sensitivity.

## 7. Design, sample size and contrasts
* Unit = scenario (fresh seeds). Complete block = 3 geometries {0, 2, 3 m inward slot shift} × 4 arms = 12 episodes. Pilot seeds
  17290–17293 (implementation check and variance only). Confirmatory seeds 17300, 17301, …. Within a block the 12 episodes run in a
  seed-keyed random permutation, in adjacent pairs, on 3 workers.
* **Primary contrast: D2 = FF − NOM** (empirical avoidance contribution), geometry-averaged min_sep_ha, 95 % interval.
* **Secondary family (Bonferroni, 97.5 %):** D3 = FF − SPD (benefit beyond the simpler policy) and D1 = RARF − FF (added hard-branch
  contribution).
* Descriptive: every contrast per geometry (95 %), RARF − NOM, paired risk differences and discordant counts for every binary endpoint.
  Per-geometry results are always shown next to the geometry average.
* Inference: cluster bootstrap over scenario blocks (all arms and geometries of a seed resampled together), 10,000 resamples, numpy seed
  20261031, percentile intervals. No null-centred p-values. An interval containing 0 is reported as unresolved, not as equivalence.
* **Sample size (rule fixed now).** s* = max(s_pilot, 0.197, 0.10), where s_pilot is the SD of the four pilot per-scenario D2 values and
  0.197 m is the SD of per-scenario D2 in rarf_ablation_v1 Exp I. N = clamp(⌈(1.96 s* / 0.10)²⌉, 12, 20) for a 95 % half-width ≤ 0.10 m
  on D2. **Compute cap 20 blocks** (240 episodes ≈ 6 h at 3 workers). If the rule asks for more, N = 20 and the achieved precision is
  reported as below target. D3 cannot reach ±0.10 m within any feasible N (Exp I SD 0.705 m would need ≈ 190 blocks), so its interval
  is reported as obtained.

## 8. Retries, invalid runs, missing data (fixed now)
* One same-seed re-run of an episode that produces no `eval/summary.json` (runner rule). The first attempt is kept in `_failed_attempts/`.
  No RTF re-run, because RTF is fixed by design.
* Crashes, timeouts, deadline misses, fallback use and failed completion are outcomes, never exclusions.
* **Missing telemetry is distinct from controller failure.** An episode whose truth stream does not cover the scored window is
  non-analysable (telemetry missing). It enters the "non-analysable := 0 m" sensitivity, and the primary analysis uses complete blocks.
  A missing agent log affects branch and timing endpoints only and is reported as such.
* Every attempt is listed in the attempt ledger (`index.jsonl`, `_failed_attempts/`), with the order and slot of each.

## Amendment A1 (2026-10-29 17:02 IST, after the 2-episode smoke test, BEFORE the pilot; no outcome-based change)
Smoke seed 17280 (RARF and FF, 3 m shift; never pooled): RTF 0.500, 0 misses, 0 skips, reference speed 1.000 m/s, onset error 5–25 ms,
publication slack p99 5.6 ms, clock jitter p99 1.7 ms. Criterion 3 failed (17.2 decisions per agent-second) only because its denominator
was a fixed 20 s. Under the plant schedule the honest swarm reaches the goal at about 24 s and lands at about 29 s, inside the scored
window. **Criterion 3 is redefined as decisions per agent per second of MISSION-phase plant time inside the window.** The pass band
[19.8, 20.2] is unchanged. No endpoint, contrast, seed, arm or threshold changed. The smoke run's outcomes were not used for anything.

## Amendment A2 (2026-10-29 17:42 IST, after pilot 1 failed the instrument gate, BEFORE any confirmatory flight)
Pilot 1 (seeds 17290–17293, 48 episodes, RTF target 0.5, 3 workers) **failed criterion 2**. Deadline misses were 0.1–1.2 % (median
0.31 %) of decisions in the two solver arms (RARF, FF) and ≤ 0.05 % in NOM / SPD. Publication slack p99 was ≈ 21 ms against the 25 ms
plant deadline. One SPD episode failed criterion 5 by 0.1 ms (onset error 0.0501 s; the times are logged to 1 ms). Criteria 1, 3 and 4
passed in every episode: RTF 0.47–0.50, 19.97–20.05 decisions per agent-second, reference speed 1.000 m/s. Only the instrument summary was
inspected. No pilot contrast was computed or looked at, because the pilot mode of the analysis computes none.
Per §5, only execution parameters change:
* **target RTF 0.5 → 0.3** (`yard_physics_rtf030.sdf`: real_time_factor 0.3, real_time_update_rate 75, physics step unchanged). The
  25 ms plant deadline then leaves ≈ 83 ms of host time instead of ≈ 50 ms; 3 workers unchanged;
* criterion 1 band = target ± 0.05, i.e. [0.25, 0.35]; criterion 5 = within one tick + 2 ms of log rounding (0.052 s);
* pilot repeated on new seeds **17294–17297** (`pilot2`). Pilot 1 is kept and is never pooled;
* planning SD s* = max(s_pilot2, s_pilot1 = 0.181, 0.197, 0.10). The rule and clamp [12, 20] are otherwise unchanged.
If pilot 2 also fails criterion 2, the confirmatory run uses the last passing configuration or is not run, and the failure is reported.
The deadline, the hold rule, the arms, the endpoints and the contrasts are unchanged.
