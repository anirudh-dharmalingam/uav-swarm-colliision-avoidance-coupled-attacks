# rarf_hocbf_v1 deviations (PLAN.md is hash-frozen; deviations are recorded here)

## Execution note, confirm interruption (2026-10-30 22:58 IST)
The host powered off at 09:20 IST, about 15 min into the confirmatory batch (9 completed episodes recorded in index.jsonl). One
in-flight episode (HCc-COUPLED-FF-g00-s17500, no eval/summary.json) is preserved unanalysed in confirm/_interrupted_attempts/ and is
re-flown from scratch. The frozen hashes verified OK before resuming; no code, spec or plan change. Completed episodes are kept by the
runner's resume rule. Outcomes of the 9 completed episodes were not inspected before resuming.

## Execution note, host power profile after the reboot (2026-10-30 23:20 IST)
After the 22:50 reboot the host came up in the `power-saver` platform profile (governor powersave, about 1.8 GHz). Every episode flown
after the 22:58 resume ran at RTF 0.23-0.27 instead of the locked 0.30 (wall 250-305 s instead of about 224 s); the 9 pre-power-off
episodes ran at 0.298-0.302. The batch was stopped at about 23:19. All 17 directories flown or in flight after the resume (10 completed, 3 killed
in flight, others started) are preserved unanalysed in confirm/_interrupted_attempts/power_saver_20261030/ together with the index
as it stood (index_full_at_stop.jsonl). Only their RTF and wall time were looked at; no outcome was inspected. The profile was set to
`performance` (cores about 4 GHz) and the batch resumed. The 9 pre-power-off episodes are kept. The la_robust_v1 simulator screen
(3 niced cores) runs alongside, as in the original chain.
- After the power-saver discovery the batch was resumed at about 23:20 under the performance profile; the first 3 episodes ran at RTF 0.299-0.302.

## Correction to the power-profile note (2026-10-30 23:30 IST, before any confirmatory outcome was inspected)
The frozen rules (rarf_plant_time_v3 FROZEN_PLAN §5 and §8, adopted by PLAN.md §4) allow no RTF re-run: an episode that completes
with a failed instrument criterion is KEPT, flagged and analysed in the primary population, and only an episode without
eval/summary.json gets one re-run. Re-flying the 10 completed power-saver episodes therefore departed from the frozen rule. It is
corrected as follows, fixed now:
* **Primary population:** for each of those 10 scenario-arm cells, the FIRST completed attempt (in
  _interrupted_attempts/power_saver_20261030/) is used, with its instrument flags (several are below the [0.25, 0.35] RTF band).
  The re-flown attempt is excluded from the primary.
* **Sensitivity (reported with equal prominence):** the same analysis with the re-flown attempts in place of the power-saver ones.
* The 3 episodes killed in flight at 23:19 and the 1 killed at the 09:20 power-off had no eval/summary.json. They take the runner's
  single re-run (§8), and the killed attempts stay in _interrupted_attempts/.
* The analysis script is not changed now. The population substitution will be implemented as a separate, versioned wrapper around
  scripts/rarf_hocbf_v1/analyze.py, with this note as its specification, before the confirmatory results are first read.

## Execution note, co-load from a Layer-1 replay (2026-10-30 23:23-23:27 IST)
A Layer-1 simulator replay (container rarf_l1_v3, 2 cores, nice 19; not part of this study) ran beside the batch for about 4 minutes,
in addition to the planned la_robust_v1 screen. Episodes completing 23:25 (RARF-g00, FF-g20, ORCA-g20 of seed 17500) ran at RTF
0.285-0.287 and had deadline misses of 1.41 %, 1.17 % and 0.24 % (criterion 2: < 0.1 %). Per §5/§8 they are kept and flagged, and
they enter the flagged-block sensitivity. No re-run. The replay was paused at 23:27 and stays paused until this batch ends.
Diagnostics only (RTF, misses) were read; no outcome was inspected.

## 2026-10-31 13:15 IST — analysis crash in the frozen binaries() helper
The primary-population run of `analyze.py confirm` (via `analyze_population.py`) raised `KeyError: 'SPD'` inside the frozen v3
`binaries()` helper, which hard-codes the pair FF−SPD; this campaign has no SPD arm (PLAN.md amendment A2). The frozen files are not
edited. The wrapper re-executes the same `binaries()` source with pairs whose arms are absent skipped; FF−NOM and RARF−FF are computed
exactly as written. No primary contrast or decision rule is affected. Recorded before the results were read.
