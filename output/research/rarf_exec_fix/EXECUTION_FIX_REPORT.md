# RARF command-execution mismatch: diagnosis, correction and regression evidence

2026-10-29. No flights were run. Historical source snapshots, results and certification labels are unchanged. The correction is a
separately versioned variant ("exec chain v2"). `tests/test_exec_x2.py::test_historical_files_unchanged` checks that the frozen
`ral26_core.py`, `ral26_policy.py`, `ral26_cs.py`, `cs_agent_node.py` and `uav_agent_node.py` still match their rarf_ablation_v1 freeze
hashes.

## 1. Every recorded post-projection violation, reproduced from archived inputs
`scripts/rarf_exec_fix/replay_violations.py` scans every honest-agent MISSION tick of E1–E5, dev, Exp I and Exp II (about 3.4 M ticks). A
violation is a sent-command jump ‖s_k − s_{k−1}‖ > 0.402 m/s² (slew 0.4 + log-rounding tolerance), the definition used by the archived
audit. Outputs: `violations_replayed.csv`, `tick_bookkeeping.csv`, `replay_summary.json`.

| Set | Arm | Violations | Max sent jump | Max selected-command change | Max ‖u_prev − published‖ at k−1 |
|---|---|---|---|---|---|
| E1 | RARF / supervised | 3 / 10 | 1.40 / 3.00 | 0.400 / 0.401 | 0 |
| E3 | **nominal** | 5 | 2.97 | 0.400 | 0.64 |
| E5 | RARF | 274 | 3.02 | 0.401 | 3.00 |
| dev | speed rule | 2 | 1.22 | 0.401 | 0.82 |
| Exp I | RARF / FF / speed rule | 6 / 3 / 5 | 2.60 / 2.57 / 3.00 | 0.401 | 2.69 / 0 / 0.91 |
| Exp II | FF | 1 | 0.89 | 0.400 | 0 |
| **All** | | **309** | **3.02** | **0.401** | |

* All 287 archived witnesses (`command_jumps_all.json`) are matched one-to-one by run, agent and time. The 22 additional violations are
  5 in the historical **nominal** arm of E3 (excluded from the archived count, which the manuscript states) and 17 in rarf_ablation_v1.
* The v1 node projection (`cs_agent_node.py:211-215`) applied to the logged selected command and velocity reproduces every logged sent
  command within 1.3 × 10⁻³ m/s². The logs are rounded to 10⁻³.
* Cause, from the projection flag at k−1 and k: onset 137, release 78, both ticks projected 94. The policy output never changed by more
  than 0.401. Every violation is introduced downstream of the policy.
* For the speed-rule arm the policy is closed-form in logged fields. Recomputing it from (`unx`, `uny`, `spd_dmin`, v) reproduces the
  logged policy output within 1.6 × 10⁻³ in all 7 cases.
* **Hard-row violations after projection cannot be reproduced**: the rows depend on packets that are not logged. Three historical
  violations occurred on hard-certified ticks (E5 s16003, s16006, s16010, all at projection onset), and v1 relabelled each one
  `UNCERTIFIED_POSTCLIP`. Whether those projected commands also broke a hard row is unknown. In Exp I none of the 16 hard selections was
  projected.

## 2. The command chain
| Step | Code | Constraint checks | Logged as |
|---|---|---|---|
| nominal | `uav_agent_node.nominal`; norm-clipped to U in `AgentPolicy.step` | – | `unx, uny` |
| optimizer | `rc.solve_hard_safety` (hard QP), `rc.least_violation_command` (soft L1 QP), `rc.authority_reserve_m` (rescue) | rows, norm, slew around `u_prev`; independent `rc.recheck` | candidate vectors not logged (`dev_hard_fb` only) |
| fallback / rescue | `ral26_policy.py:236-255` | soft: finite / norm / slew; rescue: projected into the slew ball | `soft_ok`, `rescue`, `applied` |
| planarization | `ral26_policy.py:256-260` | – | `planar_chg` (0 in every logged tick) |
| previous-command store | `ral26_policy.py:261` **`self.u_prev = u`** | – | `ux, uy` (selected) |
| attack terms | `uav_agent_node.attack_terms` (identity for honest agents) | – | – |
| speed projection | `cs_agent_node.py:211-215`: if ‖v_xy‖ > 4 m/s and u·v > 0, remove the component along v | **none**: no slew, norm or row recheck | `clip`, `spx, spy` |
| relabel | `cs_agent_node.py:220-221`: CERTIFIED becomes UNCERTIFIED_POSTCLIP if projected | label only | `status` |
| publication | `px4_iface.accel_setpoint` → TrajectorySetpoint (acceleration, altitude hold) | – | `t_pub` (host) |
| PX4 application | offboard acceptance, attitude/thrust loops | – | **unavailable** |

## 3. The previous-command reference
The slew centre of every next-tick solve is the **selected** command, stored before the speed projection (`ral26_policy.py:261`). It is
not the command published to PX4. The mismatch occurs exactly on projected ticks: E5 2,357 of 69,614 (3.4 %), E1 supervised 36 of
770,370, Exp I 95 of 417,402 across arms (`tick_bookkeeping.csv`). Every projected tick shifts the next slew ball off the published
command, so the next published change can reach ‖u_sel,k − s_{k−1}‖ ≤ 0.4 + ‖u_sel,k−1 − s_{k−1}‖. This is the release and
both-ticks mechanism.

## 4. The correction (smallest change that enforces the constraints jointly)
New modules, all separately versioned:
* `swarmsec/ral26_exec.py`: execution set X = {u_z = 0, ‖u‖ ≤ U, ‖u − u_pub,prev‖ ≤ 0.4, nᵀu ≤ c}. While ‖v_xy‖ > v_max,
  n = v_xy/‖v_xy‖ and c = max(0, nᵀu_pub,prev − 0.4); otherwise the half-space is absent. This is the old speed envelope ("no outward
  acceleration beyond v_max") made slew-compatible. **X is never empty** (witness: u_pub,prev − min(nᵀu_pub,prev, 0.4)·n), so speed,
  norm and slew never conflict. The hard QP, the soft L1 QP (rows relaxed, X hard), the max-reserve rescue and a Euclidean projection
  are re-posed over X with independent rechecks.
* `swarmsec/ral26_policy_x2.py`: `ExecV2Policy` keeps the same command laws for NOM, SPD, RARF and FF. Every source is solved over X, and
  `finalize(u_pub)` revalidates the final command and commits it as the next slew centre.
* `controllers/swarm/cs_agent_node_v2.py` with `cs.exec_chain = "v2"`: no post-policy projection for honest agents. The designated
  UAV keeps the v1 chain, so the attack is identical across arms. The default `"v1"` reproduces the historical chain.
* Semantic change, stated: the outward acceleration is unwound at 0.4 m/s² per tick instead of being cut in one tick. The commanded speed
  can exceed 4 m/s by at most 0.64 m/s, plus the PX4 tracking residual.

## 5. Behaviour when the final command fails validation (`ral26_exec.validate_final`)
| Situation | Published command | Label |
|---|---|---|
| hard selected, final command passes X and every hard row, assumptions valid | as selected | `CERTIFIED_INTERVAL` |
| hard selected, final command fails a hard row | as selected (it lies in X) | `UNCERTIFIED_FINAL` (+ reason). The certified label is never retained |
| any source, final command outside X (solver tolerance or downstream change) | closest point of X (QP), else the witness | `EXEC_REPLACED` (uncertified) |
| soft fallback | as selected | `FALLBACK_SOFT`: uncertified, positive slack relaxes the geometry, never called safe |
| rescue | as selected | `FALLBACK_RESCUE`: uncertified |
| envelope premise flagged invalid | as above | `INVALID_ASSUMPTIONS` (precedence) |
| plant-time deadline missed (cs.clock = plant) | previous set-point re-sent | `DEADLINE_MISS_HELD` |

## 6. Verification
**Focused checks** (`tests/test_exec_x2.py`, 8/8 pass in the project image):
* `execution_set_never_empty`: 20,000 random (u_prev, v); the witness always lies in X.
* `saturation_nominal_and_filter`: ‖u_nom‖ ≈ 85 m/s²; published norm ≤ U and published slew ≤ 0.4 for NOM, RARF and FF.
* `speed_onset_and_release_v1_jumps_v2_does_not`: a held outward command while speed crosses 4 m/s up and down. v1 sends a jump
  > 2.5 m/s² (the archived failure); v2's maximum published change is 0.4, and the outward component unwinds monotonically.
* `branch_switching_random_published_slew`: 8 random episodes × {RARF, FF, SPD} with random speeds around v_max. The published slew is
  ≤ 0.4 on every tick, CERTIFIED appears only with the hard branch, and FF is never certified.
* `infeasible_rows_never_certified`: a zero-time infeasible row leads to `k0_infeasible`. In a case whose rows can only be met by
  outward acceleration, v2 refuses certification. v1 certifies there and its projection then breaks the row (residual < −10⁻³), which
  shows that the defect can invalidate a certificate. Soft and rescue commands lie in X.
* `validate_final_labels`: every row of the table in §5.
* `v2_equals_v1_when_envelope_inactive`: below v_max, v2 publishes the v1 nominal law exactly and the v1 RARF command within 10⁻³, with
  identical certified/uncertified decisions.
* `historical_files_unchanged`: frozen hashes hold.

**Archived violating cases.** Each of the 309 violations was replayed with X built around the logged published command and the logged
velocity:
* open-loop, one tick: the projection of the logged selected command onto X has a maximum published change of 0.4000001 m/s², and 309
  of 309 lie in X;
* exact, for the 7 speed-rule violations: the closed-form v2 command at that tick has a maximum change of 0.4000001.

The open-loop result shows only that X removes the recorded violation. The true closed-loop v2 trajectory, with re-solved QPs and
diverging earlier commands, is not recoverable from logs.

## 7. What is and is not established (reported separately)
1. **Command-constraint compliance.** v1 is broken: 309 published-command slew violations up to 3.02 m/s² per tick, with the slew centre
   off the published command on 0.02–3.4 % of ticks. v2 makes norm, slew, planarity and the speed envelope hold on every published
   command by construction, and this is verified by the tests and the replay. It is not yet verified in flight.
2. **Conditional certificate validity.** In v1 a certified label described the pre-projection command. The three certified-and-projected
   historical ticks were relabelled, but whether their published commands met the rows is unknowable. In v2 the label requires the final
   published command to pass every hard row and X. Proposition 1's other premises (enclosure, W, held command, plant timing) are
   unaffected by this fix and remain conditional.
3. **Publication timing.** Host decision-to-publish time p99 ≤ 0.11 s and max 0.29 s; plant hold p99 0.053–0.059 s and max 0.20 s (Exp I,
   AUDIT_REPORT §B4). v2 does not change timing. The plant-time variant (`cs.clock = plant`) adds a deadline and a hold rule.
4. **PX4 application.** Unobserved in every run. No log records when, or whether, PX4 accepted or applied a set-point. The physical
   acceleration that followed a command is visible only through PX4's estimated acceleration (`ax, ay`) and the truth stream, never as an
   application event.

No historical result, label or figure is changed. A campaign flying v2 was not launched.
