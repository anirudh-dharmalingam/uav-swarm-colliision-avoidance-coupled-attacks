# Audit of existing HOCBF implementations for use as RARF comparators

2026-10-30. The original implementations are not modified.

## 1. What is implemented
| Item | Location | What it does | What it does NOT do |
|---|---|---|---|
| **Plain HOCBF** | `swarmsec/safety.py:SafetyFilter.filter` (OSQP via cvxpy, `_CachedQP`) | Exponential second-order CBF on h = ‖p_rel‖² − Ds², ψ₁ = ḣ + k₁h, row 2p_relᵀu ≥ −2‖v_rel‖² − (k₁+k₂)ḣ − k₁k₂h. Neighbour acceleration assumed 0. Squared-slack penalty ρ = 1e4 (slack makes it always feasible, never certified). Box \|u_d\| ≤ u_max, then norm clip. k_max = 8 nearest neighbours within neighbor_radius | No slew constraint, no measurement or age model, no hard guarantee once slack > 0. The QP is warm-started across solves, so the answer depends on solve history (differences up to ≈ 0.07 m/s² observed; `tests/test_hocbf_comparator.py`) |
| Plain HOCBF, vectorised | `swarmsec/le/fastqp.py` (`hocbf_rows`, `solve`: projected semi-smooth Newton) | The same row family, unit-normalised, all agents at once (leader-election study, N ≤ 64) | No acceleration bound, no slew limit. Its tests check it against the OSQP filter |
| **Acceleration-bound robust HOCBF** | `SafetyFilter.filter(..., accel_bound=ā, robust_soft=ρ_r)` | Worst case of the neighbour-acceleration term: each row is tightened by ‖A_k‖·ā_j, either hard-penalised (ρ) or as a two-tier layer (plain rows ρ, robust rows soft ρ_r). Used by D13 (constant ā) and D20 (residual-adaptive ā) | It pads neighbour **acceleration only**. It does not model falsified position/velocity reports, receiver age, transit delay, ego-estimate error, zero-order hold or actuator lag |
| Measurement-robust sampled-data HOCBF | `output/research/coupled_safety_v1/MATH_AND_ASSUMPTIONS.md` §5 | Derived lower bound over observation balls (R_p, R_v), ZOH margin η_sd, SOCP | **Not implemented and never run** (coupled_safety_v1 DEVIATIONS D7). It is not used here, and nothing may describe it as implemented |

The Layer-2 node `controllers/swarm/swarmsec_node.py` runs one centralised process. It builds the defence decision (`DefenseDecision.accel_bound`,
`robust_soft`) and calls `SafetyFilter.filter` for all agents on per-receiver views. It is a different harness from the per-agent RAL26 /
RARF one (`uav_agent_node` / `cs_agent_node`).

## 2. Existing runs and comparability
* Layer-1 simulator campaigns (E9–E18; D12, D13, D17, D20, …) use the robust HOCBF under the A10 / A12 / A11 attacks in a Python
  double-integrator simulator, with neither packets nor PX4.
* Layer-2 SITL stack runs (`swarmsec_node`, `docs/RESILIENCE_STACK.md`, report S7–S12) use a centralised controller, different
  attacks, a different geometry and a different information path.
* coupled_safety_v1, rarf_ablation_v1 and rarf_plant_time_v3 contain **no** HOCBF arm. The robust HOCBF comparator was explicitly not
  run (D7).
**None of the existing HOCBF runs is comparable to the RARF experiments.** Their numbers must not be placed in RARF result tables. No
historical experiment is relabelled.

## 3. Which implementation integrates most faithfully
`swarmsec/safety.py:SafetyFilter` is chosen. It is the only implementation with both the plain and the acceleration-bound variants, and
it is what D13 and D20 actually ran. `fastqp` reproduces only the plain rows, with a different solver. The adapter
(`swarmsec/ral26_hocbf.py:filter_one`) runs SafetyFilter's per-agent body for the ego agent only, calling its unmodified private row
builder and cached QPs. With fresh solver instances it equals `SafetyFilter.filter` exactly (600 random cases, max difference 0).

## 4. Harness adaptations (identical across arms; `swarmsec/ral26_hocbf.py` docstring)
* Information: PX4 ego estimate plus received packets propagated by receiver-local age. This is the same information the nominal law,
  the speed rule and RARF receive, and the attacker's falsified reports enter the HOCBF rows unchanged.
* Rows on the horizontal relative state (planar command, PX4 altitude hold), as RARF's rows are.
* u_max = U = 3 m/s². Then the harness's common slew limiter (0.4 m/s² per tick, the same function as the nominal arm), planar zeroing
  and the node's v1 speed-envelope projection. The downstream slew limit can modify the QP output, so each tick's final command is
  re-evaluated against the HOCBF rows and the result is reported.
* Neighbour radius 12 m, as in the nominal consensus gate; k_max = 8.
* Plant-time schedule and deadline: cs_agent_node_v3 (= v2 plus these variants), RTF 0.3, 25 ms deadline.
