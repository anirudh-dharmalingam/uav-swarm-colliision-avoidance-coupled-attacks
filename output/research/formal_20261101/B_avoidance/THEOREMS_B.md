# Manuscript B — formal results for avoidance under coupled pursuit and report falsification

Written 2026-11-01. Statements are about the planar double-integrator abstraction used by every B controller (relative position
r = p_i − p_j, velocity v, honest command ‖a_i‖ ≤ U, neighbour acceleration a_j). Machine checks: Z3 5.3.3 (`z3_avoidance.py`,
`z3_avoidance_results.json`) and numerical experiments (`numeric_checks.py`, `numeric_checks_results.json`). They are not
guarantees about PX4 execution; each result names the experiment it explains.

Barrier (exponential HOCBF, Xiao & Belta 2019): h = ‖r‖² − D², ψ₁ = ḣ + k₁h, ψ₂ = ψ̇₁ + k₂ψ₁
= 2‖v‖² + 2rᵀ(a_i − a_j) + 2(k₁ + k₂)rᵀv + k₁k₂h. Robust row (RHOCBF): 2rᵀa_i + 2‖v‖² + 2(k₁ + k₂)rᵀv + k₁k₂h − 2‖r‖ā ≥ 0.

## Theorem B1 (robust HOCBF is sound with exact information)
*Assumptions.* Exact r, v; ‖a_j‖ ≤ ā; the robust row is feasible within ‖a_i‖ ≤ U at every instant (continuous time).
*Claim.* ψ₂ ≥ 0 at every instant; with h(0) ≥ 0 and ψ₁(0) ≥ 0, h(t) ≥ 0 for all t (forward invariance).
*Proof.* Pointwise: 2rᵀa_j ≤ 2‖r‖ā (Cauchy–Schwarz), so row ≥ 0 ⇒ ψ₂ ≥ 0 (Z3 B1, all reals). Integration: ψ̇₁ ≥ −k₂ψ₁ with
ψ₁(0) ≥ 0 gives ψ₁ ≥ 0, then ḣ ≥ −k₁h gives h ≥ 0 (comparison lemma). ∎

## Theorem B2 (report falsification removes the guarantee)
*Claim.* With reports r̂ = r + e_p, v̂ = v + e_v at the declared budget ‖e_p‖ ≤ 1.25 m, ‖e_v‖ ≤ 0.5 m/s, there are states in which
the robust row holds on the report while the true ψ₂ < 0 although ‖r‖ ≥ D. *Witness.* Z3 B2 (k₁ = k₂ = 1, D = 1.5, ā = 2.5,
r = (0, 3), e_p ≈ (0.38, 0.48), e_v ≈ (0.47, 0.13), ‖a_i‖ ≤ 3). ∎
*Use.* B1's premise (exact information) is exactly what the coupled attack violates; this is why RHOCBF, though the strongest arm
empirically, reaches contact under the worst-case adaptive attacker (B-A1, B-RZ WORST) — no theorem was ever on offer there. A
measurement-robust row (MRCBF: tightening by the receiver-age enclosure of the report error) restores soundness on paper but loses
feasibility and route completion (B-H1, B-N1: MRCBF deadlock).

## Theorem B3 (input-to-state safety of the soft QP, and its limits)
*(a) ISSf algebra.* If the soft row holds with slack ξ ≥ 0 and ‖a_j‖ ≤ ā, then ψ₂ ≥ −ξ (Z3 B3a). With ξ̄ = sup ξ, the comparison
lemma gives ψ₁ ≥ −ξ̄/k₂ and h ≥ −ξ̄/(k₁k₂), i.e. ‖r‖ ≥ √(max(D² − ξ̄/(k₁k₂), 0)): a class-K degradation γ(ξ̄) = ξ̄/(k₁k₂) of the
safe set (Kolathaya & Ames, ISSf).
*(b) Slack is forced when the neighbour can out-accelerate.* At the boundary (h = 0, ψ₁ = 0, v = 0) the row reads 2rᵀa_i ≥ 2‖r‖ā,
infeasible for ‖a_i‖ ≤ U whenever ā > U (Z3 B3b: no admissible a_i at ā = 5.5 = U + ε, U = 3). So under the declared pursuit
budget the hard robust row cannot be satisfied near contact, and some slack ξ ≥ 2‖r‖(ā − U) is unavoidable there.
*(c) The slack weight saturates.* The L1 penalty is not exact here: the round actuator set ‖u‖ ≤ U removes the weak-sharpness
that exactness needs (in 47 % of 400 random RARF-like instances the solution still moves above ρ = 1e3). It converges at about
one decade per decade of ρ: median distance to the ρ = 1e4 solution 4.5e-3, 2.2e-4, 2.5e-5 m/s² at ρ = 1e1, 1e2, 1e3 (90th
percentile 0.12, 0.011, 0.001). *Use.* Explains B-ISS: the empirical gain γ̂(ε) is class-K (I1 passed for every ρ) and clearance
stops changing beyond ρ ≈ 1e2 (I2 passed, saturation ≤ 0.02 m) — the frozen ρ = 1e3 sits in the saturated regime, so ρ is not a
useful design knob; the actuator bound is.

## Theorem B4 (blocking impossibility: safety and passage are incompatible)
*Setting.* The honest UAV must cross a cross-section of half-width w of its corridor (bottleneck, gap between formation slots,
corridor wall); the attacker reaches the centre of that section first (its reachable set covers it before the honest UAV arrives).
*Claim.* Every crossing point is within w of the attacker (Z3 B4a). If w < d_safe, no honest trajectory both crosses the section
and keeps separation ≥ d_safe (Z3 B4b): every avoidance law must give up either safety or route progress. ∎
*Numerical check.* Planar double integrators, tube half-width 3 m, honest UAV not allowed to run backwards along its route,
attacker starting 20 m ahead: capture (0.77 m rotor contact) in 30/30 trials with a_A = 5.5 > a_D = 3 and in 21/30 with equal
authority; the other 9 came within 0.78 m.
*Use.* Explains the B safety–progress trade-off: MRCBF keeps clearance by stalling (route completion → 0 at N = 16), the others keep
the route and reach contact under WORST / A15.

## Theorem B5 (pursuit regimes)
For the planar relative double integrator with acceleration bounds a_A (attacker) and a_D (defender) and speed caps V_A, V_D:
* (i) V_A > V_D ⇒ capture from any initial state (pure pursuit closes at ≥ V_A − V_D once the attacker is at top speed).
* (ii) No speed caps and a_A > a_D ⇒ capture (the relative system is a double integrator whose control authority a_A dominates
  the bounded matched disturbance a_D).
* (iii) V_A = V_D in free space ⇒ a defender that flees along the line of sight is never captured; capture then needs a blocking
  geometry (B4) or a mission constraint that forbids fleeing.
*Numerical check* (30 trials per cell): capture 100 % in (i) and (ii); flee-escape 100 % in (iii) (minimum distance 13 m); a
tube-confined defender fleeing forward from a pursuer behind also escapes — tube confinement alone is not enough (prediction
corrected after the first run; recorded in numeric_checks.py).
*Use.* The declared budget gives the attacker a_A = U + ε = 5.5 > U = 3 with equal airframe speed caps: the B experiments are in the
blocking regime (honest UAVs must pass the attacker's slot), which is why no tested law — published or ours — withstands the
worst-case adaptive pursuer within the declared budget.

## What is NOT guaranteed
* Sampled-data effects (20 Hz decisions, slew limits, PX4 tracking error W) are bounded only empirically; an SD-CBF margin
  (Breeden et al.) is not derived for the implemented rows.
* RARF/FF's best-estimate rows (β = 0) have no soundness theorem; B3 covers them only through the ISSf algebra with whatever slack
  the solver reports.
* Receiver-age propagation is a zeroth-order predictor; no control-barrier-functional bound is proved. B-DLY measures it
  (graceful within the declared 0.15 s bound).

## Reproduce
```
PYTHONPATH=<dir with z3-solver 5.3.3> python3 z3_avoidance.py
python3 numeric_checks.py                     # inside dronelab:research-v1.17 (cvxpy + Clarabel)
```
