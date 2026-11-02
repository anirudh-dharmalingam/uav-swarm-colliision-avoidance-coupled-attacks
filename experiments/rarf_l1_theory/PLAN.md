# B-ISS / B-DLY: experiments behind the control-theoretic framing of manuscript B (frozen before any confirmatory run)

Written 2026-11-01 about 00:08 IST. Programme: `output/research/transactions_program_20261030/PROGRAM.md`.
Origin: an external review corpus (2026-11-01) asks that manuscript B be framed through measurement-robust CBFs, input-to-state
safety (ISSf), sampled-data CBFs and control barrier functionals. Its baseline requests (ORCA in the main analysis, an MR-CBF,
safety barrier certificates, no speed-rule-only comparison) are already met (B-H1, B-PB). Two of its claims are testable and untested:
the soft policy's slack weight sets an ISSf gain, and receiver-age propagation acts as a delay predictor. Nine smoke episodes
(seed 19200 / 20200) checked the runner and are not used.

## 1. B-ISS (`scripts/rarf_l1/run_theory.py iss`)
The FF arm executes the best-estimate L1-slack soft QP at every decision (`ral26_core.least_violation_command`, ρ = 1e3 frozen); RARF
executes it when the hard rows are not certified. ISSf (Kolathaya & Ames, IEEE L-CSS 2019) predicts h_min ≥ h_0 − γ(‖d‖) with γ class-K,
and a slack weight that trades constraint violation against deviation from u_nom.
* Factors: ρ ∈ {1e1, 1e2, 1e3, 1e4, 1e5} × pursuit ε ∈ {0, 0.5, 1.5, 2.5, 3.5, 5.0} m/s² at the declared β_p = 1.25 m, β_v = 0.5 m/s;
  arms FF, RARF; B-S2 seeds 19200–19209 × g {2, 3}. 1,200 episodes. ρ is injected by rebinding the function (no source change).
* γ̂(ε) per (arm, ρ) = max over seed × geometry of (h_min(ε = 0) − h_min(ε))⁺; mean loss alongside.
* **I1.** γ̂ is non-decreasing in ε (tolerance 0.05 m) for every (arm, ρ): the empirical class-K shape.
* **I2.** For FF at ε ≥ 2.5, mean loss is non-increasing in ρ (tolerance 0.05 m) and saturates by ρ = 1e3 (|loss(1e5) − loss(1e3)| ≤ 0.1 m),
  because the actuator ball U binds. Paired contrasts ρ − 1e3 are reported with seed-block intervals.
* **I3.** ρ = 1e3, ε > 0 rows are bit-identical to the B-S2 rows of the same seed and geometry (checked on one smoke row).

## 2. B-DLY (`scripts/rarf_l1/run_theory.py dly`)
Receiver-age propagation p_j + s v_j is a zeroth-order predictor of the delayed neighbour state (the control-barrier-functional
reading, Molnar, Kiss, Ames & Orosz). The declared transit bound in every agent stays T_net_max = 0.15 s.
* Factors: one-way FANET delay ∈ {0, 0.05, 0.10, 0.20, 0.40} s, jitter 20 % of the delay, FANET drop limit raised to delay + 0.1 s;
  threat {A0, A14}; the eight B arms; g {0, 3}; seeds 20200–20209. 1,600 episodes.
* **D1.** Within the declared bound (delay ≤ 0.10 s) no arm's A14 viol15 rate rises by more than 0.10 over zero delay.
* **D2.** Honest–honest contacts under A0 are listed per delay and arm (beyond the declared bound included).
* **D3.** RHOCBF − RARF in min_sep_ha keeps its sign at every delay (interval per delay reported).
Breakpoints beyond the declared bound (0.20, 0.40 s) are reported as findings, not as failures of a claim.

## 3. Analysis (`scripts/rarf_l1/analyze_theory.py`)
Seed-block bootstrap, 10,000 resamples, numpy seed 20261119. Layer-1 only; the B-F0 fidelity rule applies to any arm-pair claim.
