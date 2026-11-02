# Formal results, 2026-11-01

| id | statement (short) | kind | status |
|---|---|---|---|
| A1 | Byzantine quorum ⌊(M+f)/2⌋+1 ⇒ one leader per term under equivocation, all M, f | proof + Z3 + TLC (N ≤ 7) | proved / checked |
| A1-c | crash quorum ⌊M/2⌋+1 admits two leaders with one equivocator | Z3 witness + TLC trace | counterexample |
| A2 | honest quorum exists ⇔ M ≥ 3f + 1 | Z3 | proved |
| A3 | accepted leader has ≥ q − f honest eligibility votes; ineligible never admitted (q > f) | proof + TLC | proved / checked |
| A4 | energy ledger never flags an honest sender (incl. departure) | Z3, inductive | proved (original code: witness of the bug) |
| A5 | accepted claim ≤ max(c₀ − p_min Δt, 0); inflation ≤ (P_max − p_min)Δt | Z3, inductive | proved |
| A6 | bounded-authority tube (LE_METHOD Prop. 1), key inequality | Z3 (nonlinear) | proved |
| B1 | robust HOCBF row sound with exact information | Z3 + comparison lemma | proved |
| B2 | report falsification at the declared budget breaks B1 | Z3 witness | counterexample |
| B3a | soft row ⇒ ISSf, h ≥ −ξ̄/(k₁k₂) | Z3 + comparison lemma | proved |
| B3b | ā > U ⇒ robust row infeasible at the boundary (slack forced) | Z3 | proved (no witness) |
| B3c | L1 slack penalty not exact under ‖u‖ ≤ U; converges ~1 decade per decade of ρ | numerical (400 QPs) | measured |
| B4 | blocking: crossing a section of half-width w < d_safe held by the attacker ⇒ safety or passage fails | Z3 + simulation | proved / 30/30 capture |
| B5 | pursuit regimes (speed advantage, acceleration advantage without caps, equal speed in free space) | argument + simulation | supported |

Files: `A_admission/THEOREMS_A.md`, `A_admission/tla/`, `A_admission/z3_admission.py`; `B_avoidance/THEOREMS_B.md`,
`B_avoidance/z3_avoidance.py`, `B_avoidance/numeric_checks.py`, with result JSONs beside each script.
Tools: TLA+ tools v1.8.0 (sha256 011267fc…), z3-solver 5.3.3 (installed outside the repository), Java 17.
