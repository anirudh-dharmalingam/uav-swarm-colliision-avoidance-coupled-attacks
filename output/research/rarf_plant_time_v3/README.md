# rarf_plant_time_v3: deliverable index

| Deliverable | Location |
|---|---|
| Necessity decision | `output/research/paper_audit_v1/MINIMAL_REVISION_PLAN.md` (R-X2): D1 resolved without a new run; D2 / D3 require it while the causal attribution is kept |
| Frozen plan (text, amendments A1 and A2 made before any confirmatory flight) | `experiments/rarf_plant_time_v3/FROZEN_PLAN.md` |
| Frozen N, hashes, image, seeds | `experiments/rarf_plant_time_v3/FROZEN_PLAN.json` (`freeze.py verify` runs before and after the confirmatory batch) |
| Implementation | `controllers/swarm/cs_agent_node_v2.py` (`cs.clock = "plant"`, exec chain v1), `worlds/coupled_safety/yard_physics_rtf030.sdf`, `experiments/rarf_plant_time_v3/design.py`, `run_chain.sh` |
| Raw traces | `results/rarf_plant_time_v3/{smoke,pilot,pilot2,confirm}/<episode>/` (agents, eval/truth.csv, eval/clock_map.csv, controller logs) |
| Attempt ledgers | `output/research/rarf_plant_time_v3/<stage>/attempt_ledger.csv`, `results/rarf_plant_time_v3/<stage>_launch.log`, `_failed_attempts/` |
| Analysis (frozen) | `scripts/rarf_plant_time_v3/analyze.py` → `output/research/rarf_plant_time_v3/<stage>/{episodes.csv,results.json}` |
| Manuscript-ready results | `RESULTS.md` (this directory) |

Stages run: smoke (seed 17280, 2 episodes; it found the criterion-3 denominator bug, amendment A1); pilot 1 (17290–17293, RTF 0.5;
failed the deadline-miss criterion, amendment A2); pilot 2 (17294–17297, RTF 0.3; passed 48/48); confirmatory (17300–17314, N = 15).
Pilot and smoke outcomes are never pooled with the confirmatory ones, and no pilot contrast was computed.
