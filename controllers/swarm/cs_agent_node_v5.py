#!/usr/bin/env python3
"""B-PB agent: cs_agent_node_v4.CsAgentV4 (file unchanged) plus
    --variant SBC | BVC      parameters from cfg["cs"]["b3"]  (swarmsec/ral26_baselines3.py)
All other variants behave exactly as in cs_agent_node_v4.  SBC / BVC diagnostics use the v4 b2_* log columns.
Layer 1 uses it by binding cs_agent_node_v4.CsAgentV4 to CsAgentV5 before swarmsec.rarf_l1.harness.run_episode (scripts/rarf_l1/run_pb.py).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cs_agent_node_v4 as v4  # noqa: E402
from swarmsec.ral26_baselines3 import B3_VARIANTS, Baseline3Policy  # noqa: E402

_V4 = v4.CsAgentV4


class _W:
    def __init__(self, w, agent):
        self.w, self.agent = w, agent

    def writerow(self, row):
        pol = self.agent.policy
        if isinstance(pol, Baseline3Policy):
            row = {**row, **pol.diag}
            return self.w.writerow(row)
        return v4._W(self.w, self.agent).writerow(row)


class CsAgentV5(_V4):
    def __init__(self, uid, cfg, geo, run_dir, variant, attack, attacker_id, clock=None):
        kind = variant.upper() if variant.upper() in B3_VARIANTS else None
        super().__init__(uid, cfg, geo, run_dir, "D0R" if kind else variant, attack, attacker_id, clock=clock)
        if kind and not self.designated:
            R = cfg.get("ral26", {})
            self.policy = Baseline3Policy(kind, self.B, cfg.get("cs", {}).get("b3", {}), vmax=self.vmax,
                                          static_age_s=float(R.get("static_age_s", 1.0)), nominal_age_s=float(R.get("nominal_age_s", 0.45)),
                                          info=self.info)
            self.event("b3_start", variant=kind, params=cfg.get("cs", {}).get("b3", {}))
        if kind:
            self.f.close()
            self.f = (run_dir / f"agent_{uid}.csv").open("w", newline="")
            self.w = _W(csv.DictWriter(self.f, fieldnames=v4.LOG_FIELDS), self)
            self.w.w.writeheader()
