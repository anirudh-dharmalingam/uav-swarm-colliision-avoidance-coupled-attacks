#!/usr/bin/env python3
"""rarf_hocbf_v1 confirmatory / ORCA-MRCBF development agent: cs_agent_node_v3.CsAgentV3 (+ v2 plant clock; files unchanged) plus
    --variant ORCA | MRCBF      parameters from cfg["cs"]["b2"]  (swarmsec/ral26_baselines2.py)
All other variants behave exactly as in cs_agent_node_v3.  ORCA / MRCBF diagnostics are appended as b2_* columns.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cs_agent_node as cs  # noqa: E402
import cs_agent_node_v2 as v2  # noqa: E402
import cs_agent_node_v3 as v3  # noqa: E402
import px4_iface  # noqa: E402
import uav_agent_node as base  # noqa: E402
from swarmsec.ral26_baselines2 import B2_VARIANTS, Baseline2Policy  # noqa: E402

B2 = ["b2_rows", "b2_slack", "b2_status", "b2_viol", "b2_ic_ok", "b2_infeasible", "b2_vnew", "b2_ms", "b2_slew_mod", "b2_variant"]
LOG_FIELDS = v3.LOG_FIELDS + B2


class _W:
    def __init__(self, w, agent):
        self.w, self.agent = w, agent

    def writerow(self, row):
        pol = self.agent.policy
        if isinstance(pol, Baseline2Policy):
            row = {**row, **pol.diag}
        elif hasattr(pol, "diag"):
            row = {**row, **pol.diag, "hc_variant": pol.variant_label}
        return self.w.writerow(row)


class CsAgentV4(v3.CsAgentV3):
    def __init__(self, uid, cfg, geo, run_dir, variant, attack, attacker_id, clock=None):
        kind = variant.upper() if variant.upper() in B2_VARIANTS else None
        super().__init__(uid, cfg, geo, run_dir, "D0R" if kind else variant, attack, attacker_id, clock=clock)
        if kind and not self.designated:
            R = cfg.get("ral26", {})
            self.policy = Baseline2Policy(kind, self.B, cfg.get("cs", {}).get("b2", {}), vmax=self.vmax,
                                          static_age_s=float(R.get("static_age_s", 1.0)), nominal_age_s=float(R.get("nominal_age_s", 0.45)),
                                          info=self.info)
            self.event("b2_start", variant=kind, params=cfg.get("cs", {}).get("b2", {}))
        self.f.close()
        self.f = (run_dir / f"agent_{uid}.csv").open("w", newline="")
        self.w = _W(csv.DictWriter(self.f, fieldnames=LOG_FIELDS), self)
        self.w.w.writeheader()

    def close(self):
        if isinstance(self.policy, Baseline2Policy):
            self.event("b2_close", solver_fail=self.policy.solver_fail)
        super().close()


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser()
    for k in ("--id", "--config", "--geometry", "--run-dir"):
        ap.add_argument(k, required=True)
    ap.add_argument("--variant", default="D24R")
    ap.add_argument("--attack", default="A0")
    ap.add_argument("--attacker-id", type=int, default=-1)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    geo = json.loads(Path(a.geometry).read_text())
    clock = None
    if str(cfg.get("cs", {}).get("clock", "host")) == "plant":
        clock = v2.PlantClock(float(cfg.get("cs", {}).get("rtf_nominal", 0.5)))
        t_wait = v2._REAL_MONO()
        while not clock.ready() and v2._REAL_MONO() - t_wait < 60.0:
            time.sleep(0.05)
        sh = v2._shim(clock)
        base.time = sh
        cs.time = sh
        px4_iface.time = sh
    rclpy.init()
    node = CsAgentV4(int(a.id), cfg, geo, Path(a.run_dir) / "agents", a.variant, a.attack, a.attacker_id, clock=clock)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.002 if clock is not None else 0.1)
            if node.phase == "DONE" and node.land_cmd and base.time.monotonic() - node.t_phase > 1.0:
                break
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
