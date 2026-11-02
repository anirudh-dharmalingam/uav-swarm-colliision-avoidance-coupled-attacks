#!/usr/bin/env python3
"""B-RZ resilience matrix for coupled-attack avoidance (experiments/rarf_l1_resilience/PLAN.md).  Inside the image:

    python3 scripts/rarf_l1/run_rz.py --workers 14

Episodes come from experiments/rarf_l1_adaptive/design.py `ep` (the frozen rarf_hocbf_v1 / v3 builders and comparator gains, declared
budgets ε = 2.5, β_p = 1.25, β_v = 0.5).  Factors:
  threat       A0; A14 (persistent); A14_T (stops 8 s after onset); WORST (θ*(HOCBF) from B-A1, the attack that drove every
               non-MRCBF arm to contact in the transfer matrix; persistent); WORST_T (stops 8 s after onset)
  network      nominal (base FANET: no loss, T_net_max 50 ms); degraded (10 % loss, Gilbert-Elliott bursts p_gb 0.05 / p_bg 0.3,
               20 ms delay, 10 ms jitter); severe (25 % loss, p_gb 0.10, 40 ms delay, 10 ms jitter: many deliveries exceed
               T_net_max and are dropped); blackout (nominal, plus every link down for 4 s from 2 s after onset)
  arm × geometry {0, 3} × seeds 20000-20009.  3,200 episodes.
Wind is NOT in this matrix: the Layer-1 OU disturbance has no PX4 calibration (docs/RAL26_PX4_CALIBRATION.md covers no wind), and
smoke episodes with it drifted > 100 m and crashed for every arm; wind is left to the PX4 stage.
The blackout is placed in mission time when the swarm enters MISSION (the FANET clock is simulation time, which the mission start
offsets).  Every attack goes through swarmsec.rarf_l1.adaptive.make_hook (A14_THETA is bit-identical to A14), so a transient attack is the same
law switched off at t_end.  Rows add scripts/rarf_l1/metrics_rz.py columns.  Results: results/rarf_l1/rz/episodes.jsonl.
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "rarf_l1"))

BASE = "config/swarm/coupled_safety_base.toml"   # the base every B study uses (B-H1, B-S1/S2, B-A1, B-N1)
SEEDS = list(range(20000, 20010))
GEOMS = (0.0, 3.0)
T_TRANSIENT = 8.0
THREATS = ("A0", "A14", "A14_T", "WORST", "WORST_T")
NET = {"nominal": None, "degraded": {"loss": 0.10, "ge_p_gb": 0.05, "ge_p_bg": 0.3, "delay_s": 0.02, "jitter_s": 0.01},
       "severe": {"loss": 0.25, "ge_p_gb": 0.10, "ge_p_bg": 0.3, "delay_s": 0.04, "jitter_s": 0.01}, "blackout": None}
BLACKOUT = (2.0, 4.0)           # (start after onset, duration) [s]


def _design():
    sp = importlib.util.spec_from_file_location("ad_design", ROOT / "experiments/rarf_l1_adaptive/design.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def build():
    D = _design()
    star = json.loads((ROOT / "output/research/rarf_l1_adaptive/theta_star.json").read_text())
    from swarmsec.rarf_l1.adaptive import A14_THETA
    eps = []
    for s in SEEDS:
        for g in GEOMS:
            for th in THREATS:
                for nt in NET:
                    for arm in D.ARMS:
                        theta = None if th == "A0" else (A14_THETA if th.startswith("A14") else star["HOCBF"]["theta"])
                        e = D.ep(s, arm, g, "BRZ", theta=theta, suffix=f"-{th}-n{nt}")
                        if th == "A0":
                            e["overrides"]["attack"]["id"] = "A0"
                        if NET[nt]:
                            e["overrides"]["ral26"].setdefault("fanet", {}).update(NET[nt])
                        e["l1x"]["blackout"] = list(BLACKOUT) if nt == "blackout" else None
                        e["l1x"]["t_end_rel"] = T_TRANSIENT if th.endswith("_T") else None
                        e["meta"].update(threat=th, net=nt, geometry=g, seed=s)
                        eps.append(e)
    return eps


BLACKOUT_REL = {"value": None}     # set per episode in the worker before run_episode (one episode per call)


def _blackout_in_mission_time(harness):
    if getattr(harness.World, "_rz_patched", False):
        return

    class World(harness.World):
        _rz_patched = True

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._bo = BLACKOUT_REL["value"]

        def step(self):
            if self._bo is not None and any(p == "MISSION" for p in self.gcs_phase.values()):
                t0, dur = self._bo
                self.net.cfg.outages.append({"t0": self.t + t0, "dur": dur, "all": True})
                self._bo = None
            super().step()
    harness.World = World


def run_one(spec_base, common, ep, out):
    from swarmsec.rarf_l1 import adaptive, harness
    import metrics_rz
    _blackout_in_mission_time(harness)
    rd = Path(out) / ep["name"]
    t0 = time.time()
    try:
        cfg, draw = harness.resolve(spec_base, common, ep)
        x = ep.get("l1x") or {}
        l1 = {"extra_attackers": []}
        onset = float(cfg["ral26"]["attack"].get("start_s", 10.0))
        BLACKOUT_REL["value"] = (onset + x["blackout"][0], x["blackout"][1]) if x.get("blackout") else None
        t_end = None
        if cfg["attack"]["id"] != "A0" and x.get("theta") is not None:
            hook = adaptive.make_hook(x["theta"], exclude={int(cfg["ral26"]["attacker_id"])})
            if x.get("t_end_rel") is not None:
                t_end = float(cfg["ral26"]["attack"].get("start_s", 10.0)) + float(x["t_end_rel"])

                def hook2(ag, hook=hook, t_end=t_end):
                    hook(ag)
                    inner = ag.attack_terms

                    def attack_terms(tm, p, v, u_safe, u_nom):
                        if tm >= t_end:
                            return u_safe, p.copy(), v.copy(), False
                        return inner(tm, p, v, u_safe, u_nom)
                    ag.attack_terms = attack_terms
                l1["attacker_hook"] = hook2
            else:
                l1["attacker_hook"] = hook
        summ = harness.run_episode(cfg, rd, keep_agent_logs=False, l1=l1)
        row = metrics_rz.episode(rd, [], t_end)
        row.update(meta=ep.get("meta", {}), contacts=len(summ["contact_events"]), wall_s=round(time.time() - t0, 1), layer="L1")
    except Exception:
        row = {"name": ep["name"], "valid": False, "why": traceback.format_exc()[-2000:], "meta": ep.get("meta", {})}
    shutil.rmtree(rd, ignore_errors=True)
    row["name"] = ep["name"]
    return row


def work(job):
    return run_one(*job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="smoke: run only the first k episodes into results/rarf_l1/rz_smoke")
    a = ap.parse_args()
    eps = build()
    out = ROOT / "results" / "rarf_l1" / ("rz_smoke" if a.limit else "rz")
    out.mkdir(parents=True, exist_ok=True)
    f = out / "episodes.jsonl"
    done = {json.loads(l)["name"] for l in f.read_text().splitlines() if l.strip()} if f.exists() else set()
    todo = [e for e in eps if e["name"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    jobs = [(BASE, {}, e, str(out)) for e in todo]
    print(len(eps), "episodes in design;", len(jobs), "to run", flush=True)
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=8) as pool, f.open("a") as fh:
        for r in pool.imap_unordered(work, jobs):
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            if not r.get("valid"):
                print("INVALID", r["name"], str(r.get("why", ""))[-300:], flush=True)


if __name__ == "__main__":
    main()
