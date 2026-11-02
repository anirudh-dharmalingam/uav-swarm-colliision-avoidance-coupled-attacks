#!/usr/bin/env python3
"""Layer-1 runner with per-episode harness options (B-A1 adaptive attacker, B-N1 scale / multiple attackers).

Same as run_l1.py (unchanged, frozen for B-S1/B-S2) except that each spec episode may carry
    "l1x": {"theta": {...adaptive parameters...} | null, "extra_attackers": [ids], "wind": {...} | null}
theta != null installs swarmsec.rarf_l1.adaptive on the designated attacker and every extra attacker (co-attackers are never targeted);
extra attackers without theta use A14_THETA (the A14 law with co-attacker exclusion).  Rows add scripts/rarf_l1/metrics_multi.py columns.

    python3 scripts/rarf_l1/run_l1x.py --spec <spec.json> --tag <tag> --workers 14
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "rarf_l1"))


def l1_options(x):
    from swarmsec.rarf_l1 import adaptive
    x = x or {}
    extra = [int(i) for i in x.get("extra_attackers", [])]
    l1 = {"extra_attackers": extra}
    if x.get("wind"):
        l1["wind"] = x["wind"]
    th = x.get("theta")
    if th is not None or extra:
        att = set(extra) | {int(x["attacker_id"])} if "attacker_id" in x else set(extra)
        l1["attacker_hook"] = adaptive.make_hook(th if th is not None else adaptive.A14_THETA, exclude=att)
    return l1, extra


def run_one(spec_base, common, ep, out, keep=False):
    from swarmsec.rarf_l1 import harness
    import metrics_multi
    rd = Path(out) / ep["name"]
    t0 = time.time()
    try:
        cfg, draw = harness.resolve(spec_base, common, ep)
        x = dict(ep.get("l1x") or {}, attacker_id=int(cfg["ral26"]["attacker_id"]))
        l1, extra = l1_options(x)
        summ = harness.run_episode(cfg, rd, keep_agent_logs=keep, l1=l1)
        row = metrics_multi.episode(rd, extra)
        row.update(meta=ep.get("meta", {}), contacts=len(summ["contact_events"]), wall_s=round(time.time() - t0, 1), layer="L1")
    except Exception:
        row = {"name": ep["name"], "valid": False, "why": traceback.format_exc()[-2000:], "meta": ep.get("meta", {})}
    if not keep:
        shutil.rmtree(rd, ignore_errors=True)               # outcome row only (truth is not needed again for these sweeps)
    return row


def work(job):
    return run_one(*job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--keep", action="store_true", help="keep episode directories (truth, agent logs)")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    out = ROOT / "results" / "rarf_l1" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    f = out / "episodes.jsonl"
    done = {json.loads(l)["name"] for l in f.read_text().splitlines() if l.strip()} if f.exists() else set()
    eps = [e for e in spec["episodes"] if e["name"] not in done]
    jobs = [(spec.get("base", "config/swarm/ral26_base.toml"), spec.get("common", {}), e, str(out), a.keep) for e in eps]
    print(len(jobs), "episodes", flush=True)
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=8) as pool, f.open("a") as fh:
        for k, r in enumerate(pool.imap_unordered(work, jobs)):
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            if k % 50 == 0 or not r.get("valid", True):
                print(k, time.strftime("%H:%M:%S"), r.get("name"), r.get("min_sep_ha_any"), r.get("wall_s"),
                      r.get("why", "")[-300:] if not r.get("valid", True) else "", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
