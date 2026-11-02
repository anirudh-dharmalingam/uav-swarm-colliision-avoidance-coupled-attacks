#!/usr/bin/env python3
"""Run a cs_batch-format spec (experiments/*/specs/*.json) in the Layer-1 harness (swarmsec/rarf_l1).

    python3 scripts/rarf_l1/run_l1.py --spec experiments/rarf_hocbf_v1/specs/confirm.json --tag hocbf_confirm --workers 3 [--only a,b]
             [--keep-logs] [--l1 '{"wind": {"sigma": 0.3, "tau": 2.0}}']

Outputs results/rarf_l1/<tag>/<episode>/ in the PX4 layout and appends one outcome row per episode (scripts/coupled_safety/metrics.py
definitions) to results/rarf_l1/<tag>/episodes.jsonl.  Resume rule: an episode whose row exists is skipped.
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
sys.path.insert(0, str(ROOT / "scripts" / "coupled_safety"))


def work(job):
    spec_base, common, ep, out, keep, l1 = job
    from swarmsec.rarf_l1 import harness
    import metrics
    rd = Path(out) / ep["name"]
    t0 = time.time()
    try:
        cfg, draw = harness.resolve(spec_base, common, ep)
        summ = harness.run_episode(cfg, rd, keep_agent_logs=keep, l1=l1)
        row = metrics.episode(rd)
        row.update(meta=ep.get("meta", {}), contacts=len(summ["contact_events"]), wall_s=round(time.time() - t0, 1), layer="L1")
    except Exception:
        row = {"name": ep["name"], "valid": False, "why": traceback.format_exc()[-2000:], "meta": ep.get("meta", {})}
    if not keep:
        shutil.rmtree(rd / "agents", ignore_errors=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", default=None)
    ap.add_argument("--keep-logs", action="store_true")
    ap.add_argument("--l1", default="{}")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    out = ROOT / "results" / "rarf_l1" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    f = out / "episodes.jsonl"
    done = {json.loads(l)["name"] for l in f.read_text().splitlines() if l.strip()} if f.exists() else set()
    eps = [e for e in spec["episodes"] if e["name"] not in done and (not a.only or e["name"] in a.only.split(","))]
    (out / "run.json").write_text(json.dumps({"spec": a.spec, "l1": json.loads(a.l1), "n_spec": len(spec["episodes"]),
                                              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    jobs = [(spec.get("base", "config/swarm/ral26_base.toml"), spec.get("common", {}), e, str(out), a.keep_logs, json.loads(a.l1))
            for e in eps]
    print(len(jobs), "episodes", flush=True)
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=8) as pool, f.open("a") as fh:
        for k, r in enumerate(pool.imap_unordered(work, jobs)):
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            print(k, r.get("name"), r.get("min_sep_ha"), r.get("wall_s"), r.get("why", "")[-300:] if not r.get("valid", True) else "",
                  flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
