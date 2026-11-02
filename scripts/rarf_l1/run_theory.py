#!/usr/bin/env python3
"""B-ISS and B-DLY: experiments that test the control-theoretic readings of the B methods (experiments/rarf_l1_theory/PLAN.md).
Inside the image:

    python3 scripts/rarf_l1/run_theory.py iss --workers 14
    python3 scripts/rarf_l1/run_theory.py dly --workers 14

B-ISS  input-to-state-safety reading of the soft policy.  The best-estimate soft QP min 1/2||u - u_nom||^2 + rho ||xi||_1
       (swarmsec/ral26_core.least_violation_command, rho = 1e3 frozen) is the standard ISSf relaxation of a CBF-QP; ISSf predicts a
       class-K gain gamma with h_min >= h_0 - gamma(||d||) and a gain that shrinks as the slack weight grows, until the actuator
       bound U binds.  Factors: rho in {1e1, 1e2, 1e3, 1e4, 1e5} x pursuit epsilon in {0, 0.5, 1.5, 2.5, 3.5, 5.0} m/s^2 at the
       declared beta_p = 1.25 m, beta_v = 0.5 m/s, arms FF (soft policy at every decision: the clean ISSf object) and RARF (hard rows
       when certified, soft otherwise); B-S2 seeds 19200-19209 x g {2, 3} (rho = 1e3 cells at epsilon > 0 reproduce B-S2 rows: a
       bit-identity check).  1,200 episodes.  rho is injected by rebinding ral26_core.least_violation_command (no source change).
B-DLY  delay compensation.  Receiver-age propagation p_j + s v_j is a zeroth-order predictor of the delayed neighbour state (the
       control-barrier-functional reading).  Factors: one-way FANET delay in {0, 0.05, 0.10, 0.20, 0.40} s with 20 % jitter
       (t_net_max raised to delay + 0.1 s so packets are delayed, not dropped; the declared bound T_net_max = 0.15 s in the agents'
       envelopes is unchanged, so 0.20 and 0.40 are outside the declared assumption), threat {A0, A14}, the eight B arms, g {0, 3},
       seeds 20200-20209.  1,600 episodes.
Results: results/rarf_l1/{iss,dly}/episodes.jsonl.
"""
import argparse
import functools
import importlib.util
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "rarf_l1"))

BASE = "config/swarm/coupled_safety_base.toml"
OUT = ROOT / "results" / "rarf_l1"
RHOS = (1e1, 1e2, 1e3, 1e4, 1e5)
ISS_EPS = (0.0, 0.5, 1.5, 2.5, 3.5, 5.0)
DELAYS = (0.0, 0.05, 0.10, 0.20, 0.40)
DLY_SEEDS = list(range(20200, 20210))


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def work(job):
    tag, e = job
    from swarmsec import ral26_core as rc
    rho = (e.get("l1x") or {}).get("rho")
    if rho is not None:
        if not hasattr(rc, "_lvc_orig"):
            rc._lvc_orig = rc.least_violation_command
        rc.least_violation_command = functools.partial(rc._lvc_orig, rho=float(rho))
    elif hasattr(rc, "_lvc_orig"):
        rc.least_violation_command = rc._lvc_orig
    import run_l1x
    return run_l1x.run_one(BASE, {}, e, str(OUT / tag))


def run_eps(eps, tag, workers):
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    f = out / "episodes.jsonl"
    done = {json.loads(l)["name"] for l in f.read_text().splitlines() if l.strip()} if f.exists() else set()
    assert len({e["name"] for e in eps}) == len(eps)
    todo = [e for e in eps if e["name"] not in done]
    print(tag, len(eps), "episodes;", len(todo), "to run", flush=True)
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=8) as pool, f.open("a") as fh:
        for r in pool.imap_unordered(work, [(tag, e) for e in todo]):
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            if not r.get("valid", True):
                print("INVALID", r["name"], str(r.get("why", ""))[-300:], flush=True)


def iss(workers):
    BD = _load("bd", "experiments/rarf_l1_breadth/design.py")
    eps = []
    for s in BD.BUDGET_SEEDS:
        for g in (2.0, 3.0):
            for eps_ in ISS_EPS:
                for rho in RHOS:
                    for arm in ("FF", "RARF"):
                        e = BD.ep(s, [a for a in BD.ARMS if a[0] == arm][0], g, "A14", eps_, 1.25, 0.5, "BISS")
                        e["name"] += f"-rho{rho:g}"
                        e["l1x"] = {"rho": rho}
                        e["meta"].update(rho=rho)
                        eps.append(e)
    run_eps(eps, "iss", workers)


def dly(workers):
    D = _load("ad", "experiments/rarf_l1_adaptive/design.py")
    eps = []
    for s in DLY_SEEDS:
        for g in (0.0, 3.0):
            for th in ("A0", "A14"):
                for dl in DELAYS:
                    for arm in D.ARMS:
                        e = D.ep(s, arm, g, "BDLY", suffix=f"-{th}-d{dl:g}")
                        if th == "A0":
                            e["overrides"]["attack"]["id"] = "A0"
                        e["overrides"]["ral26"]["fanet"].update(delay_s=dl, jitter_s=0.2 * dl, t_net_max=dl + 0.1)
                        e["meta"].update(threat=th, delay=dl, geometry=g, seed=s)
                        eps.append(e)
    run_eps(eps, "dly", workers)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("iss", "dly"))
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    {"iss": iss, "dly": dly}[a.stage](a.workers)
