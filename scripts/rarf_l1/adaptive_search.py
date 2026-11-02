#!/usr/bin/env python3
"""B-A1 defence-aware adaptive attacker: search and test in Layer 1 (experiments/rarf_l1_adaptive/PLAN.md).  Inside the image:

    python3 scripts/rarf_l1/adaptive_search.py search --workers 14
    python3 scripts/rarf_l1/adaptive_search.py test   --workers 14

Search: cross-entropy method in the unit cube of swarmsec.rarf_l1.adaptive.BOUNDS, per arm, POP candidates per iteration (candidate 0 =
current mean, candidate 1 = A14_THETA), all candidates of an iteration on the same SEARCH episodes (seeds x geometries), seeds rotating
over iterations; elite 1/4, smoothing 0.7; objective = mean min_sep_ha (the attacker minimises it).  Selection: the 3 best candidates
over all iterations re-run on the SELECT episodes; theta* = the lowest mean there.  Budgets are the declared ones (epsilon 2.5 m/s^2,
beta_p 1.25 m, beta_v 0.5 m/s) for every candidate.
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "rarf_l1"))
_sp = importlib.util.spec_from_file_location("bad", ROOT / "experiments" / "rarf_l1_adaptive" / "design.py")
DS = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(DS)

ARMS = [a[0] for a in DS.ARMS]
POP, ITERS, ELITE, SMOOTH, SD0 = 14, 6, 4, 0.7, 0.30
SEARCH_GEOMS = (0.0, 3.0)
SEARCH_SEEDS = [list(range(19300 + 3 * it, 19303 + 3 * it)) for it in range(ITERS)]     # 19300-19317
SELECT = [(s, g) for s in range(19330, 19334) for g in (0.0, 2.0, 3.0)]
TEST = [(s, g) for s in range(19400, 19420) for g in (0.0, 2.0, 3.0)]
TRANSFER = [(s, g) for s in range(19400, 19410) for g in (2.0, 3.0)]
RNG_SEED = 20261108
OUT = ROOT / "results" / "rarf_l1"
REP = ROOT / "output" / "research" / "rarf_l1_adaptive"
BASE = "config/swarm/coupled_safety_base.toml"


def work(job):
    import run_l1x
    tag, e = job
    return run_l1x.run_one(BASE, {}, e, str(OUT / tag))


def run_eps(eps, tag, workers):
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    f = out / "episodes.jsonl"
    rows = {json.loads(l)["name"]: json.loads(l) for l in f.read_text().splitlines() if l.strip()} if f.exists() else {}
    todo = [e for e in eps if e["name"] not in rows]
    if todo:
        with mp.get_context("spawn").Pool(workers, maxtasksperchild=8) as pool, f.open("a") as fh:
            for r in pool.imap_unordered(work, [(tag, e) for e in todo]):
                fh.write(json.dumps(r, default=str) + "\n")
                fh.flush()
                rows[r["name"]] = r
                if not r.get("valid", True):
                    print("INVALID", r["name"], r.get("why", "")[-400:], flush=True)
    return [rows[e["name"]] for e in eps]


def to_theta(z):
    from swarmsec.rarf_l1.adaptive import BOUNDS
    return {k: float(lo + float(np.clip(z[i], 0, 1)) * (hi - lo)) for i, (k, (lo, hi)) in enumerate(BOUNDS.items())}


def to_z(th):
    from swarmsec.rarf_l1.adaptive import BOUNDS
    return np.array([(th[k] - lo) / (hi - lo) for k, (lo, hi) in BOUNDS.items()])


def score(rows):
    v = [r["min_sep_ha"] for r in rows if r.get("valid")]
    return float(np.mean(v)) if v else np.inf


def search(workers):
    from swarmsec.rarf_l1.adaptive import A14_THETA
    rng = np.random.default_rng(RNG_SEED)
    state = {a: (to_z(A14_THETA), np.full(len(A14_THETA), SD0)) for a in ARMS}
    hist = {a: [] for a in ARMS}
    for it in range(ITERS):
        eps, cands = [], {}
        for a in ARMS:
            mu, sd = state[a]
            Z = [mu.copy(), to_z(A14_THETA)] + [np.clip(mu + sd * rng.standard_normal(len(mu)), 0, 1) for _ in range(POP - 2)]
            cands[a] = [to_theta(z) for z in Z]
            for c, th in enumerate(cands[a]):
                for s in SEARCH_SEEDS[it]:
                    for g in SEARCH_GEOMS:
                        eps.append(DS.ep(s, a, g, "BA1s", theta=th, suffix=f"-i{it}c{c}"))
        R = {r["name"]: r for r in run_eps(eps, "adaptive_search", workers)}
        for a in ARMS:
            sc = []
            for c, th in enumerate(cands[a]):
                sc.append(score([r for nm, r in R.items() if r.get("meta", {}).get("method") == a and nm.endswith(f"-i{it}c{c}")]))
                hist[a].append({"it": it, "cand": c, "theta": th, "score": sc[-1]})
            order = np.argsort(sc)[:ELITE]
            E = np.array([to_z(cands[a][k]) for k in order])
            mu, sd = state[a]
            state[a] = (SMOOTH * E.mean(0) + (1 - SMOOTH) * mu, np.maximum(SMOOTH * E.std(0) + (1 - SMOOTH) * sd, 0.02))
            print(f"it {it} {a}: best {sc[order[0]]:.3f}  mean-cand {sc[0]:.3f}  A14 {sc[1]:.3f}  {time.strftime('%H:%M')}", flush=True)
    eps, top = [], {}
    for a in ARMS:
        top[a] = sorted(hist[a], key=lambda h: h["score"])[:3]
        for k, h in enumerate(top[a]):
            eps += [DS.ep(s, a, g, "BA1sel", theta=h["theta"], suffix=f"-r{k}") for s, g in SELECT]
        eps += [DS.ep(s, a, g, "BA1sel", theta=None, suffix="-A14") for s, g in SELECT]
    R = run_eps(eps, "adaptive_select", workers)
    star = {}
    for a in ARMS:
        sel = [score([r for r in R if r["meta"]["method"] == a and r["name"].endswith(f"-r{k}")]) for k in range(len(top[a]))]
        base = score([r for r in R if r["meta"]["method"] == a and r["name"].endswith("-A14")])
        b = int(np.argmin(sel))
        star[a] = {"theta": top[a][b]["theta"], "select_score": sel[b], "select_scores_top3": sel, "select_A14": base,
                   "search_score": top[a][b]["score"]}
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "theta_star.json").write_text(json.dumps(star, indent=1))
    (REP / "search_history.json").write_text(json.dumps(hist, indent=1))
    print(json.dumps({a: {k: v for k, v in d.items() if k != "theta"} for a, d in star.items()}, indent=1))


def test(workers):
    star = json.loads((REP / "theta_star.json").read_text())
    eps = []
    for a in ARMS:
        eps += [DS.ep(s, a, g, "BA1t", theta=None, suffix="-A14") for s, g in TEST]
        eps += [DS.ep(s, a, g, "BA1t", theta=star[a]["theta"], suffix=f"-th{a}") for s, g in TEST]
        for src in ARMS:
            if src != a:
                eps += [DS.ep(s, a, g, "BA1t", theta=star[src]["theta"], suffix=f"-th{src}") for s, g in TRANSFER]
    print(len(eps), "test episodes", flush=True)
    run_eps(eps, "adaptive_test", workers)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("search", "test"))
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    {"search": search, "test": test}[a.stage](a.workers)
