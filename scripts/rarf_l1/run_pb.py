#!/usr/bin/env python3
"""B-PB: published avoidance baselines SBC (Wang, Ames & Egerstedt 2017) and BVC (Zhou et al. 2017) through every Layer-1 B study
(experiments/rarf_l1_published/PLAN.md).  Inside the image:

    python3 scripts/rarf_l1/run_pb.py dev       --workers 14   # tuning on the rarf_hocbf_v1 dev seeds, selection rule, FROZEN_B3.json
    python3 scripts/rarf_l1/run_pb.py spectrum  --workers 14   # B-S1 design for the two arms
    python3 scripts/rarf_l1/run_pb.py budget    --workers 14   # B-S2 design
    python3 scripts/rarf_l1/run_pb.py search    --workers 14   # B-A1 CEM per new arm (same budget, seeds, objective)
    python3 scripts/rarf_l1/run_pb.py test      --workers 14   # B-A1 test cells for the new arms + the new theta* against the 8 arms
    python3 scripts/rarf_l1/run_pb.py rz        --workers 14   # B-RZ resilience design
    python3 scripts/rarf_l1/run_pb.py scale     --workers 14   # B-N1 design

Every episode is built by the frozen builders of the study it extends (same seeds, geometries, scenario draws, budgets, base config)
from a placeholder arm, with defense.id set to SBC / BVC and the frozen parameters in cs.b3; the agents run cs_agent_node_v5
(CsAgentV4 + the two policies; identical otherwise).  Episode functions are the frozen run_l1x.run_one / run_rz.run_one.
Results: results/rarf_l1/pb_<stage>/episodes.jsonl; output/research/rarf_l1_published/.
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "rarf_l1"))
sys.path.insert(0, str(ROOT / "controllers" / "swarm"))

BASE = "config/swarm/coupled_safety_base.toml"
OUT = ROOT / "results" / "rarf_l1"
REP = ROOT / "output" / "research" / "rarf_l1_published"
NEW = ("SBC", "BVC")
COMMON3 = {"SBC": {"neighbor_radius": 12.0}, "BVC": {"neighbor_radius": 12.0, "tau_pref": 1.0, "tau_goal": 1.0, "k": 1.0, "tau_acc": 0.5}}
DEV3 = {"S1": ("SBC", {"Ds": 2.5, "gamma": 1.0, "responsibility": 0.5}), "S2": ("SBC", {"Ds": 2.5, "gamma": 1.0, "responsibility": 1.0}),
        "S3": ("SBC", {"Ds": 3.0, "gamma": 1.0, "responsibility": 1.0}),
        "V1": ("BVC", {"r_s": 0.75}), "V2": ("BVC", {"r_s": 1.0}), "V3": ("BVC", {"r_s": 1.5})}
DEV_SEEDS, DEV_GEOMS = list(range(17400, 17404)), (0.0, 2.0, 3.0)


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def as_new(e, arm, params, old_label):
    """Turn a builder episode of a placeholder arm into the new arm (name, defense id, parameters, meta)."""
    e["overrides"]["defense"]["id"] = arm
    e["overrides"].setdefault("cs", {}).pop("b2", None)
    e["overrides"]["cs"]["b3"] = {**COMMON3[arm], **params}
    e["name"] = e["name"].replace(f"-{old_label}-", f"-{arm}-") if f"-{old_label}-" in e["name"] else e["name"] + f"-{arm}"
    e["meta"]["method"] = arm
    e["meta"]["b3"] = params
    return e


def frozen():
    return json.loads((REP / "FROZEN_B3.json").read_text())


# ---------------------------------------------------------------------------------------------------- workers
def _bind():
    from swarmsec.rarf_l1 import harness
    harness._install_stubs()
    import cs_agent_node_v4 as v4
    import cs_agent_node_v5 as v5
    v4.CsAgentV4 = v5.CsAgentV5


def work_l1x(job):
    _bind()
    import run_l1x
    tag, e = job
    return run_l1x.run_one(BASE, {}, e, str(OUT / tag))


def work_rz(job):
    _bind()
    import run_rz
    tag, e = job
    return run_rz.run_one(BASE, {}, e, str(OUT / tag))


def run_eps(eps, tag, workers, fn=work_l1x):
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    f = out / "episodes.jsonl"
    rows = {json.loads(l)["name"]: json.loads(l) for l in f.read_text().splitlines() if l.strip()} if f.exists() else {}
    todo = [e for e in eps if e["name"] not in rows]
    assert len({e["name"] for e in eps}) == len(eps), "duplicate episode names"
    print(tag, len(eps), "episodes;", len(todo), "to run", flush=True)
    if todo:
        with mp.get_context("spawn").Pool(workers, maxtasksperchild=8) as pool, f.open("a") as fh:
            for r in pool.imap_unordered(fn, [(tag, e) for e in todo]):
                fh.write(json.dumps(r, default=str) + "\n")
                fh.flush()
                rows[r["name"]] = r
                if not r.get("valid", True):
                    print("INVALID", r["name"], str(r.get("why", ""))[-400:], flush=True)
    return [rows[e["name"]] for e in eps]


# ---------------------------------------------------------------------------------------------------- stages
def dev(workers):
    HC = _load("hc", "experiments/rarf_hocbf_v1/design.py")
    BD = _load("bd", "experiments/rarf_l1_breadth/design.py")
    eps = []
    for key, (arm, par) in DEV3.items():
        for s in DEV_SEEDS:
            for g in DEV_GEOMS:
                e = BD.ep(s, (key, "ORCA", None, None), g, "A14", 2.5, 1.25, 0.5, "BPBdev")
                eps.append(as_new(e, arm, par, key) | {})
                eps[-1]["name"] = eps[-1]["name"] + f"-{key}"
                eps[-1]["meta"]["cand"] = key
    R = run_eps(eps, "pb_dev", workers)
    sel = {}
    for arm in NEW:
        cands = [k for k, v in DEV3.items() if v[0] == arm]
        st = {}
        for k in cands:
            rr = [r for r in R if r.get("meta", {}).get("cand") == k and r.get("valid")]
            st[k] = {"n": len(rr), "routes": int(sum(r["route_complete"] for r in rr)), "crash": int(sum(r["crash"] for r in rr)),
                     "mean_min_sep_ha": float(np.mean([r["min_sep_ha"] for r in rr])) if rr else float("nan")}
        elig = [k for k in cands if st[k]["routes"] >= 11 and st[k]["crash"] == 0]
        if elig:
            k = max(elig, key=lambda k: st[k]["mean_min_sep_ha"])
            rule = "eligible: max mean min_sep_ha"
        else:
            k = max(cands, key=lambda k: (st[k]["routes"], st[k]["mean_min_sep_ha"]))
            rule = "none eligible: most routes, ties by mean min_sep_ha"
        sel[arm] = {"selected": k, "params": DEV3[k][1], "rule": rule, "dev": st}
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "FROZEN_B3.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1), flush=True)


def spectrum(workers):
    BD = _load("bd", "experiments/rarf_l1_breadth/design.py")
    F = frozen()
    eps = []
    for s in BD.SPECTRUM_SEEDS:
        for g in (0.0, 2.0, 3.0):
            for att, (_, e_, bp, bv) in BD.SPECTRUM.items():
                for arm in NEW:
                    eps.append(as_new(BD.ep(s, (arm, "ORCA", None, None), g, att, e_, bp, bv, "BS1"), arm, F[arm]["params"], arm))
    for e in eps:
        e["meta"].update(attack=e["overrides"]["attack"]["id"])
    run_eps(eps, "pb_spectrum", workers)


def budget(workers):
    BD = _load("bd", "experiments/rarf_l1_breadth/design.py")
    F = frozen()
    eps = []
    for s in BD.BUDGET_SEEDS:
        for g in (2.0, 3.0):
            for e_ in BD.EPS:
                for bp in BD.BETA:
                    for arm in NEW:
                        eps.append(as_new(BD.ep(s, (arm, "ORCA", None, None), g, "A14", e_, bp, round(0.4 * bp, 3), "BS2"), arm,
                                          F[arm]["params"], arm))
    run_eps(eps, "pb_budget", workers)


def _AS():
    return _load("as_b", "scripts/rarf_l1/adaptive_search.py")


def search(workers):
    AS = _AS()
    DS = AS.DS
    F = frozen()
    from swarmsec.rarf_l1.adaptive import A14_THETA
    rng = np.random.default_rng(20261117)
    state = {a: (AS.to_z(A14_THETA), np.full(len(A14_THETA), AS.SD0)) for a in NEW}
    hist = {a: [] for a in NEW}

    def mk(s, a, g, tag, th, suffix):
        return as_new(DS.ep(s, (a, "ORCA", None, None), g, tag, theta=th, suffix=suffix), a, F[a]["params"], a)
    for it in range(AS.ITERS):
        cands, eps = {}, []
        for a in NEW:
            mu, sd = state[a]
            Z = [mu.copy(), AS.to_z(A14_THETA)] + [np.clip(mu + sd * rng.standard_normal(len(mu)), 0, 1) for _ in range(AS.POP - 2)]
            cands[a] = [AS.to_theta(z) for z in Z]
            for c, th in enumerate(cands[a]):
                for s in AS.SEARCH_SEEDS[it]:
                    for g in AS.SEARCH_GEOMS:
                        eps.append(mk(s, a, g, "BPBs", th, f"-i{it}c{c}"))
        R = run_eps(eps, "pb_adaptive_search", workers)
        by = {r["name"]: r for r in R}
        for a in NEW:
            sc = []
            for c, th in enumerate(cands[a]):
                rows = [by[e["name"]] for e in eps if e["meta"]["method"] == a and e["name"].endswith(f"-i{it}c{c}")]
                sc.append(AS.score(rows))
                hist[a].append({"it": it, "cand": c, "theta": th, "score": sc[-1]})
            order = np.argsort(sc)[:AS.ELITE]
            E = np.array([AS.to_z(cands[a][k]) for k in order])
            mu, sd = state[a]
            state[a] = (AS.SMOOTH * E.mean(0) + (1 - AS.SMOOTH) * mu, np.maximum(AS.SMOOTH * E.std(0) + (1 - AS.SMOOTH) * sd, 0.02))
            print(f"it {it} {a}: best {sc[order[0]]:.3f}  mean-cand {sc[0]:.3f}  A14 {sc[1]:.3f}", flush=True)
    star, eps = {}, []
    top = {a: sorted(hist[a], key=lambda h: h["score"])[:3] for a in NEW}
    for a in NEW:
        for k, h in enumerate(top[a]):
            eps += [mk(s, a, g, "BPBsel", h["theta"], f"-r{k}") for s, g in AS.SELECT]
    R = {r["name"]: r for r in run_eps(eps, "pb_adaptive_select", workers)}
    for a in NEW:
        sel = [AS.score([R[e["name"]] for e in eps if e["meta"]["method"] == a and e["name"].endswith(f"-r{k}")]) for k in range(3)]
        b = int(np.argmin(sel))
        star[a] = {"theta": top[a][b]["theta"], "select_score": sel[b], "select_scores_top3": sel, "search_score": top[a][b]["score"]}
    (REP / "theta_star.json").write_text(json.dumps(star, indent=1))
    (REP / "search_history.json").write_text(json.dumps(hist))
    print(json.dumps({a: {k: v for k, v in d.items() if k != "theta"} for a, d in star.items()}, indent=1), flush=True)


def test(workers):
    AS = _AS()
    DS = AS.DS
    F = frozen()
    star_old = json.loads((ROOT / "output/research/rarf_l1_adaptive/theta_star.json").read_text())
    star_new = json.loads((REP / "theta_star.json").read_text())
    eps = []
    for a in NEW:
        mk = lambda s, g, th, suf, a=a: as_new(DS.ep(s, (a, "ORCA", None, None), g, "BA1t", theta=th, suffix=suf), a, F[a]["params"], a)
        eps += [mk(s, g, None, "-A14") for s, g in AS.TEST]
        eps += [mk(s, g, star_new[a]["theta"], f"-th{a}") for s, g in AS.TEST]
        for src in list(star_old) + [x for x in NEW if x != a]:
            th = star_old[src]["theta"] if src in star_old else star_new[src]["theta"]
            eps += [mk(s, g, th, f"-th{src}") for s, g in AS.TRANSFER]
    for a in AS.ARMS:                                       # the new attackers against the eight existing arms
        for src in NEW:
            eps += [DS.ep(s, a, g, "BA1t", theta=star_new[src]["theta"], suffix=f"-th{src}") for s, g in AS.TRANSFER]
    run_eps(eps, "pb_adaptive_test", workers)


def rz(workers):
    Z = _load("rz_b", "scripts/rarf_l1/run_rz.py")
    D = Z._design()
    F = frozen()
    star = json.loads((ROOT / "output/research/rarf_l1_adaptive/theta_star.json").read_text())
    from swarmsec.rarf_l1.adaptive import A14_THETA
    eps = []
    for s in Z.SEEDS:
        for g in Z.GEOMS:
            for th in Z.THREATS:
                for nt in Z.NET:
                    for arm in NEW:
                        theta = None if th == "A0" else (A14_THETA if th.startswith("A14") else star["HOCBF"]["theta"])
                        e = as_new(D.ep(s, (arm, "ORCA", None, None), g, "BRZ", theta=theta, suffix=f"-{th}-n{nt}"), arm, F[arm]["params"], arm)
                        if th == "A0":
                            e["overrides"]["attack"]["id"] = "A0"
                        if Z.NET[nt]:
                            e["overrides"]["ral26"].setdefault("fanet", {}).update(Z.NET[nt])
                        e["l1x"]["blackout"] = list(Z.BLACKOUT) if nt == "blackout" else None
                        e["l1x"]["t_end_rel"] = Z.T_TRANSIENT if th.endswith("_T") else None
                        e["meta"].update(threat=th, net=nt, geometry=g, seed=s)
                        eps.append(e)
    run_eps(eps, "pb_rz", workers, fn=work_rz)


def scale(workers):
    DS = _load("ad_b", "experiments/rarf_l1_adaptive/design.py")
    F = frozen()
    eps = []
    for n, ks in DS.SCALE_N.items():
        for s in DS.SCALE_SEEDS:
            prim = DS.D.scenarios.scenario("RA", s, n)["attacker_id"]
            for k in ks:
                ex = DS.extra_ids(s, n, prim, k)
                for g in DS.SCALE_GEOMS:
                    for arm in NEW:
                        e = DS.ep(s, (arm, "ORCA", None, None), g, "BN1", theta=None, n=n, extra=ex, suffix=f"-N{n}-k{k}")
                        e = as_new(e, arm, F[arm]["params"], arm)
                        e["meta"].update(k=k)
                        eps.append(e)
    run_eps(eps, "pb_scale", workers)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("dev", "spectrum", "budget", "search", "test", "rz", "scale"))
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    globals()[a.stage](a.workers)
