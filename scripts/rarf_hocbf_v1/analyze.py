#!/usr/bin/env python3
"""rarf_hocbf_v1 analysis (experiments/rarf_hocbf_v1/PLAN.md).  Inside the image:   python3 scripts/rarf_hocbf_v1/analyze.py dev|confirm

Reuses scripts/rarf_plant_time_v3/analyze.py (frozen endpoints via outcomes.py, the instrument() timing audit, cluster bootstrap) with
this study's arm set and contrasts, plus the HOCBF audit from the hc_* log columns (honest agents, MISSION phase, scored window):
slack > 1e-6 share, solver non-optimal share, final-command row residual < -1e-6 share (QP output modified by the downstream slew limit),
slew-modified share, HOCBF solve time p99.
dev: per-candidate summaries and the frozen selection rule (PLAN.md §3) + the planning SD of the primary contrast for N.
"""
import gzip
import importlib.util
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_sp = importlib.util.spec_from_file_location("v3a", ROOT / "scripts/rarf_plant_time_v3/analyze.py")
A = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(A)
A.SEED = 20261101
A.ARMS = ("NOM", "RARF", "FF", "HOCBF", "RHOCBF", "ORCA", "MRCBF")                      # PLAN.md amendment A2
_L = 1 - 0.05 / 6
A.CONTRASTS = (("P", "RARF", "RHOCBF", 0.95, "PRIMARY"), ("S1", "RARF", "HOCBF", _L, "SECONDARY_bonferroni"),
               ("S2", "FF", "RHOCBF", _L, "SECONDARY_bonferroni"), ("S3", "RHOCBF", "HOCBF", _L, "SECONDARY_bonferroni"),
               ("S4", "RARF", "ORCA", _L, "SECONDARY_bonferroni"), ("S5", "RARF", "MRCBF", _L, "SECONDARY_bonferroni"),
               ("S6", "FF", "MRCBF", _L, "SECONDARY_bonferroni"))
FAMILIES = {"dev": (("HOCBF", ("H1", "H2", "H3")), ("RHOCBF", ("R1", "R2", "R3"))),
            "dev2": (("ORCA", ("O1", "O2", "O3")), ("MRCBF", ("M1", "M2", "M3")))}


def hc_audit(rd: Path):
    cfg = json.loads((rd / "resolved_config.json").read_text())
    R = cfg["ral26"]
    att, n, on = int(R["attacker_id"]), int(cfg["swarm"]["num_uavs"]), float(R["attack"]["start_s"])
    W = []
    for i in range(n):
        if i == att:
            continue
        f = next((p for p in (rd / "agents" / f"agent_{i}.csv.gz", rd / "agents" / f"agent_{i}.csv") if p.exists()), None)
        if f is None:
            continue
        raw = f.read_bytes()
        D = pd.read_csv(io.BytesIO(gzip.decompress(raw) if f.suffix == ".gz" else raw), low_memory=False)
        if ("hc_status" not in D or D.hc_status.isna().all()) and ("b2_status" not in D or D.b2_status.isna().all()):
            return {}
        D = D[(D.phase == "MISSION") & (D.t >= on) & (D.t <= on + 20.0)]
        W.append(D)
    if not W:
        return {}
    D = pd.concat(W)
    num = lambda c: pd.to_numeric(D[c], errors="coerce")  # noqa: E731
    if "b2_status" in D and D.b2_status.notna().any():
        return {"b2_n": len(D), "b2_slack_pos": float((num("b2_slack") > 1e-6).mean()), "b2_slack_max": float(num("b2_slack").max()),
                "b2_infeasible": float(num("b2_infeasible").fillna(0).mean()), "b2_viol_pos": float((num("b2_viol") > 1e-6).mean()),
                "b2_ic_ok": float(num("b2_ic_ok").mean()) if num("b2_ic_ok").notna().any() else math.nan,
                "b2_slew_mod": float(num("b2_slew_mod").fillna(0).mean()), "b2_ms_p99": float(np.nanquantile(num("b2_ms"), .99)),
                "b2_nonoptimal": float((~D.b2_status.astype(str).isin(["optimal", "none", "lp2", "lp3"])).mean())}
    return {"hc_n": len(D), "hc_slack_pos": float((num("hc_slack") > 1e-6).mean()), "hc_slack_max": float(num("hc_slack").max()),
            "hc_nonoptimal": float((D.hc_status.astype(str) != "optimal").mean() - (D.hc_status.astype(str) == "none").mean()),
            "hc_fail": float(num("hc_fail").fillna(0).mean()), "hc_res_final_neg": float((num("hc_res_final") < -1e-6).mean()),
            "hc_res_final_min": float(num("hc_res_final").min()),
            "hc_res_rob_neg": float((num("hc_res_rob") < -1e-6).mean()) if "hc_res_rob" in D else math.nan, "hc_slew_mod": float(num("hc_slew_mod").fillna(0).mean()),
            "hc_ms_p99": float(np.nanquantile(num("hc_ms"), .99)), "hc_rows_mean": float(num("hc_rows").mean())}


def run(stage):
    td = ROOT / "results/rarf_hocbf_v1" / stage
    spec = json.loads((ROOT / "experiments/rarf_hocbf_v1/specs" / f"{stage}.json").read_text())
    out = ROOT / "output/research/rarf_hocbf_v1" / stage
    out.mkdir(parents=True, exist_ok=True)
    meta = {e["name"]: e["meta"] for e in spec["episodes"]}
    rows = []
    for rd in sorted(p for p in td.iterdir() if p.is_dir() and not p.name.startswith("_")):
        r = {"name": rd.name}
        if not (rd / "eval" / "summary.json").exists():
            r.update(valid=False, why="no eval/summary.json (telemetry missing)")
        else:
            try:
                r.update(A.O.episode(rd))
            except Exception as exc:
                r.update(valid=False, why=repr(exc))
            try:
                r.update({k: v for k, v in A.instrument(rd).items() if k != "name"})
            except Exception as exc:
                r.update(inst_error=repr(exc), inst_pass=False)
            r.update(hc_audit(rd))
        rows.append(r)
    seen = {r["name"] for r in rows}
    rows += [{"name": nm, "valid": False, "why": "planned, not attempted"} for nm in meta if nm not in seen]
    E = pd.DataFrame(rows)
    for k in ("method", "geometry", "seed", "order"):
        E[k] = E.name.map(lambda x, k=k: meta.get(x, {}).get(k))
    E["valid"] = E.valid.astype(str).str.lower() == "true"
    if "why" not in E:
        E["why"] = ""
    E.to_csv(out / "episodes.csv", index=False)
    ok = E[E.valid]
    R = {"inventory": {"planned": len(meta), "analysable": int(E.valid.sum()), "non_analysable": E[~E.valid][["name", "why"]].to_dict("records")},
         "instrument_pass": {m: f"{int(g.inst_pass.fillna(False).astype(bool).sum())}/{len(g)}" for m, g in E.groupby("method")},
         "instrument_fail_criteria": {c: int((~E[c].fillna(False).astype(bool)).sum()) for c in ("inst_rtf", "inst_miss", "inst_rate", "inst_ref", "inst_onset") if c in E}}
    hc_cols = [c for c in E if c.startswith("hc_") or c.startswith("b2_")]
    R["hocbf_audit_by_arm_geometry"] = {f"{m}|g{g:g}": {c: float(np.nanmean(s[c])) for c in hc_cols}
                                        for (m, g), s in ok[ok[hc_cols].notna().any(axis=1)].groupby(["method", "geometry"])} if hc_cols else {}
    if stage in ("dev", "dev2"):
        summ = {}
        for m, s in ok.groupby("method"):
            summ[m] = {"n": len(s), "route_complete": int(s.route_complete.sum()), "crash": int(s.crash.sum()), "viol15": int(s.viol15.sum()),
                       "mean_min_sep_ha": float(s.min_sep_ha.mean()), "min_min_sep_ha": float(s.min_sep_ha.min()),
                       "per_geometry_mean": {f"g{g:g}": float(x.min_sep_ha.mean()) for g, x in s.groupby("geometry")},
                       "effort_rms": float(s.effort_rms.mean())}
            summ[m]["eligible"] = summ[m]["route_complete"] >= 11 and summ[m]["crash"] == 0
        R["candidates"] = summ
        sel = {}
        for fam, cands in FAMILIES[stage]:
            c = {k: summ[k] for k in cands if k in summ}
            el = [k for k in c if c[k]["eligible"]]
            pick = max(el, key=lambda k: c[k]["mean_min_sep_ha"]) if el else max(c, key=lambda k: (c[k]["route_complete"], c[k]["mean_min_sep_ha"]))
            sel[fam] = {"selected": pick, "why": "eligible, largest mean min_sep_ha" if el else "none eligible: most completions, tie by min_sep"}
        R["selection"] = sel
        R["note"] = "development only (HOCBF candidates); planning SD per PLAN.md amendment A1"
        if stage == "dev2":
            R["sample_size_rule"] = {"N": 12, "note": "PLAN.md amendment A2: compute cap 12 = clamp floor; N fixed"}
            (out / "results.json").write_text(json.dumps(R, indent=1, default=str))
            print(json.dumps({k: R[k] for k in ("inventory", "instrument_pass", "selection", "candidates")}, indent=1, default=str)[:5000])
            return
        x = ok[ok.method == sel["RHOCBF"]["selected"]].groupby("seed").min_sep_ha.mean()
        s_dev = float(x.std(ddof=1)) if len(x) > 1 else math.nan
        v3 = pd.read_csv(ROOT / "output/research/rarf_plant_time_v3/confirm/episodes.csv")       # PLAN.md amendment A1
        v3 = v3[(v3.method == "RARF") & (v3.valid.astype(str).str.lower() == "true")].groupby("seed").min_sep_ha.mean()
        s_rarf = float(v3.std(ddof=1))
        s_star = max((s_dev if math.isfinite(s_dev) else 0.0) + s_rarf, 0.10)
        n_rule = math.ceil((1.96 * s_star / 0.10) ** 2)
        R["sample_size_rule"] = {"s_dev_RHOCBF": s_dev, "s_v3_RARF": s_rarf, "s_star": s_star, "n_rule": n_rule, "N": 12, "N_rule_original_clamp_12_15": int(min(max(n_rule, 12), 15)), "capped_by_A2": n_rule > 12}
    else:
        R["contrasts"] = A.contrasts(ok)
        R["binary"] = A.binaries(ok)
        R["flagged_block_sensitivity"] = A.contrasts(ok[~ok.seed.isin(set(E[~E.inst_pass.fillna(False).astype(bool)].seed))])
        R["continuous"] = {v: {f"g{g:g}|{m}": [A.q(s[v], .5), A.q(s[v], 0), A.q(s[v], 1)] for (g, m), s in ok.groupby(["geometry", "method"])}
                           for v in ("effort_rms", "t_cross", "dec_ms_p99", "slew_sent_max", "n_slew_sent_402", "min_sep_hh") if v in ok}
        R["deadline_misses"] = {m: float(s.miss_frac.mean()) for m, s in ok.groupby("method")} if "miss_frac" in ok else {}
    (out / "results.json").write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps({k: R[k] for k in R if k in ("inventory", "instrument_pass", "selection", "sample_size_rule", "candidates")}, indent=1, default=str)[:5000])


if __name__ == "__main__":
    run(sys.argv[1])
