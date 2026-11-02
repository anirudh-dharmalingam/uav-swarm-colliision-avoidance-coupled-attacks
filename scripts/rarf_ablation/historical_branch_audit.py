#!/usr/bin/env python3
"""Experiment III (historical part): branch / label accounting and command-chain trace from EXISTING coupled_safety_v1 logs.
Host python (stdlib + numpy).  Reads agent CSVs only; writes nothing into results/coupled_safety_v1.

    python3 scripts/rarf_ablation/historical_branch_audit.py --out output/research/rarf_ablation_v1/historical

Observable per honest encounter-window tick in the historical logs: status label (CERTIFIED_INTERVAL / UNCERTIFIED / INVALID_ASSUMPTIONS /
UNCERTIFIED_POSTCLIP), hard-solve failure reason (`fail`: k0_infeasible = zero-time necessary condition; solver/recheck reasons), assumption
flag, post-policy speed-envelope clip, policy output (ux,uy) and sent set-point (spx,spy).  NOT observable historically: whether the soft
fallback or the max-reserve rescue produced an uncertified command (no column) -> reported as unresolved.
Flags overlap; the table counts each flag independently and, separately, an exclusive hierarchy
  INVALID > K0_FAIL > OTHER_HARD_FAIL > POSTCLIP > CERTIFIED (declared here, used only for the stacked view).
"""
import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def ticks(rd, uid):
    f = rd / "agents" / f"agent_{uid}.csv.gz"
    return list(csv.DictReader(gzip.open(f, "rt")))


def audit_episode(rd):
    cfg = json.loads((rd / "resolved_config.json").read_text())
    att = int(cfg["ral26"]["attacker_id"])
    on = float(cfg["ral26"]["attack"]["start_s"])
    n = int(cfg["swarm"]["num_uavs"])
    C, X = Counter(), Counter()
    jumps = []
    for i in range(n):
        if i == att:
            continue
        R = [r for r in ticks(rd, i) if r["phase"] == "MISSION"]
        prev = None
        for r in R:
            t = float(r["t"])
            if not (on <= t <= on + 20.0):
                prev = r
                continue
            st, fail = r["status"], r["fail"]
            C["ticks"] += 1
            C["label_certified"] += st == "CERTIFIED_INTERVAL"
            C["label_invalid"] += st == "INVALID_ASSUMPTIONS"
            C["label_postclip"] += st == "UNCERTIFIED_POSTCLIP"
            C["assump_bad"] += r["assump_ok"] == "0"
            C["k0_fail"] += fail == "k0_infeasible"
            C["hard_fail_other"] += bool(fail) and fail != "k0_infeasible"
            C["hard_fail_any"] += bool(fail)
            C["clip"] += r["clip"] == "1"
            C["hard_ok_but_invalid"] += (not fail) and st == "INVALID_ASSUMPTIONS"
            ex = ("INVALID" if st == "INVALID_ASSUMPTIONS" else "K0_FAIL" if fail == "k0_infeasible" else
                  "OTHER_HARD_FAIL" if fail else "POSTCLIP" if st == "UNCERTIFIED_POSTCLIP" else
                  "CERTIFIED" if st == "CERTIFIED_INTERVAL" else "UNLABELLED_OTHER")
            X[ex] += 1
            if prev is not None:
                du = math.dist((float(prev["ux"]), float(prev["uy"])), (float(r["ux"]), float(r["uy"])))
                ds = math.dist((float(prev["spx"]), float(prev["spy"])), (float(r["spx"]), float(r["spy"])))
                if ds > 0.402 or du > 0.402:
                    jumps.append({"name": rd.name, "uid": i, "t": t, "d_policy": round(du, 4), "d_sent": round(ds, 4),
                                  "clip_prev": prev["clip"], "clip": r["clip"], "status": st, "fail": fail,
                                  "speed": round(math.hypot(float(r["vx"]), float(r["vy"])), 3)})
            prev = r
    return C, X, jumps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="e1,e3,e4,e5")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, alljumps = [], []
    for tag in a.sets.split(","):
        for rd in sorted((ROOT / "results/coupled_safety_v1" / tag).iterdir()):
            if not rd.is_dir() or rd.name.startswith("_") or not (rd / "eval/summary.json").exists():
                continue
            meth = json.loads((rd / "resolved_config.json").read_text())["defense"]["id"]
            if meth == "D0R":
                continue
            C, X, J = audit_episode(rd)
            alljumps += J
            rows.append({"tag": tag, "name": rd.name, "method": meth, **C, **{"x_" + k: v for k, v in X.items()}, "n_jumps": len(J)})
    keys = sorted({k for r in rows for k in r})
    with (out / "branch_accounting_by_episode.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    agg = {}
    for r in rows:
        k = f"{r['tag']}|{r['method']}"
        A = agg.setdefault(k, Counter())
        for kk, v in r.items():
            if isinstance(v, int):
                A[kk] += v
        A["episodes"] += 1
    (out / "branch_accounting_summary.json").write_text(json.dumps(agg, indent=1))
    jc = Counter()
    for j in alljumps:
        cause = ("sent_only_clip_onset" if j["d_policy"] <= 0.402 and j["clip"] == "1" and j["clip_prev"] == "0" else
                 "sent_only_clip_release" if j["d_policy"] <= 0.402 and j["clip"] == "0" and j["clip_prev"] == "1" else
                 "sent_only_clip_both" if j["d_policy"] <= 0.402 and j["clip"] == "1" else
                 "policy_output" if j["d_policy"] > 0.402 else "other")
        j["cause"] = cause
        jc[cause] += 1
    (out / "command_jumps_all.json").write_text(json.dumps(alljumps, indent=0))
    print(json.dumps(agg, indent=0)[:3000])
    print("jumps", len(alljumps), dict(jc))


if __name__ == "__main__":
    main()
