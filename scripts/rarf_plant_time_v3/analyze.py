#!/usr/bin/env python3
"""rarf_plant_time_v3 analysis (experiments/rarf_plant_time_v3/FROZEN_PLAN.md §5-§8).  Inside the image (pandas):

    python3 scripts/rarf_plant_time_v3/analyze.py results/rarf_plant_time_v3/pilot --spec experiments/rarf_plant_time_v3/specs/pilot.json \
        --out output/research/rarf_plant_time_v3/pilot --pilot
    python3 scripts/rarf_plant_time_v3/analyze.py results/rarf_plant_time_v3/confirm --spec .../confirm.json --out .../confirm

Per episode: frozen endpoints from scripts/rarf_ablation/outcomes.py:episode (unchanged import), and instrument metrics from the v2
node's own clock columns (honest agents, MISSION phase, scored window = agent plant mission time in [start_s, start_s + 20]):
  rtf_mean (evaluator), miss_frac, skip_frac, dec_per_agent_plant_s, ref_speed (slope of s_ref vs plant mission time while s_ref < L),
  onset_err (first attacker attack_on = 1 minus start_s), pub_slack_p99 (t_plant_pub - t_plant_sched), hold_p99 / hold_max (plant time
  between publications), clock_err_p50 / p99 (agent plant estimate minus evaluator clock map at the same host instant; descriptive),
  label counts (label_final), hard_changed (applied hard with dev_hard_fb > 1e-3), dec_ms p99.
Instrument pass = criteria 1-5 of the plan.  Contrasts: cluster bootstrap over scenario blocks, 10,000 resamples, numpy seed 20261031.
"""
import argparse
import gzip
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "rarf_ablation"))
import outcomes as O  # noqa: E402  (frozen endpoints)

NB, SEED = 10000, 20261031
GEOMS, ARMS = (0.0, 2.0, 3.0), ("NOM", "RARF", "FF", "SPD")
S_EXP1_D2 = 0.197
RTF_TARGET = 0.3                     # amendment A2 (pilot 1 used 0.5)
S_PILOT1_D2 = 0.18107236812943048    # amendment A2: pilot-1 SD enters the planning SD
CONTRASTS = (("D2", "FF", "NOM", 0.95, "PRIMARY"), ("D3", "FF", "SPD", 0.975, "SECONDARY_bonferroni"),
             ("D1", "RARF", "FF", 0.975, "SECONDARY_bonferroni"))


def rd_csv(f):
    raw = f.read_bytes()
    return pd.read_csv(io.BytesIO(gzip.decompress(raw) if f.suffix == ".gz" else raw), low_memory=False)


def q(x, p):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, p)) if x.size else math.nan


def instrument(rd: Path):
    cfg = json.loads((rd / "resolved_config.json").read_text())
    R = cfg["ral26"]
    att, n = int(R["attacker_id"]), int(cfg["swarm"]["num_uavs"])
    on = float(R["attack"]["start_s"])
    L = float(np.linalg.norm(R["route"]))
    summ = json.loads((rd / "eval" / "summary.json").read_text())
    cm = pd.read_csv(rd / "eval" / "clock_map.csv").drop_duplicates("mono") if (rd / "eval" / "clock_map.csv").exists() else None
    out = {"name": rd.name, "rtf_mean_eval": summ.get("rtf_mean")}
    acc = {k: [] for k in ("slack", "hold", "clk", "dec_ms")}
    n_dec = n_miss = n_skip = 0
    mis_s = 0.0
    slopes, n_ag = [], 0
    labels = {}
    hard_changed = hard = 0
    for i in range(n):
        f = next((p for p in (rd / "agents" / f"agent_{i}.csv.gz", rd / "agents" / f"agent_{i}.csv") if p.exists()), None)
        if f is None:
            out[f"agent_{i}_log"] = "missing"
            continue
        A = rd_csv(f)
        A = A[A.phase == "MISSION"]
        if i == att:
            a = A[pd.to_numeric(A.attack_on, errors="coerce") == 1]
            out["onset_logged"] = float(a.t.iloc[0]) if len(a) else math.nan
            out["onset_err"] = out["onset_logged"] - on if len(a) else math.nan
            continue
        if "t_plant_sched" not in A:
            out["clock_columns"] = "absent"
            continue
        n_ag += 1
        W = A[(A.t >= on) & (A.t <= on + 20.0)]
        n_dec += len(W)
        if len(W) > 1:                                   # MISSION-phase plant time inside the window (agents may land before its end)
            mis_s += float(min(on + 20.0, A.t.iloc[-1]) - max(on, A.t.iloc[0]))
        n_miss += int(pd.to_numeric(W.miss, errors="coerce").fillna(0).sum())
        n_skip += int(pd.to_numeric(W.sched_skip, errors="coerce").fillna(0).sum())
        tp = pd.to_numeric(W.t_plant_pub, errors="coerce").to_numpy(float)
        ts = pd.to_numeric(W.t_plant_sched, errors="coerce").to_numpy(float)
        acc["slack"].append(tp - ts)
        acc["hold"].append(np.diff(tp))
        acc["dec_ms"].append(pd.to_numeric(W.dec_ms, errors="coerce").to_numpy(float))
        if cm is not None and len(cm) > 10:
            th = pd.to_numeric(W.t_host_dec, errors="coerce").to_numpy(float)
            mp = np.interp(th, cm.mono.to_numpy(float), cm.t_sim.to_numpy(float), left=np.nan, right=np.nan)
            pd_ = pd.to_numeric(W.t_plant_dec, errors="coerce").to_numpy(float)
            acc["clk"].append(pd_ - mp)
        s = pd.to_numeric(A.s_ref, errors="coerce").to_numpy(float)
        t = A.t.to_numpy(float)
        m = (s > 0.5) & (s < L - 0.5)
        if m.sum() > 20:
            slopes.append(float(np.polyfit(t[m], s[m], 1)[0]))
        for lab, c in W.label_final.astype(str).value_counts().items():
            labels[lab] = labels.get(lab, 0) + int(c)
        h = W[W.applied.astype(str) == "hard"]
        hard += len(h)
        hard_changed += int((pd.to_numeric(h.dev_hard_fb, errors="coerce") > 1e-3).sum())
    cat = {k: np.concatenate(v) if v else np.array([]) for k, v in acc.items()}
    # the clock map stamps the sim time of the newest pose at host receipt: it lags plant time by the (unmeasured) transport latency;
    # its offset to the agent estimate is reported after removing the episode median (jitter) and as the median itself (offset)
    clk = cat["clk"][np.isfinite(cat["clk"])]
    out.update(n_honest_logs=n_ag, n_dec_win=n_dec, miss_frac=n_miss / max(n_dec, 1), skip_frac=n_skip / max(n_dec + n_skip, 1),
               dec_per_agent_plant_s=n_dec / max(mis_s, 1e-9), mission_plant_s_in_window=mis_s / max(n_ag, 1), ref_speed=float(np.median(slopes)) if slopes else math.nan,
               pub_slack_p50=q(cat["slack"], .5), pub_slack_p99=q(cat["slack"], .99), pub_slack_max=q(cat["slack"], 1),
               hold_p99=q(cat["hold"], .99), hold_max=q(cat["hold"], 1), dec_ms_p99=q(cat["dec_ms"], .99),
               clock_offset_median=float(np.median(clk)) if clk.size else math.nan,
               clock_jitter_p99=q(np.abs(clk - np.median(clk)), .99) if clk.size else math.nan,
               hard_applied=hard, hard_changed_command=hard_changed, labels=json.dumps(labels, sort_keys=True))
    r = out
    r["inst_rtf"] = bool(r["rtf_mean_eval"] is not None and abs(r["rtf_mean_eval"] - RTF_TARGET) <= 0.05)
    r["inst_miss"] = bool(r["miss_frac"] < 0.001 and r["skip_frac"] < 0.001)
    r["inst_rate"] = bool(19.8 <= r["dec_per_agent_plant_s"] <= 20.2)
    r["inst_ref"] = bool(abs(r["ref_speed"] - 1.0) <= 0.01) if math.isfinite(r["ref_speed"]) else False
    r["inst_onset"] = bool(0.0 <= r.get("onset_err", math.nan) <= 0.05 + 0.002)     # one tick + 1 ms log rounding of each time (A2)
    r["inst_pass"] = all(r[k] for k in ("inst_rtf", "inst_miss", "inst_rate", "inst_ref", "inst_onset"))
    return r


def cboot(M, fn, level):
    rng = np.random.default_rng(SEED)
    n = M.shape[0]
    bs = np.array([fn(M[rng.integers(0, n, n)]) for _ in range(NB)])
    a = (1 - level) / 2
    return [float(fn(M)), float(np.nanquantile(bs, a)), float(np.nanquantile(bs, 1 - a))]


def cube(df, var):
    seeds = sorted(df.seed.dropna().unique())
    X = np.full((len(seeds), len(GEOMS), len(ARMS)), np.nan)
    for (s, g, a), v in df.set_index(["seed", "geometry", "method"])[var].items():
        if g in GEOMS and a in ARMS:
            X[seeds.index(s), GEOMS.index(g), ARMS.index(a)] = v
    keep = ~np.isnan(X).reshape(len(seeds), -1).any(axis=1)
    return X[keep], [int(s) for s, k in zip(seeds, keep) if k], len(seeds)


def contrasts(df, var="min_sep_ha"):
    X, seeds, n_all = cube(df, var)
    ai = {a: k for k, a in enumerate(ARMS)}
    R = {"n_blocks": len(seeds), "n_scenarios": n_all, "seeds": seeds}
    if not len(seeds):
        return R
    for lab, a, b, lev, role in CONTRASTS:
        R[f"{lab}_{a}-{b}"] = {"role": role, "level": lev,
                               "est_lo_hi": cboot(X, lambda M, a=a, b=b: float(np.mean(M[:, :, ai[a]] - M[:, :, ai[b]])), lev),
                               "per_scenario": [round(float(x), 4) for x in np.mean(X[:, :, ai[a]] - X[:, :, ai[b]], axis=1)]}
        for k, g in enumerate(GEOMS):
            R[f"g{g:g}|{a}-{b}"] = cboot(X, lambda M, a=a, b=b, k=k: float(np.mean(M[:, k, ai[a]] - M[:, k, ai[b]])), 0.95)
    for k, g in enumerate(GEOMS):
        R[f"g{g:g}|RARF-NOM"] = cboot(X, lambda M, k=k: float(np.mean(M[:, k, 1] - M[:, k, 0])), 0.95)
        for a in ARMS:
            R[f"g{g:g}|{a}|mean_min"] = [float(np.mean(X[:, k, ai[a]])), float(np.min(X[:, k, ai[a]]))]
    R["RARF-NOM"] = cboot(X, lambda M: float(np.mean(M[:, :, 1] - M[:, :, 0])), 0.95)
    return R


def binaries(df):
    out = {}
    ai = {a: k for k, a in enumerate(ARMS)}
    for var in ("viol15", "prox08", "crash", "route_complete", "dwell_complete"):
        Y, _, _ = cube(df, var)
        if not len(Y):
            continue
        d = {}
        for k, g in enumerate(GEOMS):
            for a in ARMS:
                d[f"g{g:g}|{a}"] = f"{int(np.nansum(Y[:, k, ai[a]]))}/{Y.shape[0]}"
            for a, b in (("FF", "NOM"), ("FF", "SPD"), ("RARF", "FF")):
                d[f"g{g:g}|{a}-{b}|risk_diff"] = cboot(Y, lambda M, a=a, b=b, k=k: float(np.mean(M[:, k, ai[a]] - M[:, k, ai[b]])), 0.95)
                d[f"g{g:g}|{a}-{b}|discordant"] = [int(((Y[:, k, ai[a]] == 1) & (Y[:, k, ai[b]] == 0)).sum()),
                                                   int(((Y[:, k, ai[a]] == 0) & (Y[:, k, ai[b]] == 1)).sum())]
        out[var] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag_dir")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    td, out = Path(a.tag_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(a.spec).read_text())
    meta = {e["name"]: e["meta"] for e in spec["episodes"]}
    rows, inst = [], []
    for rd in sorted(p for p in td.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if not (rd / "eval" / "summary.json").exists():
            rows.append({"name": rd.name, "valid": False, "why": "no eval/summary.json (telemetry missing)"})
            continue
        try:
            rows.append(O.episode(rd))
        except Exception as exc:
            rows.append({"name": rd.name, "valid": False, "why": repr(exc)})
        try:
            inst.append(instrument(rd))
        except Exception as exc:
            inst.append({"name": rd.name, "inst_error": repr(exc), "inst_pass": False})
    seen = {r["name"] for r in rows}
    for nm in meta:
        if nm not in seen:
            rows.append({"name": nm, "valid": False, "why": "planned, not attempted"})
    E = pd.DataFrame(rows)
    for k in ("method", "geometry", "seed", "order"):
        E[k] = E.name.map(lambda x, k=k: meta.get(x, {}).get(k))
    I = pd.DataFrame(inst)
    E = E.merge(I, on="name", how="left")
    E["valid"] = E.valid.astype(str).str.lower() == "true"
    if "why" not in E:
        E["why"] = ""
    E.to_csv(out / "episodes.csv", index=False)
    ledger = []
    for sub in ("", "_failed_attempts"):
        base = td / sub if sub else td
        if base.exists():
            for rd in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")):
                ledger.append({"name": rd.name, "attempt": "final" if not sub else "failed_first_attempt",
                               "has_summary": (rd / "eval" / "summary.json").exists(), "order": meta.get(rd.name, {}).get("order")})
    pd.DataFrame(ledger).to_csv(out / "attempt_ledger.csv", index=False)
    ok = E[E.valid]
    R = {"inventory": {"planned": len(meta), "rows": len(E), "analysable": int(E.valid.sum()),
                       "non_analysable": E[~E.valid][["name", "why"]].to_dict("records")}}
    icols = ["rtf_mean_eval", "miss_frac", "skip_frac", "dec_per_agent_plant_s", "ref_speed", "onset_err", "pub_slack_p99", "hold_p99",
             "hold_max", "dec_ms_p99", "clock_offset_median", "clock_jitter_p99"]
    R["instrument_by_arm"] = {m: {c: [q(g[c], 0), q(g[c], .5), q(g[c], 1)] for c in icols if c in g} for m, g in E.groupby("method")}
    R["instrument_pass"] = {m: f"{int(g.inst_pass.fillna(False).astype(bool).sum())}/{len(g)}" for m, g in E.groupby("method")}
    R["instrument_fail_criteria"] = {c: int((~E[c].fillna(False).astype(bool)).sum()) for c in
                                     ("inst_rtf", "inst_miss", "inst_rate", "inst_ref", "inst_onset") if c in E}
    if a.pilot:
        d2 = contrasts(ok)
        s_p = float(np.std(d2["D2_FF-NOM"]["per_scenario"], ddof=1)) if d2.get("n_blocks", 0) > 1 else math.nan
        s_star = max(v for v in (s_p, S_PILOT1_D2, S_EXP1_D2, 0.10) if math.isfinite(v))
        n_rule = math.ceil((1.96 * s_star / 0.10) ** 2)
        R["sample_size_rule"] = {"s_pilot_D2": s_p, "s_exp1_D2": S_EXP1_D2, "s_star": s_star, "n_rule": n_rule, "N": int(min(max(n_rule, 12), 20)),
                                 "capped": n_rule > 20, "expected_halfwidth_at_N": 1.96 * s_star / math.sqrt(min(max(n_rule, 12), 20))}
        R["note"] = "pilot: instrument check and variance only; contrasts are not reported"
    else:
        R["primary_and_secondary"] = contrasts(ok)
        R["binary"] = binaries(ok)
        R["flagged_block_sensitivity"] = contrasts(ok[~ok.seed.isin(set(E[~E.inst_pass.fillna(False).astype(bool)].seed))])
        W = E[E.method.notna()].copy()
        W.loc[~W.valid, "min_sep_ha"] = 0.0
        R["nonanalysable_as_zero"] = {k: v for k, v in contrasts(W).items() if k.startswith("D") or k.startswith("n_")}
        cont = {}
        for var in ("min_sep_hh", "s_end", "t_cross", "effort_rms", "slew_sent_max", "n_slew_sent_402", "dec_ms_p99", "rtf_mean"):
            if var in ok:
                cont[var] = {f"g{g:g}|{m}": [q(s[var], .5), q(s[var], 0), q(s[var], 1)] for (g, m), s in ok.groupby(["geometry", "method"])}
        R["continuous_median_min_max"] = cont
        br = {}
        for c in [c for c in ok if c.startswith("sh_")] + ["hard_applied", "hard_changed_command"]:
            if c in ok:
                br[c] = {f"g{g:g}|{m}": f"{int(s[c].sum())}/{int(s.n_win_ticks.sum())}" for (g, m), s in ok.groupby(["geometry", "method"])}
        R["branch_counts"] = br
        R["labels"] = {f"g{g:g}|{m}": json.dumps({k: int(sum(json.loads(x).get(k, 0) for x in s.labels.dropna()))
                                                 for k in sorted({k for x in s.labels.dropna() for k in json.loads(x)})})
                       for (g, m), s in ok.groupby(["geometry", "method"])}
    (out / "results.json").write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps({k: R[k] for k in R if k not in ("binary", "continuous_median_min_max", "branch_counts")}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
