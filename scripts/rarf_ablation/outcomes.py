#!/usr/bin/env python3
"""rarf_ablation_v1 per-episode outcomes (truth from the independent evaluator; agent logs only for command/branch/compute accounting).

    python3 scripts/rarf_ablation/outcomes.py results/rarf_ablation_v1/<tag> [...] --out episodes.csv      (inside the image: pandas)

Reuses scripts/coupled_safety/metrics.py:episode (frozen definitions: encounter window [onset, onset+20 s], min_sep_ha, min_sep_hh,
1.5 m violation, 0.8 m proximity, tilt > 60 deg crash proxy, registered route_complete, safe_task) and adds:
  dwell_complete   registered route completion AND the honest centroid stays in the goal region (ordered progress >= |route| - 1 m)
                   for >= 5 s contiguous before 100 s (separately named; not the registered endpoint)
  t_cross          goal-plane crossing time [s from GCS start]; s_end ordered progress [m]
  effort_rms       RMS of the SENT planar acceleration set-point (spx, spy) over honest agents, encounter window [m/s^2]
  slew_sent_max    max over honest agents/window of |sp_k - sp_{k-1}| between consecutive SENT commands [m/s^2/tick]
  n_slew_sent_402  count of such jumps > 0.402 (0.4 slew + 0.002 tolerance); slew_opt_max / n_slew_opt_402 use the policy output (ux, uy)
  branch shares    over honest encounter-window ticks: applied hard / soft / rescue, hypothetical hard_ok, k0 failure, assumption
                   invalid, executed-rows-hold, post-policy speed-envelope clip, planar change (overlapping flags, no precedence)
  dec_ms p50/p99   policy decision time; tick_gap_max max host gap between decisions; rtf_mean from the evaluator
  truth_gap_max    max sample gap of the truth stream inside the encounter window (continuous-time minima unresolved below it)
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "coupled_safety"))
import metrics  # noqa: E402

PLANNED = ["min_sep_ha", "min_sep_hh", "viol15", "prox08", "crash", "route_complete", "s_end", "dwell_complete", "t_cross",
           "effort_rms", "slew_sent_max", "n_slew_sent_402", "dec_ms_p50", "dec_ms_p99", "tx_bytes", "truth_gap_max"]
BRANCH = ["sh_applied_hard", "sh_applied_soft", "sh_applied_rescue", "sh_hard_ok", "sh_k0_fail", "sh_assump_bad",
          "sh_exec_rows_ok", "sh_clip", "sh_planar_chg", "sh_spd_brake", "n_win_ticks"]
FIELDS = ["name", "tag", "valid", "method", "variant", "cell", "geometry", "seed"] + PLANNED + BRANCH + \
         ["slew_opt_max", "n_slew_opt_402", "harm", "dur_below15_s", "rtf_mean", "tick_gap_max", "transit_max", "deadline_miss",
          "hard_ms_p50", "soft_ms_p50", "final_phases", "why"]


def read_agent(f):
    raw = f.read_bytes()
    if f.suffix == ".gz":
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


def dwell(rd, ev, L, att):
    """Contiguous seconds with ordered progress >= L - 1 (honest centroid along the route), from the truth stream."""
    cfg = json.loads((rd / "resolved_config.json").read_text())
    n = int(cfg["swarm"]["num_uavs"])
    T = pd.read_csv(rd / "eval" / "truth.csv")
    t = T.t_sim.to_numpy() - ev["t_mission_start_sim"]
    hon = [i for i in range(n) if i != att]
    cen = np.stack([T[[f"x{i}", f"y{i}"]].to_numpy() for i in hon], 1).mean(axis=1)
    route = np.array(cfg["ral26"]["route"], float)
    u = route / np.linalg.norm(route)
    ms = t >= 0
    if not ms.any():
        return 0.0, math.nan
    c0 = cen[np.argmax(ms)]
    s = (cen - c0) @ u
    inside = (s >= L - 1.0) & ms & (t <= 100.0)
    best = cur = 0.0
    for k in range(1, len(t)):
        cur = cur + (t[k] - t[k - 1]) if inside[k] and inside[k - 1] else 0.0
        best = max(best, cur)
    on = float(cfg["ral26"]["attack"]["start_s"])
    w = (t >= on) & (t <= on + metrics.ENC)
    gap = float(np.max(np.diff(t[w]))) if w.sum() > 1 else math.nan
    return best, gap


def episode(rd: Path):
    r = metrics.episode(rd)
    if not r.get("valid"):
        return r
    cfg = json.loads((rd / "resolved_config.json").read_text())
    ev = json.loads((rd / "eval" / "summary.json").read_text())
    att, on = r["attacker"], r["onset"]
    L = float(np.linalg.norm(cfg["ral26"]["route"]))
    dw, gap = dwell(rd, ev, L, att)
    r.update(dwell_s=dw, dwell_complete=int(bool(r["route_complete"]) and dw >= 5.0), truth_gap_max=gap)
    eff, sl_s, sl_o, n_s, n_o, dec, hard, soft, gaps = [], [], [], 0, 0, [], [], [], []
    cnt = {k: 0 for k in BRANCH}
    for i in range(r["n"]):
        if i == att:
            continue
        f = next((p for p in (rd / "agents" / f"agent_{i}.csv.gz", rd / "agents" / f"agent_{i}.csv") if p.exists()), None)
        if f is None:
            continue
        A = read_agent(f)
        A = A[A.phase == "MISSION"] if "phase" in A else A
        if A.empty:
            continue
        w = (A.t >= on) & (A.t <= on + metrics.ENC)
        sp = A[["spx", "spy"]].to_numpy(float)
        uo = A[["ux", "uy"]].to_numpy(float)
        ds, do = np.linalg.norm(np.diff(sp, axis=0), axis=1), np.linalg.norm(np.diff(uo, axis=0), axis=1)
        wk = w.to_numpy()[1:]
        sl_s.append(ds[wk].max() if wk.any() else math.nan)
        sl_o.append(do[wk].max() if wk.any() else math.nan)
        n_s += int((ds[wk] > 0.402).sum())
        n_o += int((do[wk] > 0.402).sum())
        eff.append(np.sum(sp[w.to_numpy()] ** 2, axis=1))
        dec.append(pd.to_numeric(A.dec_ms, errors="coerce").to_numpy())
        gaps.append(np.diff(A.t.to_numpy()))
        W = A[w]
        cnt["n_win_ticks"] += len(W)
        if "applied" in W and W["applied"].notna().any():
            cnt["sh_applied_hard"] += int((W.applied == "hard").sum())
            cnt["sh_applied_soft"] += int((W.applied == "soft").sum())
            cnt["sh_applied_rescue"] += int((W.applied == "rescue").sum())
            cnt["sh_hard_ok"] += int((pd.to_numeric(W.hard_ok, errors="coerce") == 1).sum())
            cnt["sh_k0_fail"] += int((pd.to_numeric(W.k0_fail, errors="coerce") == 1).sum())
            cnt["sh_exec_rows_ok"] += int((pd.to_numeric(W.exec_rows_ok, errors="coerce") == 1).sum())
            cnt["sh_planar_chg"] += int((pd.to_numeric(W.planar_chg, errors="coerce") == 1).sum())
            hard.append(pd.to_numeric(W.hard_ms, errors="coerce").to_numpy())
            soft.append(pd.to_numeric(W.soft_ms, errors="coerce").to_numpy())
        if "assump_ok" in W:
            cnt["sh_assump_bad"] += int((pd.to_numeric(W.assump_ok, errors="coerce") == 0).sum())
        if "clip" in W:
            cnt["sh_clip"] += int((pd.to_numeric(W["clip"], errors="coerce") == 1).sum())
        if "spd_brake" in W:
            cnt["sh_spd_brake"] += int((pd.to_numeric(W.spd_brake, errors="coerce") == 1).sum())
    cat = lambda xs: np.concatenate(xs) if xs else np.array([])  # noqa: E731
    d, h, s = cat(dec), cat(hard), cat(soft)
    r.update(effort_rms=float(np.sqrt(np.mean(cat(eff)))) if eff else math.nan,
             slew_sent_max=float(np.nanmax(sl_s)) if sl_s else math.nan, slew_opt_max=float(np.nanmax(sl_o)) if sl_o else math.nan,
             n_slew_sent_402=n_s, n_slew_opt_402=n_o,
             dec_ms_p50=float(np.nanquantile(d, .5)) if d.size else math.nan, dec_ms_p99=float(np.nanquantile(d, .99)) if d.size else math.nan,
             hard_ms_p50=float(np.nanquantile(h, .5)) if h.size and np.isfinite(h).any() else math.nan,
             soft_ms_p50=float(np.nanquantile(s, .5)) if s.size and np.isfinite(s).any() else math.nan,
             tick_gap_max=float(cat(gaps).max()) if gaps else math.nan, **cnt)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--spec", action="append", default=[], help="spec json(s): adds meta and lists planned-but-missing episodes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = {}
    for s in a.spec:
        for e in json.loads(Path(s).read_text())["episodes"]:
            meta[e["name"]] = e["meta"]
    rows, seen = [], set()
    for d in a.dirs:
        for rd in sorted(p for p in Path(d).iterdir() if p.is_dir() and not p.name.startswith("_")):
            if not (rd / "eval" / "summary.json").exists():
                rows.append({"name": rd.name, "valid": False, "why": "no eval/summary.json"})
            else:
                try:
                    rows.append(episode(rd))
                except Exception as exc:
                    rows.append({"name": rd.name, "valid": False, "why": repr(exc)})
            rows[-1]["tag"] = Path(d).name
            seen.add(rd.name)
    for nm in meta:
        if nm not in seen:
            rows.append({"name": nm, "valid": False, "why": "planned, not attempted"})
    for r in rows:
        m = meta.get(r["name"], {})
        r.update(method=m.get("method"), variant=m.get("variant"), cell=m.get("cell"), geometry=m.get("geometry"), seed=m.get("seed"))
    df = pd.DataFrame(rows)
    df = df[[c for c in FIELDS if c in df] + [c for c in df if c not in FIELDS]]
    df.to_csv(a.out, index=False)
    print(len(df), "rows ->", a.out, "| valid", int(df.valid.fillna(False).astype(bool).sum()))


if __name__ == "__main__":
    main()
