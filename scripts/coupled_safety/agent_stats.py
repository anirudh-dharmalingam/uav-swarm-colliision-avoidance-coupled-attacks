#!/usr/bin/env python3
"""Per-episode controller-side statistics from the agent logs (certification, reserve, latency, slew activity, information effect).

    python3 scripts/coupled_safety/agent_stats.py <results_root>/<tag> [...] --out agent_stats.csv        (inside the image)

For honest agents' MISSION ticks inside the encounter window [onset, onset+20 s]:
  cert_frac        share of ticks labelled CERTIFIED_INTERVAL;  postclip_frac = UNCERTIFIED_POSTCLIP;  invalid_frac = INVALID_ASSUMPTIONS
  cert_std_frac    share of ticks that WOULD have been certified under standard (report-only) information (E2 modes only)
  info_gain        share of ticks certified now but not under standard information; info_loss: the reverse
  gamma_*          authority reserve gamma (m) mean/min; gamma_std_* under standard information
All mission ticks (every honest agent): dec_ms (decision latency of the policy), solve_ms (whole control tick), e2e_ms (PX4-state age at command
publication), loop_dt (timer jitter) quantiles; slew activity = share of ticks whose command change equals the 0.4 m/s^2 per-tick limit; mean ||du||.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

QS = (0.5, 0.95, 0.99, 1.0)


def stats(rd: Path):
    cfg = json.loads((rd / "resolved_config.json").read_text())
    R = cfg["ral26"]
    att = int(R.get("attacker_id", -1))
    onset = float(R.get("attack", {}).get("start_s", 10.0))
    rows, win = [], []
    for f in sorted(rd.glob("agents/agent_*.csv.gz")):
        i = int(f.name.split("_")[1].split(".")[0])
        if i == att:
            continue
        d = pd.read_csv(f)
        d = d[d.phase == "MISSION"].copy()
        d["uid"] = i
        rows.append(d)
    if not rows:
        return {"name": rd.name, "ok": False}
    D = pd.concat(rows, ignore_index=True)
    t = D.t.astype(float)
    W = D[(t >= onset) & (t <= onset + 20.0)]
    out = {"name": rd.name, "ok": True, "ticks": int(len(D)), "win_ticks": int(len(W))}
    st = W.status
    out.update(cert_frac=float((st == "CERTIFIED_INTERVAL").mean()), postclip_frac=float((st == "UNCERTIFIED_POSTCLIP").mean()),
               invalid_frac=float((st == "INVALID_ASSUMPTIONS").mean()), gamma_mean=float(W.gamma.astype(float).replace([np.inf, -np.inf], np.nan).mean()),
               gamma_min=float(W.gamma.astype(float).replace([np.inf, -np.inf], np.nan).min()))
    cs = pd.to_numeric(W.get("cert_std"), errors="coerce") if "cert_std" in W else pd.Series(dtype=float)
    if cs.notna().any():
        cur = (st == "CERTIFIED_INTERVAL").astype(float)
        ok = cs.notna()
        out.update(cert_std_frac=float(cs[ok].mean()), info_gain=float(((cur == 1) & (cs == 0))[ok].mean()), info_loss=float(((cur == 0) & (cs == 1))[ok].mean()),
                   gamma_std_mean=float(pd.to_numeric(W.gamma_std, errors="coerce").replace([np.inf, -np.inf], np.nan).mean()))
    for k in ("dec_ms", "solve_ms", "e2e_ms", "loop_dt"):
        x = pd.to_numeric(D[k], errors="coerce").dropna()
        for q in QS:
            out[f"{k}_q{int(q * 100)}"] = float(x.quantile(q)) if q < 1 else float(x.max())
    # slew activity per agent
    act, dus = [], []
    for i, g in D.groupby("uid"):
        ux, uy = g.ux.astype(float).to_numpy(), g.uy.astype(float).to_numpy()
        du = np.hypot(np.diff(ux), np.diff(uy))
        act.append(float((du >= 0.4 - 1e-6).mean()))
        dus.append(float(du.mean()))
    out.update(slew_limit_frac=float(np.mean(act)), mean_du=float(np.mean(dus)))
    nfused = ncontra = 0
    clip = 0
    miss = 0
    for f in rd.glob("agents/agent_*_summary.json"):
        s = json.loads(f.read_text())
        c = s.get("cs", {})
        nfused += c.get("fused", 0)
        ncontra += c.get("contradictions", 0)
        clip += c.get("clip_count", 0)
        miss += s.get("deadline_miss", 0)
    out.update(fused=nfused, contradictions=ncontra, clip_count=clip, deadline_miss=miss)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = []
    for d in a.dirs:
        for rd in sorted(p for p in Path(d).iterdir() if p.is_dir() and not p.name.startswith("_")):
            if (rd / "eval" / "summary.json").exists():
                try:
                    r = stats(rd)
                except Exception as exc:
                    r = {"name": rd.name, "ok": False, "why": repr(exc)}
                r["tag"] = Path(d).name
                res.append(r)
    pd.DataFrame(res).to_csv(a.out, index=False)
    print(len(res), "->", a.out)
