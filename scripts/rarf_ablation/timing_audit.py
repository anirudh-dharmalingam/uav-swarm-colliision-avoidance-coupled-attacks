#!/usr/bin/env python3
"""Experiment III (new runs): execution-timing audit on host-monotonic and Gazebo plant time (inside the image: pandas).

    python3 scripts/rarf_ablation/timing_audit.py results/rarf_ablation_v1/<tag> [...] --out timing.csv

Clock map: eval/clock_map.csv (evaluator; sim stamp of the newest kept pose vs host wall and monotonic time AT RECEIPT).  The map
therefore lags plant time by the Gazebo->evaluator transport latency, which is NOT measured; plant times below are "plant time of the
newest pose the host had received", an approximation whose error is that latency.
Per honest agent tick (rarf_ablation_v1 columns): t_mono decision start, t_state_rx host receipt of the PX4 estimate, t_px4 PX4 estimate
timestamp, t_pub host time after publishing the set-point.  t_px4 is PX4 lockstep (plant) time plus a uXRCE-DDS timesync offset that is
piecewise constant with occasional steps (development finding: offset spread p1-p50 within 25 ms, steps of 0.4-5 s), so it is NOT host
wall time and its absolute offset (which includes transport latency) is unknown; only delivery jitter around the running offset is
measurable.
Reported per episode (honest agents, mission phase):
  plant_period_p50/p99/max   plant time between consecutive decisions (command hold as seen by the plant until the next publication) [s]
  host_period_p99/max        host time between decisions [s]
  state_jitter_p01/p99       (t_px4 - plant(t_state_rx)) minus its 2 s rolling median: delivery-latency jitter of the PX4 estimate [s]
  timesync_steps             number of steps > 0.2 s in that rolling median (PX4 timestamp offset re-syncs)
  cb_to_dec_plant_p50/p99/max   plant time from host receipt of the estimate to the decision that used it [s]
  dec_to_pub_host_p99/max    t_pub - t_mono [s]
  rtf_local_max/min          max/min over decisions of plant/host elapsed over the packet-age interval s = worst_age - T_net_max
  age_undercover_frac        fraction of decisions with plant-elapsed(s) > s, i.e. host-measured packet age SMALLER than plant age
                             (only possible when the simulator runs faster than real time locally)
NOT instrumentable without modifying PX4: controller acceptance and physical application of the set-point inside PX4.
"""
import argparse
import gzip
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

Q = lambda x, q: float(np.nanquantile(x, q)) if len(x) and np.isfinite(x).any() else math.nan  # noqa: E731


def episode(rd: Path):
    cm = rd / "eval" / "clock_map.csv"
    if not cm.exists():
        return {"name": rd.name, "clock_map": False}
    C = pd.read_csv(cm).drop_duplicates("mono")
    if len(C) < 10:
        return {"name": rd.name, "clock_map": False}
    cfg = json.loads((rd / "resolved_config.json").read_text())
    att = int(cfg["ral26"]["attacker_id"])
    T = float(cfg["ral26"]["bounds"]["T_net_max"])
    mono2sim = lambda x: np.interp(x, C.mono.to_numpy(), C.t_sim.to_numpy(), left=np.nan, right=np.nan)  # noqa: E731
    acc = {k: [] for k in ("pp", "hp", "jit", "cd", "dp", "rl", "uc")}
    steps = 0
    for i in range(int(cfg["swarm"]["num_uavs"])):
        if i == att:
            continue
        f = rd / "agents" / f"agent_{i}.csv.gz"
        if not f.exists():
            f = rd / "agents" / f"agent_{i}.csv"
        if not f.exists():
            continue
        raw = f.read_bytes()
        A = pd.read_csv(io.BytesIO(gzip.decompress(raw) if f.suffix == ".gz" else raw), low_memory=False)
        if "t_mono" not in A:
            continue
        A = A[(A.phase == "MISSION") & A.t_mono.notna()]
        tm = A.t_mono.to_numpy(float)
        sim_d = mono2sim(tm)
        acc["pp"].append(np.diff(sim_d))
        acc["hp"].append(np.diff(tm))
        off = pd.Series(A.t_px4.to_numpy(float) - mono2sim(A.t_state_rx.to_numpy(float)))
        med = off.rolling(41, center=True, min_periods=5).median()
        acc["jit"].append((off - med).to_numpy())
        steps += int((med.diff().abs() > 0.2).sum())
        acc["cd"].append(sim_d - mono2sim(A.t_state_rx.to_numpy(float)))
        acc["dp"].append(A.t_pub.to_numpy(float) - tm)
        s = np.clip(pd.to_numeric(A.worst_age, errors="coerce").to_numpy(float) - T, 0, None)
        m = s > 0.02
        pe = sim_d[m] - mono2sim(tm[m] - s[m])
        acc["rl"].append(pe / s[m])
        acc["uc"].append(pe > s[m] + 1e-3)
    cat = {k: np.concatenate(v) if v else np.array([]) for k, v in acc.items()}
    return {"name": rd.name, "clock_map": True, "n_ticks": int(len(cat["hp"])),
            "plant_period_p50": Q(cat["pp"], .5), "plant_period_p99": Q(cat["pp"], .99), "plant_period_max": Q(cat["pp"], 1),
            "host_period_p99": Q(cat["hp"], .99), "host_period_max": Q(cat["hp"], 1),
            "state_jitter_p01": Q(cat["jit"], .01), "state_jitter_p99": Q(cat["jit"], .99), "timesync_steps": steps,
            "cb_to_dec_plant_p50": Q(cat["cd"], .5), "cb_to_dec_plant_p99": Q(cat["cd"], .99), "cb_to_dec_plant_max": Q(cat["cd"], 1),
            "dec_to_pub_host_p99": Q(cat["dp"], .99), "dec_to_pub_host_max": Q(cat["dp"], 1),
            "rtf_local_min": Q(cat["rl"], 0), "rtf_local_max": Q(cat["rl"], 1),
            "age_undercover_frac": float(np.mean(cat["uc"])) if len(cat["uc"]) else math.nan}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for d in a.dirs:
        for rd in sorted(p for p in Path(d).iterdir() if p.is_dir() and not p.name.startswith("_")):
            if (rd / "eval" / "summary.json").exists():
                try:
                    rows.append(episode(rd))
                except Exception as exc:
                    rows.append({"name": rd.name, "error": repr(exc)})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(len(rows), "->", a.out)


if __name__ == "__main__":
    main()
