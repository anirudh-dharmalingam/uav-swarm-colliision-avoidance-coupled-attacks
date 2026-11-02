#!/usr/bin/env python3
"""Gate A old-vs-corrected packet-age enclosure audit on ARCHIVED PX4 flights (evaluation only).

The archived RAL26 v2 runs did NOT log delivered packets, so the per-tick envelopes cannot be reconstructed exactly.  This replay
therefore RE-PACKETIZES the logged trajectories: for every audited (receiver i, sender j, tick t) a packet is synthesised from j's
logged broadcast content (its EKF estimate + the logged report bias) at send time t_s = t - s - l with receiver elapsed time s in
{0, 0.1, 0.2} s (5 Hz broadcast) and actual transit l in {0, T/2, T} (T = declared T_net_max), then BOTH enclosures are evaluated against
the Gazebo truth.  Result labels: 'unsound-event' = truth outside the enclosure;  'premise-valid' restricts to ticks where the
declared report/ego/acceleration premises held (only there is soundness a claim).  Certification labels (CERTIFIED under old vs new
rows, same nominal command) use the synthetic packets and are therefore RECONSTRUCTIONS, not recorded labels.

    python3 scripts/coupled_safety/gateA_replay_audit.py --root results/ral26/v2_main --glob 'A14_e2.5_b1.25_D24R_s9*' --out results/coupled_safety_v1/gateA
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import ral26_audit_v2 as AU  # noqa: E402
from swarmsec import ral26_core as rc  # noqa: E402


def episode(rd: Path, tick_stride: int):
    cfg = json.loads((rd / "resolved_config.json").read_text())
    R, B = cfg["ral26"], cfg["ral26"]["bounds"]
    n = int(cfg["swarm"]["num_uavs"])
    att = int(R.get("attacker_id", -1))
    ev = json.loads((rd / "eval" / "summary.json").read_text())
    t0 = ev.get("t_mission_start_sim")
    if t0 is None:
        return None
    T = pd.read_csv(rd / "eval" / "truth.csv")
    ts = T.t_sim.to_numpy() - t0
    dt = 0.02
    grid = np.arange(max(0.0, ts[0]), ts[-1], dt)
    win = max(5, int(round(0.2 / dt)) | 1)
    P = np.stack([np.stack([np.interp(grid, ts, T[f"{c}{i}"]) for c in "xyz"], 1) for i in range(n)], 1)
    V = np.stack([np.stack([AU.sg(P[:, i, k], win, 2, 1, dt) for k in range(3)], 1) for i in range(n)], 1)
    Acc = np.stack([np.stack([AU.sg(P[:, i, k], win, 2, 2, dt) for k in range(3)], 1) for i in range(n)], 1)
    amag = np.linalg.norm(Acc, axis=2)
    ag, lag = {}, {}
    for i in range(n):
        f = rd / "agents" / f"agent_{i}.csv.gz"
        if f.exists():
            d = pd.read_csv(f)
            d = d[d.phase == "MISSION"]
            if len(d) > 100:
                ag[i] = d.reset_index(drop=True)
                lag[i], _ = AU.align(d, grid, P, i)
    if len(ag) < n:
        return None
    L = float(np.median(list(lag.values())))
    Tn = float(B["T_net_max"])
    bounds = rc.Bounds(**{k: v for k, v in B.items() if k in rc.Bounds.__dataclass_fields__})
    A_any, bp_decl, bv_decl = float(B["A_any"]), float(B["beta_p"]), float(B["beta_v"])
    ego_p, ego_v = float(B["ego_err_p"]), float(B["ego_err_v"])
    out = dict(name=rd.name, n_pairs=0, old_unsound=0, new_unsound=0, n_pairs_valid=0, old_unsound_valid=0, new_unsound_valid=0,
               old_excess_max=0.0, old_excess_valid_max=0.0, ticks=0, cert_old=0, cert_new=0, cert_old_not_new=0, cert_new_not_old=0)
    pk_cache = {}
    for i in range(n):
        d = ag[i]
        tt = d.t.to_numpy() - L
        idx = np.arange(0, len(d), tick_stride)
        for k in idx:
            t = float(tt[k])
            if not (grid[0] + 1.0 < t < grid[-1] - 0.5):
                continue
            kg = int(np.searchsorted(grid, t))
            own_h = d.loc[k, ["px", "py", "pz"]].to_numpy(float)
            own_vh = d.loc[k, ["vx", "vy", "vz"]].to_numpy(float)
            e_ego_p = float(np.linalg.norm(own_h - P[kg, i]))
            e_ego_v = float(np.linalg.norm(own_vh - V[kg, i]))
            envs_old, envs_new, valids = [], [], []
            for j in range(n):
                if j == i or np.linalg.norm(P[kg, j] - P[kg, i]) > 12.0:
                    continue
                dj = ag[j]
                tj = dj.t.to_numpy() - L
                s = float(np.random.default_rng([abs(hash(rd.name)) % 2**31, i, j, k]).choice([0.0, 0.1, 0.2]))
                ell = float(np.random.default_rng([abs(hash(rd.name)) % 2**31, i, j, k, 1]).choice([0.0, Tn / 2, Tn]))
                ts_ = t - s - ell
                if ts_ < tj[0] or ts_ > tj[-1]:
                    continue
                kj = int(np.searchsorted(tj, ts_))
                kj = min(kj, len(tj) - 1)
                rep_p = np.array([dj.rpx[kj], dj.rpy[kj], dj.pz[kj]], float)
                rep_v = np.array([dj.rvx[kj], dj.rvy[kj], dj.vz[kj]], float)
                pkt = rc.NeighborPacket(j, rep_p, rep_v, t - s, Tn, bp_decl, bv_decl, A_any)
                eo = rc.build_envelope(pkt, t, own_h, own_vh, ego_p, ego_v)
                en = rc.build_envelope_sound(pkt, t, own_h, own_vh, ego_p, ego_v)
                r_true = P[kg, i] - P[kg, j]
                v_true = V[kg, i] - V[kg, j]
                exo = max(float(np.linalg.norm(r_true - eo.r_hat)) - eo.B_p, float(np.linalg.norm(v_true - eo.v_hat)) - eo.B_v)
                exn = max(float(np.linalg.norm(r_true - en.r_hat)) - en.B_p, float(np.linalg.norm(v_true - en.v_hat)) - en.B_v)
                # premise validity of THIS pair at this tick: report error, ego error, neighbour acceleration over the packet age
                k0 = int(np.searchsorted(grid, ts_))
                rep_err_p = float(np.linalg.norm(rep_p[:2] - P[int(np.searchsorted(grid, ts_)), j, :2]))
                rep_err_v = float(np.linalg.norm(rep_v[:2] - V[int(np.searchsorted(grid, ts_)), j, :2]))
                prem = (rep_err_p <= bp_decl) and (rep_err_v <= bv_decl) and (e_ego_p <= ego_p) and (e_ego_v <= ego_v) \
                    and float(amag[k0:kg + 1, j].max(initial=0.0)) <= A_any
                out["n_pairs"] += 1
                out["old_unsound"] += int(exo > 1e-9)
                out["new_unsound"] += int(exn > 1e-9)
                out["old_excess_max"] = max(out["old_excess_max"], exo)
                if prem:
                    out["n_pairs_valid"] += 1
                    out["old_unsound_valid"] += int(exo > 1e-9)
                    out["new_unsound_valid"] += int(exn > 1e-9)
                    out["old_excess_valid_max"] = max(out["old_excess_valid_max"], exo)
                envs_old.append(eo)
                envs_new.append(en)
            if not envs_old:
                continue
            u_nom = np.array([d.unx[k], d.uny[k], 0.0])
            res = []
            for envs in (envs_old, envs_new):
                A, b, _, _ = rc.rows_for(envs, bounds)
                A = A.copy()
                if A.shape[0]:
                    A[:, 2] = 0.0
                u, cert, _, _ = rc.solve_hard_safety(u_nom, A, b, bounds.U)
                res.append(bool(cert))
            out["ticks"] += 1
            out["cert_old"] += int(res[0])
            out["cert_new"] += int(res[1])
            out["cert_old_not_new"] += int(res[0] and not res[1])
            out["cert_new_not_old"] += int(res[1] and not res[0])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--glob", default="*D24R_s9*")
    ap.add_argument("--stride", type=int, default=10, help="audit every k-th agent tick (20 Hz base)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for rd in sorted(Path(p) for p in glob.glob(str(Path(a.root) / a.glob))):
        if not (rd / "eval" / "summary.json").exists():
            continue
        try:
            r = episode(rd, a.stride)
        except Exception as exc:
            r = {"name": rd.name, "error": repr(exc)}
        if r:
            rows.append(r)
        if a.limit and len(rows) >= a.limit:
            break
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out) / "gateA_replay_by_episode.csv", index=False)
    ok = df[~df.get("error", pd.Series(dtype=object)).notna()] if "error" in df else df
    tot = {k: float(ok[k].sum()) for k in ("n_pairs", "old_unsound", "new_unsound", "n_pairs_valid", "old_unsound_valid", "new_unsound_valid",
                                           "ticks", "cert_old", "cert_new", "cert_old_not_new", "cert_new_not_old")}
    tot.update(episodes=int(len(ok)), errors=int(len(df) - len(ok)), old_excess_max_m=float(ok.old_excess_max.max()),
               old_excess_valid_max_m=float(ok.old_excess_valid_max.max()),
               old_unsound_rate=tot["old_unsound"] / max(tot["n_pairs"], 1), new_unsound_rate=tot["new_unsound"] / max(tot["n_pairs"], 1),
               old_unsound_rate_premise_valid=tot["old_unsound_valid"] / max(tot["n_pairs_valid"], 1),
               new_unsound_rate_premise_valid=tot["new_unsound_valid"] / max(tot["n_pairs_valid"], 1),
               cert_rate_old=tot["cert_old"] / max(tot["ticks"], 1), cert_rate_new=tot["cert_new"] / max(tot["ticks"], 1),
               scope="re-packetized replay of archived trajectories (recorded packets not logged); labels are reconstructions")
    (Path(a.out) / "gateA_replay_summary.json").write_text(json.dumps(tot, indent=1))
    print(json.dumps(tot, indent=1))
