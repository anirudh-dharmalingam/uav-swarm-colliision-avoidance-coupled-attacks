#!/usr/bin/env python3
"""E3 diagnostics (secondary): pairwise predicted certificate status and the restricted-model margin M versus the measured outcome.

    python3 scripts/coupled_safety/e3_diagnostic.py results/coupled_safety_v1/e3 --out e3_diag.csv     (inside the image)

For each episode the VICTIM is the honest UAV closest to the designated UAV at the minimum-separation instant.  Over the encounter window
[onset, onset+20 s]:
  pair_feasible   share of victim ticks for which the rows built from the attacker's packet ALONE (sound envelope, frozen bounds, packet age
                  taken from the logged worst age) admit some command in the U ball -- the pairwise conservative-certificate prediction
                  (the full certificate also needs every other neighbour, so cert_frac in the agent log is lower).
  M_diag          restricted-model margin at onset (Proposition 1): q = d_ha(onset) - d_safe, c = projected closing speed at onset (truth, SG 0.2 s),
                  R_q, R_c = envelope radii at the median packet age, tau = 0.1 s reaction delay, A_d = A_att = p95 of the attacker's realised
                  acceleration along the line of sight over the window, B = U - A_att.  B > 0 -> Statement 2; B <= 0 -> finite-window Statement 3
                  (H = 20 s).  A DIAGNOSTIC for the 1-D no-escape model; it is not an X500 threshold (lateral escape, slew limits are outside the model).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import ral26_audit_v2 as AU  # noqa: E402
from swarmsec import ral26_core as rc  # noqa: E402

TAU, H = 0.1, 20.0


def one(rd: Path):
    cfg = json.loads((rd / "resolved_config.json").read_text())
    R = cfg["ral26"]
    B = rc.Bounds(**{k: v for k, v in R["bounds"].items() if k in rc.Bounds.__dataclass_fields__})
    n, att = int(cfg["swarm"]["num_uavs"]), int(R["attacker_id"])
    onset = float(R["attack"]["start_s"])
    ev = json.loads((rd / "eval" / "summary.json").read_text())
    T = pd.read_csv(rd / "eval" / "truth.csv")
    ts = T.t_sim.to_numpy() - ev["t_mission_start_sim"]
    dt = 0.02
    grid = np.arange(max(0.0, ts[0]), ts[-1], dt)
    P = np.stack([np.stack([np.interp(grid, ts, T[f"{c}{i}"]) for c in "xyz"], 1) for i in range(n)], 1)
    win = 11
    Vv = np.stack([np.stack([AU.sg(P[:, i, k], win, 2, 1, dt) for k in range(3)], 1) for i in range(n)], 1)
    Aa = np.stack([np.stack([AU.sg(P[:, i, k], win, 2, 2, dt) for k in range(3)], 1) for i in range(n)], 1)
    m = (grid >= onset) & (grid <= onset + H)
    hon = [i for i in range(n) if i != att]
    d = np.stack([np.linalg.norm(P[:, att] - P[:, j], axis=1) for j in hon], 1)
    kmin = int(np.argmin(np.where(m[:, None], d, np.inf).min(axis=1)))
    victim = hon[int(np.argmin(d[kmin]))]
    ko = int(np.searchsorted(grid, onset))
    r0 = P[ko, victim] - P[ko, att]
    q = float(np.linalg.norm(r0)) - B.d_safe
    los = r0 / max(np.linalg.norm(r0), 1e-9)
    c_hat = float(-(Vv[ko, victim] - Vv[ko, att]) @ los)
    a_att = np.clip(-(Aa[m, att] @ (-los)) * 1.0, 0, None)            # attacker accel directed at the victim (line of sight at onset)
    # attacker acceleration toward the victim along the instantaneous line of sight
    los_t = (P[m, victim] - P[m, att]); los_t /= np.maximum(np.linalg.norm(los_t, axis=1, keepdims=True), 1e-9)
    a_toward = np.einsum("ij,ij->i", Aa[m, att], los_t)
    A_att = float(np.quantile(np.clip(a_toward, 0, None), 0.95))
    # pairwise feasibility from logs
    f_v = rd / "agents" / f"agent_{victim}.csv.gz"
    f_a = rd / "agents" / f"agent_{att}.csv.gz"
    dv, da = pd.read_csv(f_v), pd.read_csv(f_a)
    dv, da = dv[dv.phase == "MISSION"], da[da.phase == "MISSION"]
    tv, ta = dv.t.to_numpy(float), da.t.to_numpy(float)
    sel = (tv >= onset) & (tv <= onset + H)
    feas, ages = [], []
    for k in np.flatnonzero(sel)[::4]:
        wa = float(dv.worst_age.iloc[k])
        s = max(0.0, wa - B.T_net_max)
        j = int(np.searchsorted(ta, tv[k] - (s + B.T_net_max / 2)))
        j = min(max(j, 0), len(ta) - 1)
        pk = rc.NeighborPacket(att, np.array([da.rpx.iloc[j], da.rpy.iloc[j], da.pz.iloc[j]], float), np.array([da.rvx.iloc[j], da.rvy.iloc[j], da.vz.iloc[j]], float),
                               0.0, B.T_net_max, B.beta_p, B.beta_v, B.A_any)
        own_p = dv[["px", "py", "pz"]].iloc[k].to_numpy(float)
        own_v = dv[["vx", "vy", "vz"]].iloc[k].to_numpy(float)
        e = rc.build_envelope_sound(pk, s, own_p, own_v, B.ego_err_p, B.ego_err_v, B.sender_err_p, B.sender_err_v)
        A, b, _, _ = rc.rows_for([e], B)
        if A.shape[0] == 0:
            feas.append(1.0)
        else:
            A = A.copy(); A[:, 2] = 0.0
            g, _, _ = rc.authority_reserve_m(A, b, B.U)
            feas.append(float(g >= 0))
        ages.append(wa)
    wa_med = float(np.median(ages)) if ages else B.T_net_max + 0.1
    e = rc.build_envelope_sound(rc.NeighborPacket(att, np.array([10.0, 0, 0]), np.zeros(3), 0.0, B.T_net_max, B.beta_p, B.beta_v, B.A_any), max(0.0, wa_med - B.T_net_max),
                                np.zeros(3), np.zeros(3), B.ego_err_p, B.ego_err_v, B.sender_err_p, B.sender_err_v)
    Rq, Rc = e.B_p, e.B_v
    cbar = max(0.0, c_hat + Rc)
    Beff = B.U - A_att
    if Beff > 1e-6:
        M = q - Rq - cbar * TAU - 0.5 * A_att * TAU ** 2 - (cbar + A_att * TAU) ** 2 / (2 * Beff)
        form = "Statement2"
    else:
        Hp = max(H - TAU, 0.0)
        M = q - Rq - cbar * TAU - 0.5 * A_att * TAU ** 2 - (cbar + A_att * TAU) * Hp + 0.5 * Beff * Hp ** 2
        form = "Statement3_finite_window"
    return {"name": rd.name, "victim": victim, "clearance_onset": q + B.d_safe, "c_hat": c_hat, "A_att_p95": A_att, "B_eff": Beff, "R_q": Rq, "R_c": Rc,
            "M_diag": M, "M_form": form, "pair_feasible_frac": float(np.mean(feas)) if feas else math.nan, "median_worst_age": wa_med}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for rd in sorted(p for p in Path(a.dir).iterdir() if p.is_dir() and not p.name.startswith("_")):
        if (rd / "eval" / "summary.json").exists():
            try:
                rows.append(one(rd))
            except Exception as exc:
                rows.append({"name": rd.name, "error": repr(exc)})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(len(rows), "->", a.out)
