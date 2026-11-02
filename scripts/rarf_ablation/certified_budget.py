#!/usr/bin/env python3
"""Experiment IV (analytic part, no flights): minimum estimated clearance at which the IMPLEMENTED hard rows can be feasible.

For one neighbour, every row of `ral26_core.certificate_rows` has A_k = (tau_k^2/2) n with the same unit n and nonnegative coefficient,
so without the slew ball the best command is u = U n and the authority reserve is exactly
    gamma(r) = min_k [ tau_k^2 U/2 - b_k(r) ]
(rows built by the real function; no solver).  Dropping the slew ball can only ENLARGE the feasible set, so r_min below is a NECESSARY
clearance for feasibility of the deployed (slew-limited) rows: below it no admissible command can be certified, whatever the optimiser.
Geometry: receiver and neighbour on a line, estimated separation r, estimated closing speed c (receiver-relative), fresh packet (s = 0)
or packet age s.  Information regimes change only the declared terms the receiver may justify:
  standard  beta 1.25/0.5, sender err 0.4/1.1, ego err 0.4/1.1, A_j = A_any 8.8, T 0.15          (campaign)
  identity_honest_nb   honest neighbour known honest: beta 0, A_j = A_honest 6.2 (attacker row stays standard)
  sensor_like  trusted relative sensor 0.10 m / 0.15 m/s, latency 0.05-0.13 s, attacker authority kept A_any
  oracle_info  zero estimation/report error, transit 0.15 s kept, A_j = A_any (perfect information, unchanged physics)
  oracle_info_no_delay  as oracle_info with T = 0 and s = 0 (only physical authority remains)

    python3 scripts/rarf_ablation/certified_budget.py --out output/research/rarf_ablation_v1/certified_budget.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from swarmsec import ral26_core as rc  # noqa: E402

B0 = json.loads((ROOT / "experiments/coupled_safety_v1/bounds_declared.json").read_text())["bounds"]

REGIMES = {
    "standard": dict(bp=B0["beta_p"], bv=B0["beta_v"], sp=B0["sender_err_p"], sv=B0["sender_err_v"], ep=B0["ego_err_p"],
                     ev=B0["ego_err_v"], A=B0["A_any"], T=B0["T_net_max"]),
    "identity_honest_nb": dict(bp=0.0, bv=0.0, sp=B0["sender_err_p"], sv=B0["sender_err_v"], ep=B0["ego_err_p"], ev=B0["ego_err_v"],
                               A=B0["A_honest"], T=B0["T_net_max"]),
    "sensor_like": dict(bp=0.10, bv=0.15, sp=0.0, sv=0.0, ep=0.0, ev=0.0, A=B0["A_any"], T=0.13),
    "oracle_info": dict(bp=0.0, bv=0.0, sp=0.0, sv=0.0, ep=0.0, ev=0.0, A=B0["A_any"], T=B0["T_net_max"]),
    "oracle_info_no_delay": dict(bp=0.0, bv=0.0, sp=0.0, sv=0.0, ep=0.0, ev=0.0, A=B0["A_any"], T=0.0),
}


def reserve(r, c, s, R, U=B0["U"], W=B0["W"], h=B0["h"], M=B0["M"], d=B0["d_safe"]):
    p_own, v_own = np.zeros(3), np.zeros(3)
    pk = rc.NeighborPacket(1, np.array([r, 0.0, 0.0]), np.array([-c, 0.0, 0.0]), 0.0, R["T"], R["bp"], R["bv"], R["A"])
    e = rc.build_envelope_sound(pk, s, p_own, v_own, R["ep"], R["ev"], R["sp"], R["sv"])
    A, b, n = rc.certificate_rows(e, h, M, U, W, d)
    k0 = float(-b[0])                                   # zero-time row has no command coefficient
    g = float(np.min(A @ (U * n) - b))                   # exact best reserve without the slew ball
    return g, k0


def r_min(c, s, R, which):
    lo, hi = 0.0, 40.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        g, k0 = reserve(mid, c, s, R)
        ok = (g >= 0) if which == "all" else (k0 >= 0)
        lo, hi = (lo, mid) if ok else (mid, hi)
    return hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for name, R in REGIMES.items():
        for c in (0.0, 0.5, 1.0, 2.0):
            for s in (0.0, 0.1, 0.2):
                if name == "oracle_info_no_delay" and s > 0:
                    continue
                rows.append({"regime": name, "closing_speed_mps": c, "packet_age_s": s,
                             "r_min_zero_time_m": round(r_min(c, s, R, "k0"), 3), "r_min_all_rows_m": round(r_min(c, s, R, "all"), 3)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["packet_age_s"] == 0.0:
            print(r)


if __name__ == "__main__":
    main()
