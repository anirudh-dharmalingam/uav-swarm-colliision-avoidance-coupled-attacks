#!/usr/bin/env python3
"""Fit the Layer-1 plant model for swarmsec/rarf_l1 from PX4 SITL logs (TRAIN = rarf_plant_time_v3 confirm; never the hold-out
rarf_hocbf_v1 flights).

    python3 scripts/rarf_l1/fit_plant.py results/rarf_plant_time_v3/confirm --out swarmsec/rarf_l1/plant_fit.json

Model (horizontal axes, independent, identical):
  * actuator: commanded acceleration setpoint u (zero-order hold at the agent's 20 Hz decision grid) -> true acceleration a, a
    second-order lag with natural frequency wn, damping zeta and pure delay td:   a'' = wn^2 (u(t - td) - a) - 2 zeta wn a'
  * vertical: altitude held exactly (PX4 altitude loop; |a_z| <= 0.69 m/s^2 in HF1 is not modelled).
  * ego estimate: p_hat = p + e_p, v_hat = v + e_v, each an independent Ornstein-Uhlenbeck process per axis with stationary SD and
    correlation time fitted from (agent log estimate - truth interpolated to the agent's plant time).
Fit: true acceleration from truth positions (5-point smoothed double difference on the 50 Hz grid), aligned to the agent mission time
via eval/summary.json t_mission_start_sim; grid search over (wn, zeta, td) minimising the squared residual over every honest UAV in
[0, 60] s mission time.  Attacker UAVs are excluded (no slew limit; HF1 shows steps are not tracked by the same law).
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(rd: Path):
    ev = json.loads((rd / "eval" / "summary.json").read_text())
    cfg = json.loads((rd / "resolved_config.json").read_text())
    t0 = ev["t_mission_start_sim"]
    T = pd.read_csv(rd / "eval" / "truth.csv")
    n = int(cfg["swarm"]["num_uavs"])
    att = int(cfg["ral26"].get("attacker_id", -1))
    out = []
    for i in range(n):
        if i == att:
            continue
        f = rd / "agents" / f"agent_{i}.csv.gz"
        if not f.exists():
            continue
        A = pd.read_csv(gzip.open(f, "rt"), low_memory=False)
        A = A[A.phase == "MISSION"]
        if len(A) < 200:
            continue
        out.append((T.t_sim.to_numpy() - t0, T[f"x{i}"].to_numpy(), T[f"y{i}"].to_numpy(), A))
    return out


def true_acc(t, x):
    """5-point smoothed second derivative on the (near-uniform) truth grid."""
    dt = np.median(np.diff(t))
    k = np.array([1, 2, 0, -2, -1], float)  # smoothed first derivative stencil / (8 dt)
    v = np.convolve(x, k[::-1], mode="same") / (8 * dt)
    a = np.convolve(v, k[::-1], mode="same") / (8 * dt)
    return a


def simulate(tu, u, wn, zeta, td, t_out, h=0.005):
    """Second-order lag driven by the ZOH setpoint sequence (tu, u); returns a at t_out."""
    from scipy import signal
    t = np.arange(tu[0], t_out[-1] + h, h)
    idx = np.clip(np.searchsorted(tu, t - td, side="right") - 1, 0, len(u) - 1)
    _, a, _ = signal.lsim(([wn * wn], [1.0, 2 * zeta * wn, wn * wn]), u[idx], t - t[0], interp=False)
    return np.interp(t_out, t, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-eps", type=int, default=24)
    a = ap.parse_args()
    eps = sorted(p for p in Path(a.root).iterdir() if p.is_dir() and (p / "eval" / "summary.json").exists())
    rng = np.random.default_rng(0)
    eps = [eps[i] for i in sorted(rng.choice(len(eps), size=min(a.max_eps, len(eps)), replace=False))]
    segs, est = [], {"ep": [], "ev": []}
    for rd in eps:
        for tt, x, y, A in load(rd):
            ta = A.t.to_numpy(float)
            m = (ta >= 0) & (ta <= 60)
            if m.sum() < 200:
                continue
            ta = ta[m]
            ux, uy = A.spx.to_numpy(float)[m], A.spy.to_numpy(float)[m]
            g = (tt >= ta[0] + 1.0) & (tt <= ta[-1])
            tg = tt[g]
            axt, ayt = true_acc(tt, x)[g], true_acc(tt, y)[g]
            segs.append((ta, ux, uy, tg, axt, ayt))
            # ego estimate error at agent ticks (truth interpolated to the agent's plant time)
            px, py = A.px.to_numpy(float)[m], A.py.to_numpy(float)[m]
            vx, vy = A.vx.to_numpy(float)[m], A.vy.to_numpy(float)[m]
            xt, yt = np.interp(ta, tt, x), np.interp(ta, tt, y)
            dt = np.median(np.diff(tt))
            vxt = np.interp(ta, tt, np.gradient(x, dt))
            vyt = np.interp(ta, tt, np.gradient(y, dt))
            est["ep"].append(np.stack([px - xt, py - yt], 1))
            est["ev"].append(np.stack([vx - vxt, vy - vyt], 1))
    print(len(segs), "honest UAV segments from", len(eps), "episodes")
    best = None
    for wn in (4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0):
        for zeta in (0.4, 0.5, 0.6, 0.7, 0.85, 1.0):
            for td in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
                sse = cnt = 0.0
                for ta, ux, uy, tg, axt, ayt in segs[:12]:
                    for u, at in ((ux, axt), (uy, ayt)):
                        r = simulate(ta, u, wn, zeta, td, tg) - at
                        sse += float(r @ r)
                        cnt += r.size
                rmse = (sse / cnt) ** 0.5
                if best is None or rmse < best[0]:
                    best = (rmse, wn, zeta, td)
    rmse, wn, zeta, td = best
    # hold-out residual on the remaining segments and the zero-model reference (a = u, no dynamics)
    res, ref = [], []
    for ta, ux, uy, tg, axt, ayt in segs[12:]:
        for u, at in ((ux, axt), (uy, ayt)):
            res.append(simulate(ta, u, wn, zeta, td, tg) - at)
            idx = np.clip(np.searchsorted(ta, tg, side="right") - 1, 0, len(u) - 1)
            ref.append(u[idx] - at)
    res, ref = np.concatenate(res), np.concatenate(ref)

    def ou(E):
        E = np.concatenate(E)
        sd = float(E.std())
        ac = []
        for e in E.T:
            e = e - e.mean()
            c = np.correlate(e[:400], e[:400], "full")[399:]
            ac.append(c / c[0])
        ac = np.mean(ac, 0)
        lag = int(np.argmax(ac < np.exp(-1))) if (ac < np.exp(-1)).any() else len(ac)
        return {"sd": sd, "p99_abs": float(np.quantile(np.abs(E), 0.99)), "max_abs": float(np.abs(E).max()), "tau_s": lag * 0.05,
                "mean": E.mean(0).tolist()}

    fit = {"source": str(a.root), "episodes": [p.name for p in eps], "segments": len(segs),
           "actuator": {"wn": wn, "zeta": zeta, "td": td, "train_rmse": rmse,
                        "holdout_rmse": float(np.sqrt(np.mean(res ** 2))), "holdout_p99_abs": float(np.quantile(np.abs(res), 0.99)),
                        "zero_model_holdout_rmse": float(np.sqrt(np.mean(ref ** 2)))},
           "ego_p": ou(est["ep"]), "ego_v": ou(est["ev"])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(fit, indent=1))
    print(json.dumps({k: v for k, v in fit.items() if k != "episodes"}, indent=1))


if __name__ == "__main__":
    main()
