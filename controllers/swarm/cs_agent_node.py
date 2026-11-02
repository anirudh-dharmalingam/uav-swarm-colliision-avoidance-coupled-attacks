#!/usr/bin/env python3
"""Coupled-safety campaign per-UAV controller (subclass of the RAL26 agent; historical agent untouched).

Adds, all driven by the `cs` config section and all absent => RAL26 behaviour:
  * transit mission  : ordered along-route reference progress, smooth squeeze scheduled on ROUTE PROGRESS (not time), hold at the goal
                       then land (route_mode stays "out")
  * information modes: standard | identity | oracle | sensor (see swarmsec/ral26_cs.py; oracle/sensor topics exist only if
                       oracle_adapter.py was started, and honest agents never read them in the other modes)
  * information boundary: honest processes do not keep the attacker id unless mode == identity (the manager passes it to every agent)
  * Gate B logging  : post-certification speed-envelope clip is flagged and the tick relabelled UNCERTIFIED_POSTCLIP; decision latency
                       (dec_ms), state-to-command latency (e2e_ms), counterfactual standard-information certification (cert_std, gamma_std)
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node  # noqa: F401
from std_msgs.msg import Float64MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import uav_agent_node as base  # noqa: E402
from swarmsec import ral26_core as rc  # noqa: E402
from swarmsec.ral26_cs import CsPolicy  # noqa: E402

EXTRA = ["dec_ms", "e2e_ms", "cert_std", "gamma_std", "clip", "n_fused", "n_contra", "info", "att_req", "s_ref", "scale"]
# rarf_ablation_v1 branch accounting (empty for the historical variants, whose decisions carry no extra["br"]) and timing (every tick)
BR = ["hard_ok", "hard_why", "k0_fail", "soft_ok", "rescue", "applied", "exec_rows_ok", "planar_chg", "dev_hard_fb", "hard_ms", "soft_ms",
      "gamma_valid", "slew_clip", "spd_dmin", "spd_cap", "spd_brake", "t_mono", "t_state_rx", "t_px4", "t_pub", "tick"]
EXTRA += BR
LOG_FIELDS = base.LOG_FIELDS + EXTRA


def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


class Info:
    def __init__(self, cs):
        I = cs.get("info", {})
        self.mode = I.get("mode", "standard")
        S = I.get("sensor", {})
        self.sensor = {}
        self.sensor_cfg = {"pos_err": float(S.get("pos_err", 0.10)), "vel_err": float(S.get("vel_err", 0.15)),
                           "lat_min": float(S.get("latency_s", 0.05)), "lat_max": float(S.get("latency_s", 0.05)) + float(S.get("latency_jitter_s", 0.08)),
                           "max_age_s": float(S.get("max_age_s", 1.0))}
        self.truth = deque(maxlen=200)                       # oracle: (t_mono, {id: (p, v)})


class CsAgent(base.Agent):
    def __init__(self, uid, cfg, geo, run_dir, variant, attack, attacker_id):
        super().__init__(uid, cfg, geo, run_dir, variant, attack, attacker_id)
        cs = cfg.get("cs", {})
        self.cs = cs
        self.info = Info(cs)
        if uid != attacker_id and self.info.mode != "identity":
            self.attacker_id = -1                              # information boundary: honest agents do not know the attacker id
        R = cfg.get("ral26", {})
        A = R.get("attack", {})
        if uid == attacker_id and attack.upper() == "A0" and str(A.get("attacker_policy", "same")) != "same":
            # factorial hygiene: the designated UAV runs the SAME fixed policy in every cell (incl. A0), otherwise the A0 cell would
            # differ from PHYS/REPORT/COUPLED by 'attacker runs a safety filter' as well as by the attack
            self.variant = str(A["attacker_policy"])
            self.no_squeeze = self.variant.upper().startswith("D24R_NOSQ")
            self.att_refuse_refresh = bool(A.get("attacker_refuse_refresh", False))
        self.policy = CsPolicy(self.variant, self.B, static_age_s=float(R.get("static_age_s", 1.0)),
                               nominal_age_s=float(R.get("nominal_age_s", 0.45)), gamma_warn=float(R.get("gamma_warn", 0.5)),
                               gamma_refresh=float(R.get("gamma_refresh", 0.75)), info=self.info,
                               speed_rule=R.get("speed_rule"))
        self.hold_s = float(cs.get("hold_s", 5.0))
        self.sq_ramp = float(cs.get("squeeze_ramp_frac", 1.0 / 12.0))
        self.t_goal = None
        self.pk_rep = {}                                       # packets exactly as received (counterfactual standard information)
        self.L_route = float(np.linalg.norm(self.route))
        self.clip_count = 0
        self.fresh_state = 0.0
        # replace the log writer with the extended header
        self.f.close()
        self.f = (run_dir / f"agent_{uid}.csv").open("w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=LOG_FIELDS)
        self.w.writeheader()
        self.event("cs_start", info=self.info.mode, hold_s=self.hold_s, honest_knows_attacker=bool(self.attacker_id >= 0 and uid != attacker_id))
        if self.info.mode == "sensor":
            self.create_subscription(Float64MultiArray, f"/sensor/uav_{uid}", self._on_sensor, 20)
            self.sensor_transit = []
        if self.info.mode == "oracle":
            self.create_subscription(Float64MultiArray, "/oracle/truth", self._on_truth, 20)
            inner = self.policy.step

            def step(t, p_own, v_own, packets, u_nom, **kw):
                ego = self.oracle_state(self.i, t)
                if ego is not None:
                    p_own, v_own = ego
                return inner(t, p_own, v_own, packets, u_nom, **kw)
            self.policy.step = step

    # ---- information inputs ---------------------------------------------------------------------
    def _on_sensor(self, m):
        d = list(m.data)
        t_rx = time.monotonic()
        if len(self.sensor_transit) < 200000:
            self.sensor_transit.append(t_rx - d[0])
        k = 1
        while k + 6 < len(d) + 0:
            j = int(d[k])
            self.info.sensor[j] = (np.array(d[k + 1:k + 4]), np.array(d[k + 4:k + 7]), t_rx)
            k += 7

    def _on_truth(self, m):
        d = list(m.data)
        st = {}
        for k in range(1, len(d), 7):
            st[int(d[k])] = (np.array(d[k + 1:k + 4]), np.array(d[k + 4:k + 7]))
        self.info.truth.append((d[0], st))

    def oracle_state(self, j, t):
        tr = self.info.truth
        if not tr:
            return None
        ts = [x[0] for x in tr]
        if t <= ts[0]:
            return tr[0][1][j]
        if t >= ts[-1]:
            p, v = tr[-1][1][j]
            return p + v * (t - ts[-1]), v
        k = int(np.searchsorted(ts, t))
        (t0, a), (t1, b) = tr[k - 1], tr[k]
        w = (t - t0) / max(t1 - t0, 1e-9)
        return a[j][0] * (1 - w) + b[j][0] * w, a[j][1] * (1 - w) + b[j][1] * w

    def _on_rx(self, m):
        d = m.data
        j = int(d[0])
        if j == self.i:
            return
        t_rx = time.monotonic()
        if len(self.transit) < 200000:
            self.transit.append(t_rx - float(d[2]))
        B = self.B
        bp, bv, A_j = B.beta_p, B.beta_v, B.A_any
        p_msg, v_msg = np.array(d[3:6]), np.array(d[6:9])
        self.pk_rep[j] = rc.NeighborPacket(j, p_msg.copy(), v_msg.copy(), t_rx, B.T_net_max, bp, bv, A_j)
        mode = self.info.mode
        if mode == "identity" and self.attacker_id >= 0 and j != self.attacker_id:
            bp = bv = 0.0
            A_j = B.A_honest
        elif mode == "oracle":
            st = self.oracle_state(j, float(d[2]))
            if st is not None:
                p_msg, v_msg = st
            bp = bv = 0.0
        self.pk[j] = rc.NeighborPacket(j, p_msg, v_msg, t_rx, B.T_net_max, bp, bv, A_j)

    # ---- mission ----------------------------------------------------------------------------------
    def sched_scale(self, tm):
        if self.no_squeeze:
            return 1.0
        if self.scenario != "transit":
            return super().sched_scale(tm)
        x = self.progress / max(self.L_route, 1e-9)
        w = max(self.sq_ramp, 1e-6)
        f = smoothstep((x - 1.0 / 3.0) / w) * (1.0 - smoothstep((x - (2.0 / 3.0 - w)) / w))
        return 1.0 - self.squeeze * f

    def _counterfactual(self, now, p, v, u_nom):
        """Standard-information certification of the SAME tick (diagnostic only; never commanded)."""
        B = self.B
        envs = []
        for pkt in self.pk_rep.values():
            envs.append(rc.build_envelope_sound(pkt, now, p, v, B.ego_err_p, B.ego_err_v, B.sender_err_p, B.sender_err_v))
        A, b, _, _ = self.policy._rows(envs, B)
        un = np.array(u_nom, float)
        un[2] = 0.0
        nn = float(np.linalg.norm(un))
        if nn > B.U:
            un *= B.U / nn
        up, dmax = self.policy._up(), self.policy._dmax()
        _, cert, _, _ = rc.solve_hard_safety(un, A, b, B.U, up, dmax)
        g, _, _ = rc.authority_reserve_m(A, b, B.U, up, dmax)
        return int(cert), float(g)

    def mission_tick(self, now, p, v, loop_dt):
        t_start = time.perf_counter()
        tm = now - self.t0
        veh = self.veh
        en = None
        if self.energy is not None:
            dist_home = float(np.linalg.norm((p - (self.home_c + self.offsets[self.i]))[:2]))
            en = self.energy.screen(dist_home)
        u_nom = self.nominal(tm, p, v, self.policy.relax)
        dec = self.policy.step(now, p, v, self.pk, u_nom, energy=en)
        if dec.mode == "RETURN_REQUIRED" and self.mode != "RETURN_REQUIRED":
            self.mode = "RETURN_REQUIRED"
            self.event("return_required", t=tm, energy=en)
        u = dec.u.copy()
        u[2] = 0.0
        u_cmd, rep_p, rep_v, att_on = self.attack_terms(tm, p, v, u, u_nom)
        if dec.refresh:
            self.req_pub.publish(Float64MultiArray(data=[float(self.i), now]))
        clip = 0
        ucmd = u_cmd.copy()
        sp = float(np.linalg.norm(v[:2]))
        if sp > self.vmax and float(ucmd[:2] @ v[:2]) > 0:
            ucmd[:2] -= (ucmd[:2] @ v[:2]) / (sp * sp) * v[:2]
            clip = 1
            self.clip_count += 1
        veh.publish_mode("acceleration")
        veh.accel_setpoint(ucmd, self.alt)
        t_pub = time.monotonic()
        status = dec.status
        if clip and status == rc.CERTIFIED:
            status = "UNCERTIFIED_POSTCLIP"                 # the command actually sent was not the certified one
        rate = self.broadcast(now, rep_p, rep_v)
        cert_std, gamma_std = ("", "")
        if self.info.mode != "standard" and self.pk_rep:
            cert_std, gamma_std = self._counterfactual(now, p, v, u_nom)
        goal = self.progress >= self.L_route - 1e-6
        if goal and self.phase == "MISSION":
            if self.t_goal is None:
                self.t_goal = tm
                self.event("goal_reference_reached", t=tm)
            if tm - self.t_goal >= self.hold_s:
                self.event("hold_complete_landing", t=tm)
                self.set_phase("LAND")
        if self.mode == "RETURN_REQUIRED":
            tgt = self.home_c + self.offsets[self.i]
            if np.linalg.norm((p - tgt)[:2]) < 1.0 and np.linalg.norm(v[:2]) < 0.4:
                self.event("return_landing", t=tm)
                self.set_phase("LAND")
        comp_ms = 1e3 * (time.perf_counter() - t_start)
        e2e_ms = 1e3 * (time.monotonic() - veh.t_rx) if veh.t_rx else math.nan
        self.ticks += 1
        if comp_ms > 1e3 * self.dt:
            self.deadline_miss += 1
        a = veh.acc_enu()
        row = {
            "t": f"{tm:.3f}", "phase": self.phase, "mode": self.mode, "status": status, "gamma": f"{dec.gamma:.4f}",
            "nominal_unsafe": int(dec.nominal_unsafe), "cert_h": dec.certified_horizon, "min_res": f"{dec.min_row_residual:.4f}",
            "worst_age": f"{dec.worst_age:.3f}", "n_rows": dec.n_rows_agents, "n_excl": dec.n_excluded, "n_pkts": len(self.pk),
            "fail": dec.failure_reason, "relax": f"{dec.relax:.3f}", "refresh": int(dec.refresh), "solve_ms": f"{comp_ms:.2f}",
            "loop_dt": f"{loop_dt:.4f}", "assump_ok": int(dec.assumptions_valid),
            "px": f"{p[0]:.3f}", "py": f"{p[1]:.3f}", "pz": f"{p[2]:.3f}", "vx": f"{v[0]:.3f}", "vy": f"{v[1]:.3f}", "vz": f"{v[2]:.3f}",
            "ax": f"{a[0]:.3f}", "ay": f"{a[1]:.3f}", "unx": f"{u_nom[0]:.3f}", "uny": f"{u_nom[1]:.3f}",
            "ux": f"{u_cmd[0]:.3f}", "uy": f"{u_cmd[1]:.3f}", "spx": f"{ucmd[0]:.3f}", "spy": f"{ucmd[1]:.3f}",
            "E_wh": f"{self.energy.E:.4f}" if self.energy else "", "P_w": f"{self.power.power_w():.1f}" if self.energy else "",
            "E_margin": f"{en['margin_wh']:.4f}" if en else "", "tx_rate": rate, "attack_on": int(att_on),
            "rep_bias": f"{float(np.linalg.norm(rep_p - p)):.3f}", "nav": veh.nav_state(), "xy_valid": int(veh.xy_valid),
            "rpx": f"{rep_p[0]:.3f}", "rpy": f"{rep_p[1]:.3f}", "rvx": f"{rep_v[0]:.3f}", "rvy": f"{rep_v[1]:.3f}",
            "dec_ms": f"{dec.solve_ms:.2f}", "e2e_ms": f"{e2e_ms:.2f}", "cert_std": cert_std,
            "gamma_std": f"{gamma_std:.4f}" if gamma_std != "" else "", "clip": clip,
            "n_fused": self.policy.fused, "n_contra": self.policy.contradictions, "info": self.info.mode,
            "att_req": f"{float(np.linalg.norm(u_cmd - np.minimum(1.0, self.B.U / max(float(np.linalg.norm(u_nom)), 1e-9)) * u_nom)):.3f}",
            "s_ref": f"{self.progress:.3f}", "scale": f"{self.sched_scale(tm):.3f}"}
        br, spd = dec.extra.get("br"), dec.extra.get("spd")
        row.update({k: v for k, v in (br or {}).items() if k in BR})
        if spd is not None:
            row.update(spd_dmin=f"{spd['d_min']:.3f}", spd_cap=f"{spd['cap']:.3f}", spd_brake=spd["brake"])
        row.update(t_mono=f"{now:.4f}", t_state_rx=f"{veh.t_rx:.4f}", t_px4=f"{getattr(veh, 't_px4', math.nan):.4f}",
                   t_pub=f"{t_pub:.4f}", tick=self.ticks)
        self.w.writerow(row)
        if self.ticks % 40 == 0:
            self.f.flush()

    def close(self):
        extra = {"clip_count": self.clip_count, "fused": self.policy.fused, "contradictions": self.policy.contradictions,
                 "contra_pos": self.policy.contra_p, "contra_vel": self.policy.contra_v,
                 "sensor_used": self.policy.sensor_used, "info": self.info.mode, "t_goal": self.t_goal}
        if self.info.mode == "sensor" and getattr(self, "sensor_transit", None):
            tr = np.asarray(self.sensor_transit)
            extra["sensor_transit_s"] = {"n": int(tr.size), "p50": float(np.quantile(tr, .5)), "p99": float(np.quantile(tr, .99)),
                                         "max": float(tr.max()), "declared_lat_max": self.info.sensor_cfg["lat_max"],
                                         "frac_over_declared": float(np.mean(tr > self.info.sensor_cfg["lat_max"]))}
        super().close()
        p = self.run_dir / f"agent_{self.i}_summary.json"
        try:
            s = json.loads(p.read_text())
            s["cs"] = extra
            p.write_text(json.dumps(s, default=str))
        except Exception:
            pass


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", default="D24R")
    ap.add_argument("--attack", default="A0")
    ap.add_argument("--attacker-id", type=int, default=-1)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    geo = json.loads(Path(a.geometry).read_text())
    rclpy.init()
    node = CsAgent(a.id, cfg, geo, Path(a.run_dir) / "agents", a.variant, a.attack, a.attacker_id)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.phase == "DONE" and node.land_cmd and time.monotonic() - node.t_phase > 1.0:
                break
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
