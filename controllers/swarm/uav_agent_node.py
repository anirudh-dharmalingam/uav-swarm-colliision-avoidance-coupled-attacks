#!/usr/bin/env python3
"""RAL26 per-UAV controller process (one OS process per vehicle).

Information boundary: this process reads ONLY
  * its own PX4 estimate   /uav_<i>/fmu/out/vehicle_local_position_v1, vehicle_status_v1
  * its own FANET receiver /ral/rx/uav_<i>  (packets the bridge delivered; t_rx = local monotonic time)
  * refresh requests       /ral/req/uav_<i>
  * GCS start/land markers /ral/gcs
  * its own rotor-speed commands (energy estimate; see energy_supervisor.py)
and the shared mission plan (geometry.json, resolved config).  It never reads Gazebo poses of any
vehicle or another agent's process state.

Safety layer: swarmsec.ral26_policy.AgentPolicy (D24R / D26 variants), solved every control tick.
Compromised endpoint (research sandbox, --attack != A0 and --role attacker): the SAME process type,
whose broadcast and/or own PX4 command is manipulated within declared budgets; its physical effect is
whatever its own PX4 + Gazebo X500 produce.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from px4_iface import PX4Vehicle  # noqa: E402
from swarmsec import ral26_core as rc  # noqa: E402
from swarmsec.ral26_policy import AgentPolicy  # noqa: E402

LOG_FIELDS = ["t", "phase", "mode", "status", "gamma", "nominal_unsafe", "cert_h", "min_res", "worst_age",
              "n_rows", "n_excl", "n_pkts", "fail", "relax", "refresh", "solve_ms", "loop_dt", "assump_ok",
              "px", "py", "pz", "vx", "vy", "vz", "ax", "ay", "unx", "uny", "ux", "uy", "spx", "spy",
              "E_wh", "P_w", "E_margin", "tx_rate", "attack_on", "rep_bias", "nav", "xy_valid",
              "rpx", "rpy", "rvx", "rvy"]


def cfg_get(cfg, path, default):
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


class Agent(Node):
    def __init__(self, uid, cfg, geo, run_dir: Path, variant: str, attack: str, attacker_id: int):
        super().__init__(f"ral26_agent_{uid}")
        self.i, self.cfg, self.run_dir = uid, cfg, run_dir
        R = cfg.get("ral26", {})
        self.n = int(cfg_get(cfg, "swarm.num_uavs", 4))
        self.alt = float(cfg_get(cfg, "swarm.altitude_m", 5.0))
        self.dt = 1.0 / float(R.get("rate_hz", 20.0))
        self.cmd_mode = R.get("cmd_mode", "acceleration")
        self.tau = float(R.get("velocity_tau_s", 0.5))
        self.vmax = float(R.get("max_speed_mps", 4.0))
        self.offsets = np.array([geo["desired_offsets"][str(k)] for k in range(self.n)], float)
        self.spawn = {k: np.array(geo["spawn_positions"][str(k)], float) for k in range(self.n)}
        self.home_c = np.mean([self.spawn[k] for k in range(self.n)], axis=0)
        self.home_c[2] = self.alt
        self.route = np.array(list(R.get("route", [0.0, 0.0])) + [0.0], float)
        self.v_mission = float(R.get("v_mission", 1.0))
        self.route_mode = R.get("route_mode", "out")         # "out" | "out_back" (return to home slot, land)
        self.scenario = R.get("scenario", "bottleneck")
        self.squeeze = float(R.get("squeeze", 0.35))
        self.sq_t = (float(R.get("squeeze_t0", 8.0)), float(R.get("squeeze_ramp_s", 8.0)))
        self.breathe = float(R.get("breathe_period_s", 10.0))
        self.kp, self.kv = float(R.get("kp", 0.6)), float(R.get("kv", 1.2))
        self.ka, self.kva = float(R.get("ka", 0.5)), float(R.get("kva", 1.0))
        self.radius = float(R.get("neighbor_radius_m", 12.0))
        b = R.get("bounds", {})
        self.B = rc.Bounds(**{k: v for k, v in b.items() if k in rc.Bounds.__dataclass_fields__})
        A = R.get("attack", {})
        is_att = uid == attacker_id and attack.upper() != "A0"
        # v2: the compromised endpoint runs a FIXED policy (default "same" = legacy behaviour, where it ran
        # the defenders' own variant and therefore also relaxed under D26 -- a confound found in the v2 audit)
        att_pol = str(A.get("attacker_policy", "same"))
        if is_att and att_pol != "same":
            variant = att_pol
        self.att_refuse_refresh = is_att and bool(A.get("attacker_refuse_refresh", False))
        self.no_squeeze = variant.upper().startswith("D24R_NOSQ")
        self.variant = variant
        self.policy = AgentPolicy(variant, self.B, static_age_s=float(R.get("static_age_s", 1.0)),
                                  nominal_age_s=float(R.get("nominal_age_s", 0.45)),
                                  gamma_warn=float(R.get("gamma_warn", 0.5)),
                                  gamma_refresh=float(R.get("gamma_refresh", 0.75)))
        rc.prewarm()                                             # compile solvers before flight
        self.relax_gain = float(R.get("relax_gain", 1.0))
        self.tx_base, self.tx_fast = float(R.get("tx_rate_hz", 5.0)), float(R.get("tx_fast_hz", 20.0))
        self.fast_hold = float(R.get("refresh_hold_s", 1.0))
        self.fast_until = -1.0
        self.next_tx = 0.0
        self.seq = 0
        # attack (only for the compromised endpoint)
        self.attack = attack.upper() if uid == attacker_id else "A0"
        self.attacker_id = attacker_id
        self.eps = float(A.get("epsilon", 0.0))
        self.beta_att = float(A.get("beta_attack", 0.0))
        self.beta_v_att = float(A.get("beta_v_attack", 0.0))
        self.att_start = float(A.get("start_s", 10.0))
        self.replay_s = float(A.get("replay_s", 2.0))
        self.trig_r = float(A.get("trigger_radius_m", 8.0))
        self.hist = deque(maxlen=400)
        # demo control panel (scripts/demo): live attack switch read from <run>/live_attack.json; only when the
        # run config sets demo.live, so campaign episodes never read it
        self.live_path = run_dir.parent / "live_attack.json" if cfg.get("demo", {}).get("live") else None
        self.live, self.live_poll = {}, 0.0
        # energy
        E = R.get("energy", {})
        self.energy = None
        self.v_ret_cmd = float(E.get("v_return_cmd", 1.5))     # commanded return speed (bound uses v_return)
        if E.get("enabled", True):
            from energy_supervisor import EnergyEstimator, PowerMonitor
            e0 = E.get("E0_wh", 60.0)
            e0 = e0[uid] if isinstance(e0, list) else e0
            self.power = PowerMonitor(uid, float(E.get("inflation", 1.1)))
            self.energy = EnergyEstimator(e0, float(E.get("reserve_wh", 1.0)), float(E.get("P_return_upper_w", 400.0)),
                                          float(E.get("v_return", 1.5)), float(E.get("T_land_upper_s", 15.0)),
                                          valid=bool(E.get("model_valid", True)))
        self.veh = PX4Vehicle(self, uid, self.spawn[uid])
        self.pk = {}
        self.create_subscription(Float64MultiArray, f"/ral/rx/uav_{uid}", self._on_rx, 50)
        self.create_subscription(Float64MultiArray, f"/ral/req/uav_{uid}", self._on_req, 10)
        self.create_subscription(String, "/ral/gcs", self._on_gcs, 10)
        self.tx_pub = self.create_publisher(Float64MultiArray, "/ral/tx", 50)
        self.req_pub = self.create_publisher(Float64MultiArray, "/ral/req_tx", 10)
        self.st_pub = self.create_publisher(String, f"/ral/status/uav_{uid}", 10)
        self.phase, self.t_phase = "WAIT", time.monotonic()
        self.t0 = None
        self.land_cmd = False
        self.progress = 0.0
        self.mode = "MISSION"
        self.t_last_loop = None
        self.hold_ok_since = None
        self.ret_landing = False
        self.tx_count = 0
        self.tx_bytes = 0
        self.deadline_miss = 0
        self.ticks = 0
        self.transit = []                         # t_rx - sender stamp (same container CLOCK_MONOTONIC)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.f = (run_dir / f"agent_{uid}.csv").open("w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=LOG_FIELDS)
        self.w.writeheader()
        self.ev = (run_dir / f"agent_{uid}_events.jsonl").open("w")
        self.event("start", variant=variant, attack=self.attack, cmd_mode=self.cmd_mode, bounds=self.B.__dict__)
        self.timer = self.create_timer(self.dt, self.loop)

    # ------------------------------------------------------------------------------------------
    def event(self, name, **kw):
        self.ev.write(json.dumps({"mono": time.monotonic(), "event": name, **kw}, default=str) + "\n")
        self.ev.flush()

    def set_phase(self, p):
        self.event("phase", frm=self.phase, to=p)
        self.phase, self.t_phase = p, time.monotonic()

    def _on_rx(self, m):
        d = m.data
        j = int(d[0])
        if j == self.i:
            return
        t_rx = time.monotonic()                       # receiver-local time, never the sender's stamp
        if len(self.transit) < 200000:
            self.transit.append(t_rx - float(d[2]))   # premise check only (audit); never used for control
        self.pk[j] = rc.NeighborPacket(j, np.array(d[3:6]), np.array(d[6:9]), t_rx, self.B.T_net_max,
                                       self.B.beta_p, self.B.beta_v, self.B.A_any)

    def _on_req(self, m):
        if self.att_refuse_refresh:
            return
        if int(m.data[0]) != self.i:
            self.fast_until = time.monotonic() + self.fast_hold

    def _on_gcs(self, m):
        d = json.loads(m.data)
        if d.get("cmd") == "start" and self.t0 is None:
            self.t0 = float(d["t0"])
        if d.get("cmd") in ("land", "done"):
            self.land_cmd = True

    # ------------------------------------------------------------------------------------------
    def sched_scale(self, tm):
        if self.no_squeeze:
            return 1.0
        if self.scenario == "bottleneck":
            f = np.clip((tm - self.sq_t[0]) / max(self.sq_t[1], 1e-6), 0.0, 1.0)
            return 1.0 - self.squeeze * f
        if self.scenario == "breathing":
            return 1.0 - self.squeeze * max(0.0, math.sin(2 * math.pi * tm / self.breathe))
        return 1.0

    def nominal(self, tm, p, v, relax):
        s = self.sched_scale(tm)
        s = s + relax * self.relax_gain * (1.0 - s)          # D26 mission relaxation: restore spacing
        off = self.offsets * s
        L = float(np.linalg.norm(self.route))
        if self.mode == "RETURN_REQUIRED":
            off = self.offsets                             # full spacing on the way home
            target = self.home_c + off[self.i]
            to_home = target - p
            to_home[2] = 0.0
            dist = float(np.linalg.norm(to_home))
            v_ref = np.zeros(3)
            if dist > 0.5:
                v_ref = self.v_ret_cmd * to_home / dist * min(1.0, dist / 3.0)
            u = self.ka * (target - p) + self.kva * (v_ref - v)
        else:
            dirv = self.route / L if L > 0 else np.zeros(3)
            Ltot = 2 * L if self.route_mode == "out_back" else L
            spd = self.v_mission * (1.0 - 0.6 * relax) if self.progress < Ltot else 0.0
            self.progress = min(Ltot, self.progress + spd * self.dt)
            along = self.progress if self.progress <= L else 2 * L - self.progress
            sgn = 1.0 if self.progress <= L else -1.0
            c_ref = self.home_c + dirv * along
            v_ref = sgn * dirv * spd
            u = self.ka * (c_ref + off[self.i] - p) + self.kva * (v_ref - v)
        # consensus over received neighbours (packet centres propagated by local age)
        now = time.monotonic()
        terms_p, terms_v = [], []
        for j, q in self.pk.items():
            age = max(0.0, now - q.t_rx)
            pj = q.p_msg + age * q.v_msg
            if np.linalg.norm(pj - p) > self.radius:
                continue
            terms_p.append((p - pj) - (off[self.i] - off[j]))
            terms_v.append(v - q.v_msg)
        if terms_p:
            u += -self.kp * np.mean(terms_p, axis=0) - self.kv * np.mean(terms_v, axis=0)
        u[2] = 0.0
        return u

    # ---- compromised endpoint ------------------------------------------------------------------
    def attack_terms(self, tm, p, v, u_safe, u_nom):
        """Returns (u_cmd, rep_p, rep_v, active).  Budgets: ||delta_u|| <= epsilon (physical),
        ||rep_p - p|| <= beta_attack, ||rep_v - v|| <= beta_v_attack (self-report only)."""
        rep_p, rep_v = p.copy(), v.copy()
        if self.live_path is not None and self.i == self.attacker_id:
            self.poll_live()
            mode = self.live.get("mode", "auto")
            if mode == "off" or self.t0 is None:
                return u_safe, rep_p, rep_v, False
            if mode == "on":
                tm = max(tm, self.att_start)              # fire now, regardless of the scheduled onset
        if self.attack == "A0" or self.t0 is None or tm < self.att_start:
            return u_safe, rep_p, rep_v, False
        now = time.monotonic()
        cands = []
        for j, q in self.pk.items():
            pj = q.p_msg + max(0.0, now - q.t_rx) * q.v_msg
            cands.append((float(np.linalg.norm(pj - p)), j, pj))
        if not cands:
            return u_safe, rep_p, rep_v, False
        d, j, pj = min(cands)
        to = (pj - p)
        to[2] = 0.0
        to = to / max(np.linalg.norm(to), 1e-9)
        u, active = u_safe, False
        if self.attack in ("A8", "A14") and d <= self.trig_r:
            un = u_nom * min(1.0, self.B.U / max(float(np.linalg.norm(u_nom)), 1e-9))
            u = un + self.eps * to                          # bypasses its own safety layer; ||u|| <= U + eps
            active = True
        if self.attack == "A15" and d <= self.trig_r:
            u = (self.B.U + self.eps) * to                  # pursuit: full authority at the nearest neighbour
            active = True
        if self.attack in ("A4", "A14", "A15"):
            rep_p = p - self.beta_att * to                  # appear farther away from the target
            if self.beta_v_att > 0:
                vc = max(0.0, float(v @ to))                # hide closing speed within beta_v
                rep_v = v - min(vc, self.beta_v_att) * to
            active = True
        if self.attack == "A5":
            old = [h for h in self.hist if h[0] <= now - self.replay_s]
            if old:
                rep_p, rep_v = old[-1][1].copy(), old[-1][2].copy()
                active = True
        return u, rep_p, rep_v, active

    def poll_live(self):
        now = time.monotonic()
        if now < self.live_poll:
            return
        self.live_poll = now + 0.25
        try:
            L = json.loads(self.live_path.read_text())
        except (OSError, ValueError):
            return
        if L == self.live:
            return
        self.live = L
        if L.get("attack") in ("A4", "A5", "A8", "A14", "A15"):
            self.attack = L["attack"]
        self.eps = float(L.get("epsilon", self.eps))
        self.beta_att = float(L.get("beta", self.beta_att))
        self.trig_r = float(L.get("trigger_radius", self.trig_r))
        self.event("live_attack", **L)

    # ------------------------------------------------------------------------------------------
    def status(self):
        self.st_pub.publish(String(data=json.dumps({"id": self.i, "phase": self.phase})))

    def broadcast(self, now, p, v):
        rate = self.tx_fast if now < self.fast_until else self.tx_base
        if now < self.next_tx:
            return rate
        self.next_tx = now + 1.0 / rate
        self.seq += 1
        self.tx_pub.publish(Float64MultiArray(data=[float(self.i), float(self.seq), now, *map(float, p), *map(float, v)]))
        self.tx_count += 1
        self.tx_bytes += 72
        return rate

    def loop(self):
        now = time.monotonic()
        loop_dt = math.nan if self.t_last_loop is None else now - self.t_last_loop
        self.t_last_loop = now
        self.status()
        veh = self.veh
        if not veh.has_state():
            return
        p, v = veh.pos_enu(), veh.vel_enu()
        self.hist.append((now, p.copy(), v.copy()))
        if self.energy is not None:
            self.energy.update(now, self.power.power_w() if self.phase not in ("WAIT", "BOOT", "DONE") else 0.0)

        if self.phase == "WAIT":
            self.set_phase("BOOT")
        if self.phase == "BOOT":
            if now - self.t_phase > 2.0 and veh.bootstrap(now, self.alt):
                self.set_phase("TAKEOFF")
            else:
                veh.publish_mode("position"); veh.hover_setpoint(self.alt)
            if now - self.t_phase > 60.0:
                self.event("arm_timeout"); self.set_phase("LAND")
            return
        if self.phase in ("TAKEOFF", "HOLD"):
            veh.publish_mode("position"); veh.hover_setpoint(self.alt)
            self.broadcast(now, p, v)
            if self.phase == "TAKEOFF" and abs(p[2] - self.alt) < 0.3 and np.linalg.norm(v) < 0.3:
                self.hold_ok_since = self.hold_ok_since or now
                if now - self.hold_ok_since > 1.0:
                    self.set_phase("HOLD")
            if self.phase == "HOLD" and self.t0 is not None:
                self.set_phase("MISSION")
            if self.land_cmd:
                self.set_phase("LAND")
            return
        if self.phase == "MISSION":
            if self.land_cmd:
                self.set_phase("LAND")
            else:
                self.mission_tick(now, p, v, loop_dt)
                return
        if self.phase == "LAND":
            veh.publish_mode("position")
            if now - veh.last_land > 2.0:
                veh.request_land(); veh.last_land = now
            self.broadcast(now, p, v)
            if not veh.armed() and now - self.t_phase > 2.0:
                self.set_phase("DONE")
            return
        if self.phase == "DONE":
            self.broadcast(now, p, v)

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
        # command mapping to PX4
        if self.cmd_mode == "velocity":
            vsp = v + u_cmd * self.tau
            vsp[2] = 0.0
            sp = float(np.linalg.norm(vsp[:2]))
            if sp > self.vmax:
                vsp[:2] *= self.vmax / sp
            veh.publish_mode("velocity"); veh.velocity_setpoint(vsp, self.alt)
            spx, spy = vsp[0], vsp[1]
        else:
            ucmd = u_cmd.copy()
            sp = float(np.linalg.norm(v[:2]))
            if sp > self.vmax and float(ucmd[:2] @ v[:2]) > 0:     # speed envelope: no accel beyond vmax
                ucmd[:2] -= (ucmd[:2] @ v[:2]) / (sp * sp) * v[:2]
            veh.publish_mode("acceleration"); veh.accel_setpoint(ucmd, self.alt)
            spx, spy = ucmd[0], ucmd[1]
        rate = self.broadcast(now, rep_p, rep_v)
        if self.route_mode == "out_back" and self.mode == "MISSION" and self.progress >= 2 * float(np.linalg.norm(self.route)) - 1e-6:
            tgt = self.home_c + self.offsets[self.i]
            if np.linalg.norm((p - tgt)[:2]) < 1.0 and np.linalg.norm(v[:2]) < 0.4:
                self.event("mission_complete_landing", t=tm)
                self.set_phase("LAND")
        # return-mode landing at home
        if self.mode == "RETURN_REQUIRED":
            tgt = self.home_c + self.offsets[self.i]
            if np.linalg.norm((p - tgt)[:2]) < 1.0 and np.linalg.norm(v[:2]) < 0.4:
                self.event("return_landing", t=tm)
                self.set_phase("LAND")
        comp_ms = 1e3 * (time.perf_counter() - t_start)
        self.ticks += 1
        if comp_ms > 1e3 * self.dt:
            self.deadline_miss += 1
        a = veh.acc_enu()
        self.w.writerow({
            "t": f"{tm:.3f}", "phase": self.phase, "mode": self.mode, "status": dec.status, "gamma": f"{dec.gamma:.4f}",
            "nominal_unsafe": int(dec.nominal_unsafe), "cert_h": dec.certified_horizon, "min_res": f"{dec.min_row_residual:.4f}",
            "worst_age": f"{dec.worst_age:.3f}", "n_rows": dec.n_rows_agents, "n_excl": dec.n_excluded, "n_pkts": len(self.pk),
            "fail": dec.failure_reason, "relax": f"{dec.relax:.3f}", "refresh": int(dec.refresh), "solve_ms": f"{comp_ms:.2f}",
            "loop_dt": f"{loop_dt:.4f}", "assump_ok": int(dec.assumptions_valid),
            "px": f"{p[0]:.3f}", "py": f"{p[1]:.3f}", "pz": f"{p[2]:.3f}", "vx": f"{v[0]:.3f}", "vy": f"{v[1]:.3f}", "vz": f"{v[2]:.3f}",
            "ax": f"{a[0]:.3f}", "ay": f"{a[1]:.3f}", "unx": f"{u_nom[0]:.3f}", "uny": f"{u_nom[1]:.3f}",
            "ux": f"{u_cmd[0]:.3f}", "uy": f"{u_cmd[1]:.3f}", "spx": f"{spx:.3f}", "spy": f"{spy:.3f}",
            "E_wh": f"{self.energy.E:.4f}" if self.energy else "", "P_w": f"{self.power.power_w():.1f}" if self.energy else "",
            "E_margin": f"{en['margin_wh']:.4f}" if en else "", "tx_rate": rate, "attack_on": int(att_on),
            "rep_bias": f"{float(np.linalg.norm(rep_p - p)):.3f}", "nav": veh.nav_state(), "xy_valid": int(veh.xy_valid),
            "rpx": f"{rep_p[0]:.3f}", "rpy": f"{rep_p[1]:.3f}", "rvx": f"{rep_v[0]:.3f}", "rvy": f"{rep_v[1]:.3f}"})
        if self.ticks % 40 == 0:
            self.f.flush()

    def close(self):
        summ = {"id": self.i, "variant": self.variant, "attack": self.attack, "ticks": self.ticks,
                "deadline_miss": self.deadline_miss, "tx_count": self.tx_count, "tx_bytes": self.tx_bytes,
                "transitions": self.policy.transitions, "final_phase": self.phase, "mode": self.mode,
                "energy_used_wh_est": self.energy.used_wh if self.energy else None,
                "E_end_wh_est": self.energy.E if self.energy else None,
                "E0_wh": self.energy.E0 if self.energy else None}
        if self.transit:
            tr = np.asarray(self.transit)
            summ["transit_s"] = {"n": int(tr.size), "p50": float(np.quantile(tr, 0.5)),
                                 "p99": float(np.quantile(tr, 0.99)), "max": float(tr.max()),
                                 "frac_over_T_net_max": float(np.mean(tr > self.B.T_net_max)),
                                 "T_net_max": self.B.T_net_max}
        (self.run_dir / f"agent_{self.i}_summary.json").write_text(json.dumps(summ, default=str))
        for f in (self.f, self.ev):
            try:
                f.close()
            except Exception:
                pass
        if self.energy is not None:
            self.power.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", default="D26")
    ap.add_argument("--attack", default="A0")
    ap.add_argument("--attacker-id", type=int, default=-1)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    geo = json.loads(Path(a.geometry).read_text())
    rclpy.init()
    node = Agent(a.id, cfg, geo, Path(a.run_dir) / "agents", a.variant, a.attack, a.attacker_id)
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
