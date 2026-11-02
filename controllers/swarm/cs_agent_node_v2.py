#!/usr/bin/env python3
"""Versioned coupled-safety agent (subclass of cs_agent_node.CsAgent; the historical node and its modules are not edited).

Two independent switches in the `cs` config section.  Both default to the historical behaviour.
  cs.exec_chain = "v1" (default) | "v2"
      v2 (honest agents only): ExecV2Policy (swarmsec/ral26_policy_x2.py).  Every command source is solved over the execution set X
      (norm, slew around the PUBLISHED previous command, slew-compatible speed half-space).  The node applies no post-policy
      projection.  The final command is revalidated (ral26_exec.validate_final) and committed as the next slew centre.
      The designated UAV keeps the v1 chain, so the attack policy is identical across honest arms.
  cs.clock = "host" (default) | "plant"
      plant: every schedule runs on Gazebo plant time.  That covers the 20 Hz decision grid, route progress, squeeze schedule, attack
      onset, packet stamps and receiver-local age, broadcast timing and phase timers.
      * plant clock: last /world/default/stats simTime sample + host-monotonic elapsed x RTF estimated from consecutive samples (EWMA),
        extrapolated <= 0.5 s, monotone.  It is the same source the GCS uses for mission duration and its t0_sim marker.
      * mission zero = GCS t0_sim (plant), so agent mission time, attacker onset and the evaluator's scored window share one origin.
      * decisions are due at plant times t_k = k * dt (dt = 1 / rate_hz), polled every 2 ms of host time.  Route progress advances by the
        plant time between executed ticks.  Ticks that pass entirely without a decision are counted as `sched_skip`; there is no
        catch-up.
      * deadline: the command for t_k must be published before plant time t_k + cs.deadline_s (default 0.025 s).  Otherwise it is a
        MISS.  The late result is discarded, the previous published set-point is re-sent (held), the policy's previous-command state is
        restored, and the miss is logged.
      * receiver-local age = plant time at decision - plant time stamped at packet receipt.  Host-monotonic receipt times are kept and
        the host age of the worst packet is logged for comparison (worst_age_host).
      * host times are also logged (t_host_dec, t_host_pub).  PX4-internal acceptance and application are not observable here.
"""
from __future__ import annotations

import csv
import math
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import rclpy
from std_msgs.msg import Float64MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cs_agent_node as cs  # noqa: E402
import px4_iface  # noqa: E402
import uav_agent_node as base  # noqa: E402
from swarmsec import ral26_core as rc  # noqa: E402

_REAL_MONO = time.monotonic
V2 = ["clock", "exec_chain", "t_host_dec", "t_plant_sched", "t_plant_dec", "t_plant_pub", "t_host_pub", "miss", "sched_skip",
      "rtf_est", "worst_age_host", "label_final", "final_why", "exec_replaced", "x_active", "x_c", "sel_x", "sel_y"]
LOG_FIELDS = cs.LOG_FIELDS + V2


class PlantClock:
    def __init__(self, rtf_nominal: float = 1.0):
        from gz_stream import GzStream
        self.lock = threading.Lock()
        self.sample = None
        self.rtf = float(rtf_nominal)
        self.last = -math.inf
        self.n = 0
        self.stream = GzStream("/world/default/stats", self._on)

    def _on(self, msg):
        st = msg.get("simTime") or msg.get("sim_time") or {}
        t = float(st.get("sec", 0)) + 1e-9 * float(st.get("nsec", 0))
        h = _REAL_MONO()
        if t <= 0:
            return
        with self.lock:
            if self.sample is not None and h - self.sample[1] > 1e-3 and t > self.sample[0]:
                r = (t - self.sample[0]) / (h - self.sample[1])
                self.rtf = 0.8 * self.rtf + 0.2 * min(max(r, 0.05), 2.0)
            self.sample = (t, h)
            self.n += 1

    def ready(self):
        return self.sample is not None

    def now(self) -> float:
        with self.lock:
            if self.sample is None:
                return math.nan
            t, h = self.sample
            est = t + min(0.5, max(0.0, _REAL_MONO() - h)) * self.rtf
            est = max(est, self.last)
            self.last = est
            return est


def _shim(clock):
    """Stand-in for the `time` module inside the agent modules: monotonic() -> plant time, everything else real."""
    return types.SimpleNamespace(monotonic=clock.now, perf_counter=time.perf_counter, time=time.time, sleep=time.sleep)


class CsAgentV2(cs.CsAgent):
    def __init__(self, uid, cfg, geo, run_dir, variant, attack, attacker_id, clock=None):
        super().__init__(uid, cfg, geo, run_dir, variant, attack, attacker_id)
        C = cfg.get("cs", {})
        self.exec_chain = str(C.get("exec_chain", "v1"))
        self.clock_mode = "plant" if clock is not None else "host"
        self.clock = clock
        self.designated = uid == attacker_id
        if self.exec_chain == "v2" and not self.designated:
            from swarmsec import ral26_exec as rx
            from swarmsec.ral26_policy_x2 import ExecV2Policy
            R = cfg.get("ral26", {})
            self.policy = ExecV2Policy(self.variant, self.B, static_age_s=float(R.get("static_age_s", 1.0)),
                                       nominal_age_s=float(R.get("nominal_age_s", 0.45)), gamma_warn=float(R.get("gamma_warn", 0.5)),
                                       gamma_refresh=float(R.get("gamma_refresh", 0.75)), info=self.info,
                                       speed_rule=R.get("speed_rule"), vmax=self.vmax)
            rx.prewarm()
        self.pk_host = {}
        self.misses = self.sched_skips = 0
        self.last_sp = None
        self.t_sched = math.nan
        self.f.close()
        self.f = (run_dir / f"agent_{uid}.csv").open("w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=LOG_FIELDS)
        self.w.writeheader()
        self.event("cs_v2_start", exec_chain=self.exec_chain, clock=self.clock_mode,
                   deadline_s=float(C.get("deadline_s", 0.025)) if clock is not None else None)
        if clock is not None:
            self.deadline = float(C.get("deadline_s", 0.025))
            self.dt_nom = self.dt
            self.t_next = None
            self.t_prev_tick = None
            self.timer.cancel()
            self.destroy_timer(self.timer)
            self.timer = self.create_timer(0.002, self._sched)

    # ---- plant-time scheduling ----------------------------------------------------------------------
    def _on_gcs(self, m):
        import json
        d = json.loads(m.data)
        if self.clock_mode == "plant" and d.get("cmd") == "start" and self.t0 is None:
            if d.get("t0_sim") is None:
                return                                  # wait for a marker that carries plant time
            self.t0 = float(d["t0_sim"])
            return
        super()._on_gcs(m)

    def _on_rx(self, m):
        self.pk_host[int(m.data[0])] = _REAL_MONO()
        super()._on_rx(m)

    def _sched(self):
        if not self.clock.ready():
            return
        now = self.clock.now()
        if self.t_next is None:
            self.t_next = (math.floor(now / self.dt_nom) + 1) * self.dt_nom
            return
        if now < self.t_next:
            return
        k_time = self.t_next
        skip = int(math.floor((now - k_time) / self.dt_nom + 1e-9))
        if skip > 0:                                     # whole ticks passed without a decision: no catch-up
            self.sched_skips += skip
            k_time += skip * self.dt_nom
        self.t_next = k_time + self.dt_nom
        self.dt = self.dt_nom if self.t_prev_tick is None else k_time - self.t_prev_tick
        self.t_prev_tick = k_time
        self.t_sched = k_time
        self._skip_now = skip
        self.loop()

    # ---- decision ---------------------------------------------------------------------------------
    def mission_tick(self, now, p, v, loop_dt):
        """cs_agent_node.CsAgent.mission_tick with the two switches (changes marked v2:)."""
        t_start = time.perf_counter()
        t_host_dec = _REAL_MONO()
        tm = now - self.t0
        veh = self.veh
        en = None
        if self.energy is not None:
            dist_home = float(np.linalg.norm((p - (self.home_c + self.offsets[self.i]))[:2]))
            en = self.energy.screen(dist_home)
        u_nom = self.nominal(tm, p, v, self.policy.relax)
        u_prev_saved = self.policy.u_prev.copy()                       # v2: restored if the result misses its deadline
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
        label, why, replaced = dec.status, "", 0
        v2 = self.exec_chain == "v2" and not self.designated
        if v2:                                                         # v2: no post-policy projection; final revalidation
            ucmd, label, why, rep = self.policy.finalize(u_cmd)
            replaced = int(rep)
        else:
            ucmd = u_cmd.copy()
            sp = float(np.linalg.norm(v[:2]))
            if sp > self.vmax and float(ucmd[:2] @ v[:2]) > 0:
                ucmd[:2] -= (ucmd[:2] @ v[:2]) / (sp * sp) * v[:2]
                clip = 1
                self.clip_count += 1
        miss = 0
        t_pub_plant = self.clock.now() if self.clock is not None else math.nan
        if self.clock is not None and t_pub_plant > self.t_sched + self.deadline:
            miss = 1                                                   # v2: late result discarded, previous set-point held
            self.misses += 1
            self.policy.u_prev = u_prev_saved
            ucmd = self.last_sp if self.last_sp is not None else np.zeros(3)
            label = "DEADLINE_MISS_HELD"
        veh.publish_mode("acceleration")
        veh.accel_setpoint(ucmd, self.alt)
        if not miss:
            self.last_sp = np.array(ucmd, float).copy()
        t_pub = base.time.monotonic()                                  # node clock domain (plant in plant mode)
        t_host_pub = _REAL_MONO()
        status = label
        if not v2 and clip and status == rc.CERTIFIED:
            status = "UNCERTIFIED_POSTCLIP"
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
        e2e_ms = 1e3 * (base.time.monotonic() - veh.t_rx) if veh.t_rx else math.nan
        self.ticks += 1
        if comp_ms > 1e3 * (self.dt_nom if self.clock is not None else self.dt):
            self.deadline_miss += 1
        a = veh.acc_enu()
        worst_host = max((max(0.0, t_host_dec - h) for h in self.pk_host.values()), default=0.0)
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
        row.update({k: v for k, v in (br or {}).items() if k in cs.BR})
        if spd is not None:
            row.update(spd_dmin=f"{spd['d_min']:.3f}", spd_cap=f"{spd['cap']:.3f}", spd_brake=spd["brake"])
        row.update(t_mono=f"{now:.4f}", t_state_rx=f"{veh.t_rx:.4f}", t_px4=f"{getattr(veh, 't_px4', math.nan):.4f}",
                   t_pub=f"{t_pub:.4f}", tick=self.ticks)
        row.update(clock=self.clock_mode, exec_chain=self.exec_chain if not self.designated else "v1", t_host_dec=f"{t_host_dec:.4f}",
                   t_plant_sched=f"{self.t_sched:.4f}" if self.clock is not None else "", t_plant_dec=f"{now:.4f}" if self.clock is not None else "",
                   t_plant_pub=f"{t_pub_plant:.4f}" if self.clock is not None else "", t_host_pub=f"{t_host_pub:.4f}", miss=miss,
                   sched_skip=getattr(self, "_skip_now", 0), rtf_est=f"{self.clock.rtf:.4f}" if self.clock is not None else "",
                   worst_age_host=f"{worst_host:.3f}", label_final=label, final_why=why, exec_replaced=replaced,
                   x_active=(br or {}).get("x_active", ""), x_c=(br or {}).get("x_c", ""), sel_x=f"{u[0]:.4f}", sel_y=f"{u[1]:.4f}")
        self.w.writerow(row)
        if self.ticks % 40 == 0:
            self.f.flush()

    def close(self):
        self.event("cs_v2_close", misses=self.misses, sched_skips=self.sched_skips,
                   exec_replaced=getattr(self.policy, "exec_replaced", None),
                   clock_samples=self.clock.n if self.clock is not None else None)
        super().close()


def main():
    import argparse
    import json
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
    clock = None
    if str(cfg.get("cs", {}).get("clock", "host")) == "plant":
        clock = PlantClock(float(cfg.get("cs", {}).get("rtf_nominal", 0.5)))
        t_wait = _REAL_MONO()
        while not clock.ready() and _REAL_MONO() - t_wait < 60.0:
            time.sleep(0.05)
        sh = _shim(clock)
        base.time = sh                      # every schedule in the agent modules now reads plant time
        cs.time = sh
        px4_iface.time = sh
    rclpy.init()
    node = CsAgentV2(a.id, cfg, geo, Path(a.run_dir) / "agents", a.variant, a.attack, a.attacker_id, clock=clock)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.002 if clock is not None else 0.1)
            if node.phase == "DONE" and node.land_cmd and base.time.monotonic() - node.t_phase > 1.0:
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
