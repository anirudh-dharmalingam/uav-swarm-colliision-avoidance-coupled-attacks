#!/usr/bin/env python3
"""Leader-election study: per-UAV process on PX4 SITL + Gazebo X500 (one OS process per vehicle).

Information boundary (same as RAL26 uav_agent_node): this process reads ONLY
  * its own PX4 estimate        /uav_<i>/fmu/out/vehicle_local_position_v1, vehicle_status_v1
  * packets the FANET bridge delivered to it    /ral/rx/uav_<i>   (election packets, swarmsec.le.protocol.Fld)
  * its own proximity sensor    /le/det/uav_<i>   (anonymous detections from le_sensor_node, 10 Hz)
  * its own rotor-speed commands (shaft-power energy estimate, energy_supervisor.PowerMonitor)
  * GCS start/land markers      /ral/gcs
  * the PUBLIC mission plan and fleet registry (classes) from the resolved config.
It runs the SAME election / sensing-verification / mission-layer / CBF code as the Layer-1 simulator
(swarmsec.le), restricted to its own row: every matrix-shaped state of swarmsec.le.protocol.Election is
indexed [receiver, sender] and only row i is ever filled here.

All LE logic runs in the MISSION frame (ENU minus the home centroid at altitude); accelerations are
frame-invariant.  Commands: jerk-limited xy acceleration setpoints (HF1: slew 0.4 m/s^2 per 50 ms tick),
z position hold, speed envelope 4 m/s.

Clock: the MISSION clock (plan reference, protocol timers, attack/fault onsets, packet stamps, energy) runs on
Gazebo SIM time, read from /world/default/stats (a clock source only, like GNSS time; no neighbour state) and
interpolated between messages as monotonic x RTF, anchored at the GCS start marker.  PX4 v1.17 standalone SITL
is NOT lockstep (its estimate timestamps are real time), hence the RTF >= 0.95 validity rule.  Wall-clock time is used only for process-level timeouts (boot/arm/land) and sensor freshness.
The jerk limit is a rate (slew / nominal tick, i.e. 8 m/s^3) applied over the sim time between ticks.
Disturbance observer (D14, all schemes alike): PX4's acceleration-offboard interface bypasses its position and
velocity loops, so nothing rejects wind.  d_hat = low-pass(tau_dob) of (EKF acceleration - previous xy command),
|d_hat| <= dob_max, is subtracted from the command (standard output-disturbance feed-forward).

Compromised endpoint (research sandbox): --attack L* / --phys A*, either a fixed --attacker-id or, for the
insider attacks (attacker-id -2), the process that is LEADER when the attack starts.
Faults hit the process that is leader at fault time: F_CRASH (fail-silent: stops transmitting, lands),
F_GNSS (estimate bias ramp 0.5 m/s up to 10 m, flown and broadcast), F_BATT (x6 drain), F_RADIO (bridge
range x0.25, signalled to the bridge).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from px4_iface import PX4Vehicle  # noqa: E402
from swarmsec.le import fastqp, intervene  # noqa: E402
from swarmsec.le.cmdchain import shape_command  # noqa: E402
from swarmsec.le.adversary import INSIDER, Adversary  # noqa: E402
from swarmsec.le.control import nominal  # noqa: E402
from swarmsec.le.fleet import Fleet, FleetConfig  # noqa: E402
from swarmsec.le.mission_layer import node_step  # noqa: E402
from swarmsec.le.missions import make_plan, rotate_plan  # noqa: E402
from swarmsec.le.protocol import Election, Fld, ROLE_L, SCHEMES, frozen_params  # noqa: E402
from swarmsec.le.sensing import BasicSensor, SensorConfig  # noqa: E402

LOG = ["t", "tsim", "phase", "role", "term", "leader", "led_by", "wp", "vux", "vuy", "crx", "cry", "cusum", "n_distrust",
       "att", "E_wh", "P_w", "px", "py", "pz", "vx", "vy", "ux", "uy", "spx", "spy", "n_pkts", "n_det", "n_ghost",
       "slack", "solve_ms", "loop_dt", "leaving", "fault", "dm", "dhx", "dhy",
       # resilience campaign (ral_resilience_*): command-chain audit and local anchor/authority state
       "cobx", "coby", "scl", "unx", "uny", "ucx", "ucy", "slw", "sat", "spd", "dts", "nqc", "shrej", "crux", "cruy", "burst"]


def cfg_get(cfg, path, default):
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


class LEAgent(Node):
    def __init__(self, uid, cfg, geo, run_dir: Path, scheme, attack, attacker_id, phys):
        super().__init__(f"le_agent_{uid}")
        self.i, self.cfg = uid, cfg
        E = cfg.get("le", {})
        self.n = n = int(cfg_get(cfg, "swarm.num_uavs", 4))
        self.alt = float(cfg_get(cfg, "swarm.altitude_m", 5.0))
        self.dt = 1.0 / float(E.get("rate_hz", 20.0))
        self.proto_dt = 1.0 / float(E.get("proto_hz", 10.0))
        self.vmax = float(E.get("max_speed_mps", 4.0))
        self.slew = float(E.get("slew", 0.4))
        self.u_max = float(E.get("u_max", 3.0))
        self.dob = bool(E.get("dob", True))
        self.tau_dob = float(E.get("tau_dob", 1.5))
        self.dob_max = float(E.get("dob_max", 2.0))
        self.d_hat = np.zeros(2)
        self.spawn = {k: np.array(geo["spawn_positions"][str(k)], float) for k in range(n)}
        self.home = np.mean([self.spawn[k] for k in range(n)], axis=0)
        self.home[2] = self.alt
        off = np.array([geo["desired_offsets"][str(k)] for k in range(n)], float)
        self.offsets = off - off.mean(0)
        self.offsets[:, 2] = 0.0
        spacing = float(cfg_get(cfg, "formation.spacing_m", 6.0))
        self.spacing = spacing
        self.controller = E.get("controller", "LF")
        self.radius = float(E.get("radius_m", 2.0 * spacing))
        foot = 2 * float(np.max(np.linalg.norm(self.offsets[:, :2], axis=1))) + spacing
        self.plan = make_plan(E.get("mission", "delivery"), foot, float(E.get("cruise", 1.5)),
                              scale=float(E.get("plan_scale", 1.0)))
        th = float(E.get("heading_rad", 0.0))                 # scenario route heading (resilience campaign)
        if th:
            self.plan = rotate_plan(self.plan, th)
        self.res_mode = bool(E.get("res", False))
        self.t_boot = float(E.get("t_boot", 6.0))
        # public fleet registry (classes) and this UAV's own battery
        classes = list(E.get("classes"))
        fcfg = FleetConfig(p_lead_w=float(E.get("p_lead_w", 24.0)))
        self.fleet = Fleet(n, self.dt, np.random.default_rng(0), fcfg, classes=classes)
        self.fleet.cap = np.asarray(E.get("cap_wh"), float)          # mission-scaled nameplates (public)
        self.E = float(E.get("E0_wh")[uid])                          # own state of charge (private)
        self.p_lead = fcfg.p_lead_w
        self.e_land = float(E.get("e_land_wh", 1.0))
        self.need_rate = float(E.get("need_rate_wh_s"))
        seed = int(cfg_get(cfg, "experiment.seed", 0))
        prm = frozen_params(self.controller, **({**E.get("params", {}), "local_membership": True} if self.res_mode else E.get("params", {})))
        self.el = Election(n, scheme, self.plan, self.fleet, self.offsets, np.random.default_rng(seed * 100 + uid), prm,
                           designated=int(E.get("designated", 0)), f_need=self._need)
        self.el.t_boot = self.t_boot
        self.S = SCHEMES[scheme]
        self.gate, self.impeach = bool(self.S.get("gate")), bool(self.S.get("impeach"))
        self.use_sense = bool(self.S.get("sense"))
        self.sbcbf = bool(self.S.get("sbcbf")) and self.use_sense
        self.shield = bool(self.S.get("shield"))
        self.local_wp = bool(self.S.get("local_wp"))
        self.v2 = bool(self.S.get("iface2"))
        self.ctrl_kind = "VS" if (self.S["kind"] == "local" or self.v2) else self.controller     # LOCAL / CADLE_V2: anchor-reference tracking
        if self.v2:
            prm.delta_max = prm.v2_delta
        self.shield_r = prm.shield_tight_r if self.S.get("shield_tight") else prm.tol_c
        self.inst = E.get("install")                       # {"t","hold_until"}: experimenter intervention (swarmsec.le.intervene)
        self.att_fixed = int(E.get("attacker_id", -1))
        self._inst_done = False
        self.sensor = BasicSensor(n, np.random.default_rng(seed * 100 + 50 + uid), SensorConfig(),
                                  nav_sigma=self.fleet.nav_sigma)
        self.V = np.zeros((n, n, Fld.width(n)))
        self.V[:, :, Fld.LEADER] = -1
        self.V[:, :, Fld.VOTE] = -1
        self.t_rx = np.full((n, n), -np.inf)
        self.t_tx = np.full((n, n), -np.inf)
        self.dets = np.zeros((0, 3))
        self.t_det = -np.inf
        self.alive_self = np.zeros(n, bool)
        self.alive_self[uid] = True
        self.s_lead = 1.0
        self.u_prev = np.zeros(3)
        self.v_fly = np.zeros(3)
        self.leaving = False
        # attack / faults
        A = E.get("attack", {})
        self.attack, self.phys = attack.upper(), (phys or "none")
        # legacy: insider attacks re-target to whoever leads (attacker-id -2).  Resilience campaign: the attacker is the fixed
        # manifest identity, so no other process may ever turn itself into an adversary.
        self.insider = intervene.insider_flag(attacker_id, self.attack, self.res_mode, INSIDER)
        self.att_start = self.att_start0 = float(A.get("start_s", 25.0))
        self.compromised = (attacker_id == uid) and (self.attack != "L0" or self.phys != "none")
        # demo control panel (scripts/demo): live attack switch read from <run>/live_attack.json; only when the
        # run config sets demo.live, so campaign episodes never read it
        self.live_path = run_dir.parent / "live_attack.json" if cfg.get("demo", {}).get("live") else None
        self.live, self.live_poll = {}, 0.0
        if self.live_path is not None and (self.insider or self.compromised):
            try:
                if json.loads(self.live_path.read_text()).get("mode") == "off":
                    self.att_start = float("inf")          # held until the panel switches it ON
            except (OSError, ValueError):
                pass
        self.adv = None
        self.fault = None
        self.faults = list(E.get("faults", []))
        self.fault_t = float(E.get("fault_t", 40.0))
        self.fault_bias = np.zeros(3)
        self.drain = 1.0
        self.silent = False
        self.power = None
        if E.get("energy_power", True):
            from energy_supervisor import PowerMonitor
            self.power = PowerMonitor(uid, float(E.get("inflation", 1.0)))
        self.veh = PX4Vehicle(self, uid, self.spawn[uid])
        from gz_stream import GzStream
        self.clk_stream = GzStream("/world/default/stats", self._on_stats)
        self.create_subscription(Float64MultiArray, f"/ral/rx/uav_{uid}", self._on_rx, 200)
        self.create_subscription(Float64MultiArray, f"/le/det/uav_{uid}", self._on_det, 20)
        self.create_subscription(String, "/ral/gcs", self._on_gcs, 10)
        self.tx_pub = self.create_publisher(Float64MultiArray, "/ral/tx", 50)
        self.ctl_pub = self.create_publisher(String, "/le/ctl", 10)
        self.st_pub = self.create_publisher(String, f"/ral/status/uav_{uid}", 10)
        self.phase, self.t_phase = "WAIT", time.monotonic()
        self.t0 = None
        self.land_cmd = False
        self.hold_ok_since = None
        self.t_last_loop = None
        self.next_proto = 0.0
        self.ticks = 0
        self.tx_count = 0
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.f = (run_dir / f"le_agent_{uid}.csv").open("w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=LOG)
        self.w.writeheader()
        self.ev = (run_dir / f"le_agent_{uid}_events.jsonl").open("w")
        self.event("start", scheme=scheme, attack=self.attack, phys=self.phys, compromised=self.compromised,
                   cls=classes[uid], E0=self.E, controller=self.controller, mission=self.plan.kind)
        if self.compromised:
            self._make_adversary()
        self.timer = self.create_timer(self.dt, self.loop)

    # ------------------------------------------------------------------ helpers
    def _need(self, k, c):
        return self.plan.remaining_time(int(k), np.asarray(c)[:2]) * self.need_rate + self.e_land + 0.3

    def _make_adversary(self):
        from swarmsec.attacks import build_attack
        ph = None
        if self.phys != "none":
            ph = build_attack(self.phys if self.phys != "A14" else "A8", attacker=self.i,
                              epsilon=float(self.cfg["le"].get("attack", {}).get("epsilon", 2.0)),
                              neighbor_radius=self.radius, rng=np.random.default_rng(7), offsets=self.offsets)
        self.adv = Adversary(self.attack, self.i, self.plan, self.fleet, self.offsets, self.att_start,
                             np.random.default_rng(11), phys=ph, params=self.el.p)
        A = self.cfg["le"].get("attack", {})
        self.adv.post = A.get("post", "LSTEALTH")
        self.adv.end = float(A.get("end_s", float("inf")))
        self.adv.burst_speed = float(A.get("burst_speed", self.adv.burst_speed))
        self.adv.stall_k = int(A.get("stall_k", 0))
        self.adv.auth = dict(A.get("auth") or {})
        self.compromised = True
        self.event("compromised", attack=self.attack, phys=self.phys)

    def event(self, name, **kw):
        self.ev.write(json.dumps({"mono": time.monotonic(), "sim": self.sim_now() if hasattr(self, "veh") else None,
                                  "event": name, **kw}, default=str) + "\n")
        self.ev.flush()

    def set_phase(self, p):
        self.event("phase", frm=self.phase, to=p)
        self.phase, self.t_phase = p, time.monotonic()

    def tm(self, now):
        return now - self.t0 if self.t0 is not None else -1.0

    def _on_stats(self, msg):
        st = msg.get("simTime") or {}
        t = float(st.get("sec", 0)) + 1e-9 * float(st.get("nsec", 0))
        r = msg.get("realTimeFactor")
        if t > 0:
            self._clk = (t, time.monotonic(), float(np.clip(float(r), 0.0, 1.5)) if r is not None else 1.0)

    def sim_now(self):
        """Gazebo sim time: last /world/default/stats stamp + elapsed monotonic x RTF (<= 1 s extrapolation)."""
        c = getattr(self, "_clk", None)
        if c is None:
            return math.nan
        return c[0] + min(1.0, max(0.0, time.monotonic() - c[1])) * c[2]

    # ------------------------------------------------------------------ inputs
    def _on_rx(self, m):
        d = m.data
        j = int(d[0])
        if j == self.i or len(d) < 3 + Fld.width(self.n):
            return
        if self.t0 is None:
            return
        now = self.sim_now()
        t_send = float(d[2]) - self.t0                  # sender stamp (shared Gazebo sim clock)
        if t_send <= self.t_tx[self.i, j]:
            return
        self.V[self.i, j] = np.asarray(d[3:3 + Fld.width(self.n)], float)
        self.t_tx[self.i, j] = t_send
        self.t_rx[self.i, j] = now - self.t0

    def _on_det(self, m):
        d = np.asarray(m.data, float)
        self.dets = d.reshape(-1, 3) if d.size else np.zeros((0, 3))
        self.t_det = time.monotonic()

    def _on_gcs(self, m):
        d = json.loads(m.data)
        if d.get("cmd") == "start" and self.t0 is None:
            # anchor the GCS start instant (container monotonic) on the sim clock; the delivery latency is
            # measured exactly on the shared monotonic clock
            ts = self.sim_now()
            if not math.isfinite(ts):
                return
            self.t0 = ts - max(0.0, time.monotonic() - float(d["t0"])) * self._clk[2]
            self.event("t0", t0_sim=self.t0, t0_mono=float(d["t0"]))
        if d.get("cmd") in ("land", "done"):
            self.land_cmd = True

    # ------------------------------------------------------------------ protocol tick (10 Hz)
    def poll_live(self, t):
        """Demo panel: ON fires the attack now, OFF stops it (or holds a scheduled one), auto restores the schedule.
        A new L* kind of the same class (insider / outsider) replaces the running one.  Insider attacks: if nobody
        was compromised yet, the process that leads at the ON moment becomes the insider (as at a scheduled onset)."""
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
        k = str(L.get("attack", self.attack)).upper()
        if k != self.attack and k.startswith("L") and k != "L0" and (k in INSIDER) == (self.attack in INSIDER):
            self.attack = k
            if self.adv is not None:
                self.adv.kind = k
        mode = L.get("mode", "auto")
        self.att_start = {"on": t, "off": float("inf")}.get(mode, self.att_start0)
        if self.adv is not None:
            self.adv.start = self.att_start
            self.adv.end = float("inf") if mode == "on" else float(self.cfg["le"].get("attack", {}).get("end_s", float("inf")))
        elif mode == "on" and self.insider and not any(
                '"compromised"' in f.read_text() for f in self.run_dir.glob("le_agent_*_events.jsonl")):
            self._insider_checked = False
        if self.adv is not None or self.compromised or (self.insider and self.el.role[self.i] == ROLE_L):
            self.event("live_attack", t=t, **L)

    def protocol_tick(self, t, p, v):
        i, n, el = self.i, self.n, self.el
        if self.live_path is not None and (self.insider or self.compromised):
            self.poll_live(t)
        if self.adv is None and self.insider and t >= self.att_start - 0.5 and el.role[i] == ROLE_L \
                and (self.attack != "L0" or self.phys != "none") and not getattr(self, "_insider_checked", False):
            self._make_adversary()
        if t >= self.att_start - 0.5:
            self._insider_checked = True
        # faults hit whoever leads at fault time
        if self.faults and self.fault is None and t >= self.fault_t:
            self.fault = "none"
            if el.role[i] == ROLE_L and not self.compromised:
                self.fault = "+".join(self.faults)
                self.event("fault", faults=self.faults, t=t)
                if "F_CRASH" in self.faults:
                    self.silent = True
                if "F_BATT" in self.faults:
                    self.drain = 6.0
                if "F_RADIO" in self.faults:
                    self.ctl_pub.publish(String(data=json.dumps({"radio_mult": {str(i): 0.25}})))
        if self.fault and "F_GNSS" in self.fault and np.linalg.norm(self.fault_bias) < 10.0:
            a = 2.0 * math.pi * ((self.i * 0.618) % 1.0)
            self.fault_bias += 0.5 * self.proto_dt * np.array([math.cos(a), math.sin(a), 0.0])
        own_p = np.zeros((n, 3))
        own_v = np.zeros((n, 3))
        own_p[i], own_v[i] = p, v
        heard = ((t - self.t_rx) <= el.p.t_alive).astype(float)
        np.fill_diagonal(heard, 0.0)
        navrep = -np.ones((n, n))
        if self.use_sense:
            navrep = np.where(self.sensor.seen, 1.0 - np.clip(self.sensor.resid / el.p.nav_scale, 0, 1), -1.0)
            np.fill_diagonal(navrep, -1.0)
        nav_claim = 1.0 - np.clip(self.fleet.nav_sigma * 1.5 / el.p.nav_scale, 0, 1)
        E_vec = np.zeros(n)
        E_vec[i] = self.E
        P = el.compose(own_p, own_v, E_vec, nav_claim, heard, navrep, t)
        if self.E <= self.e_land + el.p.leave_margin_wh:
            P[i, Fld.LEAVE] = 1.0
            P[i, Fld.ECL] = 0.0
            if not self.leaving:
                self.leaving = True
                self.event("leaving", E=self.E)
        if self.adv is not None:
            P = self.adv.edit_packet(P, t, el, self.V, None)
        self.V[i, i] = P[i]
        self.t_rx[i, i] = self.t_tx[i, i] = t
        if not self.silent and (self.adv is None or self.adv.send_mask(t, el, self._view_pos()) is None):
            self._send(P[i], targets=None)
        elif not self.silent:
            mask = self.adv.send_mask(t, el, self._view_pos())
            self._send(P[i], targets=[int(k) for k in np.nonzero(mask[:, i])[0] if k != i])
        fresh_v = ((t - self.t_rx) <= 0.35)
        if self.use_sense and time.monotonic() - self.t_det < 0.3:
            dets = [np.zeros((0, 3))] * n
            dets[i] = self.dets
            cons, incons, _ = self.sensor.verify(dets, own_p, self.V[:, :, Fld.PX:Fld.PZ + 1], fresh_v,
                                                 self.alive_self, distrust=el.own_distrust())
            el.add_evidence(cons, incons)
        if self.adv is not None:
            self.adv.pre_step(t, el, self.V)
        fresh = el.step(t, self.V, self.t_rx, None, own_p, own_v, E_vec, t, self.alive_self,
                        sense_nav=self.sensor.resid if self.use_sense else None,
                        sensed=self.sensor.seen if self.use_sense else None, adversary=self.adv)
        if self.inst and self.att_fixed >= 0 and self.S["kind"] == "quorum":
            if not self._inst_done and t >= float(self.inst["t"]):
                intervene.install(el, [i], self.att_fixed, t)
                self._inst_done = True
                self.event("install", t=t, att=self.att_fixed)
            if self._inst_done and t < float(self.inst.get("hold_until", self.inst["t"])):      # no hold unless hold_until is given
                intervene.hold(el, [i], self.att_fixed, t, t_rx=self.t_rx, silent_s=self.inst.get("release_silent_s"))
        Dm = el.distrusted(self.V, fresh) if (self.impeach or self.use_sense) else np.zeros((n, n), bool)
        self.dm_mask = int(sum(1 << int(k) for k in np.nonzero(Dm[i])[0] if k != i))
        self.excl = Dm if (self.use_sense and self.S.get("tctl")) else np.zeros((n, n), bool)
        use_i = fresh[i] & ~self.excl[i]
        use_i[[k for k in range(n) if k != i and self.V[i, k, Fld.LEAVE] > 0.5]] = False
        self.v_fly, self.s_lead = node_step(el, i, self.V, use_i, self.offsets, self.plan, self.s_lead, t,
                                            self.t_boot, self.gate, self.impeach, self.t_tx[i], p, adv=self.adv,
                                            shield=self.shield, local_wp=self.local_wp, leader_la=self.v2)
        self.fresh = fresh

    def _view_pos(self):
        return self.V[self.i, :, Fld.PX:Fld.PZ + 1]

    def _send(self, row, targets):
        now = self.sim_now()
        meta = [float(self.i), float(self.tx_count), now]
        if targets is not None:
            meta[1] = -1.0 - float(sum(1 << k for k in targets))   # bridge reads a negative seq as a target mask
        self.tx_pub.publish(Float64MultiArray(data=meta + [float(x) for x in row]))
        self.tx_count += 1

    # ------------------------------------------------------------------ control tick (20 Hz)
    def control(self, t, p, v):
        i, n, el = self.i, self.n, self.el
        V = self.V[i]
        Pv = np.zeros((n, n, 3))
        Vv = np.zeros((n, n, 3))
        Pv[i] = V[:, Fld.PX:Fld.PZ + 1] + V[:, Fld.VX:Fld.VZ + 1] * np.clip(t - self.t_tx[i], 0, 1.2)[:, None]
        Vv[i] = V[:, Fld.VX:Fld.VZ + 1]
        Pv[i, i], Vv[i, i] = p, v
        fresh_c = np.zeros((n, n), bool)
        fresh_c[i] = ((t - self.t_rx[i]) <= 1.0) & ~getattr(self, "excl", np.zeros((n, n), bool))[i]
        fresh_c[i] &= self.V[i, :, Fld.LEAVE] < 0.5
        own_p = np.zeros((n, 3))
        own_v = np.zeros((n, 3))
        own_p[i], own_v[i] = p, v
        alive = self.alive_self
        u_nom = nominal(self.ctrl_kind, own_p, own_v, Pv, Vv, fresh_c, alive, el.led_by, el.v_used, el.cref_used,
                        el.scale_used, self.offsets, self.spacing, radius=self.radius, gate=self.gate, z_ref=0.0,
                        shield=self.shield, shield_r=self.shield_r, v2=self.v2, v2_kp=self.el.p.v2_kp, v2_kv=self.el.p.v2_kv)
        if self.ctrl_kind == "LF" and el.led_by[i] == i:
            u_nom[i, :2] += (self.v_fly[:2] - el.v_used[i, :2])          # an insider flies what it chose to fly
        sg = self.fleet.nav_sigma
        ds = 2.5 + 1.5 * np.sqrt(sg[:, None] ** 2 + sg[None, :] ** 2)
        cols_alive = fresh_c[i].copy()
        cols_alive[i] = True
        if self.sbcbf:
            Pv[i] = np.where(self.excl[i][:, None], 1e6, Pv[i])
            if time.monotonic() - self.t_det < 0.5 and len(self.dets):
                # unexplained detections = ghosts (last verification); conservatively use all detections not
                # within the association gate of a used claim
                rel = Pv[i] - p
                g = [d for d in self.dets if not np.any(np.linalg.norm(rel[cols_alive][:, :2] - d[:2], axis=1)
                                                        <= self.sensor.gate_a[i, cols_alive])]
                if g:
                    G = len(g)
                    vf = Vv[i, fresh_c[i]].mean(0) if fresh_c[i].any() else v
                    Pg = np.full((n, G, 3), 1e6)
                    Pg[i] = p + np.asarray(g)
                    Vg = np.zeros((n, G, 3))
                    Vg[i] = vf
                    Pv = np.concatenate([Pv, Pg], 1)
                    Vv = np.concatenate([Vv, Vg], 1)
                    ds = np.concatenate([ds, np.full((n, G), 3.0)], 1)
                    cols_alive = np.concatenate([cols_alive, np.ones(G, bool)])
        A, c, _ = fastqp.hocbf_rows(own_p, own_v, Pv, Vv, ds, 2.0, 2.0, self.radius, 6, alive=cols_alive)
        u, slack, _ = fastqp.solve(u_nom[[i]], A[[i]], c[[i]], self.u_max)
        u = u[0]
        slack = float(slack[0])
        if self.adv is not None:
            P_view = Pv[i, :n].copy()
            V_view = Vv[i, :n].copy()
            out = self.adv.physical(t, el, P_view, V_view, self.dt)
            if out is not None and out.triggered:
                base = u_nom[i] if out.bypass_self_filter else u
                u = base + out.delta_u
        return u, u_nom[i], slack

    # ------------------------------------------------------------------ main loop
    def status(self):
        self.st_pub.publish(String(data=json.dumps({"id": self.i, "phase": self.phase})))

    def loop(self):
        now = time.monotonic()
        loop_dt = math.nan if self.t_last_loop is None else now - self.t_last_loop
        self.t_last_loop = now
        self.status()
        veh = self.veh
        if not veh.has_state():
            return
        p_w, v = veh.pos_enu(), veh.vel_enu()
        p = p_w - self.home + self.fault_bias                    # own estimate in the mission frame
        t_sim = self.sim_now()
        if self.phase in ("MISSION",) and self.power is not None and math.isfinite(t_sim):
            P_w = self.power.power_w()
            prev = getattr(self, "_t_sim_prev", None)
            dts = 0.0 if prev is None else max(0.0, min(0.5, t_sim - prev))
            if math.isfinite(P_w):                     # integrate on Gazebo SIM time, not wall time
                lead = self.p_lead if self.el.role[self.i] == ROLE_L else 0.0
                self.E -= (P_w + lead) * self.drain * dts / 3600.0
        self._t_sim_prev = t_sim if math.isfinite(t_sim) else None
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
            if self.phase == "TAKEOFF" and abs(p_w[2] - self.alt) < 0.3 and np.linalg.norm(v) < 0.3:
                self.hold_ok_since = self.hold_ok_since or now
                if now - self.hold_ok_since > 1.0:
                    self.set_phase("HOLD")
            if self.phase == "HOLD" and self.t0 is not None:
                self.set_phase("MISSION")
            if self.land_cmd:
                self.set_phase("LAND")
            return
        if self.phase == "MISSION":
            ts = self.sim_now()
            if self.land_cmd or (self.leaving and self.E <= self.e_land) or (self.silent and self.tm(ts) > self.fault_t + 1.0):
                if self.leaving:
                    self.event("energy_landing", E=self.E)
                self.set_phase("LAND")
            else:
                self.mission_tick(ts, p, v, loop_dt)
                return
        if self.phase == "LAND":
            veh.publish_mode("position")
            if now - veh.last_land > 2.0:
                veh.request_land(); veh.last_land = now
            if not veh.armed() and now - self.t_phase > 2.0:
                self.set_phase("DONE")
            return

    def mission_tick(self, now, p, v, loop_dt):
        t0 = time.perf_counter()
        t = self.tm(now)
        if t >= self.next_proto:
            self.next_proto = t + self.proto_dt
            self.protocol_tick(t, p, v)
        u, un, slack = self.control(t, p, v)
        # PX4 command mapping (HF1-identified jerk limit, as a rate over sim time) + speed envelope
        dts = self.dt if getattr(self, "_t_ctl", None) is None else float(np.clip(t - self._t_ctl, 0.0, 2.0 * self.dt))
        self._t_ctl = t
        slew = self.slew * dts / self.dt
        if self.dob and self.veh.acc is not None:
            a_meas = self.veh.acc_enu()[:2]
            self.d_hat += min(1.0, dts / self.tau_dob) * (a_meas - self.u_prev[:2] - self.d_hat)
            nd_ = float(np.linalg.norm(self.d_hat))
            if nd_ > self.dob_max:
                self.d_hat *= self.dob_max / nd_
        uc2, fl = shape_command(u[:2], self.u_prev[:2], self.d_hat, self.dob, v[:2], self.u_max, self.dob_max, slew, self.vmax)
        ucx0, sat_f, slw_f, spd_f = fl["dob_out"], fl["sat"], fl["slw"], fl["spd"]
        ucmd = np.r_[uc2, 0.0]
        self.u_prev = ucmd
        self.veh.publish_mode("acceleration")
        self.veh.accel_setpoint(ucmd, self.alt)
        self.ticks += 1
        el = self.el
        i = self.i
        self.w.writerow({
            "t": f"{t:.3f}", "tsim": f"{t + self.t0:.3f}", "phase": self.phase, "role": int(el.role[i]), "term": int(el.term[i]),
            "leader": int(el.leader[i]), "led_by": int(el.led_by[i]), "wp": int(el.wp[i]),
            "vux": f"{el.v_used[i, 0]:.3f}", "vuy": f"{el.v_used[i, 1]:.3f}",
            "crx": f"{el.cref[i, 0]:.3f}", "cry": f"{el.cref[i, 1]:.3f}", "cusum": f"{el.cusum[i]:.2f}",
            "n_distrust": int(getattr(self, "excl", np.zeros((1, 1)))[i].sum()) if hasattr(self, "excl") else 0,
            "att": int(self.adv is not None and self.adv.active), "E_wh": f"{self.E:.4f}",
            "P_w": f"{self.power.power_w():.1f}" if self.power else "",
            "px": f"{p[0]:.3f}", "py": f"{p[1]:.3f}", "pz": f"{p[2]:.3f}", "vx": f"{v[0]:.3f}", "vy": f"{v[1]:.3f}",
            "ux": f"{u[0]:.3f}", "uy": f"{u[1]:.3f}", "spx": f"{ucmd[0]:.3f}", "spy": f"{ucmd[1]:.3f}",
            "n_pkts": int(((t - self.t_rx[i]) <= 1.0).sum() - 1), "n_det": len(self.dets),
            "n_ghost": 0, "slack": f"{slack:.4f}", "solve_ms": f"{1e3 * (time.perf_counter() - t0):.2f}",
            "loop_dt": f"{loop_dt:.4f}", "leaving": int(self.leaving), "fault": self.fault or "",
            "dm": getattr(self, "dm_mask", 0), "dhx": f"{self.d_hat[0]:.3f}", "dhy": f"{self.d_hat[1]:.3f}",
            "cobx": f"{el.cobs[i, 0]:.3f}", "coby": f"{el.cobs[i, 1]:.3f}", "scl": f"{el.scale_used[i]:.3f}",
            "unx": f"{un[0]:.3f}", "uny": f"{un[1]:.3f}", "ucx": f"{ucx0[0]:.3f}", "ucy": f"{ucx0[1]:.3f}",
            "slw": slw_f, "sat": sat_f, "spd": spd_f, "dts": f"{dts:.4f}",
            "nqc": len(el.obs[i].get((int(el.term[i]), int(el.leader[i])), ())) if el.leader[i] >= 0 else 0,
            "shrej": int(el.shield_reject[i]),
            "crux": f"{el.cref_used[i, 0]:.3f}", "cruy": f"{el.cref_used[i, 1]:.3f}",
            "burst": int(self.adv is not None and getattr(self.adv, "burst", False))})
        if self.ticks % 40 == 0:
            self.f.flush()

    def close(self):
        el = self.el
        summ = {"id": self.i, "ticks": self.ticks, "tx": self.tx_count, "final_phase": self.phase, "E_end": self.E,
                "compromised": self.compromised, "fault": self.fault, "stats": el.stats,
                "graw_max": np.asarray(el.graw).max(0).tolist() if el.graw else None}
        (self.run_dir / f"le_agent_{self.i}_summary.json").write_text(json.dumps(summ, default=str))
        for f in (self.f, self.ev):
            try:
                f.close()
            except Exception:
                pass
        if self.power is not None:
            self.power.close()
        self.clk_stream.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--scheme", default="CADLE")
    ap.add_argument("--attack", default="L0")
    ap.add_argument("--phys", default="none")
    ap.add_argument("--attacker-id", type=int, default=-1)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    geo = json.loads(Path(a.geometry).read_text())
    rclpy.init()
    node = LEAgent(a.id, cfg, geo, Path(a.run_dir) / "agents", a.scheme, a.attack, a.attacker_id, a.phys)
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
