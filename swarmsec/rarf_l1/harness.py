"""Layer-1 harness: stub ROS 2 / PX4 world around the unmodified coupled-safety agents (see package docstring)."""
from __future__ import annotations

import copy
import json
import math
import random
import sys
import tomllib
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CTRL = ROOT / "controllers" / "swarm"
FIT = json.loads((Path(__file__).parent / "plant_fit.json").read_text())

H = 0.005                    # plant step [s]; decisions run on the agents' own 20 Hz plant-time grid
CONTACT = 0.77               # X500 rotor-tip envelope x 2 (RAL26 v2 audit)
P_HOVER_W = 250.0            # energy is never binding at E0 = 60 Wh over 100 s; constant power keeps the estimator live


# ---------------------------------------------------------------------------------------------------- stub ROS 2 / PX4 layer
class _Msg:
    def __init__(self, data=None, **k):
        self.data = data
        self.__dict__.update(k)


class _Pub:
    def __init__(self, world, topic):
        self.world, self.topic = world, topic

    def publish(self, m):
        self.world.on_publish(self.topic, m)


class _Timer:
    def cancel(self):
        pass


class _Node:
    """Stand-in for rclpy.node.Node: routes subscriptions / publishers through the World."""
    WORLD = None

    def __init__(self, name, *a, **k):
        self._name = name

    def create_subscription(self, typ, topic, cb, qos=None):
        _Node.WORLD.subscribe(topic, cb)

    def create_publisher(self, typ, topic, qos=None):
        return _Pub(_Node.WORLD, topic)

    def create_timer(self, period, cb):
        return _Timer()

    def destroy_timer(self, t):
        pass

    def destroy_node(self):
        pass

    def get_logger(self):
        return types.SimpleNamespace(info=lambda *a: None, warn=lambda *a: None, error=lambda *a: None)


class _Clock:
    """PlantClock stand-in: plant time is the world time exactly (RTF 1, no extrapolation, no misses)."""
    def __init__(self, world):
        self.world, self.rtf, self.n = world, 1.0, 1

    def ready(self):
        return True

    def now(self):
        return self.world.t


class FakeVehicle:
    """PX4Vehicle stand-in.  State comes from the World plant; setpoints go to it."""
    def __init__(self, node, uav_id, spawn_enu):
        self.uav_id = uav_id
        self.world = _Node.WORLD
        self.spawn = np.array([float(spawn_enu[0]), float(spawn_enu[1]), 0.0])
        self.xy_valid = True
        self.t_rx = 0.0
        self.last_arm = self.last_mode = self.last_land = -1e9
        self.t_px4 = 0.0

    def has_state(self):
        return True

    def pos_enu(self):
        self.t_rx = self.world.t
        return self.world.p_hat(self.uav_id)

    def vel_enu(self):
        return self.world.v_hat(self.uav_id)

    def acc_enu(self):
        return self.world.a[self.uav_id].copy()

    def armed(self):
        return not self.world.landed[self.uav_id]

    def offboard(self):
        return True

    def nav_state(self):
        return 14

    def publish_mode(self, kind):
        pass

    def position_setpoint_enu(self, p_enu):
        self.world.cmd_pos(self.uav_id, np.asarray(p_enu, float)[:2])

    def hover_setpoint(self, alt_up):
        self.world.cmd_pos(self.uav_id, None)

    def velocity_setpoint(self, v_enu, alt_up):
        raise NotImplementedError("Layer 1 models the acceleration interface only (HF1 selection)")

    def accel_setpoint(self, a_enu, alt_up):
        self.world.cmd_acc(self.uav_id, np.asarray(a_enu, float)[:2])

    def request_offboard(self):
        pass

    def request_arm(self):
        pass

    def request_land(self):
        self.world.cmd_land(self.uav_id)

    def bootstrap(self, now, alt, retry=2.0):
        return True


def _install_stubs():
    if getattr(_install_stubs, "done", False):
        return
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *a, **k: None
    rclpy.ok = lambda: True
    rclpy.shutdown = lambda: None
    node = types.ModuleType("rclpy.node")
    node.Node = _Node
    qos = types.ModuleType("rclpy.qos")
    for nm in ("DurabilityPolicy", "HistoryPolicy", "ReliabilityPolicy"):
        setattr(qos, nm, types.SimpleNamespace(TRANSIENT_LOCAL=0, VOLATILE=0, KEEP_LAST=0, BEST_EFFORT=0, RELIABLE=0))
    qos.QoSProfile = lambda *a, **k: None
    rclpy.node, rclpy.qos = node, qos
    std = types.ModuleType("std_msgs")
    stdm = types.ModuleType("std_msgs.msg")
    stdm.Float64MultiArray = stdm.String = _Msg
    std.msg = stdm
    gz = types.ModuleType("gz_stream")

    class GzStream:
        def __init__(self, *a, **k):
            pass

        def close(self):
            pass
    gz.GzStream = GzStream
    gz.model_poses = lambda m: []
    px = types.ModuleType("px4_iface")
    px.PX4Vehicle = FakeVehicle
    px.time = types.SimpleNamespace()
    sys.modules.update({"rclpy": rclpy, "rclpy.node": node, "rclpy.qos": qos, "std_msgs": std, "std_msgs.msg": stdm,
                        "gz_stream": gz, "px4_iface": px})
    for p in (str(ROOT), str(CTRL)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import energy_supervisor as es

    class PowerMonitor:
        def __init__(self, uid, inflation=1.0):
            self.inflation = inflation

        def power_w(self):
            return self.inflation * P_HOVER_W

        def close(self):
            pass
    es.PowerMonitor = PowerMonitor
    _install_stubs.done = True


# ---------------------------------------------------------------------------------------------------- world
class World:
    def __init__(self, n, alt, spawn, rng, net, req, wind=None):
        A = FIT["actuator"]
        self.wn, self.zeta = float(A["wn"]), float(A["zeta"])
        self.n, self.alt, self.rng = n, alt, rng
        self.t = 0.0
        self.p = np.array([[s[0], s[1]] for s in spawn], float)
        self.v = np.zeros((n, 2))
        self.a = np.zeros((n, 3))
        self.ad = np.zeros((n, 2))
        self.u = np.zeros((n, 2))
        self.pos_hold = [None] * n            # None = hold the current position (PX4 position mode)
        self.mode = ["pos"] * n
        self.hold_xy = self.p.copy()
        self.z = np.full(n, alt)
        self.landing = np.zeros(n, bool)
        self.landed = np.zeros(n, bool)
        self.crashed = np.zeros(n, bool)
        self.contact_events = []
        self.subs = {}
        self.net, self.req = net, req
        self.gcs_phase = {}
        self.ep = np.zeros((n, 2))
        self.ev = np.zeros((n, 2))
        self.sd_p, self.tau_p = float(FIT["ego_p"]["sd"]), max(float(FIT["ego_p"]["tau_s"]), 0.05)
        self.sd_v, self.tau_v = float(FIT["ego_v"]["sd"]), max(float(FIT["ego_v"]["tau_s"]), 0.05)
        self.wind = wind                       # None or dict(sigma [m/s^2], tau [s]): OU disturbance acceleration per vehicle
        self.wd = np.zeros((n, 2))

    # -- pub/sub
    def subscribe(self, topic, cb):
        self.subs.setdefault(topic, []).append(cb)

    def deliver(self, topic, payload):
        for cb in self.subs.get(topic, ()):
            cb(_Msg(data=list(payload)))

    def on_publish(self, topic, m):
        if topic == "/ral/tx":
            d = list(m.data)
            self.net.send(self.t, int(d[0]), d, self.true3())
        elif topic == "/ral/req_tx":
            d = list(m.data)
            self.req.send(self.t, int(d[0]), d, self.true3())
        elif topic.startswith("/ral/status/"):
            d = json.loads(m.data)
            self.gcs_phase[int(d["id"])] = d["phase"]

    # -- vehicle interface
    def true3(self):
        return np.column_stack([self.p, self.z])

    def p_hat(self, i):
        return np.array([self.p[i, 0] + self.ep[i, 0], self.p[i, 1] + self.ep[i, 1], self.z[i]])

    def v_hat(self, i):
        return np.array([self.v[i, 0] + self.ev[i, 0], self.v[i, 1] + self.ev[i, 1], 0.0])

    def cmd_pos(self, i, xy):
        if self.mode[i] != "pos":
            self.hold_xy[i] = self.p[i]
        self.mode[i] = "pos"
        if xy is not None:
            self.hold_xy[i] = xy

    def cmd_acc(self, i, u):
        self.mode[i] = "acc"
        self.u[i] = u

    def cmd_land(self, i):
        if not self.landing[i]:
            self.landing[i] = True
            self.hold_xy[i] = self.p[i]
            self.mode[i] = "pos"

    # -- plant
    def step(self):
        h = self.h
        n = self.n
        ucmd = self.u.copy()
        for i in range(n):
            if self.mode[i] == "pos":            # PX4 position hold (velocity-damped PD; P-gain of MPC_XY_P ~ 0.95, accel capped)
                e = self.hold_xy[i] - self.p[i]
                ucmd[i] = 0.95 * 1.8 * e - 1.8 * self.v[i]
                nn = np.linalg.norm(ucmd[i])
                if nn > 3.0:
                    ucmd[i] *= 3.0 / nn
        ucmd[self.crashed] = 0.0
        acc = self.a[:, :2]
        add = self.wn ** 2 * (ucmd - acc) - 2 * self.zeta * self.wn * self.ad
        self.ad += h * add
        acc = acc + h * self.ad
        if self.wind is not None:
            s, tau = float(self.wind["sigma"]), float(self.wind["tau"])
            self.wd += -h / tau * self.wd + s * math.sqrt(2 * h / tau) * self.rng.normal(size=(n, 2))
        a_tot = acc + self.wd
        a_tot[self.crashed] = 0.0
        self.v[self.crashed] = 0.0
        self.v += h * a_tot
        self.p += h * self.v
        self.a[:, :2] = a_tot
        for i in range(n):
            if self.landing[i] and not self.landed[i]:
                self.z[i] = max(0.0, self.z[i] - 1.0 * h)
                if self.z[i] <= 0.05:
                    self.landed[i] = True
        self.z[self.crashed] = np.maximum(0.0, self.z[self.crashed] - 3.0 * h)
        self.landed |= self.crashed & (self.z <= 0.05)
        # OU ego-estimate errors
        for e, sd, tau in ((self.ep, self.sd_p, self.tau_p), (self.ev, self.sd_v, self.tau_v)):
            e += -h / tau * e + sd * math.sqrt(2 * h / tau) * self.rng.normal(size=(n, 2))
        # contact: both vehicles stop and fall (PX4 / Gazebo would flip them; recorded as a crash)
        P3 = self.true3()
        D = np.linalg.norm(P3[:, None, :] - P3[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        airborne = (self.z > 0.3) & ~self.crashed
        hit = np.argwhere((D < CONTACT) & airborne[:, None] & airborne[None, :])
        for i, j in hit:
            if i < j:
                self.contact_events.append((self.t, int(i), int(j), float(D[i, j])))
                self.crashed[[i, j]] = True
        self.t += h

    h = H


# ---------------------------------------------------------------------------------------------------- episode
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def resolve(spec_base: str, common: dict, ep: dict) -> dict:
    """Config resolution exactly as scripts/swarm/ral26_batch.py (_run_episode) does it."""
    sys.path.insert(0, str(ROOT / "scripts" / "swarm"))
    import ral26_batch as B
    base = tomllib.loads((ROOT / spec_base).read_text())
    deep_update(base, common or {})
    cfg = deep_update(copy.deepcopy(base), ep.get("overrides", {}))
    n = int(cfg["swarm"]["num_uavs"])
    draw = B.randomize_scenario(cfg)
    if cfg.get("ral26", {}).get("attacker_id", -1) == "central" or (
            cfg["attack"]["id"] != "A0" and int(cfg.get("ral26", {}).get("attacker_id", -1)) < 0):
        cfg["ral26"]["attacker_id"] = B.central(B.lattice(n, float(cfg["formation"]["spacing_m"])))
    cfg.setdefault("experiment", {})["run_id"] = ep["name"]
    return cfg, draw


def geometry(cfg) -> dict:
    """experiment_manager.write_geometry."""
    from experiment_manager import generate_lattice
    n = int(cfg["swarm"]["num_uavs"])
    desired, spawn = generate_lattice(n, float(cfg["formation"]["spacing_m"]), float(cfg["swarm"]["spawn_scale"]))
    cs = cfg.get("cs", {})
    for k, (dx, dy) in (cs.get("spawn_jitter") or {}).items():
        spawn[int(k)][0] += float(dx)
        spawn[int(k)][1] += float(dy)
    ss = cs.get("slot_shift")
    if ss:
        for arr in (desired, spawn):
            arr[int(ss["id"])][0] += float(ss["dx"])
            arr[int(ss["id"])][1] += float(ss["dy"])
    return {"desired_offsets": {str(i): desired[i] for i in range(n)}, "spawn_positions": {str(i): spawn[i] for i in range(n)}}


def run_episode(cfg: dict, rd: Path, keep_agent_logs: bool = False, l1: dict | None = None) -> dict:
    """Fly one episode in Layer 1.  `l1` = harness-only options: wind {sigma, tau}, extra_attackers (list of ids, B-N1),
    attacker_hook (callable, B-A1 adaptive attacker; applied to the designated agents' attack_terms)."""
    _install_stubs()
    l1 = l1 or {}
    import uav_agent_node as base
    import cs_agent_node as csn
    import cs_agent_node_v2 as v2
    import px4_iface
    import cs_agent_node_v4 as v4
    from swarmsec.fanet import Fanet, FanetConfig
    from swarmsec.le_v2.fanet_v2 import KeyedFanet

    rd.mkdir(parents=True, exist_ok=True)
    (rd / "agents").mkdir(exist_ok=True)
    (rd / "eval").mkdir(exist_ok=True)
    n = int(cfg["swarm"]["num_uavs"])
    R = cfg["ral26"]
    seed = int(cfg["experiment"]["seed"])
    geo = geometry(cfg)
    (rd / "resolved_config.json").write_text(json.dumps(cfg, indent=1))
    (rd / "geometry.json").write_text(json.dumps(geo))
    fc = FanetConfig.from_dict(dict(R.get("fanet", {})))
    key = R.get("res_key")
    net = KeyedFanet(n, fc, np.random.default_rng(seed + 777), key) if (R.get("sim_time_net") and key is not None) \
        else Fanet(n, fc, np.random.default_rng(seed + 777))
    req = Fanet(n, FanetConfig(**{**fc.__dict__, "packet_bytes": 16, "bucket_bytes_per_s": 1e9}), np.random.default_rng(seed + 778))
    alt = float(cfg["swarm"].get("altitude_m", 5.0))
    spawn = [geo["spawn_positions"][str(i)] for i in range(n)]
    world = World(n, alt, spawn, np.random.default_rng(seed + 901), net, req, wind=l1.get("wind"))
    _Node.WORLD = world
    clock = _Clock(world)
    sh = types.SimpleNamespace(monotonic=clock.now, perf_counter=__import__("time").perf_counter,
                               time=__import__("time").time, sleep=lambda s: None)
    base.time = csn.time = px4_iface.time = sh
    v2._REAL_MONO = clock.now
    attacker = int(R.get("attacker_id", -1))
    extra_att = [int(x) for x in l1.get("extra_attackers", [])]
    variant = cfg["defense"].get("id", "D24R")
    attack = cfg["attack"].get("id", "A0")
    agents = []
    for i in range(n):
        att_i = i if (i in extra_att and attack != "A0") else attacker
        ag = v4.CsAgentV4(i, cfg, geo, rd / "agents", variant, attack, att_i, clock=clock)
        if l1.get("attacker_hook") is not None and (i == attacker or i in extra_att):
            l1["attacker_hook"](ag)
        agents.append(ag)
    if not keep_agent_logs:
        for ag in agents:                                # decision logs are not needed for the outcome metrics
            ag.f.close()
            ag.f = open("/dev/null", "w")
            ag.w = types.SimpleNamespace(writerow=lambda row: None)
    duration = float(cfg["experiment"]["duration_s"])
    t0 = None
    t_land = None
    t_land_lim = None
    truth = []
    next_truth = 0.0
    truth_dt = 1.0 / float(R.get("truth_hz", 50.0))
    T_max = duration + 60.0
    while world.t < T_max:
        # network delivery (as fanet_bridge._deliver, 5 ms timer)
        for _, i, _j, payload in net.due(world.t):
            world.deliver(f"/ral/rx/uav_{i}", payload)
        for _, i, _j, payload in req.due(world.t):
            world.deliver(f"/ral/req/uav_{i}", payload)
        for ag in agents:
            ag._sched()
        # GCS
        ph = [world.gcs_phase.get(i) for i in range(n)]
        if t0 is None and all(p == "HOLD" for p in ph):
            t0 = world.t
            for ag in agents:
                ag._on_gcs(_Msg(data=json.dumps({"cmd": "start", "t0": t0, "t0_sim": t0})))
        if t0 is not None and t_land is None and (world.t - t0 >= duration or all(p in ("DONE", "LAND") for p in ph)):
            t_land = world.t
            for ag in agents:
                ag._on_gcs(_Msg(data=json.dumps({"cmd": "land", "t0": t0})))
        if t_land is not None and (all(p == "DONE" for p in ph) or world.t - t_land > 40.0):
            break
        if t0 is None and world.t > 120.0:
            break
        if world.t >= next_truth - 1e-9:
            P3 = world.true3()
            tilt = np.degrees(np.arctan2(np.linalg.norm(world.a[:, :2], axis=1), 9.81))
            tilt[world.crashed] = 90.0
            truth.append([world.t] + [x for i in range(n) for x in (P3[i, 0], P3[i, 1], P3[i, 2], tilt[i])])
            next_truth += truth_dt
        world.step()
    for ag in agents:
        ag.close()
    if keep_agent_logs:
        import gzip
        import shutil
        for i in range(n):
            f = rd / "agents" / f"agent_{i}.csv"
            if f.exists():
                with f.open("rb") as a, gzip.open(str(f) + ".gz", "wb") as b:
                    shutil.copyfileobj(a, b)
                f.unlink()
    cols = ["t_sim"] + [f"{c}{i}" for i in range(n) for c in ("x", "y", "z", "tilt")]
    with (rd / "eval" / "truth.csv").open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in truth:
            fh.write(",".join(f"{x:.3f}" for x in r) + "\n")
    summ = {"layer": "L1", "n": n, "attacker": attacker, "t_mission_start_sim": t0,
            "t_mission_end_sim": t_land, "rtf_mean": 1.0, "rtf_min": 1.0, "contact_events": world.contact_events,
            "extra_attackers": extra_att, "plant_fit": {k: FIT[k] for k in ("actuator", "ego_p", "ego_v")}}
    (rd / "eval" / "summary.json").write_text(json.dumps(summ, default=float))
    (rd / "episode_status.json").write_text(json.dumps({"gcs_code": 0 if t0 is not None else 2}))
    return summ
