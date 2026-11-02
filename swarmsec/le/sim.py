"""Layer-1 episode runner for the leader-election study.

Loop (physics dt = 0.05 s; protocol, radio and basic sensing at 10 Hz):
  truth -> own estimates (fleet nav model) -> packets (honest compose, adversary edits own row)
  -> PacketChannel (per-link loss/range/delay/partition) -> basic sensing scan + claim verification
  -> election step -> mission layer (leader: plan progress + command; followers: plan-consistency CUSUM,
  bounded authority) -> per-receiver nominal controller -> vectorised HOCBF filter -> physics + energy.
Ground truth is used only by the emulators (physics, radio, sensor) and by the metrics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import fastqp
from .adversary import Adversary, INSIDER, keepout_distance
from .channel import NET_PRESETS, NetConfig, PacketChannel
from .control import nominal
from .mission_layer import node_step
from . import intervene
from .protocol import Election, ElectionParams, Fld, ROLE_L, SCHEMES, frozen_params
from .sensing import BasicSensor, SensorConfig


@dataclass
class LEConfig:
    n: int = 16
    mission: str = "delivery"
    controller: str = "LF"
    scheme: str = "CADLE"
    attack: str = "L0"
    phys: str | None = None            # A2 | A4 | A8 | A9 | A14 (layered on the election attack)
    att_target: str = "random"         # random | A | B | C | leader (insider: whoever leads at attack start)
    attack_start_s: float = 25.0
    epsilon: float = 2.0
    faults: tuple = ()                 # F_CRASH F_BATT F_GNSS F_PART F_RADIO (hit the leader at fault_t)
    fault_t: float = 40.0
    net: str = "nominal"
    spacing: float = 4.0
    dt: float = 0.05
    proto_every: int = 2
    d_min: float = 1.5
    d_contact: float = 0.8
    d_safe: float = 2.5
    u_max: float = 4.0
    k_rows: int = 6
    radius: float = 8.0
    wind_sigma: float = 0.15
    wind_bound: float = 0.5
    cruise: float = 1.5
    t_boot: float = 6.0                # leaderless boot phase (election converges during it)
    duration_s: float | None = None
    battery_scale: float = 1.0
    record: bool = False
    params: dict = field(default_factory=dict)
    tol_scale: float = 1.0             # E6: scales tol_v, tol_c, tol_xt (the white-box attacker sees the scaled values)
    res: dict | None = None            # resilience campaign: scenario/intervention block (see swarmsec.le.resil); None = legacy


def _phys_attack(name, att, offsets, eps, rng, radius):
    if name is None:
        return None
    from ..attacks import build_attack
    from ..attacks.base import AttackOutput
    if name.upper() == "A14":
        push = _always(build_attack("A8", attacker=att, epsilon=eps, neighbor_radius=radius, rng=rng, offsets=offsets))
        spoof = build_attack("A4", attacker=att, epsilon=1.25, neighbor_radius=radius, rng=rng, offsets=offsets)

        class Hybrid:
            def __init__(self):
                self.offsets = offsets

            def act(self, t, p, v, nom, dt=0.05, global_lambda2=0.0):
                a = push.act(t, p, v, nom, dt=dt, global_lambda2=global_lambda2)
                b = spoof.act(t, p, v, nom)
                return AttackOutput(delta_u=a.delta_u, report_pos_bias=b.report_pos_bias,
                                    bypass_self_filter=True, triggered=a.triggered or b.triggered)
        return Hybrid()
    kw = dict(attacker=att, epsilon=eps, neighbor_radius=radius, rng=rng, offsets=offsets)
    return _always(build_attack(name, **kw))


def _always(a):
    """The selective T7 trigger of A8/A9 is tuned to another study's formation spacing and never fires at this
    study's spacing (DEVIATIONS_LE D8); here they compress continuously (A9 still ramps in)."""
    from ..attacks.triggers import make_trigger
    if getattr(a, "trigger_name", None) == "T7":
        a.trigger_name, a.trigger = "ALWAYS", make_trigger("ALWAYS")
    return a


def _true_scores(fleet, plan, k_true, c_true, alive, p_true, need_fn, params, need_tot):
    """Oracle capability of every node from ground truth (evaluation only)."""
    need = need_fn(k_true, c_true)
    e = np.clip((fleet.E - need) / (params.e_headroom * need_tot), 0, 1)
    nav = 1.0 - np.clip(np.linalg.norm(fleet.nav_error()[:, :2], axis=1) / params.nav_scale, 0, 1)
    d = np.linalg.norm(p_true[:, None, :2] - p_true[None, :, :2], axis=2)
    cov = ((d <= 0.7 * fleet.radio_range()[:, None]) & alive[None, :]).sum(1) / max(alive.sum(), 1)
    w = params.w
    s = w[0] * e + w[1] * nav + w[2] * cov
    elig = (fleet.E >= need) & (np.linalg.norm(fleet.nav_error()[:, :2], axis=1) < 2.0) & alive
    return s, elig


def run(cfg: LEConfig, seed: int = 0) -> dict:
    wall0 = time.time()
    rng = np.random.default_rng(seed)
    n = cfg.n
    from .setup import episode_setup
    SU = episode_setup(n, cfg.mission, seed, spacing=cfg.spacing, cruise=cfg.cruise, t_boot=cfg.t_boot, dt=cfg.dt,
                       battery_scale=cfg.battery_scale)
    RS = cfg.res
    keyed = None
    if RS is not None:
        from .resil import KeyedStreams, build_setup
        keyed = KeyedStreams(int(RS["key"]))
        SU = build_setup(cfg, RS, seed)
    offsets, plan, fleet, fcfg = SU.offsets, SU.plan, SU.fleet, SU.fcfg
    att_start = float((RS or {}).get("att_start", cfg.attack_start_s))
    T = cfg.duration_s or (plan.duration_s + cfg.t_boot)
    need_fn = SU.need_fn
    need_tot = need_fn(0, np.zeros(2))
    prm = frozen_params(cfg.controller, **({**cfg.params, "local_membership": True} if cfg.res is not None else cfg.params))
    if cfg.tol_scale != 1.0:
        prm.tol_v *= cfg.tol_scale
        prm.tol_c *= cfg.tol_scale
        prm.tol_xt *= cfg.tol_scale
    designated = SU.designated            # best class-A node by initial charge (public pre-flight choice)
    el = Election(n, cfg.scheme, plan, fleet, offsets, np.random.default_rng(seed + 13), prm,
                  designated=designated, f_need=need_fn)
    el.t_boot = cfg.t_boot
    S = SCHEMES[cfg.scheme]
    if S.get('iface2'):
        prm.delta_max = prm.v2_delta
    gate = bool(S.get("gate"))
    shield = bool(S.get("shield"))
    v2 = bool(S.get("iface2"))
    ctrl_kind = "VS" if (S["kind"] == "local" or v2) else cfg.controller      # LOCAL: every UAV tracks its OWN public-plan reference
    shield_r = prm.shield_tight_r if S.get("shield_tight") else prm.tol_c
    impeach = bool(S.get("impeach"))
    use_sense = bool(S.get("sense"))
    sbcbf = bool(S.get("sbcbf")) and use_sense
    t_scan = 0.0
    excl = np.zeros((n, n), bool)
    ncfg = NetConfig(**NET_PRESETS[cfg.net])
    if RS is not None and RS.get("outages"):
        ncfg.outages = [dict(o) for o in RS["outages"]]
    chan = PacketChannel(n, Fld.width(n), np.random.default_rng(seed + 14), ncfg)
    sensor = BasicSensor(n, np.random.default_rng(seed + 15), SensorConfig(**((RS or {}).get('sensor') or {})), nav_sigma=fleet.nav_sigma)

    # attacker choice
    classes = np.array(fleet.classes)
    arng = np.random.default_rng(seed + 16)
    if RS is not None and RS.get("attacker_id") is not None:
        att = int(RS["attacker_id"])                         # exogenous: identical for every defence at this scenario
    elif cfg.att_target in ("A", "B", "C"):
        pool = np.nonzero(classes == cfg.att_target)[0]
        att = int(arng.choice(pool)) if len(pool) else int(arng.integers(n))
    else:
        att = int(arng.integers(n))
    has_att = cfg.attack != "L0" or cfg.phys is not None
    if not has_att:
        att = -1

    # state
    p = offsets * 1.0
    p = p + np.random.default_rng(seed + 17).normal(scale=0.2, size=p.shape)
    p[:, 2] = -5.0
    v = np.zeros((n, 3))
    wind = np.zeros((n, 3))
    wrng = np.random.default_rng(seed + 18)
    wind_tr = keyed.stream('wind').normal(size=(int(T / cfg.dt) + 2, n, 3)) if keyed is not None else None
    adv = None
    if has_att:
        adv = Adversary(cfg.attack, att, plan, fleet, offsets, att_start,
                        np.random.default_rng(seed + 19),
                        phys=_phys_attack(cfg.phys, att, offsets, cfg.epsilon, np.random.default_rng(seed + 20), cfg.radius),
                        params=prm)
    insider = cfg.att_target == "leader" or cfg.attack in INSIDER
    insider_fixed = False
    if RS is not None and RS.get("attacker_id") is not None:
        insider_fixed = True                                 # never re-target to whoever leads: the identity is the manifest's
    if RS is not None and adv is not None:
        adv.post = RS.get("post_capture", "LSTEALTH")
        adv.end = float(RS.get("att_end", np.inf))
        adv.burst_speed = float(RS.get("burst_speed", adv.burst_speed))
        adv.stall_k = int(RS.get("stall_k", adv.stall_k))
        adv.auth = dict(RS.get("auth") or {})
    inst = (RS or {}).get("install")                          # {"t": s, "hold_until": s}: intervention (NOT natural capture)
    inst_done = False

    steps = int(T / cfg.dt)
    k_true = 0
    t_done = np.nan
    cover = None
    if plan.area is not None:
        x0, x1, y0, y1 = plan.area
        gx, gy = np.meshgrid(np.arange(x0, x1, 2.0) + 1.0, np.arange(y0, y1, 2.0) + 1.0)
        cells = np.stack([gx.ravel(), gy.ravel()], 1)
        cover = np.zeros(len(cells), bool)
    # metric accumulators (protocol ticks)
    M = dict(ticks=0, led=0, led_honest=0, att_major=0, att_any=0, split=0, dual=0, changes=0, regret=[],
             inelig=0, keepout_s=0.0, xt=[], post_ticks=0, rmse=[], led_post=0, fd_num=0, fd_den=0,
             att_d_num=0, att_d_den=0)
    min_hh = np.inf
    min_ha = np.inf
    contacts_hh = contacts_ha = 0
    major_prev = -2
    fault_hit = -1
    failover = np.nan
    leader_energy_loss = 0
    hist_major = []
    pend_leave = np.zeros(n, bool)
    navrep = -np.ones((n, n))
    heard = np.zeros((n, n))
    v_cmd_lead = np.zeros((n, 3))
    s_lead = np.ones(n)
    traj = [] if cfg.record else None
    rec_series = dict(t=[], major=[], att_lead=[], n_alive=[], xt=[]) if cfg.record else None
    applied_last = np.zeros((n, 3))
    ser = {k_: [] for k_ in ('t', 'p', 'v', 'led', 'role', 'wp', 'cobs', 'vused', 'scale', 'alive', 'att_act', 'u', 'unom',
                             'deliv', 'exp', 'E')} if RS is not None else None
    fault_applied = False

    for k in range(steps):
        t = k * cfg.dt
        alive = fleet.alive.copy()
        honest = alive.copy()
        if att >= 0:
            honest[att] = False
        if keyed is not None:
            fleet.rng = keyed.tick('nav', k)
            chan.rng = keyed.tick('net', k)
        own_p, own_v = fleet.estimate(p, v)

        # ---------------------------------------------------------------- faults (hit the current leader)
        if cfg.faults and not fault_applied and t >= cfg.fault_t:
            fault_applied = True
            maj = hist_major[-1] if hist_major else -1
            tgt = maj if maj >= 0 and maj != att else int(np.argmax(np.where(honest, fleet.E, -1)))
            fault_hit = tgt
            for f in cfg.faults:
                if f == "F_CRASH":
                    fleet.crash(tgt)
                elif f == "F_BATT":
                    fleet.battery_anomaly(tgt, 6.0)
                elif f == "F_GNSS":
                    fleet.start_gnss_fault(tgt, rate=0.5, cap=10.0)
                elif f == "F_RADIO":
                    fleet.radio_degrade(tgt, 0.25)
                elif f == "F_PART":
                    side = np.argsort(p[:, 1])[: n // 2]
                    chan.cfg.outages.append({"t0": t, "dur": 15.0, "cut": list(map(int, side))})
            alive = fleet.alive.copy()
            honest = alive.copy()
            if att >= 0:
                honest[att] = False

        # ---------------------------------------------------------------- protocol tick
        if k % cfg.proto_every == 0:
            if insider and has_att and not insider_fixed and t >= att_start - 0.5:
                # insider threat: the CURRENT leader is the compromised one (re-target the adversary)
                maj = hist_major[-1] if hist_major else -1
                if maj >= 0:
                    att = maj
                    adv.id = att
                    if adv.phys is not None:
                        adv.phys = _phys_attack(cfg.phys, att, offsets, cfg.epsilon, np.random.default_rng(seed + 20), cfg.radius)
                insider_fixed = True
                honest = alive.copy()
                honest[att] = False
            t_rx = chan.t_rx
            heard = ((t - t_rx) <= prm.t_alive).astype(float)
            np.fill_diagonal(heard, 0.0)
            if use_sense:
                navrep = np.where(sensor.seen, 1.0 - np.clip(sensor.resid / prm.nav_scale, 0, 1), -1.0)
                np.fill_diagonal(navrep, -1.0)
            nav_claim = 1.0 - np.clip(fleet.nav_sigma * 1.5 / prm.nav_scale, 0, 1)      # self-known (no fault awareness)
            Pk = el.compose(own_p, own_v, fleet.E.copy(), nav_claim, heard, navrep, t)
            pend_leave = alive & (fleet.E <= fcfg.e_land_wh + prm.leave_margin_wh)
            Pk[pend_leave, Fld.LEAVE] = 1.0
            Pk[pend_leave, Fld.ECL] = 0.0
            if adv is not None:
                Pk = adv.edit_packet(Pk, t, el, chan.view, p)
            mask = adv.send_mask(t, el, p) if adv is not None else None
            chan.step(t, Pk, p, fleet.radio_range(), alive, mask)
            V = chan.view
            if use_sense:
                if keyed is not None:
                    sensor.rng = keyed.tick('sensor', k)
                dets = sensor.scan(p, alive)
                fresh_v = (t - chan.t_rx) <= 0.35
                cons, incons, _ = sensor.verify(dets, own_p, V[:, :, Fld.PX:Fld.PZ + 1], fresh_v & alive[None, :], alive,
                                                distrust=el.own_distrust())
                t_scan = t
                el.add_evidence(cons, incons)
            if adv is not None:
                adv.pre_step(t, el, V)
            fresh = el.step(t, V, chan.t_rx, None, own_p, own_v, fleet.E, fleet.t_air, alive,
                            sense_nav=sensor.resid if use_sense else None,
                            sensed=sensor.seen if use_sense else None, adversary=adv)
            if inst is not None and att >= 0:
                if not inst_done and t >= inst["t"]:
                    intervene.install(el, np.nonzero(alive)[0], att, t)
                    inst_done = True
                if inst_done and t < inst.get("hold_until", inst["t"]):          # hold only when a hold_until is given
                    intervene.hold(el, np.nonzero(alive)[0], att, t, t_rx=chan.t_rx, silent_s=inst.get('release_silent_s'))

            # trust-aware control (CADLE `sense`): a receiver stops using the CLAIMS of nodes it distrusts
            Dm = el.distrusted(V, fresh) if (impeach or use_sense) else np.zeros((n, n), bool)
            excl = Dm if (use_sense and S.get("tctl")) else np.zeros((n, n), bool)
            if sbcbf:
                sensor.ghosts_excluding(excl)             # an excluded claim never hides its body (D6)
            # ---------------- mission layer (shared with the PX4 node: swarmsec.le.mission_layer)
            for i in range(n):
                if not alive[i]:
                    continue
                use_i = fresh[i] & alive & ~excl[i]
                v_fly, s_lead[i] = node_step(el, i, V, use_i, offsets, plan, s_lead[i], t, cfg.t_boot, gate, impeach,
                                             chan.t_tx[i], own_p[i], adv=adv, shield=shield, local_wp=bool(S.get('local_wp')), leader_la=bool(S.get('iface2')))
                if el.led_by[i] == i:
                    v_cmd_lead[i] = v_fly

            # ---------------- election metrics (truth)
            if impeach or use_sense:
                hh = honest[:, None] & honest[None, :]
                np.fill_diagonal(hh, False)
                M["fd_num"] += int((Dm & hh).sum())
                M["fd_den"] += int(hh.sum())
                if att >= 0 and t >= att_start:
                    M["att_d_num"] += int(Dm[honest, att].sum())
                    M["att_d_den"] += int(honest.sum())
            led = el.led_by.copy()
            hon_idx = np.nonzero(honest)[0]
            vals = led[hon_idx]
            vals_ok = vals[vals >= 0]
            major = -1
            if len(vals_ok):
                ids, cnt = np.unique(vals_ok, return_counts=True)
                if cnt.max() >= len(hon_idx) // 2 + 1:
                    major = int(ids[np.argmax(cnt)])
                if len(ids) > 1:
                    M["split"] += 1
            L_roles = np.nonzero((el.role == ROLE_L) & alive)[0]
            if S["kind"] == "quorum" and len(L_roles) > 1:
                terms = el.term[L_roles]
                M["dual"] += int(len(terms) != len(np.unique(terms)))
            M["ticks"] += 1
            post = t >= att_start
            if major >= 0:
                M["led"] += 1
                if major != att:
                    M["led_honest"] += 1
            if post:
                M["post_ticks"] += 1
                if major >= 0:
                    M["led_post"] += 1
                if att >= 0 and major == att:
                    M["att_major"] += 1
                if att >= 0 and np.any(vals == att):
                    M["att_any"] += 1
            if major != major_prev and major_prev != -2:
                if major >= 0 and major_prev >= 0:
                    M["changes"] += 1
                elif major >= 0 > major_prev and len(hist_major) > int(cfg.t_boot * 10):
                    M["changes"] += 1
            major_prev = major
            hist_major.append(major)
            if major >= 0 and t >= cfg.t_boot:
                hc = p[honest].mean(0) if honest.any() else p.mean(0)
                s_true, e_true = _true_scores(fleet, plan, k_true, hc, alive, p, need_fn, prm, need_tot)
                cand = e_true & honest
                if cand.any():
                    M["regret"].append(float(np.max(s_true[cand]) - s_true[major]))
                M["inelig"] += int(not e_true[major])
            if fault_hit >= 0 and np.isnan(failover) and major >= 0 and major != fault_hit:
                failover = t - cfg.fault_t
            if rec_series is not None:
                rec_series["t"].append(t)
                rec_series["major"].append(major)
                rec_series["att_lead"].append(int(att >= 0 and major == att))
                rec_series["n_alive"].append(int(alive.sum()))

        # ---------------------------------------------------------------- control
        V = chan.view
        Pv = V[:, :, Fld.PX:Fld.PZ + 1].copy()
        Vv = V[:, :, Fld.VX:Fld.VZ + 1].copy()
        idx = np.arange(n)
        Pv[idx, idx] = own_p
        Vv[idx, idx] = own_v
        age = t - chan.t_rx
        fresh_c = (age <= 1.0) & alive[None, :] & ~excl
        # dead-reckon received states to now from their SEND time stamp (packets carry it; receiver-side)
        Pv = Pv + Vv * np.clip(t - chan.t_tx, 0, 1.2)[..., None]
        Pv[idx, idx] = own_p
        u_nom = nominal(ctrl_kind, own_p, own_v, Pv, Vv, fresh_c, alive, el.led_by, el.v_used, el.cref_used,
                        el.scale_used, offsets, cfg.spacing, radius=cfg.radius, gate=gate, shield=shield, shield_r=shield_r, v2=v2, v2_kp=prm.v2_kp, v2_kv=prm.v2_kv)
        # nav-aware barrier (all schemes): public class nav sigma of both UAVs widens the safety radius
        ds = cfg.d_safe + 1.5 * np.sqrt(fleet.nav_sigma[:, None] ** 2 + fleet.nav_sigma[None, :] ** 2)
        alive_c = alive
        if sbcbf:
            # distrusted claims leave the CBF; the body itself is still there as an unexplained detection
            Pv = np.where(excl[..., None], 1e6, Pv)
            # sensing-backed safety: unexplained detections (ghosts) are obstacles in the receiver's own frame,
            # moving with the formation (mean received velocity); barrier padded by 0.5 m for their uncertainty
            G = max([len(g) for g in sensor.ghost_rel] + [0])
            if G:
                Pg = np.full((n, G, 3), 1e6)
                Vg = np.zeros((n, G, 3))
                for i in range(n):
                    g = sensor.ghost_rel[i]
                    if len(g):
                        vf = Vv[i, fresh_c[i]].mean(0) if fresh_c[i].any() else own_v[i]
                        Pg[i, :len(g)] = own_p[i] + g + vf * (t - t_scan)
                        Vg[i, :len(g)] = vf
                Pv = np.concatenate([Pv, Pg], axis=1)
                Vv = np.concatenate([Vv, Vg], axis=1)
                ds = np.concatenate([ds, np.full((n, G), cfg.d_safe + 0.5)], axis=1)
                alive_c = np.concatenate([alive, np.ones(G, bool)])
        A, c, nbr = fastqp.hocbf_rows(own_p, own_v, Pv, Vv, ds, 2.0, 2.0, cfg.radius, cfg.k_rows,
                                      alive=alive_c)
        u, slack, margin = fastqp.solve(u_nom, A, c, cfg.u_max)
        if adv is not None:
            out = adv.physical(t, el, p, v, cfg.dt)
            if out is not None and out.triggered:
                base = u_nom[att] if out.bypass_self_filter else u[att]
                u[att] = base + out.delta_u
        u[~alive] = 0.0
        if ser is not None and k % 2 == 0:
            ser['t'].append(t); ser['p'].append(p.astype(np.float32)); ser['v'].append(v.astype(np.float32))
            ser['led'].append(el.led_by.astype(np.int16)); ser['role'].append(el.role.astype(np.int8))
            ser['wp'].append(el.wp.astype(np.int16)); ser['cobs'].append(np.nan_to_num(el.cobs, nan=0.0).astype(np.float32))
            ser['vused'].append(el.v_used[:, :2].astype(np.float32)); ser['scale'].append(el.scale_used.astype(np.float32))
            ser['alive'].append(alive.copy()); ser['att_act'].append(int(adv is not None and adv.active))
            ser['u'].append(u.astype(np.float32)); ser['unom'].append(u_nom.astype(np.float32))
            ser['deliv'].append(chan.last_delivered); ser['exp'].append(chan.last_expected); ser['E'].append(fleet.E.astype(np.float32))

        # ---------------------------------------------------------------- physics + energy
        kq = cfg.dt / 2.0
        wind += -kq * wind + cfg.wind_sigma * np.sqrt(2 * kq) * (wind_tr[k] if wind_tr is not None else wrng.normal(size=wind.shape))
        wn = np.linalg.norm(wind, axis=1, keepdims=True)
        w_eff = wind * np.minimum(1.0, cfg.wind_bound / np.maximum(wn, 1e-9))
        a_tot = u + w_eff
        an = np.linalg.norm(a_tot, axis=1, keepdims=True)
        a_tot = np.where(an > 12.0, a_tot * 12.0 / an, a_tot)
        v = np.where(alive[:, None], (1 - 0.2 * cfg.dt) * v + a_tot * cfg.dt, 0.0)
        p = p + v * cfg.dt
        lead_mask = np.zeros(n)
        lead_mask[el.role == ROLE_L] = 1.0
        newly = fleet.step(u, lead_mask, cfg.u_max)
        if newly.any():
            if hist_major and hist_major[-1] >= 0 and newly[hist_major[-1]]:
                leader_energy_loss += 1
            p[newly, 2] = 0.0

        # ---------------------------------------------------------------- truth metrics (every tick)
        al = fleet.alive
        hon = al.copy()
        if att >= 0:
            hon[att] = False
        if al.sum() >= 2:
            D = np.linalg.norm(p[:, None] - p[None], axis=2)
            D[~al, :] = np.inf
            D[:, ~al] = np.inf
            np.fill_diagonal(D, np.inf)
            if t >= cfg.t_boot:
                dh = D[np.ix_(hon, hon)]
                if dh.size:
                    m_ = float(dh.min())
                    min_hh = min(min_hh, m_)
                    contacts_hh += int(m_ < cfg.d_min)
                if att >= 0 and al[att]:
                    m_ = float(D[att, hon].min()) if hon.any() else np.inf
                    min_ha = min(min_ha, m_)
                    contacts_ha += int(m_ < cfg.d_min)
        if hon.any():
            hc = p[hon].mean(0)
            if k_true < len(plan.waypoints) and np.linalg.norm(plan.waypoints[k_true] - hc[:2]) < plan.tol + 1.0:
                k_true += 1
                if k_true == len(plan.waypoints) and np.isnan(t_done):
                    t_done = t
            if t >= cfg.t_boot:
                M["keepout_s"] += float(plan.in_keepout(p[hon]).sum()) * cfg.dt
                if k % 2 == 0:
                    M["ko_min"] = min(M.get("ko_min", np.inf), min(keepout_distance(plan, q) for q in p[hon]))
                if k % 10 == 0:
                    M["xt"].append(plan.cross_track(hc, k_true))
                    sc_ = float(np.median(s_lead[hon]))
                    e = (p[hon, :2] - hc[:2]) - sc_ * (offsets[hon, :2] - offsets[hon, :2].mean(0))
                    M["rmse"].append(float(np.sqrt((e ** 2).sum(1).mean())))
            if cover is not None:
                dd = np.min(np.linalg.norm(cells[:, None, :] - p[None, hon, :2], axis=2), axis=1) if k % 4 == 0 else None
                if dd is not None:
                    cover |= dd <= 3.0
        if traj is not None and k % 4 == 0:
            traj.append(p.copy())
        if not np.isnan(t_done) and t > t_done + 10.0 and not (RS is not None and RS.get("fixed_horizon")):
            break                                          # mission over; post-mission hover is not scored

    nK = len(plan.waypoints)
    post_t = max(1, M["post_ticks"])
    res = dict(
        n=n, mission=cfg.mission, controller=cfg.controller, scheme=cfg.scheme, attack=cfg.attack,
        phys=cfg.phys or "none", att_target=cfg.att_target, faults="+".join(cfg.faults) or "none", net=cfg.net,
        tol_scale=cfg.tol_scale,
        seed=seed, attacker=att, attacker_class=(fleet.classes[att] if att >= 0 else "-"),
        avail=M["led"] / max(1, M["ticks"]), avail_honest=M["led_honest"] / max(1, M["ticks"]),
        avail_post=M["led_post"] / post_t if M["post_ticks"] else np.nan,
        att_tenure=M["att_major"] / post_t if att >= 0 else 0.0,
        att_any=M["att_any"] / post_t if att >= 0 else 0.0,
        split_frac=M["split"] / max(1, M["ticks"]), dual_leader_ticks=M["dual"],
        leader_changes=M["changes"], regret_mean=float(np.mean(M["regret"])) if M["regret"] else np.nan,
        leader_inelig_frac=M["inelig"] / max(1, M["led"]),
        failover_s=failover if cfg.faults else np.nan, fault_target=fault_hit,
        leader_energy_loss=leader_energy_loss,
        progress=k_true / nK, complete=int(k_true == nK), t_complete=t_done,
        coverage=float(cover.mean()) if cover is not None else np.nan,
        keepout_s=M["keepout_s"], keepout_any=int(M["keepout_s"] > 0), ko_min_dist=float(M.get("ko_min", np.nan)),
        burst_t=float(getattr(adv, "burst_t", np.nan)) if adv is not None else np.nan,
        xt_mean=float(np.mean(M["xt"])) if M["xt"] else np.nan, xt_max=float(np.max(M["xt"])) if M["xt"] else np.nan,
        rmse_mean=float(np.mean(M["rmse"])) if M["rmse"] else np.nan,
        min_sep_hh=min_hh, min_sep_ha=min_ha, contact_hh_ticks=contacts_hh, contact_ha_ticks=contacts_ha,
        collision_hh=int(min_hh < cfg.d_min), collision_ha=int(min_ha < cfg.d_min),
        contact_hh=int(min_hh < cfg.d_contact), contact_ha=int(min_ha < cfg.d_contact),
        departed_energy=int(fleet.departed_energy.sum()), crashed=int(fleet.crashed.sum()),
        pkts_tx=chan.tx, pkts_rx=chan.delivered, bytes_per_node_s=Fld.width(n) * 4 * 10.0,
        elections=el.stats["elections"], terms_max=el.stats["terms_max"], qc_reject=el.stats["qc_reject"],
        term_ignored=el.stats["term_ignored"], impeach_events=el.stats["impeach"], handovers=el.stats["xfer"],
        stepdown_inelig=el.stats["stepdown_inelig"], stepdown_quorum=el.stats["stepdown_quorum"],
        duration_s=T, t_end=t, wall_s=time.time() - wall0,
        false_distrust=M["fd_num"] / M["fd_den"] if M["fd_den"] else np.nan,
        att_distrusted=M["att_d_num"] / M["att_d_den"] if M["att_d_den"] else np.nan,
    )
    if el.graw:
        G = np.asarray(el.graw)
        for c, nm in enumerate(("gv", "gc", "gxt", "gs")):
            res[f"{nm}_max"] = float(G[:, c].max())
            res[f"{nm}_q999"] = float(np.quantile(G[:, c], 0.999))
            res[f"{nm}_q99"] = float(np.quantile(G[:, c], 0.99))
    if RS is not None:
        from . import rmetrics
        S_ = {k_: np.asarray(v_) for k_, v_ in ser.items()}
        honest_mask = np.ones(n, bool)
        if att >= 0:
            honest_mask[att] = False
        tm = rmetrics.episode_truth_metrics(S_['t'], S_['p'].astype(float), S_['alive'], honest_mask, plan, att, cfg.t_boot,
                                            t_att=att_start, t_g=RS.get('t_g'), q_rec=RS.get('q_rec'), q_ref=RS.get('q_ref'),
                                            d_contact=cfg.d_contact, d_coll=cfg.d_min, d_safe=cfg.d_safe)
        res.update({'r_' + k_: v_ for k_, v_ in tm.items() if not k_.startswith('_')})
        res['r_auth_s'] = rmetrics.authority_seconds(S_['t'], S_['led'], honest_mask & S_['alive'][-1], att, S_['att_act']) if att >= 0 else 0.0
        res['r_shield_reject'] = int(el.shield_reject.sum())
        res['r_energy_honest_wh'] = float((S_['E'][0][honest_mask] - S_['E'][-1][honest_mask]).sum())
        res['r_scn'] = RS.get('id')
        res['r_installed'] = int(inst_done)
        res['series'] = S_
        res['series_extra'] = {'grid': tm['_grid'], 'qW': tm['_qW'], 'sg': tm['_sg']}
    if cfg.record:
        res["traj"] = np.asarray(traj)
        res["series"] = rec_series
        res["plan"] = plan
        res["classes"] = list(fleet.classes)
    return res
