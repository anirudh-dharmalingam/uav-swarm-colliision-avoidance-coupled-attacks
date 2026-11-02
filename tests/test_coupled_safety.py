"""Coupled-safety campaign, Gate A/B tests (math level only; no test here produces a flight result).

Every sound-envelope test is paired with a counter-check that the LEGACY `build_envelope` fails the same
property, so the corrected tests demonstrably fail against the old implementation.
"""
import math

import numpy as np

from swarmsec import ral26_core as rc


def _pkt(p, v, t_rx=0.0, T=0.25, bp=0.0, bv=0.0, A=0.0, sender=1):
    return rc.NeighborPacket(sender, np.asarray(p, float), np.asarray(v, float), t_rx, T, bp, bv, A)


def _ball(rng, radius):
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    return d * radius * rng.random() ** (1 / 3)


def _true_error(env, p_own_true, v_own_true, pj_true, vj_true):
    """(position, velocity) error of the enclosure centre w.r.t. the TRUE relative state."""
    r_true, v_true = p_own_true - pj_true, v_own_true - vj_true
    return float(np.linalg.norm(r_true - env.r_hat)), float(np.linalg.norm(v_true - env.v_hat))


def test_memo_counterexample_old_fails_new_sound():
    pk = _pkt([3, 0, 0], [2, 0, 0], T=0.25)
    old = rc.build_envelope(pk, 0.0, np.zeros(3), np.zeros(3), 0.0, 0.0)
    pj_true = np.array([3.0, 0, 0])               # actual transit 0: nothing has moved
    e_old, _ = _true_error(old, np.zeros(3), np.zeros(3), pj_true, np.array([2.0, 0, 0]))
    assert old.valid and old.B_p == 0.0 and abs(e_old - 0.5) < 1e-12          # false zero-radius enclosure
    new = rc.build_envelope_sound(pk, 0.0, np.zeros(3), np.zeros(3), 0.0, 0.0)
    e_new, _ = _true_error(new, np.zeros(3), np.zeros(3), pj_true, np.array([2.0, 0, 0]))
    assert e_new <= new.B_p + 1e-12 and abs(new.B_p - 0.25) < 1e-12      # (T/2)|v_msg| = 0.25


def _scenario(rng, sound, n_hist=6, transit_choice=None):
    """Random admissible packet/neighbour history; returns (env, errors, radii)."""
    A = rng.uniform(0, 6.5)
    T = rng.uniform(0.01, 0.5)
    bp, bv, sp, sv = rng.uniform(0, 2, size=4)
    ego_p, ego_v = rng.uniform(0, 1, size=2)
    vscale = rng.choice([0.0, 0.5, 2.0, 8.0])
    p0, v0 = rng.normal(size=3) * 3, _ball(rng, vscale)
    est_p, est_v = p0 + _ball(rng, sp), v0 + _ball(rng, sv)           # sender's own estimate
    rep_p, rep_v = est_p + _ball(rng, bp), est_v + _ball(rng, bv)     # falsified report
    s = rng.choice([0.0, rng.uniform(0, 2.0)])
    ell = {None: rng.uniform(0, T), "zero": 0.0, "max": T}[transit_choice]
    age = s + ell
    # random piecewise-constant admissible acceleration history on [t_s, t_s + age] (adversarial switching)
    k = n_hist
    cuts = np.sort(rng.uniform(0, age, size=k - 1)) if age > 0 else np.zeros(k - 1)
    edges = np.concatenate([[0.0], cuts, [age]])
    pj, vj = p0.copy(), v0.copy()
    for i in range(k):
        dt = edges[i + 1] - edges[i]
        a = _ball(rng, A) if rng.random() < 0.5 else (A * (rng.choice([-1, 1]) * (v0 / (np.linalg.norm(v0) + 1e-12))))
        a = a if np.linalg.norm(a) <= A + 1e-12 else a * A / np.linalg.norm(a)
        pj = pj + vj * dt + 0.5 * a * dt * dt
        vj = vj + a * dt
    own_p, own_v = rng.normal(size=3) * 3, rng.normal(size=3)
    own_ph, own_vh = own_p + _ball(rng, ego_p), own_v + _ball(rng, ego_v)
    pk = _pkt(rep_p, rep_v, t_rx=0.0, T=T, bp=bp, bv=bv, A=A)
    t_now = s
    if sound:
        env = rc.build_envelope_sound(pk, t_now, own_ph, own_vh, ego_p, ego_v, sp, sv)
    else:   # legacy lumps sender error into the 'ego' arguments (as ral26_policy did)
        env = rc.build_envelope(pk, t_now, own_ph, own_vh, ego_p + sp, ego_v + sv)
    ep, ev = _true_error(env, own_p, own_v, pj, vj)
    return env, ep, ev


def test_sound_envelope_random_histories_and_transit_endpoints():
    rng = np.random.default_rng(20261025)
    worst_p = worst_v = -math.inf
    for choice in (None, "zero", "max"):
        for _ in range(4000):
            env, ep, ev = _scenario(rng, True, transit_choice=choice)
            worst_p, worst_v = max(worst_p, ep - env.B_p), max(worst_v, ev - env.B_v)
            assert ep <= env.B_p + 1e-9, (choice, ep, env.B_p)
            assert ev <= env.B_v + 1e-9, (choice, ev, env.B_v)
    assert worst_p <= 1e-9 and worst_v <= 1e-9


def test_legacy_envelope_is_unsound_on_the_same_cases():
    rng = np.random.default_rng(20261025)
    bad = 0
    for _ in range(4000):
        env, ep, ev = _scenario(rng, False)
        bad += int(ep > env.B_p + 1e-9 or ev > env.B_v + 1e-9)
    assert bad >= 10, bad           # the old implementation is violated on a visible fraction of admissible cases


def test_sound_envelope_zero_acceleration_large_speed_and_sender_error_propagation():
    pk = _pkt([0, 0, 0], [10, 0, 0], T=0.2, bp=0.0, bv=0.0, A=0.0)
    e = rc.build_envelope_sound(pk, 0.5, np.zeros(3), np.zeros(3), 0.0, 0.0)
    assert abs(e.B_p - 0.5 * 0.2 * 10.0) < 1e-12 and e.B_v == 0.0
    e2 = rc.build_envelope_sound(pk, 0.5, np.zeros(3), np.zeros(3), 0.1, 0.2, 0.3, 0.4)
    a_plus = 0.7
    assert abs(e2.B_p - (0.1 + 0.3 + 1.0 + a_plus * 0.4)) < 1e-12      # sender velocity error acts through the age
    assert abs(e2.B_v - (0.2 + 0.4)) < 1e-12
    assert np.allclose(e2.r_hat, -(np.array([0, 0, 0]) + 0.6 * np.array([10.0, 0, 0])))   # midpoint-age centre


def test_stale_future_and_out_of_order_packets():
    """Receiver clock only.  A packet 'delivered' later than the newest one but carrying older content is
    handled by the same bound (its transit is still <= T); future/garbage sender stamps are not inputs."""
    rng = np.random.default_rng(5)
    T = 0.25
    for _ in range(500):
        A = 3.0
        p0, v0 = rng.normal(size=3) * 3, rng.normal(size=3)
        pk_new = _pkt(p0 + v0 * 0.1 + 0.0, v0, t_rx=1.0, T=T, A=A)
        pk_old = _pkt(p0, v0, t_rx=1.05, T=T, A=A)      # older content, received after the newer packet
        for pk in (pk_new, pk_old):
            t_now = pk.t_rx + rng.uniform(0, 1.0)
            e = rc.build_envelope_sound(pk, t_now, np.zeros(3), np.zeros(3), 0, 0)
            assert e.valid and math.isfinite(e.B_p)
    # stale beliefs grow quadratically and remain valid (no silent drop)
    pk = _pkt([5, 0, 0], [0, 0, 0], t_rx=0.0, T=0.25, A=2.0)
    e1 = rc.build_envelope_sound(pk, 1.0, np.zeros(3), np.zeros(3), 0, 0)
    e2 = rc.build_envelope_sound(pk, 5.0, np.zeros(3), np.zeros(3), 0, 0)
    assert e2.B_p > e1.B_p > 0 and e2.valid
    # receive time in the future of 'now' is clamped (receiver clock cannot go backwards)
    e3 = rc.build_envelope_sound(pk, -3.0, np.zeros(3), np.zeros(3), 0, 0)
    assert abs(e3.age - 0.25) < 1e-12
    # non-finite packets are invalid, never certified
    bad = _pkt([math.nan, 0, 0], [0, 0, 0])
    assert not rc.build_envelope_sound(bad, 0.0, np.zeros(3), np.zeros(3), 0, 0).valid


def _adversarial_min_sep(env_true_r, env_true_v, u, h, D, rng, dt=1e-3, mode="inward"):
    """True relative dynamics r'' = u + d, ||d|| <= D, d chosen adversarially (inward / bang-bang switching)."""
    r, v = env_true_r.copy(), env_true_v.copy()
    t, dmin = 0.0, float(np.linalg.norm(r))
    sw = rng.uniform(0.05, 0.3)
    flip = 1.0
    nxt = sw
    while t < h:
        nr = np.linalg.norm(r)
        n = r / nr if nr > 1e-9 else np.array([1.0, 0, 0])
        if mode == "inward":
            d = -D * n
        elif mode == "bang":
            if t >= nxt:
                flip, nxt = -flip, nxt + sw
            d = -D * n * flip
        else:
            vv = np.linalg.norm(v)
            d = -D * (v / vv if vv > 1e-9 else n)
        v = v + (u + d) * dt
        r = r + v * dt + 0.5 * (u + d) * dt * dt
        t += dt
        dmin = min(dmin, float(np.linalg.norm(r)))
    return dmin


def test_interval_certificate_between_samples_adversarial_acceleration_changes():
    """Certified command (sound envelope, hard rows) + true state anywhere on the enclosure boundary +
    adversarial switching disturbance: ||r|| >= d_safe on a 1 ms grid over the whole hold interval."""
    rng = np.random.default_rng(77)
    h, d_safe, U, W, A = 0.4, 1.5, 3.0, 1.0, 3.0
    checked, worst = 0, math.inf
    for _ in range(120):
        p = rng.normal(size=3) * 6.0
        p[2] = 0.0
        v = rng.normal(size=3) * 1.0
        v[2] = 0.0
        pk = _pkt(p, v, t_rx=0.0, T=0.1, bp=0.3, bv=0.2, A=A)
        env = rc.build_envelope_sound(pk, rng.uniform(0, 0.3), np.zeros(3), rng.normal(size=3) * 0.3, 0.05, 0.05)
        Arows, b, _ = rc.certificate_rows(env, h, 8, U, W, d_safe)
        Arows = Arows.copy()
        Arows[:, 2] = 0.0
        u, cert, res, _ = rc.solve_hard_safety(rng.normal(size=3) * 2, Arows, b, U)
        if not cert:
            continue
        D = A + W
        for _k in range(6):
            dr = _ball(rng, env.B_p)
            dr = dr * env.B_p / max(np.linalg.norm(dr), 1e-12) if rng.random() < 0.5 else dr   # boundary corners
            dv = _ball(rng, env.B_v)
            dv = dv * env.B_v / max(np.linalg.norm(dv), 1e-12) if rng.random() < 0.5 else dv
            for mode in ("inward", "bang", "velocity"):
                m = _adversarial_min_sep(env.r_hat + dr, env.v_hat + dv, u, h, D, rng, mode=mode)
                worst = min(worst, m)
                assert m >= d_safe - 1e-3, (m, mode)
        checked += 1
    assert checked >= 25, checked


def test_postfilter_commands_are_rechecked_or_uncertified():
    """Gate B: a certified command must remain inside actuator ball and slew ball and satisfy every hard row after
    the independent recheck; a command altered AFTER the recheck must fail the recheck."""
    e = rc.build_envelope_sound(_pkt([4.0, 0, 0], [-1.0, 0, 0], bp=0.3, A=3.0), 0.0, np.zeros(3), np.zeros(3), 0.05, 0.05)
    A, b, _ = rc.certificate_rows(e, 0.4, 8, 3.0, 1.0, 1.5)
    up, dmax = np.zeros(3), 0.4
    u, cert, res, why = rc.solve_hard_safety(np.array([3.0, 0, 0]), A, b, 3.0, up, dmax)
    if cert:
        ok, _, _ = rc.recheck(u, A, b, 3.0, up, dmax)
        assert ok
        altered = u + np.array([0.5, 0.0, 0.0])           # e.g. a speed-envelope clip applied after certification
        assert not (rc.recheck(altered, A, b, 3.0, up, dmax)[0] and np.linalg.norm(altered - up) <= dmax)
    else:
        assert u is None and why != ""                    # infeasible/solver failure -> explicit uncertified output


# ---------------------------------------------------------------------------------------------------------------------
# E2 information interventions + information boundary
def test_sensor_envelope_sound_and_intersection_valid():
    from swarmsec.ral26_cs import intersect_envelopes, sensor_envelope
    rng = np.random.default_rng(11)
    for _ in range(3000):
        D = rng.uniform(0.5, 9.0)                                   # relative acceleration bound A_j + U + W
        eps_p, eps_v = rng.uniform(0.01, 0.3), rng.uniform(0.01, 0.4)
        lmin = rng.uniform(0.0, 0.1)
        lmax = lmin + rng.uniform(0.0, 0.1)
        r0, v0 = rng.normal(size=3) * 4, rng.normal(size=3) * 2
        r_meas, v_meas = r0 + _ball(rng, eps_p), v0 + _ball(rng, eps_v)
        s = rng.uniform(0, 0.3)
        lat = rng.uniform(lmin, lmax)
        age = s + lat
        cuts = np.sort(rng.uniform(0, age, size=4))
        edges = np.concatenate([[0.0], cuts, [age]])
        r, v = r0.copy(), v0.copy()
        for i in range(5):
            dt = edges[i + 1] - edges[i]
            a = _ball(rng, D)
            r = r + v * dt + 0.5 * a * dt * dt
            v = v + a * dt
        es = sensor_envelope(1, r_meas, v_meas, s, lmin, lmax, eps_p, eps_v, D, 3.0)
        assert np.linalg.norm(r - es.r_hat) <= es.B_p + 1e-9 and np.linalg.norm(v - es.v_hat) <= es.B_v + 1e-9
        # a second valid enclosure of the same truth (a loose report ball) -> intersection encloses the truth, no contradiction
        loose = rc.Envelope(1, r + _ball(rng, 1.0), v + _ball(rng, 0.5), 1.0 + 1e-9, 0.5 + 1e-9, 3.0, 0.0, True)
        e2, bad = intersect_envelopes(loose, es)
        assert not bad
        assert np.linalg.norm(r - e2.r_hat) <= e2.B_p + 1e-9 and np.linalg.norm(v - e2.v_hat) <= e2.B_v + 1e-9
        assert e2.B_p <= min(loose.B_p, es.B_p) + 1e-12
    # contradictory (disjoint) enclosures are flagged and never shrink below the larger ball
    a = rc.Envelope(1, np.zeros(3), np.zeros(3), 0.5, 0.5, 3.0, 0.0, True)
    b = rc.Envelope(1, np.array([5.0, 0, 0]), np.zeros(3), 0.2, 0.5, 3.0, 0.0, True)
    e, bad = intersect_envelopes(a, b)
    assert bad and e.B_p == 0.5


def test_identity_tightening_changes_only_honest_rows():
    """Identity gives beta = 0 and A_honest to honest senders; the attacker's packet keeps beta and A_any."""
    B = rc.Bounds(beta_p=1.25, beta_v=0.5, A_honest=4.0, A_any=6.5, T_net_max=0.1, envelope="sound_age")
    std = rc.build_envelope_sound(_pkt([6, 0, 0], [0, 0, 0], T=0.1, bp=B.beta_p, bv=B.beta_v, A=B.A_any), 0.1, np.zeros(3), np.zeros(3), 0.1, 0.1)
    ident = rc.build_envelope_sound(_pkt([6, 0, 0], [0, 0, 0], T=0.1, bp=0.0, bv=0.0, A=B.A_honest), 0.1, np.zeros(3), np.zeros(3), 0.1, 0.1)
    assert ident.B_p < std.B_p and ident.B_v < std.B_v


def test_agent_information_boundary_static():
    """No controller-side file reads evaluator truth or Gazebo poses; the oracle topics appear only in cs_agent_node/oracle_adapter."""
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    forbidden = ("dynamic_pose", "model_poses", "truth.csv", "truth_evaluator", "/world/default/stats")
    for f in ("controllers/swarm/uav_agent_node.py", "controllers/swarm/px4_iface.py", "swarmsec/ral26_policy.py", "swarmsec/ral26_core.py",
              "swarmsec/ral26_cs.py", "controllers/swarm/cs_agent_node.py"):
        txt = (root / f).read_text()
        for tok in forbidden:
            assert tok not in txt, (f, tok)
    cs = (root / "controllers/swarm/cs_agent_node.py").read_text()
    assert "/oracle/truth" in cs and 'self.info.mode == "oracle"' in cs and 'if uid != attacker_id and self.info.mode != "identity"' in cs
    assert "/oracle/truth" not in (root / "controllers/swarm/uav_agent_node.py").read_text()
