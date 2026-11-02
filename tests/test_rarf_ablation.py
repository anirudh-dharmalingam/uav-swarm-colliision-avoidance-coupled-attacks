"""RARF ablation campaign (rarf_ablation_v1): branch selection, command-chain accounting, enclosure boundary cases, evaluator coverage.

The historical variants (D0R, D24R) must keep their exact command law; the new variants differ only in the documented switch.
"""
import json
import sys
from pathlib import Path

import numpy as np

from swarmsec import ral26_core as rc
from swarmsec.ral26_policy import AgentPolicy

ROOT = Path(__file__).resolve().parents[1]
FROZEN = json.loads((ROOT / "experiments" / "coupled_safety_v1" / "bounds_declared.json").read_text())["bounds"]


def _bounds():
    return rc.Bounds(**{k: v for k, v in FROZEN.items() if k in rc.Bounds.__dataclass_fields__})


def _pk(j, p, v, t_rx, B):
    return rc.NeighborPacket(j, np.array(p, float), np.array(v, float), t_rx, B.T_net_max, B.beta_p, B.beta_v, B.A_any)


def _episode(rng, n_steps=60):
    """Random closed-ish sequence: own state drifts, 1-3 neighbours at 2-12 m with random velocities, packets 0-0.3 s old."""
    seq = []
    p = np.array([0.0, 0.0, 5.0])
    v = np.array([0.8, 0.1, 0.0])
    nb = {j: (p + np.r_[rng.uniform(-10, 10, 2), 0.0], np.r_[rng.uniform(-1.5, 1.5, 2), 0.0]) for j in range(1, 1 + rng.integers(1, 4))}
    for k in range(n_steps):
        t = 100.0 + 0.05 * k
        p = p + 0.05 * v
        pk = {}
        for j, (pj, vj) in nb.items():
            pj = pj + 0.05 * vj
            nb[j] = (pj, vj)
            pk[j] = (pj.copy(), vj.copy(), t - rng.uniform(0.0, 0.3))
        u_nom = np.r_[rng.normal(0, 1.5, 2), 0.0]
        seq.append((t, p.copy(), v.copy(), pk, u_nom))
    return seq


def _run(variant, seq, B, **kw):
    pol = AgentPolicy(variant, B, **kw)
    out = []
    for t, p, v, pk, u_nom in seq:
        packets = {j: _pk(j, a, b, tr, B) for j, (a, b, tr) in pk.items()}
        d = pol.step(t, p, v, packets, u_nom)
        out.append((d, pol.u_prev.copy()))
    return out


def test_shadow_rarf_has_the_historical_command_law():
    B = _bounds()
    rng = np.random.default_rng(7)
    n_hard = n_fb = 0
    for _ in range(6):
        seq = _episode(rng)
        h = _run("D24R", seq, B)
        s = _run("D24R_SH", seq, B)
        for (dh, _), (ds, _) in zip(h, s):
            assert np.allclose(dh.u, ds.u, atol=1e-9), (dh.u, ds.u)
            assert dh.status == ds.status
            assert "br" not in dh.extra and "br" in ds.extra        # historical decisions carry no accounting
            br = ds.extra["br"]
            assert br["applied"] in ("hard", "soft", "rescue")
            assert (br["applied"] == "hard") == (ds.status == rc.CERTIFIED or (br["hard_ok"] == 1 and br["assump_ok"] == 0))
            assert br["soft_ok"] in (0, 1)                           # the shadow fallback runs on EVERY tick (matched compute)
            n_hard += br["applied"] == "hard"
            n_fb += br["applied"] != "hard"
    assert n_hard and n_fb, (n_hard, n_fb)                           # both branches exercised by the random cases


def test_forced_fallback_never_applies_the_hard_command_but_logs_it():
    B = _bounds()
    rng = np.random.default_rng(11)
    seen_hard_ok = 0
    for _ in range(6):
        seq = _episode(rng)
        out = _run("D24R_FF", seq, B)
        for d, u_prev in out:
            br = d.extra["br"]
            assert br["applied"] in ("soft", "rescue")
            assert d.status in ("FALLBACK_FORCED", rc.INVALID)
            assert np.allclose(u_prev, d.u)                          # previous-command state = the command actually selected
            assert float(np.linalg.norm(d.u)) <= B.U + 1e-6
            seen_hard_ok += br["hard_ok"]
    assert seen_hard_ok > 0                                          # hypothetical hard solves did succeed and were not applied


def test_forced_fallback_equals_rarf_on_ticks_where_rarf_falls_back():
    """Same state + same previous command => FF and RARF select the identical fallback command (only the switch differs)."""
    B = _bounds()
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(4):
        for t, p, v, pk, u_nom in _episode(rng, 30):
            packets = {j: _pk(j, a, b, tr, B) for j, (a, b, tr) in pk.items()}
            up = np.r_[rng.normal(0, 0.5, 2), 0.0]
            a, f = AgentPolicy("D24R_SH", B), AgentPolicy("D24R_FF", B)
            a.u_prev, f.u_prev = up.copy(), up.copy()
            da, df = a.step(t, p, v, packets, u_nom), f.step(t, p, v, packets, u_nom)
            if da.extra["br"]["applied"] != "hard":
                assert np.allclose(da.u, df.u, atol=1e-6)
                checked += 1
    assert checked > 5


def test_zero_time_necessary_condition_is_flagged():
    """A neighbour inside the current uncertainty radius + d_safe has a k = 0 row with no command coefficient."""
    B = _bounds()
    pol = AgentPolicy("D24R_SH", B)
    p = np.array([0.0, 0.0, 5.0])
    d = pol.step(10.0, p, np.zeros(3), {1: _pk(1, p + [2.0, 0, 0], [0, 0, 0], 10.0, B)}, np.zeros(3))
    br = d.extra["br"]
    assert br["k0_fail"] == 1 and br["hard_ok"] == 0 and br["applied"] in ("soft", "rescue")
    far = pol.step(10.05, p, np.zeros(3), {1: _pk(1, p + [40.0, 0, 0], [0, 0, 0], 10.05, B)}, np.zeros(3))
    assert far.extra["br"]["k0_fail"] == 0 and far.extra["br"]["hard_ok"] == 1


def test_fresh_packet_threshold_matches_the_analytic_4137_m():
    """Appendix A constant: the k = 0 row of a static fresh packet fails iff projected separation < 4.137 m + velocity terms (=0 here)."""
    B = _bounds()
    p = np.array([0.0, 0.0, 5.0])
    for dist, expect in ((4.10, 1), (4.20, 0)):
        pk = rc.NeighborPacket(1, p + [dist, 0, 0], np.zeros(3), 10.0, B.T_net_max, B.beta_p, B.beta_v, B.A_any)
        e = rc.build_envelope_sound(pk, 10.0, p, np.zeros(3), B.ego_err_p, B.ego_err_v, B.sender_err_p, B.sender_err_v)
        A, b = rc.certificate_rows(e, B.h, B.M, B.U, B.W, B.d_safe)[:2]
        k0 = int(np.any((np.abs(A).sum(axis=1) < 1e-15) & (b > rc.ROW_TOL)))
        assert k0 == expect, (dist, k0, b[:2])


def test_speed_restriction_rule():
    B = _bounds()
    rule = {"d_slow": 6.0, "d_stop": 3.0, "v_slow": 1.0, "v_near": 0.3, "k": 2.0}
    pol = AgentPolicy("D0R_SPD", B, speed_rule=rule)
    p = np.array([0.0, 0.0, 5.0])
    v = np.array([1.5, 0.0, 0.0])
    # far: no cap, nominal unchanged (inside ball/slew)
    d = pol.step(1.0, p, v, {1: _pk(1, p + [20.0, 0, 0], [0, 0, 0], 1.0, B)}, np.array([0.2, 0, 0]))
    assert d.extra["spd"]["brake"] == 0 and np.allclose(d.u, [0.2, 0, 0])
    # near (2 m): cap 0.3 m/s, brake opposes velocity; slew-limited from the previous command
    pol2 = AgentPolicy("D0R_SPD", B, speed_rule=rule)
    d2 = pol2.step(1.0, p, v, {1: _pk(1, p + [0.0, 2.0, 0], [0, 0, 0], 1.0, B)}, np.zeros(3))
    assert d2.extra["spd"]["brake"] == 1 and abs(d2.extra["spd"]["cap"] - 0.3) < 1e-12
    assert d2.u[0] < 0 and float(np.linalg.norm(d2.u)) <= B.slew + 1e-9
    # mid distance: linear cap
    pol3 = AgentPolicy("D0R_SPD", B, speed_rule=rule)
    d3 = pol3.step(1.0, p, v, {1: _pk(1, p + [0.0, 4.5, 0], [0, 0, 0], 1.0, B)}, np.zeros(3))
    assert abs(d3.extra["spd"]["cap"] - 0.65) < 1e-9
    # packets are age-propagated exactly as in the nominal consensus
    pol4 = AgentPolicy("D0R_SPD", B, speed_rule=rule)
    d4 = pol4.step(2.0, p, v, {1: _pk(1, p + [10.0, 0, 0], [-4.0, 0, 0], 1.0, B)}, np.zeros(3))
    assert abs(d4.extra["spd"]["d_min"] - 6.0) < 1e-9


def test_historical_d0r_unchanged():
    B = _bounds()
    pol = AgentPolicy("D0R", B)
    d = pol.step(1.0, np.zeros(3), np.zeros(3), {}, np.array([5.0, 0, 1.0]))
    assert d.extra == {} and np.allclose(d.u, [0.4, 0, 0])


def test_ablation_specs_and_evaluator_cover_every_episode():
    """Specs: complete matched blocks (every scenario x geometry has all arms); the outcome extractor reports every planned field."""
    sys.path.insert(0, str(ROOT / "experiments" / "rarf_ablation_v1"))
    import design  # noqa: E402
    for tag in ("dev", "exp1", "exp2"):
        eps = design.build(tag)
        names = [e["name"] for e in eps]
        assert len(names) == len(set(names))
        blocks = {}
        for e in eps:
            m = e["meta"]
            blocks.setdefault((m["seed"], m["geometry"], m["cell"]), set()).add(m["method"])
        sizes = {len(v) for v in blocks.values()}
        assert len(sizes) == 1, (tag, sizes)
        for e in eps:
            assert e["overrides"]["ral26"]["attack"]["attacker_policy"] == "D0R"
            assert e["overrides"]["ral26"]["bounds"]["envelope"] == "sound_age"
    sys.path.insert(0, str(ROOT / "scripts" / "rarf_ablation"))
    import outcomes  # noqa: E402
    assert set(outcomes.PLANNED) <= set(outcomes.FIELDS)
