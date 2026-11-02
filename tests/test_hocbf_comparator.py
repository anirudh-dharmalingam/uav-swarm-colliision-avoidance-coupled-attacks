"""rarf_hocbf_v1: the comparator adapter must reproduce the unmodified SafetyFilter exactly, and the original files stay unchanged."""
import numpy as np

from swarmsec.ral26_hocbf import HocbfPolicy, filter_one
from swarmsec.safety import SafetyConfig, SafetyFilter
from tests.test_rarf_ablation import _bounds, _episode, _pk


def test_filter_one_equals_safetyfilter_agent0():
    rng = np.random.default_rng(1)
    for robust in (None, 2.5):
        for soft in (None, 100.0):
            if soft is not None and robust is None:
                continue
            for _ in range(200):
                # fresh instances: the original filter warm-starts OSQP across successive solves, so its answer depends on the
                # solve history (observed differences up to ~0.07 m/s^2); equality is checked without that history
                F1, F2 = SafetyFilter(SafetyConfig(u_max=3.0, d_safe=2.5)), SafetyFilter(SafetyConfig(u_max=3.0, d_safe=2.5))
                n = int(rng.integers(2, 7))
                P = np.c_[rng.uniform(-6, 6, (n, 2)), np.zeros(n)]
                V = np.c_[rng.normal(0, 1.5, (n, 2)), np.zeros(n)]
                U = np.c_[rng.normal(0, 2, (n, 2)), np.zeros(n)]
                ab = None if robust is None else np.full(n, robust)
                ref = F1.filter(P, V, U, accel_bound=ab, robust_soft=soft).u_safe[0]
                got = filter_one(F2, P, V, U[0], np.full(n, 2.5), ab, soft)[0]
                assert np.allclose(ref, got, atol=1e-6), (ref, got)


def test_hocbf_policy_limits_and_diagnostics():
    B = _bounds()
    rng = np.random.default_rng(4)
    for kind, H in (("HOCBF", {"d_safe": 2.5}), ("RHOCBF", {"d_safe": 2.5, "a_bar": 2.5, "robust_soft": 100.0})):
        pol = HocbfPolicy(kind, B, H)
        prev = np.zeros(3)
        for t, p, v, pk, u_nom in _episode(rng, 60):
            d = pol.step(t, p, v, {j: _pk(j, a, b, tr, B) for j, (a, b, tr) in pk.items()}, u_nom)
            assert np.linalg.norm(d.u) <= B.U + 1e-9 and np.linalg.norm(d.u - prev) <= B.slew + 1e-9 and d.u[2] == 0
            assert {"hc_slack", "hc_res_final", "hc_status"} <= set(pol.diag)
            prev = d.u
