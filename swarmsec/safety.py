"""Higher-order CBF-QP safety filter + safety-feasibility-margin instrumentation.

For the double integrator the pairwise safety function h_ij = ||p_i-p_j||^2 - Ds^2
has relative degree two, so we use a second-order (exponential) CBF. For agent i
against neighbour j, writing p_rel = p_i-p_j, v_rel = v_i-v_j:

    h        = ||p_rel||^2 - Ds^2
    hdot     = 2 p_rel . v_rel
    hddot    = 2||v_rel||^2 + 2 p_rel . (a_i - a_j)
    psi1     = hdot + k1 h
    HOCBF    : psi1dot + k2 psi1 >= 0

Assuming a decentralised convention a_j = 0 (each agent is responsible for its
own avoidance), the constraint is linear in the ego acceleration a_i = u_i:

    (2 p_rel) . u_i  >=  c_ij
    c_ij = -2||v_rel||^2 - (k1+k2) hdot - k1 k2 h

The safety filter solves, per agent:

    min_{u,s}  1/2||u - u_nom||^2 + rho/2 ||s||^2
    s.t.       a_k . u + s_k >= c_k,   -u_max <= u <= u_max,   s >= 0

Instrumentation (the security observables of the paper):
  * margin m_i      = min_k (a_k.u_nom - c_k)/||a_k||    (signed; <0 => nominal unsafe)
  * qp_correction   = ||u* - u_nom||                      (how hard the filter works)
  * slack           = max_k s_k                           (>0 => QP infeasibility)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import warnings

import numpy as np

try:
    import cvxpy as cp
    _HAVE_CVXPY = True
    # OSQP occasionally reports "solution may be inaccurate" on tight active sets;
    # we already polish and the residual is immaterial for these small QPs.
    warnings.filterwarnings("ignore", message="Solution may be inaccurate",
                            module="cvxpy")
except Exception:  # pragma: no cover
    _HAVE_CVXPY = False


@dataclass
class SafetyConfig:
    d_min: float = 1.5        # hard collision radius (for metrics)
    d_safe: float = 2.5       # CBF barrier activation radius (Ds)
    k1: float = 2.0           # HOCBF gain (psi1)
    k2: float = 2.0           # HOCBF gain (psi1dot)
    u_max: float = 8.0        # per-agent acceleration limit (actuator)
    slack_penalty: float = 1.0e4
    neighbor_radius: float = 10.0
    k_max: int = 8            # max CBF constraints per agent (nearest neighbours)
    connectivity_radius: float | None = None   # C3: keep links shorter than this (HOCBF)
    k_conn: int = 3           # number of connectivity constraints per agent


@dataclass
class FilterResult:
    u_safe: np.ndarray        # (N, 3)
    margin: np.ndarray        # (N,) min signed feasibility margin
    qp_correction: np.ndarray  # (N,)
    slack: np.ndarray         # (N,)
    active: np.ndarray        # (N,) bool, filter engaged
    min_margin_neighbor: np.ndarray  # (N,) index of tightest neighbour, -1 if none


class _CachedQP:
    """Fixed-size DPP QP so OSQP factorisation is reused across solves."""

    def __init__(self, k_max: int, u_max: float, rho: float):
        self.k_max = k_max
        self.A = cp.Parameter((k_max, 3))
        self.c = cp.Parameter(k_max)
        self.u_nom = cp.Parameter(3)
        u = cp.Variable(3)
        s = cp.Variable(k_max, nonneg=True)
        self._u, self._s = u, s
        cons = [self.A @ u + s >= self.c, u <= u_max, u >= -u_max]
        obj = 0.5 * cp.sum_squares(u - self.u_nom) + 0.5 * rho * cp.sum_squares(s)
        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def solve(self, A, c, u_nom):
        self.A.value = A
        self.c.value = c
        self.u_nom.value = u_nom
        self.prob.solve(
            solver=cp.OSQP,
            warm_start=True,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=20000,
            polish=True,
            verbose=False,
        )
        if self._u.value is None:
            return u_nom.copy(), np.zeros(self.k_max)
        return self._u.value.copy(), np.asarray(self._s.value).copy()


class _CachedQPTwoTier:
    """Two-tier QP: unit-normalised nominal rows (hard, penalty rho) and robust rows
    (nominal + assumed neighbour acceleration bound, SOFT with penalty rho_r). The
    robust tier therefore degrades gracefully -- it can be violated by a bounded,
    penalised amount -- instead of driving the nominal rows infeasible when the
    actuator/tracking authority is smaller than the assumed bound."""

    def __init__(self, n_rows: int, u_max: float, rho: float):
        self.R = n_rows
        m = 2 * n_rows
        self.A = cp.Parameter((m, 3))
        self.c = cp.Parameter(m)
        self.pen = cp.Parameter(m, nonneg=True)
        self.u_nom = cp.Parameter(3)
        u = cp.Variable(3)
        s = cp.Variable(m, nonneg=True)
        self._u, self._s = u, s
        cons = [self.A @ u + s >= self.c, u <= u_max, u >= -u_max]
        obj = 0.5 * cp.sum_squares(u - self.u_nom) + 0.5 * cp.sum(cp.multiply(self.pen, cp.square(s)))
        self.prob = cp.Problem(cp.Minimize(obj), cons)
        self.rho = rho

    def solve(self, A_hat, c_nom, c_rob, valid, rho_r, u_nom):
        R = self.R
        self.A.value = np.vstack([A_hat, A_hat])
        self.c.value = np.concatenate([c_nom, c_rob])
        pen = np.full(2 * R, self.rho)
        pen[R:][valid] = rho_r
        self.pen.value = pen
        self.u_nom.value = u_nom
        self.prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-6, eps_rel=1e-6,
                        max_iter=20000, polish=True, verbose=False)
        if self._u.value is None:
            return u_nom.copy(), np.zeros(R)
        return self._u.value.copy(), np.asarray(self._s.value)[:R].copy()


@dataclass
class SafetyFilter:
    cfg: SafetyConfig = field(default_factory=SafetyConfig)

    @property
    def n_rows(self) -> int:
        return self.cfg.k_max + (self.cfg.k_conn if self.cfg.connectivity_radius else 0)

    def __post_init__(self):
        self._qp = None
        self._qp2 = None
        if _HAVE_CVXPY:
            self._qp = _CachedQP(self.n_rows, self.cfg.u_max, self.cfg.slack_penalty)

    def _constraints(self, i, positions, velocities, d_safe_row, own_p=None, own_v=None):
        """Build (A, c, nbr_ids) for agent i over its k_max nearest neighbours."""
        cfg = self.cfg
        op = positions[i] if own_p is None else own_p
        ov = velocities[i] if own_v is None else own_v
        p_rel = op - positions            # (N,3)
        v_rel = ov - velocities
        dist = np.linalg.norm(p_rel, axis=1)
        dist[i] = np.inf
        order = np.argsort(dist)
        order = order[dist[order] <= cfg.neighbor_radius][: cfg.k_max]

        A = np.zeros((self.n_rows, 3))
        c = np.full(self.n_rows, -1.0e6)   # inactive rows: trivially satisfied
        nbr_ids = np.full(self.n_rows, -1, dtype=int)
        for row, j in enumerate(order):
            pr, vr = p_rel[j], v_rel[j]
            Ds = d_safe_row[j]
            h = float(pr @ pr) - Ds * Ds
            hdot = 2.0 * float(pr @ vr)
            a_k = 2.0 * pr
            c_k = -2.0 * float(vr @ vr) - (cfg.k1 + cfg.k2) * hdot - cfg.k1 * cfg.k2 * h
            A[row] = a_k
            c[row] = c_k
            nbr_ids[row] = j

        if cfg.connectivity_radius:
            # Connectivity HOCBF  h = Rc^2 - ||p||^2 >= 0 on the k_conn longest links.
            # Rows keep nbr_ids = -1 so the *collision* margin m_i ignores them.
            Rc = cfg.connectivity_radius
            cand = np.nonzero((dist <= cfg.neighbor_radius) & (dist >= 0.5 * Rc))[0]
            cand = cand[np.argsort(-dist[cand])][: cfg.k_conn]
            for r, j in enumerate(cand):
                pr, vr = p_rel[j], v_rel[j]
                h = Rc * Rc - float(pr @ pr)
                A[cfg.k_max + r] = -2.0 * pr
                c[cfg.k_max + r] = (2.0 * float(vr @ vr)
                                    + 2.0 * (cfg.k1 + cfg.k2) * float(pr @ vr)
                                    - cfg.k1 * cfg.k2 * h)
        return A, c, nbr_ids

    def margins_only(self, positions, velocities, u_nom, d_safe=None, self_pos=None, self_vel=None):
        """Defense-independent feasibility margin of `u_nom` against a FIXED
        reference barrier on the given (true) geometry. No QP. Returns
        (margin (N,), tightest-neighbour (N,))."""
        n = positions.shape[0]
        d_safe = self.cfg.d_safe if d_safe is None else d_safe
        d_mat = np.full((n, n), d_safe)
        margin = np.full(n, np.inf)
        tight = np.full(n, -1, dtype=int)
        for i in range(n):
            A, c, nbr_ids = self._constraints(i, positions, velocities, d_mat[i],
                                              None if self_pos is None else self_pos[i],
                                              None if self_vel is None else self_vel[i])
            valid = nbr_ids >= 0
            if not np.any(valid):
                continue
            rn = np.linalg.norm(A, axis=1)
            signed = np.where(valid, (A @ u_nom[i] - c) / np.maximum(rn, 1e-9), np.inf)
            k = int(np.argmin(signed))
            margin[i] = float(signed[k])
            tight[i] = int(nbr_ids[k])
        return margin, tight

    def passthrough(self, positions, velocities, u_nom, d_safe_matrix=None, self_pos=None, self_vel=None):
        """Safety filter OFF: the raw controller output (actuator-clipped) is applied.
        Margins are still computed so the damage can be measured."""
        n = positions.shape[0]
        margin, tight = self.margins_only(positions, velocities, u_nom, self.cfg.d_safe, self_pos, self_vel)
        u = u_nom.copy()
        nrm = np.linalg.norm(u, axis=1, keepdims=True)
        u = np.where(nrm > self.cfg.u_max, u * self.cfg.u_max / np.maximum(nrm, 1e-9), u)
        return FilterResult(u_safe=u, margin=margin, qp_correction=np.zeros(n), slack=np.zeros(n),
                            active=np.zeros(n, dtype=bool), min_margin_neighbor=tight)

    def filter(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        u_nom: np.ndarray,
        d_safe_matrix: np.ndarray | None = None,
        self_pos: np.ndarray | None = None,
        self_vel: np.ndarray | None = None,
        accel_bound: np.ndarray | None = None,
        robust_soft: float | None = None,
    ) -> FilterResult:
        """accel_bound[j] >= ||a_j|| is an assumed bound on neighbour j's unmodelled
        acceleration. Worst case of the relative-acceleration term -2 p.a_j is
        -2||p|| a_bar_j, so the HOCBF row becomes  A_k u >= c_k + ||A_k|| a_bar_j
        (A_k = 2 p_rel, ||A_k|| = 2||p||). Only the QP sees the tightened row: the
        reported margin / blame instrumentation stays the plain geometric one, which
        prevents a suspicion -> bound -> blame positive feedback loop."""
        n = positions.shape[0]
        cfg = self.cfg
        if d_safe_matrix is None:
            d_safe_matrix = np.full((n, n), cfg.d_safe)

        u_safe = u_nom.copy()
        margin = np.full(n, np.inf)
        slack = np.zeros(n)
        qp_corr = np.zeros(n)
        active = np.zeros(n, dtype=bool)
        tight = np.full(n, -1, dtype=int)

        for i in range(n):
            A, c, nbr_ids = self._constraints(i, positions, velocities, d_safe_matrix[i],
                                              None if self_pos is None else self_pos[i],
                                              None if self_vel is None else self_vel[i])
            row_norms = np.linalg.norm(A, axis=1)
            valid = nbr_ids >= 0
            if np.any(valid):
                signed = (A @ u_nom[i] - c) / np.maximum(row_norms, 1e-9)
                signed = np.where(valid, signed, np.inf)
                k = int(np.argmin(signed))
                margin[i] = float(signed[k])
                tight[i] = int(nbr_ids[k])

            c_qp = c
            ab = None
            if accel_bound is not None and np.any(valid):
                ab = np.asarray(accel_bound, float)[np.where(valid, nbr_ids, 0)]
                c_qp = np.where(valid, c + row_norms * ab, c)
            if (robust_soft is not None and ab is not None and self._qp is not None):
                # two-tier: hard nominal rows + soft robust rows (unit-normalised)
                if self._qp2 is None:
                    self._qp2 = _CachedQPTwoTier(self.n_rows, cfg.u_max, cfg.slack_penalty)
                rn = np.maximum(row_norms, 1e-9)
                act = c > -1.0e5                      # collision AND connectivity rows
                A_hat = np.where(act[:, None], A / rn[:, None], 0.0)
                c_hat = np.where(act, c / rn, c)
                c_rob = np.where(valid, c_hat + ab, c_hat)
                u_i, s_i = self._qp2.solve(A_hat, c_hat, c_rob, valid, float(robust_soft), u_nom[i])
            elif self._qp is not None and (np.any(valid) or np.any(c > -1.0e5)):
                u_i, s_i = self._qp.solve(A, c_qp, u_nom[i])
            else:
                u_i, s_i = u_nom[i], np.zeros(self.n_rows)

            # Enforce the hard actuator norm limit after the QP box.
            nrm = float(np.linalg.norm(u_i))
            if nrm > cfg.u_max:
                u_i = u_i * (cfg.u_max / nrm)

            u_safe[i] = u_i
            slack[i] = float(np.max(s_i)) if len(s_i) else 0.0
            qp_corr[i] = float(np.linalg.norm(u_i - u_nom[i]))
            active[i] = qp_corr[i] > 1e-3

        return FilterResult(
            u_safe=u_safe,
            margin=margin,
            qp_correction=qp_corr,
            slack=slack,
            active=active,
            min_margin_neighbor=tight,
        )
