"""Execution chain v2 for the RAL26 / RARF honest controller (separately versioned; ral26_core / ral26_policy / cs_agent_node unchanged).

Defect fixed (rarf_ablation_v1 audit, output/research/paper_audit_v1/AUDIT_REPORT.md §B6): in exec chain v1 the node applies a speed-envelope
projection AFTER the policy (remove the outward acceleration component when the planar speed exceeds v_max).  That projection is not
slew-limited, is never re-checked against the hard rows, and the policy's previous-command reference u_prev is the PRE-projection command.
Sent commands therefore jumped by up to 3.0 m/s^2 per tick while every policy-level check passed.

Exec chain v2 enforces ONE execution set X jointly inside every command source (hard QP, soft L1 QP, max-reserve rescue, nominal law):

    X(u_prev, v) = { u : u_z = 0,  ||u|| <= U,  ||u - u_prev|| <= dmax,  n^T u <= c }
    with, only while the planar speed exceeds v_max,   n = v_xy / ||v_xy||,   c = max(0, n^T u_prev - dmax)
    (otherwise the half-space is absent),

where u_prev is the command actually PUBLISHED at the previous tick.  The half-space is the old speed envelope ("no outward acceleration
beyond v_max") made slew-compatible: the outward component must fall by at least dmax per tick until it reaches zero.  X is never
empty when ||u_prev|| <= U (u_prev - min(n^T u_prev, dmax) n lies in it; see `exec_witness`), so speed, norm and slew are always jointly
satisfiable and none of them has to be traded against another.  The price: the commanded outward acceleration is unwound at dmax per tick
instead of being cut in one tick, so the commanded speed can exceed v_max by at most dt * sum_k max(0, a0 - k dmax) <= 0.64 m/s
(a0 = U = 3, dmax = 0.4, dt = 0.05) plus the PX4 tracking residual.

The final command is revalidated by `validate_final` against X and, if the hard branch was selected, against every hard row.  Labels:
    CERTIFIED_INTERVAL    hard branch selected AND the final published command passes X and every hard row AND assumptions valid
    UNCERTIFIED_FINAL     hard branch selected but the final command fails a hard row (never silently kept as certified)
    FALLBACK_SOFT         best-estimate soft L1 command (uncertified; positive slack relaxes the geometry; not called safe)
    FALLBACK_RESCUE       max-reserve command of the hard rows (uncertified)
    EXEC_REPLACED         the selected command failed X (solver tolerance / downstream change): replaced by the closest command in X
                          (Euclidean projection by QP), else by the witness; uncertified
    NO_FILTER             nominal / speed-rule arms
    INVALID_ASSUMPTIONS   as v1 (precedence over the above when an envelope premise is flagged invalid)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from swarmsec import ral26_core as rc

EXEC_TOL = 1e-6          # recheck tolerance on norm / slew / half-space [m/s^2]
FINAL_CERTIFIED = rc.CERTIFIED
UNCERTIFIED_FINAL = "UNCERTIFIED_FINAL"
FALLBACK_SOFT = "FALLBACK_SOFT"
FALLBACK_RESCUE = "FALLBACK_RESCUE"
EXEC_REPLACED = "EXEC_REPLACED"


@dataclass
class ExecSet:
    U: float
    up: np.ndarray            # previous PUBLISHED command (slew centre)
    dmax: float
    n: np.ndarray | None      # outward unit direction (planar) when the speed envelope is active
    c: float                  # half-space offset  n^T u <= c

    @property
    def active(self) -> bool:
        return self.n is not None


def exec_set(U, up, dmax, v_own, vmax) -> ExecSet:
    up = np.zeros(3) if up is None else np.asarray(up, float).copy()
    up[2] = 0.0
    v = np.asarray(v_own, float)
    sp = float(np.linalg.norm(v[:2]))
    if sp > vmax and sp > 1e-9:
        n = np.array([v[0] / sp, v[1] / sp, 0.0])
        return ExecSet(float(U), up, float(dmax), n, max(0.0, float(n @ up) - float(dmax)))
    return ExecSet(float(U), up, float(dmax), None, math.inf)


def exec_witness(X: ExecSet) -> np.ndarray:
    """A command in X built from u_prev alone (proves X is non-empty when ||u_prev|| <= U)."""
    u = X.up.copy()
    nu = float(np.linalg.norm(u))
    if nu > X.U:                                   # only if u_prev itself was outside the ball (never for v2-published commands)
        u *= X.U / nu
    if X.active:
        a = float(X.n @ u)
        if a > X.c:                                # a > 0 here; shift by min(a, dmax): n^T u -> max(0, a - dmax) = c,
            u = u - min(a, X.dmax) * X.n           # slew = min(a, dmax) <= dmax, norm decreases (a > shift / 2)
    return u


def in_exec(u, X: ExecSet, tol=EXEC_TOL):
    """(ok, why) for finiteness, planarity, norm, slew and the speed half-space."""
    if u is None:
        return False, "no_command"
    u = np.asarray(u, float)
    if not np.all(np.isfinite(u)):
        return False, "non_finite"
    if abs(u[2]) > tol:
        return False, "non_planar"
    if float(np.linalg.norm(u)) > X.U + tol:
        return False, "actuator_norm"
    if float(np.linalg.norm(u - X.up)) > X.dmax + tol:
        return False, "slew"
    if X.active and float(X.n @ u) > X.c + tol:
        return False, "speed_envelope"
    return True, ""


def recheck_final(u, A, b, X: ExecSet):
    """Every applicable hard constraint on the FINAL command: X and every hard row (row tolerance as ral26_core.ROW_TOL)."""
    ok, why = in_exec(u, X)
    if not ok:
        return False, -math.inf, why
    res = float(np.min(A @ np.asarray(u, float) - b)) if A.shape[0] else math.inf
    if res < -rc.ROW_TOL:
        return False, res, "row_violation"
    return True, res, ""


class _ExecProblems:
    """Hard QP, soft L1 QP, max-reserve and Euclidean projection, each with the speed half-space as a HARD constraint."""

    def __init__(self, max_rows: int):
        import cvxpy as cp
        self.cp, self.K = cp, max_rows
        self.A, self.b = cp.Parameter((max_rows, 3)), cp.Parameter(max_rows)
        self.un, self.up = cp.Parameter(3), cp.Parameter(3)
        self.U, self.dmax = cp.Parameter(nonneg=True), cp.Parameter(nonneg=True)
        self.n, self.c = cp.Parameter(3), cp.Parameter()
        self.rho = cp.Parameter(nonneg=True)

        def X(u):
            return [cp.norm(u, 2) <= self.U, cp.norm(u - self.up, 2) <= self.dmax, self.n @ u <= self.c, u[2] == 0]
        self.u = cp.Variable(3)
        self.qp = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(self.u - self.un)), [self.A @ self.u >= self.b] + X(self.u))
        self.us, self.s = cp.Variable(3), cp.Variable(max_rows, nonneg=True)
        self.sp = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(self.us - self.un) + self.rho * cp.sum(self.s)),
                             [self.A @ self.us + self.s >= self.b] + X(self.us))
        self.g, self.ug = cp.Variable(), cp.Variable(3)
        self.gp = cp.Problem(cp.Maximize(self.g), [self.A @ self.ug - self.b >= self.g] + X(self.ug))
        self.uq = cp.Variable(3)
        self.pp = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(self.uq - self.un)), X(self.uq))

    def load(self, A, b, X: ExecSet, un=None):
        K = A.shape[0]
        Ap, bp = np.zeros((self.K, 3)), np.full(self.K, -1e3)
        Ap[:K], bp[:K] = A, b
        self.A.value, self.b.value = Ap, bp
        self.U.value, self.dmax.value, self.up.value = X.U, X.dmax, X.up
        if X.active:
            self.n.value, self.c.value = X.n, float(X.c)
        else:
            self.n.value, self.c.value = np.zeros(3), 1e3          # inactive: 0 <= 1e3
        if un is not None:
            self.un.value = np.asarray(un, float)

    def solve(self, prob, var):
        try:
            prob.solve(solver="CLARABEL")
            return (None if var.value is None else np.asarray(var.value, float)), str(prob.status)
        except Exception as exc:
            return None, f"solver_error:{type(exc).__name__}"


_CACHE: dict[int, _ExecProblems] = {}


def _P(K: int) -> _ExecProblems:
    size = 8
    while size < K:
        size *= 2
    if size not in _CACHE:
        _CACHE[size] = _ExecProblems(size)
    return _CACHE[size]


def prewarm(sizes=(8, 16, 32, 64)):
    X = ExecSet(3.0, np.zeros(3), 0.4, np.array([1.0, 0, 0]), 0.0)
    A, b = np.array([[1.0, 0, 0]]), np.array([-1.0])
    for K in sizes:
        P = _P(K)
        P.load(A, b, X, np.zeros(3))
        for pr, v in ((P.qp, P.u), (P.sp, P.us), (P.gp, P.ug), (P.pp, P.uq)):
            P.rho.value = 1e3
            P.solve(pr, v)


def project_exec(u, X: ExecSet):
    """Closest command in X (Euclidean); witness if the solve or its recheck fails.  Returns (u, method)."""
    u = np.asarray(u, float).copy()
    u[2] = 0.0
    if in_exec(u, X)[0]:
        return u, "identity"
    P = _P(1)
    P.load(np.zeros((0, 3)), np.zeros(0), X, u)
    q, _ = P.solve(P.pp, P.uq)
    if q is not None:
        q[2] = 0.0
        if in_exec(q, X)[0]:
            return q, "qp"
    return exec_witness(X), "witness"


def solve_hard_x(u_nom, A, b, X: ExecSet):
    """Hard branch over X: (u, certified, min_row_residual, why).  Independent recheck on X and every row."""
    u_nom = np.asarray(u_nom, float)
    if A.shape[0] and np.any((np.abs(A).sum(axis=1) < 1e-15) & (b > rc.ROW_TOL)):
        return None, False, float(np.min(-b[np.abs(A).sum(axis=1) < 1e-15])), "k0_infeasible"
    ok, res, _ = recheck_final(u_nom, A, b, X)
    if ok:
        return u_nom.copy(), True, res, ""
    P = _P(max(A.shape[0], 1))
    P.load(A, b, X, u_nom)
    u, status = P.solve(P.qp, P.u)
    if u is not None:
        u[2] = 0.0 if abs(u[2]) < 1e-9 else u[2]
    ok, res, why = recheck_final(u, A, b, X)
    if not ok:
        return None, False, res, f"{status}:{why}" if status != "optimal" else why
    return u, True, res, ""


def soft_x(u_nom, A, b, X: ExecSet, rho=1e3):
    """Best-estimate soft L1 command over X (rows relaxed, X hard).  None if it fails its X recheck."""
    if A.shape[0] == 0:
        u, _ = project_exec(u_nom, X)
        return u
    P = _P(A.shape[0])
    P.load(A, b, X, u_nom)
    P.rho.value = float(rho)
    u, _ = P.solve(P.sp, P.us)
    if u is None:
        return None
    u[2] = 0.0 if abs(u[2]) < 1e-9 else u[2]
    return u if in_exec(u, X)[0] else None


def reserve_x(A, b, X: ExecSet):
    """Max-reserve command of the rows over X: (gamma, u_star, validated).  u_star always lies in X."""
    if A.shape[0] == 0:
        u = exec_witness(X)
        return math.inf, u, True
    P = _P(A.shape[0])
    P.load(A, b, X)
    u, _ = P.solve(P.gp, P.ug)
    if u is None or not in_exec(u, X)[0]:
        u = exec_witness(X) if u is None else project_exec(u, X)[0]
        return float(np.min(A @ u - b)), u, False
    g_true = float(np.min(A @ u - b))
    return g_true, u, P.g.value is not None and (abs(g_true - float(P.g.value)) < 1e-3 or g_true >= float(P.g.value))


def validate_final(u_pub, A, b, X: ExecSet, selected: str, assumptions_ok: bool):
    """Explicit final-command rule.  Returns (u_final, label, why, replaced).
    selected in {"hard", "soft", "rescue", "nominal"}.  The final command ALWAYS lies in X; a hard label survives only if every hard
    row also holds on the final command."""
    ok_x, why = in_exec(u_pub, X)
    replaced = False
    u = np.asarray(u_pub, float) if u_pub is not None else None
    if not ok_x:
        u, _ = project_exec(u if u is not None else X.up, X)
        replaced = True
    if replaced:
        label = EXEC_REPLACED
    elif selected == "hard":
        ok, _, why_r = recheck_final(u, A, b, X)
        label, why = (FINAL_CERTIFIED, "") if ok else (UNCERTIFIED_FINAL, why_r)
    elif selected == "soft":
        label = FALLBACK_SOFT
    elif selected == "rescue":
        label = FALLBACK_RESCUE
    else:
        label = "NO_FILTER"
    if not assumptions_ok and selected != "nominal":
        label = rc.INVALID
    return u, label, why, replaced
