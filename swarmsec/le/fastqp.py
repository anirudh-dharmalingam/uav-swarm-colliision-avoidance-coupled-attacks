"""Vectorised HOCBF-QP collision filter for large swarms (leader-election study, N up to 64).

Same constraint family as swarmsec.safety.SafetyFilter (second-order exponential CBF on
h_ij = ||p_ij||^2 - Ds_ij^2 with the decentralised a_j = 0 convention), solved for ALL agents at
once by dual coordinate ascent (Hildreth's method) on

    min_u,s  1/2 ||u - u_nom||^2 + rho/2 ||s||^2   s.t.  a_k.u + s_k >= c_k   (k = K nearest)
             -u_max <= u <= u_max                    (hard; rho = inf rows)

Rows are unit-normalised.  With rho < inf the dual Hessian A A^T + I/rho is positive definite, so
coordinate ascent converges linearly; `iters` sweeps are enough for the 3-D problems here (checked
against the cvxpy/OSQP filter in tests/test_le.py).  A final norm clip enforces ||u|| <= u_max.
"""
from __future__ import annotations

import numpy as np


def hocbf_rows(own_p, own_v, P, V, ds, k1, k2, radius, k_max, alive=None):
    """Per-agent HOCBF rows from each agent's OWN view.

    own_p, own_v : (N,3) ego estimates.  P, V : (N,N,3) receiver views (row i = what agent i believes
    about every j); ds : (N,N) barrier radius per pair.  Returns unit rows A (N,K,3), c (N,K), nbr (N,K)
    (-1 = unused row, satisfied trivially)."""
    n = own_p.shape[0]
    pr = own_p[:, None, :] - P                    # (N,M,3), M >= N (extra columns = sensed obstacles)
    vr = own_v[:, None, :] - V
    d = np.linalg.norm(pr, axis=2)
    d[np.arange(n), np.arange(n)] = np.inf
    if alive is not None:
        d[:, ~alive] = np.inf
    K = min(k_max, P.shape[1] - 1)
    order = np.argsort(d, axis=1)[:, :K]          # (N,K)
    dk = np.take_along_axis(d, order, axis=1)
    use = dk <= radius
    I = np.arange(n)[:, None]
    prk, vrk = pr[I, order], vr[I, order]         # (N,K,3)
    Ds = ds[I, order]
    h = (prk * prk).sum(-1) - Ds * Ds
    hdot = 2.0 * (prk * vrk).sum(-1)
    a = 2.0 * prk
    c = -2.0 * (vrk * vrk).sum(-1) - (k1 + k2) * hdot - k1 * k2 * h
    an = np.maximum(np.linalg.norm(a, axis=2), 1e-9)
    A = a / an[..., None]
    c = c / an
    A = np.where(use[..., None], A, 0.0)
    c = np.where(use, c, -1e6)
    nbr = np.where(use, order, -1)
    return A, c, nbr


def _obj(u, u_nom, A, c, rho):
    v = np.maximum(0.0, c - (A * u[:, None, :]).sum(-1))
    return 0.5 * ((u - u_nom) ** 2).sum(-1) + 0.5 * rho * (v * v).sum(-1)


def solve(u_nom, A, c, u_max, rho=1e4, iters=40, tol=1e-9, clip_norm=True):
    """Batched exact solver.  Eliminating the slacks (s_k = max(0, c_k - a_k.u) at the optimum) gives

        min_u  f(u) = 1/2 ||u - u_nom||^2 + rho/2 sum_k max(0, c_k - a_k.u)^2     s.t.  |u_d| <= u_max

    f is convex, C^1 and piecewise quadratic, so a projected semi-smooth Newton method (Bertsekas'
    two-metric projection: Newton on the coordinates not held at a bound, Armijo backtracking along the
    projected arc) terminates at the exact minimiser in a few iterations.  All agents are solved at once.
    Returns u (N,3) (then norm-clipped to u_max, as the legacy SafetyFilter does), max slack (N,), and the
    nominal margin min_k (a_k.u_nom - c_k) (N,)."""
    n, K, _ = A.shape
    act_rows = c > -1e5
    margin = np.where(act_rows, (A * u_nom[:, None, :]).sum(-1) - c, np.inf).min(axis=1) if K else np.full(n, np.inf)
    U = float(u_max)
    u = np.clip(u_nom, -U, U)
    need = (np.any(act_rows & ((A * u[:, None, :]).sum(-1) < c), axis=1)) | np.any(np.abs(u_nom) > U, axis=1)
    if not need.any():
        return u_nom.copy(), np.zeros(n), margin
    idx = np.nonzero(need)[0]
    Ai, ci, u0, ui = A[idx], np.where(act_rows[idx], c[idx], -1e6), u_nom[idx], u[idx]
    eye = np.eye(3)
    for _ in range(iters):
        r = ci - (Ai * ui[:, None, :]).sum(-1)                    # >0: violated soft row
        on = r > 0
        g = (ui - u0) - rho * ((on * r)[..., None] * Ai).sum(1)
        H = eye[None] + rho * np.einsum("mk,mki,mkj->mij", on.astype(float), Ai, Ai)
        # Bertsekas epsilon-active set: coordinates within eps of a bound whose gradient points outward are
        # held (and snapped) at the bound; without it the iterate creeps toward the face and stalls
        eps = np.minimum(1e-3, np.linalg.norm(ui - np.clip(ui - g, -U, U), axis=1, keepdims=True))
        fixed = ((ui <= -U + eps) & (g > 0)) | ((ui >= U - eps) & (g < 0))
        ui = np.where(fixed, np.sign(ui) * U, ui)
        if not np.any(np.abs(np.where(fixed, 0.0, g)) > tol):
            break
        Hm = np.where((fixed[:, :, None] | fixed[:, None, :]), 0.0, H) + fixed[:, :, None] * eye[None]
        gm = np.where(fixed, 0.0, g)
        d = -np.linalg.solve(Hm, gm[..., None])[..., 0]
        f0 = _obj(ui, u0, Ai, ci, rho)
        alpha = np.ones(len(idx))
        best = ui.copy()
        done = np.zeros(len(idx), bool)
        for _ls in range(30):
            cand = np.clip(ui + alpha[:, None] * d, -U, U)
            ok = _obj(cand, u0, Ai, ci, rho) <= f0 + 1e-4 * (gm * (cand - ui)).sum(-1) + 1e-14
            newly = ok & ~done
            best[newly] = cand[newly]
            done |= ok
            if done.all():
                break
            alpha = np.where(done, alpha, 0.5 * alpha)
        ui = best
    u_out = u_nom.copy()
    u_out[idx] = ui
    nrm = np.linalg.norm(u_out, axis=1, keepdims=True)
    if clip_norm:
        u_out = np.where(nrm > U, u_out * (U / np.maximum(nrm, 1e-12)), u_out)
    slack = np.zeros(n)
    slack[idx] = np.maximum(0.0, ci - (Ai * ui[:, None, :]).sum(-1)).max(axis=1) if K else 0.0
    return u_out, slack, margin
