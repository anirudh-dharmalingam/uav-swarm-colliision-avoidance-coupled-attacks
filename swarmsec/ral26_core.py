"""RAL26 core: conditional bounded-reachability safety rows, hard-safety solve, authority reserve,
energy screening and range-only lower bound (master prompt section 2, implemented verbatim).

Everything here is CONTROLLER-SIDE mathematics.  It never advances a vehicle: in every RAL26 flight
experiment the vehicles are PX4 SITL autopilots flying Gazebo X500 models.  The guarantees are
CONDITIONAL on the stated bounds (beta_p, beta_v, T_net_max, A_j, W_i, ego estimate errors); when a
bound is known not to hold the caller must label the result INVALID_ASSUMPTIONS, never certified.

Units: metres, seconds, m/s, m/s^2, W, Wh.  All vectors are common-frame ENU.

Status labels (section 2.5)
  CERTIFIED_INTERVAL   rows satisfied by the selected command, independently re-checked
  UNCERTIFIED          rows infeasible / solver failed / recheck failed -> emergency policy output
  INVALID_ASSUMPTIONS  a declared premise is unbounded or known violated
  RETURN_REQUIRED      energy margin < 0 (mission manager must return/land)
  MISSION / EMERGENCY  supervisor modes
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CERTIFIED = "CERTIFIED_INTERVAL"
UNCERTIFIED = "UNCERTIFIED"
INVALID = "INVALID_ASSUMPTIONS"
RETURN_REQUIRED = "RETURN_REQUIRED"
UNKNOWN_ENERGY = "UNKNOWN_ENERGY"
ENERGY_OK = "ENERGY_OK"

ROW_TOL = 1e-6          # independent recheck tolerance on hard rows [m]
NORM_TOL = 1e-7         # actuator-ball recheck tolerance [m/s^2]


# ------------------------------------------------------------------------------------------------
# 2.1 per-packet reachability
@dataclass
class NeighborPacket:
    """Receiver-side record q_ij.  t_rx is the RECEIVER's local monotonic receive time."""
    sender: int
    p_msg: np.ndarray
    v_msg: np.ndarray
    t_rx: float
    max_transit: float          # T_net_max for this link scenario (inf => unbounded)
    beta_p: float               # assumed falsification bound on the reported position [m]
    beta_v: float               # assumed falsification bound on the reported velocity [m/s]
    A_j: float                  # bound on j's TOTAL physical acceleration [m/s^2]


@dataclass
class Envelope:
    sender: int
    r_hat: np.ndarray
    v_hat: np.ndarray
    B_p: float
    B_v: float
    A_j: float
    age: float
    valid: bool
    reason: str = ""


def build_envelope(pkt: NeighborPacket, t_now: float, p_own, v_own,
                   ego_err_p: float, ego_err_v: float, fixed_age: float | None = None) -> Envelope:
    """a_ij = max(0, t - t_rx) + T_net_max  (or a caller-fixed age for the NoAoI / static variants).

    p_c = p_msg + a v_msg;  beta_p_age = beta_p + a beta_v + A a^2/2;  beta_v_age = beta_v + A a.
    """
    p_own = np.asarray(p_own, float)
    v_own = np.asarray(v_own, float)
    if fixed_age is None:
        a = max(0.0, float(t_now) - float(pkt.t_rx)) + float(pkt.max_transit)
    else:
        a = float(fixed_age)
    valid = math.isfinite(a) and math.isfinite(pkt.beta_p) and math.isfinite(pkt.beta_v) \
        and math.isfinite(pkt.A_j) and np.all(np.isfinite(pkt.p_msg)) and np.all(np.isfinite(pkt.v_msg))
    if not valid:
        return Envelope(pkt.sender, np.zeros(3), np.zeros(3), math.inf, math.inf, math.inf, a, False,
                        "unbounded transit/falsification/acceleration bound or non-finite packet")
    p_c = np.asarray(pkt.p_msg, float) + a * np.asarray(pkt.v_msg, float)
    beta_p_age = pkt.beta_p + a * pkt.beta_v + 0.5 * pkt.A_j * a * a
    beta_v_age = pkt.beta_v + pkt.A_j * a
    return Envelope(pkt.sender, p_own - p_c, v_own - np.asarray(pkt.v_msg, float),
                    beta_p_age + ego_err_p, beta_v_age + ego_err_v, pkt.A_j, a, True)


def build_envelope_sound(pkt: NeighborPacket, t_now: float, p_own, v_own, ego_err_p: float, ego_err_v: float,
                         sender_err_p: float = 0.0, sender_err_v: float = 0.0) -> Envelope:
    """Coupled-safety campaign (Gate A) packet-age enclosure that is sound for UNKNOWN actual transit.

    The receiver knows only s = max(0, t - t_rx) and that the actual transit l lies in [0, T].  With midpoint
    age a_c = s + T/2 and maximum age a_+ = s + T the neighbour's true state satisfies
        || p_j(t) - (p_msg + a_c v_msg) || <= bar_beta_p + a_+ bar_beta_v + (T/2)||v_msg|| + A_j a_+^2/2
        || v_j(t) - v_msg ||               <= bar_beta_v + A_j a_+
    where bar_beta = declared falsification bound + sender estimation error (the sender's own EKF error
    sits INSIDE its report, so it is added to beta, and it is propagated by age exactly like beta).
    Ego error is separate.  Sender timestamps are never used (an authenticated sender may lie about them);
    only receiver-local arrival time.  The legacy `build_envelope` centred the report at the MAXIMUM age
    and omitted the |a - a_c| ||v_msg|| term, which makes the enclosure false (counterexample in
    tests/test_coupled_safety.py)."""
    p_own = np.asarray(p_own, float)
    v_own = np.asarray(v_own, float)
    s = max(0.0, float(t_now) - float(pkt.t_rx))
    T = float(pkt.max_transit)
    a_c, a_p = s + 0.5 * T, s + T
    valid = (math.isfinite(a_p) and math.isfinite(pkt.beta_p) and math.isfinite(pkt.beta_v)
             and math.isfinite(pkt.A_j) and np.all(np.isfinite(pkt.p_msg)) and np.all(np.isfinite(pkt.v_msg)))
    if not valid:
        return Envelope(pkt.sender, np.zeros(3), np.zeros(3), math.inf, math.inf, math.inf, a_p, False,
                        "unbounded transit/falsification/acceleration bound or non-finite packet")
    vm = np.asarray(pkt.v_msg, float)
    bp, bv = pkt.beta_p + sender_err_p, pkt.beta_v + sender_err_v
    p_c = np.asarray(pkt.p_msg, float) + a_c * vm
    R_p = ego_err_p + bp + 0.5 * T * float(np.linalg.norm(vm)) + a_p * bv + 0.5 * pkt.A_j * a_p * a_p
    R_v = ego_err_v + bv + pkt.A_j * a_p
    return Envelope(pkt.sender, p_own - p_c, v_own - vm, R_p, R_v, pkt.A_j, a_p, True)


# ------------------------------------------------------------------------------------------------
# 2.2 full-interval sufficient rows
def direction(env: Envelope, h: float) -> np.ndarray:
    """n = (r_hat + h v_hat)/||.||; any fixed unit vector keeps the certificate sound, so degenerate
    cases fall back to r_hat, then to +x."""
    for cand in (env.r_hat + h * env.v_hat, env.r_hat):
        nn = float(np.linalg.norm(cand))
        if nn > 1e-9:
            return cand / nn
    return np.array([1.0, 0.0, 0.0])


def certificate_rows(env: Envelope, h: float, M: int, U_i: float, W_i: float, d_safe: float):
    """Rows A_k u >= b_k, k = 0..M, sufficient for ||r(tau)|| >= d_safe on [0, h] (conditional).

    L = ||v_hat|| + B_v + h (U_i + A_j + W_i);   buffer = L * Delta / 2
    A_k = tau_k^2 n / 2
    b_k = d_safe + B_p + tau_k B_v + tau_k^2 (A_j + W_i)/2 + buffer - n.(r_hat + tau_k v_hat)
    """
    n = direction(env, h)
    delta = h / M
    L = float(np.linalg.norm(env.v_hat)) + env.B_v + h * (U_i + env.A_j + W_i)
    buf = 0.5 * L * delta
    tau = delta * np.arange(M + 1)
    A = 0.5 * (tau ** 2)[:, None] * n[None, :]
    b = (d_safe + env.B_p + tau * env.B_v + 0.5 * tau ** 2 * (env.A_j + W_i) + buf
         - (n @ env.r_hat + tau * (n @ env.v_hat)))
    return A, b, n


def interval_lower_bound(env: Envelope, u, h: float, W_i: float, n=None, samples: int = 400) -> float:
    """Dense evaluation of min_tau n.z(tau,u) - R(tau): the quantity the rows bound from below
    (used by the inter-sample tests and by `certified_horizon`)."""
    n = direction(env, h) if n is None else n
    tau = np.linspace(0.0, h, samples)
    z = env.r_hat[None, :] + tau[:, None] * env.v_hat[None, :] + 0.5 * (tau ** 2)[:, None] * np.asarray(u)[None, :]
    R = env.B_p + tau * env.B_v + 0.5 * tau ** 2 * (env.A_j + W_i)
    return float(np.min(z @ n - R))


def exclusion_safe(env: Envelope, h: float, U_i: float, W_i: float, d_safe: float) -> bool:
    """Sound neighbour exclusion by reachable envelope: for ANY ||u|| <= U_i,
    ||r(tau)|| >= ||r_hat|| - B_p - tau(||v_hat|| + B_v) - tau^2 (U_i + A_j + W_i)/2 >= d_safe."""
    lb = (float(np.linalg.norm(env.r_hat)) - env.B_p - h * (float(np.linalg.norm(env.v_hat)) + env.B_v)
          - 0.5 * h * h * (U_i + env.A_j + W_i))
    return lb >= d_safe


# ------------------------------------------------------------------------------------------------
# conic solves (Clarabel through cvxpy, compiled once per row budget) + independent recheck
class _ConicProblems:
    def __init__(self, max_rows: int):
        import cvxpy as cp
        self.cp = cp
        self.K = max_rows
        self.A = cp.Parameter((max_rows, 3))
        self.b = cp.Parameter(max_rows)
        self.un = cp.Parameter(3)
        self.U = cp.Parameter(nonneg=True)
        self.up = cp.Parameter(3)                    # previous command (slew constraint centre)
        self.dmax = cp.Parameter(nonneg=True)        # max change per control tick (jerk * dt)
        u = cp.Variable(3)
        self.u = u
        self.qp = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(u - self.un)),
                             [self.A @ u >= self.b, cp.norm(u, 2) <= self.U, cp.norm(u - self.up, 2) <= self.dmax])
        # emergency (UNCERTIFIED) fallback: per-row L1 exact-penalty slack, every row relaxed only as needed
        self.rho = cp.Parameter(nonneg=True)
        self.us, self.s = cp.Variable(3), cp.Variable(max_rows, nonneg=True)
        self.sp = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(self.us - self.un) + self.rho * cp.sum(self.s)),
                             [self.A @ self.us + self.s >= self.b, cp.norm(self.us, 2) <= self.U,
                              cp.norm(self.us - self.up, 2) <= self.dmax])
        g = cp.Variable()
        self.g, self.ug = g, cp.Variable(3)
        self.gp = cp.Problem(cp.Maximize(g), [self.A @ self.ug - self.b >= g, cp.norm(self.ug, 2) <= self.U,
                                              cp.norm(self.ug - self.up, 2) <= self.dmax])

    def _load(self, A, b, U, un=None, up=None, dmax=None):
        K = A.shape[0]
        Ap = np.zeros((self.K, 3))
        bp = np.full(self.K, -1e3)              # inactive padding rows: 0 >= -1e3
        Ap[:K], bp[:K] = A, b
        self.A.value, self.b.value, self.U.value = Ap, bp, float(U)
        self.up.value = np.zeros(3) if up is None else np.asarray(up, float)
        self.dmax.value = 1e3 if dmax is None else float(dmax)
        if un is not None:
            self.un.value = np.asarray(un, float)

    def solve_qp(self, A, b, U, un, up=None, dmax=None):
        self._load(A, b, U, un, up, dmax)
        try:
            self.qp.solve(solver="CLARABEL")
            return self.u.value, str(self.qp.status)
        except Exception as exc:                 # solver error -> caller marks UNCERTIFIED
            return None, f"solver_error:{type(exc).__name__}"

    def solve_soft(self, A, b, U, un, up=None, dmax=None, rho=1e3):
        self._load(A, b, U, un, up, dmax)
        self.rho.value = float(rho)
        try:
            self.sp.solve(solver="CLARABEL")
            return self.us.value, str(self.sp.status)
        except Exception as exc:
            return None, f"solver_error:{type(exc).__name__}"

    def solve_gamma(self, A, b, U, up=None, dmax=None):
        self._load(A, b, U, None, up, dmax)
        try:
            self.gp.solve(solver="CLARABEL")
            return self.g.value, self.ug.value, str(self.gp.status)
        except Exception as exc:
            return None, None, f"solver_error:{type(exc).__name__}"


_CACHE: dict[int, _ConicProblems] = {}


def _problems(K: int) -> _ConicProblems:
    size = 8
    while size < K:
        size *= 2
    if size not in _CACHE:
        _CACHE[size] = _ConicProblems(size)
    return _CACHE[size]


def prewarm(sizes=(8, 16, 32, 64)):
    """Compile every conic problem once (cvxpy canonicalisation costs 1.5-2.3 s per problem on first
    use, which stalled the 20 Hz loop at mission start and tripped PX4's offboard-loss failsafe)."""
    for K in sizes:
        P = _problems(K)
        A = np.zeros((1, 3)); A[0, 0] = 1.0
        b = np.array([-1.0])
        P.solve_qp(A, b, 3.0, np.zeros(3), np.zeros(3), 0.4)
        P.solve_gamma(A, b, 3.0, np.zeros(3), 0.4)
        P.solve_soft(A, b, 3.0, np.zeros(3), np.zeros(3), 0.4)


def recheck(u, A, b, U, up=None, dmax=None):
    """Independent validation (never trusts the solver): finite, norm, slew, every hard row."""
    if u is None:
        return False, -math.inf, "no_solution"
    u = np.asarray(u, float)
    if not np.all(np.isfinite(u)):
        return False, -math.inf, "non_finite"
    if float(np.linalg.norm(u)) > U + NORM_TOL:
        return False, -math.inf, "actuator_norm"
    if up is not None and dmax is not None and float(np.linalg.norm(u - np.asarray(up))) > dmax + NORM_TOL:
        return False, -math.inf, "slew"
    res = float(np.min(A @ u - b)) if A.shape[0] else math.inf
    if res < -ROW_TOL:
        return False, res, "row_violation"
    return True, res, ""


def solve_hard_safety(u_nom, A, b, U, up=None, dmax=None):
    """min 0.5||u-u_nom||^2 s.t. A u >= b, ||u||_2 <= U, ||u - u_prev|| <= dmax (identified jerk limit).
    No slack, no post-solve clipping.

    Returns (u, certified, min_row_residual, failure_reason).  u is None if no certified command."""
    u_nom = np.asarray(u_nom, float)
    if A.shape[0] == 0 and up is None:
        nn = float(np.linalg.norm(u_nom))
        u = u_nom if nn <= U else u_nom * (U / nn)
        return u, True, math.inf, ""
    if np.any((np.abs(A).sum(axis=1) < 1e-15) & (b > ROW_TOL)):
        return None, False, float(np.min(-b[np.abs(A).sum(axis=1) < 1e-15])), "k0_infeasible"
    # cheap path: nominal already satisfies everything
    ok, res, _ = recheck(u_nom, A, b, U, up, dmax)
    if ok:
        return u_nom.copy(), True, res, ""
    u, status = _problems(max(A.shape[0], 1)).solve_qp(A if A.shape[0] else np.zeros((1, 3)),
                                                      b if A.shape[0] else np.array([-1.0]), U, u_nom, up, dmax)
    ok, res, why = recheck(u, A, b, U, up, dmax)
    if not ok:
        return None, False, res, f"{status}:{why}" if status != "optimal" else why
    return np.asarray(u, float), True, res, ""


def authority_reserve_m(A, b, U, up=None, dmax=None):
    """gamma_i = max_{||u||<=U, ||u-u_prev||<=dmax} min_k (A_k u - b_k)  [m]; returns (gamma, u_star, validated).

    gamma >= 0 is sufficient for feasibility of THESE conservative rows -- not recursive
    feasibility and not proof that a collision-free motion exists or does not exist."""
    if A.shape[0] == 0:
        return math.inf, (np.zeros(3) if up is None else np.asarray(up, float)), True
    g, u, status = _problems(A.shape[0]).solve_gamma(A, b, U, up, dmax)
    if g is None or u is None:
        return -math.inf, (np.zeros(3) if up is None else np.asarray(up, float)), False
    u = np.asarray(u, float)
    if float(np.linalg.norm(u)) > U:
        u = u * (U / float(np.linalg.norm(u)))
    if up is not None and dmax is not None:          # project back into the slew ball if solver tolerance left it
        du = u - np.asarray(up, float)
        if float(np.linalg.norm(du)) > dmax:
            u = np.asarray(up, float) + du * (dmax / float(np.linalg.norm(du)))
    if not np.all(np.isfinite(u)):
        return -math.inf, np.zeros(3), False
    g_true = float(np.min(A @ u - b))           # independent value of the returned command
    return g_true, u, abs(g_true - float(g)) < 1e-3 or g_true >= float(g)


def least_violation_command(u_nom, A, b, U, gamma_star=None, up=None, dmax=None, rho=1e3):
    """Emergency (UNCERTIFIED) command shared by D24R and every D26 variant: the conventional
    soft-constrained fallback  min 0.5||u-u_nom||^2 + rho * sum_k s_k  s.t. A u + s >= b, s >= 0,
    ||u|| <= U, slew.  Each row is relaxed only as much as needed (a uniform relaxation set by the worst,
    often stale, row would leave the nearest neighbour unprotected).  Never certified; the result is
    re-validated for finiteness, norm and slew.  Returns None if that fails (caller then uses u*)."""
    if A.shape[0] == 0:                     # nothing threatening under these rows: nominal within ball/slew
        u = np.asarray(u_nom, float).copy()
        if float(np.linalg.norm(u)) > U:
            u *= U / float(np.linalg.norm(u))
        if up is not None and dmax is not None and float(np.linalg.norm(u - np.asarray(up))) > dmax:
            du = u - np.asarray(up)
            u = np.asarray(up) + du * (dmax / float(np.linalg.norm(du)))
        return u
    u, _ = _problems(A.shape[0]).solve_soft(A, b, U, u_nom, up, dmax, rho)
    if u is None:
        return None
    u = np.asarray(u, float)
    ok = np.all(np.isfinite(u)) and float(np.linalg.norm(u)) <= U + 1e-6 and (
        up is None or dmax is None or float(np.linalg.norm(u - np.asarray(up))) <= dmax + 1e-6)
    return u if ok else None


# ------------------------------------------------------------------------------------------------
# 2.3 energy
def update_energy_wh(E_wh: float, P_w: float, dt_s: float) -> float:
    return max(0.0, E_wh - P_w * dt_s / 3600.0)


def energy_supervisor(E_remaining_wh, P_return_upper_w, T_return_upper_s, E_reserve_wh, valid=True):
    """E_return_upper = P_return_upper T_return_upper / 3600; margin = E - E_return_upper - E_reserve."""
    if not valid or not all(math.isfinite(x) for x in (E_remaining_wh, P_return_upper_w, T_return_upper_s)):
        return {"mode": UNKNOWN_ENERGY, "margin_wh": math.nan, "E_return_upper_wh": math.nan}
    e_ret = P_return_upper_w * T_return_upper_s / 3600.0
    margin = E_remaining_wh - e_ret - E_reserve_wh
    return {"mode": RETURN_REQUIRED if margin < 0 else ENERGY_OK, "margin_wh": margin, "E_return_upper_wh": e_ret}


# ------------------------------------------------------------------------------------------------
# 2.4 optional range-only observation
def minimal_range_lower_bound(z, sigma, v_close_bound, age, h):
    """d_lower = max(0, z - sigma - V_close (age + h)); only meaningful with valid coverage."""
    return max(0.0, float(z) - float(sigma) - float(v_close_bound) * (float(age) + float(h)))


# ------------------------------------------------------------------------------------------------
def combined_resource_state(gamma, energy_margin_wh, tracking_valid, network_valid, energy_mode=ENERGY_OK):
    """Coupled eligibility (gamma >= 0) and (E_margin >= 0) and tracking/network validity."""
    energy_ok = energy_mode == ENERGY_OK and (energy_margin_wh is None or energy_margin_wh >= 0)
    return {"eligible": bool(gamma >= 0 and energy_ok and tracking_valid and network_valid),
            "safety_feasible": bool(gamma >= 0), "energy_ok": bool(energy_ok),
            "tracking_valid": bool(tracking_valid), "network_valid": bool(network_valid)}


@dataclass
class Bounds:
    """Declared premises of one agent's certificate."""
    U: float = 3.0              # ego actuator ball [m/s^2] (commanded)
    W: float = 1.0              # ego tracking residual bound (PX4-identified) [m/s^2]
    A_honest: float = 4.0       # total physical acceleration of an honest neighbour [m/s^2]
    A_any: float = 6.0          # total physical acceleration of ANY neighbour incl. compromised
    beta_p: float = 1.25        # assumed falsification bound, position [m]
    beta_v: float = 0.5         # assumed falsification bound, velocity [m/s]
    T_net_max: float = 0.25     # bounded transit [s]
    ego_err_p: float = 0.15     # own-estimate error bound vs truth [m]
    ego_err_v: float = 0.2      # [m/s]
    sender_err_p: float = 0.0   # bound on an HONEST sender's own-estimate error inside its report [m]
    sender_err_v: float = 0.0   # (v2 audit: beta_p bounds falsification relative to the sender's estimate,
                                #  so total report error vs truth is beta + sender estimate error) [m/s]
    d_safe: float = 1.5         # certified separation [m]
    h: float = 0.6              # certificate horizon (held command) [s]
    slew: float = math.inf      # max ||u_k - u_{k-1}|| per control tick (identified jerk * dt) [m/s^2]
    M: int = 6                  # subdivisions
    tracking_valid: bool = True # False when HF1 found no bound that covers the regime
    envelope: str = "legacy"    # "legacy" (historical, unsound for unknown transit) | "sound_age" (build_envelope_sound)
    extra: dict = field(default_factory=dict)


def rows_for(envs, B: Bounds):
    """Stack rows of every neighbour that the sound exclusion test cannot discard."""
    As, bs, used, n_excl = [], [], [], 0
    for e in envs:
        if not e.valid:
            continue
        if exclusion_safe(e, B.h, B.U, B.W, B.d_safe):
            n_excl += 1
            continue
        A, b, _ = certificate_rows(e, B.h, B.M, B.U, B.W, B.d_safe)
        As.append(A)
        bs.append(b)
        used.append(e.sender)
    if not As:
        return np.zeros((0, 3)), np.zeros(0), used, n_excl
    return np.vstack(As), np.concatenate(bs), used, n_excl
