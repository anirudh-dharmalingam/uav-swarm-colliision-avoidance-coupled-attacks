"""Per-receiver nominal controllers driven by the elected leader (each UAV uses ITS OWN view and leader).

  LF     leader-follower: follower tracks its slot relative to the leader it acts on
         u_i = -kp (p_i - p_L - s(d_i - d_L)) - kv (v_i - v_L);  leader: u_L = kvm (v_cmd - v_L)
  VS     virtual structure: every UAV tracks c_ref + s d_i with velocity v_cmd (c_ref/v_cmd from the leader)
  CONS   relative-position consensus on the formation + velocity tracking of the leader's mission command
  FLOCK  Olfati-Saber alpha-lattice (spacing s*d, no slots) + navigational feedback to the leader's command
A UAV with no valid leader holds formation (consensus, zero mission velocity).
With bounded authority (gate), followers' LF/VS slot targets are additionally clipped to within r_auth of
the plan-consistent target.
"""
from __future__ import annotations

import numpy as np

CONTROLLERS = ("LF", "VS", "CONS", "FLOCK")


def _cons(i, own_p, own_v, P, V, offsets, scale, use, kp, kv):
    js = np.nonzero(use[i])[0]
    if len(js) == 0:
        return -kv * own_v[i]
    e = (own_p[i] - P[i, js]) - scale[i] * (offsets[i] - offsets[js])
    ve = own_v[i] - V[i, js]
    return -kp * e.mean(0) - kv * ve.mean(0)


def _sig(z, eps=0.1):
    return (np.sqrt(1.0 + eps * z * z) - 1.0) / eps


def _flock(i, own_p, own_v, P, V, use, d, r):
    js = np.nonzero(use[i])[0]
    u = np.zeros(3)
    if len(js) == 0:
        return u
    eps, a, b, h = 0.1, 5.0, 5.0, 0.9
    c = abs(a - b) / np.sqrt(4 * a * b)
    diff = P[i, js] - own_p[i]
    dist = np.linalg.norm(diff, axis=1)
    z = _sig(dist)
    nij = diff / np.sqrt(1.0 + eps * dist * dist)[:, None]
    r_a, d_a = _sig(r), _sig(d)
    zz = z / r_a
    bump = np.where(zz < h, 1.0, np.where(zz < 1.0, 0.5 * (1 + np.cos(np.pi * (zz - h) / (1 - h))), 0.0))
    s1 = (z - d_a) + c
    phi = 0.5 * ((a + b) * s1 / np.sqrt(1 + s1 * s1) + (a - b))
    # gains re-tuned for metre-level nav error (the Layer-1 C4 gains c1=0.6 assume 3 cm noise and saturate
    # the 4 m/s^2 actuator here: |u| 7-9 m/s^2 with the fleet's 0.08-0.4 m nav bias)
    u += 0.15 * ((bump * phi)[:, None] * nij).sum(0)
    u += 1.0 * (bump[:, None] * (V[i, js] - own_v[i])).sum(0) / max(len(js), 1)
    return u


def nominal(kind, own_p, own_v, P, V, fresh, alive, led_by, v_used, cref_used, scale, offsets, spacing,
            radius=8.0, kp=0.9, kv=1.4, kvm=1.0, gate=False, r_auth=0.45, z_ref=-5.0, shield=False, shield_r=1.5, v2=False, v2_kp=0.3, v2_kv=1.8):
    n = own_p.shape[0]
    u = np.zeros((n, 3))
    D = np.linalg.norm(own_p[:, None, :] - P, axis=2)
    use = fresh & alive[None, :] & (D <= radius)
    np.fill_diagonal(use, False)
    for i in range(n):
        if not alive[i]:
            continue
        L = int(led_by[i])
        vm = v_used[i]
        if L < 0:
            u[i] = _cons(i, own_p, own_v, P, V, offsets, scale, use, kp, kv) - kvm * own_v[i]
            continue
        if kind == "LF":
            if L == i:
                u[i] = kvm * (vm - own_v[i]) + 0.3 * _cons(i, own_p, own_v, P, V, offsets, scale, use, kp, 0.0)
            else:
                tgt = P[i, L] + scale[i] * (offsets[i] - offsets[L])
                v_ref = V[i, L]
                # r_auth = (1.8 / 1.2) * delta_max: the leader's body can bias the slot loop by at most delta_max
                # of velocity, so with the command clip the follower's total authority deviation is <= 2 delta_max
                if gate and np.isfinite(cref_used[i, 0]):
                    plan_t = np.r_[cref_used[i], P[i, L, 2]] + scale[i] * offsets[i]
                    dd = tgt - plan_t
                    nd = np.linalg.norm(dd[:2])
                    if nd > r_auth:
                        tgt = plan_t + dd * (r_auth / nd)
                    # bounded authority covers the velocity reference too: the follower damps toward its own
                    # bounded command, not toward the leader's actual (unbounded) velocity
                    v_ref = np.r_[vm[:2], V[i, L, 2]]
                elif shield and np.isfinite(cref_used[i, 0]):
                    # conventional shield, body channel: the leader's body is accepted as the slot reference only while
                    # it is within shield_r of the plan-consistent slot (else the plan-consistent slot is used)
                    plan_t = np.r_[cref_used[i], P[i, L, 2]] + scale[i] * offsets[i]
                    dd = tgt - plan_t
                    if np.linalg.norm(dd[:2]) > shield_r:
                        tgt = plan_t
                        v_ref = np.r_[vm[:2], V[i, L, 2]]
                u[i] = -1.2 * (own_p[i] - tgt) - 1.8 * (own_v[i] - v_ref) + 0.3 * _cons(
                    i, own_p, own_v, P, V, offsets, scale, use, kp, 0.0)
        elif kind == "VS":
            c = cref_used[i] if np.isfinite(cref_used[i, 0]) else own_p[i, :2] - scale[i] * offsets[i, :2]
            tgt = np.r_[c, own_p[i, 2]] + scale[i] * offsets[i]
            tgt[2] = own_p[i, 2] if not np.isfinite(P[i, L, 2]) else P[i, L, 2]
            if v2:   # CADLE_V2 interface: reduced anchor-position gain (kp2/kv2 << 1) so that anchor error enters the velocity loop at (kp2/kv2) x error
                u[i] = -v2_kp * (own_p[i] - tgt) - v2_kv * (own_v[i] - vm)
            else:
                u[i] = -1.0 * (own_p[i] - tgt) - 1.8 * (own_v[i] - vm)
        elif kind == "CONS":
            u[i] = _cons(i, own_p, own_v, P, V, offsets, scale, use, kp, kv) + kvm * (vm - own_v[i])
        elif kind == "FLOCK":
            u[i] = _flock(i, own_p, own_v, P, V, use, spacing * scale[i], min(1.5 * spacing * scale[i], radius)) \
                   + kvm * (vm - own_v[i])
        else:
            raise ValueError(kind)
    u[:, 2] = -1.0 * (own_p[:, 2] - z_ref) - 1.4 * own_v[:, 2]          # altitude hold (planar missions)
    return u
