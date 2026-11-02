"""Per-UAV mission step shared by the Layer-1 simulator and the PX4 per-vehicle node.

For node i (after the election step) this decides what i acts on:
  leader   : plan progress k (advanced on its own observed formation centre), reference c_ref = observed
             centre, command v = plan(k, c_ref); an insider adversary may override what it broadcasts/flies
  follower : reads its accepted leader's (k, c_ref, v_cmd, scale) from the leader's latest packet,
             updates the plan-consistency CUSUM once per new packet (impeach), and with bounded authority
             (gate) uses  plan(k, c_used) + clip(v_cmd - plan(k, c_used), delta_max)  where
             c_used = c_obs + clip(c_ref - c_obs, tol_c)
  leaderless: hover (v = 0, no reference)
`c_obs` = median over the claims i currently uses (fresh, alive, not distrusted) of (p_j - s d_j).
"""
from __future__ import annotations

import numpy as np

from .protocol import Fld


def observed_centre(V_i, use_i, offsets, scale, own_p_i):
    js = np.nonzero(use_i)[0]
    if not len(js):
        return own_p_i[:2].copy()
    return np.median(V_i[js, Fld.PX:Fld.PY + 1] - scale * offsets[js, :2], axis=0)


def node_step(el, i, V, use_i, offsets, plan, s_prev, t, t_boot, gate, impeach, t_tx_i, own_p_i, adv=None, shield=False, local_wp=False, leader_la=False):
    """Returns (v_fly (3,), scale) and updates el.wp/cref/v_used/scale_used/cref_used for node i."""
    p = el.p
    L = int(el.led_by[i])
    cobs = observed_centre(V[i], use_i, offsets, s_prev, own_p_i)
    el.cobs[i] = cobs
    if L == i:
        cr = cobs
        if leader_la and t >= t_boot:
            # v2: leader and followers share ONE arrival radius (plan.tol + tol_c); mixed radii desynchronise the indices (development defect)
            k0_ = int(el.wp[i])
            kk = k0_ + 1 if (k0_ < len(plan.waypoints) and np.linalg.norm(plan.waypoints[k0_] - cr) < plan.tol + el.p.tol_c) else k0_
        else:
            kk = plan.advance(int(el.wp[i]), cr) if t >= t_boot else int(el.wp[i])
        vc, sc = plan.command(kk, cr)
        if t < t_boot:
            vc = np.zeros(3)
        v_fly = vc
        if adv is not None and i == adv.id:
            cr, vc, kk, sc, v_fly = adv.leader_override(t, el, cr, vc, kk, sc, cobs)
        el.wp[i] = kk
        el.cref[i] = cr
        el.v_used[i] = vc
        el.scale_used[i] = sc
        el.cref_used[i] = cr
        return np.asarray(v_fly, float), float(sc)
    if L >= 0:
        kL = int(V[i, L, Fld.WP])
        crL = V[i, L, Fld.CRX:Fld.CRY + 1]
        cmd = np.r_[V[i, L, Fld.CMDX:Fld.CMDY + 1], 0.0]
        sL = float(V[i, L, Fld.SCALE])
        # the plan-consistency statistic is always computed and logged (calibration/audit); it only ACTS
        # through the CUSUM when the scheme impeaches
        if V[i, L, Fld.ROLE] == 2:                     # only packets in which L claims leadership are commands
            el.update_cusum(i, L, V, cobs, float(t_tx_i[L]))
        if gate:
            dc = crL - cobs
            nd = np.linalg.norm(dc)
            cr_used = cobs + (dc * min(1.0, p.tol_c / nd) if nd > 1e-9 else dc)
            # waypoint authority is bounded too: the follower moves to the next segment only when its own
            # bounded reference has reached the current waypoint (no corner cutting by index skipping)
            k0 = int(el.wp[i])
            if local_wp:
                # CADLE_LA (resilience campaign): advance on this UAV's OWN physical arrival, no leader announcement needed
                if k0 < len(plan.waypoints) and np.linalg.norm(plan.waypoints[k0] - cr_used) < plan.tol + p.tol_c:
                    el.wp[i] = k0 + 1
            elif kL == k0 + 1 and k0 < len(plan.waypoints) and \
                    np.linalg.norm(plan.waypoints[k0] - cr_used) < plan.tol + p.tol_c:
                el.wp[i] = kL
            vp, sp_ = plan.command(int(el.wp[i]), cr_used)
            dv = cmd - vp
            ndv = np.linalg.norm(dv)
            el.v_used[i] = vp + (dv * min(1.0, p.delta_max / ndv) if ndv > 1e-9 else dv)
            el.scale_used[i] = sp_ if abs(sL - sp_) > 0.05 else sL
            # the slot loops use the reference as a POSITION anchor, where a tol_c offset would act as a steady
            # velocity bias of (k_p/k_v) tol_c >> delta_max; the anchor gets its own, smaller budget r_anchor
            el.cref_used[i] = cobs + (dc * min(1.0, p.r_anchor / nd) if nd > 1e-9 else dc)
        elif shield:
            # conventional plan/command shield (resilience campaign, baseline C): the packet is accepted verbatim iff
            # every channel is inside the monitor's frozen benign-calibrated tolerance and the index does not jump;
            # otherwise this UAV executes its LOCAL public plan for the packet.  The waypoint advances on this UAV's
            # own physical arrival only (no leader announcement is needed or trusted).
            p = el.p
            k0 = int(el.wp[i])
            _, cr_l, v_pl, s_pl = el._expected(i, L, V, float(t_tx_i[L]))
            raw = (float(np.linalg.norm(cmd[:2] - v_pl[:2])), float(np.linalg.norm(crL - cobs)),
                   float(plan.cross_track(np.r_[crL, 0.0], kL)), abs(sL - s_pl))
            tol_ref = p.shield_tight_r if el.S.get("shield_tight") else p.tol_c
            ok = (raw[0] <= p.tol_v and raw[1] <= tol_ref and raw[2] <= p.tol_xt and raw[3] <= 0.05 and kL <= k0 + 1)
            if k0 < len(plan.waypoints) and np.linalg.norm(plan.waypoints[k0] - cobs) < plan.tol + p.tol_c:
                el.wp[i] = k0 + 1       # same guard radius as the CADLE gate (a stricter radius desynchronises k from the leader)
            if ok:
                el.v_used[i] = np.r_[cmd[:2], 0.0] if cmd.shape[0] == 3 else cmd
                el.scale_used[i] = sL
                el.cref_used[i] = crL
            else:
                vp, sp_ = plan.command(int(el.wp[i]), cobs)
                el.v_used[i] = vp
                el.scale_used[i] = sp_
                el.cref_used[i] = cobs
            el.shield_reject[i] += int(not ok)
        else:
            el.wp[i] = kL
            el.v_used[i] = cmd
            el.scale_used[i] = sL
            el.cref_used[i] = crL
        el.cref[i] = el.cref_used[i]
        return el.v_used[i].copy(), float(el.scale_used[i])
    el.v_used[i] = 0.0
    el.cref_used[i] = np.nan
    return np.zeros(3), float(el.scale_used[i])
