"""Experimental INTERVENTIONS for the resilience campaign (shared by the Layer-1 simulator and the PX4 agent node).

These are NOT part of any defence: they are the experimenter's way of putting a fixed compromised UAV into office so that
containment (what an accepted insider can do) can be measured separately from admission (whether it is elected).

install(): at the intervention instant every UAV's local protocol state is set as if the attacker had just won a term-T_INST
           election: the attacker holds the leader role; every other UAV has the attacker as its accepted leader of term T_INST
           and has itself seen q signed votes for it (synthetic, receiver-local evidence).  Row-wise: the PX4 node applies it to
           its own row only, Layer 1 loops over all rows.
hold():    while the hold lasts, a UAV cannot withdraw from the installed leader (impeachment, lease loss and handover are
           overridden after every protocol tick).  Everything else (trust, sensing, gating, command shaping, control) runs
           unchanged.  The hold turns detection-based removal off in ALL arms alike, so the remaining differences measure
           authority limiting, not detection.
Schemes without a leader role (LOCAL) are unaffected: they have no authority to install, which is reported as such.
"""
from __future__ import annotations

import numpy as np

from .protocol import ROLE_F, ROLE_L

T_INST = 1000


def has_leader_role(el) -> bool:
    return el.kind == "quorum"


def install(el, rows, att, t):
    if not has_leader_role(el):
        return False
    for i in rows:
        q = el.quorum(i)
        el.obs[i][(T_INST, att)] = set(range(q))                      # q distinct signed votes seen locally
        el.term[i] = el.cterm[i] = T_INST
        el.vote[i] = att
        el.vterm[i] = T_INST
        el.cusum[i] = 0.0
        el.xfer[i] = -1
        el.rejected[i, att] = -1
        el.t_last_hb[i] = t
        el.cov_since[i] = np.inf
        el.better_since[i] = np.inf
        if i == att:
            el.role[i], el.leader[i] = ROLE_L, att
        else:
            el.role[i], el.leader[i] = ROLE_F, att
    return True


def hold(el, rows, att, t, t_rx=None, silent_s=None):
    """silent_s (v2): a receiver stops holding once the held leader has been silent (no packet received) for silent_s seconds, e.g. because it landed
    for energy -- a fail-silent insider is not in office.  Default None reproduces the v1 behaviour (hold regardless)."""
    if not has_leader_role(el):
        return
    for i in rows:
        if silent_s is not None and t_rx is not None and i != att and (t - float(t_rx[i, att])) > silent_s:
            continue
        el.rejected[i, att] = -1
        el.t_last_hb[i] = t
        if i == att:
            el.role[i], el.leader[i], el.led_by[i] = ROLE_L, att, att
        else:
            el.role[i], el.leader[i], el.led_by[i] = ROLE_F, att, att


def insider_flag(attacker_id, attack, res_mode, insider_attacks):
    """May a process turn itself into the adversary when it happens to lead?  Legacy insider threats re-target to whoever leads
    (attacker id -2).  In the resilience campaign the attacker is the FIXED manifest identity: never re-targeted."""
    return (attacker_id == -2) if res_mode else (attacker_id == -2 or attack in insider_attacks)
