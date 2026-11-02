"""Post-filter command chain of the PX4 agent, as a pure function so that it can be audited and tested.

Order (identical for every scheme): CBF/QP output u -> disturbance-observer feed-forward (u - d_hat) -> norm bound
(u_max + dob_max) -> jerk (slew) limit relative to the previous command over the elapsed sim time -> speed-envelope
projection (remove the outward component of the command when |v| > vmax).  The FINAL command is what is sent to PX4; the
formulation's theorem premises concern this command, not the pre-limiter nominal.  Every limiter reports whether it acted.
"""
from __future__ import annotations

import numpy as np


def shape_command(u_xy, u_prev_xy, d_hat, dob, v_xy, u_max, dob_max, slew, vmax):
    u_c = np.asarray(u_xy, float) - (d_hat if dob else 0.0)
    dob_out = u_c.copy()
    nu = float(np.linalg.norm(u_c))
    sat = nu > u_max + dob_max
    if sat:
        u_c = u_c * ((u_max + dob_max) / nu)
    du = u_c - u_prev_xy
    nd = float(np.linalg.norm(du))
    slw = nd > slew
    if slw:
        du = du * (slew / max(nd, 1e-9))
    uc = np.asarray(u_prev_xy, float) + du
    sp = float(np.linalg.norm(v_xy))
    spd = sp > vmax and float(uc @ np.asarray(v_xy, float)) > 0
    if spd:
        uc = uc - (uc @ v_xy) / (sp * sp) * np.asarray(v_xy, float)
    return uc, {"dob_out": dob_out, "sat": int(sat), "slw": int(slw), "spd": int(spd)}
