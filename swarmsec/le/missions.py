"""Leader-executed missions with a PUBLIC plan (shared before take-off).

The leader owns mission progress (waypoint index k) and broadcasts the mission command
(v_cmd, spacing scale s).  Because the plan is public, any follower can recompute the plan-consistent
command from (k, the leader's position) and check the leader against it (swarmsec.le.protocol).

Missions (world frame, start centroid at the origin, units m):
  search    boustrophedon sweep of a W x H area; outcome = fraction of 2 m cells imaged (camera radius 3 m)
  patrol    square loop, two laps; outcome = waypoints reached
  corridor  straight transit through a gate; outside the gate width at the gate is a wall (keep-out);
            formation squeezes to `squeeze` while centre is in the gate
  delivery  multi-leg route that skirts a circular no-fly zone (NFZ); outcome = arrival, NFZ intrusion

Keep-out regions are axis-aligned boxes or circles; the steering adversary aims at the nearest one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MISSIONS = ("search", "patrol", "corridor", "delivery")


@dataclass
class Plan:
    kind: str
    waypoints: np.ndarray                 # (K,2)
    cruise: float
    tol: float
    duration_s: float
    keepout_circles: list = field(default_factory=list)    # [(cx, cy, r)]
    keepout_boxes: list = field(default_factory=list)      # [(x0, x1, y0, y1)]
    gate: tuple | None = None             # (x0, x1, half_width) corridor gate
    squeeze: float = 1.0
    area: tuple | None = None             # search area (x0, x1, y0, y1)

    def path_length_from(self, k, c):
        if k >= len(self.waypoints):
            return 0.0
        L = float(np.linalg.norm(self.waypoints[k] - c[:2]))
        for a, b in zip(self.waypoints[k:-1], self.waypoints[k + 1:]):
            L += float(np.linalg.norm(b - a))
        return L

    def remaining_time(self, k, c):
        return self.path_length_from(k, c) / max(self.cruise, 1e-6)

    lookahead: float = 8.0

    def command(self, k, c):
        """Plan-consistent mission velocity (3,) and spacing scale for progress k and centre estimate c.
        Line following (pure pursuit): aim at the point `lookahead` metres ahead of c's projection on the
        current segment (prev waypoint -> waypoint k), never beyond waypoint k; speed slows near waypoint k."""
        if k >= len(self.waypoints):
            return np.zeros(3), self.spacing(c)
        wpk = self.waypoints[k]
        prev = np.zeros(2) if k == 0 else self.waypoints[k - 1]
        seg = wpk - prev
        Ls = float(np.linalg.norm(seg))
        c2 = np.asarray(c, float)[:2]
        if Ls > 1e-6:
            u = seg / Ls
            s_proj = float(np.clip(np.dot(c2 - prev, u), 0.0, Ls))
            aim = prev + min(Ls, s_proj + self.lookahead) * u
        else:
            aim = wpk
        d = aim - c2
        nd = float(np.linalg.norm(d))
        dw = float(np.linalg.norm(wpk - c2))
        if nd < 1e-6:
            return np.zeros(3), self.spacing(c)
        sp = self.cruise * min(1.0, dw / 6.0)
        return np.array([d[0] / nd * sp, d[1] / nd * sp, 0.0]), self.spacing(c)

    def advance(self, k, c):
        """Waypoint index after checking arrival (one step at most per call)."""
        if k < len(self.waypoints) and np.linalg.norm(self.waypoints[k] - c[:2]) < self.tol:
            return k + 1
        return k

    def spacing(self, c):
        if self.gate is None:
            return 1.0
        x0, x1, _ = self.gate
        ramp = 8.0
        x = float(c[0])
        if x0 - ramp <= x <= x1 + ramp:
            f = 1.0 if x0 <= x <= x1 else 1.0 - min(abs(x - x0), abs(x - x1)) / ramp
            return 1.0 - (1.0 - self.squeeze) * f
        return 1.0

    def in_keepout(self, p):
        """Boolean (N,) for points p (N,>=2) inside any keep-out region."""
        p = np.atleast_2d(p)
        m = np.zeros(len(p), bool)
        for cx, cy, r in self.keepout_circles:
            m |= np.hypot(p[:, 0] - cx, p[:, 1] - cy) < r
        for x0, x1, y0, y1 in self.keepout_boxes:
            m |= (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)
        return m

    def nearest_keepout_dir(self, c):
        """Unit xy direction from c to the nearest keep-out point (steering adversary's aim)."""
        best, bd = None, np.inf
        for cx, cy, r in self.keepout_circles:
            d = np.array([cx, cy]) - c[:2]
            dist = np.linalg.norm(d) - r
            if dist < bd:
                bd, best = dist, d
        for x0, x1, y0, y1 in self.keepout_boxes:
            q = np.array([np.clip(c[0], x0, x1), np.clip(c[1], y0, y1)])
            d = q - c[:2]
            if np.linalg.norm(d) < bd:
                bd, best = np.linalg.norm(d), d
        if best is None:
            return np.array([0.0, 1.0, 0.0])
        nb = max(np.linalg.norm(best), 1e-9)
        return np.array([best[0] / nb, best[1] / nb, 0.0])

    def cross_track(self, c, k):
        """Distance of centre c from the planned polyline segment it is on (k = progress)."""
        pts = np.vstack([np.zeros(2), self.waypoints])
        k = int(np.clip(k, 0, len(self.waypoints) - 1))
        a, b = pts[k], pts[k + 1]
        ab = b - a
        t = float(np.clip(np.dot(c[:2] - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0))
        return float(np.linalg.norm(c[:2] - (a + t * ab)))


def make_plan(kind: str, footprint: float, cruise: float = 1.5, scale: float = 1.0) -> Plan:
    """Mission plan scaled to the swarm footprint (diameter of the formation, m).  `scale` shrinks every
    route length / keep-out position along the route (PX4 flights use 0.5); clearances to the formation stay
    footprint-based."""
    if scale != 1.0:
        return _scaled(make_plan(kind, footprint, cruise), scale, footprint)
    f = max(footprint, 8.0)
    if kind == "search":
        W = 70.0
        lane = 1.2 * f
        rows = 3
        pts = []
        for r in range(rows):
            y = r * lane
            xs = (0.0, W) if r % 2 == 0 else (W, 0.0)
            pts += [(xs[0], y), (xs[1], y)]
        wp = np.array(pts[1:], float)
        L = W * rows + lane * (rows - 1)
        return Plan(kind, wp, cruise, 3.0, L / cruise * 1.5 + 30,
                    keepout_boxes=[(20.0, 45.0, -f - 14.0, -f - 2.0)],
                    area=(-f / 2, W + f / 2, -f / 2, (rows - 1) * lane + f / 2))
    if kind == "patrol":
        s = 50.0
        loop = [(s, 0.0), (s, s), (0.0, s), (0.0, 0.0)]
        wp = np.array(loop * 2, float)
        return Plan(kind, wp, cruise, 3.0, 8 * s / cruise * 1.5 + 30,
                    keepout_circles=[(s / 2, s / 2, 10.0)])
    if kind == "corridor":
        Lx = 130.0
        gate = (55.0, 80.0, 0.5 * f + 3.0)
        wp = np.array([(Lx, 0.0)], float)
        hw = gate[2]
        return Plan(kind, wp, cruise, 3.0, Lx / cruise * 1.5 + 30,
                    keepout_boxes=[(gate[0], gate[1], hw, hw + 40.0), (gate[0], gate[1], -hw - 40.0, -hw)],
                    gate=gate, squeeze=0.85)
    if kind == "delivery":
        wp = np.array([(60.0, 0.0), (60.0, 60.0), (0.0, 60.0)], float)
        r = 12.0
        cx = 30.0
        cy = f / 2 + r + 6.0
        return Plan(kind, wp, cruise, 3.0, 180.0 / cruise * 1.5 + 30,
                    keepout_circles=[(cx, cy, r)])
    if kind == "sharp":      # v2 held-out geometry: a 135-degree hairpin then a 90-degree turn (the delivery route has two 90-degree corners)
        wp = np.array([(60.0, 0.0), (20.0, 40.0), (60.0, 80.0)], float)
        L = float(np.linalg.norm(wp[0])) + float(np.linalg.norm(wp[1] - wp[0])) + float(np.linalg.norm(wp[2] - wp[1]))
        return Plan(kind, wp, cruise, 3.0, L / cruise * 1.5 + 30, keepout_circles=[(55.0, 40.0, 6.0)])
    raise ValueError(kind)


def _scaled(pl: Plan, a: float, footprint: float) -> Plan:
    """Shrink route geometry by a; keep-outs keep their size and their clearance from the route."""
    f = max(footprint, 8.0)
    wp = pl.waypoints * a
    if pl.kind == "search":
        wp[:, 1] = pl.waypoints[:, 1]                          # lanes stay footprint-spaced
    circ = []
    for cx, cy, r in pl.keepout_circles:
        if pl.kind == "delivery":
            Ly = float(wp[1, 1])                               # NFZ fits between the outbound and return legs
            r2 = float(np.clip((Ly - f) / 2.0 - 6.0, 2.0, r))
            circ.append((cx * a, f / 2 + r2 + 6.0, r2))        # same 6 m clearance from the first leg
        else:
            circ.append((cx * a, cy * a, r * a if pl.kind == "patrol" else r))
    boxes = []
    for x0, x1, y0, y1 in pl.keepout_boxes:
        boxes.append((x0 * a, x1 * a, y0, y1))                 # lateral clearance is footprint-based
    gate = None if pl.gate is None else (pl.gate[0] * a, pl.gate[1] * a, pl.gate[2])
    area = None
    if pl.area is not None:
        x0, x1, y0, y1 = pl.area
        area = (x0 * a, x1 * a, y0, y1)
    L = float(np.linalg.norm(wp[0])) + sum(float(np.linalg.norm(b - c)) for b, c in zip(wp[1:], wp[:-1]))
    return Plan(pl.kind, wp, pl.cruise, pl.tol, L / pl.cruise * 1.5 + 30, keepout_circles=circ,
                keepout_boxes=boxes, gate=gate, squeeze=pl.squeeze, area=area)


def rotate_plan(pl: Plan, theta: float) -> Plan:
    """The same mission rotated about the start centroid by theta (rad): waypoints and circular keep-outs rotate, so the
    route heading relative to the formation lattice, wind and sensing geometry changes (resilience campaign scenarios).
    Only plans whose geometry is rotation-covariant are supported (delivery, patrol); box keep-outs (search, corridor)
    are axis-aligned and are refused rather than silently mis-rotated."""
    if theta == 0.0:
        return pl
    if pl.keepout_boxes or pl.gate is not None or pl.area is not None:
        raise ValueError(f"plan '{pl.kind}' has axis-aligned geometry; rotation unsupported")
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    wp = pl.waypoints @ R.T
    circ = [(float(R[0, 0] * cx + R[0, 1] * cy), float(R[1, 0] * cx + R[1, 1] * cy), r) for cx, cy, r in pl.keepout_circles]
    out = Plan(pl.kind, wp, pl.cruise, pl.tol, pl.duration_s, keepout_circles=circ, keepout_boxes=[], gate=None,
               squeeze=pl.squeeze, area=None)
    out.lookahead = pl.lookahead
    return out
