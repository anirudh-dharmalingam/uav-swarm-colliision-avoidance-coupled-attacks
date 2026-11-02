#!/usr/bin/env python3
"""Generate the coupled-safety inspection-yard Gazebo worlds (60 x 40 m, flight altitude 5 m).

Two products, built from the PX4 default world (`worlds/coupled_safety/base/default_px4.sdf`, world name stays "default" so every existing
topic /world/default/... and the evaluator/FANET/GCS keep working):

  yard_physics.sdf   confirmatory-run world: ground + PX4 physics UNCHANGED, static COLLIDABLE buildings/racks/fences placed outside
                     the swept flight envelope of every preregistered heading / spawn perturbation / slot shift (>= 4 m lateral),
                     non-collidable launch-pad paint.
  yard_h<deg>.sdf    the same world PLUS a scenario overlay for one route heading: floor-painted guidance lane (outer width = formation
                     lateral span + 4 m, middle-third width = 0.8 * span + 4 m), start/goal gates, inspection-staging platform and an
                     illuminated route, ALL visual-only (no <collision>, no inertia, static).  Used for video reruns; scene_equivalence()
                     proves the collision/physics blocks are identical to yard_physics.sdf.

Mission-relative coordinates (origin = formation centroid at take-off, +x = heading 0) are mapped to the Gazebo world by
`to_world(p, origin, heading)`; nothing here hard-codes the spawn origin (cameras are produced from the same transform).

    python3 scripts/demo/make_coupled_scene.py --out worlds/coupled_safety [--headings -20,-10,0,10,20] [--check]
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "swarm"))

ALT = 5.0
ROUTE_M = 24.0
SQUEEZE = 0.8
LANE_MARGIN = 4.0
CLEAR = 4.0                    # required lateral clearance between the swept flight envelope and any collidable object
HEADING_MAX = 20.0
JITTER = 0.25
SLOT_SHIFT_MAX = 3.0           # E3 slot shift bound used for the envelope (frozen value is checked against it by the builder)
TRACK_MARGIN = 0.0             # nominal swept envelope = path + lattice + jitter + slot shift; realised excursions are checked post hoc (AUDIT)
YARD = (-15.0, 45.0, -20.0, 20.0)   # x0, x1, y0, y1 (60 x 40 m)


def lattice(n=6, spacing=6.0):
    from experiment_manager import generate_lattice
    return np.array(generate_lattice(n, spacing, 1.0)[0], float)[:, :2]


def to_world(p, origin=(0.0, 0.0), heading_rad=0.0):
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    return (origin[0] + c * p[0] - s * p[1], origin[1] + s * p[0] + c * p[1])


def lateral_span(offsets, heading_rad):
    n = np.array([-math.sin(heading_rad), math.cos(heading_rad)])
    z = offsets @ n
    return float(z.max() - z.min())


def swept_envelope(offsets, headings_deg, jitter=JITTER, slot=SLOT_SHIFT_MAX, margin=TRACK_MARGIN):
    """Axis-aligned bounds of every UAV position over every preregistered heading (centroid path + unrotated lattice offsets)."""
    xs, ys = [], []
    for h in headings_deg:
        th = math.radians(h)
        for t in np.linspace(0, ROUTE_M, 25):
            c = np.array([t * math.cos(th), t * math.sin(th)])
            for o in offsets:
                xs.append(c[0] + o[0]); ys.append(c[1] + o[1])
    pad = jitter + slot + margin
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def box(name, x, y, z, sx, sy, sz, rgba, collide=True, yaw=0.0, emissive=None):
    mat = f"<ambient>{rgba}</ambient><diffuse>{rgba}</diffuse>" + (f"<emissive>{emissive}</emissive>" if emissive else "")
    col = (f'<collision name="c"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>' if collide else "")
    return (f'<model name="{name}"><static>true</static><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.5f}</pose><link name="l">'
            f'{col}<visual name="v"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry><material>{mat}</material></visual></link></model>')


def cyl(name, x, y, z, r, h, rgba, emissive=None):
    mat = f"<ambient>{rgba}</ambient><diffuse>{rgba}</diffuse>" + (f"<emissive>{emissive}</emissive>" if emissive else "")
    return (f'<model name="{name}"><static>true</static><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose><link name="l">'
            f'<visual name="v"><geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry><material>{mat}</material></visual></link></model>')


def sphere(name, x, y, z, r, rgba, emissive):
    mat = f"<ambient>{rgba}</ambient><diffuse>{rgba}</diffuse><emissive>{emissive}</emissive>"
    return (f'<model name="{name}"><static>true</static><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose><link name="l">'
            f'<visual name="v"><geometry><sphere><radius>{r}</radius></sphere></geometry><material>{mat}</material></visual></link></model>')


def static_yard(env):
    """Collidable structures outside `env` + CLEAR; returns (xml list, list of (name, (x0,x1,y0,y1)) collision footprints)."""
    ex0, ex1, ey0, ey1 = env
    items, fp = [], []

    def add(name, x, y, sx, sy, sz, rgba):
        items.append(box(name, x, y, sz / 2, sx, sy, sz, rgba))
        fp.append((name, (x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2)))
    depth = 2.0
    yN = ey1 + CLEAR + depth / 2                    # building centre lines: inner face = envelope + 4 m
    yS = ey0 - CLEAR - depth / 2
    for k, x in enumerate((-4.0, 16.0, 34.0)):
        add(f"hangar_N{k}", x, yN, 12.0, depth, 6.0, "0.55 0.58 0.62 1")
    for k, x in enumerate((-3.0, 6.0, 15.0, 24.0)):
        add(f"rack_S{k}", x, yS, 6.0, depth, 3.0, "0.70 0.45 0.20 1")
    xE = ex1 + CLEAR + 0.2
    add("fence_E", xE + 0.1, 0.0, 0.2, yN - yS - 2 * depth, 2.5, "0.40 0.40 0.45 1")
    xW = ex0 - CLEAR - 0.2
    add("fence_W", xW - 0.1, 0.0, 0.2, yN - yS - 2 * depth, 2.5, "0.40 0.40 0.45 1")
    return items, fp


def pads(offsets):
    return [cyl(f"pad{i}", float(o[0]), float(o[1]), 0.005, 0.9, 0.01, "0.95 0.85 0.10 1") for i, o in enumerate(offsets)]


def overlay(offsets, heading_deg):
    """Visual-only scenario overlay for one heading (mission frame rotated by the heading about the take-off centroid)."""
    th = math.radians(heading_deg)
    span = lateral_span(offsets, th)
    w_out, w_mid = span + LANE_MARGIN, SQUEEZE * span + LANE_MARGIN
    m = []
    third = ROUTE_M / 3.0
    segs = [(0.0, third, w_out, "lane_a"), (third, 2 * third, w_mid, "lane_b"), (2 * third, ROUTE_M, w_out, "lane_c")]
    for a, b, w, nm in segs:
        L = b - a
        cx, cy = to_world((a + L / 2, 0.0), heading_rad=th)
        for sgn, side in ((1, "L"), (-1, "R")):
            ex, ey = to_world((a + L / 2, sgn * w / 2), heading_rad=th)
            m.append(box(f"{nm}_{side}", ex, ey, 0.011, L, 0.15, 0.02, "1.0 1.0 1.0 1", collide=False, yaw=th))
    # lane narrowing diagonals (smooth entry / exit visual) and centre dashes
    for k in range(int(ROUTE_M // 2)):
        x = 1.0 + 2.0 * k
        cx, cy = to_world((x, 0.0), heading_rad=th)
        m.append(box(f"dash{k}", cx, cy, 0.011, 0.9, 0.12, 0.02, "0.2 0.8 1.0 1", collide=False, yaw=th, emissive="0.1 0.5 0.8 1"))
    # illuminated route: emissive beads along the route at flight altitude
    for k in range(25):
        x, y = to_world((k * ROUTE_M / 24.0, 0.0), heading_rad=th)
        m.append(sphere(f"bead{k}", x, y, ALT, 0.10, "1 0.9 0.2 1", "1 0.8 0.1 1"))
    # start / goal gates (two posts + beam, centred on the lane, beam above the flight altitude)
    for nm, s in (("start", 0.0), ("goal", ROUTE_M)):
        for sgn in (1, -1):
            px, py = to_world((s, sgn * (w_out / 2 + 0.6)), heading_rad=th)
            m.append(box(f"gate_{nm}_post{'L' if sgn > 0 else 'R'}", px, py, 4.0, 0.25, 0.25, 8.0, "0.9 0.3 0.1" + " 1", collide=False))
        bx, by = to_world((s, 0.0), heading_rad=th)
        m.append(box(f"gate_{nm}_beam", bx, by, 8.0, 0.25, w_out + 1.5, 0.25, "0.9 0.3 0.1 1", collide=False, yaw=th))
    # inspection-staging platform under the goal (visual-only slab)
    gx, gy = to_world((ROUTE_M, 0.0), heading_rad=th)
    m.append(box("staging_platform", gx, gy, 0.03, 9.0, 9.0, 0.06, "0.25 0.5 0.35 1", collide=False, yaw=th))
    return m, {"heading_deg": heading_deg, "lateral_span_m": span, "outer_width_m": w_out, "middle_third_width_m": w_mid}


def camera_model(name, topic, pos, target, hfov, w=1920, h=1080, rate=30.0):
    """Static camera (no collision, no inertia) at world `pos` looking at world `target` (yaw/pitch derived from the two points, so nothing assumes a
    spawn origin).  The sensors system is already loaded by PX4's server.config; a camera without subscribers costs nothing."""
    dx, dy, dz = target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, math.hypot(dx, dy))                  # gz pitch > 0 looks down
    return (f'<model name="{name}"><static>true</static><pose>{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f} 0 {pitch:.5f} {yaw:.5f}</pose><link name="l">'
            f'<sensor name="cam" type="camera"><camera><horizontal_fov>{hfov}</horizontal_fov><image><width>{w}</width><height>{h}</height></image>'
            f'<clip><near>0.1</near><far>300</far></clip></camera><always_on>0</always_on><update_rate>{rate}</update_rate><visualize>false</visualize>'
            f'<topic>{topic}</topic></sensor></link></model>')


def video_cameras(heading_deg, w=1920, h=1080, rate=30.0):
    """Two fixed views in MISSION-RELATIVE coordinates mapped to the world through the route heading: a fixed oblique view containing the whole
    encounter (route 0..24 m + formation) and a top-down geometry view."""
    th = math.radians(heading_deg)
    mid = to_world((ROUTE_M / 2, 0.0), heading_rad=th)
    ob = to_world((-7.0, -16.0), heading_rad=th)
    return [camera_model("cam_oblique", "/cs/cam_oblique", (ob[0], ob[1], 13.0), (mid[0], mid[1], 3.0), 1.15, w, h, rate),
            camera_model("cam_top", "/cs/cam_top", (mid[0], mid[1], 34.0), (mid[0], mid[1], 0.0), 0.95, w, h, rate)]


def build_world(base_sdf: str, extra_xml: list) -> str:
    i = base_sdf.rindex("</world>")
    return base_sdf[:i] + "\n    " + "\n    ".join(extra_xml) + "\n  " + base_sdf[i:]


def collision_signature(sdf: str):
    """Everything that can influence physics: <physics>, gravity, and every <collision> block (with its model pose)."""
    phys = re.findall(r"<physics.*?</physics>", sdf, flags=re.S) + re.findall(r"<gravity>.*?</gravity>", sdf)
    models = re.findall(r"<model name=\"([^\"]+)\">(.*?)</model>", sdf, flags=re.S)
    col = []
    for name, body in models:
        if "<collision" in body:
            pose = re.search(r"<pose>(.*?)</pose>", body).group(1)
            col.append((name, pose, tuple(re.findall(r"<collision.*?</collision>", body, flags=re.S))))
    return phys, sorted(col)


def scene_equivalence(a: str, b: str):
    pa, ca = collision_signature(a)
    pb, cb = collision_signature(b)
    return pa == pb and ca == cb


def check_clearance(fp, env):
    ex0, ex1, ey0, ey1 = env
    worst = math.inf
    for name, (x0, x1, y0, y1) in fp:
        dx = max(x0 - ex1, ex0 - x1, 0.0)
        dy = max(y0 - ey1, ey0 - y1, 0.0)
        worst = min(worst, math.hypot(dx, dy) if (dx > 0 and dy > 0) else max(dx, dy))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="worlds/coupled_safety")
    ap.add_argument("--headings", default="-20,-10,0,10,20", help="overlay worlds to write (video reruns); the physics world covers [-20,20]; any decimal heading is allowed")
    ap.add_argument("--check", action="store_true", help="exit non-zero if a clearance/equivalence check fails")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--spacing", type=float, default=6.0)
    a = ap.parse_args()
    out = Path(a.out)
    base = (out / "base" / "default_px4.sdf").read_text()
    offs = lattice(a.n, a.spacing)
    env = swept_envelope(offs, np.linspace(-HEADING_MAX, HEADING_MAX, 41))
    items, fp = static_yard(env)
    clear = check_clearance(fp, env)
    phys = build_world(base, items + pads(offs))
    (out / "yard_physics.sdf").write_text(phys)
    report = {"yard_m": [60, 40], "altitude_m": ALT, "swept_envelope_xyxy": env, "min_clearance_to_collidable_m": clear,
              "required_clearance_m": CLEAR, "collidable_models": [n for n, _ in fp], "overlays": {}}
    ok = clear >= CLEAR - 1e-9
    for h in [float(x) for x in a.headings.split(",") if x != ""]:
        ov, dims = overlay(offs, h)
        w = build_world(phys, ov)
        tag = f"{int(h):+d}" if h == int(h) else f"{h:+.2f}"
        name = f"yard_h{tag}.sdf"
        (out / name).write_text(w)
        eq = scene_equivalence(phys, w)
        n_col = len(re.findall(r"<collision", "".join(ov)))
        wv = build_world(w, video_cameras(h))
        vname = f"video_h{tag}.sdf"
        (out / vname).write_text(wv)
        eqv = scene_equivalence(phys, wv)
        report["overlays"][name] = {**dims, "physics_collision_equivalent": eq, "overlay_collision_elements": n_col,
                                    "video_world": vname, "video_world_physics_equivalent": eqv}
        ok &= eqv
        ok &= eq and n_col == 0
    (out / "scene_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if a.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
