#!/usr/bin/env python3
"""Host-side RAL26 batch runner: one Docker container per PX4/Gazebo episode, K in parallel.

Each container gets its own network namespace, so Gazebo transport, uXRCE-DDS and PX4 UDP ports of
concurrent episodes are fully isolated; vehicle physics settings are never changed for throughput.
Inside a container: headless `gz sim -s`, MicroXRCEAgent, scripts/swarm/experiment_manager.py (RAL26
mode: one PX4 + X500 per UAV, one controller process per UAV, FANET bridge, GCS, isolated evaluator).

    python3 scripts/swarm/ral26_batch.py --spec experiments/ral26/hf0.json --workers 2

Spec JSON: {"tag": "...", "base": "config/swarm/ral26_base.toml", "episodes": [{"name": .., "overrides": {..}}]}
Results: results/ral26/<tag>/<name>/ (+ results/ral26/<tag>/index.jsonl, one row per episode).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN_CACHE = Path(os.environ.get("RAL_PX4_CACHE", Path.home() / "Documents/Disaster Swarm/.cache"))
IMAGE = os.environ.get("RAL_IMAGE", "dronelab:research-v1.17")

INNER = r"""
set +e
source ~/.dronelab_env >/dev/null 2>&1
cd /workspace && bash scripts/setup/px4_build.sh >/dev/null 2>&1
export GZ_SIM_RESOURCE_PATH=/opt/PX4-Autopilot/Tools/simulation/gz/models
export GZ_SIM_SERVER_CONFIG_PATH=/opt/PX4-Autopilot/Tools/simulation/gz/server.config
# per-episode isolation: concurrent containers share docker's bridge, so DDS and Gazebo transport
# discovery must be separated logically (unique domain / partition) and kept on loopback
export ROS_DOMAIN_ID={dom}
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export GZ_PARTITION=ral26_{dom}_{part}
export GZ_IP=127.0.0.1
mkdir -p {rd}/controller
WORLD=/opt/PX4-Autopilot/Tools/simulation/gz/worlds/default.sdf
{world_setup}
gz sim -s -r $WORLD > {rd}/controller/gz.log 2>&1 &
sleep 6
MicroXRCEAgent udp4 -p 8888 > /dev/null 2>&1 &
sleep 2
RAL_RUN_DIR={rd} RAL_INSTANCE_ROOT=/tmp/px4i timeout {tmo} python3 scripts/swarm/experiment_manager.py --config {rd}/config.toml > {rd}/controller/manager.log 2>&1
echo "manager_exit=$?" >> {rd}/controller/manager.log
pkill -f px4 ; pkill -f "gz sim" ; pkill -f MicroXRCEAgent
for f in {rd}/agents/*.csv; do [ -f "$f" ] && gzip -f "$f"; done
rm -rf {rd}/px4
chown -R {uid}:{gid} {rd}
"""


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def toml_dump(d, prefix=""):
    lines, tables = [], []
    for k, v in d.items():
        if isinstance(v, dict) and not (v and all(isinstance(x, dict) for x in v.values()) and False):
            if isinstance(v, dict):
                tables.append((k, v))
                continue
        lines.append(f"{k} = {json.dumps(v)}")
    out = ""
    if prefix and lines:
        out += f"[{prefix}]\n"
    out += "\n".join(lines) + ("\n" if lines else "")
    for k, v in tables:
        out += "\n" + toml_dump(v, f"{prefix}.{k}" if prefix else k)
    return out


def lattice(n, spacing):
    sys.path.insert(0, str(ROOT / "scripts" / "swarm"))
    from experiment_manager import generate_lattice
    return generate_lattice(n, spacing, 1.0)[0]


def central(offsets):
    import numpy as np
    o = np.asarray(offsets, float)
    return int(np.argmin(np.linalg.norm(o - o.mean(axis=0), axis=1)))


import queue
SLOTS: "queue.Queue[int]" = queue.Queue()
RTF_RERUN = None                         # spec key "rtf_rerun_below": re-run once below this RTF
RESULTS_ROOT = "results/ral26"          # spec key "results_root" overrides (leader-election study: results/le_px4)


def run_episode(tag, base, ep, cpus, timeout):
    slot = SLOTS.get()
    try:
        row = _run_episode(tag, base, ep, cpus, timeout, slot)
        rd = ROOT / RESULTS_ROOT / tag / ep["name"]
        if not row.get("skipped") and not (rd / "eval" / "summary.json").exists():
            # infrastructure start failure (e.g. PX4 exited before ROS discovery): keep the failed
            # attempt for the record and retry once; the index row records it
            import shutil
            fail = rd.parent / "_failed_attempts" / ep["name"]
            fail.parent.mkdir(exist_ok=True)
            shutil.rmtree(fail, ignore_errors=True)
            shutil.move(str(rd), str(fail))
            row = _run_episode(tag, base, ep, cpus, timeout, slot)
            row["retried_after_infra_failure"] = True
        elif RTF_RERUN is not None and not row.get("skipped") and (row.get("rtf_mean") or 0.0) < RTF_RERUN \
                and not row.get("retried_after_infra_failure"):
            # pre-registered RTF rule (PREREG_LE_ADDENDUM §5): an episode below the RTF threshold is re-run
            # once with the same seed; the slow attempt is kept for the record
            import shutil
            fail = rd.parent / "_rtf_attempts" / ep["name"]
            fail.parent.mkdir(exist_ok=True)
            shutil.rmtree(fail, ignore_errors=True)
            shutil.move(str(rd), str(fail))
            first = row.get("rtf_mean")
            row = _run_episode(tag, base, ep, cpus, timeout, slot)
            row["rerun_after_rtf"] = first
        return row
    finally:
        SLOTS.put(slot)


def randomize_scenario(cfg):
    """v2: the seed draws the SCENARIO (identical for every defence at that seed -> paired design).
    Before v2 nothing in an ideal-network episode depended on the seed (fixed geometry, central attacker),
    so 'seeds' were replicates differing only by SITL/OS timing jitter."""
    import math
    import random
    R = cfg["ral26"]
    spec = R.get("randomize")
    if not spec:
        return None
    n = int(cfg["swarm"]["num_uavs"])
    r = random.Random(1000003 * int(cfg["experiment"]["seed"]) + 17)
    L = math.hypot(*R.get("route", [20.0, 0.0])[:2])
    th = r.uniform(0.0, 2.0 * math.pi)
    draw = {"attacker_id": r.randrange(n), "route_heading_rad": th,
            "route": [L * math.cos(th), L * math.sin(th)],
            "attack_start_s": r.uniform(*spec.get("start_s", [8.0, 16.0])),
            "trigger_radius_m": r.uniform(*spec.get("trigger_radius_m", [6.0, 12.0])),
            "squeeze_t0": r.uniform(*spec.get("squeeze_t0", [6.0, 10.0]))}
    R["route"] = draw["route"]
    R["squeeze_t0"] = draw["squeeze_t0"]
    R.setdefault("attack", {})
    R["attack"]["start_s"] = draw["attack_start_s"]
    R["attack"]["trigger_radius_m"] = draw["trigger_radius_m"]
    if cfg["attack"]["id"] != "A0":
        R["attacker_id"] = draw["attacker_id"]
    R["scenario_draw"] = draw
    return draw


def world_setup(wind, rd):
    """Shell lines that build a wind-enabled `default` world (scripts/setup/gz_wind_world.py); '' = still air."""
    if not wind or float(wind.get("speed", 0.0)) <= 0.0:
        return ""
    keys = ("speed", "heading_deg", "gust_pct", "gust_period_s", "meander_deg", "turb_std", "k")
    args = " ".join(f"--{k.replace('_', '-')} {float(wind[k])}" for k in keys if k in wind)
    return ("python3 scripts/setup/gz_wind_world.py --out /tmp/gzw " + args + f" > {rd}/controller/wind.log 2>&1 && {{\n"
            "export GZ_SIM_RESOURCE_PATH=/tmp/gzw/models:$GZ_SIM_RESOURCE_PATH\n"
            "export GZ_SIM_SERVER_CONFIG_PATH=/tmp/gzw/server.config\n"
            "WORLD=/tmp/gzw/worlds/default.sdf; }")


def _run_episode(tag, base, ep, cpus, timeout, slot):
    cfg = deep_update(copy.deepcopy(base), ep.get("overrides", {}))
    n = int(cfg["swarm"]["num_uavs"])
    draw = randomize_scenario(cfg)
    if "le" not in cfg and (cfg.get("ral26", {}).get("attacker_id", -1) == "central" or (
            cfg["attack"]["id"] != "A0" and int(cfg.get("ral26", {}).get("attacker_id", -1)) < 0)):
        cfg["ral26"]["attacker_id"] = central(lattice(n, float(cfg["formation"]["spacing_m"])))
    rd_host = ROOT / RESULTS_ROOT / tag / ep["name"]
    if (rd_host / "eval" / "summary.json").exists():
        return {"name": ep["name"], "skipped": True}
    rd_host.mkdir(parents=True, exist_ok=True)
    (rd_host / "config.toml").write_text(toml_dump(cfg))
    rd = f"/workspace/{RESULTS_ROOT}/{tag}/{ep['name']}"
    inner = INNER.format(rd=rd, tmo=int(timeout), uid=os.getuid(), gid=os.getgid(), dom=10 + slot,
                         part=abs(hash(ep["name"])) % 100000, world_setup=world_setup(cfg.get("wind"), rd))
    cname = f"ral26_{tag}_{ep['name']}".replace(".", "_")[:120]
    t0 = time.time()
    cmd = ["docker", "run", "--rm", "--name", cname, f"--cpus={cpus}", "--shm-size=512m",
           "-v", f"{ROOT}:/workspace", "-v", f"{MAIN_CACHE}:/workspace/.cache", IMAGE, "bash", "-lc", inner]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 120)
    row = {"tag": tag, "name": ep["name"], "slot": slot, "wall_s": round(time.time() - t0, 1), "docker_rc": p.returncode,
           "attack": cfg["attack"]["id"], "defense": cfg["defense"]["id"], "seed": cfg["experiment"]["seed"], "n": n,
           "meta": ep.get("meta", {}), "scenario_draw": draw}
    try:
        row["status"] = json.loads((rd_host / "episode_status.json").read_text())
    except Exception:
        row["status"] = None
    try:
        s = json.loads((rd_host / "eval" / "summary.json").read_text())
        row.update({k: s.get(k) for k in ("min_sep_hh", "min_sep_ha", "collision_hh", "collision_ha", "rtf_mean",
                                          "rtf_min", "formation_rmse_hh", "centroid_progress_m", "censored")})
    except Exception as exc:
        row["eval_error"] = str(exc)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cpus", type=float, default=6.0)
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--only", default=None, help="comma list of episode names")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    base = tomllib.loads((ROOT / spec.get("base", "config/swarm/ral26_base.toml")).read_text())
    deep_update(base, spec.get("common", {}))
    eps = spec["episodes"]
    if a.only:
        keep = set(a.only.split(","))
        eps = [e for e in eps if e["name"] in keep]
    global RESULTS_ROOT, RTF_RERUN
    RESULTS_ROOT = spec.get("results_root", RESULTS_ROOT)
    RTF_RERUN = spec.get("rtf_rerun_below")
    idx = ROOT / RESULTS_ROOT / spec["tag"] / "index.jsonl"
    idx.parent.mkdir(parents=True, exist_ok=True)
    for k in range(a.workers):
        SLOTS.put(k)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_episode, spec["tag"], base, e, a.cpus, a.timeout): e for e in eps}
        for f in cf.as_completed(futs):
            try:
                row = f.result()
            except Exception as exc:
                row = {"name": futs[f]["name"], "error": repr(exc)}
            if not row.get("skipped"):
                with idx.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
