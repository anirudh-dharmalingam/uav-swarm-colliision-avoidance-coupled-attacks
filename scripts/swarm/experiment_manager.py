#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path


PROJECT_ROOT = Path("/workspace")

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "swarm"
    / "exp001_consensus.toml"
)

PX4_DIR = Path("/opt/PX4-Autopilot")

STAGE = os.environ.get("DS_STAGE", "all")     # all | px4 | ros (see _stage_methods)

BUILD_DIR = (
    PX4_DIR
    / "build"
    / "px4_sitl_default"
)

PX4_BIN = (
    BUILD_DIR
    / "bin"
    / "px4"
)

PROFILES = {
    "core": (
        "gz_x500",
        4001,
    ),

    "depth": (
        "gz_x500_depth",
        4002,
    ),

    "vision": (
        "gz_x500_vision",
        4005,
    ),

    "lidar_down": (
        "gz_x500_lidar_down",
        4016,
    ),

    "lidar_2d": (
        "gz_x500_lidar_2d",
        4013,
    ),

    "lidar_front": (
        "gz_x500_lidar_front",
        4017,
    ),
}


class ProcessLogger:

    def __init__(
        self,
        process,
        label,
        logfile,
    ):

        self.process = process
        self.label = label
        self.logfile = logfile

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

    def start(self):
        self.thread.start()

    def _run(self):

        self.logfile.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.logfile.open(
            "w",
            encoding="utf-8",
        ) as handle:

            for line in self.process.stdout:

                handle.write(line)
                handle.flush()

                print(
                    f"[{self.label}] "
                    f"{line}",
                    end="",
                    flush=True,
                )


def run_command(command):

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )


def command_exists(name):

    result = run_command(
        ["bash", "-lc", f"command -v {name}"]
    )

    return result.returncode == 0


def load_config(path):

    with path.open(
        "rb"
    ) as handle:

        return tomllib.load(handle)


def save_json(path, data):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def generate_lattice(
    n,
    spacing,
    scale,
):

    if n <= 0:
        raise ValueError(
            "num_uavs must be > 0"
        )

    def build(
        distance
    ):

        columns = max(
            1,
            math.ceil(
                math.sqrt(n)
            ),
        )

        positions = []

        for i in range(n):

            row = i // columns
            col = i % columns

            x = (
                col * distance
                + (
                    0.5 * distance
                    if row % 2
                    else 0.0
                )
            )

            y = (
                row
                * distance
                * math.sqrt(3.0)
                / 2.0
            )

            positions.append(
                [
                    x,
                    y,
                    0.0,
                ]
            )

        centroid = [
            sum(p[k] for p in positions)
            / len(positions)
            for k in range(3)
        ]

        for p in positions:

            for k in range(3):
                p[k] -= centroid[k]

        return positions

    desired = build(
        spacing
    )

    spawn = build(
        spacing * scale
    )

    return desired, spawn


class ExperimentManager:

    def __init__(
        self,
        config_path,
        num_uavs=None,
        sensor_profile=None,
        duration=None,
        attack=None,
        defense=None,
        network=None,
    ):

        self.config_path = Path(
            config_path
        ).resolve()

        self.config = load_config(
            self.config_path
        )

        swarm_cfg = self.config[
            "swarm"
        ]

        if num_uavs is not None:
            swarm_cfg[
                "num_uavs"
            ] = num_uavs

        self.num_uavs = int(
            swarm_cfg["num_uavs"]
        )

        if sensor_profile is not None:

            if sensor_profile not in PROFILES:
                raise ValueError(
                    f"Unknown sensor profile: "
                    f"{sensor_profile}"
                )

            model, autostart = PROFILES[
                sensor_profile
            ]

            swarm_cfg[
                "model"
            ] = model

            swarm_cfg[
                "autostart"
            ] = autostart

            self.config[
                "sensors"
            ] = {
                "profile":
                    sensor_profile
            }

        self.config.setdefault("attack", {})
        self.config.setdefault("defense", {})
        self.config.setdefault("network", {})
        self.config.setdefault("attacks", {"profile": "none", "enabled": False})
        self.config.setdefault("learning", {"mode": "classical"})

        if attack is not None:
            self.config["attack"]["id"] = attack
            self.config["attacks"] = {"profile": attack, "enabled": attack != "A0"}
        if defense is not None:
            self.config["defense"]["id"] = defense
        if network is not None:
            self.config["network"]["emulation"] = network
        self.config["network"].setdefault("backend", "ideal_ros2")

        if duration is not None:

            self.config[
                "experiment"
            ][
                "duration_s"
            ] = float(
                duration
            )

        timestamp = dt.datetime.now(
            dt.timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )

        experiment_id = self.config[
            "experiment"
        ]["id"]

        tag = "_".join(
            x for x in (
                self.config["attack"].get("id"),
                self.config["defense"].get("id"),
            ) if x
        )
        if tag:
            experiment_id = f"{experiment_id}_{tag}"

        seed = self.config[
            "experiment"
        ]["seed"]

        self.run_id = (
            f"{timestamp}_"
            f"{experiment_id}_"
            f"n{self.num_uavs}_"
            f"s{seed}"
        )

        self.run_dir = (
            Path(os.environ["RAL_RUN_DIR"])
            if os.environ.get("RAL_RUN_DIR")
            else PROJECT_ROOT
            / "results"
            / "swarm"
            / self.run_id
        )
        self.aux_processes = {}

        self.px4_dir = (
            self.run_dir
            / "px4"
        )

        self.controller_dir = (
            self.run_dir
            / "controller"
        )

        self.px4_processes = {}

        self.controller_process = None

        self.shutdown_requested = False
    def ensure_px4_build(self):
        """
        Ensure the PX4 SITL build required by this experiment exists.

        The experiment manager owns the build lifecycle so that a clean
        checkout can reproduce an experiment without requiring a manually
        pre-built PX4 tree.
        """

        if not PX4_DIR.is_dir():
            raise RuntimeError(
                f"PX4 source directory not found: {PX4_DIR}"
            )

        if not (
            PX4_DIR / "CMakeLists.txt"
        ).is_file():
            raise RuntimeError(
                f"Invalid PX4 source tree: {PX4_DIR}"
            )

        if PX4_BIN.is_file():
            print("[OK] PX4 SITL build")
            return

        print()
        print(
            "=============================================="
        )
        print(
            " PX4 SITL build not found"
        )
        print(
            " Building PX4 SITL automatically"
        )
        print(
            "=============================================="
        )

        build = subprocess.run(
            [
                "make",
                "px4_sitl_default",
            ],
            cwd=str(PX4_DIR),
            text=True,
        )

        if build.returncode != 0:
            raise RuntimeError(
                "PX4 SITL build failed."
            )

        if not PX4_BIN.is_file():
            raise RuntimeError(
                "PX4 build completed but executable "
                f"was not found at:\n{PX4_BIN}"
            )

        print("[OK] PX4 SITL build")
    def check_environment(self):

        print()
        print(
            "=============================================="
        )
        print(
            " DroneLab Scalable Swarm Experiment"
        )
        print(
            "=============================================="
        )
        print(
            f" UAV count: {self.num_uavs}"
        )
        print(
            f" Run ID:    {self.run_id}"
        )
        print(
            "=============================================="
        )

        if STAGE != "ros":                # the ROS 2 stage container never runs PX4
            self.ensure_px4_build()

        if not command_exists("ros2"):

            raise RuntimeError(
                "ros2 command not found"
            )

        if not command_exists("gz"):

            raise RuntimeError(
                "gz command not found"
            )

        if not command_exists("ss"):

            raise RuntimeError(
                "ss command not found"
            )

        # Verify the external XRCE-DDS agent.
        ss = run_command(
            ["ss", "-lun"]
        )

        if ":8888" not in ss.stdout:

            raise RuntimeError(
                "No UDP listener detected on "
                "port 8888.\n"
                "Start the external XRCE-DDS Agent first:\n"
                "MicroXRCEAgent udp4 -p 8888"
            )

        print()
        print(
            "[OK] PX4 binary"
        )
        print(
            "[OK] ROS 2"
        )
        print(
            "[OK] Gazebo executable"
        )
        print(
            "[OK] XRCE-DDS UDP 8888"
        )
        existing = self.existing_px4_instances()

        if existing:

            print()
            print(
                "ERROR: PX4 SITL instances are already running."
            )

            print(
                "The experiment manager will NOT reuse them."
            )

            print()

            for line in existing:
                print(
                    f"  {line}"
                )

            print()
            print(
                "Stop them before starting a new experiment:"
            )

            print(
                "  pkill -f "
                "'/opt/PX4-Autopilot/build/px4_sitl_default/bin/px4'"
            )

            raise RuntimeError(
                "Stale PX4 SITL processes detected."
            )

    def write_geometry(self):

        formation_cfg = self.config[
            "formation"
        ]

        spacing = float(
            formation_cfg[
                "spacing_m"
            ]
        )

        scale = float(
            self.config[
                "swarm"
            ][
                "spawn_scale"
            ]
        )

        desired, spawn = generate_lattice(
            self.num_uavs,
            spacing,
            scale,
        )

        cs = self.config.get("cs", {})                    # coupled-safety campaign hooks (absent => identical geometry)
        for k, (dx, dy) in (cs.get("spawn_jitter") or {}).items():
            spawn[int(k)][0] += float(dx)
            spawn[int(k)][1] += float(dy)
        ss = cs.get("slot_shift")                         # E3: shift one UAV's slot (desired AND spawn) along an xy direction
        if ss:
            for arr in (desired, spawn):
                arr[int(ss["id"])][0] += float(ss["dx"])
                arr[int(ss["id"])][1] += float(ss["dy"])

        geometry = {
            "desired_offsets": {
                str(i): desired[i]
                for i in range(
                    self.num_uavs
                )
            },

            "spawn_positions": {
                str(i): spawn[i]
                for i in range(
                    self.num_uavs
                )
            },
        }

        path = (
            self.run_dir
            / "geometry.json"
        )

        save_json(
            path,
            geometry,
        )

        return path, geometry

    def write_manifest(
        self,
        geometry,
    ):

        manifest = {
            "run_id":
                self.run_id,

            "experiment":
                self.config[
                    "experiment"
                ],

            "swarm":
                self.config[
                    "swarm"
                ],

            "formation":
                self.config[
                    "formation"
                ],

            "network":
                self.config[
                    "network"
                ],

            "attacks":
                self.config[
                    "attacks"
                ],

            "learning":
                self.config[
                    "learning"
                ],

            "geometry":
                geometry,

            "git": self.git_info(),

            "processes": {},
        }

        self.config[
            "experiment"
        ][
            "run_id"
        ] = self.run_id

        save_json(
            self.run_dir
            / "manifest.json",
            manifest,
        )

        save_json(
            self.run_dir
            / "resolved_config.json",
            self.config,
        )

        save_json(
            self.run_dir
            / "current_processes.json",
            manifest,
        )
    def existing_px4_instances(self):

        result = subprocess.run(
            [
                "pgrep",
                "-af",
                "/opt/PX4-Autopilot/build/px4_sitl_default/bin/px4",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        lines = [
            line
            for line in result.stdout.splitlines()
            if line.strip()
        ]

        return lines
    def git_info(self):

        try:

            sha = run_command(
                [
                    "git",
                    "-C",
                    str(
                        PROJECT_ROOT
                    ),
                    "rev-parse",
                    "HEAD",
                ]
            ).stdout.strip()

            status = run_command(
                [
                    "git",
                    "-C",
                    str(
                        PROJECT_ROOT
                    ),
                    "status",
                    "--porcelain",
                ]
            ).stdout

            return {
                "commit":
                    sha,
                "dirty":
                    bool(status.strip()),
            }

        except Exception:

            return {
                "commit":
                    None,
                "dirty":
                    None,
            }

    def spawn_px4(
        self,
        uav_id,
        spawn_position,
    ):

        model = self.config[
            "swarm"
        ]["model"]

        autostart = int(
            self.config[
                "swarm"
            ]["autostart"]
        )

        namespace = (
            f"uav_{uav_id}"
        )

        instance_root = os.environ.get("RAL_INSTANCE_ROOT")
        instance_dir = (
            Path(instance_root) / f"instance_{uav_id}"
            if instance_root
            else BUILD_DIR
            / "swarm_instances"
            / f"instance_{uav_id}"
        )

        instance_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        if instance_root:
            # private rootfs per instance (concurrent episodes must not share params/dataman):
            # PX4 with -w uses the working dir as rootfs, so link the read-only etc/ into it
            etc = instance_dir / "etc"
            if not etc.exists():
                etc.symlink_to(BUILD_DIR / "etc")

        logfile = (
            self.px4_dir
            / f"uav_{uav_id}.log"
        )

        pose = ",".join(
            [
                f"{spawn_position[0]:.6f}",
                f"{spawn_position[1]:.6f}",
                "0.0",
                "0.0",
                "0.0",
                "0.0",
            ]
        )

        env = os.environ.copy()

        env.update(
            {
                "PX4_GZ_STANDALONE":
                    "1",

                "PX4_SIM_MODEL":
                    model,

                "PX4_SYS_AUTOSTART":
                    str(autostart),

                "PX4_GZ_MODEL_POSE":
                    pose,

                "PX4_UXRCE_DDS_NS":
                    namespace,

                "ROS_DOMAIN_ID":
                    os.environ.get(
                        "ROS_DOMAIN_ID",
                        "42",
                    ),

                # Keep the autonomous SITL baseline
                # independent of a GCS heartbeat.
                "PX4_PARAM_NAV_DLL_ACT":
                    "0",
            }
        )
        # RAL26: extra PX4 parameters from the config (e.g. SDLOG_MODE=-1 to keep runs compact)
        for k, v in self.config.get("px4_params", {}).items():
            env[f"PX4_PARAM_{k}"] = str(v)

        print()
        print(
            f"[PX4-{uav_id}] "
            f"Starting instance "
            f"{uav_id}"
        )

        print(
            f"[PX4-{uav_id}] "
            f"Model: {model}"
        )

        print(
            f"[PX4-{uav_id}] "
            f"DDS namespace: "
            f"/{namespace}/"
        )

        print(
            f"[PX4-{uav_id}] "
            f"Pose: {pose}"
        )

        process = subprocess.Popen(
            [str(PX4_BIN), "-w", str(instance_dir), "-i", str(uav_id)]
            if instance_root
            else [
                str(PX4_BIN),
                "-i",
                str(uav_id),
            ],

            cwd=str(
                instance_dir
            ),

            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            text=True,

            bufsize=1,

            start_new_session=True,
        )
        process.stdin.write(
            "param set NAV_DLL_ACT 0\n"
        )
        process.stdin.flush()
        self.px4_processes[
            uav_id
        ] = process

        logger = ProcessLogger(
            process,
            f"PX4-{uav_id}",
            logfile,
        )

        logger.start()

        return process

    def start_px4_swarm(
        self,
        geometry,
    ):

        spawn = geometry[
            "spawn_positions"
        ]

        for i in range(
            self.num_uavs
        ):

            self.spawn_px4(
                i,
                spawn[str(i)],
            )

            # Stagger startup slightly.
            time.sleep(0.25)

    def ros_topics(self):

        result = run_command(
            [
                "ros2",
                "topic",
                "list",
            ]
        )

        if result.returncode != 0:
            return set()

        return set(
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        )

    def wait_for_px4_ros(
        self,
        timeout_s=90.0,
    ):

        required = {
            f"/uav_{i}"
            f"/fmu/out/"
            f"vehicle_local_position_v1"
            for i in range(
                self.num_uavs
            )
        }

        print()
        print(
            "=============================================="
        )
        print(
            " Waiting for all PX4 ROS 2 interfaces"
        )
        print(
            "=============================================="
        )

        deadline = (
            time.monotonic()
            + timeout_s
        )

        last_count = -1

        while (
            time.monotonic()
            < deadline
        ):

            topics = self.ros_topics()

            found = (
                required
                & topics
            )

            count = len(found)

            if count != last_count:

                print(
                    f"  PX4 ROS state topics: "
                    f"{count}/{self.num_uavs}"
                )

                last_count = count

            if found == required:

                print(
                    "  [OK] All PX4 ROS 2 state interfaces detected."
                )

                return

            # Fail early if a PX4 has already died.
            dead = [
                i
                for i, process
                in self.px4_processes.items()
                if process.poll()
                is not None
            ]

            if dead:

                raise RuntimeError(
                    f"PX4 processes exited early: "
                    f"{dead}"
                )

            time.sleep(1.0)

        raise TimeoutError(
            "Timed out waiting for all "
            "PX4 ROS 2 state topics."
        )

    def start_controller(self):

        script = self.config.get(
            "controller", {}
        ).get("script", "consensus/consensus_node.py")

        controller = (
            PROJECT_ROOT
            / "controllers"
            / "swarm"
            / script
        )

        resolved_config = (
            self.run_dir
            / "resolved_config.json"
        )

        geometry = (
            self.run_dir
            / "geometry.json"
        )

        logfile = (
            self.controller_dir
            / "controller.log"
        )

        env = os.environ.copy()

        print()
        print(
            "=============================================="
        )
        print(
            " Starting swarm controller"
        )
        print(
            "=============================================="
        )

        process = subprocess.Popen(
            [
                sys.executable,
                str(controller),
                "--config",
                str(resolved_config),
                "--geometry",
                str(geometry),
                "--run-dir",
                str(self.run_dir),
            ] + self.controller_extra_args(),

            cwd=str(
                controller.parent
            ),

            env=env,

            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            text=True,

            bufsize=1,

            start_new_session=True,
        )

        self.controller_process = process

        logger = ProcessLogger(
            process,
            "CTRL",
            logfile,
        )

        logger.start()

        return process

    def controller_extra_args(self):

        args = []

        if "swarmsec" in self.config.get(
            "controller", {}
        ).get("script", ""):

            args += [
                "--attack",
                self.config["attack"].get("id", "A0"),
                "--defense",
                self.config["defense"].get("id", "D0"),
            ]

        return args

    def write_process_manifest(self):

        processes = {
            "controller_pid": (
                self.controller_process.pid
                if self.controller_process
                else None
            ),
            "px4_pids": {
                str(i): p.pid
                for i, p in self.px4_processes.items()
            },
        }

        path = (
            self.run_dir
            / "current_processes.json"
        )

        data = json.loads(path.read_text())
        data["processes"] = processes
        save_json(path, data)

    @staticmethod
    def terminate_process_group(
        process,
        name,
    ):

        if process is None:
            return

        if process.poll() is not None:
            return

        try:

            print(
                f"[STOP] {name}"
            )

            os.killpg(
                process.pid,
                signal.SIGINT,
            )

            process.wait(
                timeout=5
            )

        except (
            ProcessLookupError,
            subprocess.TimeoutExpired,
        ):
            pass

        if process.poll() is None:

            try:

                os.killpg(
                    process.pid,
                    signal.SIGTERM,
                )

                process.wait(
                    timeout=3
                )

            except (
                ProcessLookupError,
                subprocess.TimeoutExpired,
            ):
                pass

        if process.poll() is None:

            try:

                os.killpg(
                    process.pid,
                    signal.SIGKILL,
                )

            except ProcessLookupError:
                pass

    def cleanup(self):

        print()
        print(
            "=============================================="
        )
        print(
            " Cleaning experiment processes"
        )
        print(
            "=============================================="
        )

        self.terminate_process_group(
            self.controller_process,
            "controller",
        )

        for name, proc in list(self.aux_processes.items()):
            self.terminate_process_group(proc, name)

        for i in sorted(
            self.px4_processes.keys(),
            reverse=True,
        ):

            self.terminate_process_group(
                self.px4_processes[i],
                f"PX4-{i}",
            )

        print(
            "[OK] Experiment processes stopped."
        )

        print(
            "[OK] External Gazebo left untouched."
        )

        print(
            "[OK] External XRCE-DDS Agent left untouched."
        )

    def run(self):

        self.run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.px4_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.controller_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            self.check_environment()

            geometry_path, geometry = (
                self.write_geometry()
            )

            self.write_manifest(
                geometry
            )

            if STAGE != "ros":
                print()
                print(
                    "[1/4] Starting PX4 swarm"
                )

                self.start_px4_swarm(
                    geometry
                )

                self.write_process_manifest()

            if STAGE == "px4":
                return self.hold_px4_stage()

            print()
            print(
                "[2/4] Waiting for ROS 2 discovery"
            )

            self.wait_for_px4_ros(
                timeout_s=float(os.environ.get("DS_ROS_WAIT_S", "90"))
            )

            mode = self.config.get("controller", {}).get("mode", "legacy")
            if mode in ("ral26", "hf1", "le"):
                return self.run_ral26(mode)

            # every legacy run also gets the isolated ground-truth evaluator (RAL26 metrics)
            if self.config.get("ral26", {}).get("evaluator", True):
                self.start_aux("evaluator", [
                    "truth_evaluator.py", "--n", str(self.num_uavs), "--run-dir", str(self.run_dir),
                    "--geometry", str(self.run_dir / "geometry.json"),
                    "--config", str(self.run_dir / "resolved_config.json"),
                    "--attacker", str(self.config.get("ral26", {}).get("attacker_id", -1))])

            print()
            print(
                "[3/4] Starting controller"
            )

            process = (
                self.start_controller()
            )

            self.write_process_manifest()

            print()
            print(
                "[4/4] Experiment running"
            )

            return_code = process.wait()
            ev = self.aux_processes.get("evaluator")
            if ev is not None:
                try:
                    ev.wait(timeout=20)          # exits on the controller's "done" marker
                except subprocess.TimeoutExpired:
                    self.terminate_process_group(ev, "evaluator")
            (self.run_dir / "episode_status.json").write_text(
                json.dumps({"gcs_code": 0 if return_code == 0 else return_code, "legacy_controller": True}))

            events_path = (
                self.run_dir
                / "events.jsonl"
            )

            completed = False

            if events_path.exists():

                text = events_path.read_text(
                    encoding="utf-8"
                )

                completed = (
                    '"event": "experiment_complete"'
                    in text
                )

            if return_code == 0 and completed:

                print(
                    " EXPERIMENT COMPLETED"
                )

            else:

                print(
                    " EXPERIMENT FAILED"
                )

                print(
                    f"Controller return code: "
                    f"{return_code}"
                )

                raise RuntimeError(
                    "Swarm controller exited "
                    "without completing the experiment."
                )

            print()
            print(
                "=============================================="
            )

            if return_code == 0:
                print(
                    " EXPERIMENT COMPLETED"
                )
            else:
                print(
                    " EXPERIMENT CONTROLLER FAILED"
                )

            print(
                "=============================================="
            )

            print(
                f"Results: {self.run_dir}"
            )

            return return_code

        finally:

            self.cleanup()


def _stage_methods():
    """Demo split (scripts/demo): DS_STAGE=px4 runs only the PX4 instances and holds them until the ROS 2
    stage (another container, same run dir) touches `ros_stage_done`; DS_STAGE=ros runs everything except PX4."""

    def hold_px4_stage(self):
        self.wait_for_px4_ros(timeout_s=float(os.environ.get("DS_ROS_WAIT_S", "90")))
        (self.run_dir / "px4_stage_ready").write_text(str(time.time()))
        print("[PX4 stage] all vehicles up; holding until the ROS 2 stage finishes")
        done = self.run_dir / "ros_stage_done"
        watch = PreflightWatch(self.px4_dir, self.px4_processes, self.run_dir / "live" / "preflight.json")
        while not done.exists():
            dead = [i for i, p in self.px4_processes.items() if p.poll() is not None]
            if dead:
                raise RuntimeError(f"PX4 processes exited early: {dead}")
            watch.tick()
            time.sleep(1.0)
        watch.tick()
        return 0

    ExperimentManager.hold_px4_stage = hold_px4_stage


class PreflightWatch:
    """Demo-only PX4 preflight watchdog (DS_STAGE=px4).  Tails each uav_<i>.log and publishes per-vehicle
    arming health to live/preflight.json for the Swarm Lab panel.

    SITL failure mode it repairs: when the host is CPU-starved (e.g. a PX4 campaign running alongside the
    Lab), Gazebo IMU messages arrive with gaps ("Accel #0 fail: TIMEOUT"), EKF2 absorbs them into a bogus
    accelerometer bias / vertical-velocity innovation and commander refuses to arm ("Arming denied").  The
    bias does not decay within the agents' 60 s arm window, so after a denial caused by an estimator check
    the watchdog restarts ekf2 on that instance (fresh filter, re-aligns in a few seconds) -- at most
    MAX_RESETS times, never once the vehicle has armed."""

    ESTIMATOR = ("Accelerometer Bias", "Gyro Bias", "velocity unstable", "position unstable", "height unstable",
                 "ekf2 missing data", "heading unstable", "yaw estimate")
    MIN_GAP_S, MAX_RESETS = 12.0, 3

    def __init__(self, px4_dir, procs, out):
        self.procs, self.out = procs, out
        self.files = {i: (Path(px4_dir) / f"uav_{i}.log") for i in procs}
        self.pos = {i: 0 for i in procs}
        self.v = {i: {"state": "booting", "fails": [], "denials": 0, "imu_timeouts": 0, "ekf_resets": 0,
                      "t_reset": -1e9} for i in procs}
        self.dirty = True

    def _line(self, i, s, now):
        v = self.v[i]
        if "Ready for takeoff" in s and v["state"] not in ("armed", "flown"):
            v["state"], v["fails"] = "ready", []
        elif "Preflight Fail:" in s and v["state"] not in ("armed", "flown"):   # post-landing checks are not preflight
            r = s.split("Preflight Fail:", 1)[1].strip()
            if r not in v["fails"]:
                v["fails"] = (v["fails"] + [r])[-6:]
            v["state"] = "failing"
        elif "Accel #" in s and "TIMEOUT" in s:
            v["imu_timeouts"] += 1
        elif "Arming denied" in s and v["state"] not in ("armed", "flown"):
            v["denials"] += 1
            v["state"] = "denied"
            est = [f for f in v["fails"] if any(k in f for k in self.ESTIMATOR)]
            if est and v["ekf_resets"] < self.MAX_RESETS and now - v["t_reset"] > self.MIN_GAP_S:
                self._reset_ekf(i, est, now)
        elif "Armed by" in s:
            v["state"], v["fails"] = "armed", []
        elif "Disarmed" in s and v["state"] == "armed":
            v["state"] = "flown"
        else:
            return
        self.dirty = True

    def _reset_ekf(self, i, why, now):
        p = self.procs[i]
        try:
            p.stdin.write("ekf2 stop\n")
            p.stdin.flush()
            time.sleep(0.3)
            p.stdin.write("ekf2 start\n")
            p.stdin.flush()
        except (OSError, ValueError) as exc:
            print(f"[PX4 stage] uav_{i}: ekf2 restart failed: {exc!r}")
            return
        v = self.v[i]
        v["ekf_resets"] += 1
        v["t_reset"] = now
        v["state"], v["fails"] = "recovering", []
        print(f"[PX4 stage] uav_{i}: arming denied by estimator checks ({'; '.join(why)}) "
              f"-> restarted ekf2 ({v['ekf_resets']}/{self.MAX_RESETS})")

    def tick(self):
        now = time.monotonic()
        for i, f in self.files.items():
            try:
                with f.open("rb") as h:
                    h.seek(self.pos[i])
                    chunk = h.read()
            except OSError:
                continue
            chunk = chunk[:chunk.rfind(b"\n") + 1]          # complete lines only; the rest is re-read next tick
            self.pos[i] += len(chunk)
            for s in chunk.decode("utf-8", errors="replace").splitlines():
                self._line(i, s, now)
        if self.dirty:
            self.dirty = False
            self.out.parent.mkdir(parents=True, exist_ok=True)
            rows = {str(i): {k: x for k, x in v.items() if k != "t_reset"} for i, v in self.v.items()}
            tmp = self.out.with_suffix(".tmp")
            tmp.write_text(json.dumps({"t": time.time(), "vehicles": rows}, indent=1))
            tmp.replace(self.out)


_stage_methods()


def _ral26_methods():
    """RAL26 launch path: one controller process per UAV + FANET bridge + GCS + isolated evaluator."""

    def start_aux(self, name, argv):
        script = PROJECT_ROOT / "controllers" / "swarm" / argv[0]
        log = self.controller_dir / f"{name}.log"
        proc = subprocess.Popen([sys.executable, str(script)] + list(argv[1:]), cwd=str(script.parent),
                                env=os.environ.copy(), stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                start_new_session=True)
        self.aux_processes[name] = proc
        return proc

    def run_ral26(self, mode):
        R = self.config.setdefault("ral26", {})
        rc = self.run_dir / "resolved_config.json"
        rc.write_text(json.dumps(self.config, indent=1))
        geo = self.run_dir / "geometry.json"
        attacker = int(R.get("attacker_id", -1))
        duration = float(self.config["experiment"]["duration_s"])
        print("[3/4] Starting RAL26 processes (per-UAV controllers, bridge, GCS, evaluator)")
        self.start_aux("evaluator", ["truth_evaluator.py", "--n", str(self.num_uavs), "--run-dir", str(self.run_dir),
                                     "--geometry", str(geo), "--config", str(rc), "--attacker", str(attacker)])
        if mode in ("ral26", "le"):
            self.start_aux("bridge", ["fanet_bridge.py", "--n", str(self.num_uavs), "--config", str(rc),
                                      "--run-dir", str(self.run_dir)])
        if mode == "ral26" and self.config.get("cs", {}).get("info", {}).get("mode") in ("oracle", "sensor"):
            self.start_aux("oracle", ["oracle_adapter.py", "--n", str(self.num_uavs), "--config", str(rc)])
        if mode == "le":
            self.start_aux("sensor", ["le_sensor_node.py", "--n", str(self.num_uavs), "--config", str(rc)])
        for i in range(self.num_uavs):
            if mode == "le":
                L = self.config.get("le", {})
                argv = ["le_agent_node.py", "--id", str(i), "--config", str(rc), "--geometry", str(geo),
                        "--run-dir", str(self.run_dir), "--scheme", L.get("scheme", "CADLE"),
                        "--attack", L.get("attack_id", "L0"), "--phys", L.get("phys", "none"),
                        "--attacker-id", str(int(L.get("attacker_id", -1)))]
            elif mode == "ral26":
                argv = [self.config.get("controller", {}).get("script", "uav_agent_node.py"), "--id", str(i), "--config", str(rc), "--geometry", str(geo),
                        "--run-dir", str(self.run_dir), "--variant", self.config["defense"].get("id", "D26"),
                        "--attack", self.config["attack"].get("id", "A0"), "--attacker-id", str(attacker)]
            else:
                argv = ["hf1_tracking_node.py", "--id", str(i), "--config", str(rc), "--geometry", str(geo),
                        "--run-dir", str(self.run_dir), "--mode", self.config.get("hf1", {}).get("mode", "acceleration"),
                        "--seed", str(self.config["experiment"]["seed"])]
            self.start_aux(f"agent_{i}", argv)
        gcs = self.start_aux("gcs", ["gcs_coordinator.py", "--n", str(self.num_uavs), "--duration", str(duration),
                                     "--run-dir", str(self.run_dir)])
        print("[4/4] Experiment running")
        t_lim = float(R.get("wall_timeout_s", duration + 260.0))
        try:
            code = gcs.wait(timeout=t_lim)
        except subprocess.TimeoutExpired:
            code = "timeout"
        ev = self.aux_processes["evaluator"]
        try:
            ev.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.terminate_process_group(ev, "evaluator")
        time.sleep(2.0)
        for i in range(self.num_uavs):
            p = self.aux_processes.get(f"agent_{i}")
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.terminate_process_group(p, f"agent_{i}")
        (self.run_dir / "episode_status.json").write_text(json.dumps({"gcs_code": code}))
        print(f"GCS exit: {code}; results: {self.run_dir}")
        return 0 if code == 0 else 1

    ExperimentManager.start_aux = start_aux
    ExperimentManager.run_ral26 = run_ral26


_ral26_methods()


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Scalable DroneLab "
            "PX4/Gazebo/ROS 2 swarm runner"
        )
    )

    parser.add_argument(
        "--config",
        default=str(
            DEFAULT_CONFIG
        ),
    )

    parser.add_argument(
        "--num-uavs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--sensor-profile",
        choices=sorted(
            PROFILES.keys()
        ),
        default=None,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
    )

    parser.add_argument("--attack", default=None)
    parser.add_argument("--defense", default=None)
    parser.add_argument("--network", default=None)

    return parser.parse_args()


def main():

    args = parse_args()

    manager = ExperimentManager(
        config_path=args.config,
        num_uavs=args.num_uavs,
        sensor_profile=args.sensor_profile,
        duration=args.duration,
        attack=args.attack,
        defense=args.defense,
        network=args.network,
    )

    return manager.run()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )