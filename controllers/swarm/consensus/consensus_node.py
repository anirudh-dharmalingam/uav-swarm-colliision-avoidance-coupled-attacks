#!/usr/bin/env python3

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import rclpy
from rclpy.node import Node

from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from px4_msgs.msg import (
    BatteryStatus,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleCommandAck,
    VehicleLocalPosition,
    VehicleStatus,
)

from formation_consensus import FormationConsensus
from rigidity_cbf import RigidityCBFController

Vector3 = Tuple[float, float, float]


PX4_ARMED = VehicleStatus.ARMING_STATE_ARMED
PX4_OFFBOARD = VehicleStatus.NAVIGATION_STATE_OFFBOARD


class UAVAgent:

    def __init__(
        self,
        node: Node,
        uav_id: int,
        spawn_position: Vector3,
    ):
        self.node = node
        self.uav_id = uav_id

        self.namespace = f"/uav_{uav_id}"

        # PX4 system IDs are instance + 1.
        self.target_system = uav_id + 1

        self.spawn_position = spawn_position

        self.position_local = None
        self.velocity_local = None
        self.battery_remaining = float("nan")

        self.status = None
        self.last_position_wall = 0.0

        self.last_arm_command = 0.0
        self.last_offboard_command = 0.0
        self.last_land_command = 0.0

        self.last_ack_command = None
        self.last_ack_result = None

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.offboard_pub = node.create_publisher(
            OffboardControlMode,
            f"{self.namespace}/fmu/in/offboard_control_mode",
            qos,
        )

        self.setpoint_pub = node.create_publisher(
            TrajectorySetpoint,
            f"{self.namespace}/fmu/in/trajectory_setpoint",
            qos,
        )

        self.command_pub = node.create_publisher(
            VehicleCommand,
            f"{self.namespace}/fmu/in/vehicle_command",
            qos,
        )

        node.create_subscription(
            VehicleLocalPosition,
            f"{self.namespace}/fmu/out/vehicle_local_position_v1",
            self.position_callback,
            qos,
        )

        node.create_subscription(
            VehicleStatus,
            f"{self.namespace}/fmu/out/vehicle_status_v1",
            self.status_callback,
            qos,
        )

        node.create_subscription(
            BatteryStatus,
            f"{self.namespace}/fmu/out/battery_status_v1",
            self.battery_callback,
            qos,
        )

        node.create_subscription(
            VehicleCommandAck,
            f"{self.namespace}/fmu/out/vehicle_command_ack",
            self.ack_callback,
            qos,
        )

    def position_callback(self, msg):

        self.position_local = (
            float(msg.x),
            float(msg.y),
            float(msg.z),
        )

        self.velocity_local = (
            float(msg.vx),
            float(msg.vy),
            float(msg.vz),
        )

        self.last_position_wall = time.monotonic()

    def status_callback(self, msg):
        self.status = msg

    def battery_callback(self, msg):

        value = getattr(
            msg,
            "remaining",
            float("nan"),
        )

        try:
            self.battery_remaining = float(value)
        except (TypeError, ValueError):
            self.battery_remaining = float("nan")

    def ack_callback(self, msg):

        self.last_ack_command = int(
            msg.command
        )

        self.last_ack_result = int(
            msg.result
        )

    def has_state(self) -> bool:
        return (
            self.position_local is not None
            and self.velocity_local is not None
        )

    def state_age(self) -> float:

        if self.last_position_wall == 0.0:
            return float("inf")

        return (
            time.monotonic()
            - self.last_position_wall
        )

    def common_position(self) -> Vector3:

        if not self.has_state():
            raise RuntimeError(
                f"UAV {self.uav_id} has no position"
            )

        # Each PX4 instance has its own local origin.
        #
        # The known Gazebo spawn translation is used to
        # express local PX4 position in the common swarm frame.
        #
        # This is suitable for the controlled baseline.
        # Later, this transform becomes part of the estimator/
        # attackable state pipeline.

        return (
            self.spawn_position[0]
            + self.position_local[0],
            self.spawn_position[1]
            + self.position_local[1],
            self.position_local[2],
        )

    def common_velocity(self) -> Vector3:

        if not self.has_state():
            raise RuntimeError(
                f"UAV {self.uav_id} has no velocity"
            )

        return self.velocity_local

    def is_armed(self) -> bool:

        return (
            self.status is not None
            and self.status.arming_state
            == PX4_ARMED
        )

    def is_offboard(self) -> bool:

        return (
            self.status is not None
            and self.status.nav_state
            == PX4_OFFBOARD
        )

    def preflight_pass(self) -> bool:

        return (
            self.status is not None
            and bool(
                getattr(
                    self.status,
                    "pre_flight_checks_pass",
                    False,
                )
            )
        )

    def timestamp_us(self) -> int:

        return int(
            self.node.get_clock().now().nanoseconds
            / 1000
        )

    def publish_offboard_mode(self):

        msg = OffboardControlMode()

        msg.timestamp = self.timestamp_us()

        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False

        if hasattr(
            msg,
            "thrust_and_torque",
        ):
            msg.thrust_and_torque = False

        if hasattr(
            msg,
            "direct_actuator",
        ):
            msg.direct_actuator = False

        self.offboard_pub.publish(msg)

    def publish_position_setpoint(
        self,
        position: Vector3,
    ):

        msg = TrajectorySetpoint()

        msg.timestamp = self.timestamp_us()

        msg.position = [
            float(position[0]),
            float(position[1]),
            float(position[2]),
        ]

        msg.velocity = [
            float("nan"),
            float("nan"),
            float("nan"),
        ]

        msg.acceleration = [
            float("nan"),
            float("nan"),
            float("nan"),
        ]

        msg.yaw = 0.0

        self.setpoint_pub.publish(msg)

    def send_command(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
    ):

        msg = VehicleCommand()

        msg.timestamp = self.timestamp_us()

        msg.command = command

        msg.param1 = float(param1)
        msg.param2 = float(param2)

        msg.target_system = self.target_system
        msg.target_component = 1

        msg.source_system = 1
        msg.source_component = 1

        msg.from_external = True

        self.command_pub.publish(msg)

    def request_offboard(self):

        self.send_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            1.0,
            6.0,
        )

    def request_arm(self):

        self.send_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            1.0,
        )

    def request_land(self):

        self.send_command(
            VehicleCommand.VEHICLE_CMD_NAV_LAND
        )


class FormationConsensusNode(Node):

    def __init__(
        self,
        config_path: Path,
        geometry_path: Path,
        run_dir: Path,
    ):
        super().__init__("swarm_formation_consensus")

        self.config = json.loads(
            config_path.read_text()
        )

        geometry = json.loads(
            geometry_path.read_text()
        )

        self.run_dir = run_dir

        self.num_uavs = int(
            self.config["swarm"]["num_uavs"]
        )

        swarm_cfg = self.config["swarm"]

        self.altitude = -abs(
            float(
                swarm_cfg.get(
                    "altitude_m",
                    swarm_cfg.get("altitude", 5.0),
                )
            )
        )

        control_cfg = self.config["control"]

        self.kp = float(
            control_cfg["kp"]
        )

        self.kv = float(
            control_cfg["kv"]
        )

        self.max_speed = float(
            control_cfg["max_speed_mps"]
        )

        self.bootstrap_s = float(
            control_cfg["bootstrap_s"]
        )

        self.arm_retry_s = float(
            control_cfg["arm_retry_s"]
        )

        self.land_timeout_s = float(
            control_cfg["land_timeout_s"]
        )

        formation_cfg = self.config["formation"]

        self.neighbor_radius = float(
            formation_cfg["neighbor_radius_m"]
        )

        self.state_timeout = float(
            formation_cfg["state_timeout_s"]
        )

        mission = self.config["mission"]

        self.mission_velocity = (
            float(mission["vx_mps"]),
            float(mission["vy_mps"]),
            float(mission["vz_mps"]),
        )

        reward_cfg = self.config["reward"]

        self.reward_formation_weight = float(
            reward_cfg["formation_weight"]
        )

        self.reward_control_weight = float(
            reward_cfg["control_weight"]
        )

        self.reward_collision_weight = float(
            reward_cfg["collision_weight"]
        )

        self.desired_offsets = {
            int(k): (
                float(v[0]),
                float(v[1]),
                float(v[2]),
            )
            for k, v in geometry[
                "desired_offsets"
            ].items()
        }

        self.spawn_positions = {
            int(k): (
                float(v[0]),
                float(v[1]),
                float(v[2]),
            )
            for k, v in geometry[
                "spawn_positions"
            ].items()
        }

        self.agents = {
            i: UAVAgent(
                self,
                i,
                self.spawn_positions[i],
            )
            for i in range(self.num_uavs)
        }
        self.consensus = RigidityCBFController(
            kp=self.kp,
            kv=self.kv,
            max_speed=self.max_speed,
            min_separation=float(
                self.config["formation"]["min_separation_m"]
            ),
            cbf_alpha=float(
                self.config["formation"]["cbf_alpha"]
            ),
            cbf_slack_penalty=float(
                self.config["formation"]["cbf_slack_penalty"]
            ),
        )

        self.phase = "WAITING"

        self.phase_start = time.monotonic()
        self.last_loop = time.monotonic()

        self.command_positions: Dict[int, list] = {}

        self.step = 0
        self.controller_start = None
        self.land_start = None

        data_dir = self.run_dir / "data"
        data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.data_files = {}
        self.data_writers = {}

        for i in range(self.num_uavs):

            path = (
                data_dir
                / f"uav_{i}.jsonl"
            )

            handle = path.open(
                "w",
                encoding="utf-8",
            )

            self.data_files[i] = handle

        metrics_path = (
            self.run_dir / "metrics.csv"
        )

        self.metrics_file = metrics_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )

        self.metrics_writer = csv.DictWriter(
            self.metrics_file,
            fieldnames=[
                "wall_time",
                "step",
                "phase",
                "formation_rmse",
                "minimum_distance",
                "velocity_disagreement",
                "armed_count",
                "offboard_count",
            ],
        )

        self.metrics_writer.writeheader()

        self.events_file = (
            self.run_dir / "events.jsonl"
        ).open(
            "w",
            encoding="utf-8",
        )

        self.timer = self.create_timer(
            0.05,
            self.control_loop,
        )

        self.log_event(
            "controller_started",
            {
                "num_uavs": self.num_uavs,
            },
        )

    def log_event(
        self,
        name: str,
        data: dict,
    ):

        record = {
            "wall_time": time.time(),
            "event": name,
            "data": data,
        }

        self.events_file.write(
            json.dumps(record)
            + "\n"
        )

        self.events_file.flush()

    def all_have_state(self) -> bool:

        return all(
            agent.has_state()
            for agent in self.agents.values()
        )

    def get_positions(self):

        return {
            i: agent.common_position()
            for i, agent in self.agents.items()
        }

    def get_velocities(self):

        return {
            i: agent.common_velocity()
            for i, agent in self.agents.items()
        }

    def build_neighbors(
        self,
        agent_id: int,
        positions: Dict[int, Vector3],
        velocities: Dict[int, Vector3],
    ):

        neighbors = {}

        own = positions[agent_id]

        for neighbor_id in range(
            self.num_uavs
        ):

            if neighbor_id == agent_id:
                continue

            age = self.agents[
                neighbor_id
            ].state_age()

            if age > self.state_timeout:
                continue

            dx = own[0] - positions[
                neighbor_id
            ][0]

            dy = own[1] - positions[
                neighbor_id
            ][1]

            dz = own[2] - positions[
                neighbor_id
            ][2]

            distance = math.sqrt(
                dx * dx
                + dy * dy
                + dz * dz
            )

            if distance <= self.neighbor_radius:

                neighbors[neighbor_id] = {
                    "position":
                        positions[neighbor_id],
                    "velocity":
                        velocities[neighbor_id],
                    "age_s":
                        age,
                }

        return neighbors

    @staticmethod
    def vector_norm(v: Vector3) -> float:

        return math.sqrt(
            v[0] ** 2
            + v[1] ** 2
            + v[2] ** 2
        )

    def formation_rmse(
        self,
        positions: Dict[int, Vector3],
    ) -> float:

        if not positions:
            return float("nan")

        centroid = (
            sum(
                p[0]
                for p in positions.values()
            )
            / len(positions),

            sum(
                p[1]
                for p in positions.values()
            )
            / len(positions),

            sum(
                p[2]
                for p in positions.values()
            )
            / len(positions),
        )

        squared_errors = []

        for i, position in positions.items():

            desired = self.desired_offsets[i]

            relative = (
                position[0] - centroid[0],
                position[1] - centroid[1],
                position[2] - centroid[2],
            )

            dx = relative[0] - desired[0]
            dy = relative[1] - desired[1]
            dz = relative[2] - desired[2]

            squared_errors.append(
                dx * dx
                + dy * dy
                + dz * dz
            )

        return math.sqrt(
            sum(squared_errors)
            / len(squared_errors)
        )

    def minimum_distance(
        self,
        positions: Dict[int, Vector3],
    ) -> float:

        minimum = float("inf")

        ids = sorted(
            positions.keys()
        )

        for a in range(len(ids)):

            for b in range(
                a + 1,
                len(ids),
            ):

                p = positions[ids[a]]
                q = positions[ids[b]]

                distance = math.sqrt(
                    (p[0] - q[0]) ** 2
                    + (p[1] - q[1]) ** 2
                    + (p[2] - q[2]) ** 2
                )

                minimum = min(
                    minimum,
                    distance,
                )

        return minimum

    def velocity_disagreement(
        self,
        velocities: Dict[int, Vector3],
    ) -> float:

        if len(velocities) < 2:
            return 0.0

        ids = sorted(
            velocities.keys()
        )

        errors = []

        for a in range(len(ids)):

            for b in range(
                a + 1,
                len(ids),
            ):

                va = velocities[ids[a]]
                vb = velocities[ids[b]]

                errors.append(
                    self.vector_norm(
                        (
                            va[0] - vb[0],
                            va[1] - vb[1],
                            va[2] - vb[2],
                        )
                    )
                )

        return (
            sum(errors)
            / len(errors)
        )

    def log_metrics(
        self,
        positions,
        velocities,
    ):

        armed_count = sum(
            int(agent.is_armed())
            for agent in self.agents.values()
        )

        offboard_count = sum(
            int(agent.is_offboard())
            for agent in self.agents.values()
        )

        self.metrics_writer.writerow(
            {
                "wall_time": time.time(),
                "step": self.step,
                "phase": self.phase,
                "formation_rmse":
                    self.formation_rmse(
                        positions
                    ),
                "minimum_distance":
                    self.minimum_distance(
                        positions
                    ),
                "velocity_disagreement":
                    self.velocity_disagreement(
                        velocities
                    ),
                "armed_count": armed_count,
                "offboard_count":
                    offboard_count,
            }
        )

        self.metrics_file.flush()

    def log_transition(
        self,
        agent_id: int,
        state: dict,
        neighbors: dict,
        action: Vector3,
        reward: float,
        formation_error: float,
        collision_penalty: float,
    ):

        record = {
            "run_id": self.config[
                "experiment"
            ].get("run_id"),
            "episode_id": self.config[
                "experiment"
            ]["episode_id"],
            "client_id": agent_id,
            "uav_id": agent_id,
            "step": self.step,
            "wall_time": time.time(),
            "phase": self.phase,

            "state": state,

            "observation": {
                "neighbor_count":
                    len(neighbors),
                "neighbors": {
                    str(k): {
                        "position":
                            list(v["position"]),
                        "velocity":
                            list(v["velocity"]),
                        "age_s":
                            v["age_s"],
                    }
                    for k, v in neighbors.items()
                },
            },

            "action": {
                "velocity_command":
                    list(action),
            },

            "reward": reward,

            "reward_components": {
                "formation":
                    formation_error,
                "collision_penalty":
                    collision_penalty,
            },

            "network_backend":
                self.config[
                    "network"
                ]["backend"],

            "attack_profile":
                self.config[
                    "attacks"
                ]["profile"],
        }

        handle = self.data_files[
            agent_id
        ]

        handle.write(
            json.dumps(record)
            + "\n"
        )

        handle.flush()

    def control_loop(self):

        now = time.monotonic()

        dt = now - self.last_loop

        self.last_loop = now

        dt = max(
            0.01,
            min(dt, 0.10),
        )

        for agent in self.agents.values():
            agent.publish_offboard_mode()

        if not self.all_have_state():
            return

        if self.controller_start is None:
            self.controller_start = now
            self.phase_start = now

        elapsed = (
            now
            - self.controller_start
        )

        positions = self.get_positions()
        velocities = self.get_velocities()

        # ----------------------------------------------
        # PHASE 1: WAIT / BOOTSTRAP
        # ----------------------------------------------

        if self.phase == "WAITING":

            self.phase = "BOOTSTRAP"
            self.phase_start = now

            self.log_event(
                "bootstrap_started",
                {},
            )

        if self.phase == "BOOTSTRAP":

            for i, agent in self.agents.items():

                target = (
                    agent.spawn_position[0],
                    agent.spawn_position[1],
                    self.altitude,
                )

                agent.publish_position_setpoint(
                    target
                )

            bootstrap_elapsed = (
                now - self.phase_start
            )

            if bootstrap_elapsed >= self.bootstrap_s:

                for agent in self.agents.values():

                    if (
                        not agent.is_offboard()
                        and
                        (
                            now
                            - agent.last_offboard_command
                            >= self.arm_retry_s
                        )
                    ):

                        agent.request_offboard()

                        agent.last_offboard_command = now

                for agent in self.agents.values():

                    if (
                        agent.is_offboard()
                        and not agent.is_armed()
                        and
                        (
                            now
                            - agent.last_arm_command
                            >= self.arm_retry_s
                        )
                    ):

                        agent.request_arm()

                        agent.last_arm_command = now

            if all(
                agent.is_armed()
                and agent.is_offboard()
                for agent in self.agents.values()
            ):

                self.phase = "FORMATION"

                self.phase_start = now

                for i, position in positions.items():

                    self.command_positions[i] = [
                        position[0],
                        position[1],
                        self.altitude,
                    ]

                self.log_event(
                    "formation_started",
                    {},
                )

            elif elapsed > 30.0:

                failed = []

                for i, agent in self.agents.items():

                    failed.append(
                        {
                            "uav_id": i,
                            "state": (
                                "armed"
                                if agent.is_armed()
                                else "disarmed"
                            ),
                            "offboard":
                                agent.is_offboard(),
                            "preflight":
                                agent.preflight_pass(),
                        }
                    )

                self.log_event(
                    "arm_timeout",
                    {"vehicles": failed},
                )

                self.phase = "ERROR"

            self.step += 1

            self.log_metrics(
                positions,
                velocities,
            )

            return

        # ----------------------------------------------
        # PHASE 2: DISTRIBUTED FORMATION CONSENSUS
        # ----------------------------------------------

        if self.phase == "FORMATION":

            if elapsed >= float(
                self.config[
                    "experiment"
                ]["duration_s"]
            ):

                self.phase = "LANDING"

                self.land_start = now

                self.log_event(
                    "landing_started",
                    {},
                )

            else:

                for i, agent in self.agents.items():

                    neighbors = self.build_neighbors(
                        i,
                        positions,
                        velocities,
                    )

                    action = self.consensus.compute(
                        agent_id=i,
                        own_position=positions[i],
                        own_velocity=velocities[i],
                        neighbor_states=neighbors,
                        desired_offsets=self.desired_offsets,
                        mission_velocity=self.mission_velocity,
                    )

                    cmd = self.command_positions[i]

                    cmd[0] += (
                        action[0] * dt
                    )

                    cmd[1] += (
                        action[1] * dt
                    )

                    # We hold the common altitude.
                    cmd[2] = self.altitude

                    agent.publish_position_setpoint(
                        (
                            cmd[0],
                            cmd[1],
                            cmd[2],
                        )
                    )

                    desired = self.desired_offsets[i]

                    # Agent-level formation error.
                    error_terms = []

                    for neighbor_id, neighbor in neighbors.items():

                        desired_relative = (
                            desired[0]
                            - self.desired_offsets[
                                neighbor_id
                            ][0],

                            desired[1]
                            - self.desired_offsets[
                                neighbor_id
                            ][1],

                            desired[2]
                            - self.desired_offsets[
                                neighbor_id
                            ][2],
                        )

                        actual_relative = (
                            positions[i][0]
                            - positions[
                                neighbor_id
                            ][0],

                            positions[i][1]
                            - positions[
                                neighbor_id
                            ][1],

                            positions[i][2]
                            - positions[
                                neighbor_id
                            ][2],
                        )

                        error_terms.append(
                            self.vector_norm(
                                (
                                    actual_relative[0]
                                    - desired_relative[0],

                                    actual_relative[1]
                                    - desired_relative[1],

                                    actual_relative[2]
                                    - desired_relative[2],
                                )
                            )
                        )

                    formation_error = (
                        sum(error_terms)
                        / len(error_terms)
                        if error_terms
                        else 0.0
                    )

                    effort = self.vector_norm(
                        action
                    )

                    min_distance = (
                        self.minimum_distance(
                            positions
                        )
                    )

                    safety_distance = (
                        0.5
                        * float(
                            self.config[
                                "formation"
                            ]["spacing_m"]
                        )
                    )

                    collision_penalty = 0.0

                    if (
                        min_distance
                        < safety_distance
                    ):

                        collision_penalty = (
                            self.reward_collision_weight
                            * (
                                safety_distance
                                - min_distance
                            )
                        )

                    reward = (
                        -self.reward_formation_weight
                        * formation_error
                        -self.reward_control_weight
                        * effort
                        -collision_penalty
                    )

                    state = {
                        "position":
                            list(positions[i]),
                        "velocity":
                            list(velocities[i]),
                        "battery_remaining":
                            self.agents[
                                i
                            ].battery_remaining,
                        "armed":
                            self.agents[
                                i
                            ].is_armed(),
                        "offboard":
                            self.agents[
                                i
                            ].is_offboard(),
                    }

                    self.log_transition(
                        i,
                        state,
                        neighbors,
                        action,
                        reward,
                        formation_error,
                        collision_penalty,
                    )

            self.step += 1

            self.log_metrics(
                positions,
                velocities,
            )

            return

        # ----------------------------------------------
        # PHASE 3: LAND
        # ----------------------------------------------

        if self.phase == "LANDING":

            for agent in self.agents.values():

                if (
                    now
                    - agent.last_land_command
                    >= 3.0
                ):

                    agent.request_land()

                    agent.last_land_command = now

            positions = self.get_positions()

            all_disarmed = all(
                not agent.is_armed()
                for agent in self.agents.values()
            )

            landing_elapsed = (
                now - self.land_start
            )

            if (
                all_disarmed
                or landing_elapsed
                > self.land_timeout_s
            ):

                self.log_event(
                    "experiment_complete",
                    {
                        "all_disarmed":
                            all_disarmed,
                        "landing_elapsed_s":
                            landing_elapsed,
                    },
                )

                self.phase = "DONE"

            self.step += 1

            self.log_metrics(
                positions,
                velocities,
            )

            return

        if self.phase in (
            "DONE",
            "ERROR",
        ):

            self.close_logs()

            raise SystemExit(
                0
                if self.phase == "DONE"
                else 1
            )

    def close_logs(self):

        for handle in self.data_files.values():

            try:
                handle.close()
            except Exception:
                pass

        try:
            self.metrics_file.close()
        except Exception:
            pass

        try:
            self.events_file.close()
        except Exception:
            pass

    def destroy_node(self):

        self.close_logs()

        super().destroy_node()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--geometry",
        required=True,
    )

    parser.add_argument(
        "--run-dir",
        required=True,
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    ).resolve()

    geometry_path = Path(
        args.geometry
    ).resolve()

    run_dir = Path(
        args.run_dir
    ).resolve()

    rclpy.init()

    node = FormationConsensusNode(
        config_path,
        geometry_path,
        run_dir,
    )

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except SystemExit:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()