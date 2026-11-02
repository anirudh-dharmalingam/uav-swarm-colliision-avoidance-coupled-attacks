"""One PX4 SITL vehicle seen from ITS OWN companion process (namespaced uXRCE-DDS topics).

Frames: common ENU world (Gazebo); PX4 local NED with origin at the vehicle's spawn point:
  N = y_enu - spawn_y,  E = x_enu - spawn_x,  D = -z_up.
Offboard command interfaces:
  velocity      : xy velocity setpoint, z position hold (the legacy v = V + u*tau mapping)
  acceleration  : xy acceleration setpoint (NaN position/velocity on xy), z position hold.
                  Still tracked by PX4's attitude/rate loops -- NOT a direct motor acceleration.
"""
from __future__ import annotations

import time

import numpy as np
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus

NAN = float("nan")
QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                 durability=DurabilityPolicy.VOLATILE)


class PX4Vehicle:
    def __init__(self, node, uav_id: int, spawn_enu):
        self.node, self.uav_id = node, uav_id
        self.ns = f"/uav_{uav_id}"
        self.target_system = uav_id + 1
        self.spawn = np.array([float(spawn_enu[0]), float(spawn_enu[1]), 0.0])
        self.pos = self.vel = self.acc = None
        self.xy_valid = False
        self.status = None
        self.t_rx = 0.0
        self.last_arm = self.last_mode = self.last_land = -1e9
        self.offboard_pub = node.create_publisher(OffboardControlMode, f"{self.ns}/fmu/in/offboard_control_mode", QOS)
        self.sp_pub = node.create_publisher(TrajectorySetpoint, f"{self.ns}/fmu/in/trajectory_setpoint", QOS)
        self.cmd_pub = node.create_publisher(VehicleCommand, f"{self.ns}/fmu/in/vehicle_command", QOS)
        node.create_subscription(VehicleLocalPosition, f"{self.ns}/fmu/out/vehicle_local_position_v1", self._on_pos, QOS)
        node.create_subscription(VehicleStatus, f"{self.ns}/fmu/out/vehicle_status_v1", self._on_status, QOS)

    def _on_pos(self, m):
        self.pos = np.array([float(m.x), float(m.y), float(m.z)])
        self.vel = np.array([float(m.vx), float(m.vy), float(m.vz)])
        self.acc = np.array([float(m.ax), float(m.ay), float(m.az)])
        self.xy_valid = bool(m.xy_valid and m.v_xy_valid)
        self.t_rx = time.monotonic()
        self.t_px4 = 1e-6 * float(m.timestamp)        # PX4 (lockstep sim) time of the estimate

    def _on_status(self, m):
        self.status = m

    def has_state(self):
        return self.pos is not None

    @staticmethod
    def ned2enu(v):
        return np.array([v[1], v[0], -v[2]])

    def pos_enu(self):
        return self.spawn + self.ned2enu(self.pos)

    def vel_enu(self):
        return self.ned2enu(self.vel)

    def acc_enu(self):
        return self.ned2enu(self.acc)

    def armed(self):
        return self.status is not None and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def offboard(self):
        return self.status is not None and self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def nav_state(self):
        return -1 if self.status is None else int(self.status.nav_state)

    # ---- commands ------------------------------------------------------------------------------
    def _ts(self):
        return int(self.node.get_clock().now().nanoseconds / 1000)

    def publish_mode(self, kind: str):
        m = OffboardControlMode()
        m.timestamp = self._ts()
        m.position = kind in ("position", "velocity", "acceleration")
        m.velocity = kind == "velocity"
        m.acceleration = kind == "acceleration"
        m.attitude = m.body_rate = False
        self.offboard_pub.publish(m)

    def position_setpoint_enu(self, p_enu):
        loc = np.asarray(p_enu, float) - self.spawn
        s = TrajectorySetpoint()
        s.timestamp = self._ts()
        s.position = [float(loc[1]), float(loc[0]), float(-p_enu[2])]
        s.velocity = [NAN, NAN, NAN]
        s.acceleration = [NAN, NAN, NAN]
        s.yaw = 0.0
        self.sp_pub.publish(s)

    def hover_setpoint(self, alt_up: float):
        s = TrajectorySetpoint()
        s.timestamp = self._ts()
        s.position = [0.0, 0.0, -float(alt_up)]
        s.velocity = [NAN, NAN, NAN]
        s.acceleration = [NAN, NAN, NAN]
        s.yaw = 0.0
        self.sp_pub.publish(s)

    def velocity_setpoint(self, v_enu, alt_up: float):
        s = TrajectorySetpoint()
        s.timestamp = self._ts()
        s.position = [NAN, NAN, -float(alt_up)]
        s.velocity = [float(v_enu[1]), float(v_enu[0]), NAN]
        s.acceleration = [NAN, NAN, NAN]
        s.yaw = 0.0
        self.sp_pub.publish(s)

    def accel_setpoint(self, a_enu, alt_up: float):
        s = TrajectorySetpoint()
        s.timestamp = self._ts()
        s.position = [NAN, NAN, -float(alt_up)]
        s.velocity = [NAN, NAN, NAN]
        s.acceleration = [float(a_enu[1]), float(a_enu[0]), NAN]
        s.yaw = 0.0
        self.sp_pub.publish(s)

    def _cmd(self, command, p1=0.0, p2=0.0):
        m = VehicleCommand()
        m.timestamp = self._ts()
        m.command = command
        m.param1, m.param2 = float(p1), float(p2)
        m.target_system, m.target_component = self.target_system, 1
        m.source_system, m.source_component = 1, 1
        m.from_external = True
        self.cmd_pub.publish(m)

    def request_offboard(self):
        self._cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

    def request_arm(self):
        self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

    def request_land(self):
        self._cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)

    # ---- shared arm/takeoff state machine --------------------------------------------------------
    def bootstrap(self, now, alt, retry=2.0):
        """Stream hover setpoints; request offboard then arm.  Returns True once armed+offboard."""
        self.publish_mode("position")
        self.hover_setpoint(alt)
        if not self.offboard() and now - self.last_mode >= retry:
            self.request_offboard()
            self.last_mode = now
        if self.offboard() and not self.armed() and now - self.last_arm >= retry:
            self.request_arm()
            self.last_arm = now
        return self.armed() and self.offboard()
