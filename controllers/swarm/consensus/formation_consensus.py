#!/usr/bin/env python3

import math
from typing import Dict, Tuple


Vector3 = Tuple[float, float, float]


class FormationConsensus:
    """
    Distributed velocity-form formation consensus.

    For UAV i:

        v_i =
            v_mission
            - kp * mean_j(
                (p_i - p_j) - delta_ij
            )
            - kv * mean_j(
                v_i - v_j
            )

    delta_ij is the desired relative displacement
    between agents i and j.

    The controller uses only neighbor information.
    """

    def __init__(
        self,
        kp: float,
        kv: float,
        max_speed: float,
        normalize_by_degree: bool = True,
    ):
        if kp < 0.0:
            raise ValueError("kp must be non-negative")

        if kv < 0.0:
            raise ValueError("kv must be non-negative")

        if max_speed <= 0.0:
            raise ValueError("max_speed must be positive")

        self.kp = kp
        self.kv = kv
        self.max_speed = max_speed
        self.normalize_by_degree = normalize_by_degree

    @staticmethod
    def _sub(a: Vector3, b: Vector3) -> Vector3:
        return (
            a[0] - b[0],
            a[1] - b[1],
            a[2] - b[2],
        )

    @staticmethod
    def _add(a: Vector3, b: Vector3) -> Vector3:
        return (
            a[0] + b[0],
            a[1] + b[1],
            a[2] + b[2],
        )

    @staticmethod
    def _scale(a: Vector3, scalar: float) -> Vector3:
        return (
            a[0] * scalar,
            a[1] * scalar,
            a[2] * scalar,
        )

    def _clamp_speed(self, velocity: Vector3) -> Vector3:

        speed = math.sqrt(
            velocity[0] ** 2
            + velocity[1] ** 2
            + velocity[2] ** 2
        )

        if speed <= self.max_speed:
            return velocity

        scale = self.max_speed / speed

        return self._scale(
            velocity,
            scale,
        )

    def compute(
        self,
        agent_id: int,
        own_position: Vector3,
        own_velocity: Vector3,
        neighbor_states: Dict[int, dict],
        desired_offsets: Dict[int, Vector3],
        mission_velocity: Vector3,
    ) -> Vector3:

        if not neighbor_states:
            return self._clamp_speed(
                mission_velocity
            )

        position_error = (0.0, 0.0, 0.0)
        velocity_error = (0.0, 0.0, 0.0)

        count = 0

        for neighbor_id, neighbor in neighbor_states.items():

            if neighbor_id not in desired_offsets:
                continue

            neighbor_position = neighbor["position"]
            neighbor_velocity = neighbor["velocity"]

            # Actual relative displacement:
            #
            # p_i - p_j
            actual_relative = self._sub(
                own_position,
                neighbor_position,
            )

            # Desired relative displacement:
            #
            # delta_i - delta_j
            desired_relative = self._sub(
                desired_offsets[agent_id],
                desired_offsets[neighbor_id],
            )

            formation_error = self._sub(
                actual_relative,
                desired_relative,
            )

            relative_velocity_error = self._sub(
                own_velocity,
                neighbor_velocity,
            )

            position_error = self._add(
                position_error,
                formation_error,
            )

            velocity_error = self._add(
                velocity_error,
                relative_velocity_error,
            )

            count += 1

        if count == 0:
            return self._clamp_speed(
                mission_velocity
            )

        if self.normalize_by_degree:
            position_error = self._scale(
                position_error,
                1.0 / count,
            )

            velocity_error = self._scale(
                velocity_error,
                1.0 / count,
            )

        correction = self._add(
            self._scale(
                position_error,
                -self.kp,
            ),
            self._scale(
                velocity_error,
                -self.kv,
            ),
        )

        command = self._add(
            mission_velocity,
            correction,
        )

        return self._clamp_speed(command)