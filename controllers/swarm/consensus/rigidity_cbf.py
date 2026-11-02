#!/usr/bin/env python3

import math
from typing import Dict, Tuple


Vector3 = Tuple[float, float, float]


class RigidityCBFController:
    """
    Distributed formation controller with:

      1. Rigidity/formation maintenance
      2. Relative-velocity damping
      3. Control-barrier-function-style collision avoidance
      4. Mission velocity tracking
      5. Global velocity saturation

    The controller remains decentralized:
    each UAV uses only its own state and the
    states of currently available neighbors.

    Nominal controller:

        v_nom =
            v_mission
            - kp * formation_error
            - kv * relative_velocity_error

    Safety layer:

        For each neighbor j:

            h_ij = ||p_i - p_j||^2 - d_safe^2

        The barrier correction pushes the UAV away from
        neighbors when the safety margin becomes small.

    Interface intentionally matches the previous
    FormationConsensus.compute() method.
    """

    def __init__(
        self,
        kp: float,
        kv: float,
        max_speed: float,
        min_separation: float = 2.5,
        cbf_alpha: float = 2.0,
        cbf_slack_penalty: float = 100.0,
        normalize_by_degree: bool = True,
    ):
        if kp < 0.0:
            raise ValueError("kp must be non-negative")

        if kv < 0.0:
            raise ValueError("kv must be non-negative")

        if max_speed <= 0.0:
            raise ValueError("max_speed must be positive")

        if min_separation <= 0.0:
            raise ValueError("min_separation must be positive")

        if cbf_alpha <= 0.0:
            raise ValueError("cbf_alpha must be positive")

        if cbf_slack_penalty < 0.0:
            raise ValueError(
                "cbf_slack_penalty must be non-negative"
            )

        self.kp = float(kp)
        self.kv = float(kv)
        self.max_speed = float(max_speed)

        self.min_separation = float(min_separation)
        self.cbf_alpha = float(cbf_alpha)
        self.cbf_slack_penalty = float(cbf_slack_penalty)

        self.normalize_by_degree = normalize_by_degree

        # Useful telemetry for the experiment logger.
        self.last_cbf_active = False
        self.last_min_barrier = float("inf")
        self.last_cbf_intervention = 0.0
        self.last_neighbor_count = 0

    # ---------------------------------------------------------
    # Vector operations
    # ---------------------------------------------------------

    @staticmethod
    def _add(a: Vector3, b: Vector3) -> Vector3:
        return (
            a[0] + b[0],
            a[1] + b[1],
            a[2] + b[2],
        )

    @staticmethod
    def _sub(a: Vector3, b: Vector3) -> Vector3:
        return (
            a[0] - b[0],
            a[1] - b[1],
            a[2] - b[2],
        )

    @staticmethod
    def _scale(a: Vector3, scalar: float) -> Vector3:
        return (
            a[0] * scalar,
            a[1] * scalar,
            a[2] * scalar,
        )

    @staticmethod
    def _dot(a: Vector3, b: Vector3) -> float:
        return (
            a[0] * b[0]
            + a[1] * b[1]
            + a[2] * b[2]
        )

    @staticmethod
    def _norm(a: Vector3) -> float:
        return math.sqrt(
            a[0] ** 2
            + a[1] ** 2
            + a[2] ** 2
        )

    # ---------------------------------------------------------
    # Speed limiting
    # ---------------------------------------------------------

    def _clamp_speed(self, velocity: Vector3) -> Vector3:

        speed = self._norm(velocity)

        if speed <= self.max_speed:
            return velocity

        if speed < 1e-9:
            return (0.0, 0.0, 0.0)

        return self._scale(
            velocity,
            self.max_speed / speed,
        )

    # ---------------------------------------------------------
    # Nominal formation controller
    # ---------------------------------------------------------

    def _nominal_velocity(
        self,
        agent_id: int,
        own_position: Vector3,
        own_velocity: Vector3,
        neighbor_states: Dict[int, dict],
        desired_offsets: Dict[int, Vector3],
        mission_velocity: Vector3,
    ) -> Vector3:

        position_error = (0.0, 0.0, 0.0)
        velocity_error = (0.0, 0.0, 0.0)

        count = 0

        if agent_id not in desired_offsets:
            return self._clamp_speed(
                mission_velocity
            )

        for neighbor_id, neighbor in neighbor_states.items():

            if neighbor_id == agent_id:
                continue

            if neighbor_id not in desired_offsets:
                continue

            try:
                neighbor_position = tuple(
                    neighbor["position"]
                )

                neighbor_velocity = tuple(
                    neighbor["velocity"]
                )
            except (KeyError, TypeError, ValueError):
                continue

            # Actual relative position:
            #
            # p_i - p_j
            actual_relative = self._sub(
                own_position,
                neighbor_position,
            )

            # Desired relative position:
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

        self.last_neighbor_count = count

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

        formation_term = self._scale(
            position_error,
            -self.kp,
        )

        damping_term = self._scale(
            velocity_error,
            -self.kv,
        )

        correction = self._add(
            formation_term,
            damping_term,
        )

        return self._clamp_speed(
            self._add(
                mission_velocity,
                correction,
            )
        )

    # ---------------------------------------------------------
    # CBF safety layer
    # ---------------------------------------------------------

    def _cbf_correction(
        self,
        own_position: Vector3,
        own_velocity: Vector3,
        neighbor_states: Dict[int, dict],
    ) -> Vector3:

        correction = (0.0, 0.0, 0.0)

        self.last_cbf_active = False
        self.last_min_barrier = float("inf")

        d_safe = self.min_separation

        for neighbor_id, neighbor in neighbor_states.items():

            try:
                neighbor_position = tuple(
                    neighbor["position"]
                )

                neighbor_velocity = tuple(
                    neighbor["velocity"]
                )

            except (KeyError, TypeError, ValueError):
                continue

            relative_position = self._sub(
                own_position,
                neighbor_position,
            )

            relative_velocity = self._sub(
                own_velocity,
                neighbor_velocity,
            )

            distance_sq = self._dot(
                relative_position,
                relative_position,
            )

            distance = math.sqrt(
                max(distance_sq, 1e-12)
            )

            # Barrier function:
            #
            # h = ||p_i-p_j||² - d_safe²
            #
            # h >= 0 is the desired safe region.

            h = distance_sq - d_safe ** 2

            self.last_min_barrier = min(
                self.last_min_barrier,
                h,
            )

            # Time derivative:
            #
            # h_dot = 2 r^T v_rel

            h_dot = 2.0 * self._dot(
                relative_position,
                relative_velocity,
            )

            # CBF condition:
            #
            # h_dot + alpha*h >= 0
            #
            # If this is violated, the nominal velocity is
            # driving the agents toward an unsafe state.

            cbf_value = (
                h_dot
                + self.cbf_alpha * h
            )

            if cbf_value >= 0.0:
                continue

            self.last_cbf_active = True

            # Radial direction away from the neighbor.
            direction = self._scale(
                relative_position,
                1.0 / distance,
            )

            # Required corrective radial velocity.
            required = (
                -cbf_value
                / max(
                    2.0 * distance,
                    1e-6,
                )
            )

            # Never generate a negative safety correction.
            required = max(
                0.0,
                required,
            )

            safety_term = self._scale(
                direction,
                required,
            )

            correction = self._add(
                correction,
                safety_term,
            )

        return correction

    # ---------------------------------------------------------
    # Public controller interface
    # ---------------------------------------------------------

    def compute(
        self,
        agent_id: int,
        own_position: Vector3,
        own_velocity: Vector3,
        neighbor_states: Dict[int, dict],
        desired_offsets: Dict[int, Vector3],
        mission_velocity: Vector3,
    ) -> Vector3:

        # 1. Nominal formation/mission command.
        nominal_velocity = self._nominal_velocity(
            agent_id=agent_id,
            own_position=own_position,
            own_velocity=own_velocity,
            neighbor_states=neighbor_states,
            desired_offsets=desired_offsets,
            mission_velocity=mission_velocity,
        )

        # 2. Safety correction.
        cbf_correction = self._cbf_correction(
            own_position=own_position,
            own_velocity=own_velocity,
            neighbor_states=neighbor_states,
        )

        # 3. Safe command.
        safe_velocity = self._add(
            nominal_velocity,
            cbf_correction,
        )

        safe_velocity = self._clamp_speed(
            safe_velocity
        )

        # 4. Telemetry.
        self.last_cbf_intervention = self._norm(
            self._sub(
                safe_velocity,
                nominal_velocity,
            )
        )

        return safe_velocity