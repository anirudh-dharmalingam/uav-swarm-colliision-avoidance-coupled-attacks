"""Per-vehicle energy estimation and screening for the RAL26 agent (section 2.3).

Power source: the vehicle's OWN rotor-speed commands (PX4 -> Gazebo motor model), i.e. onboard-known
actuator data comparable to ESC telemetry.  P_est = inflation * sum_k c_m c_T w_k^3 is the Gazebo motor
model's shaft power, an ESTIMATE of propulsion power (no ESC/motor efficiency); it is labelled as such
everywhere.  PX4's simulated battery percentage is NOT used for energy: it drains synthetically.

The supervisor needs a return bound valid for the flown envelope:
  T_return_upper = dist_home / v_return + T_land_upper
  E_return_upper = P_return_upper T_return_upper / 3600
with P_return_upper taken from the HF1/HF4 calibration (or the mode is UNKNOWN_ENERGY).
"""
from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from swarmsec import ral26_core as rc  # noqa: E402

C_T, C_M = 8.54858e-06, 0.016


class PowerMonitor:
    def __init__(self, uav_id: int, inflation: float = 1.0):
        from gz_stream import GzStream
        self.inflation = inflation
        self.P = math.nan
        self.t = None
        self.lock = threading.Lock()
        self.stream = GzStream(f"/x500_{uav_id}/command/motor_speed", self._on)

    def _on(self, msg):
        w = np.asarray(msg.get("velocity", []), float)
        if w.size:
            with self.lock:
                self.P = self.inflation * float(np.sum(C_M * C_T * np.abs(w) ** 3))

    def power_w(self):
        with self.lock:
            return self.P

    def close(self):
        self.stream.close()


class EnergyEstimator:
    def __init__(self, E0_wh, reserve_wh, P_return_upper_w, v_return, T_land_upper_s, valid=True):
        self.E = float(E0_wh)
        self.E0 = float(E0_wh)
        self.reserve = float(reserve_wh)
        self.P_ret = float(P_return_upper_w)
        self.v_ret = float(v_return)
        self.T_land = float(T_land_upper_s)
        self.valid = bool(valid)
        self.t_last = None
        self.used_wh = 0.0

    def update(self, t, P_w):
        if self.t_last is not None and math.isfinite(P_w):
            dt = max(0.0, t - self.t_last)
            E_new = rc.update_energy_wh(self.E, P_w, dt)
            self.used_wh += self.E - E_new
            self.E = E_new
        self.t_last = t

    def screen(self, dist_home_m):
        T_ret = dist_home_m / max(self.v_ret, 1e-6) + self.T_land
        s = rc.energy_supervisor(self.E, self.P_ret, T_ret, self.reserve, valid=self.valid)
        s["E_wh"] = self.E
        s["T_return_upper_s"] = T_ret
        return s
