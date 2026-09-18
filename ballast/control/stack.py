"""The full cascade, from position reference to the two physical actuators.

::

    position error --> desired accel --> desired thrust axis  (position.py)
                                     |
                                     +-> thrust magnitude  --> rotor
                                     |
                       attitude error --> roll/pitch torque  (attitude.py)
                                     |
                                     +-> CoM offset --> slider  (allocation.py)

The inner loop is fast relative to nothing in particular: this airframe's peak
angular acceleration is around 10 rad/s^2, roughly an order of magnitude below a
comparable quadrotor's.  The gains below are tuned to that reality rather than
borrowed from multirotor practice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..dynamics import Control, origin_position, origin_velocity, unpack
from ..params import Vehicle
from .allocation import allocate
from .attitude import AttitudeController, AttitudeGains
from .position import PositionController, PositionGains

__all__ = ["ControllerGains", "FlightController"]


@dataclass
class ControllerGains:
    position: PositionGains = field(default_factory=PositionGains)
    attitude: AttitudeGains = field(default_factory=AttitudeGains)


class FlightController:
    """Callable with the signature the simulator expects.

    ``controller(vehicle, x, t, dt, p_ref, v_ref, a_ref) -> Control``
    """

    def __init__(self, gains: ControllerGains | None = None):
        self.gains = gains or ControllerGains()
        self.position = PositionController(self.gains.position)
        self.attitude = AttitudeController(self.gains.attitude)
        self.last_allocation = None
        self.last_accel_cmd = None

    def reset(self) -> None:
        self.position.reset()

    def __call__(self, veh: Vehicle, x: np.ndarray, t: float, dt: float,
                 p_ref: np.ndarray, v_ref: np.ndarray,
                 a_ref: np.ndarray) -> Control:
        st = unpack(x)
        p = origin_position(veh, x)
        v = origin_velocity(veh, x)

        cmd = self.position(p, v, p_ref, v_ref, a_ref, dt)
        self.last_accel_cmd = cmd

        # Thrust is the projection of the desired specific force onto the axis
        # the vehicle actually has right now -- not onto the one it wants.
        b3_now = st.R @ np.array([0.0, 0.0, 1.0])
        thrust = veh.total_mass * float(cmd.accel @ b3_now)
        thrust = float(np.clip(thrust, 0.5, veh.rotor.thrust_max))

        torque_xy = self.attitude(st.R, st.omega, cmd.b3_des)
        alloc = allocate(veh, torque_xy, thrust)
        self.last_allocation = alloc

        return Control(thrust=thrust,
                       slider_x=float(alloc.slider_cmd[0]),
                       slider_y=float(alloc.slider_cmd[1]))
