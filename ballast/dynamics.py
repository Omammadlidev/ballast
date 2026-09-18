"""Full 18-state vehicle dynamics.

State layout
------------
=========  =====================================================
 indices    quantity
=========  =====================================================
 ``0:3``    system centre-of-mass position, world frame [m]
 ``3:6``    system centre-of-mass velocity, world frame [m/s]
 ``6:10``   attitude quaternion, body -> world ``[w,x,y,z]``
 ``10:13``  body angular rate, body frame [rad/s]
 ``13``     rotor speed [rad/s]
 ``14:16``  slider position ``(sx, sy)`` [m]
 ``16:18``  slider velocity ``(sx_dot, sy_dot)`` [m/s]
=========  =====================================================

The *system* centre of mass is integrated rather than an airframe-fixed point,
because Newton's second law is then exactly ``M * a_G = F``.  The airframe
origin -- where sensors and payload actually live -- is recovered by
:func:`origin_position`.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import math3d as m3
from .multibody import angular_acceleration
from .params import Vehicle

__all__ = [
    "NX", "Control", "State", "initial_state", "unpack",
    "origin_position", "origin_velocity", "derivative", "Diagnostics",
]

NX = 18

I_P = slice(0, 3)
I_V = slice(3, 6)
I_Q = slice(6, 10)
I_W = slice(10, 13)
I_ROTOR = 13
I_S = slice(14, 16)
I_SD = slice(16, 18)


class Control(NamedTuple):
    """Commands sent to the vehicle's two physical actuators."""

    thrust: float          # commanded rotor thrust [N]
    slider_x: float        # commanded slider position [m]
    slider_y: float        # commanded slider position [m]

    def as_array(self) -> np.ndarray:
        return np.array([self.thrust, self.slider_x, self.slider_y])


class State(NamedTuple):
    """Unpacked view of the state vector."""

    p_com: np.ndarray
    v_com: np.ndarray
    quat: np.ndarray
    omega: np.ndarray
    rotor_speed: float
    slider: np.ndarray       # (sx, sy)
    slider_rate: np.ndarray  # (sx_dot, sy_dot)
    R: np.ndarray


class Diagnostics(NamedTuple):
    thrust: float
    com_body: np.ndarray
    drag_body: np.ndarray
    torque_body: np.ndarray
    omega_dot: np.ndarray
    tilt: float
    heading: float


def unpack(x: np.ndarray) -> State:
    q = m3.quat_normalize(x[I_Q])
    return State(
        p_com=x[I_P], v_com=x[I_V], quat=q, omega=x[I_W],
        rotor_speed=float(x[I_ROTOR]), slider=x[I_S], slider_rate=x[I_SD],
        R=m3.quat_to_rot(q),
    )


def initial_state(veh: Vehicle, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
                  quat=(1.0, 0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0),
                  slider=(0.0, 0.0), trimmed: bool = True) -> np.ndarray:
    """Build a state vector, by default with the rotor already at hover speed."""
    x = np.zeros(NX)
    x[I_P] = position
    x[I_V] = velocity
    x[I_Q] = m3.quat_normalize(np.asarray(quat, dtype=float))
    x[I_W] = omega
    x[I_ROTOR] = veh.rotor.omega_for(veh.hover_thrust) if trimmed else 0.0
    x[I_S] = slider
    return x


def origin_position(veh: Vehicle, x: np.ndarray) -> np.ndarray:
    """Airframe origin (sensor/payload location) in world coordinates."""
    st = unpack(x)
    s = veh.slider.position(st.slider[0], st.slider[1])
    return st.p_com - st.R @ veh.com_body(s)


def origin_velocity(veh: Vehicle, x: np.ndarray) -> np.ndarray:
    """World velocity of the airframe origin, including slider-motion coupling."""
    st = unpack(x)
    s = veh.slider.position(st.slider[0], st.slider[1])
    s_dot = np.array([st.slider_rate[0], st.slider_rate[1], 0.0])
    c = veh.com_body(s)
    c_dot = veh.com_body(s_dot)
    return st.v_com - st.R @ (c_dot + np.cross(st.omega, c))


def _slider_acceleration(veh: Vehicle, s_xy: np.ndarray, sd_xy: np.ndarray,
                         cmd_xy: np.ndarray) -> np.ndarray:
    """Second-order position servo with travel and rate limits."""
    sp = veh.slider
    cmd = sp.clamp(np.asarray(cmd_xy, dtype=float))
    acc = sp.omega_n ** 2 * (cmd - s_xy) - 2.0 * sp.zeta * sp.omega_n * sd_xy
    # Soft rate limit: refuse to accelerate further once the axis is at speed.
    over = np.abs(sd_xy) >= sp.rate_max
    acc = np.where(over & (np.sign(acc) == np.sign(sd_xy)), 0.0, acc)
    return acc


def derivative(veh: Vehicle, x: np.ndarray, u: Control,
               wind: np.ndarray | None = None,
               with_diagnostics: bool = False):
    """Time derivative of the state vector.

    ``wind`` is the world-frame air velocity; ``None`` means still air.
    """
    st = unpack(x)
    R = st.R
    v_wind = np.zeros(3) if wind is None else np.asarray(wind, dtype=float)

    s = veh.slider.position(st.slider[0], st.slider[1])
    s_dot = np.array([st.slider_rate[0], st.slider_rate[1], 0.0])
    s_ddot_xy = _slider_acceleration(veh, st.slider, st.slider_rate,
                                     np.array([u.slider_x, u.slider_y]))
    s_ddot = np.array([s_ddot_xy[0], s_ddot_xy[1], 0.0])

    c = veh.com_body(s)
    thrust = veh.rotor.thrust(st.rotor_speed)
    f_thrust_body = np.array([0.0, 0.0, thrust])

    v_rel_body = R.T @ (st.v_com - v_wind)
    f_drag_body = veh.aero.drag_body(v_rel_body)

    torque = (np.cross(veh.rotor.position - c, f_thrust_body)
              + np.cross(veh.aero.cop - c, f_drag_body)
              + np.array([0.0, 0.0, veh.rotor.residual_yaw_torque(thrust)])
              + veh.aero.rate_damping(st.omega))

    rot = angular_acceleration(veh, st.omega, s, s_dot, s_ddot, torque)

    a_com = R @ (f_thrust_body + f_drag_body) / veh.total_mass \
        - np.array([0.0, 0.0, veh.gravity])

    omega_cmd = veh.rotor.omega_for(u.thrust)
    rotor_dot = (omega_cmd - st.rotor_speed) / veh.rotor.tau_spool

    dx = np.empty(NX)
    dx[I_P] = st.v_com
    dx[I_V] = a_com
    dx[I_Q] = m3.quat_derivative(st.quat, st.omega)
    dx[I_W] = rot.omega_dot
    dx[I_ROTOR] = rotor_dot
    dx[I_S] = st.slider_rate
    dx[I_SD] = s_ddot_xy

    if not with_diagnostics:
        return dx
    diag = Diagnostics(
        thrust=thrust, com_body=c, drag_body=f_drag_body, torque_body=torque,
        omega_dot=rot.omega_dot, tilt=m3.tilt_angle(R), heading=m3.heading_angle(R),
    )
    return dx, diag
