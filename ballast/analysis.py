"""Envelope studies: how much can this airframe actually do?

These are the numbers that decide whether the concept is viable, and they all
come from the same two constraints -- bounded slider travel and bounded thrust.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .params import Vehicle

__all__ = ["AuthorityPoint", "authority_envelope", "wind_envelope",
           "max_steady_tilt", "slider_bandwidth_limit"]


class AuthorityPoint(NamedTuple):
    thrust: float
    torque_max: float
    angular_accel: float
    steady_tilt_deg: float
    lateral_accel: float


def authority_envelope(veh: Vehicle, thrusts=None) -> list[AuthorityPoint]:
    """Steering authority as a function of rotor thrust.

    Both the available torque and the tilt it can hold scale with thrust, so a
    heavily loaded vehicle is not automatically a sluggish one -- but a
    *descending* one, at reduced thrust, is.
    """
    if thrusts is None:
        thrusts = np.linspace(0.3 * veh.hover_thrust, veh.rotor.thrust_max, 25)
    J = float(veh.inertia_about_com(veh.slider.position(0.0, 0.0))[0, 0])
    out = []
    for f in np.atleast_1d(thrusts):
        tau = float(f) * veh.com_offset_max
        # Holding a steady tilt costs no torque, so the tilt limit is set by
        # thrust alone: enough to carry the weight *and* push sideways.
        ratio = float(f) / veh.hover_thrust
        tilt = (np.degrees(np.arccos(np.clip(1.0 / ratio, -1.0, 1.0)))
                if ratio >= 1.0 else 0.0)
        lat = veh.gravity * np.tan(np.radians(tilt))
        out.append(AuthorityPoint(thrust=float(f), torque_max=tau,
                                  angular_accel=tau / J, steady_tilt_deg=tilt,
                                  lateral_accel=lat))
    return out


def max_steady_tilt(veh: Vehicle) -> float:
    """Largest tilt the vehicle can hold while still supporting its weight [deg]."""
    return float(np.degrees(np.arccos(np.clip(1.0 / veh.thrust_to_weight, -1.0, 1.0))))


def wind_envelope(veh: Vehicle, speeds=None) -> list[tuple]:
    """Steady wind speed vs. the tilt and torque needed to hold station.

    Returns ``(speed, tilt_deg, aero_torque, torque_margin)``.  The margin is
    what is left of the slider's authority after the aerodynamic moment about
    the centre of mass has been cancelled; once it reaches zero the vehicle can
    no longer be trimmed.
    """
    if speeds is None:
        speeds = np.arange(0.0, 20.1, 0.5)
    lever = float(veh.aero.cop[2] - veh.com_body(veh.slider.position(0.0, 0.0))[2])
    cd = float(veh.aero.cd_area[0])
    out = []
    for v in np.atleast_1d(speeds):
        drag = cd * v * v
        tilt = np.degrees(np.arctan2(drag, veh.hover_thrust))
        thrust = np.hypot(drag, veh.hover_thrust)
        aero_torque = abs(lever) * drag
        margin = thrust * veh.com_offset_max - aero_torque
        out.append((float(v), float(tilt), float(aero_torque), float(margin)))
    return out


def slider_bandwidth_limit(veh: Vehicle) -> dict:
    """Frequencies at which the slider stops being able to follow a command."""
    sp = veh.slider
    wn, z = sp.omega_n, sp.zeta
    w_bw = wn * np.sqrt(max(1 - 2 * z * z + np.sqrt(max(4 * z ** 4 - 4 * z * z + 2, 0.0)), 0.0))
    # Rate limit: a sinusoid of amplitude A at frequency w needs peak speed A*w.
    w_rate = sp.rate_max / sp.travel
    return {
        "servo_bandwidth_rad_s": float(w_bw),
        "servo_bandwidth_hz": float(w_bw / (2 * np.pi)),
        "rate_limited_rad_s": float(w_rate),
        "rate_limited_hz": float(w_rate / (2 * np.pi)),
        "limiting": "rate" if w_rate < w_bw else "servo",
    }
