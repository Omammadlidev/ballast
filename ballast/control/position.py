"""Outer position loop: PID on position, producing a desired thrust direction.

The commanded tilt is clipped to a configurable cone.  That cone is the honest
expression of this airframe's limits -- with a steering torque of order
0.09 N.m, a large commanded tilt is a promise the slider cannot keep, and
letting the outer loop ask for one only winds the vehicle up.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..params import G

__all__ = ["PositionGains", "PositionController", "AccelCommand"]


@dataclass
class PositionGains:
    k_pos: np.ndarray = field(default_factory=lambda: np.array([0.9, 0.9, 6.0]))
    k_vel: np.ndarray = field(default_factory=lambda: np.array([1.5, 1.5, 4.2]))
    k_int: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.25, 1.2]))
    integral_limit: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 2.0]))
    tilt_limit_deg: float = 25.0


@dataclass
class AccelCommand:
    accel: np.ndarray        # desired specific force, world frame [m/s^2]
    b3_des: np.ndarray       # desired thrust axis (unit, world)
    tilt_clipped: bool


class PositionController:
    """PID position loop with anti-windup and a tilt cone."""

    def __init__(self, gains: PositionGains | None = None, gravity: float = G):
        self.gains = gains or PositionGains()
        self.gravity = gravity
        self.integral = np.zeros(3)

    def reset(self) -> None:
        self.integral = np.zeros(3)

    def __call__(self, p: np.ndarray, v: np.ndarray, p_ref: np.ndarray,
                 v_ref: np.ndarray, a_ref: np.ndarray, dt: float) -> AccelCommand:
        g = self.gains
        e_p = p - p_ref
        self.integral = np.clip(self.integral + e_p * dt,
                                -g.integral_limit, g.integral_limit)
        accel = (-g.k_pos * e_p - g.k_vel * (v - v_ref) - g.k_int * self.integral
                 + a_ref + np.array([0.0, 0.0, self.gravity]))

        norm = float(np.linalg.norm(accel))
        if norm < 1e-9:
            return AccelCommand(accel, np.array([0.0, 0.0, 1.0]), False)
        b3 = accel / norm

        cz = np.cos(np.radians(g.tilt_limit_deg))
        clipped = False
        if b3[2] < cz:
            clipped = True
            horiz = b3[:2]
            hn = float(np.linalg.norm(horiz))
            if hn > 1e-9:
                b3 = np.concatenate([horiz / hn * np.sqrt(1.0 - cz * cz), [cz]])
            else:
                b3 = np.array([0.0, 0.0, 1.0])
        return AccelCommand(accel, b3, clipped)
