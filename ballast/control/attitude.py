r"""Geometric attitude control on :math:`SO(3)`, restricted to roll and pitch.

Uses the Lee/Leok/McClamroch error metric

.. math::
    e_R = \tfrac12 \left(R_d^\top R - R^\top R_d\right)^\vee ,

which is well defined for arbitrarily large attitude errors -- no Euler-angle
singularity, no quaternion sign ambiguity.

The yaw channel is deliberately *not* driven.  The vehicle has no yaw actuator,
so the desired attitude is built around the vehicle's **current** heading: the
controller asks only for the thrust axis it wants, and lets the heading go where
the residual rotor torque takes it.  This keeps ``e_R`` free of any yaw
component that could steal roll/pitch authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import math3d as m3

__all__ = ["AttitudeGains", "AttitudeController", "desired_rotation"]


def desired_rotation(b3_des: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """Attitude with thrust axis ``b3_des``, keeping the present heading.

    The current body x-axis is projected onto the plane normal to ``b3_des``;
    if it happens to be parallel (the vehicle is pointing straight along the
    commanded thrust axis) any perpendicular direction will do.
    """
    b3 = b3_des / np.linalg.norm(b3_des)
    b1c = m3.project_perp(R_current @ np.array([1.0, 0.0, 0.0]), b3)
    if np.linalg.norm(b1c) < 1e-6:
        b1c = m3.project_perp(np.array([1.0, 0.0, 0.0]), b3)
        if np.linalg.norm(b1c) < 1e-6:
            b1c = m3.project_perp(np.array([0.0, 1.0, 0.0]), b3)
    b1 = b1c / np.linalg.norm(b1c)
    b2 = np.cross(b3, b1)
    return np.column_stack([b1, b2, b3])


@dataclass
class AttitudeGains:
    """Proportional-derivative gains on the roll and pitch channels."""

    k_rot: np.ndarray = field(default_factory=lambda: np.array([14.0, 14.0]))
    k_rate: np.ndarray = field(default_factory=lambda: np.array([3.4, 3.4]))


class AttitudeController:
    """Produce the roll/pitch torque that drives ``b3`` onto its command."""

    def __init__(self, gains: AttitudeGains | None = None):
        self.gains = gains or AttitudeGains()
        self.last_error = np.zeros(3)

    def __call__(self, R: np.ndarray, omega: np.ndarray,
                 b3_des: np.ndarray) -> np.ndarray:
        R_des = desired_rotation(b3_des, R)
        e_R = m3.vee(R_des.T @ R - R.T @ R_des)
        self.last_error = e_R
        g = self.gains
        return -g.k_rot * e_R[:2] - g.k_rate * omega[:2]
