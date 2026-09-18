r"""Control allocation: body torque :math:`\to` slider position.

The rotor thrust is fixed in the body frame, so the only steering moment comes
from the offset between the thrust line and the centre of mass.  With thrust
:math:`F = f\,\hat z_b` applied at :math:`r_t` and the centre of mass at
:math:`c`,

.. math::
    \tau = (r_t - c) \times f\hat z_b
         = f\,\begin{bmatrix} -c_y \\ c_x \\ 0 \end{bmatrix}.

Two things follow, and they define the whole vehicle.

1. **Inversion is trivial** -- :math:`c_x = \tau_y / f`,
   :math:`c_y = -\tau_x / f`, and the slider command is
   :math:`s = (M/m_s)\,c`.  The vertical lever :math:`r_{t,z}-c_z` does not
   appear: it scales no torque at all.
2. **Yaw is unreachable.**  The third row is identically zero, so no slider
   position produces a moment about the thrust axis.  Heading is left to the
   anti-torque vanes and is not commanded here.

Authority is bounded by the travel limit, giving a maximum steering torque of
:math:`f \cdot (m_s/M)\, s_{\max}` -- a few hundredths of a newton-metre.  The
allocator therefore saturates by **preserving torque direction** and scaling
magnitude, which keeps the vehicle tilting the right way when it cannot tilt
hard enough.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..params import Vehicle

__all__ = ["Allocation", "allocate", "achievable_torque"]


class Allocation(NamedTuple):
    slider_cmd: np.ndarray    # (sx, sy) commanded slider position [m]
    com_cmd: np.ndarray       # (cx, cy) commanded CoM offset [m]
    torque_cmd: np.ndarray    # (tau_x, tau_y) actually requested [N.m]
    saturated: bool
    demand_ratio: float       # requested torque / available torque


def allocate(veh: Vehicle, torque_xy: np.ndarray, thrust: float) -> Allocation:
    """Map a desired roll/pitch torque to a slider position command."""
    f = max(float(thrust), 1e-3)
    tau = np.asarray(torque_xy, dtype=float)[:2]

    com = np.array([tau[1] / f, -tau[0] / f])
    slider = com * (veh.total_mass / veh.slider.mass)

    travel = veh.slider.travel
    norm = float(np.linalg.norm(slider))
    demand = norm / travel if travel > 0 else np.inf
    saturated = norm > travel
    if saturated:
        slider = slider * (travel / norm)
        com = slider * (veh.slider.mass / veh.total_mass)
        tau = np.array([-com[1] * f, com[0] * f])

    return Allocation(slider_cmd=slider, com_cmd=com, torque_cmd=tau,
                      saturated=saturated, demand_ratio=demand)


def achievable_torque(veh: Vehicle, thrust: float) -> float:
    """Largest roll/pitch torque available at a given thrust [N.m]."""
    return float(thrust) * veh.com_offset_max
