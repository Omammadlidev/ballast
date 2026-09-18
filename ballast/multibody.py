r"""Exact rotational dynamics of a rigid shell carrying a moving point mass.

Most moving-mass vehicle models cheat: they shift the centre of mass, keep the
inertia tensor frozen, and drop the reaction terms.  That is fine for sizing and
wrong for control design, because the neglected terms are first order in the
very signal the controller commands.

Derivation
----------
Let the airframe shell have mass :math:`m_b` with its own centre of mass at the
body origin :math:`O`, and let a point mass :math:`m_s` sit at body position
:math:`s(t)`.  Write :math:`M = m_b + m_s` and the reduced mass
:math:`\mu = m_b m_s / M`.

The system centre of mass in body coordinates is

.. math:: c = \frac{m_s}{M}\, s .

Measuring each body from :math:`G` gives :math:`\rho_b = -c` and
:math:`\rho_s = s - c = \frac{m_b}{M} s`.  Their parallel-axis contributions
collapse into one reduced-mass term:

.. math::
    J_G(s) = J_b + m_b\big(\lVert\rho_b\rVert^2 I - \rho_b\rho_b^\top\big)
                 + m_s\big(\lVert\rho_s\rVert^2 I - \rho_s\rho_s^\top\big)
           = J_b + \mu\big(s^\top s\, I - s s^\top\big).

The slider's motion relative to the shell also stores angular momentum.  With
:math:`\dot\rho_b = -\frac{m_s}{M}\dot s` and
:math:`\dot\rho_s = \frac{m_b}{M}\dot s`,

.. math::
    h_{\mathrm{rel}} = \sum_i m_i\,\rho_i \times \dot\rho_i
                     = \mu\, (s \times \dot s).

So :math:`H_G = J_G \omega + h_{\mathrm{rel}}`, and transporting the derivative
into the rotating body frame,

.. math::
    J_G \dot\omega = \tau_G - \omega \times H_G - \dot J_G\,\omega
                     - \dot h_{\mathrm{rel}},

with :math:`\dot J_G = \mu\,(2 (s\cdot\dot s) I - \dot s s^\top - s \dot s^\top)`
and :math:`\dot h_{\mathrm{rel}} = \mu\,(s \times \ddot s)`.

Every term is retained below.  The three that a simplified model would drop --
:math:`\dot J_G \omega`, :math:`h_{\mathrm{rel}}` inside the gyroscopic product,
and :math:`\dot h_{\mathrm{rel}}` -- are exactly the ones driven by slider
motion, so they matter most when the controller is working hardest.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .params import Vehicle

__all__ = ["RotationalTerms", "inertia_terms", "angular_acceleration"]


class RotationalTerms(NamedTuple):
    """Breakdown of the rotational equation, useful for analysis and tests."""

    inertia: np.ndarray            # J_G
    inertia_rate: np.ndarray       # dJ_G/dt
    internal_momentum: np.ndarray  # h_rel
    internal_torque: np.ndarray    # d h_rel / dt
    angular_momentum: np.ndarray   # H_G
    gyroscopic: np.ndarray         # -omega x H_G
    applied_torque: np.ndarray     # tau_G
    omega_dot: np.ndarray


def inertia_terms(veh: Vehicle, s: np.ndarray, s_dot: np.ndarray):
    """Return ``(J_G, dJ_G/dt, h_rel)`` for a slider state."""
    mu = veh.reduced_mass
    J = veh.shell_inertia + mu * (float(s @ s) * np.eye(3) - np.outer(s, s))
    J_dot = mu * (2.0 * float(s @ s_dot) * np.eye(3)
                  - np.outer(s_dot, s) - np.outer(s, s_dot))
    h_rel = mu * np.cross(s, s_dot)
    return J, J_dot, h_rel


def angular_acceleration(
    veh: Vehicle,
    omega: np.ndarray,
    s: np.ndarray,
    s_dot: np.ndarray,
    s_ddot: np.ndarray,
    applied_torque: np.ndarray,
) -> RotationalTerms:
    """Solve the full moving-mass Euler equation for ``omega_dot``.

    ``applied_torque`` must already be expressed about the *system* centre of
    mass, in body coordinates.
    """
    mu = veh.reduced_mass
    J, J_dot, h_rel = inertia_terms(veh, s, s_dot)
    h_rel_dot = mu * np.cross(s, s_ddot)
    H = J @ omega + h_rel
    gyroscopic = -np.cross(omega, H)
    rhs = applied_torque + gyroscopic - J_dot @ omega - h_rel_dot
    omega_dot = np.linalg.solve(J, rhs)
    return RotationalTerms(
        inertia=J,
        inertia_rate=J_dot,
        internal_momentum=h_rel,
        internal_torque=h_rel_dot,
        angular_momentum=H,
        gyroscopic=gyroscopic,
        applied_torque=applied_torque,
        omega_dot=omega_dot,
    )
