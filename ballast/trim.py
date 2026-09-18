r"""Trim solution and linear analysis.

Hover trim is analytic for this airframe: level attitude, slider centred, rotor
thrust equal to weight.  What is *not* obvious is how the open-loop vehicle
behaves around that point, and that is what the linearisation answers.

Error-state parametrisation
---------------------------
The 18-element state carries a 4-element quaternion for 3 rotational degrees of
freedom, so a naive Jacobian is rank deficient and its eigenvalues are polluted
by the redundant direction.  We linearise instead in 17 error coordinates

.. math::
    \delta x = (\delta p,\ \delta v,\ \delta\theta,\ \delta\omega,
                \ \delta\Omega_r,\ \delta s,\ \delta\dot s),

where an attitude perturbation acts multiplicatively,
:math:`q \leftarrow q \otimes \delta q(\delta\theta)`, and is read back with the
matrix logarithm.  The resulting :math:`A` matrix has exactly the structural
zero modes one expects -- the three position integrators and the free heading --
and nothing else spurious.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import math3d as m3
from .dynamics import I_Q, Control, derivative, initial_state, unpack
from .params import Vehicle

__all__ = ["TrimPoint", "hover_trim", "linearize", "Mode", "modes", "describe_modes"]

NE = 17          # error-state dimension
_E_Q = slice(6, 9)


class TrimPoint(NamedTuple):
    state: np.ndarray
    control: Control
    residual: float


def hover_trim(veh: Vehicle, altitude: float = 0.0) -> TrimPoint:
    """Level hover with the slider centred.

    The residual reported is the norm of the state derivative.  It is not zero:
    the uncancelled fraction of rotor reaction torque leaves a yaw acceleration
    that no slider position can remove, and reporting it honestly is the point.
    """
    x = initial_state(veh, position=(0.0, 0.0, altitude))
    u = Control(thrust=veh.hover_thrust, slider_x=0.0, slider_y=0.0)
    dx = derivative(veh, x, u)
    return TrimPoint(state=x, control=u, residual=float(np.linalg.norm(dx)))


def _perturb(x: np.ndarray, idx: int, h: float) -> np.ndarray:
    """Apply an error-state perturbation of size ``h`` at error index ``idx``."""
    y = x.copy()
    if idx < 6:                                   # position, velocity
        y[idx] += h
    elif idx < 9:                                 # attitude (multiplicative)
        dtheta = np.zeros(3)
        dtheta[idx - 6] = h
        y[I_Q] = m3.quat_normalize(
            m3.quat_multiply(x[I_Q], m3.quat_from_axis_angle(dtheta, np.linalg.norm(dtheta))))
    else:                                         # rates, rotor, slider
        y[idx + 1] += h
    return y


def _error_derivative(veh: Vehicle, x: np.ndarray, u: Control,
                      x_ref: np.ndarray) -> np.ndarray:
    """State derivative expressed in the 17 error coordinates about ``x_ref``."""
    dx = derivative(veh, x, u)
    out = np.empty(NE)
    out[0:6] = dx[0:6]
    # attitude rate in error coordinates is just the body rate
    out[6:9] = unpack(x).omega     # attitude error rate is the body rate
    out[9:12] = dx[10:13]          # angular acceleration
    out[12] = dx[13]               # rotor spool
    out[13:15] = dx[14:16]         # slider velocity
    out[15:17] = dx[16:18]         # slider acceleration
    return out


def linearize(veh: Vehicle, trim: TrimPoint | None = None, h: float = 1e-6):
    """Central-difference Jacobians ``(A, B)`` in error coordinates."""
    trim = trim or hover_trim(veh)
    x0, u0 = trim.state, trim.control

    A = np.zeros((NE, NE))
    for i in range(NE):
        xp = _perturb(x0, i, h)
        xm = _perturb(x0, i, -h)
        A[:, i] = (_error_derivative(veh, xp, u0, x0)
                   - _error_derivative(veh, xm, u0, x0)) / (2 * h)

    B = np.zeros((NE, 3))
    u_arr = np.array(u0)
    for j in range(3):
        du = np.zeros(3)
        du[j] = h if j == 0 else h * 100.0        # slider scale is much smaller
        up = Control(*(u_arr + du))
        um = Control(*(u_arr - du))
        B[:, j] = (_error_derivative(veh, x0, up, x0)
                   - _error_derivative(veh, x0, um, x0)) / (2 * du[j])
    return A, B


class Mode(NamedTuple):
    eigenvalue: complex
    frequency_hz: float
    damping_ratio: float
    time_constant: float
    label: str


def _label(lam: complex, tol: float) -> str:
    if abs(lam) < tol:
        return "rigid-body integrator"
    if abs(lam.imag) < tol:
        return "stable first-order" if lam.real < 0 else "DIVERGENT first-order"
    if lam.real < -tol:
        return "damped oscillation"
    if lam.real > tol:
        return "DIVERGENT oscillation"
    return "undamped oscillation"


def modes(A: np.ndarray, tol: float | None = None) -> list[Mode]:
    """Eigen-decomposition of ``A``, sorted most-unstable first.

    ``tol`` defaults to a *relative* threshold.  The horizontal channel of any
    thrust-vectored hovering vehicle is a chain of integrators
    (``x <- v <- theta <- omega``), whose eigenvalues are analytically zero but
    come out of a finite-difference Jacobian as a small cluster spread around
    the origin.  An absolute tolerance would mislabel half of that cluster as
    divergent; a relative one keeps the classification honest.
    """
    if tol is None:
        tol = 1e-4 * max(float(np.linalg.norm(A, 2)), 1.0)
    out: list[Mode] = []
    for lam in np.linalg.eigvals(A):
        wn = abs(lam)
        zeta = -lam.real / wn if wn > tol else float("nan")
        tc = -1.0 / lam.real if abs(lam.real) > tol else float("inf")
        out.append(Mode(eigenvalue=lam, frequency_hz=abs(lam.imag) / (2 * np.pi),
                        damping_ratio=float(zeta), time_constant=float(tc),
                        label=_label(lam, tol)))
    return sorted(out, key=lambda m: -m.eigenvalue.real)


def describe_modes(ms: list[Mode], tol: float = 1e-6) -> str:
    lines = [f"{'eigenvalue':>24s}  {'f [Hz]':>8s}  {'zeta':>7s}  "
             f"{'tau [s]':>10s}   character"]
    lines.append("-" * 76)
    rigid = sum(1 for m in ms if m.label == "rigid-body integrator")
    seen: set[tuple] = set()
    for m in ms:
        if m.label == "rigid-body integrator":
            continue
        key = (round(m.eigenvalue.real, 6), round(abs(m.eigenvalue.imag), 6))
        if key in seen:
            continue
        seen.add(key)
        conj = "" if abs(m.eigenvalue.imag) < tol else " (x2)"
        ev = f"{m.eigenvalue.real:+.4f}{m.eigenvalue.imag:+.4f}j"
        tau = "       inf" if not np.isfinite(m.time_constant) else f"{m.time_constant:10.3f}"
        zeta = "      -" if not np.isfinite(m.damping_ratio) else f"{m.damping_ratio:7.3f}"
        lines.append(f"{ev:>24s}  {m.frequency_hz:8.3f}  {zeta}  {tau}   "
                     f"{m.label}{conj}")
    if rigid:
        lines.append(f"{'0 (x%d)' % rigid:>24s}  {'':>8s}  {'':>7s}  {'':>10s}   "
                     f"rigid-body integrators: position, horizontal")
        lines.append(f"{'':>24s}  {'':>8s}  {'':>7s}  {'':>10s}   "
                     f"translation chain, and free heading")
    return "\n".join(lines)
