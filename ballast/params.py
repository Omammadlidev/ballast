"""Vehicle, actuator and environment parameters.

Body frame convention
---------------------
Origin ``O`` sits at the **airframe shell's own centre of mass**, so the shell
contributes no first mass moment about ``O``.  ``+z`` points up through the
rotor, ``+x`` forward, ``+y`` left.  The world frame is ENU with gravity along
``-z``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = 9.80665


def _v(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=float)


@dataclass
class RotorParams:
    """Single fixed-pitch rotor with trimmed anti-torque stator vanes.

    The vanes sit in the slipstream, so the torque they recover scales with
    thrust exactly as the rotor reaction torque does.  Their cancellation ratio
    is therefore thrust-independent and set only by build tolerance.
    """

    kf: float = 9.0e-6            # thrust = kf * omega^2   [N/(rad/s)^2]
    ctau: float = 0.018           # reaction torque = ctau * thrust  [m]
    spin_dir: float = 1.0         # +1 = rotor turns about +z_body
    vane_ratio: float = 0.995     # fraction of reaction torque cancelled by vanes
    tau_spool: float = 0.050      # first-order spin-up time constant [s]
    thrust_max: float = 14.0      # [N]
    thrust_min: float = 0.0       # [N]
    position: np.ndarray = field(default_factory=lambda: _v(0.0, 0.0, 0.120))

    @property
    def omega_max(self) -> float:
        return float(np.sqrt(self.thrust_max / self.kf))

    def thrust(self, omega: float) -> float:
        return float(self.kf * omega * omega)

    def omega_for(self, thrust: float) -> float:
        t = float(np.clip(thrust, self.thrust_min, self.thrust_max))
        return float(np.sqrt(max(t, 0.0) / self.kf))

    def residual_yaw_torque(self, thrust: float) -> float:
        """Reaction torque left over after the vanes, about ``+z_body``."""
        return -self.spin_dir * self.ctau * (1.0 - self.vane_ratio) * thrust


@dataclass
class SliderParams:
    """The internal 2-DOF moving mass -- the vehicle's only steering actuator.

    The mass travels in a plane a fixed distance below the shell centre of mass
    and is driven by two position-servo axes modelled as a second-order lag with
    hard end-stops.
    """

    mass: float = 0.15            # [kg]
    plane_z: float = -0.030       # slider plane height in body frame [m]
    travel: float = 0.060         # per-axis half-travel [m]
    omega_n: float = 25.0         # servo natural frequency [rad/s]
    zeta: float = 0.80            # servo damping ratio
    rate_max: float = 0.80        # per-axis speed limit [m/s]

    def position(self, sx: float, sy: float) -> np.ndarray:
        return _v(sx, sy, self.plane_z)

    def clamp(self, s_xy: np.ndarray) -> np.ndarray:
        """Clamp a *commanded* offset to the circular travel envelope."""
        n = float(np.linalg.norm(s_xy))
        return s_xy if n <= self.travel else s_xy * (self.travel / n)


@dataclass
class AeroParams:
    """Quasi-steady aerodynamics.

    Translational drag acts at a centre of pressure that is deliberately placed
    slightly *below* the centre of mass, which makes the airframe mildly
    weathervane-stable without generating moments the slider cannot overpower.
    """

    cd_area: np.ndarray = field(default_factory=lambda: _v(0.012, 0.012, 0.040))
    cop: np.ndarray = field(default_factory=lambda: _v(0.0, 0.0, -0.030))
    rate_damp_lin: np.ndarray = field(default_factory=lambda: _v(3.0e-3, 3.0e-3, 2.5e-2))
    rate_damp_quad: np.ndarray = field(default_factory=lambda: _v(6.0e-4, 6.0e-4, 3.0e-3))
    rho: float = 1.225

    def drag_body(self, v_rel_body: np.ndarray) -> np.ndarray:
        """Quadratic drag force in body axes for a body-frame relative wind."""
        return -self.cd_area * np.abs(v_rel_body) * v_rel_body * (self.rho / 1.225)

    def rate_damping(self, omega_body: np.ndarray) -> np.ndarray:
        mag = float(np.linalg.norm(omega_body))
        return -(self.rate_damp_lin + self.rate_damp_quad * mag) * omega_body


@dataclass
class Vehicle:
    """Complete single-rotor, moving-mass vehicle."""

    shell_mass: float = 0.60      # airframe excluding the slider [kg]
    shell_inertia: np.ndarray = field(
        default_factory=lambda: np.diag([9.0e-3, 9.0e-3, 3.5e-3]))
    rotor: RotorParams = field(default_factory=RotorParams)
    slider: SliderParams = field(default_factory=SliderParams)
    aero: AeroParams = field(default_factory=AeroParams)
    gravity: float = G

    # ---- derived quantities -------------------------------------------------
    @property
    def total_mass(self) -> float:
        return self.shell_mass + self.slider.mass

    @property
    def reduced_mass(self) -> float:
        """``mb*ms/M`` -- appears in both the inertia and the coupling terms."""
        return self.shell_mass * self.slider.mass / self.total_mass

    @property
    def hover_thrust(self) -> float:
        return self.total_mass * self.gravity

    @property
    def thrust_to_weight(self) -> float:
        return self.rotor.thrust_max / self.hover_thrust

    @property
    def com_offset_max(self) -> float:
        """Largest lateral centre-of-mass shift the slider can produce [m]."""
        return self.slider.mass / self.total_mass * self.slider.travel

    @property
    def control_torque_max(self) -> float:
        """Largest steering torque available at hover thrust [N.m]."""
        return self.hover_thrust * self.com_offset_max

    @property
    def angular_accel_max(self) -> float:
        """Peak roll/pitch acceleration at hover thrust [rad/s^2].

        Uses the inertia about the *system* centre of mass with the slider
        centred, not the bare shell inertia -- the slider's own parallel-axis
        contribution is part of what has to be turned.
        """
        J = self.inertia_about_com(self.slider.position(0.0, 0.0))
        return self.control_torque_max / float(J[0, 0])

    def com_body(self, s: np.ndarray) -> np.ndarray:
        """System centre of mass in body coordinates for slider position ``s``."""
        return (self.slider.mass / self.total_mass) * s

    def inertia_about_com(self, s: np.ndarray) -> np.ndarray:
        """Exact system inertia about the *instantaneous* system centre of mass.

        Both bodies move relative to the system CoM when the slider moves; the
        two parallel-axis contributions collapse into a single reduced-mass term.
        """
        return self.shell_inertia + self.reduced_mass * (
            float(s @ s) * np.eye(3) - np.outer(s, s))

    def internal_momentum(self, s: np.ndarray, s_dot: np.ndarray) -> np.ndarray:
        """Angular momentum stored in the slider's motion relative to the shell."""
        return self.reduced_mass * np.cross(s, s_dot)

    def summary(self) -> str:
        return (
            f"total mass          {self.total_mass:.3f} kg "
            f"(shell {self.shell_mass:.3f} + slider {self.slider.mass:.3f})\n"
            f"hover thrust        {self.hover_thrust:.3f} N\n"
            f"thrust/weight       {self.thrust_to_weight:.2f}\n"
            f"slider mass ratio   {self.slider.mass / self.total_mass * 100:.1f} %\n"
            f"max CoM offset      {self.com_offset_max * 1000:.1f} mm\n"
            f"max steering torque {self.control_torque_max:.4f} N.m\n"
            f"peak ang. accel     {self.angular_accel_max:.2f} rad/s^2"
        )
